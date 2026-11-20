"""
リアルタイム推定の処理。オフラインの前処理と同じ計算を、届いたデータから逐次行う。

    Sample (部位ごと, 約 60 Hz)
      → 机置きデータの補正 (ジャイロのバイアス・圧力のゼロ点)       dataset._apply_static_calib と同じ
      → 時計のずれの補正 (インソールと IMU の clock_shift)          dataset._load_and_align_data と同じ
      → 100 Hz の共通の時刻に線形補間                                同上 (np.interp と同じ値)
      → T ポーズ: search_sec 秒ためて静止区間を探す                  dataset._apply_tpose_calib (テスト時) と同じ
      → IMU の取り付け向きの補正・圧力の体重基準化・正規化 (stats.json)
      → 直近 window フレームで推論し、delay フレーム前の推定を出力     scripts/realtime_sim.py と同じ方式
        (前回の窓の推定と線形に混ぜて、推論ごとのつなぎ目をなめらかにする)
      → SMPL の順運動学 (関節位置・メッシュ変形用の変換行列)

状態: wait (全デバイスの受信待ち) → [desk (机置き)] → ready (T ポーズの開始待ち) → tpose → run
      run 中も request_tpose() で T ポーズのキャリブレーションをやり直せる。
"""
import json
import glob
import os
import re
import time
from collections import deque

import numpy as np
import torch

from dataset import calibration as calib
from dataset.dataset import InsoleIMUDataset, SENSOR_RE, subject_key
from models import KinematicFusionSystem
from utils.paths import get_paths
from utils.rotation import rot6d_to_matrix, matrix_to_axis_angle
from utils.smpl import SMPLBank, PARENTS, NUM_BETAS

from realtime.sources import PARTS, FOOT_PARTS, NUM_PRESSURE

IMU_ORDER = ('AL', 'AR', 'FL', 'FR', 'BC')   # モデルの IMU 入力の並び (dataset と同じ)
FPS = 100.0


# ----------------------------------------------------------------------
# 100 Hz への逐次の補間
# ----------------------------------------------------------------------
class Resampler:
    """
    部位ごとに届くサンプル (時刻 [s], 値) を 100 Hz の共通の時刻 k / 100 に線形補間する。
    全部位のデータがそろった時刻まで出力する。途切れている部位 (stalled: 一定時間届いていない部位。
    呼び出し側が実際の受信時刻で判定する) は待たずに最後の値のまま進める (切断で全体が止まらないように)。
    """
    def __init__(self, parts, fps=FPS):
        self.buf = {p: deque() for p in parts}
        self.dt = 1.0 / fps
        self.k = None
        self.held = {p: 0 for p in parts}

    def push(self, part, t, vec):
        b = self.buf[part]
        if b and t <= b[-1][0]:
            return   # 重複・逆転したタイムスタンプは捨てる
        b.append((t, vec))

    def pop_frames(self, stalled=()):
        if any(not b for b in self.buf.values()):
            return []
        if self.k is None:
            self.k = int(np.ceil(max(b[0][0] for b in self.buf.values()) / self.dt - 1e-9))
        active = [b[-1][0] for p, b in self.buf.items() if p not in stalled]
        if not active:
            return []
        limit = min(active)
        out = []
        while self.k * self.dt <= limit:
            t = self.k * self.dt
            vals = {}
            for p, b in self.buf.items():
                while len(b) >= 2 and b[1][0] <= t:
                    b.popleft()
                if len(b) >= 2 and b[0][0] <= t:
                    (t0, v0), (t1, v1) = b[0], b[1]
                    vals[p] = v0 + (t - t0) / (t1 - t0) * (v1 - v0)
                else:
                    vals[p] = b[0][1]
                    if b[0][0] < t:
                        self.held[p] += 1
            out.append((self.k, t, vals))
            self.k += 1
        return out


# ----------------------------------------------------------------------
# 設定の解決
# ----------------------------------------------------------------------
def resolve_clock_shift(cfg_value, paths, subject, log=print):
    """
    時計のずれ [s] {part: shift}。センサー時刻 = timestamp_ms / 1000 - shift。
    'auto' なら学習時の同期結果 (sync/train.json の clock_shift_sec) の被験者ごとの平均を使う。
    """
    if isinstance(cfg_value, dict):
        return {p: float(cfg_value.get(p, 0.0)) for p in PARTS}
    shift = {p: 0.0 for p in PARTS}
    if cfg_value in (None, 'none', 0):
        return shift
    try:
        with open(paths.sync_train, encoding='utf-8') as f:
            sync = json.load(f)
    except FileNotFoundError:
        log(f"[calib] 注意: {paths.sync_train} が無いため、時計のずれは補正しません")
        return shift
    rows = [v['clock_shift_sec'] for k, v in sync.items() if subject_key(k) == subject and 'clock_shift_sec' in v]
    if not rows:
        rows = [v['clock_shift_sec'] for v in sync.values() if 'clock_shift_sec' in v]
    if rows:
        shift = {p: float(np.mean([r.get(p, 0.0) for r in rows])) for p in PARTS}
    return shift


def load_static_calib(data_dir, subject, log=print):
    """
    机置きデータ (data/sensor/calib/D?S?A0_*.csv) のキャリブレーション値 (dataset._static_calib と同じ探し方:
    同じ D?S? → 同じ収集回 D? の他の被験者)。無ければ None。
    """
    files = {}
    for path in glob.glob(os.path.join(data_dir, 'sensor', 'calib', '*.csv')):
        m = SENSOR_RE.match(os.path.basename(path))
        if m and re.search(r'A0$', m.group(1), re.IGNORECASE):
            files.setdefault(m.group(1).upper(), {})[m.group(2).upper()] = path
    coll = re.match(r'(D\d+)', subject).group(1)
    same = sorted(d for d in files if subject_key(d) == subject)
    other = sorted(d for d in files if d.startswith(coll + 'S') and d not in same)
    for did in same + other:
        raw = {p: InsoleIMUDataset._read_sensor_csv(path, with_pressure=(p in FOOT_PARTS))
               for p, path in files[did].items()}
        return did, calib.static_calibration(raw)
    return None, None


# ----------------------------------------------------------------------
# 本体
# ----------------------------------------------------------------------
class RealtimePipeline:
    def __init__(self, config, rt, device, log=print):
        self.log = log
        self.device = device
        ds_cfg = config.get('dataset', {})
        if ds_cfg.get('target', 'skeleton') != 'smpl':
            raise ValueError("リアルタイム推定は dataset.target: smpl のモデルのみ対応しています")
        paths = get_paths(config)

        # --- モデルと統計量 (test.py と同じ) ---
        self.model = KinematicFusionSystem(config=config).to(device)
        self.model.load_state_dict(torch.load(paths.weights, map_location=device))
        self.model.eval()
        with open(paths.stats, 'r') as f:
            stats = json.load(f)
        self.subject = str(rt.get('subject') or next(iter(stats))).upper()
        if self.subject not in stats:
            raise ValueError(f"stats.json に被験者 {self.subject} がありません (あるもの: {list(stats)})。"
                             f"realtime.yaml の subject を確認してください")
        st = stats[self.subject]
        self.norm = {k: np.asarray(st[k], dtype=np.float64)
                     for k in ('lp_mean', 'lp_std', 'rp_mean', 'rp_std', 'imu_mean', 'imu_std')}

        # --- キャリブレーションの設定 (学習時と同じであることを確認) ---
        self.cc = calib.merge_config(config.get('calibration'))
        self.cc['tpose'].update((rt.get('tpose') or {}).get('override') or {})
        trained = st.get('calib_flags', 'disabled')
        if trained != calib.config_flags(self.cc):
            raise ValueError(f"キャリブレーションの設定が学習時と異なります (学習時: {trained} / "
                             f"現在: {calib.config_flags(self.cc)})")
        self.enabled = self.cc['enabled']
        self.ref = ({p: np.asarray(v) for p, v in st['tpose_gravity_ref'].items()}
                    if self.enabled and st.get('tpose_gravity_ref') else None)
        self.fallback = {
            'gravity': ({p: np.asarray(v) for p, v in st['tpose_gravity'].items()}
                        if st.get('tpose_gravity') else None),
            'press': st.get('tpose_press'),
        }

        # --- 体型 ---
        subj = {k.upper(): v for k, v in ((config.get('smpl') or {}).get('subjects') or {}).items()}.get(self.subject, {})
        betas = np.zeros(NUM_BETAS)
        b = list(subj.get('betas') or [])[:NUM_BETAS]
        betas[:len(b)] = b
        self.gender = subj.get('gender', 'neutral')
        self.betas = torch.tensor(betas, dtype=torch.float32, device=device)[None]
        self.bank = SMPLBank(config, device)
        self.J_rest = self.bank.rest_joints(self.betas, [self.gender])            # [1, 24, 3]

        # --- 時計のずれ・机置きデータ ---
        self.clock_shift = resolve_clock_shift(rt.get('clock_shift_sec', 'auto'), paths, self.subject, log)
        self.static_mode = str(rt.get('static_calib', 'file')).lower()
        self.static, self.static_source = None, None
        if self.enabled and self.static_mode == 'file':
            self.static_source, self.static = load_static_calib(str(paths.data_dir), self.subject, log)
            if self.static is None:
                log(f"[calib] 注意: {self.subject} の机置きデータが無いため、ジャイロのバイアスと圧力のゼロ点は補正しません")
        self.desk_sec = float(rt.get('desk_sec', 10.0))

        # --- 推論の設定 ---
        inf = rt.get('inference') or {}
        self.W = int(inf.get('window') or ds_cfg.get('seq_len', 200))
        self.delay = int(inf.get('delay', 30))
        self.hop = int(inf.get('hop', 5))
        self.crossfade = bool(inf.get('crossfade', True))
        if not 0 <= self.delay <= self.W - 1:
            raise ValueError(f"inference.delay は 0〜{self.W - 1} フレームにしてください: {self.delay}")
        tp = rt.get('tpose') or {}
        self.auto_tpose = bool(tp.get('auto_start', True))
        self.skip_tpose = bool(tp.get('skip', False))

        # --- 状態 ---
        self.resampler = Resampler(PARTS)
        self.max_wait = float(rt.get('max_wait_sec', 0.3))   # これ以上届かない部位は途切れたとみなす [s]
        self.last_arrival = {}
        self.frames = deque(maxlen=self.W)       # (k, t, {part: vec}) 補間後・机置き補正後 (向きの補正前)
        self.state = 'wait'
        self.seen = set()
        self.desk_buf = {p: [] for p in PARTS}
        self.desk_t0 = None
        self.tpose_buf = []
        self.align, self.scale = None, None
        self.calib_result = None
        self.last_emitted = None
        self.prev = None
        self.infer_ms = deque(maxlen=50)
        self.message = 'デバイスからの受信を待っています'
        self._tpose_requested = False

        log(f"[pipeline] 被験者 {self.subject} ({self.gender}), 窓 {self.W} / 遅延 {self.delay} / "
            f"推論間隔 {self.hop} フレーム, device {device}")
        log("[pipeline] 時計のずれの補正 [s]: " + ", ".join(f"{p} {v:+.2f}" for p, v in self.clock_shift.items()))
        if self.static_source:
            log(f"[pipeline] 机置きデータ: {self.static_source}")

    # ------------------------------------------------------------------
    # 外部からの操作
    # ------------------------------------------------------------------
    def request_tpose(self):
        """T ポーズのキャリブレーションを (やり直し) 開始する。"""
        if self.state in ('ready', 'run', 'tpose'):
            self._start_tpose()
        else:
            self._tpose_requested = True

    def _start_tpose(self):
        self.state = 'tpose'
        self.tpose_buf = []
        self._tpose_requested = False
        sec = self.cc['tpose']['search_sec']
        self.message = f"T ポーズを 3 秒以上キープしてください ({sec:.0f} 秒間のデータから探します)"
        self.log(f"[calib] {self.message}")

    # ------------------------------------------------------------------
    # サンプルの入力
    # ------------------------------------------------------------------
    def push(self, s):
        """Sample を 1 つ処理し、出力 (推論結果のまとまり) のリストを返す。"""
        if self.state == 'wait':
            self.seen.add(s.part)
            if len(self.seen) < len(PARTS):
                self.message = f"受信待ち: {', '.join(p for p in PARTS if p not in self.seen)}"
                return []
            self.log("[pipeline] 全デバイスの受信を確認しました")
            if self.enabled and self.static_mode == 'live':
                self.state = 'desk'
                self.message = f"全センサーを机に置いて静止させてください ({self.desk_sec:.0f} 秒, インソールは無負荷)"
                self.log(f"[calib] {self.message}")
            else:
                self._after_static()
            return []

        if self.state == 'desk':
            self._push_desk(s)
            return []

        imu, press = s.imu, s.press
        if self.enabled and self.static is not None:
            c = self.static.get(s.part)
            if c is not None:
                if self.cc['gyro_bias']:
                    imu = imu.copy()
                    imu[:3] -= c['gyro_bias']
                if self.cc['pressure_zero'] and press is not None and c['press_zero'] is not None:
                    press = np.clip(press - c['press_zero'], 0.0, None)
        vec = imu if press is None else np.concatenate([imu, press])
        self.resampler.push(s.part, s.t_ms / 1000.0 - self.clock_shift[s.part], vec)
        now = time.monotonic()
        self.last_arrival[s.part] = now
        stalled = {p for p, a in self.last_arrival.items() if now - a > self.max_wait}

        outputs = []
        for fr in self.resampler.pop_frames(stalled):
            self.frames.append(fr)
            if self.state == 'tpose':
                self.tpose_buf.append(fr)
                need = int(self.cc['tpose']['search_sec'] * FPS)
                self.message = (f"T ポーズを 3 秒以上キープしてください (残り "
                                f"{max(0, need - len(self.tpose_buf)) / FPS:.0f} 秒)")
                if len(self.tpose_buf) >= need:
                    self._finish_tpose()
            elif self.state == 'run':
                out = self._maybe_infer()
                if out is not None:
                    outputs.append(out)
        return outputs

    def _after_static(self):
        if not self.enabled or self.skip_tpose:
            self._set_calibration(*self._fallback_calibration('T ポーズを使わない設定'))
            return
        if self.auto_tpose or self._tpose_requested:
            self._start_tpose()
        else:
            self.state = 'ready'
            self.message = 'センサーを装着したら「T ポーズ開始」を押してください'
            self.log(f"[calib] {self.message} (コンソールでは Enter)")

    def _push_desk(self, s):
        t = s.t_ms / 1000.0
        if self.desk_t0 is None:
            self.desk_t0 = t
        self.desk_buf[s.part].append((s.t_ms, s.imu, s.press))
        left = self.desk_sec - (t - self.desk_t0)
        self.message = f"全センサーを机に置いて静止させてください (残り {max(0.0, left):.0f} 秒)"
        if left > 0 or any(len(v) < 10 for v in self.desk_buf.values()):
            return
        raw = {p: (np.array([x[0] for x in v]), np.stack([x[1] for x in v]),
                   np.stack([x[2] for x in v]) if p in FOOT_PARTS else None)
               for p, v in self.desk_buf.items()}
        self.static = calib.static_calibration(raw)
        self.static_source = 'live'
        moving = {p: round(c['gyro_p95'], 2) for p, c in self.static.items() if c['gyro_p95'] > 5.0}
        self.log("[calib] 机置きのキャリブレーション完了 (ジャイロのバイアス "
                 + ", ".join(f"{p} {np.linalg.norm(c['gyro_bias']):.2f}" for p, c in self.static.items()) + " deg/s)")
        if moving:
            self.log(f"  警告: 角速度が大きい部位があります {moving} deg/s (静止していない可能性)")
        self.desk_buf = {p: [] for p in PARTS}
        self.auto_tpose = False      # 装着してから T ポーズを始める
        self._after_static()

    # ------------------------------------------------------------------
    # T ポーズ
    # ------------------------------------------------------------------
    def _fallback_calibration(self, reason):
        fb = self.fallback
        return fb['gravity'], fb['press'], f"{reason} → 学習データの平均 (stats)", None

    def _finish_tpose(self):
        tc = self.cc['tpose']
        t = np.array([f[1] for f in self.tpose_buf])
        raw = {p: (None,
                   np.stack([f[2][p][:6] for f in self.tpose_buf]),
                   np.stack([f[2][p][6:] for f in self.tpose_buf]) if p in FOOT_PARTS else None)
               for p in PARTS}
        sens_t = {p: t for p in PARTS}
        cands = []
        for t0, t1 in calib.find_static_runs(sens_t, raw, (t[0], t[-1] + 1e-6), tc):
            w = calib.window_stats(sens_t, raw, t0, t1)
            if w is None:
                continue
            w['angle'] = {p: calib.angle_deg(w['gravity'][p], self.ref[p]) for p in PARTS} if self.ref else None
            w['accepted'] = self.ref is not None and max(w['angle'].values()) <= tc['max_angle_deg']
            cands.append(w)
        acc = [w for w in cands if w['accepted']]
        info = [{'t_sec': [round(w['t'][0] - t[0], 2), round(w['t'][1] - t[0], 2)], 'accepted': w['accepted'],
                 'max_angle_deg': round(max(w['angle'].values()), 1) if w['angle'] else None} for w in cands]
        if acc:
            w = max(acc, key=lambda w: w['t'][1] - w['t'][0])
            res = (w['gravity'], w['press'],
                   f"T ポーズ {w['t'][0] - t[0]:.1f}〜{w['t'][1] - t[0]:.1f} s", info)
        else:
            why = ('静止区間が見つかりません' if not cands
                   else f"静止区間はあるが T ポーズの向きと違います (最大 {min(i['max_angle_deg'] for i in info)}° 以上)")
            res = self._fallback_calibration(f"T ポーズ無し ({why})")
            res = res[:3] + (info,)
        self._set_calibration(*res)

    def _set_calibration(self, gravity, press, source, cands):
        cc = self.cc
        self.align, align_deg = None, {}
        if cc['imu_align'] and gravity is not None and self.ref is not None:
            self.align = {p: calib.rotation_between(gravity[p], self.ref[p]) for p in PARTS}
            align_deg = {p: round(calib.angle_deg(gravity[p], self.ref[p]), 1) for p in PARTS}
        self.scale = None
        if cc['pressure_scale'] != 'none' and press:
            if cc['pressure_scale'] == 'total' and press['FL'] + press['FR'] > 0:
                self.scale = {f: 100.0 / (press['FL'] + press['FR']) for f in FOOT_PARTS}
            elif cc['pressure_scale'] == 'per_foot' and min(press.values()) > 0:
                self.scale = {f: 50.0 / press[f] for f in FOOT_PARTS}
        self.calib_result = {
            'source': source, 'imu_align_deg': align_deg or None,
            'press_sum': {f: round(float(v), 1) for f, v in press.items()} if press else None,
            'pressure_scale': self.scale, 'candidates': cands, 'static': self.static_source,
        }
        self.log(f"[calib] {source}")
        if align_deg:
            self.log("  取り付け向きの補正角: " + ", ".join(f"{p} {v:.1f}°" for p, v in align_deg.items()))
        if press:
            self.log(f"  T ポーズの圧力合計: 左 {press['FL']:.0f} / 右 {press['FR']:.0f}")
        self.state = 'run'
        self.prev = None
        self.message = '推定中'

    # ------------------------------------------------------------------
    # 正規化・推論
    # ------------------------------------------------------------------
    def normalized_window(self, frames):
        """補間後のフレーム列 → モデルの入力 (lp [T, 112], rp [T, 112], imu [T, 5, 6]) (正規化済み)。"""
        imu = np.stack([[f[2][p][:6] for p in IMU_ORDER] for f in frames])          # [T, 5, 6]
        lp = np.stack([f[2]['FL'][6:] for f in frames])
        rp = np.stack([f[2]['FR'][6:] for f in frames])
        if self.align is not None:
            for i, p in enumerate(IMU_ORDER):
                R = self.align[p]
                imu[:, i, 0:3] = imu[:, i, 0:3] @ R.T
                imu[:, i, 3:6] = imu[:, i, 3:6] @ R.T
        if self.scale is not None:
            lp, rp = lp * self.scale['FL'], rp * self.scale['FR']
        n = self.norm
        return ((lp - n['lp_mean']) / n['lp_std'], (rp - n['rp_mean']) / n['rp_std'],
                (imu - n['imu_mean']) / n['imu_std'])

    def _maybe_infer(self):
        if len(self.frames) < self.W:
            return None
        newest = self.frames[-1][0]
        target = newest - self.delay
        if self.last_emitted is not None and target - self.last_emitted < self.hop:
            return None
        k0 = newest - self.W + 1
        if self.last_emitted is None:
            first = max(k0, target - self.hop + 1)
        else:
            first = max(self.last_emitted + 1, k0)   # 推論が遅れたときは、その間のフレームもまとめて出力する
        frames = list(self.frames)
        lp, rp, imu = self.normalized_window(frames)

        t_start = time.perf_counter()
        dev = self.device
        with torch.no_grad():
            f32 = lambda x: torch.as_tensor(x, dtype=torch.float32, device=dev)[None]  # noqa: E731
            trans, pose6d, contact, _ = self.model(f32(lp)[..., None], f32(rp)[..., None], f32(imu), self.betas)
            trans, pose6d, contact = trans[0], pose6d[0], torch.sigmoid(contact[0])
            idx = torch.arange(first - k0, target - k0 + 1, device=dev)
            tr, p6, c = trans[idx], pose6d[idx], contact[idx]
            if self.crossfade and self.prev is not None:
                pk0, ptr, pp6, pc = self.prev
                ks = torch.arange(first, target + 1, device=dev)
                n = len(ks)
                ok = (ks >= pk0) & (ks < pk0 + self.W)
                alpha = torch.where(ok, torch.arange(1, n + 1, device=dev, dtype=torch.float32) / n,
                                    torch.ones(n, device=dev))
                pi = (ks - pk0).clamp(0, self.W - 1)
                a = alpha[:, None]
                tr = (1 - a) * ptr[pi] + a * tr
                p6 = (1 - a[..., None]) * pp6[pi] + a[..., None] * p6
                c = (1 - a) * pc[pi] + a * c
            self.prev = (k0, trans, pose6d, contact)
            R = rot6d_to_matrix(p6)                                                  # [n, 24, 3, 3]
            G, J = self.bank.forward_kinematics(R, self.J_rest)
            J = J + tr[:, None]
            # メッシュ変形 (線形ブレンドスキニング) 用の変換 A_j = [G_j | J_j - G_j J_rest_j]
            t_j = J - (G @ self.J_rest[0][None, :, :, None])[..., 0]
            A = torch.cat([G, t_j[..., None]], dim=-1)                               # [n, 24, 3, 4]
            aa = matrix_to_axis_angle(R).reshape(len(R), -1)
            if dev.type == 'cuda':
                torch.cuda.synchronize()
        self.infer_ms.append((time.perf_counter() - t_start) * 1000)
        self.last_emitted = target

        t_of = {f[0]: f[1] for f in frames}
        latest = frames[-1][2]
        return {
            'k': list(range(first, target + 1)),
            't': [t_of[k] for k in range(first, target + 1)],
            'A': A.reshape(len(A), -1).cpu().numpy(),
            'joints': J.cpu().numpy(),
            'contact': c.cpu().numpy(),
            'poses': aa.cpu().numpy(),
            'trans': tr.cpu().numpy(),
            'pressure': {f: (latest[f][6:] * (self.scale[f] if self.scale else 1.0)) for f in FOOT_PARTS},
        }

    # ------------------------------------------------------------------
    # 表示用
    # ------------------------------------------------------------------
    def status(self):
        im = list(self.infer_ms)
        return {
            'state': self.state, 'message': self.message,
            'infer_ms': round(float(np.median(im)), 1) if im else None,
            'held_frames': dict(self.resampler.held),
            'calib': self.calib_result,
            'progress': (len(self.tpose_buf) / (self.cc['tpose']['search_sec'] * FPS)
                         if self.state == 'tpose' else None),
        }

    def viewer_model(self):
        """ブラウザでメッシュを変形するためのデータ (体型 β のレストポーズ頂点・面・スキニングの重み 上位 4 つ)。"""
        smpl = self.bank._model(self.gender)
        v, _ = smpl.shaped(self.betas)
        w = smpl.weights
        top_w, top_i = w.topk(4, dim=1)
        top_w = top_w / top_w.sum(1, keepdim=True)
        return {
            'verts': np.round(v[0].cpu().numpy().reshape(-1), 5).tolist(),
            'faces': smpl.faces.cpu().numpy().reshape(-1).astype(int).tolist(),
            'skin_idx': top_i.cpu().numpy().reshape(-1).astype(int).tolist(),
            'skin_w': np.round(top_w.cpu().numpy().reshape(-1), 4).tolist(),
            'parents': PARENTS,
            'subject': self.subject, 'gender': self.gender,
            'window': self.W, 'delay': self.delay, 'hop': self.hop, 'fps': FPS,
            'pressure_shape': [8, 14], 'num_pressure': NUM_PRESSURE,
        }
