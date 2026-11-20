"""
リアルタイム動作のシミュレーション (target: smpl)。

学習済みモデルをそのまま使い、テストデータを 1 フレームずつ届くストリームとして扱って、
「直近 window フレームの窓で推論し、窓の末尾から delay フレーム前の推定を出力する」方式の
精度と遅延を、オフライン推論 (test.py と同じ 2 秒チャンク) と比べる。

    uv run scripts/realtime_sim.py                              # 既定: 窓 200, 遅延 0/5/10/20/30/50/100 フレーム
    uv run scripts/realtime_sim.py --delays 0 20 50 --hop 5     # 5 フレーム (50 ms) ごとに推論して 5 フレームずつ出力
    uv run scripts/realtime_sim.py --ids D2S1T1 --window 100    # 窓を 1 秒にする

方式 (window = W, delay = d, hop = h):
    フレーム e まで受信した時点で、窓 [e-W+1, e] を推論し、窓内の位置 W-1-d-k (k = 0..h-1) の推定を
    フレーム e-d-k の出力とする。h フレームごとに推論するので、出力の遅延は d 〜 d+h-1 フレーム
    (+ 推論の計算時間)。h = 1 なら毎フレーム推論し、遅延はちょうど d フレーム。
    d = 0 は窓の最後のフレーム (未来の情報なし)、d = W/2 付近なら前後の文脈がそろう。

    評価するフレームは、全ての遅延で出力がそろう範囲 (最初の窓がたまるまでの W-1 フレームと、
    最大の遅延の分の末尾を除く) に限り、オフライン推論もその範囲で比べる。

この評価で分かること / 分からないこと:
    - モデルの構造 (双方向 GRU・マスクなしの Transformer) が未来のフレームを使うことによる精度の低下を測る
    - 前処理 (時刻同期・キャリブレーション・正規化) はオフラインと同じ値を使う。T ポーズは収録の最初のものを
      使う設定 (calibration.tpose.prefer: start) ならリアルタイムでも同じ手順で求められるが、
      インソールと IMU の時計のずれ (sync.groups) は mocap から推定した値なので、実機では別の方法で求める必要がある
    - 通信・センサーの遅延は含まない

オフラインのパイプライン (dataset/, models/, test.py, evaluate.py) は変更せず、読み込むだけ。
出力 (オフラインの結果は上書きしない):
    <output_dir>/realtime/realtime_eval_w{window}_h{hop}.txt   結果の表
    <output_dir>/realtime/realtime_eval_w{window}_h{hop}.csv   同じ内容の CSV
"""
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset.dataset import InsoleIMUDataset                                   # noqa: E402
from models import KinematicFusionSystem                                       # noqa: E402
from utils.paths import load_config, get_paths, resolve_ids, DEFAULT_CONFIG    # noqa: E402
from utils.rotation import rot6d_to_matrix, geodesic_angle                     # noqa: E402
from utils.smpl import SMPLBank, BODY_JOINTS                                   # noqa: E402
from evaluate import (contact_scores, _global_rot, _jerk, _foot_speed,         # noqa: E402
                      SIP_JOINTS, SMPL_PARTS, FPS)

INPUT_KEYS = ('left_pressure', 'right_pressure', 'imu_data')
GT_KEYS = ('gt_smpl_pose', 'gt_smpl_trans', 'gt_smpl_joints', 'gt_contact')


# ----------------------------------------------------------------------
# データ: テスト用データセットのチャンクを収録ごとの連続したストリームに戻す
# ----------------------------------------------------------------------
def load_streams(dataset):
    """
    InsoleIMUDataset (mode='test') のチャンクは収録ごとに先頭から seq_len ずつ重なりなく並んでいるので、
    収録ごとに連結すれば正規化済みの連続した入力・正解に戻る (オフライン推論と同じフレームが対象になる)。
    """
    L = dataset.seq_len
    counts = []
    for did in dataset.data_ids:
        n = int(dataset.sync_info[did]['frames_kept'].split('/')[0])
        counts.append(len(range(0, n - L, L)))
    if sum(counts) != len(dataset):
        raise RuntimeError(f"チャンク数が収録のフレーム数と合いません ({sum(counts)} != {len(dataset)})。"
                           f"dataset.py のチャンク分割が変わった可能性があります")
    streams, i = [], 0
    for did, c in zip(dataset.data_ids, counts):
        items = [dataset[k] for k in range(i, i + c)]
        i += c
        if not items:
            continue
        s = {k: torch.cat([it[k] for it in items]) for k in INPUT_KEYS + GT_KEYS}
        s.update(data_id=did, subject=items[0]['data_id'], gender=items[0]['gender'],
                 body_params=items[0]['body_params'])
        streams.append(s)
    return streams


# ----------------------------------------------------------------------
# 推論
# ----------------------------------------------------------------------
def _forward(model, s, idx, device):
    """idx [B, W] (フレーム番号) の窓をまとめて推論する。"""
    idx = torch.as_tensor(idx, device=device)
    lp, rp, imu = (s['dev'][k][idx] for k in INPUT_KEYS)
    bp = s['dev']['body_params'].expand(idx.shape[0], -1)
    trans, pose6d, contact, _ = model(lp, rp, imu, bp)
    return trans, pose6d, torch.sigmoid(contact)


def _empty(N):
    return dict(trans=torch.full((N, 3), float('nan')),
                pose6d=torch.full((N, 24, 6), float('nan')),
                contact=torch.full((N, 2), float('nan')))


def run_offline(model, s, L, batch, device):
    """test.py と同じ: 重なりのない L フレームのチャンクごとに推論する。"""
    N = len(s['imu_data'])
    out = _empty(N)
    starts = np.arange(0, N - L + 1, L)
    for b in range(0, len(starts), batch):
        st = starts[b:b + batch]
        idx = st[:, None] + np.arange(L)[None]
        tr, p6, c = _forward(model, s, idx, device)
        f = torch.as_tensor(idx.reshape(-1))
        out['trans'][f] = tr.reshape(-1, 3).cpu()
        out['pose6d'][f] = p6.reshape(-1, 24, 6).cpu()
        out['contact'][f] = c.reshape(-1, 2).cpu()
    return out


def run_streaming(model, s, W, hop, delays, batch, device):
    """窓をずらしながら推論し、遅延ごとの出力を返す {delay: {trans, pose6d, contact}}。"""
    N = len(s['imu_data'])
    out = {d: _empty(N) for d in delays}
    ends = np.arange(W - 1, N, hop)
    ks = np.arange(hop)
    for b in range(0, len(ends), batch):
        e = ends[b:b + batch]
        idx = (e[:, None] - W + 1) + np.arange(W)[None]
        tr, p6, c = _forward(model, s, idx, device)
        tr, p6, c = tr.cpu(), p6.cpu(), c.cpu()
        for d in delays:
            pos = torch.as_tensor(W - 1 - d - ks)                       # 窓内の位置 [h]
            f = torch.as_tensor((e[:, None] - d - ks[None]).reshape(-1))  # 出力するフレーム [B*h]
            out[d]['trans'][f] = tr[:, pos].reshape(-1, 3)
            out[d]['pose6d'][f] = p6[:, pos].reshape(-1, 24, 6)
            out[d]['contact'][f] = c[:, pos].reshape(-1, 2)
    return out, ends


def measure_compute(model, s, W, device, n=50):
    """1 回の更新 (バッチ 1, 窓 W) にかかる計算時間 [ms] (中央値, 95 パーセンタイル)。"""
    idx = np.arange(W)[None]
    times = []
    with torch.no_grad():
        for i in range(n + 10):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _forward(model, s, idx, device)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            if i >= 10:
                times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times)), float(np.percentile(times, 95))


# ----------------------------------------------------------------------
# 評価 (evaluate.py の SMPL の指標と同じ定義)
# ----------------------------------------------------------------------
def frame_metrics(pred, s, frames, bank, chunk=8192):
    """frames の範囲の予測について、フレームごとの誤差と関節位置を返す。"""
    dev = bank.device
    bp = s['body_params'][None].to(dev)
    J_rest = bank.rest_joints(bp, [s['gender']])                    # [1, 24, 3]
    pj, ld, gd = [], [], []
    for i in range(0, len(frames), chunk):
        f = frames[i:i + chunk]
        Rp = rot6d_to_matrix(pred['pose6d'][f].to(dev))
        Rg = rot6d_to_matrix(s['gt_smpl_pose'][f].to(dev))
        _, J = bank.forward_kinematics(Rp, J_rest)
        pj.append((J + pred['trans'][f].to(dev)[:, None]).cpu())
        ld.append(torch.rad2deg(geodesic_angle(Rp, Rg)).cpu())
        gd.append(torch.rad2deg(geodesic_angle(_global_rot(Rp), _global_rot(Rg))).cpu())
    pj = torch.cat(pj).numpy()
    gj = s['gt_smpl_joints'][frames].numpy()
    return dict(
        pj=pj, gj=gj,
        err=np.linalg.norm(pj - gj, axis=-1) * 1000,
        pa=np.linalg.norm((pj - pj[:, :1]) - (gj - gj[:, :1]), axis=-1) * 1000,
        loc=torch.cat(ld).numpy(), glob=torch.cat(gd).numpy(),
        root_h=np.abs(pred['trans'][frames, 1].numpy() - s['gt_smpl_trans'][frames, 1].numpy()) * 1000,
        p_contact=pred['contact'][frames].numpy(), g_contact=s['gt_contact'][frames].numpy(),
    )


def summarize(per_stream):
    """収録ごとの frame_metrics をまとめて指標にする (ジャーク・足の速さは収録ごとに計算してフレーム数で重み付け)。"""
    cat = {k: np.concatenate([m[k] for m in per_stream]) for k in
           ('err', 'pa', 'loc', 'glob', 'root_h', 'p_contact', 'g_contact')}
    n = np.array([len(m['err']) for m in per_stream], dtype=float)
    w = n / n.sum()
    body = BODY_JOINTS
    res = {
        'MPJPE': cat['err'][:, body].mean(),
        'PA-MPJPE': cat['pa'][:, body].mean(),
        'Local': cat['loc'][:, body].mean(),
        'Global': cat['glob'][:, body].mean(),
        'SIP': cat['glob'][:, SIP_JOINTS].mean(),
        'RootH': cat['root_h'].mean(),
        'Jerk': float(np.sum(w * [_jerk(m['pj']) for m in per_stream])),
        'FootSpd': float(np.sum(w * [_foot_speed(m['pj'], m['g_contact']) for m in per_stream])),
    }
    for i, side in enumerate(('L', 'R')):
        res[f'F1_{side}'] = contact_scores(cat['p_contact'][:, i], cat['g_contact'][:, i])[1]
    for name, idx in SMPL_PARTS.items():
        res[f'part:{name}'] = cat['err'][:, idx].mean()
    res['_gt_jerk'] = float(np.sum(w * [_jerk(m['gj']) for m in per_stream]))
    res['_gt_footspd'] = float(np.sum(w * [_foot_speed(m['gj'], m['g_contact']) for m in per_stream]))
    return res


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--ids', nargs='+', default=None,
                        help='テストに使う収録 ID (ワイルドカード可)。既定は test.py と同じ')
    parser.add_argument('--window', type=int, default=None, help='窓の長さ [フレーム] (既定: dataset.seq_len)')
    parser.add_argument('--delays', type=int, nargs='+', default=[0, 5, 10, 20, 30, 50, 100],
                        help='出力の遅延 [フレーム, 100 Hz]')
    parser.add_argument('--hop', type=int, default=1, help='推論する間隔 [フレーム] (既定 1 = 毎フレーム)')
    parser.add_argument('--batch', type=int, default=64, help='シミュレーションでまとめて推論する窓の数')
    parser.add_argument('--device', default=None, help='cuda / cpu (既定: 使えれば cuda)')
    args = parser.parse_args()

    config = load_config(args.config)
    paths = get_paths(config)
    ds_cfg = config.get('dataset', {})
    if ds_cfg.get('target', 'skeleton') != 'smpl':
        raise SystemExit("このスクリプトは dataset.target: smpl の場合のみ対応しています")
    L = ds_cfg.get('seq_len', 200)
    W = args.window or L
    hop = args.hop
    delays = sorted(set(args.delays))
    bad = [d for d in delays if d < 0 or d + hop - 1 > W - 1]
    if bad:
        raise SystemExit(f"遅延 {bad} は使えません (0 <= delay かつ delay + hop - 1 <= window - 1 = {W - 1})")

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if device.type == 'cpu' and hop == 1:
        print("[realtime] 注意: CPU で毎フレーム推論するシミュレーションは時間がかかります (--hop 5 などを推奨)")

    model = KinematicFusionSystem(config=config).to(device)
    model.load_state_dict(torch.load(paths.weights, map_location=device))
    model.eval()
    with open(paths.stats, 'r') as f:
        stats = json.load(f)

    # test.py と同じ設定でテストデータを読み込む (ファイルへの書き出しはしない)
    dataset = InsoleIMUDataset(
        str(paths.data_dir), seq_len=L, mode='test', stats=stats,
        sync_cfg=config.get('sync'), target='smpl', smpl_cfg=config.get('smpl'),
        calib_cfg=config.get('calibration'),
        ids=resolve_ids(config, 'test', args.ids),
        contact_source=ds_cfg.get('contact_source', 'pressure'),
        canonical_heading=ds_cfg.get('canonical_heading', True),
        floor_normalize=ds_cfg.get('floor_normalize', True))
    streams = load_streams(dataset)
    bank = SMPLBank(config, device)

    modes = ['offline'] + [f'd={d}' for d in delays]
    per_mode = {m: [] for m in modes}
    n_eval = 0
    t_start = time.perf_counter()
    for s in streams:
        N = len(s['imu_data'])
        s['dev'] = {k: s[k].to(device) for k in INPUT_KEYS}
        s['dev']['body_params'] = s['body_params'][None].to(device)
        print(f"[realtime] {s['data_id']}: {N} フレーム, 窓 {W}, hop {hop}, 遅延 {delays} を推論中...")
        with torch.no_grad():
            off = run_offline(model, s, L, args.batch, device)
            stream, ends = run_streaming(model, s, W, hop, delays, args.batch, device)
        last = int(ends[-1]) - max(delays)
        frames = torch.arange(W - 1, last + 1)
        if len(frames) < 4:
            print(f"  {s['data_id']}: 評価できるフレームがないためスキップ")
            continue
        n_eval += len(frames)
        preds = {'offline': off, **{f'd={d}': stream[d] for d in delays}}
        for m in modes:
            if torch.isnan(preds[m]['trans'][frames]).any():
                raise RuntimeError(f"{m}: 評価範囲に推定のないフレームがあります")
            per_mode[m].append(frame_metrics(preds[m], s, frames, bank))
        del s['dev']
    print(f"[realtime] シミュレーション完了 ({time.perf_counter() - t_start:.0f} 秒)")

    # 計算時間 (バッチ 1)
    s0 = streams[0]
    s0['dev'] = {k: s0[k].to(device) for k in INPUT_KEYS}
    s0['dev']['body_params'] = s0['body_params'][None].to(device)
    comp_med, comp_p95 = measure_compute(model, s0, W, device)

    results = {m: summarize(per_mode[m]) for m in modes}
    write_report(paths, args, W, L, hop, delays, device, dataset, streams, n_eval,
                 comp_med, comp_p95, results)


def write_report(paths, args, W, L, hop, delays, device, dataset, streams, n_eval,
                 comp_med, comp_p95, results):
    out_dir = paths.output_dir / 'realtime'
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"realtime_eval_w{W}_h{hop}"
    gpu = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'

    def latency(m):
        if m == 'offline':
            return '-'
        d = int(m.split('=')[1])
        return f"{d / FPS:.2f}" + (f"-{(d + hop - 1) / FPS:.2f}" if hop > 1 else '')

    cols = [('MPJPE', 'MPJPE', '{:.2f}'), ('PA-MPJPE', 'PA-MPJPE', '{:.2f}'), ('Local', 'Local', '{:.2f}'),
            ('Global', 'Global', '{:.2f}'), ('SIP', 'SIP', '{:.2f}'), ('RootH', 'RootH', '{:.2f}'),
            ('Jerk', 'Jerk', '{:.0f}'), ('FootSpd', 'FootSpd', '{:.2f}'),
            ('F1_L', 'F1 L', '{:.3f}'), ('F1_R', 'F1 R', '{:.3f}')]
    head = f"{'mode':<9} | {'delay [s]':>9} | " + " | ".join(f"{h:>8}" for _, h, _ in cols)
    calib = {did: dataset.calib_info.get(did, {}).get('source', '-') for did in dataset.data_ids}
    lines = [
        "=" * len(head),
        "Realtime simulation (sliding window)",
        "=" * len(head),
        f"Recordings          : {', '.join(s['data_id'] for s in streams)}",
        f"Evaluated frames    : {n_eval} (各収録の最初の {W - 1} フレーム (窓がたまるまで) と末尾 {max(delays)} フレームを除く)",
        f"Window / hop        : {W} フレーム ({W / FPS:.2f} s) / {hop} フレーム ({hop / FPS * 1000:.0f} ms ごとに推論)",
        f"Offline baseline    : test.py と同じ {L} フレームの重なりのないチャンク",
        f"Compute per update  : {comp_med:.1f} ms (median), {comp_p95:.1f} ms (p95)  [batch 1, {gpu}]",
        f"  -> 処理できる更新頻度の上限 ≒ {1000 / comp_med:.0f} Hz (センサーは 100 Hz)",
        f"Calibration source  : {calib}",
        "",
        "delay [s]: 窓の末尾から何秒前のフレームを出力するか (hop > 1 のときは最新〜最古の出力の遅延)。",
        "           実際の遅延 = delay + 計算時間 + 通信・センサーの遅延",
        "MPJPE / PA-MPJPE / RootH [mm], Local / Global / SIP [deg], Jerk [m/s^3], FootSpd [cm/s], F1: 接地",
        "-" * len(head),
        head,
        "-" * len(head),
    ]
    for m, r in results.items():
        lines.append(f"{m:<9} | {latency(m):>9} | " + " | ".join(f"{f.format(r[k]):>8}" for k, _, f in cols))
    r0 = next(iter(results.values()))
    lines += ["-" * len(head),
              f"GT: Jerk {r0['_gt_jerk']:.0f}, FootSpd {r0['_gt_footspd']:.2f}",
              "",
              "部位ごとの MPJPE [mm]",
              f"{'mode':<9} | " + " | ".join(f"{p:>9}" for p in SMPL_PARTS),
              "-" * (12 + 12 * len(SMPL_PARTS))]
    for m, r in results.items():
        lines.append(f"{m:<9} | " + " | ".join(f"{r['part:' + p]:>9.2f}" for p in SMPL_PARTS))
    lines.append("=" * len(head))

    text = "\n".join(lines)
    print("\n" + text)
    (out_dir / f'{stem}.txt').write_text(text, encoding='utf-8')

    keys = [k for k, _, _ in cols] + [f'part:{p}' for p in SMPL_PARTS]
    csv = ["mode,delay_frames,hop,window," + ",".join(keys)]
    for m, r in results.items():
        d = '' if m == 'offline' else m.split('=')[1]
        csv.append(f"{m},{d},{hop},{W}," + ",".join(f"{r[k]:.4f}" for k in keys))
    (out_dir / f'{stem}.csv').write_text("\n".join(csv) + "\n", encoding='utf-8')
    print(f"\n結果を {out_dir / (stem + '.txt')} と {stem}.csv に保存しました")


if __name__ == '__main__':
    main()
