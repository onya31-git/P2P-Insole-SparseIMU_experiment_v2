"""
mocap の BVH を SMPL のパラメータにフィッティングし、SMPL 形式のデータセットを作る。

入力:
    data/skeleton/{train,test}/<ID>_skeleton.bvh   (scripts/prepare_data.py が raw からコピー)
    config.yaml の smpl.subjects.<D?S?>            (被験者ごとの性別と体型 β。β は固定値として使う)
出力:
    data/smpl/{train,test}/<ID>_smpl.npz            (AMASS 互換のキー + 確認用の値)
        poses           [T, 72]   SMPL の姿勢 (軸角, 局所回転)。手 (22, 23) は単位回転
        trans           [T, 3]    並進 [m] (mocap のワールド座標, y が鉛直)
        betas           [10]      体型 (config の値そのまま)
        gender                    'male' / 'female' / 'neutral'
        mocap_framerate           100
        joints          [T, 24, 3] SMPL の関節位置 [m] (poses, trans, betas から計算した値)
        offsets         [24, 3]   SMPL 関節 -> mocap 関節の定義の差 (親関節の座標系, m)
        offset_mask     [24]      位置の対応がある関節 (True)
        mocap_joint_names [24]    対応する BVH の関節名
        fit_err_mm      [T, 24]   関節位置のフィッティング誤差 [mm] (対応が無い関節は NaN)
        ori_err_deg     [T, 24]   BVH のグローバル回転との角度差 [deg]
    data/smpl/fit_report.txt      フィッティング誤差のまとめ (骨長の比較を含む)

手順 (被験者ごと):
    0. 初期値 (リターゲット): BVH のグローバル回転 G_b に、レストポーズの骨の向きの違いを
       補正する定数回転 C を掛けて SMPL のグローバル回転 G_s = G_b C とし、局所回転に直す
    1. 段階 1: 全収録から間引いたフレームで、姿勢・並進と「関節の定義の差」
       (オフセット: SMPL 関節から見た mocap 関節の位置, 被験者ごとに一定) を同時に推定する
    2. 段階 2: オフセットを固定し、収録ごとに全フレームの姿勢・並進を推定する
    損失: 関節位置, BVH のグローバル回転 (骨軸まわりのねじれを決める), 膝・肘のヒンジ性,
          位置の対応が無い関節を初期値の近くに保つ項, 関節の加速度 (段階 2 のみ, 既定は無効)
    ※ 加速度の項はフレーム間を結合するため収束が大きく遅れる。mocap (100 Hz) 自体が滑らかで、
      無しでもジッターは mocap と同程度 (レポートの「ジッター比」) なので既定の重みは 0。
    β はフィッティングでは変えない (config の値をそのまま使う)。

使い方:
    uv run scripts/fit_smpl.py                  # 未作成の収録だけフィッティング
    uv run scripts/fit_smpl.py --overwrite      # すべて作り直す
    uv run scripts/fit_smpl.py --ids D1S1A1     # 指定した収録 (と同じ被験者の段階 1) のみ
"""
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset.dataset import subject_key                                   # noqa: E402
from utils.bvh import load_bvh                                            # noqa: E402
from utils.paths import load_config, get_paths, resolve, DEFAULT_CONFIG   # noqa: E402
from utils.rotation import (rot6d_to_matrix, matrix_to_rot6d, matrix_to_axis_angle,  # noqa: E402
                            geodesic_angle, minimal_rotation)
from utils.smpl import SMPL, NUM_JOINTS, PARENTS, JOINT_NAMES, NUM_BETAS  # noqa: E402

# ----------------------------------------------------------------------
# SMPL 関節と BVH 関節の対応
#   (BVH 関節, 位置の重み, 回転の重み, 回転の補正 C の決め方)
#   補正: (SMPL の子関節, BVH の子関節) -> レストポーズでの骨の向きを一致させる最小回転
#         'same:k' -> SMPL 関節 k と同じ補正 (局所回転を BVH と同じ意味にする)
#         None     -> 補正なし (C = I。胴体などレストポーズの向きが同じとみなす関節)
# ----------------------------------------------------------------------
CORR = {
    0:  ('Hips',           0.5, 0.5, None),
    1:  ('LeftUpLeg',      0.5, 1.0, (4, 'LeftLeg')),
    2:  ('RightUpLeg',     0.5, 1.0, (5, 'RightLeg')),
    3:  ('Spine1',         0.2, 0.3, None),
    4:  ('LeftLeg',        1.0, 1.0, (7, 'LeftFoot')),
    5:  ('RightLeg',       1.0, 1.0, (8, 'RightFoot')),
    6:  ('Spine3',         0.2, 0.3, None),
    7:  ('LeftFoot',       1.0, 1.0, (10, 'LeftToeBase')),
    8:  ('RightFoot',      1.0, 1.0, (11, 'RightToeBase')),
    9:  ('Spine4',         0.3, 0.5, None),
    10: ('LeftToeBase',    1.0, 0.3, 'same:7'),
    11: ('RightToeBase',   1.0, 0.3, 'same:8'),
    12: ('Neck',           0.3, 0.5, None),
    13: ('LeftShoulder',   0.3, 0.5, (16, 'LeftArm')),
    14: ('RightShoulder',  0.3, 0.5, (17, 'RightArm')),
    15: ('Head',           0.3, 0.5, None),
    16: ('LeftArm',        0.7, 1.0, (18, 'LeftForeArm')),
    17: ('RightArm',       0.7, 1.0, (19, 'RightForeArm')),
    18: ('LeftForeArm',    1.0, 1.0, (20, 'LeftHand')),
    19: ('RightForeArm',   1.0, 1.0, (21, 'RightHand')),
    20: ('LeftHand',       1.0, 1.0, (22, 'LeftHand_End')),
    21: ('RightHand',      1.0, 1.0, (23, 'RightHand_End')),
    22: ('LeftHand_End',   0.3, 0.0, 'same:20'),
    23: ('RightHand_End',  0.3, 0.0, 'same:21'),
}
HINGE = [4, 5, 18, 19]                    # 膝・肘: 局所 z 軸まわり (外転方向) の回転を抑える
PRIOR = [3, 6, 9, 10, 11, 12, 15, 22, 23]  # 位置だけでは決まりにくい関節 (初期値の近くに保つ)
PARENT_FRAME = [0 if p < 0 else p for p in PARENTS]   # オフセットは親関節の座標系で持つ

# 骨長の比較 (β の確認用): (名前, SMPL 関節 a, b, BVH 関節 a, b)
BONES = [
    ('大腿 (左)', 1, 4, 'LeftUpLeg', 'LeftLeg'), ('下腿 (左)', 4, 7, 'LeftLeg', 'LeftFoot'),
    ('足 (左)', 7, 10, 'LeftFoot', 'LeftToeBase'),
    ('上腕 (左)', 16, 18, 'LeftArm', 'LeftForeArm'), ('前腕 (左)', 18, 20, 'LeftForeArm', 'LeftHand'),
    ('骨盤幅', 1, 2, 'LeftUpLeg', 'RightUpLeg'), ('肩幅', 16, 17, 'LeftArm', 'RightArm'),
    ('股関節中央-首', None, 12, None, 'Neck'),
]


def log(msg, report):
    print(msg)
    report.append(msg)


# ----------------------------------------------------------------------
# 準備
# ----------------------------------------------------------------------
def bvh_targets(bvh, device):
    """SMPL 関節の順に並べた BVH のグローバル回転 [T, 24, 3, 3] と位置 [T, 24, 3] (m)。"""
    G, P = bvh.forward_kinematics(device=device, dtype=torch.float64)
    idx = [bvh.index[CORR[s][0]] for s in range(NUM_JOINTS)]
    return G[:, idx].float(), (P[:, idx] / 1000.0).float()


def rest_correction(J_rest, bvh, device):
    """レストポーズの骨の向きの違いを補正する定数回転 C [24, 3, 3]。"""
    Pb = torch.as_tensor(bvh.rest_positions() / 1000.0, dtype=torch.float32, device=device)
    C = torch.eye(3, device=device).repeat(NUM_JOINTS, 1, 1)
    for s, (b, _, _, rule) in CORR.items():
        if isinstance(rule, tuple):
            sc, bc = rule
            C[s] = minimal_rotation(J_rest[sc] - J_rest[s], Pb[bvh.index[bc]] - Pb[bvh.index[b]])
    for s, (_, _, _, rule) in CORR.items():
        if isinstance(rule, str) and rule.startswith('same:'):
            C[s] = C[int(rule.split(':')[1])]
    return C


def retarget(G_b, C):
    """SMPL の目標グローバル回転 G_b C と、それに対応する局所回転 [T, 24, 3, 3]。"""
    G_t = G_b @ C
    local = torch.empty_like(G_t)
    local[:, 0] = G_t[:, 0]
    for j in range(1, NUM_JOINTS):
        local[:, j] = G_t[:, PARENTS[j]].transpose(-1, -2) @ G_t[:, j]
    return G_t, local


def init_transl(smpl, local, J_rest, P_t):
    """左右の股関節の中点が BVH と一致するように並進の初期値を決める。"""
    with torch.no_grad():
        _, joints = smpl.forward_kinematics(local, J_rest.expand(local.shape[0], -1, -1))
        return (P_t[:, [1, 2]] - joints[:, [1, 2]]).mean(1)


# ----------------------------------------------------------------------
# 最適化
# ----------------------------------------------------------------------
class Fitter:
    def __init__(self, smpl, J_rest, cfg, device):
        self.smpl, self.J_rest, self.device = smpl, J_rest, device
        w = cfg.get('weights', {})
        self.w = {k: float(w.get(k, d)) for k, d in
                  dict(pos=1.0, ori=5.0, offset=0.05, hinge=10.0, prior=1.0, smooth=0.0).items()}
        self.lr = float(cfg.get('lr', 0.01))
        self.pos_w = torch.tensor([CORR[s][1] for s in range(NUM_JOINTS)], device=device)
        self.ori_w = torch.tensor([CORR[s][2] for s in range(NUM_JOINTS)], device=device)
        self.offset_mask = self.pos_w > 0

    def model_joints(self, R, transl, offsets):
        glob_R, joints = self.smpl.forward_kinematics(R, self.J_rest.expand(R.shape[0], -1, -1))
        joints = joints + transl[:, None]
        mocap = joints + (glob_R[:, PARENT_FRAME] @ offsets[..., None])[..., 0]
        return glob_R, joints, mocap

    def losses(self, R, transl, offsets, G_t, P_t, R_init, smooth):
        glob_R, joints, mocap = self.model_joints(R, transl, offsets)
        e_pos = ((mocap - P_t) * 100).pow(2).sum(-1)                       # [T, 24] cm^2
        L = {
            'pos': (e_pos * self.pos_w).sum(-1).mean() / self.pos_w.sum(),
            'ori': (((glob_R - G_t) ** 2).sum((-1, -2)) * self.ori_w).sum(-1).mean() / self.ori_w.sum(),
            'offset': ((offsets * 100).pow(2).sum(-1))[self.offset_mask].mean(),
            'hinge': (((R[:, HINGE, 1, 0] - R[:, HINGE, 0, 1]) / 2) ** 2).mean(),
            'prior': ((R[:, PRIOR] - R_init[:, PRIOR]) ** 2).sum((-1, -2)).mean(),
        }
        if smooth and self.w['smooth'] > 0 and R.shape[0] > 2:
            acc = joints[2:] - 2 * joints[1:-1] + joints[:-2]
            L['smooth'] = (acc * 100).pow(2).sum(-1).mean()
        total = sum(self.w[k] * v for k, v in L.items())
        return total, L

    def run(self, local_init, transl_init, G_t, P_t, offsets_init, iters, fit_offsets, smooth, tag):
        rot6d = matrix_to_rot6d(local_init).clone().requires_grad_(True)
        transl = transl_init.clone().requires_grad_(True)
        offsets = offsets_init.clone().requires_grad_(fit_offsets)
        groups = [{'params': [rot6d], 'lr': self.lr}, {'params': [transl], 'lr': self.lr * 0.2}]
        if fit_offsets:
            groups.append({'params': [offsets], 'lr': self.lr * 0.2})
        opt = torch.optim.Adam(groups)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=self.lr * 0.01)
        t0 = time.time()
        for it in range(iters):
            opt.zero_grad()
            total, L = self.losses(rot6d_to_matrix(rot6d), transl, offsets * self.offset_mask[:, None],
                                   G_t, P_t, local_init, smooth)
            total.backward()
            opt.step()
            sched.step()
            if it % 100 == 0 or it == iters - 1:
                print(f'  [{tag}] iter {it:4d}  total {total.item():8.4f}  '
                      + '  '.join(f'{k} {v.item():.4f}' for k, v in L.items())
                      + f'  ({time.time() - t0:.0f}s)')
        with torch.no_grad():
            return rot6d_to_matrix(rot6d), transl.detach(), (offsets * self.offset_mask[:, None]).detach()


# ----------------------------------------------------------------------
# 評価・保存
# ----------------------------------------------------------------------
@torch.no_grad()
def foot_heights(smpl, R, transl, betas, stride=10, chunk=500):
    """足 (足首・つま先に属する頂点) の最下点の高さ [m] (stride フレームごと)。"""
    foot_v = torch.isin(smpl.weights.argmax(1), torch.tensor([7, 8, 10, 11], device=R.device))
    idx = torch.arange(0, R.shape[0], stride, device=R.device)
    out = []
    for s in range(0, len(idx), chunk):
        i = idx[s:s + chunk]
        v = smpl(R[i], betas, transl[i], return_verts=True).verts
        out.append(v[:, foot_v, 1].min(1).values)
    return torch.cat(out)


def bone_report(J_rest, bvh, report):
    Pb = bvh.rest_positions() / 1000.0
    Js = J_rest.cpu().numpy()
    log('  骨長の比較 (レストポーズ, mm)   SMPL(β)  mocap   差', report)
    for name, sa, sb, ba, bb in BONES:
        if sa is None:   # 股関節中央からの距離
            ls = np.linalg.norm(Js[sb] - (Js[1] + Js[2]) / 2)
            lb = np.linalg.norm(Pb[bvh.index[bb]] - (Pb[bvh.index['LeftUpLeg']] + Pb[bvh.index['RightUpLeg']]) / 2)
        else:
            ls = np.linalg.norm(Js[sb] - Js[sa])
            lb = np.linalg.norm(Pb[bvh.index[bb]] - Pb[bvh.index[ba]])
        log(f'    {name:12s} {ls * 1000:8.1f} {lb * 1000:7.1f} {(ls - lb) * 1000:+6.1f}', report)


def jitter(x):
    """関節ごとの加速度の大きさの平均 [T, J, 3] -> [J]"""
    return np.linalg.norm(x[2:] - 2 * x[1:-1] + x[:-2], axis=-1).mean(0)


def summarize(tag, fit_err, ori_err, foot_h, jitter_ratio, report):
    names = [f'{JOINT_NAMES[s]}' for s in range(NUM_JOINTS)]
    m = np.nanmean(fit_err, 0)
    o = np.mean(ori_err, 0)
    log(f'  [{tag}] 関節位置の誤差: 平均 {np.nanmean(fit_err):.1f} mm, '
        f'95 パーセンタイル {np.nanpercentile(fit_err, 95):.1f} mm', report)
    log('    ' + ', '.join(f'{names[s]} {m[s]:.1f}' for s in range(NUM_JOINTS) if CORR[s][1] > 0), report)
    log(f'  [{tag}] グローバル回転の差 [deg]: '
        + ', '.join(f'{names[s]} {o[s]:.1f}' for s in range(NUM_JOINTS) if CORR[s][2] > 0), report)
    fh = foot_h * 1000
    log(f'  [{tag}] 足裏の最下点の高さ [mm] (床 y=0 と仮定): p1 {np.percentile(fh, 1):.1f}, '
        f'p5 {np.percentile(fh, 5):.1f}, p50 {np.percentile(fh, 50):.1f}, '
        f'-10 mm 未満 (めり込み) {np.mean(fh < -10):.1%}', report)
    log(f'  [{tag}] ジッター比 (フィット / mocap, 1 に近いほど良い): 平均 {np.mean(jitter_ratio):.2f}, '
        f'最大 {np.max(jitter_ratio):.2f}', report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--ids', nargs='*', default=None, help='対象の収録 ID (例 D1S1A1)')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    config = load_config(args.config)
    fcfg = config.get('fit_smpl') or {}
    scfg = config.get('smpl') or {}
    device = torch.device(fcfg.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    data_dir = get_paths(config).data_dir
    out_root = resolve(scfg.get('data_dir', 'data/smpl'))
    subjects_cfg = {k.upper(): v for k, v in (scfg.get('subjects') or {}).items()}

    # 収録の一覧 {subject: [(data_id, split, bvh_path)]}
    recs = {}
    for p in sorted(data_dir.glob('skeleton/*/*_skeleton.bvh')):
        data_id, split = p.name.split('_')[0].upper(), p.parent.name
        recs.setdefault(subject_key(data_id), []).append((data_id, split, p))
    if not recs:
        sys.exit(f'{data_dir / "skeleton"} に BVH がありません。先に scripts/prepare_data.py を実行してください。')
    wanted = {i.upper() for i in args.ids} if args.ids else None

    report = []
    for subj, items in recs.items():
        todo = [(d, s, p) for d, s, p in items
                if (wanted is None or d in wanted)
                and (args.overwrite or not (out_root / s / f'{d}_smpl.npz').exists())]
        if not todo:
            print(f'{subj}: 作成済みのためスキップ (--overwrite で作り直し)')
            continue
        if subj not in subjects_cfg:
            sys.exit(f'config.yaml の smpl.subjects に {subj} (gender, betas) がありません')
        gender = subjects_cfg[subj].get('gender', 'neutral')
        betas_list = list(subjects_cfg[subj].get('betas') or [])
        if len(betas_list) > NUM_BETAS:
            print(f'警告: {subj} の β が {len(betas_list)} 次元あります。先頭 {NUM_BETAS} 次元だけ使います')
        betas = SMPL.prepare_betas(betas_list).to(device)

        log(f'\n===== {subj} (gender={gender}, betas={np.round(betas.cpu().numpy(), 3).tolist()}) =====', report)
        if not torch.any(betas != 0):
            log('  警告: β がすべて 0 (平均体型) です。config.yaml の smpl.subjects に収集した β を設定してください', report)

        smpl = SMPL.from_config(config, gender).to(device)
        J_rest = smpl.rest_joints(betas)[0]
        fitter = Fitter(smpl, J_rest, fcfg, device)

        # ---------- 準備: 全収録の目標値と初期値 ----------
        data = {}
        for data_id, split, path in items:
            print(f'  BVH 読み込み: {path.name}')
            bvh = load_bvh(path)
            G_b, P_t = bvh_targets(bvh, device)
            C = rest_correction(J_rest, bvh, device)
            G_t, local = retarget(G_b, C)
            data[data_id] = dict(split=split, bvh=bvh, G_t=G_t, P_t=P_t, local=local,
                                 transl=init_transl(smpl, local, J_rest, P_t))
        bone_report(J_rest, data[items[0][0]]['bvh'], report)

        # ---------- 段階 1: オフセット (関節の定義の差) の推定 ----------
        stride = int(fcfg.get('offset_stride', 10))
        cat = lambda k: torch.cat([d[k][::stride] for d in data.values()])  # noqa: E731
        offsets0 = torch.zeros(NUM_JOINTS, 3, device=device)
        print(f'  段階 1: オフセット推定 ({cat("P_t").shape[0]} フレーム)')
        _, _, offsets = fitter.run(cat('local'), cat('transl'), cat('G_t'), cat('P_t'), offsets0,
                                   int(fcfg.get('offset_iters', 800)), fit_offsets=True, smooth=False,
                                   tag='stage1')
        off_mm = offsets.norm(dim=-1).cpu().numpy() * 1000
        log('  オフセット [mm]: ' + ', '.join(f'{JOINT_NAMES[s]} {off_mm[s]:.0f}'
                                             for s in range(NUM_JOINTS) if CORR[s][1] > 0), report)

        # ---------- 段階 2: 収録ごとに全フレーム ----------
        for data_id, split, _ in todo:
            d = data[data_id]
            T = d['P_t'].shape[0]
            print(f'  段階 2: {data_id} ({T} フレーム)')
            R, transl, _ = fitter.run(d['local'], d['transl'], d['G_t'], d['P_t'], offsets,
                                      int(fcfg.get('pose_iters', 600)), fit_offsets=False, smooth=True,
                                      tag=data_id)
            with torch.no_grad():
                glob_R, joints, mocap = fitter.model_joints(R, transl, offsets)
                fit_err = ((mocap - d['P_t']).norm(dim=-1) * 1000).cpu().numpy()
                fit_err[:, ~fitter.offset_mask.cpu().numpy()] = np.nan
                ori_err = np.degrees(geodesic_angle(glob_R, d['G_t']).cpu().numpy())
                R[:, 22:24] = torch.eye(3, device=device)       # 手は観測できないので単位回転に固定
                _, joints, _ = fitter.model_joints(R, transl, offsets)
                poses = matrix_to_axis_angle(R).reshape(T, -1)
                fh = foot_heights(smpl, R, transl, betas).cpu().numpy()
                m = fitter.offset_mask.cpu().numpy()
                jr = jitter(mocap.cpu().numpy())[m] / np.maximum(jitter(d['P_t'].cpu().numpy())[m], 1e-9)

            log(f'\n  --- {data_id} ({split}, {T} フレーム) ---', report)
            summarize(data_id, fit_err, ori_err, fh, jr, report)

            out = out_root / split / f'{data_id}_smpl.npz'
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out,
                poses=poses.cpu().numpy().astype(np.float32),
                trans=transl.cpu().numpy().astype(np.float32),
                betas=betas.cpu().numpy().astype(np.float32),
                gender=np.array(gender),
                mocap_framerate=np.array(1.0 / d['bvh'].frame_time),
                joints=joints.cpu().numpy().astype(np.float32),
                offsets=offsets.cpu().numpy().astype(np.float32),
                offset_mask=fitter.offset_mask.cpu().numpy(),
                mocap_joint_names=np.array([CORR[s][0] for s in range(NUM_JOINTS)]),
                fit_err_mm=fit_err.astype(np.float32),
                ori_err_deg=ori_err.astype(np.float32),
            )
            log(f'  保存: {out.relative_to(ROOT)}', report)

    if report:
        rp = out_root / 'fit_report.txt'
        rp.parent.mkdir(parents=True, exist_ok=True)
        with open(rp, 'a', encoding='utf-8') as f:
            f.write(f'\n##### {time.strftime("%Y-%m-%d %H:%M:%S")} #####\n' + '\n'.join(report) + '\n')
        print(f'\nレポートを {rp.relative_to(ROOT)} に追記しました')


if __name__ == '__main__':
    main()
