"""
test.py が書き出した予測 CSV から評価指標を計算する。

使い方:
    uv run evaluate.py [--config config.yaml]

入力:
    outputs/results/predictions_pos.csv, predictions_rot.csv, predictions_contact.csv
出力:
    outputs/results/evaluation.txt
        MPJPE      : 関節位置の平均誤差 [mm]
        Rot L2     : クオータニオン (W,X,Y,Z) の L2 距離 min(|p-g|, |p+g|) の平均
        Rot Angle  : 予測を正規化したうえでの回転の角度差 2*arccos(|<p,g>|) [deg] の平均
        Bone       : 骨長 (KINEMATIC_TREE の 17 本) の誤差 |L_pred - L_gt| の平均 [mm] と相対誤差 [%]
        Contact    : 接地の正解率と F1 (予測確率 0.5 で 0/1 に判定)
    回転誤差は q と -q を同じ回転として扱う (符号に依存しない)。

dataset.target = smpl のときは outputs/results/smpl_predictions.npz から次を計算する:
    MPJPE        : SMPL の関節位置の誤差 [mm] (22 関節 / 24 関節, 骨盤をそろえた値も)
    PVE          : メッシュの頂点誤差 [mm]
    Rot error    : 局所回転 / グローバル回転の角度差 [deg], SIP 誤差 (股関節・肩のグローバル回転)
    Jerk         : 関節の加加速度 [m/s^3] (ジッター)。予測はチャンク (2 秒) の境目で不連続になりうる
    Foot speed   : 正解で接地しているフレームの足の速さ [cm/s] (足の滑り)
    mocap MPJPE  : 元の mocap の 16 関節との誤差 [mm]。正解 SMPL 自体の誤差 (フィッティング誤差) も併記
"""
import os
import argparse

import pandas as pd
import numpy as np
import torch

from dataset.dataset import KINEMATIC_TREE, MOCAP_TO_SMPL, JOINT_NAMES as MOCAP_JOINT_NAMES
from utils.rotation import axis_angle_to_matrix, geodesic_angle
from utils.smpl import SMPLBank, PARENTS, BODY_JOINTS, JOINT_NAMES as SMPL_JOINT_NAMES
from utils.paths import load_config, get_paths, DEFAULT_CONFIG

NUM_JOINTS = 18
JOINT_NAMES = [
    "CenterOfGravity", "SpinePosition", "Spine4Position", "HeadPosition",
    "LShoulder", "LElbow", "LWrist",
    "RShoulder", "RElbow", "RWrist",
    "LHip", "LKnee", "LAnkle", "LToe",
    "RHip", "RKnee", "RAnkle", "RToe"
]
BODY_PARTS = {
    "Spine/Torso": [0, 1, 2],
    "Head": [3],
    "Arms": [4, 5, 6, 7, 8, 9],
    "Legs": [10, 11, 12, 13, 14, 15, 16, 17]
}


def quat_errors(pred_quat, gt_quat):
    """
    符号に依存しないクオータニオンの誤差を返す。
    pred_quat, gt_quat : [N, J, 4]
    戻り値: (L2 距離 [N, J], 角度差 [deg] [N, J])
    """
    l2 = np.minimum(np.linalg.norm(pred_quat - gt_quat, axis=-1),
                    np.linalg.norm(pred_quat + gt_quat, axis=-1))
    p = pred_quat / np.clip(np.linalg.norm(pred_quat, axis=-1, keepdims=True), 1e-8, None)
    g = gt_quat / np.clip(np.linalg.norm(gt_quat, axis=-1, keepdims=True), 1e-8, None)
    dot = np.clip(np.abs(np.sum(p * g, axis=-1)), 0.0, 1.0)
    angle = np.degrees(2.0 * np.arccos(dot))
    return l2, angle


def contact_scores(pred_prob, gt):
    """接地の正解率と F1 を返す (pred_prob >= 0.5 を接地とみなす)。"""
    pred = pred_prob >= 0.5
    gt = gt >= 0.5
    tp = np.sum(pred & gt)
    precision = tp / max(np.sum(pred), 1)
    recall = tp / max(np.sum(gt), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return np.mean(pred == gt), f1


def evaluate(config_path=DEFAULT_CONFIG, pve_stride=None):
    config = load_config(config_path)
    paths = get_paths(config, makedirs=True)
    if (config.get('dataset') or {}).get('target', 'skeleton') == 'smpl':
        return evaluate_smpl(config, paths, pve_stride)

    if not os.path.exists(paths.pred_pos):
        print(f"Error: {paths.pred_pos} not found. Run test.py first.")
        return

    # ---------------- 位置 ----------------
    print(f"Loading {paths.pred_pos}...")
    df_pos = pd.read_csv(paths.pred_pos)
    pred_pos = df_pos[[c for c in df_pos.columns if c.startswith('Pred_')]].values.reshape(-1, NUM_JOINTS, 3)
    gt_pos = df_pos[[c for c in df_pos.columns if c.startswith('GT_')]].values.reshape(-1, NUM_JOINTS, 3)
    print(f"Loaded {pred_pos.shape[0]} frames for positions.")
    per_joint_mpjpe = np.mean(np.linalg.norm(pred_pos - gt_pos, axis=-1), axis=0)

    # 骨長の誤差 (実寸)
    parents = [u for u, _ in KINEMATIC_TREE]
    children = [v for _, v in KINEMATIC_TREE]
    bone_pred = np.linalg.norm(pred_pos[:, children] - pred_pos[:, parents], axis=-1)
    bone_gt = np.linalg.norm(gt_pos[:, children] - gt_pos[:, parents], axis=-1)
    bone_err_mm = np.mean(np.abs(bone_pred - bone_gt))
    bone_err_pct = 100 * np.mean(np.abs(bone_pred - bone_gt) / np.clip(bone_gt, 1.0, None))

    # ---------------- 回転 ----------------
    per_joint_l2 = per_joint_angle = None
    if os.path.exists(paths.pred_rot):
        print(f"Loading {paths.pred_rot}...")
        df_rot = pd.read_csv(paths.pred_rot)
        pred_rot = df_rot[[c for c in df_rot.columns if c.startswith('RotPred_')]].values.reshape(-1, NUM_JOINTS, 6)
        gt_rot = df_rot[[c for c in df_rot.columns if c.startswith('RotGT_')]].values.reshape(-1, NUM_JOINTS, 6)
        # 先頭4次元がクオータニオン (W, X, Y, Z)
        l2, angle = quat_errors(pred_rot[:, :, :4], gt_rot[:, :, :4])
        per_joint_l2, per_joint_angle = l2.mean(axis=0), angle.mean(axis=0)
    else:
        print(f"Warning: {paths.pred_rot} not found. Rotational Error will not be calculated.")

    # ---------------- 接地 ----------------
    contact = None
    if os.path.exists(paths.pred_contact):
        print(f"Loading {paths.pred_contact}...")
        df_c = pd.read_csv(paths.pred_contact)
        contact = {side: contact_scores(df_c[f'Pred_{side}'].values, df_c[f'GT_{side}'].values)
                   for side in ('L', 'R')}
        contact['GT rate'] = (df_c['GT_L'].mean(), df_c['GT_R'].mean())
    else:
        print(f"Warning: {paths.pred_contact} not found. Contact will not be evaluated.")

    # ---------------- 出力 ----------------
    def fmt(v):
        return "N/A" if v is None else f"{v:.4f}"

    def rows(title, groups):
        out = [f"{title:<20} | {'MPJPE [mm]':<11} | {'Rot L2':<8} | {'Rot [deg]':<9}", "-" * 60]
        for name, idx in groups:
            l2_v = None if per_joint_l2 is None else np.mean(per_joint_l2[idx])
            ang_v = None if per_joint_angle is None else np.mean(per_joint_angle[idx])
            out.append(f"{name:<20} | {np.mean(per_joint_mpjpe[idx]):<11.4f} | {fmt(l2_v):<8} | "
                       + ("N/A" if ang_v is None else f"{ang_v:.2f}"))
        return out

    lines = ["=" * 60, "Evaluation Results", "=" * 60,
             f"Overall MPJPE [mm]        : {np.mean(per_joint_mpjpe):.4f}",
             f"Bone length error [mm / %]: {bone_err_mm:.2f} / {bone_err_pct:.2f}"]
    if per_joint_l2 is not None:
        lines += [f"Overall Rot L2 (quat)     : {np.mean(per_joint_l2):.4f}",
                  f"Overall Rot Angle [deg]   : {np.mean(per_joint_angle):.2f}"]
    if contact is not None:
        for side, label in (('L', 'Left '), ('R', 'Right')):
            acc, f1 = contact[side]
            lines.append(f"Contact {label} acc / F1   : {acc:.4f} / {f1:.4f}")
        lines.append(f"Contact GT rate (L / R)   : {contact['GT rate'][0]:.2f} / {contact['GT rate'][1]:.2f}")
    lines.append("-" * 60)
    lines += rows('Body Part', BODY_PARTS.items())
    lines.append("-" * 60)
    lines += rows('Joint Name', [(n, [i]) for i, n in enumerate(JOINT_NAMES)])
    lines.append("=" * 60)

    output_text = "\n".join(lines)
    print("\n" + output_text)
    with open(paths.evaluation, 'w', encoding='utf-8') as f:
        f.write(output_text)
    print(f"\nResults saved to {paths.evaluation}")




# ======================================================================
# dataset.target = smpl
# ======================================================================
SMPL_PARTS = {
    "Torso": [0, 3, 6, 9],
    "Head/Neck": [12, 15],
    "Arms": [13, 14, 16, 17, 18, 19, 20, 21],
    "Legs": [1, 2, 4, 5, 7, 8, 10, 11],
}
SIP_JOINTS = [1, 2, 16, 17]          # 股関節・肩 (DIP / TransPose などで使われる SIP 誤差)
FOOT = ([7, 10], [8, 11])            # 左右の (足首, つま先)
FPS = 100.0


def _global_rot(R):
    """局所回転 [N, 24, 3, 3] -> グローバル回転 (回転だけなので体型に依存しない)。"""
    G = [R[:, 0]]
    for j in range(1, len(PARENTS)):
        G.append(G[PARENTS[j]] @ R[:, j])
    return torch.stack(G, 1)


def _pve(z, bank, stride, chunk=256):
    """頂点誤差 [mm] (stride フレームごと)。"""
    idx = np.arange(0, len(z['pred_poses']), stride)
    errs = []
    for g in np.unique(z['gender'][idx]):
        ig = idx[z['gender'][idx] == g]
        for s in range(0, len(ig), chunk):
            i = ig[s:s + chunk]

            def t(k):
                return torch.as_tensor(z[k][i], device=bank.device)
            vp = bank(t('pred_poses'), t('betas'), t('pred_trans'), str(g), return_verts=True).verts
            vg = bank(t('gt_poses'), t('betas'), t('gt_trans'), str(g), return_verts=True).verts
            errs.append(((vp - vg).norm(dim=-1).mean(-1) * 1000).cpu().numpy())
    return np.concatenate(errs)


def _foot_speed(J, contact):
    """正解で接地しているフレームの足 (足首・つま先の遅い方) の速さ [cm/s] の平均。"""
    vel = np.linalg.norm(np.diff(J, axis=0), axis=-1) * FPS * 100       # [N-1, 24] cm/s
    c = (contact[1:] >= 0.5) & (contact[:-1] >= 0.5)
    vals = [vel[c[:, s]][:, list(j)].min(-1) for s, j in enumerate(FOOT)]
    n = sum(len(v) for v in vals)
    return float(np.mean(np.concatenate(vals))) if n else float('nan')


def _jerk(J):
    """関節の加加速度の大きさ [m/s^3] の平均 (ジッターの指標)。"""
    return float(np.linalg.norm(np.diff(J, n=3, axis=0), axis=-1).mean() * FPS ** 3)


def _mocap_mpjpe(pred_m, mocap_mm):
    """
    mocap の関節 (MOCAP_TO_SMPL の 16 関節) との誤差 [mm] [N, 16]。
    水平方向の基準が異なる (SMPL は並進, mocap は CenterOfGravity) ため、
    両方とも左右の股関節の中点を水平方向の原点にそろえてから比べる。
    """
    m_idx = list(MOCAP_TO_SMPL.keys())
    s_idx = list(MOCAP_TO_SMPL.values())
    p = pred_m[:, s_idx] * 1000
    g = mocap_mm[:, m_idx]
    hips = [m_idx.index(10), m_idx.index(14)]
    hip_p = p[:, hips].mean(1, keepdims=True)
    hip_g = g[:, hips].mean(1, keepdims=True)
    hip_p[..., 1] = 0
    hip_g[..., 1] = 0
    return np.linalg.norm((p - hip_p) - (g - hip_g), axis=-1)


def evaluate_smpl(config, paths, pve_stride=None):
    if not os.path.exists(paths.smpl_pred):
        print(f"Error: {paths.smpl_pred} not found. Run test.py first.")
        return
    print(f"Loading {paths.smpl_pred}...")
    z = dict(np.load(paths.smpl_pred))
    N = len(z['pred_poses'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bank = SMPLBank(config, device)

    pj, gj = z['pred_joints'], z['gt_joints']
    err = np.linalg.norm(pj - gj, axis=-1) * 1000                                  # [N, 24]
    pa = np.linalg.norm((pj - pj[:, :1]) - (gj - gj[:, :1]), axis=-1) * 1000       # 骨盤をそろえた誤差

    Rp = axis_angle_to_matrix(torch.as_tensor(z['pred_poses']).reshape(N, 24, 3))
    Rg = axis_angle_to_matrix(torch.as_tensor(z['gt_poses']).reshape(N, 24, 3))
    loc_deg = torch.rad2deg(geodesic_angle(Rp, Rg)).numpy()                        # [N, 24]
    glob_deg = torch.rad2deg(geodesic_angle(_global_rot(Rp), _global_rot(Rg))).numpy()

    if pve_stride is None:
        pve_stride = 1 if device.type == 'cuda' else 10
    print(f"Computing PVE (every {pve_stride} frame(s))...")
    pve = _pve(z, bank, pve_stride)

    mocap_pred = _mocap_mpjpe(z['pred_mocap'], z['gt_mocap_pos'])
    mocap_fit = _mocap_mpjpe(z['gt_fit_mocap'], z['gt_mocap_pos'])
    contact = {side: contact_scores(z['pred_contact'][:, i], z['gt_contact'][:, i])
               for i, side in enumerate(('L', 'R'))}
    root_h = np.abs(z['pred_trans'][:, 1] - z['gt_trans'][:, 1]).mean() * 1000

    body = BODY_JOINTS
    lines = ["=" * 64, "Evaluation Results (SMPL)", "=" * 64,
             f"Frames                         : {N}",
             f"MPJPE [mm] (22 joints)         : {err[:, body].mean():.2f}",
             f"MPJPE [mm] (24 joints)         : {err.mean():.2f}",
             f"Pelvis-aligned MPJPE [mm]      : {pa[:, body].mean():.2f}",
             f"PVE [mm]                       : {pve.mean():.2f}",
             f"Local rot error [deg]          : {loc_deg[:, body].mean():.2f}",
             f"Global rot error [deg]         : {glob_deg[:, body].mean():.2f}",
             f"SIP error [deg] (hips, shldrs) : {glob_deg[:, SIP_JOINTS].mean():.2f}",
             f"Root height error [mm]         : {root_h:.2f}",
             f"Jerk [m/s^3] pred / GT         : {_jerk(pj):.1f} / {_jerk(gj):.1f}",
             f"Foot speed in contact [cm/s]   : {_foot_speed(pj, z['gt_contact']):.2f} / "
             f"{_foot_speed(gj, z['gt_contact']):.2f}   (pred / GT, 小さいほど滑りが少ない)",
             "-" * 64,
             "mocap の関節 (16 関節, 股関節の中点を水平方向の原点にそろえて比較)",
             f"  予測 vs mocap MPJPE [mm]     : {mocap_pred.mean():.2f}",
             f"  正解 SMPL vs mocap [mm]      : {mocap_fit.mean():.2f}   (フィッティング誤差 = 下限の目安)",
             "-" * 64]
    for side, label in (('L', 'Left '), ('R', 'Right')):
        acc, f1 = contact[side]
        lines.append(f"Contact {label} acc / F1        : {acc:.4f} / {f1:.4f}")
    lines += ["-" * 64,
              f"{'Body Part':<16} | {'MPJPE [mm]':>10} | {'Local [deg]':>11} | {'Global [deg]':>12}",
              "-" * 64]
    for name, idx in SMPL_PARTS.items():
        lines.append(f"{name:<16} | {err[:, idx].mean():>10.2f} | {loc_deg[:, idx].mean():>11.2f} | "
                     f"{glob_deg[:, idx].mean():>12.2f}")
    lines += ["-" * 64,
              f"{'Joint Name':<16} | {'MPJPE [mm]':>10} | {'Local [deg]':>11} | {'Global [deg]':>12}",
              "-" * 64]
    for j in body:
        lines.append(f"{SMPL_JOINT_NAMES[j]:<16} | {err[:, j].mean():>10.2f} | {loc_deg[:, j].mean():>11.2f} | "
                     f"{glob_deg[:, j].mean():>12.2f}")
    lines += ["-" * 64, f"{'mocap Joint':<16} | {'pred [mm]':>10} | {'fit [mm]':>10}", "-" * 64]
    for k, m in enumerate(MOCAP_TO_SMPL):
        name = MOCAP_JOINT_NAMES[m].replace('Positions', '').replace('Position', '')
        lines.append(f"{name:<16} | {mocap_pred[:, k].mean():>10.2f} | {mocap_fit[:, k].mean():>10.2f}")
    lines.append("=" * 64)

    output_text = "\n".join(lines)
    print("\n" + output_text)
    with open(paths.evaluation, 'w', encoding='utf-8') as f:
        f.write(output_text)
    print(f"\nResults saved to {paths.evaluation}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--pve-stride', type=int, default=None,
                        help='PVE を計算するフレーム間隔 (既定: GPU なら 1, CPU なら 10)')
    args = parser.parse_args()
    evaluate(args.config, args.pve_stride)
