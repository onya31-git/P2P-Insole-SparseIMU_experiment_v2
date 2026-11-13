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
"""
import os
import argparse

import pandas as pd
import numpy as np

from dataset.dataset import KINEMATIC_TREE
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


def evaluate(config_path=DEFAULT_CONFIG):
    paths = get_paths(load_config(config_path), makedirs=True)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    args = parser.parse_args()
    evaluate(args.config)
