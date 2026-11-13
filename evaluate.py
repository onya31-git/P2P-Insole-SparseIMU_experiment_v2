"""
test.py が書き出した予測 CSV から評価指標を計算する。

使い方:
    uv run evaluate.py [--config config.yaml]

入力:
    outputs/results/predictions_pos.csv, outputs/results/predictions_rot.csv
出力:
    outputs/results/evaluation.txt
        MPJPE : 関節位置の平均誤差 (スケルトンCSVの単位)
        MPJAE : クオータニオン (W,X,Y,Z) の L2 距離の平均
"""
import os
import argparse

import pandas as pd
import numpy as np

from utils.paths import load_config, get_paths, DEFAULT_CONFIG


def evaluate(config_path=DEFAULT_CONFIG):
    paths = get_paths(load_config(config_path), makedirs=True)
    pos_file = paths.pred_pos
    rot_file = paths.pred_rot
    output_file = paths.evaluation

    if not os.path.exists(pos_file):
        print(f"Error: {pos_file} not found. Run test.py first.")
        return

    print(f"Loading {pos_file}...")
    df_pos = pd.read_csv(pos_file)

    pred_pos_cols = [c for c in df_pos.columns if c.startswith('Pred_')]
    gt_pos_cols = [c for c in df_pos.columns if c.startswith('GT_')]

    num_joints = 18
    pred_pos = df_pos[pred_pos_cols].values.reshape(-1, num_joints, 3)
    gt_pos = df_pos[gt_pos_cols].values.reshape(-1, num_joints, 3)
    num_frames = pred_pos.shape[0]
    print(f"Loaded {num_frames} frames for positions.")

    # Calculate MPJPE (位置の誤差)
    diff_pos = pred_pos - gt_pos
    distances_pos = np.linalg.norm(diff_pos, axis=-1)
    overall_mpjpe = np.mean(distances_pos)
    per_joint_mpjpe = np.mean(distances_pos, axis=0)

    overall_mpjae = None
    per_joint_mpjae = None

    if os.path.exists(rot_file):
        print(f"Loading {rot_file}...")
        df_rot = pd.read_csv(rot_file)

        pred_rot_cols = [c for c in df_rot.columns if c.startswith('RotPred_')]
        gt_rot_cols = [c for c in df_rot.columns if c.startswith('RotGT_')]

        # Reshape to [Frames, Joints, 6]
        pred_rot = df_rot[pred_rot_cols].values.reshape(-1, num_joints, 6)
        gt_rot = df_rot[gt_rot_cols].values.reshape(-1, num_joints, 6)

        # クオータニオン（最初の4次元）を取り出す
        pred_quat = pred_rot[:, :, :4]
        gt_quat = gt_rot[:, :, :4]

        # Calculate Error for Quaternions (ユークリッド距離)
        # ※厳密にはクオータニオンの測地距離 2*arccos(|q1・q2|) を使用することもありますが、
        # 既存コードのL2ノルム(ユークリッド距離)ベースの評価に合わせます。
        diff_quat = pred_quat - gt_quat
        distances_rot = np.linalg.norm(diff_quat, axis=-1)

        overall_mpjae = np.mean(distances_rot)
        per_joint_mpjae = np.mean(distances_rot, axis=0)
        print(f"Loaded {pred_rot.shape[0]} frames for rotations (Quaternions).")
    else:
        print(f"Warning: {rot_file} not found. Rotational Error will not be calculated.")

    joint_names = [
        "CenterOfGravity", "SpinePosition", "Spine4Position", "HeadPosition",
        "LShoulder", "LElbow", "LWrist",
        "RShoulder", "RElbow", "RWrist",
        "LHip", "LKnee", "LAnkle", "LToe",
        "RHip", "RKnee", "RAnkle", "RToe"
    ]

    body_parts = {
        "Spine/Torso": [0, 1, 2],
        "Head": [3],
        "Arms": [4, 5, 6, 7, 8, 9],
        "Legs": [10, 11, 12, 13, 14, 15, 16, 17]
    }

    lines = []
    lines.append("="*50)
    lines.append("Evaluation Results (Quaternion Model)")
    lines.append("="*50)
    lines.append(f"Overall Average MPJPE: {overall_mpjpe:.4f}")
    if overall_mpjae is not None:
        lines.append(f"Overall Average MPJAE (Quat L2): {overall_mpjae:.4f}")
    lines.append("-" * 50)

    lines.append(f"{'Body Part':<20} | {'MPJPE (Pos)':<12} | {'Error (Rot)':<12}")
    lines.append("-" * 50)

    for part_name, indices in body_parts.items():
        part_mpjpe = np.mean([per_joint_mpjpe[i] for i in indices])
        mpjpe_str = f"{part_mpjpe:.4f}"
        if per_joint_mpjae is not None:
            part_mpjae = np.mean([per_joint_mpjae[i] for i in indices])
            mpjae_str = f"{part_mpjae:.4f}"
        else:
            mpjae_str = "N/A"
        lines.append(f"{part_name:<20} | {mpjpe_str:<12} | {mpjae_str:<12}")

    lines.append("-" * 50)
    lines.append(f"{'Joint Name':<20} | {'MPJPE (Pos)':<12} | {'Error (Rot)':<12}")
    lines.append("-" * 50)

    for i in range(num_joints):
        name = joint_names[i] if i < len(joint_names) else f"Joint_{i}"
        mpjpe_str = f"{per_joint_mpjpe[i]:.4f}"
        mpjae_str = f"{per_joint_mpjae[i]:.4f}" if per_joint_mpjae is not None else "N/A"
        lines.append(f"{name:<20} | {mpjpe_str:<12} | {mpjae_str:<12}")

    lines.append("="*50)

    output_text = "\n".join(lines)

    print("\n" + output_text)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output_text)
    print(f"\nResults saved to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    args = parser.parse_args()
    evaluate(args.config)
