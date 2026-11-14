"""
BVH の読み込み (オイラー角の規約・階層) が正しいかを確認する。

BVH を順運動学で計算した関節位置と、同じ収録のスケルトン CSV (mocap 書き出し) の
関節位置を比較し、関節ごとの誤差 [mm] を表示する。誤差が十分小さければ
(目安: 数 mm 以下) BVH の解釈は正しく、SMPL フィッティングの入力に使える。

使い方:
    uv run scripts/check_bvh.py                 # data/skeleton/{train,test}/*.bvh をすべて確認
    uv run scripts/check_bvh.py --frames 2000   # 先頭 N フレームだけ (速い)
"""
import sys
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.bvh import load_bvh                       # noqa: E402
from utils.paths import load_config, get_paths      # noqa: E402

# BVH の関節名 -> スケルトン CSV の Position 列名
BVH_TO_CSV = {
    'LeftUpLeg': 'LHipPositions', 'LeftLeg': 'LKneePositions',
    'LeftFoot': 'LAnklePositions', 'LeftToeBase': 'LToePositions', 'LeftToeBase_End': 'LToesEndPosition',
    'RightUpLeg': 'RHipPositions', 'RightLeg': 'RKneePositions',
    'RightFoot': 'RAnklePositions', 'RightToeBase': 'RToePositions', 'RightToeBase_End': 'RToesEndPosition',
    'Spine': 'SpinePosition', 'Spine1': 'Spine1Position', 'Spine2': 'Spine2Position',
    'Spine3': 'Spine3Position', 'Spine4': 'Spine4Position', 'Neck': 'NeckPosition',
    'Head': 'HeadPosition', 'Head_End': 'HeadEndPosition',
    'LeftShoulder': 'LClaviclePosition', 'LeftArm': 'LShoulderPositions',
    'LeftForeArm': 'LElbowPositions', 'LeftHand': 'LWristPositions', 'LeftHand_End': 'LHandEndPosition',
    'RightShoulder': 'RClaviclePosition', 'RightArm': 'RShoulderPositions',
    'RightForeArm': 'RElbowPositions', 'RightHand': 'RWristPositions', 'RightHand_End': 'RHandEndPosition',
}


def read_csv_positions(csv_path, names, max_frames=None):
    """スケルトン CSV から指定した Position 列 [T, len(names), 3] (mm) を読む。"""
    with open(csv_path, 'r') as f:
        header = [next(f) for _ in range(5)]
    row2 = header[2].rstrip('\n').split(',')
    cols = []
    for n in names:
        idx = row2.index(n)               # 同名が複数ある場合は最初の列
        cols += [idx, idx + 1, idx + 2]
    data = np.loadtxt(csv_path, delimiter=',', skiprows=5, usecols=cols, max_rows=max_frames, ndmin=2)
    return data.reshape(len(data), len(names), 3)


def check(bvh_path, csv_path, max_frames=None):
    bvh = load_bvh(bvh_path)
    T = bvh.num_frames if max_frames is None else min(max_frames, bvh.num_frames)
    _, pos = bvh.forward_kinematics(slice(0, T))
    pos = pos.numpy()

    pairs = [(b, c) for b, c in BVH_TO_CSV.items() if b in bvh.index]
    gt = read_csv_positions(csv_path, [c for _, c in pairs], max_frames=T)
    n = min(len(gt), T)
    print(f'\n{bvh_path.name}  vs  {csv_path.name}  ({n} フレーム, BVH 全 {bvh.num_frames})')
    errs = []
    for k, (b, c) in enumerate(pairs):
        e = np.linalg.norm(pos[:n, bvh.index[b]] - gt[:n, k], axis=-1)
        errs.append(e)
        print(f'  {b:18s} -> {c:20s}  mean {e.mean():8.3f} mm   max {e.max():8.3f} mm')
    errs = np.stack(errs)
    print(f'  全関節: mean {errs.mean():.3f} mm, max {errs.max():.3f} mm')
    return errs.mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None)
    parser.add_argument('--frames', type=int, default=None, help='先頭 N フレームのみ確認')
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    skel_root = get_paths(cfg).data_dir / 'skeleton'

    results = {}
    for bvh_path in sorted(skel_root.glob('*/*_skeleton.bvh')):
        csv_path = bvh_path.with_suffix('.csv')
        if not csv_path.exists():
            print(f'skip {bvh_path.name}: 対応する CSV がありません')
            continue
        results[bvh_path.stem] = check(bvh_path, csv_path, args.frames)
    if not results:
        sys.exit(f'{skel_root} に *_skeleton.bvh がありません。先に scripts/prepare_data.py を実行してください。')
    print('\n=== まとめ (全関節平均誤差 [mm]) ===')
    for k, v in results.items():
        print(f'  {k}: {v:.3f}' + ('' if v < 5 else '   <-- 大きい: BVH の解釈を確認してください'))


if __name__ == '__main__':
    main()
