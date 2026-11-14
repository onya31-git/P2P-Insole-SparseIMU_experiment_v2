"""
推論結果 (outputs/results/smpl_predictions.npz) から SMPL の 3D メッシュを書き出す。

    uv run scripts/export_mesh.py --start 0 --end 300 --step 1   # OBJ (予測・正解) + AMASS 形式の npz
    uv run scripts/export_mesh.py --no-obj                        # npz のみ (全フレーム)

出力 (outputs/results/mesh/):
    pred/frame_000000.obj ...   予測のメッシュ (m, y が鉛直)
    gt/frame_000000.obj ...     正解 (SMPL フィッティング結果) のメッシュ
    pred_amass.npz, gt_amass.npz
        AMASS と同じキー (poses [N, 72], trans [N, 3], betas, gender, mocap_framerate)。
        Blender の SMPL アドオンや aitviewer などで読み込める。
        ※ 並進の水平成分 (x, z) は学習データと同じくルート相対 (0) なので、その場で動く。
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.paths import load_config, get_paths, DEFAULT_CONFIG   # noqa: E402
from utils.smpl import SMPLBank                                  # noqa: E402


def write_obj(path, verts, faces):
    with open(path, 'w') as f:
        f.write(''.join(f'v {x:.5f} {y:.5f} {z:.5f}\n' for x, y, z in verts))
        f.write(''.join(f'f {a + 1} {b + 1} {c + 1}\n' for a, b, c in faces))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=300, help='-1 で最後まで')
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--no-obj', action='store_true', help='OBJ を書き出さない (npz のみ)')
    args = parser.parse_args()

    config = load_config(args.config)
    paths = get_paths(config, makedirs=True)
    if not paths.smpl_pred.exists():
        sys.exit(f'{paths.smpl_pred} がありません。dataset.target: smpl で test.py を実行してください。')
    z = dict(np.load(paths.smpl_pred))
    out_dir = paths.mesh_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    genders = np.unique(z['gender'])
    for key in ('pred', 'gt'):
        np.savez(out_dir / f'{key}_amass.npz',
                 poses=z[f'{key}_poses'], trans=z[f'{key}_trans'], betas=z['betas'][0],
                 gender=np.array(str(z['gender'][0])), mocap_framerate=z['mocap_framerate'])
    print(f'保存: {out_dir / "pred_amass.npz"}, {out_dir / "gt_amass.npz"} ({len(z["pred_poses"])} フレーム)')
    if len(genders) > 1 or len(np.unique(z['subject'])) > 1:
        print('警告: 複数の被験者が含まれています。AMASS 形式の npz の betas / gender は先頭の被験者の値です')
    if args.no_obj:
        return

    N = len(z['pred_poses'])
    end = N if args.end == -1 else min(args.end, N)
    frames = np.arange(args.start, end, max(1, args.step))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bank = SMPLBank(config, device)
    for key in ('pred', 'gt'):
        (out_dir / key).mkdir(exist_ok=True)
    for s in range(0, len(frames), 128):
        i = frames[s:s + 128]
        for g in np.unique(z['gender'][i]):
            ig = i[z['gender'][i] == g]

            def t(k):
                return torch.as_tensor(z[k][ig], device=device)
            faces = bank.faces(str(g)).cpu().numpy()
            for key in ('pred', 'gt'):
                verts = bank(t(f'{key}_poses'), t('betas'), t(f'{key}_trans'), str(g), return_verts=True).verts
                for n, v in zip(ig, verts.cpu().numpy()):
                    write_obj(out_dir / key / f'frame_{n:06d}.obj', v, faces)
    print(f'保存: {out_dir / "pred"}, {out_dir / "gt"} に OBJ を {len(frames)} フレームずつ')


if __name__ == '__main__':
    main()
