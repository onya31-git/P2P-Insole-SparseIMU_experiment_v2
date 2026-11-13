"""
学習済みモデルでテストデータを推論し、予測結果を CSV に書き出す。
入力の正規化・出力の逆正規化には学習時に保存した統計量 (stats.json) を使う。

使い方:
    uv run test.py [--config config.yaml]
    uv run evaluate.py      # 続けて評価指標を計算

入力:
    outputs/checkpoints/model.pth, outputs/checkpoints/stats.json
出力:
    outputs/results/predictions_pos.csv  関節位置 (予測 Pred_* / 正解 GT_*)
    outputs/results/predictions_rot.csv  関節回転 (予測 RotPred_* / 正解 RotGT_*)
    outputs/sync/test.json               テストデータの時刻同期結果
"""
import json
import argparse

import torch

from dataset import get_dataloader
from models import KinematicFusionSystem
from utils.export_csv import export_predictions_to_csv, export_rotations_to_csv
from utils.paths import load_config, get_paths, DEFAULT_CONFIG


def test(config_path=DEFAULT_CONFIG):
    config = load_config(config_path)
    paths = get_paths(config, makedirs=True)

    ds_cfg = config.get('dataset', {})
    te_cfg = config.get('test', {})

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = KinematicFusionSystem(config=config).to(device)
    model.load_state_dict(torch.load(paths.weights, map_location=device))
    model.eval()

    with open(paths.stats, 'r') as f:
        id_stats = json.load(f)

    dataloader = get_dataloader(
        str(paths.data_dir),
        batch_size=te_cfg.get('batch_size', 1),
        seq_len=ds_cfg.get('seq_len', 200),
        mode='test',
        num_workers=ds_cfg.get('num_workers', 0),
        stats=id_stats,
        sync_cfg=config.get('sync'),
    )
    with open(paths.sync_test, 'w', encoding='utf-8') as f:
        json.dump(dataloader.dataset.sync_info, f, indent=2, ensure_ascii=False)
    print(f"[test] 同期結果を {paths.sync_test} に保存しました")

    # データセットが実際に使った統計量で逆正規化する
    stats_tensors = {
        key: {
            'pos_mean': torch.tensor(st['pos_mean'], device=device, dtype=torch.float32),
            'pos_std':  torch.tensor(st['pos_std'],  device=device, dtype=torch.float32),
        }
        for key, st in dataloader.dataset.id_stats.items()
    }

    all_pred_pos, all_gt_pos, all_pred_rot, all_gt_rot = [], [], [], []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            lp = batch['left_pressure'].to(device)
            rp = batch['right_pressure'].to(device)
            imu = batch['imu_data'].to(device)
            bp = batch['body_params'].to(device)
            gt_pos = batch['gt_joint_pos'].to(device)
            gt_rot = batch['gt_joint_rot'].to(device)

            pred_pos, pred_rot, _, _ = model(lp, rp, imu, bp)

            pred_pos_denorm = torch.zeros_like(pred_pos)
            gt_pos_denorm = torch.zeros_like(gt_pos)
            for b_i, key in enumerate(batch['data_id']):
                st = stats_tensors[key]
                pred_pos_denorm[b_i] = pred_pos[b_i] * st['pos_std'] + st['pos_mean']
                gt_pos_denorm[b_i] = gt_pos[b_i] * st['pos_std'] + st['pos_mean']

            all_pred_pos.append(pred_pos_denorm.cpu().numpy().tolist())
            all_gt_pos.append(gt_pos_denorm.cpu().numpy().tolist())
            all_pred_rot.append(pred_rot.cpu().numpy().tolist())
            all_gt_rot.append(gt_rot.cpu().numpy().tolist())

            if batch_idx % 10 == 0:
                print(f"Test Step [{batch_idx}] processed")

    print("============\nTest Complete. Processing finished.\n============")
    export_predictions_to_csv(all_pred_pos, all_gt_pos, paths.pred_pos)
    export_rotations_to_csv(all_pred_rot, all_gt_rot, paths.pred_rot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    args = parser.parse_args()
    test(args.config)
