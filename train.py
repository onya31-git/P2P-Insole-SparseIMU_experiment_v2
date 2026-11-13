"""
モデルを学習する。

使い方:
    uv run train.py [--config config.yaml]

出力:
    outputs/checkpoints/model.pth   学習済み重み (エポックごとに上書き保存)
    outputs/checkpoints/stats.json  正規化統計量 (test.py で使用)
    outputs/sync/train.json         学習データの時刻同期結果
"""
import json
import argparse

import torch

from dataset import get_dataloader
from dataset.dataset import KINEMATIC_TREE
from losses.pose_loss import PoseLoss
from losses.cycle_loss import CycleLoss
from models import KinematicFusionSystem
from utils.paths import load_config, get_paths, DEFAULT_CONFIG


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        return super().default(obj)



def train(config_path=DEFAULT_CONFIG):
    config = load_config(config_path)
    paths = get_paths(config, makedirs=True)

    ds_cfg = config.get('dataset', {})
    tr_cfg = config.get('training', {})
    cy_cfg = config.get('cycle_loss', {})

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = KinematicFusionSystem(config=config).to(device)
    pose_loss_fn = PoseLoss(lambdas=config.get('losses')).to(device)
    cycle_loss_fn = CycleLoss(
        pressure_nodes=cy_cfg.get('pressure_nodes', 112),
        imu_channels=cy_cfg.get('imu_channels', 6),
    ).to(device)

    # CycleLoss のデコーダーも学習対象に含める
    params = list(model.parameters()) + list(cycle_loss_fn.parameters())
    optimizer = torch.optim.Adam(params, lr=tr_cfg.get('learning_rate', 1e-4))

    dataloader = get_dataloader(
        str(paths.data_dir),
        batch_size=tr_cfg.get('batch_size', 32),
        seq_len=ds_cfg.get('seq_len', 200),
        mode='train',
        num_workers=ds_cfg.get('num_workers', 0),
        sync_cfg=config.get('sync'),
    )

    with open(paths.sync_train, 'w', encoding='utf-8') as f:
        json.dump(dataloader.dataset.sync_info, f, indent=2, ensure_ascii=False)
    print(f"[train] 同期結果を {paths.sync_train} に保存しました")
    with open(paths.stats, 'w') as f:
        json.dump(dataloader.dataset.id_stats, f, cls=NumpyEncoder)
    print(f"[train] 統計量を {paths.stats} に保存しました (keys: {list(dataloader.dataset.id_stats.keys())})")

    cycle_w = tr_cfg.get('cycle_loss_weight', 0.1)
    num_epochs = tr_cfg.get('num_epochs', 50)

    for epoch in range(num_epochs):
        model.train()
        cycle_loss_fn.train()
        for batch_idx, batch in enumerate(dataloader):
            lp = batch['left_pressure'].to(device)
            rp = batch['right_pressure'].to(device)
            imu = batch['imu_data'].to(device)
            bp = batch['body_params'].to(device)
            gt_pos = batch['gt_joint_pos'].to(device)
            gt_rot = batch['gt_joint_rot'].to(device)
            gt_contact = batch['gt_contact'].to(device)
            pos_mean = batch['pos_mean'].to(device)
            pos_std = batch['pos_std'].to(device)

            optimizer.zero_grad()
            pred_pos, pred_rot, pred_contact, body_latent = model(lp, rp, imu, bp)

            loss_pose, _ = pose_loss_fn(pred_pos, gt_pos, pred_rot, gt_rot, pred_contact,
                                        gt_contact, kinematic_tree=KINEMATIC_TREE,
                                        pos_mean=pos_mean, pos_std=pos_std)
            loss_cycle, _ = cycle_loss_fn(body_latent, lp, rp, imu)
            loss = loss_pose + cycle_w * loss_cycle

            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}], Loss: {loss.item():.4f}")

        torch.save(model.state_dict(), paths.weights)
        print(f"Epoch [{epoch+1}/{num_epochs}] 終了: 重みを '{paths.weights}' に保存しました。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    args = parser.parse_args()
    train(args.config)
