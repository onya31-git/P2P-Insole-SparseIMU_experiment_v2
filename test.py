"""
学習済みモデルでテストデータを推論し、予測結果を書き出す。
入力の正規化・出力の逆正規化には学習時に保存した統計量 (stats.json) を使う。

使い方:
    uv run test.py [--config config.yaml]
    uv run evaluate.py      # 続けて評価指標を計算

入力:
    outputs/checkpoints/model.pth, outputs/checkpoints/stats.json
出力 (dataset.target = skeleton):
    outputs/results/predictions_pos.csv  関節位置 (予測 Pred_* / 正解 GT_*)
    outputs/results/predictions_rot.csv  関節回転 (予測 RotPred_* / 正解 RotGT_*)
    outputs/results/predictions_contact.csv  接地 (予測確率 Pred_L/R / 正解 GT_L/R)
    outputs/sync/test.json               テストデータの時刻同期結果
出力 (dataset.target = smpl):
    outputs/results/smpl_predictions.npz SMPL の予測と正解 (下記), 接地の CSV, 同期結果
        pred_poses / gt_poses   [N, 72]     局所回転 (軸角)  ※ AMASS と同じ形式
        pred_trans / gt_trans   [N, 3]      並進 [m] (水平 x, z はルート相対)
        pred_joints / gt_joints [N, 24, 3]  SMPL の関節位置 [m]
        pred_mocap / gt_fit_mocap [N, 24, 3] mocap の関節定義に合わせた位置 [m]
                                (SMPL 関節 + フィッティング時のオフセット。gt_fit_mocap は正解の SMPL から計算)
        gt_mocap_pos            [N, 18, 3]  mocap の関節位置 [mm] (元のスケルトン CSV)
        betas [N, 10], gender [N], subject [N], pred_contact / gt_contact [N, 2]
"""
import json
import argparse

import numpy as np
import torch

from dataset import get_dataloader
from models import KinematicFusionSystem
from utils.export_csv import export_predictions_to_csv, export_rotations_to_csv, export_contacts_to_csv
from utils.paths import load_config, get_paths, DEFAULT_CONFIG
from utils.rotation import rot6d_to_matrix, matrix_to_axis_angle
from utils.smpl import SMPLBank, PARENTS


def test(config_path=DEFAULT_CONFIG):
    config = load_config(config_path)
    paths = get_paths(config, makedirs=True)

    ds_cfg = config.get('dataset', {})
    te_cfg = config.get('test', {})
    target = ds_cfg.get('target', 'skeleton')

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
        target=target,
        smpl_cfg=config.get('smpl'),
    )
    with open(paths.sync_test, 'w', encoding='utf-8') as f:
        json.dump(dataloader.dataset.sync_info, f, indent=2, ensure_ascii=False)
    print(f"[test] 同期結果を {paths.sync_test} に保存しました")

    if target == 'smpl':
        return test_smpl(config, paths, model, dataloader, device)

    all_pred_pos, all_gt_pos, all_pred_rot, all_gt_rot = [], [], [], []
    all_pred_contact, all_gt_contact = [], []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            lp = batch['left_pressure'].to(device)
            rp = batch['right_pressure'].to(device)
            imu = batch['imu_data'].to(device)
            bp = batch['body_params'].to(device)
            gt_pos = batch['gt_joint_pos'].to(device)
            gt_rot = batch['gt_joint_rot'].to(device)

            pred_pos, pred_rot, pred_contact, _ = model(lp, rp, imu, bp)
            all_pred_contact.append(torch.sigmoid(pred_contact).cpu().numpy())
            all_gt_contact.append(batch['gt_contact'].numpy())

            # データセットが実際に使った統計量で mm に戻す
            pos_mean = batch['pos_mean'].to(device).unsqueeze(1)   # [B, 1, J, 3]
            pos_std = batch['pos_std'].to(device).unsqueeze(1)
            pred_pos_denorm = pred_pos * pos_std + pos_mean
            gt_pos_denorm = gt_pos * pos_std + pos_mean

            all_pred_pos.append(pred_pos_denorm.cpu().numpy().tolist())
            all_gt_pos.append(gt_pos_denorm.cpu().numpy().tolist())
            all_pred_rot.append(pred_rot.cpu().numpy().tolist())
            all_gt_rot.append(gt_rot.cpu().numpy().tolist())

            if batch_idx % 10 == 0:
                print(f"Test Step [{batch_idx}] processed")

    print("============\nTest Complete. Processing finished.\n============")
    export_predictions_to_csv(all_pred_pos, all_gt_pos, paths.pred_pos)
    export_rotations_to_csv(all_pred_rot, all_gt_rot, paths.pred_rot)
    export_contacts_to_csv(all_pred_contact, all_gt_contact, paths.pred_contact)


def test_smpl(config, paths, model, dataloader, device):
    """SMPL のパラメータを推定し、関節位置・mocap 相当の関節位置とあわせて npz に保存する。"""
    bank = SMPLBank(config, device)
    smpl_info = dataloader.dataset.smpl_info
    parent_frame = [0 if p < 0 else p for p in PARENTS]

    keys = ['pred_poses', 'gt_poses', 'pred_trans', 'gt_trans', 'pred_joints', 'gt_joints',
            'pred_mocap', 'gt_fit_mocap', 'gt_mocap_pos', 'betas', 'gender', 'subject',
            'pred_contact', 'gt_contact']
    out = {k: [] for k in keys}

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            lp = batch['left_pressure'].to(device)
            rp = batch['right_pressure'].to(device)
            imu = batch['imu_data'].to(device)
            bp = batch['body_params'].to(device)
            pred_trans, pred_6d, pred_contact, _ = model(lp, rp, imu, bp)

            B, T = pred_trans.shape[:2]
            genders, subjects = list(batch['gender']), list(batch['data_id'])
            J_rest = bank.rest_joints(bp, genders)[:, None]                       # [B, 1, 24, 3]
            offsets = torch.stack([torch.as_tensor(smpl_info[s]['offsets']) for s in subjects]).to(device)
            offsets = offsets[:, None]                                            # [B, 1, 24, 3]

            def run(trans, pose_6d):
                R = rot6d_to_matrix(pose_6d)
                G, J = bank.forward_kinematics(R, J_rest)
                J = J + trans[:, :, None]
                mocap = J + (G[:, :, parent_frame] @ offsets[..., None])[..., 0]
                return matrix_to_axis_angle(R).reshape(B, T, -1), J, mocap

            p_aa, p_J, p_mocap = run(pred_trans, pred_6d)
            gt_trans = batch['gt_smpl_trans'].to(device)
            g_aa, _, g_mocap = run(gt_trans, batch['gt_smpl_pose'].to(device))

            # mocap の関節位置を mm に戻す
            st_mean = batch['pos_mean'].unsqueeze(1)
            st_std = batch['pos_std'].unsqueeze(1)
            gt_mocap_mm = batch['gt_joint_pos'] * st_std + st_mean

            flat = lambda x: x.reshape((B * T,) + tuple(x.shape[2:])).cpu().numpy()  # noqa: E731
            out['pred_poses'].append(flat(p_aa))
            out['gt_poses'].append(flat(g_aa))
            out['pred_trans'].append(flat(pred_trans))
            out['gt_trans'].append(flat(gt_trans))
            out['pred_joints'].append(flat(p_J))
            out['gt_joints'].append(flat(batch['gt_smpl_joints']))
            out['pred_mocap'].append(flat(p_mocap))
            out['gt_fit_mocap'].append(flat(g_mocap))
            out['gt_mocap_pos'].append(flat(gt_mocap_mm))
            out['betas'].append(np.repeat(bp.cpu().numpy(), T, axis=0))
            out['gender'].append(np.repeat(np.array(genders), T))
            out['subject'].append(np.repeat(np.array(subjects), T))
            out['pred_contact'].append(flat(torch.sigmoid(pred_contact)))
            out['gt_contact'].append(flat(batch['gt_contact']))

            if batch_idx % 10 == 0:
                print(f"Test Step [{batch_idx}] processed")

    print("============\nTest Complete. Processing finished.\n============")
    res = {k: np.concatenate(v).astype(np.float32) if k not in ('gender', 'subject') else np.concatenate(v)
           for k, v in out.items()}
    res['mocap_framerate'] = np.array(100.0)
    np.savez_compressed(paths.smpl_pred, **res)
    print(f"Saved SMPL predictions to {paths.smpl_pred} ({len(res['pred_poses'])} frames)")
    export_contacts_to_csv([res['pred_contact']], [res['gt_contact']], paths.pred_contact)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    args = parser.parse_args()
    test(args.config)
