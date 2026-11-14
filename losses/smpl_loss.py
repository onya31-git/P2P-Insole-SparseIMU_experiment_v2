"""
SMPL 形式の正解に対する損失 (dataset.target = 'smpl' のとき使う)。

モデルの出力 (並進, 局所回転 6D, 接地) を SMPL の順運動学に通し、
関節位置・グローバル回転でも正解と比べる。体型 β は被験者の値 (正解と同じ) を使う。
SMPL の順運動学では骨長が常に正しいので、骨長の損失は不要。

    rot        局所回転 (回転行列の要素の L1)                      手を除く 22 関節
    glob_rot   グローバル回転 (順運動学後の回転行列の L1)           手を除く 22 関節
    joint      関節位置の平均ユークリッド距離 [m]                    24 関節
    trans      並進の L1 [m]
    contact    接地の BCE
    smooth     関節速度の差の L1 [m/s]
    foot_skate 正解で接地しているフレームでの、足 (足首・つま先) の速度の誤差 [m/s]。
               足の滑りを抑えるため、接地中の足の速度を全体の smooth とは別に重み付けして合わせる。
               ※ 速度そのものを 0 にする形にしないのは、正解の足も接地ラベル中に平均 0.1 m/s 程度
                 動いているため (ラベルの閾値 0.4 m/s 未満・mocap のノイズ)。0 に寄せると正解からずれる
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.rotation import rot6d_to_matrix, geodesic_angle
from utils.smpl import BODY_JOINTS

SKEL_FPS = 100.0
FOOT_JOINTS = ([7, 10], [8, 11])      # 左 (足首, つま先), 右 (足首, つま先)

DEFAULT_LAMBDAS = {'rot': 1.0, 'glob_rot': 1.0, 'joint': 5.0, 'trans': 1.0,
                   'contact': 0.1, 'smooth': 0.1, 'foot_skate': 0.1}


class SMPLPoseLoss(nn.Module):
    def __init__(self, smpl_bank, lambdas=None):
        super().__init__()
        self.bank = smpl_bank
        self.lambdas = {**DEFAULT_LAMBDAS, **(lambdas or {})}
        self.bce = nn.BCEWithLogitsLoss()

    def smpl_outputs(self, trans, pose_6d, betas, genders):
        """
        (並進, 局所回転 6D) -> 局所回転行列, グローバル回転, 関節位置 [m]
        trans [B, T, 3], pose_6d [B, T, 24, 6], betas [B, 10], genders: 長さ B のリスト
        """
        R = rot6d_to_matrix(pose_6d)                                     # [B, T, 24, 3, 3]
        J_rest = self.bank.rest_joints(betas, list(genders))             # [B, 24, 3]
        glob_R, joints = self.bank.forward_kinematics(R, J_rest[:, None])
        return R, glob_R, joints + trans[:, :, None]

    def forward(self, pred_trans, pred_pose_6d, pred_contact, batch):
        """
        batch: データセットのバッチ (gt_smpl_pose, gt_smpl_trans, gt_smpl_joints, gt_contact,
               body_params, gender) をデバイスに載せたもの
        戻り値: (合計, 各項の値の dict, 予測の関節位置 [B, T, 24, 3])
        """
        betas, genders = batch['body_params'], batch['gender']
        R_p, G_p, J_p = self.smpl_outputs(pred_trans, pred_pose_6d, betas, genders)
        R_g = rot6d_to_matrix(batch['gt_smpl_pose'])
        G_g, _ = self.bank.forward_kinematics(R_g, self.bank.rest_joints(betas, list(genders))[:, None])
        J_g = batch['gt_smpl_joints']
        gt_contact = batch['gt_contact']

        body = BODY_JOINTS
        L = {
            'rot': F.l1_loss(R_p[:, :, body], R_g[:, :, body]),
            'glob_rot': F.l1_loss(G_p[:, :, body], G_g[:, :, body]),
            'joint': (J_p - J_g).norm(dim=-1).mean(),
            'trans': F.l1_loss(pred_trans, batch['gt_smpl_trans']),
            'contact': self.bce(pred_contact, gt_contact),
        }
        if J_p.shape[1] > 1:
            vel_p = (J_p[:, 1:] - J_p[:, :-1]) * SKEL_FPS                # [B, T-1, 24, 3] m/s
            vel_g = (J_g[:, 1:] - J_g[:, :-1]) * SKEL_FPS
            L['smooth'] = F.l1_loss(vel_p, vel_g)
            # 正解で接地している区間 (前後 2 フレームとも接地) の足の速さ
            c = gt_contact[:, 1:] * gt_contact[:, :-1]                    # [B, T-1, 2]
            skate = 0.0
            for side, joints in enumerate(FOOT_JOINTS):
                err = (vel_p[:, :, joints] - vel_g[:, :, joints]).norm(dim=-1).mean(-1)   # [B, T-1]
                skate = skate + (err * c[..., side]).sum() / c[..., side].sum().clamp(min=1.0)
            L['foot_skate'] = skate / 2
        else:
            L['smooth'] = L['foot_skate'] = pred_trans.new_zeros(())

        total = sum(self.lambdas[k] * v for k, v in L.items())
        with torch.no_grad():
            info = {f'loss_{k}': v.item() for k, v in L.items()}
            info['mpjpe_mm'] = L['joint'].item() * 1000
            info['rot_deg'] = torch.rad2deg(geodesic_angle(R_p[:, :, body], R_g[:, :, body])).mean().item()
            info['total_pose'] = total.item()
        return total, info, J_p
