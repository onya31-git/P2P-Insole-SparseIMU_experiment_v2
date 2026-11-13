import torch
import torch.nn as nn
import torch.nn.functional as F

class PoseLoss(nn.Module):
    """
    Computes multiple loss components for pose estimation:
    - Position (MPJPE / L1)
    - Rotation (L1 or Geodesic on 6D)
    - Bone Length (実寸 [mm] での骨長の相対誤差)
    - Foot Contact
    - Temporal Smoothness
    """
    def __init__(self, lambdas=None):
        super(PoseLoss, self).__init__()
        # Default loss weights
        self.lambdas = lambdas or {
            'pos': 1.0,
            'rot': 1.0,
            'bone': 0.5,
            'contact': 0.1,
            'smooth': 0.01   # 0.1 → 0.01: 急激な動作変化への過剰抑制を防ぐ
        }
        self.bce = nn.BCEWithLogitsLoss()
        
    def forward(self, pred_pos, gt_pos, pred_rot, gt_rot, pred_contact, gt_contact, kinematic_tree=None,
                pos_mean=None, pos_std=None):
        """
        Inputs:
        pred_pos, gt_pos: [B, T, J, 3]  正規化された位置
        pred_rot, gt_rot: [B, T, J, 6]
        pred_contact: [B, T, 2] # logits
        gt_contact: [B, T, 2]   # labels (0 or 1)
        kinematic_tree: list of tuples (parent_idx, child_idx) defining bones
        pos_mean, pos_std: [B, J, 3]  位置の正規化統計量。骨長は pos * pos_std + pos_mean で
                           実寸 [mm] に戻してから計算する (関節・軸ごとにスケールが異なる正規化空間では
                           骨長が意味を持たないため)。kinematic_tree を渡す場合は必須。
        """
        # Position Loss (L1 MPJPE)
        loss_pos = F.l1_loss(pred_pos, gt_pos)
        
        # Rotation Loss (L1 on 6D representation)
        # Note: Proper geodesic loss could be implemented if converted to rotation matrices
        loss_rot = F.l1_loss(pred_rot, gt_rot)
        
        # Contact Loss (BCE)
        loss_contact = self.bce(pred_contact, gt_contact)
        
        # Smoothness Loss (Temporal difference)
        # diff: [B, T-1, J, 3]
        if pred_pos.shape[1] > 1:
            diff_pred = pred_pos[:, 1:] - pred_pos[:, :-1]
            diff_gt = gt_pos[:, 1:] - gt_pos[:, :-1]
            loss_smooth = F.l1_loss(diff_pred, diff_gt)
        else:
            loss_smooth = torch.tensor(0.0, device=pred_pos.device)
            
        # Bone Length Loss: 骨長の相対誤差 |L_pred - L_gt| / L_gt (実寸で計算するので単位に依存しない)
        loss_bone = torch.tensor(0.0, device=pred_pos.device)
        if kinematic_tree is not None:
            if pos_mean is None or pos_std is None:
                raise ValueError("骨長の損失には pos_mean / pos_std (実寸に戻すための統計量) が必要です")
            mean, std = pos_mean.unsqueeze(1), pos_std.unsqueeze(1)   # [B, 1, J, 3]
            pred_mm, gt_mm = pred_pos * std + mean, gt_pos * std + mean
            parents = [u for u, _ in kinematic_tree]
            children = [v for _, v in kinematic_tree]
            bone_pred = torch.norm(pred_mm[:, :, children] - pred_mm[:, :, parents], dim=-1)  # [B, T, num_bones]
            bone_gt = torch.norm(gt_mm[:, :, children] - gt_mm[:, :, parents], dim=-1)
            loss_bone = (torch.abs(bone_pred - bone_gt) / bone_gt.clamp(min=1.0)).mean()

        total_loss = (
            self.lambdas['pos'] * loss_pos + 
            self.lambdas['rot'] * loss_rot + 
            self.lambdas['bone'] * loss_bone + 
            self.lambdas['contact'] * loss_contact + 
            self.lambdas['smooth'] * loss_smooth
        )
        
        return total_loss, {
            'loss_pos': loss_pos.item(),
            'loss_rot': loss_rot.item(),
            'loss_bone': loss_bone.item(),
            'loss_contact': loss_contact.item(),
            'loss_smooth': loss_smooth.item(),
            'total_pose': total_loss.item()
        }
