import torch
import torch.nn as nn
import torch.nn.functional as F

class PoseLoss(nn.Module):
    """
    Computes multiple loss components for pose estimation:
    - Position (MPJPE / L1)
    - Rotation (L1 or Geodesic on 6D)
    - Bone Length
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
        
    def forward(self, pred_pos, gt_pos, pred_rot, gt_rot, pred_contact, gt_contact, kinematic_tree=None):
        """
        Inputs:
        pred_pos, gt_pos: [B, T, J, 3]
        pred_rot, gt_rot: [B, T, J, 6]
        pred_contact: [B, T, 2] # logits
        gt_contact: [B, T, 2]   # labels (0 or 1)
        kinematic_tree: list of tuples (parent_idx, child_idx) defining bones
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
            
        # Bone Length Loss
        loss_bone = torch.tensor(0.0, device=pred_pos.device)
        if kinematic_tree is not None:
            bone_preds = []
            bone_gts = []
            for u, v in kinematic_tree:
                # [B, T, 3]
                bone_vec_pred = pred_pos[:, :, v, :] - pred_pos[:, :, u, :]
                bone_len_pred = torch.norm(bone_vec_pred, p=2, dim=-1)
                
                bone_vec_gt = gt_pos[:, :, v, :] - gt_pos[:, :, u, :]
                bone_len_gt = torch.norm(bone_vec_gt, p=2, dim=-1)
                
                bone_preds.append(bone_len_pred)
                bone_gts.append(bone_len_gt)
            
            # [num_bones, B, T] -> [B, T, num_bones]
            bone_preds = torch.stack(bone_preds, dim=-1)
            bone_gts = torch.stack(bone_gts, dim=-1)
            loss_bone = F.l1_loss(bone_preds, bone_gts)
            
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
