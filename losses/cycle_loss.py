import torch
import torch.nn as nn
import torch.nn.functional as F

class CycleLoss(nn.Module):
    """
    Computes Cycle Consistency Losses by reconstructing 
    Pressure and IMU signals from the estimated Pose.
    """
    def __init__(self, pose_dim=512, pressure_nodes=35, pressure_channels=1, 
                 imu_sensor_count=5, imu_channels=9, lambdas=None):
        super(CycleLoss, self).__init__()
        
        self.lambdas = lambdas or {
            'pressure': 1.0,
            'imu': 1.0
        }
        
        # Simple MLPs for reconstruction (Decoders)
        # Note: In practice, these decoders are trained jointly or separately
        
        # Reconstruct Pressure [B, T, Np, C] from Body Latent or Pose [B, T, D]
        self.pressure_decoder_l = nn.Sequential(
            nn.Linear(pose_dim, 256),
            nn.ReLU(),
            nn.Linear(256, pressure_nodes * pressure_channels)
        )
        self.pressure_decoder_r = nn.Sequential(
            nn.Linear(pose_dim, 256),
            nn.ReLU(),
            nn.Linear(256, pressure_nodes * pressure_channels)
        )
        self.pressure_nodes = pressure_nodes
        self.pressure_channels = pressure_channels
        
        # Reconstruct IMU [B, T, 5, F]
        self.imu_decoder = nn.Sequential(
            nn.Linear(pose_dim, 256),
            nn.ReLU(),
            nn.Linear(256, imu_sensor_count * imu_channels)
        )
        self.imu_sensor_count = imu_sensor_count
        self.imu_channels = imu_channels
        
    def forward(self, body_latent, gt_left_pressure, gt_right_pressure, gt_imu):
        """
        Inputs:
        body_latent: [B, T, pose_dim] (or could be explicit pose features)
        gt_left_pressure: [B, T, Np, C]
        gt_right_pressure: [B, T, Np, C]
        gt_imu: [B, T, 5, F]
        """
        # Note: Depending on the specific architecture described in the paper, cycle decoders might
        # take the estimated pose (Positions + Rotations) as input instead of the latent representation.
        # This implementation takes the intermediate body_latent for simplicity but can be adopted.
        
        B, T, _ = body_latent.shape
        
        # Reconstruct Left Pressure
        pred_lp = self.pressure_decoder_l(body_latent)
        pred_lp = pred_lp.view(B, T, self.pressure_nodes, self.pressure_channels)
        loss_pressure_l = F.mse_loss(pred_lp, gt_left_pressure)
        
        # Reconstruct Right Pressure
        pred_rp = self.pressure_decoder_r(body_latent)
        pred_rp = pred_rp.view(B, T, self.pressure_nodes, self.pressure_channels)
        loss_pressure_r = F.mse_loss(pred_rp, gt_right_pressure)
        
        loss_pressure = loss_pressure_l + loss_pressure_r
        
        # Reconstruct IMU
        pred_imu = self.imu_decoder(body_latent)
        pred_imu = pred_imu.view(B, T, self.imu_sensor_count, self.imu_channels)
        loss_imu = F.mse_loss(pred_imu, gt_imu)
        
        total_cycle = (
            self.lambdas['pressure'] * loss_pressure + 
            self.lambdas['imu'] * loss_imu
        )
        
        return total_cycle, {
            'loss_cycle_pressure': loss_pressure.item(),
            'loss_cycle_imu': loss_imu.item(),
            'total_cycle': total_cycle.item()
        }
