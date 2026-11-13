import torch
import torch.nn as nn

class TemporalIMUEncoder(nn.Module):
    """
    Encodes temporal sequences for a single IMU.
    """
    def __init__(self, in_features, hidden_dim=128, num_layers=2):
        super(TemporalIMUEncoder, self).__init__()
        self.gru = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(self, x):
        # x: [B, T, F]
        out, _ = self.gru(x)
        out = self.fc(out) # [B, T, hidden_dim]
        return out

class IMUEncoder(nn.Module):
    """
    Main IMU Encoder Module processing 5 IMUs:
    LeftWrist, RightWrist, LeftAnkle, RightAnkle, Pelvis.
    """
    def __init__(self, in_features=9, hidden_dim=128):
        # in_features: 3 (Mag) + 3 (Gyro) + 3 (Acc) = 9
        super(IMUEncoder, self).__init__()
        
        # Shared encoder for limbs
        self.limb_encoder = TemporalIMUEncoder(in_features, hidden_dim)
        
        # Specific encoder for pelvis
        self.pelvis_encoder = TemporalIMUEncoder(in_features, hidden_dim)
        
    def forward(self, imu_data):
        """
        imu_data: [B, T, 5, F]
        Assume order: 0:LeftWrist, 1:RightWrist, 2:LeftAnkle, 3:RightAnkle, 4:Pelvis
        """
        lw = imu_data[:, :, 0, :]
        rw = imu_data[:, :, 1, :]
        la = imu_data[:, :, 2, :]
        ra = imu_data[:, :, 3, :]
        pelvis = imu_data[:, :, 4, :]
        
        # Encode with shared limb encoder
        lw_feat = self.limb_encoder(lw)
        rw_feat = self.limb_encoder(rw)
        la_feat = self.limb_encoder(la)
        ra_feat = self.limb_encoder(ra)
        
        # Encode pelvis
        pelvis_feat = self.pelvis_encoder(pelvis)
        
        return lw_feat, rw_feat, la_feat, ra_feat, pelvis_feat

if __name__ == '__main__':
    B, T, F = 2, 10, 10
    model = IMUEncoder(in_features=F)
    imu_data = torch.randn(B, T, 5, F)
    lw, rw, la, ra, p = model(imu_data)
    print("IMUEncoder outputs (LeftWrist, Pelvis shape):", lw.shape, p.shape)
