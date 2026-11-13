import torch
import torch.nn as nn

class FootFusion(nn.Module):
    """
    Fuses Foot Pressure features and Ankle IMU features using Cross-Attention.
    """
    def __init__(self, embed_dim=128, num_heads=4):
        super(FootFusion, self).__init__()
        
        # Cross Attention
        # query, key, value embeddings
        # We can use pressure as query, IMU as key/value, or vice versa.
        # Here we use pressure as query since we want to ground the IMU data with floor contact info.
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        
        # Feature fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, left_pressure_feat, right_pressure_feat, left_ankle_feat, right_ankle_feat):
        """
        Inputs: [B, T, D] where D is embed_dim
        Outputs: 
        left_foot_feat: [B, T, D]
        right_foot_feat: [B, T, D]
        """
        # Cross attention for left foot
        # query: pressure, key/value: ankle IMU
        # MultiheadAttention expects [B, T, D] if batch_first=True
        left_attn_out, _ = self.cross_attn(query=left_pressure_feat, 
                                           key=left_ankle_feat, 
                                           value=left_ankle_feat)
        
        # Combine residual or concat
        left_concat = torch.cat([left_pressure_feat, left_attn_out], dim=-1) # [B, T, D*2]
        left_foot_feat = self.fusion_mlp(left_concat) # [B, T, D]
        
        # Cross attention for right foot
        right_attn_out, _ = self.cross_attn(query=right_pressure_feat, 
                                            key=right_ankle_feat, 
                                            value=right_ankle_feat)
        
        right_concat = torch.cat([right_pressure_feat, right_attn_out], dim=-1)
        right_foot_feat = self.fusion_mlp(right_concat)
        
        return left_foot_feat, right_foot_feat

if __name__ == '__main__':
    B, T, D = 2, 10, 128
    model = FootFusion(embed_dim=D)
    lp = torch.randn(B, T, D)
    rp = torch.randn(B, T, D)
    la = torch.randn(B, T, D)
    ra = torch.randn(B, T, D)
    
    lf, rf = model(lp, rp, la, ra)
    print("FootFusion outputs (Left Foot shape):", lf.shape)
