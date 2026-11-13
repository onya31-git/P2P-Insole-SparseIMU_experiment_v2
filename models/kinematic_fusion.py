# kinematic_fusion.py

import torch
import torch.nn as nn

class KinematicFusion(nn.Module):
    """
    Fuses upper body and lower body latent representations using a 
    Kinematic Graph Transformer or a standard Transformer Encoder.
    """
    def __init__(self, upper_dim=256, lower_dim=256, hidden_dim=512, num_heads=8, num_layers=2):
        super(KinematicFusion, self).__init__()
        
        # Project inputs to same hidden dim
        self.proj_upper = nn.Linear(upper_dim, hidden_dim // 2)
        self.proj_lower = nn.Linear(lower_dim, hidden_dim // 2)
        
        # Transformer Encoder for fusion
        # Concatenate upper and lower features across embedding dimension
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, upper_latent, lower_latent):
        """
        Inputs:
        upper_latent: [B, T, upper_dim]
        lower_latent: [B, T, lower_dim]
        
        Output:
        body_latent: [B, T, hidden_dim]
        """
        # Project
        u_feat = self.proj_upper(upper_latent) # [B, T, hidden_dim // 2]
        l_feat = self.proj_lower(lower_latent) # [B, T, hidden_dim // 2]
        
        # Concat features per timestep
        combined = torch.cat([u_feat, l_feat], dim=-1) # [B, T, hidden_dim]
        
        # Pass through Transformer to allow temporal/kinematic interactions
        # (Alternatively, could treat upper/lower as separate tokens [B, T, 2, hidden_dim])
        body_latent = self.transformer(combined) # [B, T, hidden_dim]
        
        return body_latent

if __name__ == '__main__':
    B, T, U_dim, L_dim, H_dim = 2, 10, 256, 256, 512
    model = KinematicFusion(upper_dim=U_dim, lower_dim=L_dim, hidden_dim=H_dim)
    u = torch.randn(B, T, U_dim)
    l = torch.randn(B, T, L_dim)
    
    body = model(u, l)
    print("BodyLatent shape:", body.shape)
