import torch
import torch.nn as nn

class UpperBodyNet(nn.Module):
    """
    Estimates upper body skeletal latent representation from wrist features,
    pelvis features, body parameters, and lower body latent representation.
    """
    def __init__(self, feature_dim=128, lower_latent_dim=256, body_param_dim=8, hidden_dim=256, num_layers=2):
        super(UpperBodyNet, self).__init__()
        
        # Embed body parameters
        self.body_param_embed = nn.Sequential(
            nn.Linear(body_param_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Input concatenation dimension
        # LeftWrist + RightWrist + Pelvis + LowerBodyLatent + BodyParam 
        # = 3 * feature_dim + lower_latent_dim + feature_dim
        input_dim = feature_dim * 4 + lower_latent_dim
        
        # Temporal Network
        self.temporal_net = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Output projection
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, left_wrist, right_wrist, pelvis, lower_body_latent, body_params):
        """
        Inputs:
        left_wrist, right_wrist, pelvis: [B, T, feature_dim]
        lower_body_latent: [B, T, lower_latent_dim]
        body_params: [B, body_param_dim]
        
        Output:
        upper_body_latent: [B, T, hidden_dim]
        """
        B, T, _ = left_wrist.shape
        
        # Embed and expand body params to sequence length
        bp_embed = self.body_param_embed(body_params) # [B, feature_dim]
        bp_embed = bp_embed.unsqueeze(1).expand(-1, T, -1) # [B, T, feature_dim]
        
        # Concatenate all features
        x = torch.cat([left_wrist, right_wrist, pelvis, lower_body_latent, bp_embed], dim=-1) # [B, T, input_dim]
        
        # Pass through temporal network
        out, _ = self.temporal_net(x) # [B, T, hidden_dim*2]
        
        # Project
        upper_body_latent = self.fc_out(out) # [B, T, hidden_dim]
        
        return upper_body_latent

if __name__ == '__main__':
    B, T, F_dim, L_dim, BP_dim = 2, 10, 128, 256, 8
    model = UpperBodyNet(feature_dim=F_dim, lower_latent_dim=L_dim, body_param_dim=BP_dim)
    lw = torch.randn(B, T, F_dim)
    rw = torch.randn(B, T, F_dim)
    p = torch.randn(B, T, F_dim)
    lb = torch.randn(B, T, L_dim)
    bp = torch.randn(B, BP_dim)
    
    latent = model(lw, rw, p, lb, bp)
    print("UpperBodyLatent shape:", latent.shape)
