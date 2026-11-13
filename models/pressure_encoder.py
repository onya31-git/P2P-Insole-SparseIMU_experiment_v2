import torch
import torch.nn as nn

class GraphEncoder(nn.Module):
    """
    Encodes spatial features from foot pressure sensors using MLP or Graph Conv.
    Assuming Np sensors without explicit graph structure for now, we use a PointNet-like
    or transformer-like approach, or simple MLPs across the sensor dimension.
    """
    def __init__(self, num_sensors, in_channels=1, hidden_dim=64):
        super(GraphEncoder, self).__init__()
        self.num_sensors = num_sensors
        
        # Simple MLP to process each sensor's features (e.g. pressure, delta, etc.)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Aggregate across sensors (e.g., pooling or flattening)
        # Here we flatten, though max-pooling could also work depending on the exact topology
        self.fc_out = nn.Sequential(
            nn.Linear(num_sensors * hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2)
        )
        
    def forward(self, x):
        # x: [B, T, Np, in_channels] or [B, Np, in_channels]
        shape = x.shape
        if len(shape) == 4:
            B, T, Np, C = shape
            x = x.view(-1, Np, C) # [B*T, Np, C]
        else:
            B, Np, C = shape
            
        # Extract node features
        x = self.mlp(x) # [B*T, Np, hidden_dim]
        
        # Flatten and project
        x = x.view(x.shape[0], -1) # [B*T, Np*hidden_dim]
        x = self.fc_out(x)         # [B*T, hidden_dim * 2]
        
        if len(shape) == 4:
            x = x.view(B, T, -1)
            
        return x

class TemporalEncoder(nn.Module):
    """
    Encodes temporal dynamics of the pressure features.
    """
    def __init__(self, in_features, hidden_dim=128, num_layers=2):
        super(TemporalEncoder, self).__init__()
        self.gru = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        # Output dim is hidden_dim * 2 because of bidirectional
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x):
        # x: [B, T, in_features]
        out, _ = self.gru(x) # out: [B, T, hidden_dim * 2]
        out = self.fc(out)   # out: [B, T, hidden_dim]
        return out

class PressureEncoder(nn.Module):
    """
    Main Pressure Encoder Module processing Left and Right Foot pressure data.
    """
    def __init__(self, num_sensors=35, in_channels=1, spatial_dim=64, temporal_dim=128):
        super(PressureEncoder, self).__init__()
        
        # Shared GraphEncoder for left and right
        self.graph_encoder = GraphEncoder(num_sensors, in_channels, spatial_dim)
        
        # Shared TemporalEncoder
        spatial_out_dim = spatial_dim * 2
        self.temporal_encoder = TemporalEncoder(spatial_out_dim, temporal_dim)
        
    def forward(self, left_pressure, right_pressure):
        """
        left_pressure: [B, T, Np, C]
        right_pressure: [B, T, Np, C]
        """
        # Encode spatial features
        left_spatial = self.graph_encoder(left_pressure)   # [B, T, spatial_out_dim]
        right_spatial = self.graph_encoder(right_pressure) # [B, T, spatial_out_dim]
        
        # Encode temporal features
        left_feat = self.temporal_encoder(left_spatial)    # [B, T, temporal_dim]
        right_feat = self.temporal_encoder(right_spatial)  # [B, T, temporal_dim]
        
        return left_feat, right_feat

if __name__ == '__main__':
    # Test script
    B, T, Np, C = 2, 10, 89, 1
    model = PressureEncoder(num_sensors=Np, in_channels=C)
    left = torch.randn(B, T, Np, C)
    right = torch.randn(B, T, Np, C)
    
    out_left, out_right = model(left, right)
    print("PressureEncoder output shape:", out_left.shape)
