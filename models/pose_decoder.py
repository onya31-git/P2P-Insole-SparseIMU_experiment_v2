import torch
import torch.nn as nn

class PoseDecoder(nn.Module):
    """
    Decodes body latent representation into Joint Positions, 
    Joint Rotations (6D), and Foot Contact states.
    Incorporates concepts of Differentiable IK (Forward Kinematics).
    """
    def __init__(self, in_features=512, num_joints=18):
        super(PoseDecoder, self).__init__()
        self.num_joints = num_joints
        
        # Joint Rotations (6D representation -> 6 params per joint)
        self.rotation_decoder = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_joints * 6)
        )
        
        # Joint Positions (3D -> 3 params per joint)
        # Note: In a full Diff-IK setup, positions might be derived mathematically 
        # from rotations + bone lengths + root translation. Here we provide a direct prediction path as well.
        self.position_decoder = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_joints * 3)
        )
        
        # Foot Contact (Left and Right -> 2 params, logits)
        self.contact_decoder = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        
    def forward(self, body_latent):
        """
        Input: body_latent [B, T, in_features]
        Output:
        joint_positions [B, T, J, 3]
        joint_rotations [B, T, J, 6]
        foot_contact    [B, T, 2] # logits
        """
        B, T, F = body_latent.shape
        
        # [B, T, J*6]
        rotations_6d = self.rotation_decoder(body_latent)
        # [B, T, J, 6]
        rotations_6d = rotations_6d.view(B, T, self.num_joints, 6)
        
        # [B, T, J*3]
        positions = self.position_decoder(body_latent)
        # [B, T, J, 3]
        positions = positions.view(B, T, self.num_joints, 3)
        
        # [B, T, 2]
        contact_logits = self.contact_decoder(body_latent)
        
        # Optional: Forward Kinematics (FK) layer can be applied here to compute 
        # exact positions from rotations and predefined skeletal bone lengths.
        # joint_positions_fk = forward_kinematics(rotations_6d, bone_lengths)
        
        return positions, rotations_6d, contact_logits

if __name__ == '__main__':
    B, T, F = 2, 10, 512
    model = PoseDecoder(in_features=F, num_joints=17)
    body = torch.randn(B, T, F)
    pos, rot, contact = model(body)
    print("Positions shape:", pos.shape)
    print("Rotations shape:", rot.shape)
    print("Contact shape:", contact.shape)
