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

class SMPLPoseDecoder(nn.Module):
    """
    body_latent から SMPL のパラメータを推定する (dataset.target = 'smpl' のとき使う)。

    出力:
        trans    [B, T, 3]      SMPL の並進 [m] (学習データでは水平 x, z = 0, y は絶対値)
        pose_6d  [B, T, 24, 6]  各関節の局所回転 (6D 表現)。手 (22, 23) は観測できないため
                                常に単位回転とし、22 関節だけを推定する
        contact  [B, T, 2]      左右の足の接地 (logits)

    関節位置・メッシュは学習モデルの外で SMPL (utils/smpl.py) に通して求める
    (SMPL のデータを state_dict に含めないため)。
    """
    NUM_PRED_JOINTS = 22

    def __init__(self, in_features=512, hidden=256, init_trans_y=0.0):
        super().__init__()
        J = self.NUM_PRED_JOINTS
        self.rotation_decoder = nn.Sequential(
            nn.Linear(in_features, hidden), nn.ReLU(), nn.Linear(hidden, J * 6))
        # 並進: MLP + 線形のスキップ経路。MLP だけだと、しゃがみ・座りなど頻度の低い低い姿勢で
        # ReLU がすべて不活性になり出力が定数 (≒ 平均の腰の高さ) に張り付くことがあったため
        self.trans_decoder = nn.Sequential(
            nn.Linear(in_features, hidden), nn.ReLU(), nn.Linear(hidden, 3))
        self.trans_skip = nn.Linear(in_features, 3)
        self.contact_decoder = nn.Sequential(
            nn.Linear(in_features, 64), nn.ReLU(), nn.Linear(64, 2))

        identity_6d = torch.tensor([1.0, 0, 0, 0, 1.0, 0])
        with torch.no_grad():
            # 学習開始時の出力を「単位回転」「平均的な腰の高さ」付近にする
            last = self.rotation_decoder[-1]
            last.weight.mul_(0.1)
            last.bias.copy_(identity_6d.repeat(J))
            self.trans_decoder[-1].bias.copy_(torch.tensor([0.0, init_trans_y, 0.0]))
            # スキップ経路は 0 から始める (ランダム初期化だと学習初期の並進が大きく乱れ、
            # 関節位置の損失に引っ張られて膝が骨軸まわりに 180° ねじれた局所解に落ちることがあった)
            self.trans_skip.weight.zero_()
            self.trans_skip.bias.zero_()
        self.register_buffer('hand_6d', identity_6d.repeat(2, 1), persistent=False)

    def forward(self, body_latent):
        B, T, _ = body_latent.shape
        pose_6d = self.rotation_decoder(body_latent).view(B, T, self.NUM_PRED_JOINTS, 6)
        hands = self.hand_6d.expand(B, T, 2, 6)
        pose_6d = torch.cat([pose_6d, hands], dim=2)                     # [B, T, 24, 6]
        trans = self.trans_decoder(body_latent) + self.trans_skip(body_latent)
        contact = self.contact_decoder(body_latent)
        return trans, pose_6d, contact


if __name__ == '__main__':
    B, T, F = 2, 10, 512
    model = PoseDecoder(in_features=F, num_joints=17)
    body = torch.randn(B, T, F)
    pos, rot, contact = model(body)
    print("Positions shape:", pos.shape)
    print("Rotations shape:", rot.shape)
    print("Contact shape:", contact.shape)
