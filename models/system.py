"""
モデル全体 (KinematicFusionSystem)。

入力  : 左右インソール圧力 + 5 部位の IMU (+ 体型パラメータ)
出力  : 18 関節の位置 / 回転 / 接地確率, および中間表現 body_latent

    PressureEncoder ─┐
                     ├─ FootFusion ─ LowerBodyNet ─┐
    IMUEncoder ──────┤                             ├─ KinematicFusion ─ PoseDecoder
                     └──────────── UpperBodyNet ───┘

※ 属性名 (pressure_encoder など) は学習済み重み (state_dict) のキーになるため変更しないこと。
"""
import torch.nn as nn

from models.pressure_encoder import PressureEncoder
from models.imu_encoder import IMUEncoder
from models.foot_fusion import FootFusion
from models.lower_body_net import LowerBodyNet
from models.upper_body_net import UpperBodyNet
from models.kinematic_fusion import KinematicFusion
from models.pose_decoder import PoseDecoder


class KinematicFusionSystem(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        # config の model セクションが無ければ各クラスのデフォルト値を使う
        mdl = (config or {}).get('model', {})

        self.pressure_encoder = PressureEncoder(**mdl.get('pressure_encoder', {}))
        self.imu_encoder = IMUEncoder(**mdl.get('imu_encoder', {}))
        self.foot_fusion = FootFusion(**mdl.get('foot_fusion', {}))
        self.lower_body_net = LowerBodyNet(**mdl.get('lower_body_net', {}))
        self.upper_body_net = UpperBodyNet(**mdl.get('upper_body_net', {}))
        self.kinematic_fusion = KinematicFusion(**mdl.get('kinematic_fusion', {}))
        self.pose_decoder = PoseDecoder(**mdl.get('pose_decoder', {}))

    def forward(self, left_pressure, right_pressure, imu_data, body_params):
        # 1. Encoders
        lp_feat, rp_feat = self.pressure_encoder(left_pressure, right_pressure)
        lw_feat, rw_feat, la_feat, ra_feat, pelvis_feat = self.imu_encoder(imu_data)

        # 2. Foot Fusion
        lf_feat, rf_feat = self.foot_fusion(lp_feat, rp_feat, la_feat, ra_feat)

        # 3. Lower Body
        lower_latent = self.lower_body_net(lf_feat, rf_feat, pelvis_feat, body_params)

        # 4. Upper Body
        upper_latent = self.upper_body_net(lw_feat, rw_feat, pelvis_feat, lower_latent, body_params)

        # 5. Kinematic Fusion
        body_latent = self.kinematic_fusion(upper_latent, lower_latent)

        # 6. Pose Decoder
        pos, rot, contact = self.pose_decoder(body_latent)

        return pos, rot, contact, body_latent
