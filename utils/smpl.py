"""
SMPL 人体モデル (PyTorch 実装。smplx / chumpy / scipy に依存しない)。

    smpl = SMPL.from_config(config, gender='male').to(device)
    out = smpl(pose_rotmat, betas, transl, return_verts=True)
    out.joints   [B, 24, 3]  関節位置 [m]
    out.verts    [B, 6890, 3] 頂点 [m]  (return_verts=True のとき)
    out.glob_rot [B, 24, 3, 3] 各関節のグローバル回転

モデルファイル:
    SMPL 公式サイトの pkl (v1.0.0 の basicModel_*_lbs_10_207_0_v1.0.0.pkl など) をそのまま読む。
    pkl 内の chumpy 配列・scipy 疎行列は専用の unpickler で numpy 配列に変換する。
    ライセンス上モデルファイルは再配布できないため、body_models/ は git 管理外にしている。
    また SMPL のパラメータを学習モデルの state_dict に含めないよう、SMPL は
    KinematicFusionSystem の外 (損失・後処理) で使う。

座標系: y 軸が上, 正面が +z, 左が +x (mocap の BVH と同じ向き)。単位は m。
"""
import pickle
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from utils.rotation import axis_angle_to_matrix

NUM_JOINTS = 24
NUM_BETAS = 10

JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hand', 'right_hand',
]
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]

# 疎センサーからは観測できない手 (22, 23) を除いた 22 関節。学習の損失・評価はこの関節で行う
BODY_JOINTS = list(range(22))

# モデルファイル名の候補 (大文字小文字は区別しない)
_MODEL_PATTERNS = {
    'male': ['basicmodel_m_lbs_10_207_0_v1.*.pkl', 'smpl_male.pkl', '*_male*.pkl'],
    'female': ['basicmodel_f_lbs_10_207_0_v1.*.pkl', 'smpl_female.pkl', '*_female*.pkl'],
    'neutral': ['basicmodel_neutral_lbs_10_207_0_v1.*.pkl', 'smpl_neutral.pkl', '*neutral*.pkl'],
}

SMPLOutput = namedtuple('SMPLOutput', ['joints', 'verts', 'glob_rot'])


# ----------------------------------------------------------------------
# pkl の読み込み (chumpy / scipy なし)
# ----------------------------------------------------------------------
class _Stub:
    """chumpy.Ch / scipy.sparse.csc_matrix の代わり。state をそのまま保持する。"""
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.state = state


class _SMPLUnpickler(pickle.Unpickler):
    # 想定外のクラスを読み込まないよう、許可するモジュールを限定する
    _ALLOWED = ('numpy', 'builtins', 'copy_reg', 'copyreg', '__builtin__', 'collections')

    def find_class(self, module, name):
        if module.startswith('chumpy') or module.startswith('scipy'):
            return type(name, (_Stub,), {})
        if module.split('.')[0] in self._ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f'SMPL pkl に想定外のクラスがあります: {module}.{name}')


def _to_array(v):
    if isinstance(v, _Stub):
        st = v.state
        if 'x' in st:                                   # chumpy.Ch
            return np.asarray(st['x'])
        if 'indptr' in st:                              # scipy.sparse.csc_matrix -> 密行列
            n_rows, n_cols = st['_shape']
            dense = np.zeros((n_rows, n_cols))
            indptr, indices, data = st['indptr'], st['indices'], st['data']
            for c in range(n_cols):
                s, e = indptr[c], indptr[c + 1]
                dense[indices[s:e], c] = data[s:e]
            return dense
        raise ValueError(f'未対応のオブジェクトです: {list(st)}')
    return np.asarray(v)


def load_smpl_pkl(path):
    with open(path, 'rb') as f:
        d = _SMPLUnpickler(f, encoding='latin1').load()
    return {
        'v_template': _to_array(d['v_template']),          # [6890, 3]
        'shapedirs': _to_array(d['shapedirs']),            # [6890, 3, 10]
        'posedirs': _to_array(d['posedirs']),              # [6890, 3, 207]
        'J_regressor': _to_array(d['J_regressor']),        # [24, 6890]
        'weights': _to_array(d['weights']),                # [6890, 24]
        'faces': np.asarray(d['f'], dtype=np.int64),       # [13776, 3]
        'parents': _to_array(d['kintree_table'])[0].astype(np.int64),
    }


def find_model_file(model_dir, gender):
    model_dir = Path(model_dir)
    if gender not in _MODEL_PATTERNS:
        raise ValueError(f'gender は male / female / neutral のいずれかです: {gender}')
    files = {p.name.lower(): p for p in model_dir.glob('*.pkl')}
    for pat in _MODEL_PATTERNS[gender]:
        for name, p in sorted(files.items()):
            if Path(name).match(pat):
                return p
    raise FileNotFoundError(f'{model_dir} に SMPL ({gender}) の pkl が見つかりません '
                            f'(候補: {_MODEL_PATTERNS[gender]})')


# ----------------------------------------------------------------------
# モデル本体
# ----------------------------------------------------------------------
class SMPL(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        d = load_smpl_pkl(model_path)
        if list(d['parents'][1:]) != PARENTS[1:]:
            raise ValueError(f'{model_path}: 関節の親子関係が SMPL の定義と異なります')
        f32 = lambda a: torch.as_tensor(a, dtype=torch.float32)  # noqa: E731
        self.model_path = str(model_path)
        self.register_buffer('v_template', f32(d['v_template']), persistent=False)
        self.register_buffer('shapedirs', f32(d['shapedirs']), persistent=False)
        self.register_buffer('posedirs', f32(d['posedirs']).reshape(-1, 207).T.contiguous(),
                             persistent=False)                           # [207, 6890*3]
        self.register_buffer('J_regressor', f32(d['J_regressor']), persistent=False)
        self.register_buffer('weights', f32(d['weights']), persistent=False)
        self.register_buffer('faces', torch.as_tensor(d['faces']), persistent=False)
        self.parents = PARENTS

    @classmethod
    def from_config(cls, config, gender):
        model_dir = (config.get('smpl') or {}).get('model_dir', 'body_models')
        from utils.paths import resolve
        return cls(find_model_file(resolve(model_dir), gender))

    @staticmethod
    def prepare_betas(betas):
        """β を NUM_BETAS 次元にそろえる (足りなければ 0 埋め, 多ければ切り捨て)。"""
        betas = torch.as_tensor(betas, dtype=torch.float32)
        if betas.shape[-1] < NUM_BETAS:
            betas = torch.cat([betas, betas.new_zeros(betas.shape[:-1] + (NUM_BETAS - betas.shape[-1],))], -1)
        return betas[..., :NUM_BETAS]

    def shaped(self, betas):
        """体型 β のレストポーズ頂点 [B, 6890, 3] と関節 [B, 24, 3]。"""
        betas = betas.to(self.v_template)
        v = self.v_template + torch.einsum('vck,bk->bvc', self.shapedirs, betas)
        J = torch.einsum('jv,bvc->bjc', self.J_regressor, v)
        return v, J

    def rest_joints(self, betas):
        return self.shaped(betas.reshape(-1, NUM_BETAS))[1]

    def forward_kinematics(self, rotmat, J_rest):
        """
        rotmat : [B, 24, 3, 3] 局所回転, J_rest : [B, 24, 3]
        戻り値 : グローバル回転 [B, 24, 3, 3], 関節位置 [B, 24, 3] (ルートは J_rest[:, 0])
        """
        return self.forward_kinematics_static(rotmat, J_rest)

    @staticmethod
    def forward_kinematics_static(rotmat, J_rest):
        """forward_kinematics と同じ (親子関係は全 SMPL 共通なのでモデルに依存しない)。"""
        glob_R = [rotmat[:, 0]]
        glob_p = [J_rest[:, 0]]
        for j in range(1, NUM_JOINTS):
            p = PARENTS[j]
            glob_R.append(glob_R[p] @ rotmat[:, j])
            glob_p.append(glob_p[p] + (glob_R[p] @ (J_rest[:, j] - J_rest[:, p])[..., None])[..., 0])
        return torch.stack(glob_R, 1), torch.stack(glob_p, 1)

    def forward(self, pose, betas, transl=None, return_verts=False):
        """
        pose   : [B, 24, 3, 3] 回転行列 または [B, 72] / [B, 24, 3] 軸角
        betas  : [B, 10] または [10] (全フレーム共通)
        transl : [B, 3] または None
        """
        if pose.dim() == 4:
            rotmat = pose
        else:
            rotmat = axis_angle_to_matrix(pose.reshape(pose.shape[0], NUM_JOINTS, 3))
        B = rotmat.shape[0]
        betas = betas.reshape(-1, NUM_BETAS).to(rotmat)
        if betas.shape[0] == 1:
            v_shaped, J_rest = self.shaped(betas)
            v_shaped, J_rest = v_shaped.expand(B, -1, -1), J_rest.expand(B, -1, -1)
        else:
            v_shaped, J_rest = self.shaped(betas)

        glob_R, joints = self.forward_kinematics(rotmat, J_rest)

        verts = None
        if return_verts:
            eye = torch.eye(3, dtype=rotmat.dtype, device=rotmat.device)
            pose_feat = (rotmat[:, 1:] - eye).reshape(B, -1)                  # [B, 207]
            v_posed = v_shaped + (pose_feat @ self.posedirs).reshape(B, -1, 3)
            # A_j = [G_j | p_j - G_j J_j]  (レストポーズ -> 姿勢の剛体変換)
            t = joints - (glob_R @ J_rest[..., None])[..., 0]
            T = torch.einsum('vj,bjmn->bvmn', self.weights, glob_R)          # [B, 6890, 3, 3]
            tt = torch.einsum('vj,bjm->bvm', self.weights, t)                 # [B, 6890, 3]
            verts = (T @ v_posed[..., None])[..., 0] + tt

        if transl is not None:
            transl = transl.reshape(B, 1, 3).to(joints)
            joints = joints + transl
            if verts is not None:
                verts = verts + transl
        return SMPLOutput(joints=joints, verts=verts, glob_rot=glob_R)


class SMPLBank:
    """
    性別ごとの SMPL をまとめて扱う (バッチ内に複数の被験者・性別が混在してもよい)。
    モデルは必要になった性別だけ読み込む。学習モデルの state_dict には含めない。

        bank = SMPLBank(config, device)
        J_rest = bank.rest_joints(betas, genders)            # [B, 24, 3]
        glob_R, joints = bank.forward_kinematics(rotmat, J_rest)
    """
    def __init__(self, config, device='cpu'):
        self.config = config
        self.device = torch.device(device)
        self.models = {}

    def _model(self, gender):
        if gender not in self.models:
            self.models[gender] = SMPL.from_config(self.config, gender).to(self.device)
        return self.models[gender]

    def rest_joints(self, betas, genders):
        """betas [B, 10], genders: 長さ B の文字列リスト (または 1 つの文字列) -> [B, 24, 3]"""
        if isinstance(genders, str):
            genders = [genders] * betas.shape[0]
        out = torch.empty(betas.shape[0], NUM_JOINTS, 3, device=betas.device, dtype=betas.dtype)
        for g in sorted(set(genders)):
            idx = [i for i, x in enumerate(genders) if x == g]
            out[idx] = self._model(g).rest_joints(betas[idx]).to(out)
        return out

    def forward_kinematics(self, rotmat, J_rest):
        """rotmat [..., 24, 3, 3], J_rest [..., 24, 3] (先頭の次元はそろえる)"""
        lead = rotmat.shape[:-3]
        R = rotmat.reshape(-1, NUM_JOINTS, 3, 3)
        J = J_rest.expand(lead + (NUM_JOINTS, 3)).reshape(-1, NUM_JOINTS, 3)
        glob_R, joints = SMPL.forward_kinematics_static(R, J)
        return glob_R.reshape(lead + (NUM_JOINTS, 3, 3)), joints.reshape(lead + (NUM_JOINTS, 3))

    def __call__(self, pose, betas, transl, gender, return_verts=False):
        """1 人分 (性別が 1 つ) の SMPL を計算する。引数は SMPL.forward と同じ。"""
        return self._model(gender)(pose, betas, transl, return_verts=return_verts)

    def faces(self, gender):
        return self._model(gender).faces
