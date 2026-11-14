"""
回転表現の相互変換 (PyTorch。numpy 配列は torch.as_tensor で渡す)。

    軸角 (axis-angle)   [..., 3]   SMPL のパラメータ形式 (データセットの保存形式)
    回転行列            [..., 3, 3]
    6D 表現             [..., 6]   ネットワークの出力形式 (Zhou et al., CVPR 2019)
                                   回転行列の第 1, 2 列を並べたもの
    オイラー角          BVH の読み込みのみ (euler_to_matrix)

いずれも列ベクトル規約 (v' = R v) で、合成は R_parent @ R_local。
"""
import torch
import torch.nn.functional as F


def axis_angle_to_matrix(aa):
    """[..., 3] -> [..., 3, 3] (Rodrigues)。角度 0 付近も安定に計算する。"""
    angle = torch.linalg.norm(aa, dim=-1, keepdim=True)            # [..., 1]
    small = angle < 1e-6
    axis = aa / torch.where(small, torch.ones_like(angle), angle)
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    K = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], dim=-1).reshape(aa.shape[:-1] + (3, 3))
    s = torch.sin(angle)[..., None]
    c = torch.cos(angle)[..., None]
    eye = torch.eye(3, dtype=aa.dtype, device=aa.device).expand(K.shape)
    R = eye + s * K + (1 - c) * (K @ K)
    # 角度が極小のときは 1 次近似 I + [aa]_x (勾配を保つ)
    ax, ay, az = aa.unbind(-1)
    K0 = torch.stack([zero, -az, ay, az, zero, -ax, -ay, ax, zero], dim=-1).reshape(K.shape)
    return torch.where(small[..., None], eye + K0, R)


def matrix_to_quaternion(R):
    """[..., 3, 3] -> [..., 4] (W, X, Y, Z), W >= 0。"""
    m = R
    tr = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    cands = torch.stack([
        1.0 + tr,
        1.0 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2],
        1.0 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2],
        1.0 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2],
    ], dim=-1)
    k = cands.argmax(dim=-1, keepdim=True)                          # 数値的に最も安定な分岐
    s = torch.sqrt(torch.clamp(cands.gather(-1, k), min=1e-12)) * 2  # = 4 * 成分
    s = s.squeeze(-1)
    q_w = torch.stack([s / 4,
                       (m[..., 2, 1] - m[..., 1, 2]) / s,
                       (m[..., 0, 2] - m[..., 2, 0]) / s,
                       (m[..., 1, 0] - m[..., 0, 1]) / s], -1)
    q_x = torch.stack([(m[..., 2, 1] - m[..., 1, 2]) / s,
                       s / 4,
                       (m[..., 0, 1] + m[..., 1, 0]) / s,
                       (m[..., 0, 2] + m[..., 2, 0]) / s], -1)
    q_y = torch.stack([(m[..., 0, 2] - m[..., 2, 0]) / s,
                       (m[..., 0, 1] + m[..., 1, 0]) / s,
                       s / 4,
                       (m[..., 1, 2] + m[..., 2, 1]) / s], -1)
    q_z = torch.stack([(m[..., 1, 0] - m[..., 0, 1]) / s,
                       (m[..., 0, 2] + m[..., 2, 0]) / s,
                       (m[..., 1, 2] + m[..., 2, 1]) / s,
                       s / 4], -1)
    q = torch.stack([q_w, q_x, q_y, q_z], dim=-2)                   # [..., 4(分岐), 4]
    q = q.gather(-2, k[..., None].expand(k.shape[:-1] + (1, 4))).squeeze(-2)
    q = F.normalize(q, dim=-1)
    return torch.where(q[..., :1] < 0, -q, q)


def matrix_to_axis_angle(R):
    """[..., 3, 3] -> [..., 3]。角度は [0, π]。"""
    q = matrix_to_quaternion(R)                                     # W >= 0
    xyz = q[..., 1:]
    sin_half = torch.linalg.norm(xyz, dim=-1, keepdim=True)
    angle = 2 * torch.atan2(sin_half, q[..., :1])
    # sin(θ/2) が小さいときは θ / sin(θ/2) ≈ 2
    scale = torch.where(sin_half < 1e-6, torch.full_like(angle, 2.0), angle / sin_half.clamp(min=1e-12))
    return xyz * scale


def rot6d_to_matrix(d6):
    """[..., 6] -> [..., 3, 3]。Gram-Schmidt で正規直交化する (列 = b1, b2, b3)。"""
    a1, a2 = d6[..., 0:3], d6[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def matrix_to_rot6d(R):
    """[..., 3, 3] -> [..., 6] (第 1 列, 第 2 列)。"""
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def geodesic_angle(R1, R2):
    """2 つの回転の角度差 [rad]。[..., 3, 3] -> [...]"""
    R = R1.transpose(-1, -2) @ R2
    cos = (R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1) / 2
    return torch.acos(torch.clamp(cos, -1 + 1e-7, 1 - 1e-7))


def _basic_rot(axis, angle):
    c, s = torch.cos(angle), torch.sin(angle)
    one, zero = torch.ones_like(angle), torch.zeros_like(angle)
    if axis == 'X':
        m = [one, zero, zero, zero, c, -s, zero, s, c]
    elif axis == 'Y':
        m = [c, zero, s, zero, one, zero, -s, zero, c]
    else:
        m = [c, -s, zero, s, c, zero, zero, zero, one]
    return torch.stack(m, -1).reshape(angle.shape + (3, 3))


def euler_to_matrix(angles_rad, order):
    """
    オイラー角 -> 回転行列。order は BVH の CHANNELS の並び (例 'XYZ')。
    BVH の規約どおり R = R_order[0] @ R_order[1] @ R_order[2] (内因性・記載順)。
    angles_rad : [..., 3] (order の順)
    """
    R = None
    for i, ax in enumerate(order):
        Ri = _basic_rot(ax, angles_rad[..., i])
        R = Ri if R is None else R @ Ri
    return R


def minimal_rotation(a, b):
    """ベクトル a を b の向きに回す最小回転 [3, 3] (a, b は [3])。"""
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    v = torch.cross(a, b, dim=-1)
    c = (a * b).sum(-1)
    s = torch.linalg.norm(v)
    if s < 1e-8:
        if c > 0:
            return torch.eye(3, dtype=a.dtype, device=a.device)
        # 正反対: a に直交する任意の軸で 180°
        tmp = torch.tensor([1.0, 0, 0], dtype=a.dtype, device=a.device)
        if abs(a[0]) > 0.9:
            tmp = torch.tensor([0, 1.0, 0], dtype=a.dtype, device=a.device)
        axis = F.normalize(torch.cross(a, tmp, dim=-1), dim=-1)
        return axis_angle_to_matrix(axis * torch.pi)
    return axis_angle_to_matrix(v / s * torch.atan2(s, c))
