"""
BVH (mocap のスケルトン書き出し) の読み込みと順運動学。

    bvh = load_bvh('skeleton1.bvh')
    glob_R, glob_p = bvh.forward_kinematics()   # [T, J, 3, 3], [T, J, 3] (mm, mocap のワールド座標)

関節には End Site も含める (名前は '<親>_End')。End Site は回転チャンネルを持たないので、
回転は親と同じ (局所回転 = 単位行列) として扱う。

規約:
    - 局所回転 R_local = Rx @ Ry @ Rz のように CHANNELS の記載順に掛ける (内因性)
    - p_child = p_parent + R_global_parent @ offset_child
    - R_global_child = R_global_parent @ R_local_child
"""
from dataclasses import dataclass, field

import numpy as np
import torch

from utils.rotation import euler_to_matrix


@dataclass
class BVH:
    names: list                       # 関節名 (End Site は '<親>_End')
    parents: list                     # 親インデックス (ルートは -1)
    offsets: np.ndarray               # [J, 3] 親からのオフセット (mm)
    rot_order: list                   # 関節ごとの回転チャンネル順 ('XYZ' など, End Site は None)
    frame_time: float
    root_pos: np.ndarray              # [T, 3] ルート位置 (mm)
    euler_deg: np.ndarray             # [T, J, 3] 回転 (deg, rot_order の順。End Site は 0)
    index: dict = field(default_factory=dict)

    def __post_init__(self):
        self.index = {n: i for i, n in enumerate(self.names)}

    @property
    def num_frames(self):
        return len(self.root_pos)

    def local_rotations(self, frames=slice(None), device='cpu', dtype=torch.float64):
        """局所回転行列 [T, J, 3, 3]。"""
        eul = torch.as_tensor(np.deg2rad(self.euler_deg[frames]), dtype=dtype, device=device)
        R = torch.eye(3, dtype=dtype, device=device).repeat(eul.shape[0], len(self.names), 1, 1)
        for j, order in enumerate(self.rot_order):
            if order is not None:
                R[:, j] = euler_to_matrix(eul[:, j], order)
        return R

    def forward_kinematics(self, frames=slice(None), device='cpu', dtype=torch.float64):
        """グローバル回転 [T, J, 3, 3] とグローバル位置 [T, J, 3] (mm) を返す (torch)。"""
        R_loc = self.local_rotations(frames, device, dtype)
        root = torch.as_tensor(self.root_pos[frames], dtype=dtype, device=device)
        off = torch.as_tensor(self.offsets, dtype=dtype, device=device)
        T, J = R_loc.shape[:2]
        glob_R = [None] * J
        glob_p = [None] * J
        for j in range(J):
            p = self.parents[j]
            if p < 0:
                glob_R[j] = R_loc[:, j]
                glob_p[j] = root + off[j]
            else:
                glob_R[j] = glob_R[p] @ R_loc[:, j]
                glob_p[j] = glob_p[p] + (glob_R[p] @ off[j])
        return torch.stack(glob_R, 1), torch.stack(glob_p, 1)

    def rest_positions(self):
        """レストポーズ (回転 0) での関節位置 [J, 3] (ルート = 原点)。"""
        pos = np.zeros((len(self.names), 3))
        for j, p in enumerate(self.parents):
            pos[j] = self.offsets[j] if p < 0 else pos[p] + self.offsets[j]
        return pos


def load_bvh(path):
    with open(path, 'r', encoding='utf-8') as f:
        tokens_lines = f.read().split('\n')

    names, parents, offsets, orders, channels = [], [], [], [], []
    stack = []
    i = 0
    # ---------------- HIERARCHY ----------------
    while i < len(tokens_lines):
        tok = tokens_lines[i].split()
        i += 1
        if not tok:
            continue
        key = tok[0]
        if key in ('ROOT', 'JOINT'):
            names.append(tok[1])
            parents.append(stack[-1] if stack else -1)
            offsets.append(None)
            orders.append(None)
            channels.append([])
            stack.append(len(names) - 1)
        elif key == 'End':
            parent = stack[-1]
            names.append(f'{names[parent]}_End')
            parents.append(parent)
            offsets.append(None)
            orders.append(None)
            channels.append([])
            stack.append(len(names) - 1)
        elif key == 'OFFSET':
            offsets[stack[-1]] = [float(v) for v in tok[1:4]]
        elif key == 'CHANNELS':
            chs = tok[2:2 + int(tok[1])]
            channels[stack[-1]] = chs
            rot = ''.join(c[0].upper() for c in chs if c.lower().endswith('rotation'))
            orders[stack[-1]] = rot or None
        elif key == '}':
            stack.pop()
        elif key == 'MOTION':
            break

    # ---------------- MOTION ----------------
    num_frames = int(tokens_lines[i].split(':')[1])
    frame_time = float(tokens_lines[i + 1].split(':')[1])
    data = np.loadtxt(tokens_lines[i + 2:i + 2 + num_frames], dtype=np.float64, ndmin=2)
    if data.shape[0] != num_frames:
        raise ValueError(f'{path}: フレーム数が一致しません ({data.shape[0]} != {num_frames})')

    J = len(names)
    root_pos = np.zeros((num_frames, 3))
    euler = np.zeros((num_frames, J, 3))
    col = 0
    for j in range(J):
        rot_k = 0
        for ch in channels[j]:
            c = ch.lower()
            if c.endswith('position'):
                root_pos[:, 'xyz'.index(c[0])] = data[:, col]
            else:
                euler[:, j, rot_k] = data[:, col]
                rot_k += 1
            col += 1
    if col != data.shape[1]:
        raise ValueError(f'{path}: チャンネル数 ({col}) とデータの列数 ({data.shape[1]}) が一致しません')

    return BVH(names=names, parents=parents, offsets=np.asarray(offsets, dtype=np.float64),
               rot_order=orders, frame_time=frame_time, root_pos=root_pos, euler_deg=euler)
