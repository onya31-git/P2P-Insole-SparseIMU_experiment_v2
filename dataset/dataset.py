"""
インソール (圧力 + IMU) + スパース IMU のデータセット。

ディレクトリ構成 (data_dir = config.yaml の paths.data_dir):
    {data_dir}/sensor/{train,test}/D1S1A1_AL.csv      ... センサーデータ
    {data_dir}/skeleton/{train,test}/D1S1A1_skeleton.csv ... スケルトン (クオータニオン形式)
    {smpl_dir}/{train,test}/D1S1A1_smpl.npz            ... SMPL パラメータ (target='smpl' のとき。
                                                           scripts/fit_smpl.py で BVH から作成)

ファイル命名規則 (詳細は data/README.md):
    D: Data collection Number, S: Subject Number,
    A: Action Section Number (A0 はキャリブレーション), T: Test 用データ
    部位: AL=arm left, AR=arm right, FL=foot left, FR=foot right, BC=back center
    ※ "D1S1A3_FR_.csv" や "D1S1A1skeleton.csv" のようなアンダースコアの揺れも正規表現で吸収する。

センサーCSVの形式:
    timestamp_ms, P1..PN, Mag_x..z, Gyro_x..z, Acc_x..z, frame_seq, time_ref
    - AL / AR / BC : IMU のみ使用 (Gyro 3 + Acc 3 = 6 次元)
    - FL / FR      : 圧力 8x14 = 112 点 (P1..P112) + IMU (Gyro + Acc)
    - timestamp_ms は全センサー共通のエポックミリ秒

時刻同期:
    センサー5台は共通時計で互いに同期しているが、mocap (スケルトン) との
    オフセットは収録ごとに 0.4〜1.8 秒程度ばらつく。そこで収録ごとに
    「センサーの角速度ノルム」と「対応する関節の速度」の相互相関から
    オフセットを1つ推定し、センサー側をずらしてスケルトンに合わせる
    (estimate_sync_offset)。センサー範囲外となるフレームは切り捨てる。

正解の形式 (target):
    'skeleton' : mocap の 18 関節の位置 (正規化済み) と回転 (従来の方式)
    'smpl'     : 上に加えて SMPL の姿勢 (6D 回転)・並進・関節位置。SMPL の npz は BVH と同じ
                 フレーム (= スケルトン CSV の行) に対応するので、CSV と同じフレームを切り出す。
    いずれの場合も body_params には被験者の β (config の smpl.subjects) を入れる (無ければ 0)。
"""
import os
import re
import glob
from itertools import islice

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from utils.rotation import axis_angle_to_matrix, matrix_to_rot6d
from utils.smpl import NUM_BETAS


SENSOR_RE = re.compile(r'^(D\d+S\d+[AT]\d+)_?(AL|AR|BC|FL|FR)_?\.csv$', re.IGNORECASE)
SKELETON_RE = re.compile(r'^(D\d+S\d+[AT]\d+)_?skeleton(?:_q)?_?\.csv$', re.IGNORECASE)
SUBJECT_RE = re.compile(r'^(D\d+S\d+)', re.IGNORECASE)

IMU_COLS = ['Gyro_x', 'Gyro_y', 'Gyro_z', 'Acc_x', 'Acc_y', 'Acc_z']
PRESSURE_ROWS, PRESSURE_COLS = 8, 14
NUM_PRESSURE = PRESSURE_ROWS * PRESSURE_COLS  # 112

# スケルトンCSVのヘッダーに現れる関節名 (この順序がモデル出力の関節インデックスになる)
JOINT_NAMES = [
    "CenterOfGravity", "SpinePosition", "Spine4Position", "HeadPosition",
    "LShoulderPositions", "LElbowPositions", "LWristPositions",
    "RShoulderPositions", "RElbowPositions", "RWristPositions",
    "LHipPositions", "LKneePositions", "LAnklePositions", "LToePositions",
    "RHipPositions", "RKneePositions", "RAnklePositions", "RToePositions",
]

# 骨 (親関節, 子関節) の一覧。JOINT_NAMES のインデックス。骨長の損失と評価に使う
KINEMATIC_TREE = [
    (0, 1), (1, 2), (2, 3),
    (2, 4), (4, 5), (5, 6),
    (2, 7), (7, 8), (8, 9),
    (0, 10), (10, 11), (11, 12), (12, 13),
    (0, 14), (14, 15), (15, 16), (16, 17)
]

# スケルトンCSVで Angles 列がクオータニオンではなくヒンジ角になっている関節 (肘)
HINGE_JOINTS = {"LElbowPositions", "RElbowPositions"}

SKEL_FPS = 100.0

# mocap の関節 (JOINT_NAMES のインデックス) -> 対応する SMPL 関節 (scripts/fit_smpl.py の CORR と同じ対応)。
# CenterOfGravity と SpinePosition には対応する SMPL 関節が無い。
MOCAP_TO_SMPL = {
    2: 9, 3: 15,
    4: 16, 5: 18, 6: 20, 7: 17, 8: 19, 9: 21,
    10: 1, 11: 4, 12: 7, 13: 10, 14: 2, 15: 5, 16: 8, 17: 11,
}

# 接地ラベル (compute_foot_contact) の判定に使う関節と閾値
FOOT_JOINTS = [("LAnklePositions", "LToePositions"), ("RAnklePositions", "RToePositions")]
# 閾値はインソール圧力の合計 (足裏の物理的な接地) との一致度が最も高くなるように決めた
# (一致率 91〜94%, Cohen's κ 0.70〜0.79; 高さは 15 mm が最良, 速さは 300 mm/s 以上でほぼ横ばい)
CONTACT_HEIGHT_MARGIN = 15.0   # 床からの高さ [mm]
CONTACT_SPEED_THRESH = 400.0   # 速さ [mm/s]

# 同期推定に使う (センサー部位, スケルトン関節インデックス) の対応
#   JOINT_NAMES の順序: 0 CoG, 6 LWrist, 9 RWrist, 12 LAnkle, 16 RAnkle
SYNC_PAIRS = [('AL', 6), ('AR', 9), ('FL', 12), ('FR', 16), ('BC', 0)]


def subject_key(data_id):
    """'D1S1A2' -> 'D1S1'。正規化統計量はこのキー (データ収集回 x 被験者) 単位で持つ。"""
    m = SUBJECT_RE.match(data_id)
    return m.group(1).upper() if m else data_id


def _hinge_to_quat(included_angle):
    """
    肘のヒンジ角をクオータニオン (W, X, Y, Z) に変換する。

    スケルトンCSVの肘 (L/RElbowAngles) は 4 列のうち Y 列に
    「上腕と前腕のなす角 [rad]」(伸展で π) が入っており、X, Z 列は常に 0、
    W 列は屈曲に連動する別の量で、クオータニオンではない。
    そこで屈曲角 θ = π - なす角 を求め、ローカル X 軸まわりの回転
    q = (cos θ/2, sin θ/2, 0, 0) として表す。

    ※ 回転軸 (X) は仮定。同じ軸まわりの回転同士の差は角度差だけで決まるため、
      学習の損失・評価値には影響しない。
    """
    theta = np.pi - included_angle
    q = np.zeros((len(theta), 4))
    q[:, 0] = np.cos(theta / 2)
    q[:, 1] = np.sin(theta / 2)
    return q


def compute_foot_contact(pos, height_margin=CONTACT_HEIGHT_MARGIN, speed_thresh=CONTACT_SPEED_THRESH):
    """
    mocap の足の動きから接地ラベルを作る。

    左右それぞれ、かかと側 (足首) とつま先のどちらかが
      「床から height_margin [mm] 以内の高さ」かつ「速さ speed_thresh [mm/s] 未満」
    であれば接地 (1) とする。床の高さは収録ごと・関節ごとに、その収録中の
    高さの 1 パーセンタイルとする (足首とつま先は床に着いても高さが異なるため)。

    Parameters
    ----------
    pos : [T, J, 3]  グローバル座標の関節位置 [mm] (ルート相対化する前, y が鉛直)
    戻り値 : [T, 2]  左足, 右足 (float, 0 or 1)
    """
    contact = np.zeros((len(pos), 2))
    for side, joints in enumerate(FOOT_JOINTS):
        for j in joints:
            p = pos[:, JOINT_NAMES.index(j)]
            floor = np.percentile(p[:, 1], 1)
            speed = np.linalg.norm(np.gradient(p, axis=0), axis=1) * SKEL_FPS
            on = (p[:, 1] < floor + height_margin) & (speed < speed_thresh)
            contact[:, side] = np.maximum(contact[:, side], on)
    return contact


def _smooth(x, win):
    return np.convolve(x, np.ones(win) / win, mode='same')


def _zscore(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def _xcorr_curve(a, b, max_lag):
    """z-score 済みの a, b について r(L) = mean_t a[t] * b[t+L] を L=-max_lag..max_lag で返す (FFT)。"""
    n = len(a)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    c = np.fft.irfft(np.conj(np.fft.rfft(a, nfft)) * np.fft.rfft(b, nfft), nfft)
    lags = np.arange(-max_lag, max_lag + 1)
    vals = c[lags % nfft] / (n - np.abs(lags))  # 重なり長で割って相関係数に近づける
    return lags, vals


def estimate_sync_offset(skel_t, skel_pos, sensor_t, sensor_gyro, max_lag_sec=5.0, smooth_sec=0.2):
    """
    センサーとスケルトンの時刻オフセット (秒) を推定する。

    Parameters
    ----------
    skel_t : [T]  スケルトン時刻 (秒, 100 Hz 等間隔)
    skel_pos : [T, J, 3]  スケルトンのグローバル関節位置 (ルート相対化する前)
    sensor_t : {part: [N] 秒}  共通原点からのセンサー時刻
    sensor_gyro : {part: [N, 3]}  ジャイロ
    戻り値 : (offset_sec, peak_r, {part: (offset_sec, peak_r)})
        スケルトン時刻 t に対応するセンサー時刻は t + offset_sec。
    """
    t_grid = skel_t
    win = max(1, int(round(smooth_sec * SKEL_FPS)))
    max_lag = int(round(max_lag_sec * SKEL_FPS))

    curves, per_part = [], {}
    for part, j in SYNC_PAIRS:
        if part not in sensor_t:
            continue
        vel = np.gradient(skel_pos[:, j, :], axis=0) * SKEL_FPS
        a = _zscore(_smooth(np.linalg.norm(vel, axis=1), win))
        gyro = np.linalg.norm(sensor_gyro[part], axis=1)
        b = _zscore(_smooth(np.interp(t_grid, sensor_t[part], gyro), win))
        lags, r = _xcorr_curve(a, b, max_lag)
        curves.append(r)
        k = int(np.argmax(r))
        per_part[part] = (lags[k] / SKEL_FPS, float(r[k]))

    mean_r = np.mean(curves, axis=0)
    k = int(np.argmax(mean_r))
    return lags[k] / SKEL_FPS, float(mean_r[k]), per_part


class InsoleIMUDataset(Dataset):
    """
    出力テンソルの形状 (1 サンプル = seq_len フレームのチャンク):
        left_pressure / right_pressure : [T, 112, 1]
        imu_data                       : [T, 5, 6]  順序: AL, AR, FL, FR, BC
        gt_joint_pos                   : [T, 18, 3]
        gt_joint_rot                   : [T, 18, 6] (先頭4次元が関節の相対回転 W,X,Y,Z, 残り2次元は0)
        gt_contact                     : [T, 2]     左足, 右足の接地 (1=接地, compute_foot_contact)
        body_params                    : [10]       被験者の体型 β (config の smpl.subjects。無ければ 0)
        data_id                        : 正規化に使った被験者キー (例: 'D1S1')
        pos_mean / pos_std             : [18, 3]    位置の正規化統計量 (pos_mm = pos * pos_std + pos_mean)
      target='smpl' のとき追加:
        gt_smpl_pose                   : [T, 24, 6] SMPL の局所回転 (6D)
        gt_smpl_trans                  : [T, 3]     SMPL の並進 [m]。水平 (x, z) は 0 (ルート相対), y は絶対値
        gt_smpl_joints                 : [T, 24, 3] SMPL の関節位置 [m] (水平方向は並進基準)
        gender                         : 'male' / 'female' / 'neutral'

    正規化統計量:
        学習時 (stats=None) は被験者キー (例: D1S1) ごとに、その被験者の
        全アクションを結合して計算する。テスト時は学習時に保存した統計量を
        stats 引数で渡すことで、学習時と同じ正規化を適用する
        (train=D1S1A*, test=D1S1T* と ID が異なるため)。
    """

    def __init__(self, data_path, seq_len=64, mode='train', stats=None,
                 exclude_calibration=True, sync_cfg=None, target='skeleton', smpl_cfg=None):
        """
        Parameters
        ----------
        data_path : str
            sensor/ と skeleton/ を含むデータディレクトリ。
        mode : 'train' or 'test'
            読み込むサブディレクトリ名。
        stats : dict or None
            {subject_key: {pos_mean, pos_std, lp_mean, ...}}。
            None の場合はこのデータから計算する。
        exclude_calibration : bool
            True の場合 A0 (キャリブレーション) を読み込まない。
        sync_cfg : dict or None
            時刻同期の設定 (config.yaml の sync セクション)。
              enabled      : bool   自動推定を行うか (既定 True)
              max_lag_sec  : float  探索範囲 ±秒 (既定 5.0)
              min_corr     : float  これ未満の相関なら警告 (既定 0.3)
              offsets      : {data_id: 秒}  手動指定 (自動推定より優先)
        target : 'skeleton' or 'smpl'
            正解の形式 (モジュールの docstring 参照)。
        smpl_cfg : dict or None
            config.yaml の smpl セクション (data_dir, subjects)。
        """
        if target not in ('skeleton', 'smpl'):
            raise ValueError(f"target は 'skeleton' か 'smpl' です: {target}")
        self.target = target
        smpl_cfg = smpl_cfg or {}
        from utils.paths import resolve
        self.smpl_dir = os.path.join(str(resolve(smpl_cfg.get('data_dir', 'data/smpl'))), mode)
        self.subjects = {k.upper(): v for k, v in (smpl_cfg.get('subjects') or {}).items()}
        # 被験者ごとの SMPL の情報 {subject_key: {gender, betas, offsets, offset_mask}} (評価で使う)
        self.smpl_info = {}
        sync_cfg = sync_cfg or {}
        self.sync_enabled = sync_cfg.get('enabled', True)
        self.sync_max_lag = float(sync_cfg.get('max_lag_sec', 5.0))
        self.sync_min_corr = float(sync_cfg.get('min_corr', 0.3))
        self.sync_manual = {k.upper(): float(v) for k, v in (sync_cfg.get('offsets') or {}).items()}
        # 収録ごとの同期結果 {data_id: {offset_sec, source, peak_r, per_sensor, frames_kept}}
        self.sync_info = {}
        self.mode = mode
        self.input_dir = os.path.join(data_path, 'sensor', mode)
        self.skeleton_dir = os.path.join(data_path, 'skeleton', mode)
        self.seq_len = seq_len
        self.exclude_calibration = exclude_calibration

        self.BP = NUM_BETAS   # 体型パラメータ (SMPL の β) の次元
        self.J = len(JOINT_NAMES)

        self.id_stats = {}
        if stats is not None:
            self.id_stats = {k: {kk: np.asarray(vv) if isinstance(vv, list) else vv
                                 for kk, vv in st.items()}
                             for k, st in stats.items()}
        self._stats_given = stats is not None

        self.data_seqs = self._load_and_align_data()
        self.num_samples = len(self.data_seqs)
        if self.num_samples == 0:
            raise RuntimeError(
                f"[Dataset] 有効なシーケンスがありません: {self.skeleton_dir}, {self.input_dir}")

    # ------------------------------------------------------------------
    # スケルトンCSVの解析
    # ------------------------------------------------------------------
    def _parse_skeleton(self, csv_path):
        """
        スケルトンCSV (mocap の書き出し) を読み込む。
        3行目 (関節名) から各関節の Position と Angles の列を探し、
        4行目 (軸ラベル) で Angles の成分の並びを判定する。

        回転は親関節に対する相対回転 (関節角) で、W,X,Y,Z の順に並べ替えて返す
        (CSV 上は X,Y,Z,W の順)。例外として次の関節は単位クオータニオンに変換する:
          - CenterOfGravity : Angles 列が無いので単位クオータニオン [1,0,0,0]
          - 肘 (HINGE_JOINTS) : CSV にはクオータニオンではなくヒンジ角が入っている
                                (詳細は _hinge_to_quat)
        符号は W >= 0 に揃える (q と -q は同じ回転のため)。

        戻り値: (time_sec [T], pos [T, J, 3], rot [T, J, 6])
                rot の先頭4次元が W,X,Y,Z, 残り2次元は 0
        """
        with open(csv_path, 'r') as f:
            head = list(islice(f, 6))
        if len(head) < 6:
            return None
        row2 = head[2].rstrip('\n').split(',')   # 関節名
        row3 = head[3].rstrip('\n').split(',')   # 軸ラベル (X, Y, Z, W)

        pos_cols = []
        for j in JOINT_NAMES:
            if j in row2:
                idx = row2.index(j)
                pos_cols.extend([idx, idx + 1, idx + 2])
            else:
                pos_cols.extend([0, 0, 0])

        df = pd.read_csv(csv_path, skiprows=5, header=None, low_memory=False)
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

        time_sec = df[0].values * (1.0 / SKEL_FPS)
        pos_data = df[pos_cols].values.reshape(-1, self.J, 3)

        # 既定は単位クオータニオン (W, X, Y, Z) = (1, 0, 0, 0)
        rot_quat = np.zeros((len(df), self.J, 4))
        rot_quat[..., 0] = 1.0
        for ji, j in enumerate(JOINT_NAMES):
            if j == "CenterOfGravity":
                continue  # Angles 列が無い
            ang_name = j.replace("Positions", "Angles").replace("Position", "Angles")
            if ang_name not in row2:
                print(f"  警告: {os.path.basename(csv_path)} に {ang_name} が無いため単位回転で埋めます")
                continue
            idx = row2.index(ang_name)
            axis_col = {row3[idx + k].strip(): idx + k for k in range(4)}
            if set(axis_col) != {'W', 'X', 'Y', 'Z'}:
                raise ValueError(f"{csv_path}: {ang_name} の軸ラベルが想定外です ({list(axis_col)})")

            if j in HINGE_JOINTS:
                rot_quat[:, ji] = _hinge_to_quat(df[axis_col['Y']].values)
            else:
                rot_quat[:, ji] = df[[axis_col[a] for a in 'WXYZ']].values
                norm = np.linalg.norm(rot_quat[:, ji], axis=1)
                if np.max(np.abs(norm - 1.0)) > 1e-2:
                    print(f"  警告: {ang_name} に単位クオータニオンでない値があります "
                          f"(|q| = {norm.min():.3f}〜{norm.max():.3f})")

        # q と -q は同じ回転を表す。CSV では途中で符号が反転するフレームがあり、
        # 学習の正解値が不連続になるため W >= 0 の側に揃える
        # (このデータでは |W| >= 0.18 で 180° 付近の回転が無いため、揃えても不連続は生じない)
        rot_quat[rot_quat[..., 0] < 0] *= -1

        # モデルの回転出力 (6次元) に合わせ、4次元のクオータニオンを 0 でパディングする
        rot_6d = np.zeros((rot_quat.shape[0], self.J, 6))
        rot_6d[:, :, :4] = rot_quat

        return time_sec, pos_data, rot_6d

    # ------------------------------------------------------------------
    # センサーCSVの解析
    # ------------------------------------------------------------------
    @staticmethod
    def _read_sensor_csv(csv_path, with_pressure):
        df = pd.read_csv(csv_path, low_memory=False)
        df = df.sort_values('timestamp_ms').drop_duplicates('timestamp_ms')
        t_ms = df['timestamp_ms'].to_numpy(dtype=np.float64)
        imu = df[IMU_COLS].apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy(np.float64)
        press = None
        if with_pressure:
            p_cols = [f'P{i}' for i in range(1, NUM_PRESSURE + 1)]
            missing = [c for c in p_cols if c not in df.columns]
            if missing:
                raise ValueError(f"{csv_path}: 圧力列が不足しています ({missing[:3]}...)")
            press = df[p_cols].apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy(np.float64)
        return t_ms, imu, press

    def _find_sensor_files(self):
        """{data_id: {part: path}} を返す。"""
        files = {}
        for path in glob.glob(os.path.join(self.input_dir, '*.csv')):
            m = SENSOR_RE.match(os.path.basename(path))
            if not m:
                continue
            data_id, part = m.group(1).upper(), m.group(2).upper()
            files.setdefault(data_id, {})[part] = path
        return files

    def _find_skeleton_files(self):
        """{data_id: path} を返す。"""
        files = {}
        for path in glob.glob(os.path.join(self.skeleton_dir, '*.csv')):
            m = SKELETON_RE.match(os.path.basename(path))
            if m:
                files[m.group(1).upper()] = path
        return files

    # ------------------------------------------------------------------
    # 読み込み・時刻合わせ
    # ------------------------------------------------------------------
    def _load_and_align_data(self):
        skel_files = self._find_skeleton_files()
        sensor_files = self._find_sensor_files()
        if not skel_files:
            print(f"[Dataset] スケルトンCSVが見つかりません: {self.skeleton_dir}")
            return []

        def interp_feat(t_target, t_src, feat_src):
            res = np.zeros((len(t_target), feat_src.shape[1]))
            for i in range(feat_src.shape[1]):
                res[:, i] = np.interp(t_target, t_src, feat_src[:, i])
            return res

        # 各IDを読み込み (統計量計算のため一旦すべて保持)
        sessions = []
        for data_id in sorted(skel_files):
            if self.exclude_calibration and re.search(r'A0$', data_id):
                continue
            parts = sensor_files.get(data_id, {})
            missing = [p for p in ('AL', 'AR', 'BC', 'FL', 'FR') if p not in parts]
            if missing:
                print(f"[Dataset] ID={data_id}: センサーファイル不足 {missing} のためスキップ")
                continue

            print(f"[Dataset] ID={data_id} を読み込み中...")
            try:
                skel_t, pos, rot = self._parse_skeleton(skel_files[data_id])
            except Exception as e:
                print(f"  スケルトン読み込み失敗: {e}")
                continue

            key = subject_key(data_id)
            betas = self._subject_betas(key)
            smpl = None
            if self.target == 'smpl':
                smpl = self._load_smpl(data_id, key, betas, len(skel_t))
                if smpl is None:
                    continue

            raw = {}
            for part in ('AL', 'AR', 'BC'):
                t, imu, _ = self._read_sensor_csv(parts[part], with_pressure=False)
                raw[part] = (t, imu, None)
            for part in ('FL', 'FR'):
                raw[part] = self._read_sensor_csv(parts[part], with_pressure=True)

            # 全センサー共通のエポック時計なので、最も早い開始時刻を共通原点にする
            t0 = min(r[0][0] for r in raw.values())
            sens_t = {part: (r[0] - t0) / 1000.0 for part, r in raw.items()}

            # mocap との時刻オフセット (センサー時刻 = スケルトン時刻 + offset)
            offset, info = self._resolve_sync_offset(
                data_id, skel_t, pos, sens_t,
                {part: r[1][:, 0:3] for part, r in raw.items()})  # IMU_COLS の先頭3列 = Gyro

            # ずらした結果センサーの記録範囲外になるフレームは補間できないので切り捨てる
            tq = skel_t + offset
            lo = max(t[0] for t in sens_t.values())
            hi = min(t[-1] for t in sens_t.values())
            keep = (tq >= lo) & (tq <= hi)
            if keep.sum() < self.seq_len:
                print(f"  ID={data_id}: 同期後の有効フレームが {keep.sum()} しかないためスキップ")
                continue
            skel_t, pos, rot, tq = skel_t[keep], pos[keep], rot[keep], tq[keep]
            if smpl is not None:
                smpl = {k: v[keep] for k, v in smpl.items()}
            info['frames_kept'] = f"{int(keep.sum())}/{len(keep)}"
            self.sync_info[data_id] = info

            aligned = {}
            for part, (_, imu, press) in raw.items():
                aligned[part] = (
                    interp_feat(tq, sens_t[part], imu),
                    interp_feat(tq, sens_t[part], press) if press is not None else None,
                )

            # IMU の順序: AL=左手首, AR=右手首, FL=左足首, FR=右足首, BC=骨盤 (IMUEncoder の入力順)
            imu_combined = np.stack(
                [aligned['AL'][0], aligned['AR'][0], aligned['FL'][0],
                 aligned['FR'][0], aligned['BC'][0]], axis=1)          # [T, 5, 6]
            press_fl = aligned['FL'][1]                                  # [T, 112]
            press_fr = aligned['FR'][1]                                  # [T, 112]

            # 接地ラベル (ルート相対化する前のグローバル座標で計算する)
            contact = compute_foot_contact(pos)                          # [T, 2]

            # ルート (CenterOfGravity) 相対座標へ変換
            # 水平方向 (x, z) のみルート相対にし、鉛直方向 (y) は絶対値を保持する
            # （ジャンプ等の上下動を可視化で確認できるようにするため）
            root_pos = pos[:, 0:1, :].copy()
            root_pos[..., 1] = 0.0
            pos = pos - root_pos

            print(f"  同期: offset={offset:+.2f}s ({info['source']}"
                  + (f", r={info['peak_r']:.2f}" if info.get('peak_r') is not None else '')
                  + f"), 使用フレーム {info['frames_kept']}")

            print(f"  接地率: 左 {contact[:, 0].mean():.0%}, 右 {contact[:, 1].mean():.0%}")

            sessions.append(dict(data_id=data_id, key=key, pos=pos, rot=rot,
                                 imu=imu_combined, lp=press_fl, rp=press_fr, contact=contact,
                                 betas=betas, smpl=smpl))

        # ------------------------------------------------------------------
        # 被験者キーごとに統計量を計算 (stats が与えられていない場合)
        # ------------------------------------------------------------------
        if not self._stats_given:
            for key in sorted({s['key'] for s in sessions}):
                group = [s for s in sessions if s['key'] == key]
                self.id_stats[key] = self._compute_group_stats(group)
                print(f"[Dataset] 統計量を計算しました: {key} "
                      f"({[s['data_id'] for s in group]})")
        else:
            for key in sorted({s['key'] for s in sessions}):
                if key not in self.id_stats:
                    group = [s for s in sessions if s['key'] == key]
                    print(f"[Dataset] 警告: 統計量に {key} がないため、このデータから計算します")
                    self.id_stats[key] = self._compute_group_stats(group)

        # ------------------------------------------------------------------
        # seq_len ごとのチャンクに分割
        # ------------------------------------------------------------------
        all_seqs = []
        for s in sessions:
            n = s['pos'].shape[0]
            count = 0
            for i in range(0, n - self.seq_len, self.seq_len):
                e = i + self.seq_len
                chunk = {
                    'data_id':        s['key'],                        # 正規化に使うキー
                    'left_pressure':  s['lp'][i:e, :, np.newaxis],     # [T, 112, 1]
                    'right_pressure': s['rp'][i:e, :, np.newaxis],     # [T, 112, 1]
                    'imu_data':       s['imu'][i:e],                   # [T, 5, 6]
                    'gt_joint_pos':   s['pos'][i:e],                   # [T, J, 3]
                    'gt_joint_rot':   s['rot'][i:e],                   # [T, J, 6]
                    'gt_contact':     s['contact'][i:e],               # [T, 2] 左, 右
                    'body_params':    s['betas'],                      # [10] 体型 β
                }
                if s['smpl'] is not None:
                    chunk['gt_smpl_pose'] = s['smpl']['pose6d'][i:e]       # [T, 24, 6]
                    chunk['gt_smpl_trans'] = s['smpl']['trans'][i:e]       # [T, 3]
                    chunk['gt_smpl_joints'] = s['smpl']['joints'][i:e]     # [T, 24, 3]
                    chunk['gender'] = self.smpl_info[s['key']]['gender']
                all_seqs.append(chunk)
                count += 1
            print(f"  ID={s['data_id']}: チャンク {count} 個 (累計 {len(all_seqs)})")
        return all_seqs

    def _subject_betas(self, key):
        """config の smpl.subjects から被験者の β [10] を返す (無ければ 0)。"""
        if self.target == 'smpl' and key not in self.subjects:
            raise ValueError(f"config.yaml の smpl.subjects に {key} がありません")
        betas = np.zeros(self.BP)
        b = list((self.subjects.get(key) or {}).get('betas') or [])[:self.BP]
        betas[:len(b)] = b
        return betas

    def _load_smpl(self, data_id, key, betas, num_frames):
        """SMPL の npz を読み、正解の形式に変換する (フレームはスケルトン CSV と同じ)。"""
        path = os.path.join(self.smpl_dir, f'{data_id}_smpl.npz')
        if not os.path.exists(path):
            print(f"  ID={data_id}: {path} がないためスキップ (scripts/fit_smpl.py を実行してください)")
            return None
        z = np.load(path)
        if len(z['poses']) != num_frames:
            raise ValueError(f"{path}: フレーム数 {len(z['poses'])} がスケルトン CSV ({num_frames}) と異なります")
        if np.max(np.abs(z['betas'] - betas)) > 1e-4:
            raise ValueError(f"{path}: フィッティング時の β が config.yaml の {key} の β と異なります。"
                             f"scripts/fit_smpl.py --overwrite で作り直してください")
        gender = str(z['gender'])
        cfg_gender = (self.subjects.get(key) or {}).get('gender', gender)
        if cfg_gender != gender:
            raise ValueError(f"{path}: フィッティング時の性別 ({gender}) が config.yaml ({cfg_gender}) と異なります。"
                             f"scripts/fit_smpl.py --overwrite で作り直してください")
        self.smpl_info[key] = dict(gender=gender, betas=betas, offsets=z['offsets'],
                                   offset_mask=z['offset_mask'])

        T = num_frames
        aa = torch.as_tensor(z['poses'], dtype=torch.float32).reshape(T, -1, 3)
        pose6d = matrix_to_rot6d(axis_angle_to_matrix(aa)).numpy()          # [T, 24, 6]
        trans = z['trans'].astype(np.float64)
        root_h = trans.copy()
        root_h[:, 1] = 0.0                                                  # 水平方向 (x, z) だけ取り除く
        return dict(pose6d=pose6d, trans=trans - root_h,
                    joints=z['joints'] - root_h[:, None, :])

    def _resolve_sync_offset(self, data_id, skel_t, skel_pos, sens_t, sens_gyro):
        """手動指定 > 自動推定 > 0 の優先順でオフセット (秒) を決める。"""
        if data_id in self.sync_manual:
            return self.sync_manual[data_id], {'offset_sec': self.sync_manual[data_id],
                                               'source': 'manual', 'peak_r': None}
        if not self.sync_enabled:
            return 0.0, {'offset_sec': 0.0, 'source': 'disabled', 'peak_r': None}

        offset, peak_r, per_part = estimate_sync_offset(
            skel_t, skel_pos, sens_t, sens_gyro, max_lag_sec=self.sync_max_lag)
        info = {'offset_sec': offset, 'source': 'auto', 'peak_r': peak_r,
                'per_sensor': {p: {'offset_sec': o, 'r': r} for p, (o, r) in per_part.items()}}
        print("  センサー別推定: " + ", ".join(
            f"{p} {o:+.2f}s(r={r:.2f})" for p, (o, r) in per_part.items()))
        if peak_r < self.sync_min_corr:
            print(f"  警告: ID={data_id} の同期相関が低い (r={peak_r:.2f})。"
                  f"config の sync.offsets で手動指定を検討してください。")
        if abs(abs(offset) - self.sync_max_lag) < 0.05:
            print(f"  警告: ID={data_id} の推定値が探索範囲の端 ({offset:+.2f}s) です。"
                  f"sync.max_lag_sec を広げてください。")
        return offset, info

    @staticmethod
    def _compute_group_stats(group):
        pos = np.concatenate([s['pos'] for s in group], axis=0)
        lp = np.concatenate([s['lp'] for s in group], axis=0)
        rp = np.concatenate([s['rp'] for s in group], axis=0)
        imu = np.concatenate([s['imu'] for s in group], axis=0)

        # 位置は外れ値に強いロバスト統計量 (中央値 / 5-95 パーセンタイル幅)
        pos_std = (np.percentile(pos, 95, axis=0) - np.percentile(pos, 5, axis=0)) / 2.0
        pos_std = np.clip(pos_std, a_min=1e-4, a_max=None)
        return {
            'pos_mean': np.median(pos, axis=0),
            'pos_std':  pos_std,
            'lp_mean':  float(np.mean(lp)),
            'lp_std':   float(np.std(lp) + 1e-6),
            'rp_mean':  float(np.mean(rp)),
            'rp_std':   float(np.std(rp) + 1e-6),
            'imu_mean': np.mean(imu, axis=0),
            'imu_std':  np.std(imu, axis=0) + 1e-6,
        }

    # ------------------------------------------------------------------
    # Dataset インターフェース
    # ------------------------------------------------------------------
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """チャンクを被験者キーの統計量で正規化してテンソルにする。"""
        chunk = self.data_seqs[idx]
        st = self.id_stats.get(chunk['data_id'], next(iter(self.id_stats.values())))
        norm = {
            'gt_joint_pos':   (st['pos_mean'], st['pos_std']),
            'left_pressure':  (st['lp_mean'], st['lp_std']),
            'right_pressure': (st['rp_mean'], st['rp_std']),
            'imu_data':       (st['imu_mean'], st['imu_std']),
        }
        out = {}
        for k, v in chunk.items():
            if k in ('data_id', 'gender'):
                out[k] = v  # 文字列はそのまま
                continue
            if k in norm:
                mean, std = norm[k]
                v = (v - mean) / std
            out[k] = torch.tensor(v, dtype=torch.float32)
        # 位置を mm に戻すための統計量 (骨長の損失・テスト時の逆正規化で使う)
        out['pos_mean'] = torch.tensor(st['pos_mean'], dtype=torch.float32)  # [J, 3]
        out['pos_std'] = torch.tensor(st['pos_std'], dtype=torch.float32)    # [J, 3]
        return out


def get_dataloader(data_path, batch_size=32, seq_len=64, mode='train',
                   num_workers=0, stats=None, sync_cfg=None, target='skeleton', smpl_cfg=None):
    dataset = InsoleIMUDataset(data_path, seq_len=seq_len, mode=mode, stats=stats,
                               sync_cfg=sync_cfg, target=target, smpl_cfg=smpl_cfg)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=num_workers,
        pin_memory=(num_workers > 0),
    )


if __name__ == '__main__':
    # 動作確認: python -m dataset.dataset
    from utils.paths import load_config, get_paths
    cfg = load_config()
    loader = get_dataloader(str(get_paths(cfg).data_dir), batch_size=2,
                            seq_len=cfg['dataset'].get('seq_len', 200), mode='train',
                            sync_cfg=cfg.get('sync'), target=cfg['dataset'].get('target', 'skeleton'),
                            smpl_cfg=cfg.get('smpl'))
    print(f"Dataset size: {len(loader.dataset)} sequences")
    print(f"Stats keys  : {list(loader.dataset.id_stats.keys())}")
    print(f"Sync        : { {k: round(v['offset_sec'], 2) for k, v in loader.dataset.sync_info.items()} }")
    batch = next(iter(loader))
    for k, v in batch.items():
        print(f"{k:15s}: {tuple(v.shape) if hasattr(v, 'shape') else v}")
