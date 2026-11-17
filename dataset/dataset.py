"""
インソール (圧力 + IMU) + スパース IMU のデータセット。

ディレクトリ構成 (data_dir = config.yaml の paths.data_dir):
    {data_dir}/sensor/{train,test}/D1S1A1_AL.csv      ... センサーデータ
    {data_dir}/skeleton/{train,test}/D1S1A1_skeleton.csv ... スケルトン (クオータニオン形式)
    {smpl_dir}/{train,test}/D1S1A1_smpl.npz            ... SMPL パラメータ (target='smpl' のとき。
                                                           scripts/fit_smpl.py で BVH から作成)

使う収録の選び方:
    ids を指定しない : mode と同じ名前のフォルダ ({train,test}/) にある収録をすべて使う
    ids を指定する   : train/ と test/ のどちらにあっても、ID が一致する収録を使う
                       (例 ['D2S1A1', 'D2S1A2'] / ワイルドカード可 ['D2S1A*', 'D1*'])

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
    mocap (スケルトン) とセンサーのオフセットは収録ごとに 0.4〜1.8 秒程度ばらつく。そこで収録ごとに
    「センサーの角速度ノルム」と「対応する関節の速度」の相互相関からオフセットを推定し、
    センサー側をずらしてスケルトンに合わせる。センサー範囲外となるフレームは切り捨てる。
    センサーの時計はデバイスの種類ごとにずれることがある (D2 ではインソール FL / FR だけ約 0.5 秒ずれていた) ため、
    同じ時計のセンサーのまとまり (config の sync.groups) ごとにオフセットを推定し、
    まとまり間の差はセンサー時刻の補正 (clock_shift_sec) として先に取り除く。

正解の形式 (target):
    'skeleton' : mocap の 18 関節の位置 (正規化済み) と回転 (従来の方式)
    'smpl'     : 上に加えて SMPL の姿勢 (6D 回転)・並進・関節位置。SMPL の npz は BVH と同じ
                 フレーム (= スケルトン CSV の行) に対応するので、CSV と同じフレームを切り出す。
    いずれの場合も body_params には被験者の β (config の smpl.subjects) を入れる (無ければ 0)。

キャリブレーション (config の calibration。詳細は dataset/calibration.py):
    - 机置きデータ (sensor/calib/D?S?A0_*.csv) → ジャイロのバイアスと圧力のゼロ点を引く
    - 収録の最初 / 最後の T ポーズ → IMU の取り付け向き (重力に対する傾き) の補正と、
      圧力を T ポーズ時の合計 (体重) で割る体重基準化
    結果は outputs/calib/{train,test}.json に保存される。
"""
import os
import re
import glob
import fnmatch
from itertools import islice

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from dataset import calibration as calib
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

# 接地ラベル (compute_pressure_contact) の閾値: 圧力の合計が p1 + 比率 x (p95 - p1) を超えたら接地
#   D1 では mocap による判定 (compute_foot_contact) と κ 0.82〜0.83 で一致する。D2 のように mocap の
#   高さがドリフトしたり体の位置がその場に固定されていたり (歩行で立脚中の足が後ろへ動く) しても使える
CONTACT_PRESS_FRAC = 0.1

# 接地ラベル (compute_foot_contact。contact_source = 'mocap' のとき) の判定に使う関節と閾値
FOOT_JOINTS = [("LAnklePositions", "LToePositions"), ("RAnklePositions", "RToePositions")]
# 閾値はインソール圧力の合計 (足裏の物理的な接地) との一致度が最も高くなるように決めた
# (一致率 91〜94%, Cohen's κ 0.70〜0.79; 高さは 15 mm が最良, 速さは 300 mm/s 以上でほぼ横ばい)
CONTACT_HEIGHT_MARGIN = 15.0   # 床からの高さ [mm]
CONTACT_SPEED_THRESH = 400.0   # 速さ [mm/s]

# 同期推定に使う (センサー部位, スケルトン関節インデックス) の対応
#   JOINT_NAMES の順序: 0 CoG, 6 LWrist, 9 RWrist, 12 LAnkle, 16 RAnkle
SYNC_PAIRS = [('AL', 6), ('AR', 9), ('FL', 12), ('FR', 16), ('BC', 0)]
# 同じ時計を共有するセンサーのまとまり (config の sync.groups の既定値): 腕・背中の IMU / インソール
DEFAULT_SYNC_GROUPS = [['AL', 'AR', 'BC'], ['FL', 'FR']]


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


def rot_y(theta):
    """鉛直軸 (y) まわりに theta [rad] 回す回転行列 (z 軸を x 軸の方へ回す)。"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


# 床の高さの推定 (estimate_floor) に使う関節と窓
FLOOR_JOINTS = ['LToePositions', 'RToePositions', 'LAnklePositions', 'RAnklePositions']
FLOOR_WIN_SEC = 10.0


def estimate_floor(pos, contact, win_sec=FLOOR_WIN_SEC, step=50, min_count=20):
    """
    mocap の床の高さ [mm] をフレームごとに推定する。

    どちらかの足が接地しているフレームの「足首・つま先のうち最も低い関節の高さ」の、前後 win_sec / 2 秒の中央値。
    mocap の床の高さは収録ごとに違い (D2 では約 50 mm), 収録中にもドリフトするため、
    この値を引いて「接地中のつま先の関節の高さ ≒ 0」にそろえる (関節の中心の高さなので床そのものではない)。

    Parameters
    ----------
    pos : [T, J, 3]  関節位置 [mm] (y が鉛直)
    contact : [T, 2]  接地ラベル
    戻り値 : [T] または None (接地フレームが足りないとき)
    """
    low = pos[:, [JOINT_NAMES.index(j) for j in FLOOR_JOINTS], 1].min(axis=1)
    on = contact.max(axis=1) > 0
    T, half = len(pos), int(win_sec * SKEL_FPS / 2)
    centers = np.arange(0, T, step)
    vals = np.full(len(centers), np.nan)
    for k, i in enumerate(centers):
        sl = slice(max(0, i - half), i + half)
        x = low[sl][on[sl]]
        if len(x) >= min_count:
            vals[k] = np.median(x)
    ok = ~np.isnan(vals)
    if not ok.any():
        return None
    return np.interp(np.arange(T), centers[ok], vals[ok])


def compute_pressure_contact(press_l, press_r, frac=CONTACT_PRESS_FRAC):
    """
    インソール圧力の合計から接地ラベルを作る。

    足を上げてもインソールには靴の締め付けなどで一定の荷重が残るため、ゼロではなく
    収録中の圧力の合計の 1 パーセンタイル (足を上げたとき) を基準にし、95 パーセンタイル
    (荷重時) との間の frac の位置を閾値とする。

    Parameters
    ----------
    press_l, press_r : [T, 112]  時刻を合わせた左右の圧力 (正規化する前)
    戻り値 : [T, 2]  左足, 右足 (float, 0 or 1)
    """
    contact = np.zeros((len(press_l), 2))
    for side, p in enumerate((press_l, press_r)):
        s = p.sum(axis=1)
        lo, hi = np.percentile(s, 1), np.percentile(s, 95)
        contact[:, side] = s > lo + frac * (hi - lo)
    return contact


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


def sync_curves(skel_t, skel_pos, sensor_t, sensor_gyro, max_lag_sec=5.0, smooth_sec=0.2):
    """
    センサーごとに「角速度ノルム」と「対応する関節の速度」の相互相関をラグの関数として求める。

    Parameters
    ----------
    skel_t : [T]  スケルトン時刻 (秒, 100 Hz 等間隔)
    skel_pos : [T, J, 3]  スケルトンのグローバル関節位置 (ルート相対化する前)
    sensor_t : {part: [N] 秒}  共通原点からのセンサー時刻
    sensor_gyro : {part: [N, 3]}  ジャイロ
    戻り値 : (lags_sec [L], {part: r [L]})
        ラグ L の相関はスケルトン時刻 t とセンサー時刻 t + L の対応の良さ。
    """
    win = max(1, int(round(smooth_sec * SKEL_FPS)))
    max_lag = int(round(max_lag_sec * SKEL_FPS))
    lags, curves = np.arange(-max_lag, max_lag + 1), {}
    for part, j in SYNC_PAIRS:
        if part not in sensor_t:
            continue
        vel = np.gradient(skel_pos[:, j, :], axis=0) * SKEL_FPS
        a = _zscore(_smooth(np.linalg.norm(vel, axis=1), win))
        gyro = np.linalg.norm(sensor_gyro[part], axis=1)
        b = _zscore(_smooth(np.interp(skel_t, sensor_t[part], gyro), win))
        lags, curves[part] = _xcorr_curve(a, b, max_lag)
    return lags / SKEL_FPS, curves


def _peak(lags_sec, curves):
    """複数センサーの相関の平均が最大になるラグ (秒) と、その相関。"""
    r = np.mean(curves, axis=0)
    k = int(np.argmax(r))
    return float(lags_sec[k]), float(r[k])


def estimate_sync_offset(skel_t, skel_pos, sensor_t, sensor_gyro, max_lag_sec=5.0, smooth_sec=0.2):
    """
    センサーとスケルトンの時刻オフセット (秒) を全センサー共通の 1 つの値として推定する。
    戻り値 : (offset_sec, peak_r, {part: (offset_sec, peak_r)})
        スケルトン時刻 t に対応するセンサー時刻は t + offset_sec。
    """
    lags, curves = sync_curves(skel_t, skel_pos, sensor_t, sensor_gyro, max_lag_sec, smooth_sec)
    per_part = {p: _peak(lags, [c]) for p, c in curves.items()}
    offset, r = _peak(lags, list(curves.values()))
    return offset, r, per_part


class InsoleIMUDataset(Dataset):
    """
    出力テンソルの形状 (1 サンプル = seq_len フレームのチャンク):
        left_pressure / right_pressure : [T, 112, 1]
        imu_data                       : [T, 5, 6]  順序: AL, AR, FL, FR, BC
        gt_joint_pos                   : [T, 18, 3]
        gt_joint_rot                   : [T, 18, 6] (先頭4次元が関節の相対回転 W,X,Y,Z, 残り2次元は0)
        gt_contact                     : [T, 2]     左足, 右足の接地 (1=接地。contact_source に応じて
                                                compute_pressure_contact / compute_foot_contact)
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
                 exclude_calibration=True, sync_cfg=None, target='skeleton', smpl_cfg=None,
                 calib_cfg=None, ids=None, contact_source='pressure', canonical_heading=True,
                 floor_normalize=True):
        """
        Parameters
        ----------
        data_path : str
            sensor/ と skeleton/ を含むデータディレクトリ。
        mode : 'train' or 'test'
            学習かテストか (シャッフルの有無・T ポーズの確認方法が変わる)。
            ids を指定しない場合は、読み込むサブディレクトリ名も兼ねる。
        contact_source : 'pressure' or 'mocap'
            接地ラベルの作り方。pressure: インソール圧力の合計 (compute_pressure_contact) /
            mocap: 足の高さと速さ (compute_foot_contact。mocap の高さのドリフトや、体の位置が固定された歩行に弱い)
        canonical_heading : bool
            target='smpl' のとき、収録ごとに骨盤の平均の向きが +z になるよう正解 (SMPL・mocap の関節位置) を
            鉛直軸まわりに回す (回した角度は heading_info に入る)
        floor_normalize : bool
            target='smpl' のとき、接地中の足の高さから床の高さを推定し (estimate_floor)、正解の高さ
            (SMPL の並進・関節, mocap の関節位置) から引く。mocap の床の高さは収録ごとに違い、収録中にもドリフトするため
        ids : list of str, str or None
            使う収録の ID (ワイルドカード可)。指定すると train/ と test/ のどちらのフォルダに
            あるかに関係なく ID で探す。None なら mode のフォルダの収録をすべて使う。
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
        calib_cfg : dict or None
            config.yaml の calibration セクション (dataset/calibration.py の DEFAULTS を参照)。
            None なら既定値 (有効)。テスト時は学習時 (stats) と同じ設定である必要がある。
        """
        if target not in ('skeleton', 'smpl'):
            raise ValueError(f"target は 'skeleton' か 'smpl' です: {target}")
        self.target = target
        smpl_cfg = smpl_cfg or {}
        from utils.paths import resolve
        self.smpl_root = str(resolve(smpl_cfg.get('data_dir', 'data/smpl')))
        if isinstance(ids, str):
            ids = [ids]
        self.ids = [str(i).upper() for i in ids] if ids else None
        self.data_ids = []   # 実際に使った収録の ID (_load_and_align_data で設定)
        self.subjects = {k.upper(): v for k, v in (smpl_cfg.get('subjects') or {}).items()}
        # 被験者ごとの SMPL の情報 {subject_key: {gender, betas, offsets, offset_mask}} (評価で使う)
        self.smpl_info = {}
        sync_cfg = sync_cfg or {}
        self.sync_enabled = sync_cfg.get('enabled', True)
        self.sync_max_lag = float(sync_cfg.get('max_lag_sec', 5.0))
        self.sync_min_corr = float(sync_cfg.get('min_corr', 0.3))
        self.sync_manual = {k.upper(): float(v) for k, v in (sync_cfg.get('offsets') or {}).items()}
        # 同じ時計を共有するセンサーのまとまり (先頭のまとまりのオフセットが基準)
        groups = sync_cfg.get('groups') or DEFAULT_SYNC_GROUPS
        self.sync_groups = [[str(p).upper() for p in g] for g in groups]
        listed = sorted(p for g in self.sync_groups for p in g)
        if listed != sorted(p for p, _ in SYNC_PAIRS):
            raise ValueError(f"config の sync.groups には AL, AR, BC, FL, FR を 1 回ずつ書いてください: {groups}")
        # 収録ごとの同期結果 {data_id: {offset_sec, source, peak_r, per_sensor, frames_kept}}
        self.sync_info = {}
        self.mode = mode
        self.data_path = data_path
        self.calib = calib.merge_config(calib_cfg)
        # 収録ごとのキャリブレーション結果 (JSON に保存する) / 被験者キーごとの A0 の値 / 統計量に加える値
        self.calib_info = {}
        self._a0_cache = {}
        self._calib_stats = {}
        if contact_source not in ('pressure', 'mocap'):
            raise ValueError(f"contact_source は 'pressure' か 'mocap' です: {contact_source}")
        self.contact_source = contact_source
        self.canonical_heading = bool(canonical_heading) and target == 'smpl'
        self.heading_info = {}   # 収録ごとに回した向き [rad] (canonical_heading)
        self.floor_normalize = bool(floor_normalize) and target == 'smpl'
        self.floor_info = {}     # 収録ごとの床の高さ [mm] {p5, p50, p95} (floor_normalize)
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
        if self._stats_given and self.id_stats:
            trained = next(iter(self.id_stats.values())).get('calib_flags', 'disabled')
            current = calib.config_flags(self.calib)
            if trained != current:
                raise ValueError(
                    f"キャリブレーションの設定が学習時と異なります (学習時: {trained} / 現在: {current})。"
                    f"config.yaml の calibration を学習時と同じにするか、再学習してください")
            first = next(iter(self.id_stats.values()))
            for name, label in (('canonical_heading', '向きの正規化'), ('floor_normalize', '床の高さの補正')):
                trained, current = bool(first.get(name, False)), getattr(self, name)
                if trained != current:
                    raise ValueError(
                        f"{label} (config の dataset.{name}) が学習時と異なります "
                        f"(学習時: {trained} / 現在: {current})。設定を学習時と同じにするか、再学習してください")

        self.data_seqs = self._load_and_align_data()
        self.num_samples = len(self.data_seqs)
        if self.num_samples == 0:
            where = f"ID {self.ids}" if self.ids else f"sensor/{mode}, skeleton/{mode}"
            raise RuntimeError(f"[Dataset] 有効なシーケンスがありません ({where})")

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

    def _search_dirs(self, root):
        """
        収録を探すフォルダの一覧。ids 未指定なら root/<mode>、
        指定ありなら root の下のすべてのサブフォルダ (train, test など。机置きデータの calib は除く)。
        """
        if self.ids is None:
            return [os.path.join(root, self.mode)]
        return sorted(d for d in glob.glob(os.path.join(root, '*'))
                      if os.path.isdir(d) and os.path.basename(d) != 'calib')

    @staticmethod
    def _check_unique(found, key, path):
        if key in found and os.path.normpath(found[key]) != os.path.normpath(path):
            raise ValueError(f"同じ収録のファイルが複数のフォルダにあります: {found[key]} / {path}")

    def _find_sensor_files(self):
        """{data_id: {part: path}} を返す。"""
        files = {}
        for d in self._search_dirs(os.path.join(self.data_path, 'sensor')):
            for path in glob.glob(os.path.join(d, '*.csv')):
                m = SENSOR_RE.match(os.path.basename(path))
                if not m:
                    continue
                data_id, part = m.group(1).upper(), m.group(2).upper()
                self._check_unique(files.setdefault(data_id, {}), part, path)
                files[data_id][part] = path
        return files

    def _find_skeleton_files(self):
        """{data_id: path} を返す。ids 指定時は一致する収録だけに絞る。"""
        files = {}
        for d in self._search_dirs(os.path.join(self.data_path, 'skeleton')):
            for path in glob.glob(os.path.join(d, '*.csv')):
                m = SKELETON_RE.match(os.path.basename(path))
                if m:
                    self._check_unique(files, m.group(1).upper(), path)
                    files[m.group(1).upper()] = path
        if self.ids is None:
            return files

        available = sorted(files)
        selected = set()
        for pat in self.ids:
            hit = [i for i in available if fnmatch.fnmatchcase(i, pat)]
            if not hit:
                raise ValueError(f"ID '{pat}' に一致する収録 (スケルトン CSV) がありません。"
                                 f"使える ID: {', '.join(available)}")
            selected.update(hit)
        return {i: files[i] for i in sorted(selected)}

    def _smpl_path(self, data_id):
        """SMPL の npz のパス (ids 指定時は train / test のどちらのフォルダにあっても探す)。"""
        if self.ids is None:
            return os.path.join(self.smpl_root, self.mode, f'{data_id}_smpl.npz')
        hits = glob.glob(os.path.join(self.smpl_root, '*', f'{data_id}_smpl.npz'))
        if len(hits) > 1:
            raise ValueError(f"同じ収録の SMPL ファイルが複数のフォルダにあります: {hits}")
        return hits[0] if hits else os.path.join(self.smpl_root, '*', f'{data_id}_smpl.npz')

    # ------------------------------------------------------------------
    # 読み込み・時刻合わせ
    # ------------------------------------------------------------------
    def _load_and_align_data(self):
        skel_files = self._find_skeleton_files()
        sensor_files = self._find_sensor_files()
        if not skel_files:
            print(f"[Dataset] スケルトンCSVが見つかりません: {os.path.join(self.data_path, 'skeleton', self.mode)}")
            return []
        print(f"[Dataset] {self.mode} に使う収録: {', '.join(sorted(skel_files))}")

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

            # 机置きデータ (A0) のジャイロのバイアスと圧力のゼロ点を引く
            if self.calib['enabled']:
                self._apply_static_calib(data_id, key, raw)

            # 全センサー共通のエポック時計なので、最も早い開始時刻を共通原点にする
            t0 = min(r[0][0] for r in raw.values())
            sens_t = {part: (r[0] - t0) / 1000.0 for part, r in raw.items()}

            # mocap との時刻オフセット (センサー時刻 = スケルトン時刻 + offset)
            offset, info = self._resolve_sync_offset(
                data_id, skel_t, pos, sens_t,
                {part: r[1][:, 0:3] for part, r in raw.items()})  # IMU_COLS の先頭3列 = Gyro
            # センサーのまとまり間の時計のずれを補正 (以降は全センサー共通の offset で対応する)
            sens_t = {part: t - info['clock_shift_sec'].get(part, 0.0) for part, t in sens_t.items()}

            # T ポーズの候補 (収録の最初 / 最後の静止区間)。採否と補正は全収録を読んだ後に行う
            tpose = (self._tpose_candidates(sens_t, raw, skel_t, pos, offset)
                     if self.calib['enabled'] else None)

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

            # 接地ラベル (mocap はルート相対化する前のグローバル座標で計算する)
            if self.contact_source == 'pressure':
                contact = compute_pressure_contact(press_fl, press_fr)   # [T, 2]
            else:
                contact = compute_foot_contact(pos)                      # [T, 2]

            # 床の高さをそろえる (接地ラベルを使うので、その後に行う)
            if self.floor_normalize:
                floor = estimate_floor(pos, contact)
                if floor is None:
                    print(f"  警告: ID={data_id} は接地フレームが少なく床の高さを推定できないため、高さを補正しません")
                else:
                    pos = pos.copy()
                    pos[..., 1] -= floor[:, None]
                    smpl['trans'] = smpl['trans'].copy()
                    smpl['trans'][:, 1] -= floor / 1000.0
                    smpl['joints'] = smpl['joints'].copy()
                    smpl['joints'][..., 1] -= floor[:, None] / 1000.0
                    q = np.percentile(floor, [5, 50, 95])
                    self.floor_info[data_id] = dict(zip(('p5', 'p50', 'p95'), q.round(1).tolist()))
                    print(f"  床の高さの補正: p5 {q[0]:.0f} / p50 {q[1]:.0f} / p95 {q[2]:.0f} mm")

            # ルート (CenterOfGravity) 相対座標へ変換
            # 水平方向 (x, z) のみルート相対にし、鉛直方向 (y) は絶対値を保持する
            # （ジャンプ等の上下動を可視化で確認できるようにするため）
            root_pos = pos[:, 0:1, :].copy()
            root_pos[..., 1] = 0.0
            pos = pos - root_pos
            if data_id in self.heading_info:
                # SMPL の正解と同じ回転で mocap の関節位置の向きもそろえる (評価で比べるため)
                pos = pos @ rot_y(-self.heading_info[data_id]).T
                print(f"  向きの正規化: 骨盤の平均の向き {np.degrees(self.heading_info[data_id]):+.0f}° → 0°")

            print(f"  同期: offset={offset:+.2f}s ({info['source']}"
                  + (f", r={info['peak_r']:.2f}" if info.get('peak_r') is not None else '')
                  + f"), 使用フレーム {info['frames_kept']}")

            print(f"  接地率: 左 {contact[:, 0].mean():.0%}, 右 {contact[:, 1].mean():.0%}")

            sessions.append(dict(data_id=data_id, key=key, pos=pos, rot=rot,
                                 imu=imu_combined, lp=press_fl, rp=press_fr, contact=contact,
                                 betas=betas, smpl=smpl, tpose=tpose))

        # ------------------------------------------------------------------
        # T ポーズによる IMU の取り付け向きの補正と圧力の体重基準化
        # ------------------------------------------------------------------
        self.data_ids = [s['data_id'] for s in sessions]
        if self.calib['enabled']:
            self._apply_tpose_calib(sessions)
        else:
            self._calib_stats = {s['key']: {'calib_flags': 'disabled'} for s in sessions}

        # ------------------------------------------------------------------
        # 被験者キーごとに統計量を計算 (stats が与えられていない場合)
        # ------------------------------------------------------------------
        if not self._stats_given:
            for key in sorted({s['key'] for s in sessions}):
                group = [s for s in sessions if s['key'] == key]
                self.id_stats[key] = self._compute_group_stats(group)
                self.id_stats[key].update(self._calib_stats.get(key, {}))
                self.id_stats[key]['canonical_heading'] = self.canonical_heading
                self.id_stats[key]['floor_normalize'] = self.floor_normalize
                print(f"[Dataset] 統計量を計算しました: {key} "
                      f"({[s['data_id'] for s in group]})")
        else:
            for key in sorted({s['key'] for s in sessions}):
                if key not in self.id_stats:
                    group = [s for s in sessions if s['key'] == key]
                    print(f"[Dataset] 警告: 統計量に {key} がないため、このデータから計算します")
                    self.id_stats[key] = self._compute_group_stats(group)
                    self.id_stats[key].update(self._calib_stats.get(key, {}))
                    self.id_stats[key]['canonical_heading'] = self.canonical_heading
                    self.id_stats[key]['floor_normalize'] = self.floor_normalize

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
        path = self._smpl_path(data_id)
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
        aa = torch.as_tensor(z['poses'], dtype=torch.float64).reshape(T, -1, 3)
        R = axis_angle_to_matrix(aa)                                        # [T, 24, 3, 3]
        trans = z['trans'].astype(np.float64)
        joints = z['joints'].astype(np.float64)
        if self.canonical_heading:
            # 収録中の骨盤の平均の向きが +z (SMPL の正面) になるよう、世界座標を鉛直軸まわりに回す。
            # Mag を使わないため絶対的な向きはセンサーから分からず、mocap の世界座標の向きは収録環境しだい
            # (被験者が -z を向いていると、骨盤を 180° 反転させた局所解に陥り学習が進まない)
            fwd = R[:, 0, :, 2].numpy()                                     # 骨盤の前方向 (世界座標)
            yaw = np.arctan2(fwd[:, 0], fwd[:, 2])
            theta = float(np.arctan2(np.sin(yaw).mean(), np.cos(yaw).mean()))
            Ry = rot_y(-theta)
            R[:, 0] = torch.as_tensor(Ry) @ R[:, 0]
            pelvis = joints[:, 0]                                           # = 骨盤のレスト位置 + 並進
            trans = pelvis @ Ry.T - (pelvis - trans)
            joints = joints @ Ry.T
            self.heading_info[data_id] = theta
        pose6d = matrix_to_rot6d(R).float().numpy()                         # [T, 24, 6]
        root_h = trans.copy()
        root_h[:, 1] = 0.0                                                  # 水平方向 (x, z) だけ取り除く
        return dict(pose6d=pose6d, trans=trans - root_h,
                    joints=joints - root_h[:, None, :])

    def _resolve_sync_offset(self, data_id, skel_t, skel_pos, sens_t, sens_gyro):
        """
        手動指定 > 自動推定 > 0 の優先順でオフセット (秒) を決める。
        戻り値 : (offset_sec, info)
            offset_sec は基準のまとまり (sync.groups の先頭) のオフセット。他のまとまりとの差は
            info['clock_shift_sec'] = {part: 秒} に入れ、呼び出し側でセンサー時刻から引く
            (補正後はどのセンサーも「スケルトン時刻 t ↔ センサー時刻 t + offset_sec」で対応する)。
        """
        no_shift = {p: 0.0 for p in sens_t}
        if data_id in self.sync_manual:
            return self.sync_manual[data_id], {'offset_sec': self.sync_manual[data_id], 'source': 'manual',
                                               'peak_r': None, 'clock_shift_sec': no_shift}
        if not self.sync_enabled:
            return 0.0, {'offset_sec': 0.0, 'source': 'disabled', 'peak_r': None, 'clock_shift_sec': no_shift}

        lags, curves = sync_curves(skel_t, skel_pos, sens_t, sens_gyro, max_lag_sec=self.sync_max_lag)
        per_part = {p: _peak(lags, [c]) for p, c in curves.items()}
        all_offset, all_r = _peak(lags, list(curves.values()))
        print("  センサー別推定: " + ", ".join(
            f"{p} {o:+.2f}s(r={r:.2f})" for p, (o, r) in per_part.items()))

        # まとまりごとに推定。相関が低いまとまりは全センサー共通の推定値を使う
        group_res = []
        for g in self.sync_groups:
            o, r = _peak(lags, [curves[p] for p in g])
            fallback = len(self.sync_groups) > 1 and r < self.sync_min_corr
            if fallback:
                print(f"  警告: ID={data_id} の {'/'.join(g)} の同期相関が低い (r={r:.2f}) ため、"
                      f"全センサー共通の推定値 {all_offset:+.2f}s を使います")
                o = all_offset
            group_res.append({'parts': g, 'offset_sec': o, 'r': r, 'fallback': fallback})
        offset, peak_r = group_res[0]['offset_sec'], group_res[0]['r']
        shift = {p: g['offset_sec'] - offset for g in group_res for p in g['parts']}
        if len(group_res) > 1:
            print("  まとまり別: " + ", ".join(
                f"{'/'.join(g['parts'])} {g['offset_sec']:+.2f}s(r={g['r']:.2f})" for g in group_res)
                + f" → 時計の補正 " + ", ".join(f"{p} {s:+.2f}s" for p, s in shift.items() if s))

        info = {'offset_sec': offset, 'source': 'auto', 'peak_r': peak_r,
                'clock_shift_sec': shift,
                'groups': [{k: v for k, v in g.items()} for g in group_res],
                'all_sensors': {'offset_sec': all_offset, 'r': all_r},
                'per_sensor': {p: {'offset_sec': o, 'r': r} for p, (o, r) in per_part.items()}}
        if peak_r < self.sync_min_corr:
            print(f"  警告: ID={data_id} の同期相関が低い (r={peak_r:.2f})。"
                  f"config の sync.offsets で手動指定を検討してください。")
        for g in group_res:
            if abs(abs(g['offset_sec']) - self.sync_max_lag) < 0.05:
                print(f"  警告: ID={data_id} の推定値が探索範囲の端 ({g['offset_sec']:+.2f}s) です。"
                      f"sync.max_lag_sec を広げてください。")
        return offset, info

    # ------------------------------------------------------------------
    # キャリブレーション (処理の中身は dataset/calibration.py)
    # ------------------------------------------------------------------
    def _static_calib(self, key):
        """
        被験者キーに対応する机置きデータ (A0) のキャリブレーション値 (無ければ None)。
        A0 は収集の直前に 1 回だけ取るので、同じ D?S? → 同じ収集回 D? の他の被験者 の順に探す。
        """
        if key in self._a0_cache:
            return self._a0_cache[key]
        files = {}
        for path in glob.glob(os.path.join(self.data_path, 'sensor', 'calib', '*.csv')):
            m = SENSOR_RE.match(os.path.basename(path))
            if m and re.search(r'A0$', m.group(1), re.IGNORECASE):
                files.setdefault(m.group(1).upper(), {})[m.group(2).upper()] = path
        coll = re.match(r'(D\d+)', key).group(1)
        same_subject = sorted(d for d in files if subject_key(d) == key)
        same_collection = sorted(d for d in files if d.startswith(coll + 'S') and d not in same_subject)

        res = None
        for did in same_subject + same_collection:
            raw = {p: self._read_sensor_csv(path, with_pressure=(p in calib.FOOT_PARTS))
                   for p, path in files[did].items()}
            res = {'source': did, 'parts': calib.static_calibration(raw)}
            moving = {p: round(c['gyro_p95'], 2) for p, c in res['parts'].items() if c['gyro_p95'] > 5.0}
            print(f"[Calib] {key}: 机置きデータ {did} を使用 (ジャイロのバイアス "
                  + ", ".join(f"{p} {np.linalg.norm(c['gyro_bias']):.2f}" for p, c in res['parts'].items())
                  + " deg/s)")
            if moving:
                print(f"  警告: {did} の角速度が大きい部位があります {moving} deg/s (静止していない可能性)")
            break
        if res is None:
            print(f"[Calib] 警告: {key} の机置きデータ (sensor/calib/{coll}S?A0_*.csv) が無いため、"
                  f"ジャイロのバイアスと圧力のゼロ点は補正しません")
        self._a0_cache[key] = res
        return res

    def _apply_static_calib(self, data_id, key, raw):
        """raw (センサーごとの (t, imu, press)) のジャイロのバイアスと圧力のゼロ点を引いた値に置き換える。"""
        a0 = self._static_calib(key)
        info = self.calib_info.setdefault(data_id, {})
        info['static_calib'] = a0['source'] if a0 else None
        if a0 is None:
            return
        for part, (t, imu, press) in list(raw.items()):
            c = a0['parts'].get(part)
            if c is None:
                print(f"  警告: 机置きデータに {part} が無いため {part} は補正しません")
                continue
            if self.calib['gyro_bias']:
                imu = imu.copy()
                imu[:, :3] -= c['gyro_bias']
            if self.calib['pressure_zero'] and press is not None and c['press_zero'] is not None:
                press = np.clip(press - c['press_zero'], 0.0, None)
            raw[part] = (t, imu, press)

    def _tpose_candidates(self, sens_t, raw, skel_t, pos, offset):
        """
        収録の最初 / 最後の search_sec 秒の静止区間を T ポーズの候補として返す。
        {'start': [候補, ...], 'end': [...]}。候補は calibration.window_stats の値に
        mocap_ratio (mocap で T ポーズと判定されたフレームの割合, 確認しないときは None) と
        verified (True / False / None = 未確認) を加えたもの。mocap での確認は学習時のみ。
        """
        tc = self.calib['tpose']
        lo = max(t[0] for t in sens_t.values())
        hi = min(t[-1] for t in sens_t.values())
        span_ref = float(np.percentile(np.linalg.norm(pos[:, 6] - pos[:, 9], axis=1), 99))
        use_mocap = self.mode == 'train' and tc['mocap_check']
        out = {}
        for region, rng in (('start', (lo, min(hi, lo + tc['search_sec']))),
                            ('end', (max(lo, hi - tc['search_sec']), hi))):
            cands = []
            for t0, t1 in calib.find_static_runs(sens_t, raw, rng, tc):
                w = calib.window_stats(sens_t, raw, t0, t1)
                if w is None:
                    continue
                w['mocap_ratio'] = (calib.mocap_tpose_ratio(skel_t, pos, offset, t0, t1, tc, span_ref)
                                    if use_mocap else None)
                w['verified'] = None if w['mocap_ratio'] is None else w['mocap_ratio'] >= tc['mocap_min_ratio']
                cands.append(w)
            out[region] = cands
        return out

    def _tpose_reference(self, sessions):
        """部位ごとの基準の重力方向 (全被験者共通)。学習時は T ポーズの平均、テスト時は stats の値。"""
        if self._stats_given:
            for st in self.id_stats.values():
                ref = st.get('tpose_gravity_ref')
                if ref:
                    return {p: np.asarray(v, dtype=np.float64) for p, v in ref.items()}
            print("[Calib] 警告: stats に T ポーズの基準が無いため、IMU の取り付け向きは補正しません")
            return None
        pool = [w for s in sessions for c in s['tpose'].values() for w in c if w['verified'] is True]
        if not pool:
            # mocap で確認できない (mocap_check: false など) ときは各区間の最初の静止区間を使う
            pool = [c[0] for s in sessions for c in s['tpose'].values() if c]
            if pool:
                print("[Calib] 警告: mocap で確認できた T ポーズが無いため、各収録の最初 / 最後の静止区間を"
                      "基準の計算に使います (腕を下ろした静止が混ざっていないか calib/train.json で確認してください)")
        if not pool:
            print("[Calib] 警告: T ポーズが 1 つも見つからないため、IMU の取り付け向きの補正と"
                  "圧力の体重基準化は行いません (calibration.tpose の設定を確認してください)")
            return None
        return {p: calib.mean_direction([w['gravity'][p] for w in pool]) for p in calib.PARTS}

    def _apply_tpose_calib(self, sessions):
        """
        T ポーズの採否を決め、IMU の取り付け向きの補正と圧力の体重基準化を sessions にその場で適用する。
          1. 基準の重力方向 ref (部位ごと) を決める
          2. 各収録の T ポーズ候補を判定し (mocap で確認済みならその結果、未確認なら ref との角度)、
             start / end それぞれで最も長いものを選ぶ。prefer の側を優先して使う
          3. T ポーズが無い収録は被験者の平均 (学習時は他の収録、テスト時は stats) で代用する
          4. 重力方向を ref に移す最小回転を Gyro と Acc に掛け、圧力を T ポーズの合計で割る
        """
        cc, tc = self.calib, self.calib['tpose']
        ref = self._tpose_reference(sessions)
        order = ('start', 'end') if tc['prefer'] == 'start' else ('end', 'start')

        for s in sessions:
            best = {}
            for region in ('start', 'end'):
                accepted = []
                for w in s['tpose'][region]:
                    w['angle_to_ref'] = ({p: calib.angle_deg(w['gravity'][p], ref[p]) for p in calib.PARTS}
                                         if ref else None)
                    if w['verified'] is not None:
                        w['accepted'] = bool(w['verified'])
                    else:
                        w['accepted'] = ref is not None and max(w['angle_to_ref'].values()) <= tc['max_angle_deg']
                    if w['accepted']:
                        accepted.append(w)
                best[region] = max(accepted, key=lambda w: w['t'][1] - w['t'][0]) if accepted else None
            s['tpose_best'] = best
            s['tpose_used'] = next(((r, best[r]) for r in order if best[r] is not None), None)

        # 被験者ごとの平均 (T ポーズが無い収録の代用と、テスト時のために stats に保存する)
        fallback = {}
        for key in sorted({s['key'] for s in sessions}):
            st = self.id_stats.get(key) if self._stats_given else None
            if st is not None and 'tpose_gravity' in st:
                g = st.get('tpose_gravity')
                fallback[key] = {
                    'gravity': {p: np.asarray(v, dtype=np.float64) for p, v in g.items()} if g else None,
                    'press': st.get('tpose_press'),
                    'source': '学習データの平均 (stats)'}
            else:
                used = [s['tpose_used'][1] for s in sessions if s['key'] == key and s['tpose_used']]
                fallback[key] = {
                    'gravity': ({p: calib.mean_direction([w['gravity'][p] for w in used]) for p in calib.PARTS}
                                if used else None),
                    'press': ({f: float(np.mean([w['press'][f] for w in used])) for f in calib.FOOT_PARTS}
                              if used else None),
                    'source': f"他の収録の平均 ({len(used)} 収録)"}
            self._calib_stats[key] = {
                'calib_flags': calib.config_flags(cc),
                'tpose_gravity_ref': ref,
                'tpose_gravity': fallback[key]['gravity'],
                'tpose_press': fallback[key]['press'],
            }

        imu_order = ('AL', 'AR', 'FL', 'FR', 'BC')   # imu_combined の並び
        for s in sessions:
            fb = fallback[s['key']]
            if s['tpose_used']:
                region, w = s['tpose_used']
                gravity, press, source = w['gravity'], w['press'], f"T ポーズ ({region})"
            elif fb['gravity'] is not None:
                gravity, press, source = fb['gravity'], fb['press'], f"T ポーズ無し → {fb['source']}"
            else:
                gravity, press, source = None, None, 'T ポーズ無し → 補正しない'

            align_deg = {}
            if cc['imu_align'] and gravity is not None and ref is not None:
                for i, p in enumerate(imu_order):
                    R = calib.rotation_between(gravity[p], ref[p])
                    s['imu'][:, i, 0:3] = s['imu'][:, i, 0:3] @ R.T   # Gyro
                    s['imu'][:, i, 3:6] = s['imu'][:, i, 3:6] @ R.T   # Acc
                    align_deg[p] = round(calib.angle_deg(gravity[p], ref[p]), 2)

            scale = None
            if cc['pressure_scale'] != 'none' and press:
                if cc['pressure_scale'] == 'total' and press['FL'] + press['FR'] > 0:
                    # 左右の合計 (≒ 体重) を 100 とする
                    scale = {f: 100.0 / (press['FL'] + press['FR']) for f in calib.FOOT_PARTS}
                elif cc['pressure_scale'] == 'per_foot' and min(press.values()) > 0:
                    scale = {f: 50.0 / press[f] for f in calib.FOOT_PARTS}
                if scale:
                    s['lp'] = s['lp'] * scale['FL']
                    s['rp'] = s['rp'] * scale['FR']

            slip = None
            if s['tpose_best']['start'] and s['tpose_best']['end']:
                # 最初と最後の T ポーズの重力方向の差 (バンドのずれ・姿勢の違いの目安)
                slip = {p: round(calib.angle_deg(s['tpose_best']['start']['gravity'][p],
                                                 s['tpose_best']['end']['gravity'][p]), 2)
                        for p in calib.PARTS}

            def _cand_info(w):
                return {'t_sec': [round(w['t'][0], 2), round(w['t'][1], 2)],
                        'mocap_ratio': None if w['mocap_ratio'] is None else round(w['mocap_ratio'], 3),
                        'accepted': w['accepted'],
                        'angle_to_ref_deg': ({p: round(v, 2) for p, v in w['angle_to_ref'].items()}
                                             if w['angle_to_ref'] else None),
                        'press_sum': {f: round(v, 1) for f, v in w['press'].items()}}

            info = self.calib_info.setdefault(s['data_id'], {})
            info.update({
                'source': source,
                'tpose_candidates': {r: [_cand_info(w) for w in s['tpose'][r]] for r in ('start', 'end')},
                'imu_align_deg': align_deg or None,
                'pressure_scale': {f: round(v, 6) for f, v in scale.items()} if scale else None,
                'start_end_diff_deg': slip,
            })

            msg = f"[Calib] {s['data_id']}: {source}"
            if s['tpose_used']:
                w = s['tpose_used'][1]
                msg += f" {w['t'][0]:.2f}〜{w['t'][1]:.2f}s"
                if w['mocap_ratio'] is not None:
                    msg += f" (mocap で確認 {w['mocap_ratio']:.0%})"
            print(msg)
            if align_deg:
                print("  取り付け向きの補正角: " + ", ".join(f"{p} {v:.1f}°" for p, v in align_deg.items()))
            if press:
                print(f"  T ポーズの圧力合計: 左 {press['FL']:.0f} / 右 {press['FR']:.0f}")
            if slip:
                print("  最初と最後の T ポーズの重力方向の差: " + ", ".join(f"{p} {v:.1f}°" for p, v in slip.items()))
            rejected = [(r, w) for r in ('start', 'end') for w in s['tpose'][r] if not w['accepted']]
            for r, w in rejected:
                why = (f"mocap で T ポーズと判定されたのは {w['mocap_ratio']:.0%}" if w['mocap_ratio'] is not None
                       else (f"基準との角度差 最大 {max(w['angle_to_ref'].values()):.0f}°" if w['angle_to_ref']
                             else '基準なし'))
                print(f"  不採用の静止区間 ({r}) {w['t'][0]:.2f}〜{w['t'][1]:.2f}s: {why}")

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
                   num_workers=0, stats=None, sync_cfg=None, target='skeleton', smpl_cfg=None,
                   calib_cfg=None, ids=None, contact_source='pressure', canonical_heading=True,
                   floor_normalize=True):
    dataset = InsoleIMUDataset(data_path, seq_len=seq_len, mode=mode, stats=stats,
                               sync_cfg=sync_cfg, target=target, smpl_cfg=smpl_cfg,
                               calib_cfg=calib_cfg, ids=ids, contact_source=contact_source,
                               canonical_heading=canonical_heading, floor_normalize=floor_normalize)
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
                            smpl_cfg=cfg.get('smpl'), calib_cfg=cfg.get('calibration'),
                            ids=cfg['dataset'].get('train_ids'),
                            contact_source=cfg['dataset'].get('contact_source', 'pressure'),
                            canonical_heading=cfg['dataset'].get('canonical_heading', True),
                            floor_normalize=cfg['dataset'].get('floor_normalize', True))
    print(f"Dataset size: {len(loader.dataset)} sequences")
    print(f"Stats keys  : {list(loader.dataset.id_stats.keys())}")
    print(f"Sync        : { {k: round(v['offset_sec'], 2) for k, v in loader.dataset.sync_info.items()} }")
    batch = next(iter(loader))
    for k, v in batch.items():
        print(f"{k:15s}: {tuple(v.shape) if hasattr(v, 'shape') else v}")
