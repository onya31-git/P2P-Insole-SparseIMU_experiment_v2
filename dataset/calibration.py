"""
机置きデータ (A0) と T ポーズを使ったセンサーのキャリブレーション。

収録の手順 (前提):
  - データ収集の直前に 1 回、全センサーを机に置いて静止させた記録 (A0) を取る
      → ジャイロのバイアス / 圧力センサーの無負荷時のゼロ点
  - 各収録の最初と最後に T ポーズを 3 秒キープする
      → (1) IMU の取り付け向きの補正: T ポーズ中の重力方向 (センサー座標系) を基準の向きにそろえる
        (2) 圧力の体重基準化: T ポーズ (両足で静止して立つ) の圧力の合計で割る
  地磁気 (Mag) は信頼度が低いため使わない。そのため取り付け向きの補正は重力に対する傾き (2 自由度) だけで、
  重力軸まわりの向き (yaw) は補正しない。

T ポーズの検出 (センサーのみで行う。テスト時・実運用時と同じ条件にするため):
  収録の最初 / 最後の search_sec 秒から「全 IMU の角速度が小さく、加速度の大きさが重力に近く、
  両足に荷重がある」区間 (静止区間) を探す。腕を下ろして静止している区間も同じ条件を満たすため、
  静止区間が T ポーズかどうかは次の方法で確かめる:
    - 学習時: mocap の腕の向き (肩 → 手首がほぼ水平で、両手首が十分に離れている) で確認する
    - テスト時 (または mocap で確認できない区間): 重力方向が基準 (学習データの T ポーズの平均) から
      max_angle_deg 以内か。腕を下ろすと手首センサーの重力方向は約 70° 変わるので区別できる
  条件を満たす区間のうち最も長いものを使う。
"""
import numpy as np


# キャリブレーションで扱う部位 (IMU の並び順 AL, AR, FL, FR, BC とは異なるので、部位名の辞書で受け渡す)
PARTS = ('AL', 'AR', 'BC', 'FL', 'FR')
FOOT_PARTS = ('FL', 'FR')
# mocap の T ポーズ判定に使う (肩, 手首) の関節インデックス (dataset.JOINT_NAMES の順序)
ARM_JOINTS = ((4, 6), (7, 9))
SKEL_FPS = 100.0

DEFAULTS = {
    'enabled': True,
    'gyro_bias': True,          # A0 の平均角速度をジャイロのバイアスとして引く
    'pressure_zero': True,      # A0 の圧力 (無負荷) をセルごとのゼロ点として引く
    'imu_align': True,          # T ポーズの重力方向で IMU の取り付け向きを補正する
    'pressure_scale': 'total',  # total: 左右合計 (体重) で割る / per_foot: 左右それぞれで割る / none
    'tpose': {
        'search_sec': 10.0,     # 収録の最初 / 最後の何秒から T ポーズを探すか
        'min_sec': 0.5,         # 静止区間の最短 [s]
        'trim_sec': 0.1,        # 静止区間の両端から除く時間 [s]
        'smooth_sec': 0.2,      # 静止判定の移動平均の幅 [s]
        'gyro_thresh': 20.0,    # 角速度のノルムの上限 [deg/s] (T ポーズ中も腕は 10 deg/s 程度揺れる)
        'acc_tol': 0.1,         # 加速度の大きさと重力の差の上限 (重力に対する比)
        'foot_load_ratio': 0.3,  # 各足の圧力合計がその収録の中央値の何倍以上なら荷重ありとするか
        'prefer': 'start',      # start: 最初の T ポーズを優先 (無ければ最後) / end: その逆
        'max_angle_deg': 45.0,  # 重力方向の基準との差の上限 [deg] (mocap で確認できないとき)
        'mocap_check': True,    # 学習時に mocap で T ポーズかどうかを確認する
        'mocap_arm_elev_deg': 30.0,  # 肩 → 手首の水平からの角度の上限 [deg]
        'mocap_min_ratio': 0.8,      # 区間内で T ポーズと判定されたフレームの割合の下限
    },
}


def merge_config(cfg):
    """config.yaml の calibration セクションに既定値を補う。"""
    cfg = cfg or {}
    out = {k: v for k, v in DEFAULTS.items() if k != 'tpose'}
    out.update({k: v for k, v in cfg.items() if k != 'tpose'})
    out['tpose'] = dict(DEFAULTS['tpose'], **(cfg.get('tpose') or {}))
    if out['pressure_scale'] not in ('total', 'per_foot', 'none'):
        raise ValueError(f"calibration.pressure_scale は total / per_foot / none です: {out['pressure_scale']}")
    if out['tpose']['prefer'] not in ('start', 'end'):
        raise ValueError(f"calibration.tpose.prefer は start / end です: {out['tpose']['prefer']}")
    return out


def config_flags(cfg):
    """学習時とテスト時で前処理が一致しているかを確かめるための文字列。"""
    if not cfg['enabled']:
        return 'disabled'
    return (f"gyro_bias={int(cfg['gyro_bias'])},pressure_zero={int(cfg['pressure_zero'])},"
            f"imu_align={int(cfg['imu_align'])},pressure_scale={cfg['pressure_scale']}")


# ----------------------------------------------------------------------
# 机置きデータ (A0)
# ----------------------------------------------------------------------
def static_calibration(raw, trim_ratio=0.1):
    """
    机置きの記録から部位ごとのジャイロのバイアスと圧力のゼロ点を求める。

    raw: {part: (t_ms [N], imu [N, 6] (Gyro 3 + Acc 3), press [N, 112] or None)}
    戻り値: {part: {gyro_bias [3], gyro_p95 (バイアスを引いた後の角速度の 95 パーセンタイル),
                    press_zero [112] or None}}
    センサーを置いた直後・持ち上げる直前の揺れを避けるため、前後 trim_ratio を除いて中央値を取る。
    """
    out = {}
    for part, (_, imu, press) in raw.items():
        n = len(imu)
        k = int(n * trim_ratio)
        sl = slice(k, n - k) if n - 2 * k >= 10 else slice(None)
        gyro = imu[sl, :3]
        bias = np.median(gyro, axis=0)
        out[part] = {
            'gyro_bias': bias,
            'gyro_p95': float(np.percentile(np.linalg.norm(gyro - bias, axis=1), 95)),
            'press_zero': np.median(press[sl], axis=0) if press is not None else None,
        }
    return out


# ----------------------------------------------------------------------
# T ポーズの検出
# ----------------------------------------------------------------------
def _moving_average(x, k):
    if k <= 1:
        return x
    xp = np.pad(x, (k // 2, k - 1 - k // 2), mode='edge')
    return np.convolve(xp, np.ones(k) / k, mode='valid')


def find_static_runs(sens_t, raw, t_range, tcfg, fps=SKEL_FPS):
    """
    t_range (センサー時刻 [s]) の中の静止区間を [(t0, t1), ...] (時刻順) で返す。
    sens_t: {part: 時刻 [s]}, raw: {part: (t_ms, imu, press)}
    """
    a, b = t_range
    grid = np.arange(a, b, 1.0 / fps)
    if len(grid) < 2:
        return []
    k = max(1, int(round(tcfg['smooth_sec'] * fps)))
    static = np.ones(len(grid), dtype=bool)
    for part in PARTS:
        t, (_, imu, press) = sens_t[part], raw[part]
        gyro = np.interp(grid, t, np.linalg.norm(imu[:, :3], axis=1))
        acc_norm = np.linalg.norm(imu[:, 3:], axis=1)
        g = np.median(acc_norm)   # 重力の大きさ (加速度の単位に依存しないよう収録全体の中央値を使う)
        acc_dev = np.interp(grid, t, np.abs(acc_norm - g)) / g
        static &= _moving_average(gyro, k) < tcfg['gyro_thresh']
        static &= _moving_average(acc_dev, k) < tcfg['acc_tol']
        if press is not None:
            total = press.sum(axis=1)
            static &= np.interp(grid, t, total) > tcfg['foot_load_ratio'] * np.median(total)

    runs = []
    edges = np.diff(np.concatenate([[0], static.astype(int), [0]]))
    for i0, i1 in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
        t0, t1 = grid[i0] + tcfg['trim_sec'], grid[i1 - 1] - tcfg['trim_sec']
        if t1 - t0 >= tcfg['min_sec']:
            runs.append((float(t0), float(t1)))
    return runs


def window_stats(sens_t, raw, t0, t1):
    """区間 [t0, t1] の各 IMU の重力方向 (単位ベクトル) と各足の圧力の合計の平均。"""
    gravity, press = {}, {}
    for part in PARTS:
        t, (_, imu, p) = sens_t[part], raw[part]
        m = (t >= t0) & (t <= t1)
        if m.sum() < 3:
            return None
        acc = imu[m, 3:].mean(axis=0)
        gravity[part] = acc / np.linalg.norm(acc)
        if p is not None:
            press[part] = float(p[m].sum(axis=1).mean())
    return {'t': (t0, t1), 'gravity': gravity, 'press': press}


def mocap_tpose_ratio(skel_t, pos, offset, t0, t1, tcfg, span_ref):
    """
    センサー時刻の区間 [t0, t1] のうち、mocap で T ポーズと判定されるフレームの割合。
    mocap の記録が区間の半分未満しか無いときは確認できないので None を返す。

    T ポーズの条件: 左右とも肩 → 手首が水平から mocap_arm_elev_deg 以内、かつ
                    両手首の距離がその収録の 99 パーセンタイル (span_ref) の 85% 以上
    pos: [T, J, 3] mm (y が上), センサー時刻 = スケルトン時刻 + offset
    """
    ts = skel_t + offset
    m = (ts >= t0) & (ts <= t1)
    if m.sum() < 0.5 * (t1 - t0) * SKEL_FPS:
        return None
    p = pos[m]
    ok = np.ones(len(p), dtype=bool)
    for sh, wr in ARM_JOINTS:
        v = p[:, wr] - p[:, sh]
        elev = np.degrees(np.arcsin(np.clip(np.abs(v[:, 1]) / (np.linalg.norm(v, axis=1) + 1e-9), 0, 1)))
        ok &= elev < tcfg['mocap_arm_elev_deg']
    ok &= np.linalg.norm(p[:, 6] - p[:, 9], axis=1) > 0.85 * span_ref
    return float(ok.mean())


# ----------------------------------------------------------------------
# 回転
# ----------------------------------------------------------------------
def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(unit(a), unit(b)), -1.0, 1.0))))


def rotation_between(a, b):
    """
    単位ベクトル a を b に移す最小の回転行列 (回転軸は a × b)。

    T ポーズでは前腕などの体節の長軸が水平 (重力に垂直) なので、バンドが体節の長軸まわりに
    ずれた場合、その前後の重力方向の差を埋める最小回転の軸は長軸に一致し、ずれを正確に戻せる。
    """
    a, b = unit(a), unit(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1e-8:
        if c > 0:
            return np.eye(3)
        axis = unit(np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0]))
        return 2.0 * np.outer(axis, axis) - np.eye(3)   # 反平行: a に垂直な軸まわりに 180°
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1.0 - c) / s ** 2)


def mean_direction(vectors):
    """単位ベクトルの平均方向。"""
    return unit(np.mean([unit(v) for v in vectors], axis=0))
