"""
センサーデータの受信元。どちらも受け取ったデータを Sample にして queue に入れるスレッド。

    WebSocketSource : センサーサーバー (NHO プロトコル) からリアルタイムに受信する。
                      API は get_IMU_v4.py と同じ (POST /api/token でトークン → /stream/ws に接続して
                      デバイスごとに subscribe → "update" イベントを受け取る)
    ReplaySource    : 収録済みの CSV (data/sensor/*/<ID>_<部位>.csv) を実時間で再生する
                      (デバイスが無いときの動作確認用)

"update" イベントの data のうち使うもの (2026-11-19 にサーバーの snapshot で確認):
    timestamp_ms   デバイスのエポックミリ秒 (学習用 CSV の timestamp_ms と同じ時計)
    imu.gyro [3]   角速度 [deg/s]、imu.acc [3] 加速度 [g]、imu.mag [3] (推定には使わない)
    p [112]        圧力 8×14 (行優先, CSV の P1..P112 と同じ順序)。インソール (FL / FR) のみ使う
"""
import csv
import json
import time
import queue
import threading
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PARTS = ('AL', 'AR', 'BC', 'FL', 'FR')
FOOT_PARTS = ('FL', 'FR')
NUM_PRESSURE = 112


@dataclass
class Sample:
    part: str
    t_ms: float                      # デバイスのエポックミリ秒
    imu: np.ndarray                  # [6] Gyro (deg/s) 3 + Acc (g) 3
    press: np.ndarray | None         # [112] (FL / FR のみ)
    mag: np.ndarray | None = None
    seq: int | None = None


class DeviceStats:
    """部位ごとの受信状況 (直近 2 秒の受信レート・最後に受信してからの時間)。"""
    def __init__(self, parts):
        self.lock = threading.Lock()
        self.count = {p: 0 for p in parts}
        self.recent = {p: deque(maxlen=400) for p in parts}

    def update(self, part):
        with self.lock:
            self.count[part] += 1
            self.recent[part].append(time.monotonic())

    def snapshot(self):
        now = time.monotonic()
        out = {}
        with self.lock:
            for p, r in self.recent.items():
                out[p] = {
                    'count': self.count[p],
                    'rate_hz': sum(1 for x in r if now - x < 2.0) / 2.0,
                    'age_s': (now - r[-1]) if r else None,
                }
        return out


class SourceBase(threading.Thread):
    def __init__(self, out_queue, log=print):
        super().__init__(daemon=True)
        self.out = out_queue
        self.log = log
        self.stats = DeviceStats(PARTS)
        self.stop_event = threading.Event()
        self.finished = threading.Event()   # 再生の終了 (WebSocket では使わない)
        self._warned = set()

    def emit(self, sample):
        self.stats.update(sample.part)
        self.out.put(sample)

    def warn_once(self, key, msg):
        if key not in self._warned:
            self._warned.add(key)
            self.log(msg)

    def stop(self):
        self.stop_event.set()

    def now_ms(self):
        """センサーの時計で見た現在時刻 [ms] (遅延の計算に使う)。"""
        return time.time() * 1000.0


# ----------------------------------------------------------------------
# WebSocket (センサーサーバー)
# ----------------------------------------------------------------------
class WebSocketSource(SourceBase):
    def __init__(self, server_cfg, devices, username, password, out_queue, log=print):
        super().__init__(out_queue, log)
        base = server_cfg['base_url'].rstrip('/')
        prefix = server_cfg.get('app_prefix', '')
        self.api_url = base + prefix
        self.ws_url = base.replace('https://', 'wss://').replace('http://', 'ws://') + prefix + '/stream/ws'
        self.devices = {p: str(u).upper() for p, u in devices.items()}
        self.uid_to_part = {u: p for p, u in self.devices.items()}
        self.username, self.password = username, password
        self.timeout = float(server_cfg.get('timeout_sec', 10.0))

    def run(self):
        backoff = 1.0
        while not self.stop_event.is_set():
            try:
                self._session()
                backoff = 1.0
            except Exception as e:   # 切断・ネットワークエラーは再接続する
                if self.stop_event.is_set():
                    break
                self.log(f"[source] 接続エラー: {type(e).__name__}: {e} → {backoff:.0f} 秒後に再接続します")
                self.stop_event.wait(backoff)
                backoff = min(backoff * 2, 30.0)

    def _token(self):
        body = json.dumps({'username': self.username, 'password': self.password}).encode()
        req = urllib.request.Request(f"{self.api_url}/api/token", data=body, method='POST',
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)['token']

    def _session(self):
        import websocket   # websocket-client
        token = self._token()
        ws = websocket.create_connection(self.ws_url, header=[f"Authorization: Bearer {token}"],
                                         timeout=self.timeout)
        try:
            ws.settimeout(1.0)
            hello = json.loads(ws.recv())
            self.log(f"[source] サーバーに接続しました ({hello.get('type')}, {self.ws_url})")
            for part, uid in self.devices.items():
                ws.send(json.dumps({'type': 'subscribe', 'device_uid': uid}, separators=(',', ':')))
            while not self.stop_event.is_set():
                try:
                    raw = ws.recv()
                except websocket.WebSocketTimeoutException:
                    continue
                ev = json.loads(raw)
                typ = ev.get('type')
                if typ == 'update':
                    self._handle(ev)
                elif typ == 'subscribed':
                    uid = ev.get('device_uid')
                    self.log(f"[source] 購読しました: {self.uid_to_part.get(uid, '?')} = {uid}")
                elif typ == 'snapshot':
                    pass   # 購読時に送られる最後のデータ (古いことがある) は使わない
                else:
                    self.warn_once(('type', typ), f"[source] 受信メッセージ: {str(ev)[:200]}")
        finally:
            try:
                ws.close()
            except Exception:
                pass

    def _handle(self, ev):
        uid = str(ev.get('device_uid', '')).upper()
        part = self.uid_to_part.get(uid)
        if part is None:
            return
        d = ev.get('data') or {}
        imu = d.get('imu') or d
        acc, gyro, t = imu.get('acc'), imu.get('gyro'), d.get('timestamp_ms')
        if acc is None or gyro is None or t is None or len(acc) < 3 or len(gyro) < 3:
            self.warn_once(('imu', part), f"[source] {part} ({uid}) の IMU データ / timestamp_ms がありません: "
                                          f"{list(d.keys())}")
            return
        press = None
        if part in FOOT_PARTS:
            p = d.get('p')
            if p is None or len(p) < NUM_PRESSURE:
                self.warn_once(('p', part), f"[source] {part} ({uid}) の圧力 (data.p, {NUM_PRESSURE} 点) がありません")
                return
            press = np.nan_to_num(np.array(p[:NUM_PRESSURE], dtype=np.float64))
        mag = imu.get('mag')
        self.emit(Sample(part, float(t),
                         np.nan_to_num(np.array([*gyro[:3], *acc[:3]], dtype=np.float64)), press,
                         None if mag is None else np.array(mag[:3], dtype=np.float64), d.get('frame_id')))


# ----------------------------------------------------------------------
# 収録済み CSV の再生
# ----------------------------------------------------------------------
def find_sensor_csvs(data_dir, data_id):
    """data/sensor/*/<ID>_<部位>.csv を探す {part: path}。"""
    from dataset.dataset import SENSOR_RE
    files = {}
    for path in sorted(Path(data_dir, 'sensor').glob('*/*.csv')):
        m = SENSOR_RE.match(path.name)
        if m and m.group(1).upper() == data_id.upper():
            files[m.group(2).upper()] = path
    missing = [p for p in PARTS if p not in files]
    if missing:
        raise FileNotFoundError(f"{data_id} のセンサー CSV が見つかりません: {missing} ({Path(data_dir, 'sensor')})")
    return files


class ReplaySource(SourceBase):
    """収録済みの CSV を speed 倍速で再生する (タイムスタンプは CSV の値をそのまま使う)。"""
    def __init__(self, data_dir, data_id, out_queue, speed=1.0, start_sec=0.0, log=print):
        super().__init__(out_queue, log)
        from dataset.dataset import InsoleIMUDataset
        self.speed = float(speed)
        ts, parts, rows = [], [], []
        self.data = {}
        for i, (part, path) in enumerate(find_sensor_csvs(data_dir, data_id).items()):
            t, imu, press = InsoleIMUDataset._read_sensor_csv(path, with_pressure=part in FOOT_PARTS)
            self.data[part] = (imu, press)
            ts.append(t)
            parts.append(np.full(len(t), i))
            rows.append(np.arange(len(t)))
        self.part_names = list(self.data)
        t = np.concatenate(ts)
        order = np.argsort(t, kind='stable')
        self.t, self.p, self.r = t[order], np.concatenate(parts)[order], np.concatenate(rows)[order]
        if start_sec > 0:
            keep = self.t >= self.t[0] + start_sec * 1000
            self.t, self.p, self.r = self.t[keep], self.p[keep], self.r[keep]
        self.t0 = float(self.t[0])
        self.wall0 = None
        self.log(f"[source] {data_id} を再生します ({(self.t[-1] - self.t[0]) / 1000:.0f} 秒, {self.speed:g} 倍速)")

    def samples(self):
        """全サンプルを時刻順に返す (待たない。テスト用)。"""
        for t, p, r in zip(self.t, self.p, self.r):
            part = self.part_names[p]
            imu, press = self.data[part]
            yield Sample(part, float(t), imu[r], None if press is None else press[r])

    def run(self):
        self.wall0 = time.monotonic()
        for s in self.samples():
            if self.stop_event.is_set():
                break
            wait = (s.t_ms - self.t0) / 1000.0 / self.speed - (time.monotonic() - self.wall0)
            if wait > 0.002:
                self.stop_event.wait(wait)
            self.emit(s)
        if not self.stop_event.is_set():
            self.log("[source] 再生が終わりました")
        self.finished.set()

    def now_ms(self):
        if self.wall0 is None:
            return self.t0
        return self.t0 + (time.monotonic() - self.wall0) * 1000.0 * self.speed


# ----------------------------------------------------------------------
# 受信データの記録 (学習用 CSV と同じ列)
# ----------------------------------------------------------------------
class RawRecorder:
    """受信した Sample を部位ごとの CSV に書く (data/sensor/ の CSV と同じ列。後でオフラインの評価に使える)。"""
    def __init__(self, out_dir):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.files, self.writers = {}, {}

    def write(self, s):
        if s.part not in self.writers:
            f = open(self.dir / f"{s.part}.csv", 'w', newline='', encoding='utf-8')
            n = NUM_PRESSURE if s.part in FOOT_PARTS else 1
            w = csv.writer(f)
            w.writerow(['timestamp_ms'] + [f'P{i}' for i in range(1, n + 1)]
                       + ['Mag_x', 'Mag_y', 'Mag_z', 'Gyro_x', 'Gyro_y', 'Gyro_z', 'Acc_x', 'Acc_y', 'Acc_z',
                          'frame_seq', 'time_ref'])
            self.files[s.part], self.writers[s.part] = f, w
        p = list(s.press) if s.press is not None else [0.0]
        mag = list(s.mag) if s.mag is not None else [0.0, 0.0, 0.0]
        self.writers[s.part].writerow([int(s.t_ms)] + p + mag + list(s.imu) + [s.seq, 'device'])

    def close(self):
        for f in self.files.values():
            f.close()


def drain(q, max_items=2000):
    """queue から取れるだけ取り出す。"""
    items = []
    try:
        while len(items) < max_items:
            items.append(q.get_nowait())
    except queue.Empty:
        pass
    return items
