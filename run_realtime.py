"""
センサーから受信したデータで全身の姿勢をリアルタイムに推定し、ブラウザに表示する。

使い方:
    uv run run_realtime.py                          # センサーサーバーから受信して推定 (設定は realtime.yaml)
    uv run run_realtime.py --replay D2S1T1          # 収録済みの CSV を実時間で再生して推定 (デバイス無しの動作確認)
    uv run run_realtime.py --save --record          # 推定結果 (npz) と受信データ (CSV) も保存する

手順 (センサーから受信するとき):
    1. 全デバイスの電源を入れ、装着する (static_calib: live のときは先に机に置く)
    2. 起動すると全デバイスの受信を待ち、続けて T ポーズの検出を始める → T ポーズを 3 秒以上キープ
    3. 推定が始まる。バンドがずれたら画面のボタン (またはコンソールで Enter) で T ポーズをやり直す
    Ctrl+C で終了する。

認証: 環境変数 NH_USERNAME / NH_PASSWORD (無ければ起動時に入力する)。

学習済みモデル・正規化の統計量・キャリブレーションの設定は config.yaml (オフラインと共通) から読む。
オフラインのパイプラインのファイルは変更しない。

出力 (--save / --record のとき, <output_dir>/realtime/live/<開始時刻>/):
    predictions.npz   poses [N, 72] (軸角, AMASS と同じ), trans [N, 3], contact [N, 2], time [N] (センサー時刻 [s]),
                      betas, gender, mocap_framerate
    raw/<部位>.csv    受信したデータ (data/sensor/ の CSV と同じ列)
"""
import os
import sys
import time
import queue
import getpass
import argparse
import threading
from collections import deque
from datetime import datetime

import numpy as np
import torch
import yaml

from utils.paths import load_config, get_paths, resolve, DEFAULT_CONFIG
from realtime.sources import WebSocketSource, ReplaySource, RawRecorder, drain
from realtime.pipeline import RealtimePipeline

DEFAULT_RT_CONFIG = resolve('realtime.yaml')


class Log:
    """コンソールに出しつつ、画面に表示するために直近の行を残す。"""
    def __init__(self, n=40):
        self.lines = deque(maxlen=n)
        self.lock = threading.Lock()

    def __call__(self, msg):
        print(msg, flush=True)
        with self.lock:
            self.lines.append(f"{datetime.now():%H:%M:%S} {msg}")

    def text(self):
        with self.lock:
            return "\n".join(self.lines)


def credentials(rt):
    srv = rt.get('server') or {}
    user = os.environ.get('NH_USERNAME') or srv.get('username') or input('センサーサーバーのユーザー名: ')
    pw = os.environ.get('NH_PASSWORD') or srv.get('password') or getpass.getpass('パスワード: ')
    return user, pw


def main():
    parser = argparse.ArgumentParser(description='リアルタイム推定')
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--rt-config', default=DEFAULT_RT_CONFIG, help='リアルタイム推定の設定 (既定 realtime.yaml)')
    parser.add_argument('--replay', default=None, help='受信の代わりに再生する収録 ID (例 D2S1T1)')
    parser.add_argument('--speed', type=float, default=1.0, help='再生の速さ (倍)')
    parser.add_argument('--start-sec', type=float, default=0.0, help='再生を始める位置 [s]')
    parser.add_argument('--duration', type=float, default=None, help='この秒数で終了する')
    parser.add_argument('--save', action='store_true', help='推定結果を npz に保存する')
    parser.add_argument('--record', action='store_true', help='受信データを CSV に保存する')
    parser.add_argument('--no-viewer', action='store_true', help='ブラウザ表示を使わない')
    parser.add_argument('--no-browser', action='store_true', help='ブラウザを自動で開かない')
    parser.add_argument('--port', type=int, default=None)
    parser.add_argument('--device', default=None, help='cuda / cpu (既定は realtime.yaml の inference.device)')
    args = parser.parse_args()

    config = load_config(args.config)
    with open(resolve(args.rt_config), encoding='utf-8') as f:
        rt = yaml.safe_load(f) or {}
    paths = get_paths(config)
    log = Log()

    dev_name = args.device or (rt.get('inference') or {}).get('device', 'cuda')
    if dev_name == 'cuda' and not torch.cuda.is_available():
        log("[main] CUDA が使えないため CPU で推論します (推論が遅れる場合は realtime.yaml の inference.hop を大きくする)")
        dev_name = 'cpu'
    pipeline = RealtimePipeline(config, rt, torch.device(dev_name), log)

    q = queue.Queue()
    if args.replay:
        source = ReplaySource(str(paths.data_dir), args.replay, q, speed=args.speed, start_sec=args.start_sec, log=log)
    else:
        user, pw = credentials(rt)
        source = WebSocketSource(rt['server'], rt['devices'], user, pw, q, log=log)

    viewer = None
    if not args.no_viewer:
        from realtime.viewer import ViewerServer
        vc = rt.get('viewer') or {}
        viewer = ViewerServer(vc.get('host', '127.0.0.1'), args.port or int(vc.get('port', 8766)),
                              pipeline.viewer_model(), on_command=lambda c: q.put(('cmd', c)), log=log)
        viewer.start(open_browser=not args.no_browser)

    def stdin_loop():   # コンソールで Enter → T ポーズをやり直す
        try:
            for _ in sys.stdin:
                q.put(('cmd', 'tpose'))
        except Exception:
            pass
    if sys.stdin is not None and sys.stdin.isatty():
        threading.Thread(target=stdin_loop, daemon=True).start()

    out_dir = paths.output_dir / 'realtime' / 'live' / datetime.now().strftime('%Y%m%d_%H%M%S')
    recorder = RawRecorder(out_dir / 'raw') if args.record else None
    saved = {k: [] for k in ('poses', 'trans', 'contact', 'time')}
    lat = deque(maxlen=200)

    def handle(out):
        now_ms = source.now_ms()
        lat.append(now_ms / 1000.0 - out['t'][-1])
        if viewer is not None:
            viewer.publish('frames', {
                'k': out['k'], 't': out['t'], 'now_ms': now_ms,
                'A': np.round(out['A'], 4).reshape(-1).tolist(),
                'joints': np.round(out['joints'], 4).tolist(),
                'contact': np.round(out['contact'], 3).tolist(),
                'pressure': {f: np.round(v, 2).tolist() for f, v in out['pressure'].items()},
            })
        if args.save:
            saved['poses'].append(out['poses'])
            saved['trans'].append(out['trans'])
            saved['contact'].append(out['contact'])
            saved['time'].append(np.asarray(out['t']))

    log("[main] 開始します (Ctrl+C で終了)")
    source.start()
    t_start = time.monotonic()
    last_status = last_print = 0.0
    try:
        while True:
            try:
                items = [q.get(timeout=0.05)] + drain(q)
            except queue.Empty:
                items = []
            for it in items:
                if isinstance(it, tuple):
                    if it[1] == 'tpose':
                        pipeline.request_tpose()
                    continue
                if recorder is not None:
                    recorder.write(it)
                for out in pipeline.push(it):
                    handle(out)

            now = time.monotonic()
            if now - last_status > 0.5:
                last_status = now
                st = pipeline.status()
                st['devices'] = source.stats.snapshot()
                st['log'] = log.text()
                if viewer is not None:
                    viewer.publish('status', st)
                if now - last_print > 5.0:
                    last_print = now
                    rates = ", ".join(f"{p} {d['rate_hz']:.0f}" for p, d in st['devices'].items())
                    extra = ''
                    if st['state'] == 'run' and lat:
                        extra = (f", 遅延 {np.median(lat):.2f} s (センサー → 出力), "
                                 f"推論 {st['infer_ms'] or 0:.0f} ms")
                    print(f"[status] {st['state']}: {st['message']} | 受信 [Hz] {rates}{extra}", flush=True)
            if source.finished.is_set() and q.empty():
                log("[main] 入力が終わったので終了します")
                break
            if args.duration and now - t_start > args.duration:
                log(f"[main] {args.duration:g} 秒経過したので終了します")
                break
    except KeyboardInterrupt:
        log("[main] 停止しました")
    finally:
        source.stop()
        if recorder is not None:
            recorder.close()
            log(f"[main] 受信データを保存しました: {out_dir / 'raw'}")
        if args.save and saved['poses']:
            out_dir.mkdir(parents=True, exist_ok=True)
            res = {k: np.concatenate(v).astype(np.float32 if k != 'time' else np.float64) for k, v in saved.items()}
            np.savez_compressed(out_dir / 'predictions.npz', **res,
                                betas=pipeline.betas[0].cpu().numpy(), gender=pipeline.gender,
                                mocap_framerate=np.array(100.0))
            log(f"[main] 推定結果を保存しました: {out_dir / 'predictions.npz'} ({len(res['poses'])} フレーム)")
        if viewer is not None:
            viewer.stop()


if __name__ == '__main__':
    main()
