"""
予測した SMPL メッシュをブラウザで確認する Web UI (dataset.target = smpl)。

使い方:
    uv run mesh_viewer.py                    # http://127.0.0.1:8765/ がブラウザで開く
    uv run mesh_viewer.py --port 8000 --no-browser
    uv run mesh_viewer.py --cpu              # GPU を使わない

入力: <output_dir>/results/smpl_predictions.npz (test.py の出力), body_models/ の SMPL
画面: mesh_viewer.html (このスクリプトがローカル HTTP サーバーとして配信する)

visualize.py --mesh と違い、メッシュは表示中のフレームの周辺だけをサーバー側 (GPU) で
その都度計算してブラウザに送るため、テストデータ全体 (数万フレーム) をそのまま再生できる。
    - 予測と正解のメッシュを重ねる / 並べる、頂点ごとの誤差をヒートマップで表示
    - 全フレームの MPJPE と接地 (予測・正解) のタイムライン (クリックで移動, ホイールで拡大)
    - 現在フレームの MPJPE / PVE / 腰の高さ / 接地確率 / 誤差の大きい関節
    - 誤差の大きい区間の一覧, スクリーンショット

API (ブラウザ側が使う):
    GET /api/meta        フレーム数・頂点数・関節の親子関係・被験者などの JSON
    GET /api/faces       メッシュの面 (uint32 [F, 3])
    GET /api/series      フレームごとの指標 (float32 [N, K], 列名は meta.series)
    GET /api/chunk?c=K   フレーム K*CHUNK 〜 (K+1)*CHUNK のメッシュと関節
                         (int16 [n, 2, V*3 + 24*3], mm。2 は 予測 / 正解)
"""
import json
import argparse
import threading
import webbrowser
from collections import OrderedDict
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

import numpy as np
import torch

from utils.paths import ROOT, load_config, get_paths, DEFAULT_CONFIG
from utils.smpl import SMPLBank, PARENTS, JOINT_NAMES, BODY_JOINTS, NUM_JOINTS

HTML_PATH = ROOT / 'mesh_viewer.html'
CHUNK = 100          # 1 回の要求で返すフレーム数 (100 Hz で 1 秒分, 約 8 MB)
CACHE_CHUNKS = 48    # サーバー側で保持する計算済みチャンク数

SERIES = ['mpjpe', 'root_err', 'pred_root_y', 'gt_root_y',
          'pred_contact_l', 'pred_contact_r', 'gt_contact_l', 'gt_contact_r']


def to_mm_int16(x):
    return np.clip(np.round(x * 1000.0), -32767, 32767).astype(np.int16)


class MeshSource:
    """smpl_predictions.npz を読み、要求されたフレームの SMPL メッシュを計算する。"""

    def __init__(self, config, device):
        self.paths = get_paths(config)
        src = self.paths.smpl_pred
        if not src.exists():
            raise SystemExit(f'{src} がありません。dataset.target: smpl で test.py を実行してください。')
        with np.load(src) as z:
            self.z = {k: z[k] for k in z.files}
        z = self.z
        self.N = len(z['pred_poses'])
        self.fps = float(z['mocap_framerate'])
        self.device = device
        self.bank = SMPLBank(config, device)
        self.lock = threading.Lock()   # GPU の計算とキャッシュは 1 スレッドずつ
        self.cache = OrderedDict()

        genders = [str(g) for g in np.unique(z['gender'])]
        self.faces = self.bank.faces(genders[0]).cpu().numpy().astype(np.uint32)
        for g in genders[1:]:   # SMPL の面は性別によらず共通のはずだが念のため確認する
            if not np.array_equal(self.faces, self.bank.faces(g).cpu().numpy()):
                raise SystemExit('性別ごとの SMPL モデルで面の定義が異なります')
        self.V = int(self.faces.max()) + 1
        self.series = self._series()

    def _series(self):
        z = self.z
        err = np.linalg.norm(z['pred_joints'][:, BODY_JOINTS] - z['gt_joints'][:, BODY_JOINTS], axis=-1)
        cols = {
            'mpjpe': err.mean(1) * 1000.0,
            'root_err': np.abs(z['pred_trans'][:, 1] - z['gt_trans'][:, 1]) * 1000.0,
            'pred_root_y': z['pred_trans'][:, 1] * 1000.0,
            'gt_root_y': z['gt_trans'][:, 1] * 1000.0,
            'pred_contact_l': z['pred_contact'][:, 0], 'pred_contact_r': z['pred_contact'][:, 1],
            'gt_contact_l': z['gt_contact'][:, 0], 'gt_contact_r': z['gt_contact'][:, 1],
        }
        return np.stack([cols[k] for k in SERIES], 1).astype(np.float32)

    def subjects(self):
        """被験者が連続する区間 (テストデータは被験者ごとに並んでいる)。"""
        s, out = self.z['subject'], []
        starts = np.r_[0, np.nonzero(s[1:] != s[:-1])[0] + 1]
        ends = np.r_[starts[1:], len(s)]
        for a, b in zip(starts, ends):
            out.append({'name': str(s[a]), 'gender': str(self.z['gender'][a]),
                        'betas': [round(float(x), 3) for x in self.z['betas'][a]],
                        'start': int(a), 'end': int(b)})
        return out

    def meta(self):
        try:
            source = str(self.paths.smpl_pred.relative_to(ROOT)).replace('\\', '/')
        except ValueError:
            source = str(self.paths.smpl_pred)
        return {
            'frames': self.N, 'fps': self.fps, 'chunk': CHUNK,
            'verts': self.V, 'faces': len(self.faces), 'joints': NUM_JOINTS,
            'parents': PARENTS, 'joint_names': JOINT_NAMES, 'body_joints': BODY_JOINTS,
            'series': SERIES, 'subjects': self.subjects(), 'source': source,
            'device': str(self.device),
        }

    def chunk(self, c):
        """チャンク c の予測・正解のメッシュと関節 (int16, mm) をバイト列で返す。"""
        with self.lock:
            if c in self.cache:
                self.cache.move_to_end(c)
                return self.cache[c]
            idx = np.arange(c * CHUNK, min((c + 1) * CHUNK, self.N))
            if len(idx) == 0:
                raise IndexError(c)
            z, V = self.z, self.V
            out = np.empty((len(idx), 2, V * 3 + NUM_JOINTS * 3), np.int16)
            genders = z['gender'][idx]
            with torch.no_grad():
                for g in np.unique(genders):
                    sel = np.nonzero(genders == g)[0]
                    ig = idx[sel]

                    def t(k):
                        return torch.as_tensor(z[k][ig], device=self.device)
                    for k, key in enumerate(('pred', 'gt')):
                        verts = self.bank(t(f'{key}_poses'), t('betas'), t(f'{key}_trans'), str(g),
                                          return_verts=True).verts
                        out[sel, k, :V * 3] = to_mm_int16(verts.cpu().numpy()).reshape(len(ig), -1)
                        out[sel, k, V * 3:] = to_mm_int16(z[f'{key}_joints'][ig]).reshape(len(ig), -1)
            data = out.tobytes()
            self.cache[c] = data
            while len(self.cache) > CACHE_CHUNKS:
                self.cache.popitem(last=False)
            return data


def make_handler(source):
    meta_json = json.dumps(source.meta(), ensure_ascii=False).encode('utf-8')
    faces_bin = source.faces.tobytes()
    series_bin = source.series.tobytes()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, body, ctype, status=200):
            self.send_response(status)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            try:
                if url.path in ('/', '/index.html'):
                    # 毎回読み直す (HTML を編集したらブラウザの再読み込みだけで反映される)
                    self._send(HTML_PATH.read_bytes(), 'text/html; charset=utf-8')
                elif url.path == '/api/meta':
                    self._send(meta_json, 'application/json')
                elif url.path == '/api/faces':
                    self._send(faces_bin, 'application/octet-stream')
                elif url.path == '/api/series':
                    self._send(series_bin, 'application/octet-stream')
                elif url.path == '/api/chunk':
                    c = int(parse_qs(url.query).get('c', ['-1'])[0])
                    if c < 0:
                        raise IndexError(c)
                    self._send(source.chunk(c), 'application/octet-stream')
                else:
                    self._send(b'not found', 'text/plain', 404)
            except (IndexError, ValueError):
                self._send(b'bad request', 'text/plain', 400)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass   # ブラウザ側が要求を取り消した

        def log_message(self, fmt, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description='予測した SMPL メッシュを確認する Web UI')
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-browser', action='store_true', help='ブラウザを自動で開かない')
    parser.add_argument('--cpu', action='store_true', help='GPU があっても CPU で計算する')
    args = parser.parse_args()

    config = load_config(args.config)
    if (config.get('dataset') or {}).get('target', 'skeleton') != 'smpl':
        parser.error('dataset.target が smpl のときだけ使えます')
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    print('Loading predictions and SMPL model...')
    source = MeshSource(config, device)
    source.chunk(0)   # SMPL の読み込みと最初のチャンクの計算を先に済ませる
    handler = make_handler(source)

    server = None
    for port in range(args.port, args.port + 20):
        try:
            server = ThreadingHTTPServer((args.host, port), handler)
            break
        except OSError:
            continue
    if server is None:
        raise SystemExit(f'ポート {args.port}〜{args.port + 19} がすべて使用中です (--port で指定してください)')
    server.daemon_threads = True

    url = f'http://{args.host}:{server.server_address[1]}/'
    print(f'{source.N:,} frames ({source.N / source.fps:.1f} s), device: {device}')
    print(f'Serving {url}  (Ctrl+C で終了)')
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n終了します')
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
