"""
リアルタイム推定の表示用 HTTP サーバー (標準ライブラリのみ)。

    GET  /             realtime/viewer.html
    GET  /api/model    メッシュのデータ (体型 β のレストポーズ頂点・面・スキニングの重み)
    GET  /api/events   Server-Sent Events: frames (推定結果のまとまり) / status (状態, 約 2 Hz)
    POST /api/tpose    T ポーズのキャリブレーションを (やり直し) 開始する

メッシュはブラウザ側で線形ブレンドスキニング (関節ごとの変換 A_j を頂点の重みで混ぜる) で変形する。
SMPL のポーズ補正 (posedirs) は省略するため、オフラインのメッシュ (mesh_viewer.py) とはわずかに形が異なる。
"""
import json
import queue
import threading
import webbrowser
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path

HTML = Path(__file__).with_name('viewer.html')


class Broadcaster:
    """SSE の接続ごとの queue にメッセージを配る (表示が遅いクライアントの分は古いものから捨てる)。"""
    def __init__(self):
        self.lock = threading.Lock()
        self.clients = []

    def add(self):
        q = queue.Queue(maxsize=300)
        with self.lock:
            self.clients.append(q)
        return q

    def remove(self, q):
        with self.lock:
            if q in self.clients:
                self.clients.remove(q)

    def publish(self, event, data):
        msg = f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode()
        with self.lock:
            clients = list(self.clients)
        for q in clients:
            try:
                q.put_nowait(msg)
            except queue.Full:
                try:
                    q.get_nowait()
                    q.put_nowait(msg)
                except (queue.Empty, queue.Full):
                    pass


class ViewerServer:
    def __init__(self, host, port, model_payload, on_command, log=print):
        self.hub = Broadcaster()
        self.model_bytes = json.dumps(model_payload, separators=(',', ':')).encode()
        self.on_command = on_command
        self.log = log
        self.httpd = ThreadingHTTPServer((host, port), self._handler())
        self.httpd.daemon_threads = True
        self.url = f"http://{host}:{self.httpd.server_address[1]}/"

    def start(self, open_browser=True):
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.log(f"[viewer] {self.url} で表示しています")
        if open_browser:
            webbrowser.open(self.url)

    def stop(self):
        self.httpd.shutdown()

    def publish(self, event, data):
        self.hub.publish(event, data)

    def _handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, body, ctype, status=200):
                self.send_response(status)
                self.send_header('Content-Type', ctype)
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = self.path.split('?')[0]
                if path in ('/', '/index.html'):
                    self._send(HTML.read_bytes(), 'text/html; charset=utf-8')
                elif path == '/api/model':
                    self._send(server.model_bytes, 'application/json')
                elif path == '/api/events':
                    self._events()
                else:
                    self._send(b'not found', 'text/plain', 404)

            def do_POST(self):
                path = self.path.split('?')[0]
                if path == '/api/tpose':
                    server.on_command('tpose')
                    self._send(b'{"ok":true}', 'application/json')
                else:
                    self._send(b'not found', 'text/plain', 404)

            def _events(self):
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('Connection', 'keep-alive')
                self.end_headers()
                q = server.hub.add()
                try:
                    while True:
                        try:
                            msg = q.get(timeout=10)
                        except queue.Empty:
                            msg = b": ping\n\n"
                        self.wfile.write(msg)
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                    pass
                finally:
                    server.hub.remove(q)

            def log_message(self, fmt, *args):
                pass

        return Handler
