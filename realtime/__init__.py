"""
センサーから受信したデータで全身の姿勢をリアルタイムに推定する (run_realtime.py から使う)。

    sources.py   センサーサーバー (WebSocket) からの受信 / 収録済み CSV の再生
    pipeline.py  逐次の前処理 (オフラインと同じ計算)・推論・SMPL の順運動学
    viewer.py    ブラウザ表示用の HTTP サーバー (Server-Sent Events) と viewer.html

オフラインのパイプライン (dataset/, models/, train.py, test.py ...) は変更せず、読み込んで使うだけ。
"""
