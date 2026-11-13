"""
予測スケルトンと正解スケルトンをブラウザで 3D 表示する。

使い方:
    uv run visualize.py [--start 0] [--end 3000] [--step 1] [--config config.yaml]

入力: outputs/results/predictions_pos.csv (test.py の出力)
出力: outputs/results/visualize_data.js  (visualize.html が読み込む)
      ローカル HTTP サーバーを立ててブラウザで visualize.html を開く。
"""
import json
import functools
import pandas as pd
import webbrowser
import numpy as np
import http.server
import socketserver
import threading
import time
import argparse

from utils.paths import ROOT, load_config, get_paths, DEFAULT_CONFIG


def visualize():
    parser = argparse.ArgumentParser(description='Visualize Skeleton Predictions.')
    parser.add_argument('--start', type=int, default=0, help='Start frame index')
    parser.add_argument('--end', type=int, default=3000, help='End frame index (use -1 for all frames)')
    parser.add_argument('--step', type=int, default=1,
                        help='Visualize every N-th frame (e.g. 10 -> frames 0, 10, 20, ...)')
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    args = parser.parse_args()
    paths = get_paths(load_config(args.config), makedirs=True)

    start_idx = args.start
    end_idx = args.end
    step = args.step
    if step < 1:
        parser.error('--step must be >= 1')

    print(f"Loading {paths.pred_pos.name} (frames {start_idx} to {end_idx if end_idx != -1 else 'end'}, every {step} frame(s))...")
    try:
        df = pd.read_csv(paths.pred_pos)
    except FileNotFoundError:
        print(f"Error: {paths.pred_pos} not found. Run test.py first.")
        return

    pred_cols = [c for c in df.columns if c.startswith('Pred_')]
    gt_cols = [c for c in df.columns if c.startswith('GT_')]

    flat_pred = df[pred_cols].values.reshape(-1, 18, 3)
    flat_gt = df[gt_cols].values.reshape(-1, 18, 3)

    total_frames = flat_pred.shape[0]
    if end_idx == -1 or end_idx > total_frames:
        end_idx = total_frames

    flat_pred_sliced = flat_pred[start_idx:end_idx:step]
    flat_gt_sliced = flat_gt[start_idx:end_idx:step]
    # 間引き後も元のフレーム番号を表示できるように保持する
    frame_indices = list(range(total_frames))[start_idx:end_idx:step]
    if len(frame_indices) == 0:
        print(f"Error: no frames selected (total frames: {total_frames}).")
        return
    print(f"Selected {len(frame_indices)} frames (of {total_frames} total).")

    # Calculate global bounding box for the sliced frames
    all_points = np.concatenate([flat_pred_sliced, flat_gt_sliced], axis=0)
    # NaN might be present, so use nanmin, nanmax
    min_vals = np.nanmin(all_points, axis=(0, 1)).tolist()
    max_vals = np.nanmax(all_points, axis=(0, 1)).tolist()

    # Optional padding
    padding = [ (max_vals[i] - min_vals[i]) * 0.1 for i in range(3) ]
    for i in range(3):
        if padding[i] == 0: padding[i] = 1.0 # fallback
        min_vals[i] -= padding[i]
        max_vals[i] += padding[i]

    # Convert back to list for JSON serialization
    pred_list = flat_pred_sliced.tolist()
    gt_list = flat_gt_sliced.tolist()

    output_data = {
        'pred_pos_flat': pred_list,
        'gt_pos_flat': gt_list,
        'frame_indices': frame_indices,
        'step': step,
        'bounds': {
            'x': [min_vals[0], max_vals[0]],
            'y': [min_vals[1], max_vals[1]],
            'z': [min_vals[2], max_vals[2]]
        }
    }

    # visualize.html が読み込む JS ファイルとして保存
    with open(paths.visualize_data, 'w') as f:
        f.write(f"const skeletonData = {json.dumps(output_data)};\n")
    print(f"Saved visualization data to {paths.visualize_data}.")

    # Start a local HTTP server in a background thread to ensure it opens in a web browser
    PORT = 8000

    # Find an open port
    # プロジェクトルートを公開する (visualize.html と outputs/ を同じサーバーから配信)
    Handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT))

    # Try different ports if 8000 is occupied
    for port in range(8000, 8010):
        try:
            httpd = socketserver.TCPServer(("", port), Handler)
            PORT = port
            break
        except OSError:
            continue
    else:
        print("Could not find an open port.")
        return

    def serve():
        httpd.serve_forever()

    server_thread = threading.Thread(target=serve, daemon=True)
    server_thread.start()

    data_url = paths.visualize_data.relative_to(ROOT).as_posix()
    url = f'http://localhost:{PORT}/visualize.html?data={data_url}&t={int(time.time())}'
    print(f"Server started at {url}")
    print(f"Opening visualization in browser...")
    webbrowser.open(url)

    print("\nPress Ctrl+C to close the server and exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.shutdown()
        httpd.server_close()

if __name__ == '__main__':
    visualize()
