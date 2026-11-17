"""
設定ファイルの読み込みと、入出力パスの一元管理。

出力ファイルの場所はすべてここで決まる (config.yaml の paths.output_dir 配下):

    outputs/
    ├── checkpoints/
    │   ├── model.pth          学習済み重み              (train.py が保存 / test.py が読込)
    │   ├── stats.json         正規化統計量              (train.py が保存 / test.py が読込)
    │   └── data_ids.json      学習に使った収録 ID       (train.py が保存 / test.py が重複を確認)
    ├── sync/
    │   ├── train.json         学習データの時刻同期結果  (train.py)
    │   └── test.json          テストデータの時刻同期結果 (test.py)
    ├── calib/
    │   ├── train.json         学習データのキャリブレーション結果 (train.py)
    │   └── test.json          テストデータのキャリブレーション結果 (test.py)
    └── results/
        ├── predictions_pos.csv 関節位置の予測と正解      (test.py)
        ├── predictions_rot.csv 関節回転の予測と正解      (test.py)
        ├── predictions_contact.csv 接地の予測と正解      (test.py)
        ├── smpl_predictions.npz SMPL の予測と正解        (test.py, dataset.target = smpl のとき)
        ├── mesh/               メッシュ (OBJ)            (scripts/export_mesh.py)
        ├── evaluation.txt      評価指標 (MPJPE 等)       (evaluate.py)
        └── visualize_data.js   可視化用データ            (visualize.py)

相対パスはプロジェクトルート (このリポジトリの最上位) 基準で解決するため、
どのディレクトリからスクリプトを実行しても同じ場所に読み書きする。
"""
from pathlib import Path
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'config.yaml'


def resolve(path):
    """相対パスをプロジェクトルート基準の絶対パスにする。"""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_config(config_path=DEFAULT_CONFIG):
    p = Path(config_path)
    if not p.is_absolute() and not p.exists():
        p = ROOT / p
    with open(p, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def resolve_ids(config, mode, cli_ids=None):
    """
    学習 / テストに使う収録 ID を決める。コマンドラインの --ids > config の dataset.{train,test}_ids。
    どちらも無ければ None (= data/sensor/<mode>/ などのフォルダにある収録をすべて使う)。
    """
    if cli_ids:
        return list(cli_ids)
    ids = (config.get('dataset') or {}).get(f'{mode}_ids')
    if isinstance(ids, str):
        ids = [ids]
    return list(ids) if ids else None


def get_paths(config, makedirs=False):
    """config から入出力パスをまとめて返す。"""
    cfg = config.get('paths', {})
    out = resolve(cfg.get('output_dir', 'outputs'))
    ckpt, sync, results = out / 'checkpoints', out / 'sync', out / 'results'
    calib = out / 'calib'
    if makedirs:
        for d in (ckpt, sync, results, calib):
            d.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        data_dir=resolve(cfg.get('data_dir', 'data')),
        output_dir=out,
        weights=ckpt / 'model.pth',
        stats=ckpt / 'stats.json',
        data_ids=ckpt / 'data_ids.json',
        sync_train=sync / 'train.json',
        sync_test=sync / 'test.json',
        calib_train=calib / 'train.json',
        calib_test=calib / 'test.json',
        pred_pos=results / 'predictions_pos.csv',
        pred_rot=results / 'predictions_rot.csv',
        pred_contact=results / 'predictions_contact.csv',
        smpl_pred=results / 'smpl_predictions.npz',
        mesh_dir=results / 'mesh',
        evaluation=results / 'evaluation.txt',
        visualize_data=results / 'visualize_data.js',
    )
