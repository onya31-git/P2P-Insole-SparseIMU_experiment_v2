"""
生データ (data/raw) から学習用データ (data/sensor, data/skeleton) を作成する。

処理内容:
    - センサー : data/raw/<収録>/sensor/<デバイス>/<録画>/<録画>.data.csv をそのままコピーし、
                 data/sensor/<split>/<ID>_<部位>.csv にリネームする
    - スケルトン: data/raw/<収録>/skeleton/csv/<名前>.csv の区切り文字 ';' を ',' に、
                 改行を LF に変換して data/skeleton/<split>/<ID>_skeleton.csv に保存する

新しい収録を追加するときは COLLECTIONS に対応表を追記する。

使い方:
    python scripts/prepare_data.py              # 既存ファイルはスキップ
    python scripts/prepare_data.py --overwrite  # 既存ファイルも作り直す
"""
import sys
import shutil
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 収録ごとの対応表
COLLECTIONS = {
    '20261101DC_newInsole': {
        # デバイスID (センサーフォルダ名) -> 装着部位
        'devices': {'DCDC': 'AL', 'AED0': 'AR', 'CCC4': 'BC', 'AB04': 'FL', 'AB00': 'FR'},
        # (録画フォルダ名, スケルトンCSV名 or None, データID, 保存先 split)
        'recordings': [
            ('062457', None,        'D1S1A0', 'calib'),   # キャリブレーション (mocap なし)
            ('065635', 'skeleton1', 'D1S1A1', 'train'),
            ('070352', 'skeleton2', 'D1S1A2', 'train'),
            ('071141', 'skeleton3', 'D1S1A3', 'train'),
            ('071844', 'skeleton4', 'D1S1T1', 'test'),
        ],
    },
}


def convert_skeleton(src, dst):
    """mocap 書き出しCSV (セミコロン区切り・CRLF) をカンマ区切り・LF に変換する。"""
    with open(src, 'r', encoding='utf-8', newline=None) as fin, \
         open(dst, 'w', encoding='utf-8', newline='\n') as fout:
        for line in fin:
            fout.write(line.replace(';', ','))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-dir', default=ROOT / 'data' / 'raw', type=Path)
    parser.add_argument('--out-dir', default=ROOT / 'data', type=Path)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    tasks = []  # (処理関数, src, dst)
    for name, col in COLLECTIONS.items():
        base = args.raw_dir / name
        for rec, skel, data_id, split in col['recordings']:
            for device, part in col['devices'].items():
                src = base / 'sensor' / device / rec / f'{rec}.data.csv'
                dst = args.out_dir / 'sensor' / split / f'{data_id}_{part}.csv'
                tasks.append((shutil.copyfile, src, dst))
            if skel is not None:
                src = base / 'skeleton' / 'csv' / f'{skel}.csv'
                dst = args.out_dir / 'skeleton' / split / f'{data_id}_skeleton.csv'
                tasks.append((convert_skeleton, src, dst))

    missing = [str(src) for _, src, _ in tasks if not src.exists()]
    if missing:
        sys.exit('生データが見つかりません:\n  ' + '\n  '.join(missing))

    for func, src, dst in tasks:
        if dst.exists() and not args.overwrite:
            print(f'skip   {dst.relative_to(args.out_dir)}')
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        func(src, dst)
        print(f'write  {dst.relative_to(args.out_dir)}  <-  {src.relative_to(args.raw_dir)}')


if __name__ == '__main__':
    main()
