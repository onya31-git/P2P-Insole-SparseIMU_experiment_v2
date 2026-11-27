"""
生データ (data/raw) から学習用データ (data/sensor, data/skeleton) を作成する。

処理内容:
    - センサー : data/raw/<収録>/sensor/<デバイス>/<録画>/<録画>.data.csv をそのままコピーし、
                 data/sensor/<split>/<ID>_<部位>.csv にリネームする
    - スケルトン: data/raw/<収録>/skeleton/csv/<名前>.csv の区切り文字 ';' を ',' に、
                 改行を LF に変換して data/skeleton/<split>/<ID>_skeleton.csv に保存する
    - BVH       : data/raw/<収録>/skeleton/bvh/<名前>.bvh をそのままコピーし、
                 data/skeleton/<split>/<ID>_skeleton.bvh に保存する (SMPL フィッティングの入力)

新しい収録を追加するときは COLLECTIONS に対応表を追記する。

使い方:
    uv run scripts/prepare_data.py              # 既存ファイルはスキップ
    uv run scripts/prepare_data.py --overwrite  # 既存ファイルも作り直す
"""
import sys
import shutil
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 収録ごとの対応表
COLLECTIONS = {
    '20261101DC1': {
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
    # A2 の内容が変わった (D1: 足踏み + その場でのジョグ → D2: 歩行 + 移動するジョグ)
    '20261116DC2': {
        # デバイスID (センサーフォルダ名) -> 装着部位
        'devices': {'DCDC': 'AL', 'AED0': 'AR', 'CCC4': 'BC', 'AB04': 'FL', 'AB00': 'FR'},
        # (録画フォルダ名, スケルトンCSV名 or None, データID, 保存先 split)
        'recordings': [
            ('091606', None,        'D2S1A0', 'calib'),   # キャリブレーション (mocap なし)
            ('093126', 'skeleton1', 'D2S1A1', 'train'),
            ('094201', 'skeleton2', 'D2S1A2', 'train'),
            ('095129', 'skeleton3', 'D2S1A3', 'train'),
            ('100316', 'skeleton4', 'D2S1T1', 'test'),
            ('101429', 'skeleton5', 'D2S1A4', 'train'),
        ],
    },
    '20261119DC3': {
        # デバイスID (センサーフォルダ名) -> 装着部位
        'devices': {'DCDC': 'AL', 'AED0': 'AR', 'CCC4': 'BC', 'AB04': 'FL', 'AB00': 'FR'},
        # (録画フォルダ名, スケルトンCSV名 or None, データID, 保存先 split)
        'recordings': [
            ('074219', None,        'D3S2A0', 'calib'),   # キャリブレーション (mocap なし)
            ('075837', 'skeleton1', 'D3S2A1', 'train'),
            ('080556', 'skeleton2', 'D3S2A2', 'train'),
            ('081305', 'skeleton3', 'D3S2A3', 'train'),
            ('082047', 'skeleton4', 'D3S2T1', 'test'),
        ],
    },
    '20261120DC4': {
            # デバイスID (センサーフォルダ名) -> 装着部位
            'devices': {'DCDC': 'AL', 'AED0': 'AR', 'CCC4': 'BC', 'AB04': 'FL', 'AB00': 'FR'},
            # (録画フォルダ名, スケルトンCSV名 or None, データID, 保存先 split)
            'recordings': [
                ('064316', None,        'D4S3A0', 'calib'),   # キャリブレーション (mocap なし)
                ('065914', 'skeleton1', 'D4S3A1', 'train'),
                ('071020', 'skeleton2', 'D4S3A2', 'train'),
                ('072004', 'skeleton3', 'D4S3A3', 'train'),
                ('072844', 'skeleton4', 'D4S3T1', 'test'),
            ],
    },
    '20261123DC5': {
            # デバイスID (センサーフォルダ名) -> 装着部位
            'devices': {'DCDC': 'AL', 'AED0': 'AR', 'CCC4': 'BC', 'AB04': 'FL', 'AB00': 'FR'},
            # (録画フォルダ名, スケルトンCSV名 or None, データID, 保存先 split)
            'recordings': [
                ('064529', None,        'D5S4A0', 'calib'),   # キャリブレーション (mocap なし)
                ('070224', 'skeleton1', 'D5S4A1', 'train'),
                ('071227', 'skeleton2', 'D5S4A2', 'train'),
                ('072058', 'skeleton3', 'D5S4A3', 'train'),
                ('073009', 'skeleton4', 'D5S4T1', 'test'),
            ],
    },
    '20261126DC6': {
    # デバイスID (センサーフォルダ名) -> 装着部位
    'devices': {'DCDC': 'AL', 'AED0': 'AR', 'CCC4': 'BC', 'AB04': 'FL', 'AB00': 'FR'},
    # (録画フォルダ名, スケルトンCSV名 or None, データID, 保存先 split)
    'recordings': [
        ('070023', None,        'D6S5A0', 'calib'),   # キャリブレーション (mocap なし)
        ('072315', 'skeleton1', 'D6S5A1', 'train'),
        ('073230', 'skeleton2', 'D6S5A2', 'train'),
        ('074055', 'skeleton3', 'D6S5A3', 'train'),
        ('074922', 'skeleton4', 'D6S5T1', 'test'),
        ],
    },
    '20261127DC7': {
            # デバイスID (センサーフォルダ名) -> 装着部位
            'devices': {'DCDC': 'AL', 'AED0': 'AR', 'CCC4': 'BC', 'AB04': 'FL', 'AB00': 'FR'},
            # (録画フォルダ名, スケルトンCSV名 or None, データID, 保存先 split)
            'recordings': [
                ('040345', None,        'D7S6A0', 'calib'),   # キャリブレーション (mocap なし)
                ('041620', 'skeleton1', 'D7S6A1', 'train'),
                ('042518', 'skeleton2', 'D7S6A2', 'train'),
                ('043350', 'skeleton3', 'D7S6A3', 'train'),
                ('044616', 'skeleton4', 'D7S6T1', 'test'),
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
                src = base / 'skeleton' / 'bvh' / f'{skel}.bvh'
                dst = args.out_dir / 'skeleton' / split / f'{data_id}_skeleton.bvh'
                tasks.append((shutil.copyfile, src, dst))

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
