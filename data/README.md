# データ

```
data/
├── raw/                         生データ (収録したまま。編集しない)
│   └── 20261101DC_newInsole/
│       ├── sensor/<デバイスID>/<録画>/<録画>.data.csv, .meta.json
│       └── skeleton/{csv,bvh}/skeleton1〜4.*
├── sensor/                      学習用センサーデータ
│   ├── train/   D1S1A1〜A3_{AL,AR,BC,FL,FR}.csv
│   ├── test/    D1S1T1_{AL,AR,BC,FL,FR}.csv
│   └── calib/   D1S1A0_{AL,AR,BC,FL,FR}.csv   (キャリブレーション。学習・テストでは使わない)
└── skeleton/                    学習用スケルトン (mocap)
    ├── train/   D1S1A1〜A3_skeleton.csv
    └── test/    D1S1T1_skeleton.csv
```

`sensor/` と `skeleton/` は `python scripts/prepare_data.py` で `raw/` から再生成できます。

## 命名規則 `D1S1A1_AL.csv`

| 記号 | 意味 |
|---|---|
| **D** | データ収集回 (Data collection number) |
| **S** | 被験者 (Subject number) |
| **A** | 動作セクション (Action section number)。`A0` はキャリブレーション |
| **T** | テスト用データ |
| 部位 | `AL` 左腕 / `AR` 右腕 / `BC` 背中 / `FL` 左足 / `FR` 右足 |

正規化統計量は `D?S?` (収録回 × 被験者) 単位で計算されます。

## 生データとの対応 (20261101DC_newInsole → D1S1)

デバイス → 部位

| デバイスID | DCDC | AED0 | CCC4 | AB04 | AB00 |
|---|---|---|---|---|---|
| 部位 | AL | AR | BC | FL | FR |

録画 → データID

| 録画 (センサー) | スケルトン | データID | 用途 |
|---|---|---|---|
| 062457 | なし | D1S1A0 | calib |
| 065635 | skeleton1 | D1S1A1 | train |
| 070352 | skeleton2 | D1S1A2 | train |
| 071141 | skeleton3 | D1S1A3 | train |
| 071844 | skeleton4 | D1S1T1 | test |

新しい収録を追加するときは `scripts/prepare_data.py` の `COLLECTIONS` に対応表を追記してください。

## ファイル形式

- **センサー CSV**: `timestamp_ms, P1..P112, Mag_x..z, Gyro_x..z, Acc_x..z, frame_seq, time_ref` (約 60 Hz)
  - `timestamp_ms` は全デバイス共通のエポックミリ秒
  - 圧力 `P1..P112` は 8 行 × 14 列を行優先で並べたもの (足センサー FL / FR のみ使用)
  - IMU は Gyro + Acc の 6 次元を使用 (Mag は不使用)
- **スケルトン CSV**: mocap の書き出し (100 Hz)。3 行目が関節名ヘッダー、6 行目以降がデータ。
  各関節に `*Position(s)` (X, Y, Z, 単位 mm) と `*Angles` (4 列) の列がある。
  - `*Angles` は親関節に対する相対回転のクオータニオンで、**列順は X, Y, Z, W** (4 行目の軸ラベル参照)。
    読み込み時に W, X, Y, Z の順へ並べ替えている。
  - **肘 (`LElbowAngles`, `RElbowAngles`) だけはクオータニオンではない**。
    Y 列に「上腕と前腕のなす角 [rad]」(伸展で π) が入っており、X, Z 列は常に 0。
    読み込み時に屈曲角 θ = π − なす角 から X 軸まわりの回転 (cos θ/2, sin θ/2, 0, 0) に変換している
    (`dataset/dataset.py` の `_hinge_to_quat`)。
  - `CenterOfGravity` には回転列が無いため、単位クオータニオンで埋めている。
  生データはセミコロン区切りのため、`prepare_data.py` でカンマ区切りに変換している。
