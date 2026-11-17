# データ

```
data/
├── raw/                         生データ (収録したまま。編集しない)
│   ├── 20261101DC_atsuya/       D1 (旧フォルダ名 20261101DC_newInsole)
│   │   ├── sensor/<デバイスID>/<録画>/<録画>.data.csv, .meta.json
│   │   └── skeleton/{csv,bvh}/skeleton1〜4.*
│   └── 20261116DC_atsuya2/      D2
│       ├── sensor/<デバイスID>/<録画>/...
│       └── skeleton/{csv,bvh}/skeleton1〜5.*
├── smpl/                        SMPL 形式のデータセット (scripts/fit_smpl.py が BVH から作成, git 管理外)
│   ├── train/   D1S1A1〜A3_smpl.npz, D2S1A1〜A4_smpl.npz
│   ├── test/    D1S1T1_smpl.npz, D2S1T1_smpl.npz
│   └── fit_report.txt           フィッティング誤差のまとめ
├── sensor/                      学習用センサーデータ
│   ├── train/   D1S1A1〜A3_{AL,AR,BC,FL,FR}.csv, D2S1A1〜A4_*.csv
│   ├── test/    D1S1T1_{AL,AR,BC,FL,FR}.csv, D2S1T1_*.csv
│   └── calib/   D1S1A0_*.csv, D2S1A0_*.csv   (机置きデータ。キャリブレーションにだけ使う)
└── skeleton/                    学習用スケルトン (mocap)
    ├── train/   D1S1A1〜A3_skeleton.{csv,bvh}, D2S1A1〜A4_skeleton.*
    └── test/    D1S1T1_skeleton.{csv,bvh}, D2S1T1_skeleton.*
```

`sensor/` と `skeleton/` は `python scripts/prepare_data.py` で `raw/` から再生成できます。

## 命名規則 `D1S1A1_AL.csv`

| 記号 | 意味 |
|---|---|
| **D** | データ収集回 (Data collection number) |
| **S** | 被験者 (Subject number) |
| **A** | 動作セクション (Action section number)。`A0` は机置きデータ (キャリブレーション用) |
| **T** | テスト用データ |
| 部位 | `AL` 左腕 / `AR` 右腕 / `BC` 背中 / `FL` 左足 / `FR` 右足 |

正規化統計量は `D?S?` (収録回 × 被験者) 単位で計算されます。

## 収録の手順 (キャリブレーション用。2026-11-15 以降の収集から)

- **机置きデータ (`A0`)**: データ収集の直前に 1 回だけ、全センサー (インソールは無負荷) を机に置いて静止させて記録する。
  データ ID はその収集回の最初の被験者の `D?S?A0` とし、`sensor/calib/` に置く。
  同じ収集回 `D?` の他の被験者の収録にも、この机置きデータが使われる (ジャイロのバイアスと圧力のゼロ点)。
- **T ポーズ**: 各収録の最初と最後に T ポーズを 3 秒キープする (IMU の取り付け向きの補正と圧力の体重基準化)。
  最初の T ポーズを使い、最初が見つからない場合は最後の T ポーズを使う。両方あれば、その差を
  `outputs/calib/*.json` の `start_end_diff_deg` で確認できる (バンドのずれの目安)。
- D1S1 は T ポーズが 1 秒前後と短く、センサーの記録開始が mocap より遅いため、A3 には T ポーズが残っていない
  (他の収録の平均で代用している)。D2S1 からは最初と最後に約 5 秒の T ポーズがある。

## 収集回ごとの注意

- **動作の内容**: A2 の内容が D1 と D2 で異なる。
  D1: 足踏み + その場でのジョグ / D2: 歩行 + 移動するジョグ (A4 は D2 から)。
  ただし D2 の mocap でも体の水平位置はほぼ固定 (収録中の重心の移動範囲 0.3 m 程度) で、向きも一定のため、
  歩行・ジョグでは立脚中の足が後ろへ動く (トレッドミル上と同じ見え方)。並進の水平成分はルート相対にしているので
  学習には影響しないが、足の速さを使う mocap の接地判定は使えない (`dataset.contact_source: pressure` を使う)。
- **mocap の高さ**: D2 (特に D2S1T1) は足の高さが収録中に 40〜80 mm ドリフトし、瞬間的な外れ値 (−300 mm 程度) もある。
- **センサーの時計**: D2 ではインソール (FL / FR, デバイス AB04 / AB00) の時刻だけが腕・背中の IMU より約 0.5 秒ずれている
  (CSV の記録開始時刻も 0.52 秒早い)。`config.yaml` の `sync.groups` でまとまりごとに同期して補正している
  (補正量は `outputs/sync/*.json` の `clock_shift_sec`。D2 の 5 収録で −0.48〜−0.54 秒、D1 は −0.05 秒以内)。
- **圧力の大きさ**: D2 のインソールは D1 より値が約 4 倍大きく、足を上げても合計で約 39000 の荷重が残る
  (正規化は `D?S?` ごとなので学習には影響しない)。

## 生データとの対応 (20261101DC_atsuya → D1S1)

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

## 生データとの対応 (20261116DC_atsuya2 → D2S1)

デバイス → 部位は D1 と同じ。

| 録画 (センサー) | スケルトン | データID | 用途 |
|---|---|---|---|
| 091606 | なし | D2S1A0 | calib |
| 093126 | skeleton1 | D2S1A1 | train |
| 094201 | skeleton2 | D2S1A2 | train |
| 095129 | skeleton3 | D2S1A3 | train |
| 100316 | skeleton4 | D2S1T1 | test |
| 101429 | skeleton5 | D2S1A4 | train |

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
- **BVH**: 同じ mocap の書き出し (100 Hz, mm)。CSV と同じフレーム数で、階層は Hips をルートに
  Spine〜Spine4・Neck・Head・腕・脚・つま先 (+ End Site)。回転は CHANNELS の記載順 (X, Y, Z) の内因性オイラー角 [deg]。
  順運動学の結果は CSV の関節位置と 0.03 mm 以内で一致する (`scripts/check_bvh.py`)。
  レストポーズは T ポーズで、y が上・+z が正面・+x が左 (SMPL と同じ向き)。
- **SMPL npz** (`data/smpl/`): AMASS と同じキー `poses` [T, 72] (軸角), `trans` [T, 3] (m), `betas`, `gender`,
  `mocap_framerate` に加え、`joints` [T, 24, 3], `offsets` (SMPL 関節 → mocap 関節の定義の差),
  `fit_err_mm`, `ori_err_deg` を含む。フレームは BVH / CSV と同じ (データセット読み込み時に CSV と同じ範囲を切り出す)。
  生データはセミコロン区切りのため、`prepare_data.py` でカンマ区切りに変換している。
