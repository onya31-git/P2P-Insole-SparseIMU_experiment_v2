# P2P-Insole + Sparse IMU (v2)

圧力センサー付きインソール (左右) と、体に付けた少数の IMU から、
モーションキャプチャ相当の全身 18 関節の位置・回転 (クオータニオン) を推定するシステム。

- 入力: 圧力 8×14 = 112 点 ×左右 + IMU (Gyro 3 + Acc 3) ×5 部位 (左右手首・左右足・背中) + 被験者の体型 β
- 正解: `config.yaml` の `dataset.target` で切り替える
  - `smpl` (既定): mocap の BVH を SMPL 人体モデルにフィッティングしたパラメータ (100 Hz)
  - `skeleton`: mocap のスケルトン (100 Hz, 18 関節の位置 + クオータニオン)
- 出力:
  - `smpl`: SMPL の姿勢 (24 関節の局所回転) と並進 → SMPL で関節位置と 3D メッシュを得る / 左右の足の接地
  - `skeleton`: 18 関節の位置 (CenterOfGravity 基準の相対座標) / 回転 / 左右の足の接地

## 使い方

環境構築は [uv](https://docs.astral.sh/uv/) で行います (依存関係は `pyproject.toml` / `uv.lock`)。
PyTorch は CUDA 13.2 版 (`cu132`) を PyTorch 公式インデックスから取得する設定です。

```bash
uv sync                          # .venv を作成して依存ライブラリをインストール

uv run scripts/prepare_data.py   # (生データから作り直すときのみ) data/raw → data/sensor, data/skeleton
uv run scripts/check_bvh.py      # (任意) BVH の順運動学が CSV の関節位置を再現するか確認
uv run scripts/fit_smpl.py       # SMPL フィッティング → data/smpl/{train,test}/*_smpl.npz, data/smpl/fit_report.txt
uv run train.py                  # 学習        → <output_dir>/checkpoints/, <output_dir>/sync/train.json
uv run test.py                   # 推論        → <output_dir>/results/smpl_predictions.npz (skeleton: predictions_*.csv)
uv run evaluate.py               # 評価        → <output_dir>/results/evaluation.txt
uv run visualize.py --end 3000   # 3D 可視化   → ブラウザで visualize.html が開く
uv run visualize.py --mesh --end 600 --step 3   # SMPL のメッシュも表示 (フレーム数は数百程度に絞る)
uv run scripts/export_mesh.py --end 300         # メッシュを OBJ と AMASS 形式の npz に書き出す
uv run mesh_viewer.py                           # SMPL メッシュの Web UI → http://127.0.0.1:8765/ (全フレーム再生可)
```

`<output_dir>` は `config.yaml` の `paths.output_dir` (既定 `outputs/smpl`)。

### SMPL を使う準備

1. [SMPL 公式サイト](https://smpl.is.tue.mpg.de/) から SMPL のモデル (pkl) を入手し `body_models/` に置く
   (`basicmodel_m_lbs_10_207_0_v1.0.0.pkl` など。ライセンス上再配布できないため git 管理外)。
   chumpy / scipy は不要 (`utils/smpl.py` が pkl を直接読む)。
2. `config.yaml` の `smpl.subjects` に被験者ごとの性別と β (10 次元) を書く。
   β はフィッティングでは固定値として使い、学習ではモデルの入力 (`body_params`) にもなる。
   **β を変えたら `scripts/fit_smpl.py --overwrite` で作り直す** (データセット読み込み時に一致を確認している)。
3. `scripts/fit_smpl.py` を実行し、`data/smpl/fit_report.txt` で誤差を確認する
   (関節位置の誤差・BVH との回転の差・足裏のめり込み・ジッター比・SMPL と mocap の骨長の比較)。

ライブラリを追加するときは `uv add <パッケージ名>` を使います。

すべてのスクリプトは `--config <yaml>` で設定ファイルを切り替えられます (既定: `config.yaml`)。
パスはプロジェクトルート基準で解決されるので、どのディレクトリから実行しても同じ場所に読み書きします。

### 学習・テストに使う収録の指定

収録 ID (`D2S1A1` など) で指定できます。ワイルドカード (`*`, `?`) も使えます。

```bash
uv run train.py --ids D2S1A1 D2S1A2 "D1S1A*"   # 学習に使う収録
uv run test.py  --ids D2S1T1                   # テストに使う収録
```

- 毎回同じ指定にするなら `config.yaml` の `dataset.train_ids` / `dataset.test_ids` に書きます (`--ids` が優先)。
- ID を指定すると、`data/{sensor,skeleton,smpl}/` の `train/`・`test/` のどちらのフォルダにあっても探します。
  どちらも指定しなければ従来どおり、`train/` にある収録で学習し `test/` にある収録でテストします。
- 学習に使った ID は `<output_dir>/checkpoints/data_ids.json` に保存され、
  テストの収録と重なっていると `test.py` が警告します (例: `D1*` は `D1S1T1` も含むので注意)。
- 存在しない ID を指定すると、使える ID の一覧つきでエラーになります。

## フォルダ構成

```
.
├── config.yaml           設定 (入出力パス・同期・SMPL・学習・モデルのハイパーパラメータ)
├── train.py              学習
├── test.py               推論 (テストデータ) → 予測 CSV
├── evaluate.py           予測 CSV から MPJPE などを計算
├── visualize.py          予測と正解のスケルトンをブラウザで表示 (visualize.html を使用)
├── visualize.html
├── mesh_viewer.py        予測と正解の SMPL メッシュを確認する Web UI のサーバー (target: smpl)
├── mesh_viewer.html      その画面 (three.js)
├── pyproject.toml        依存ライブラリ (uv)
├── uv.lock
│
├── dataset/
│   ├── dataset.py        CSV 読み込み・センサーと mocap の時刻同期・正規化・チャンク分割
│   └── calibration.py    机置きデータと T ポーズによるセンサーのキャリブレーション
├── models/
│   ├── system.py         モデル全体 KinematicFusionSystem (下図)
│   └── *.py              各サブモジュール
├── losses/
│   ├── pose_loss.py      位置 / 回転 / 骨長 / 接地 / 平滑化の損失 (target: skeleton)
│   ├── smpl_loss.py      SMPL の回転 / 関節位置 / 並進 / 接地 / 平滑化 / 足の滑りの損失 (target: smpl)
│   └── cycle_loss.py     潜在表現からセンサー入力を再構成する損失
├── utils/
│   ├── paths.py          設定読み込みと入出力パスの一元管理
│   ├── export_csv.py     予測結果の CSV 書き出し
│   ├── rotation.py       回転表現の変換 (軸角 / 回転行列 / 6D / オイラー角)
│   ├── bvh.py            BVH の読み込みと順運動学
│   └── smpl.py           SMPL 人体モデル (順運動学・メッシュ)
├── scripts/
│   ├── prepare_data.py   生データ → 学習用データの変換
│   ├── check_bvh.py      BVH の解釈の確認 (順運動学と CSV の関節位置の比較)
│   ├── fit_smpl.py       BVH → SMPL パラメータのフィッティング
│   └── export_mesh.py    推論結果のメッシュを OBJ / AMASS 形式で書き出す
│
├── body_models/          SMPL のモデル (pkl, git 管理外)
│
├── data/                 データ (詳細は data/README.md)
│   ├── raw/              収録したままの生データ (編集しない)
│   ├── sensor/{train,test,calib}/     D1S1A1_AL.csv ...
│   ├── skeleton/{train,test}/         D1S1A1_skeleton.csv, D1S1A1_skeleton.bvh ...
│   └── smpl/{train,test}/             D1S1A1_smpl.npz ...  (fit_smpl.py が作成)
│
└── outputs/              生成物 (すべてスクリプトが出力。消しても再生成できる)
    ├── checkpoints/
    │   ├── model.pth          学習済み重み         train.py → test.py
    │   ├── stats.json         正規化統計量         train.py → test.py
    │   └── data_ids.json      学習に使った収録 ID  train.py → test.py (重複の確認)
    ├── sync/
    │   ├── train.json         時刻同期の推定結果   train.py
    │   └── test.json                               test.py
    ├── calib/
    │   ├── train.json         キャリブレーション結果 train.py
    │   └── test.json                               test.py
    └── results/
        ├── predictions_pos.csv  関節位置 (Pred_* / GT_*)   test.py → evaluate.py, visualize.py
        ├── predictions_rot.csv  関節回転 (RotPred_* / RotGT_*) test.py → evaluate.py
        ├── predictions_contact.csv 接地 (Pred_L/R 確率 / GT_L/R) test.py → evaluate.py
        ├── evaluation.txt       評価結果             evaluate.py
        └── visualize_data.js    可視化用データ       visualize.py → visualize.html
```

出力先を変えたい場合 (実験ごとに分けるなど) は `config.yaml` の `paths.output_dir` を変更します。

## 処理の流れ

1. **読み込み** (`dataset/dataset.py`)
   収録 ID (例 `D1S1A1`) ごとにセンサー 5 個とスケルトンを読み込む。机置きデータ (`A0`) は学習・テストのデータには含めない
   (キャリブレーションにだけ使う)。
2. **時刻同期**
   mocap とのずれは収録ごとに 0.4〜1.8 秒程度ある。
   「ジャイロのノルム」と「対応関節の速度」の相互相関から収録ごとにオフセットを推定し、
   センサーをスケルトンの 100 Hz に線形補間して揃える。結果は `outputs/sync/*.json`。
   センサーの時計はデバイスの種類ごとにずれることがある (D2 ではインソール FL / FR だけ約 0.5 秒ずれていた) ため、
   同じ時計のセンサーのまとまり (`sync.groups`, 既定は腕・背中の IMU / インソール) ごとにオフセットを推定し、
   まとまり間の差 (`clock_shift_sec`) をセンサー時刻から取り除いてから揃える。
   推定がうまくいかない場合は `config.yaml` の `sync.offsets` で手動指定できる。
2.5 **キャリブレーション** (`dataset/calibration.py`, `config.yaml` の `calibration`)
   収録の手順として、データ収集の直前に 1 回、全センサーを机に置いた静止記録 (`sensor/calib/D?S?A0_*.csv`) を取り、
   各収録の最初と最後に T ポーズを 3 秒キープする。地磁気 (Mag) は信頼度が低いため使わない。
   - 机置きデータ → ジャイロのバイアスと、圧力の無負荷時のゼロ点をセルごとに引く
     (同じ `D?S?` → 同じ収集回 `D?` の他の被験者の順に探す)
   - T ポーズの検出: 収録の最初 / 最後 10 秒から「全 IMU が静止し、両足に荷重がある」区間を探す。
     腕を下ろした静止と区別するため、学習時は mocap の腕の向き、テスト時 (センサーのみ) は
     重力方向が基準 (学習データの T ポーズの平均) から 45° 以内かで T ポーズと判定する
   - IMU の取り付け向きの補正: T ポーズ中の重力方向 (センサー座標系) を基準の向きに移す最小回転を Gyro / Acc に掛ける。
     Mag を使わないので補正は傾きの 2 自由度だけ (重力軸まわりの向きは補正しない)。
     T ポーズでは体節の長軸が水平なので、バンドが長軸まわりにずれた分は正確に戻せる
   - 圧力の体重基準化: T ポーズ中の左右の圧力の合計 (≒ 体重) を 100 として割る
   - T ポーズが見つからない収録は被験者の平均 (学習時は他の収録、テスト時は `stats.json`) で代用する
   - 結果 (使った区間・補正角・不採用の区間とその理由・最初と最後の T ポーズの差) は `outputs/calib/*.json`。
     設定は `stats.json` に記録され、テスト時に学習時と異なるとエラーになる (設定を変えたら再学習)
3. **前処理**
   関節位置は水平方向 (x, z) のみ CenterOfGravity 基準にする (高さ y は絶対値のまま)。
   被験者 (`D1S1` など) ごとに統計量を計算して正規化し、`seq_len` (200 フレーム = 2 秒) ごとに分割。
   テスト時は学習時の統計量 (`stats.json`) をそのまま使う。
   target: smpl のときは、収録ごとに骨盤の平均の向きが +z (SMPL の正面) になるよう正解 (SMPL と mocap の関節位置) を
   鉛直軸まわりに回す (`dataset.canonical_heading`)。Mag を使わないため絶対的な向きはセンサーから分からず、
   mocap の世界座標では被験者が -z (約 170°) を向いているため、そのままだと骨盤を 180° 反転させた局所解に
   陥ることがある (D2 で発生)。
   また、接地中の足首・つま先の最も低い関節の高さ (前後 10 秒の中央値) を床の高さとして正解の高さから引く
   (`dataset.floor_normalize`, `estimate_floor`)。mocap の床の高さは収録ごとに違い (D2 では約 50 mm)、収録中にもドリフトするため。
   この 2 つの設定は `stats.json` に記録され、テスト時に学習時と異なるとエラーになる (変えたら再学習)。
   回転は W ≥ 0 に符号を揃える (q と −q は同じ回転で、CSV では途中で符号が反転するため)。
   接地ラベルは既定ではインソール圧力の合計から作る (`dataset.contact_source: pressure`, `compute_pressure_contact`):
   足を上げても靴の締め付けで荷重が残るため、収録中の合計の 1 パーセンタイルを基準に、95 パーセンタイルとの間の
   10% を超えたら接地とする。D1 では mocap による従来の判定と κ 0.83 で一致する。
   従来の mocap による判定 (`contact_source: mocap`, `compute_foot_contact`: 足首かつま先が「床から 15 mm 以内」かつ
   「400 mm/s 未満」) は、D2 では mocap の高さが収録中にドリフトし、歩行・ジョグで体の位置がその場に固定されている
   (立脚中の足が後ろへ動く) ため、接地率が 10% 前後になり使えない。
4. **SMPL フィッティング** (`scripts/fit_smpl.py`, target: smpl のとき事前に 1 回)
   BVH (CSV の関節位置を 0.03 mm 以内で再現することを `check_bvh.py` で確認済み) を SMPL にフィッティングする。
   - 初期値: BVH のグローバル回転に、レストポーズの骨の向きの違いを補正する定数回転を掛けて SMPL の回転にする
   - 段階 1: 間引いたフレームで、SMPL と mocap の関節の定義の差 (被験者ごとに一定のオフセット) を推定
   - 段階 2: オフセットを固定して全フレームの姿勢・並進を推定
     (関節位置 + BVH のグローバル回転 (骨軸まわりのねじれを決める) + 膝・肘のヒンジ性など)
   - 前腕の骨軸まわりのねじれ (回内・回外) は BVH の値が信頼できない (可動域を超える・左右で中心がずれる) ため使わず、
     中間位 (T ポーズで手のひらが下) に寄せて可動域内に制限する。手首は BVH の局所回転 (前腕に対する屈曲など) に合わせる。
     手のひらの向きは実測ではなく推定である点に注意 (`config.yaml` の `fit_smpl.weights.arm_ori` などで調整)
   - β は config の値で固定。手 (22, 23) は観測できないので固定値 (指を軽く曲げた形, 予測も同じ値)
5. **モデル** (`models/system.py`)

   ```
   圧力 (左右) ─ PressureEncoder ─┐
                                  ├─ FootFusion (cross-attention) ─ LowerBodyNet (Transformer) ─┐
   IMU (足)   ─ IMUEncoder (GRU) ─┘                                                            │
   IMU (手首・背中) ───────────────────────────────── UpperBodyNet (GRU) ←── 下半身の潜在表現 ┤
                                                                                              │
                                        KinematicFusion (Transformer) ← 上半身 + 下半身 ──────┘
                                                  └─ PoseDecoder → 関節位置 / 回転 / 接地         (skeleton)
                                                     SMPLPoseDecoder → 並進 / 局所回転 (6D) / 接地 (smpl)
   ```
   体型 β は `body_params` として LowerBodyNet / UpperBodyNet に入る。
   SMPL 自体 (順運動学・メッシュ) は学習モデルの外にあり、state_dict には含まれない。
6. **損失**
   - smpl: `SMPLPoseLoss` — 予測を SMPL の順運動学に通し、局所回転・グローバル回転・関節位置 [m]・並進・接地・
     関節速度・接地中の足の速度を正解と比べる (骨長は SMPL で常に正しいので骨長の損失は不要)
   - skeleton: `PoseLoss` (位置 L1・回転 L1・骨長・接地 BCE・平滑化)。骨長は実寸 [mm] に戻してから相対誤差で計算する
   - どちらも + 0.1 × `CycleLoss` (潜在表現からセンサー入力を再構成)
7. **評価**
   - smpl: MPJPE (SMPL 関節, 骨盤をそろえた値も)、PVE (頂点誤差)、局所・グローバル回転誤差、SIP 誤差、
     ジャーク、接地中の足の速さ、元の mocap 関節に対する MPJPE (正解 SMPL 自体の誤差も併記)、接地の正解率・F1
   - skeleton: MPJPE、回転誤差 (クオータニオンの L2 距離と角度差 [deg])、骨長の誤差、接地の正解率・F1

## 補足

- 回転は各関節の親に対する相対回転のクオータニオン (W, X, Y, Z) で、モデル出力の 6 次元の先頭 4 次元に入れ残りを 0 にしている。
  スケルトン CSV は X, Y, Z, W 順で、肘だけヒンジ角が入っているため読み込み時に変換している (data/README.md 参照)。
- `body_params` は被験者の SMPL の β (`config.yaml` の `smpl.subjects`)。被験者が 1 人の間は値が一定なので、
  β による汎化の効果は複数の被験者のデータが揃ってから意味を持つ。
- target: smpl の並進は、従来の位置と同じく水平方向 (x, z) をルート相対 (0) にし、高さ (y) だけ推定している。
- 回転の読み込み修正 (列順・肘) 以前に旧環境 (`d:/20260218P2P-Insole+sparseIMU`) で学習した重みは
  `checkpoints/best_model_quat_v2.pth` として旧環境に残っている (回転の定義が異なるため現在のコードとは互換性がない)。
