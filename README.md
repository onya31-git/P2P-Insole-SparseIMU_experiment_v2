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

## フォルダ構成

```
.
├── config.yaml           設定 (入出力パス・同期・SMPL・学習・モデルのハイパーパラメータ)
├── train.py              学習
├── test.py               推論 (テストデータ) → 予測 CSV
├── evaluate.py           予測 CSV から MPJPE などを計算
├── visualize.py          予測と正解のスケルトンをブラウザで表示 (visualize.html を使用)
├── visualize.html
├── pyproject.toml        依存ライブラリ (uv)
├── uv.lock
│
├── dataset/
│   └── dataset.py        CSV 読み込み・センサーと mocap の時刻同期・正規化・チャンク分割
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
    │   └── stats.json         正規化統計量         train.py → test.py
    ├── sync/
    │   ├── train.json         時刻同期の推定結果   train.py
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
   収録 ID (例 `D1S1A1`) ごとにセンサー 5 個とスケルトンを読み込む。キャリブレーション (`A0`) は除外。
2. **時刻同期**
   センサー同士は共通時計だが、mocap とのずれは収録ごとに 0.4〜1.8 秒程度ある。
   「ジャイロのノルム」と「対応関節の速度」の相互相関から収録ごとにオフセットを推定し、
   センサーをスケルトンの 100 Hz に線形補間して揃える。結果は `outputs/sync/*.json`。
   推定がうまくいかない場合は `config.yaml` の `sync.offsets` で手動指定できる。
3. **前処理**
   関節位置は水平方向 (x, z) のみ CenterOfGravity 基準にする (高さ y は絶対値のまま)。
   被験者 (`D1S1` など) ごとに統計量を計算して正規化し、`seq_len` (200 フレーム = 2 秒) ごとに分割。
   テスト時は学習時の統計量 (`stats.json`) をそのまま使う。
   回転は W ≥ 0 に符号を揃える (q と −q は同じ回転で、CSV では途中で符号が反転するため)。
   接地ラベルは mocap から作る: 左右それぞれ足首かつま先が「床から 15 mm 以内」かつ「400 mm/s 未満」なら接地
   (`compute_foot_contact`)。閾値はインソール圧力の合計との一致度が最も高くなるよう決めた (一致率 91〜94%)。
4. **SMPL フィッティング** (`scripts/fit_smpl.py`, target: smpl のとき事前に 1 回)
   BVH (CSV の関節位置を 0.03 mm 以内で再現することを `check_bvh.py` で確認済み) を SMPL にフィッティングする。
   - 初期値: BVH のグローバル回転に、レストポーズの骨の向きの違いを補正する定数回転を掛けて SMPL の回転にする
   - 段階 1: 間引いたフレームで、SMPL と mocap の関節の定義の差 (被験者ごとに一定のオフセット) を推定
   - 段階 2: オフセットを固定して全フレームの姿勢・並進を推定
     (関節位置 + BVH のグローバル回転 (骨軸まわりのねじれを決める) + 膝・肘のヒンジ性など)
   - β は config の値で固定。手 (22, 23) は観測できないので単位回転
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
