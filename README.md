# P2P-Insole + Sparse IMU (v2)

圧力センサー付きインソール (左右) と、体に付けた少数の IMU から、
モーションキャプチャ相当の全身 18 関節の位置・回転 (クオータニオン) を推定するシステム。

- 入力: 圧力 8×14 = 112 点 ×左右 + IMU (Gyro 3 + Acc 3) ×5 部位 (左右手首・左右足・背中)
- 正解: mocap のスケルトン (100 Hz, 18 関節の位置 + クオータニオン)
- 出力: 18 関節の位置 (CenterOfGravity 基準の相対座標) / 回転 / 左右の足の接地

## 使い方

環境構築は [uv](https://docs.astral.sh/uv/) で行います (依存関係は `pyproject.toml` / `uv.lock`)。
PyTorch は CUDA 13.2 版 (`cu132`) を PyTorch 公式インデックスから取得する設定です。

```bash
uv sync                          # .venv を作成して依存ライブラリをインストール

uv run scripts/prepare_data.py   # (生データから作り直すときのみ) data/raw → data/sensor, data/skeleton
uv run train.py                  # 学習        → outputs/checkpoints/, outputs/sync/train.json
uv run test.py                   # 推論        → outputs/results/predictions_*.csv, outputs/sync/test.json
uv run evaluate.py               # 評価        → outputs/results/evaluation.txt
uv run visualize.py --end 3000   # 3D 可視化   → ブラウザで visualize.html が開く
```

ライブラリを追加するときは `uv add <パッケージ名>` を使います。

すべてのスクリプトは `--config <yaml>` で設定ファイルを切り替えられます (既定: `config.yaml`)。
パスはプロジェクトルート基準で解決されるので、どのディレクトリから実行しても同じ場所に読み書きします。

## フォルダ構成

```
.
├── config.yaml           設定 (入出力パス・同期・学習・モデルのハイパーパラメータ)
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
│   ├── pose_loss.py      位置 / 回転 / 骨長 / 接地 / 平滑化の損失
│   └── cycle_loss.py     潜在表現からセンサー入力を再構成する損失
├── utils/
│   ├── paths.py          設定読み込みと入出力パスの一元管理
│   └── export_csv.py     予測結果の CSV 書き出し
├── scripts/
│   └── prepare_data.py   生データ → 学習用データの変換
│
├── data/                 データ (詳細は data/README.md)
│   ├── raw/              収録したままの生データ (編集しない)
│   ├── sensor/{train,test,calib}/     D1S1A1_AL.csv ...
│   └── skeleton/{train,test}/         D1S1A1_skeleton.csv ...
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
4. **モデル** (`models/system.py`)

   ```
   圧力 (左右) ─ PressureEncoder ─┐
                                  ├─ FootFusion (cross-attention) ─ LowerBodyNet (Transformer) ─┐
   IMU (足)   ─ IMUEncoder (GRU) ─┘                                                            │
   IMU (手首・背中) ───────────────────────────────── UpperBodyNet (GRU) ←── 下半身の潜在表現 ┤
                                                                                              │
                                        KinematicFusion (Transformer) ← 上半身 + 下半身 ──────┘
                                                  └─ PoseDecoder → 関節位置 / 回転 / 接地
   ```
5. **損失** `PoseLoss` (位置 L1・回転 L1・骨長・接地 BCE・平滑化) + 0.1 × `CycleLoss` (潜在表現からセンサー入力を再構成)
   骨長の損失は、位置を正規化統計量で実寸 [mm] に戻してから骨長の相対誤差 |L̂ − L| / L で計算する
   (正規化は関節・軸ごとにスケールが違うため、正規化空間の距離は骨長として意味を持たない)。
6. **評価** MPJPE (関節位置誤差 [mm])、回転誤差 (クオータニオンの L2 距離と角度差 [deg], いずれも q と −q を区別しない)、
   骨長の誤差 [mm / %]、接地の正解率・F1 を出力。位置と回転は部位別・関節別にも出す。

## 補足

- 回転は各関節の親に対する相対回転のクオータニオン (W, X, Y, Z) で、モデル出力の 6 次元の先頭 4 次元に入れ残りを 0 にしている。
  スケルトン CSV は X, Y, Z, W 順で、肘だけヒンジ角が入っているため読み込み時に変換している (data/README.md 参照)。
- `body_params` (体型) は現状ダミー (0 埋め) で、学習には実質使われていない。
- 回転の読み込み修正 (列順・肘) 以前に旧環境 (`d:/20260218P2P-Insole+sparseIMU`) で学習した重みは
  `checkpoints/best_model_quat_v2.pth` として旧環境に残っている (回転の定義が異なるため現在のコードとは互換性がない)。
