# Delicious Detector Ver2

Face Landmarkerで取得した顔ランドマークの時系列から、飲用時の顔動作を解析するプロジェクトです。現在は、水（`water = 0`）と炭酸水（`sparkling_water = 1`）の二値分類に向けたデータ収集、正規化、手作り特徴量生成までを実装しています。

機械学習モデル、LOSO（Leave-One-Subject-Out）評価、1D-CNNは今後実装予定です。

## 現在実装されている処理

- Raspberry PiカメラからPCへの映像送信
- 約10秒間の動画・FaceMesh座標の記録
- MediaPipe Face Landmarkerによる478点の取得
- ラベル表とraw FaceMesh CSVの検査
- 鼻根中心化、目の水平化、眼間距離によるスケール正規化
- FaceMesh未検出フレームのスキップ
- 可変フレーム数の全フレーム正規化
- 目、口、口角、眉に関する21個の手作り特徴量生成
- 手動除外設定と実行時の自動除外記録
- 特徴量のNaN、Inf、定数列、極端値のsanity check

## プライバシーとGit管理

実験データには被験者ID、顔ランドマーク、動画、アンケート回答が含まれます。そのため、以下は`.gitignore`でGitHubへの掲載対象外にしています。

- `data/`
- `facemesh_csv/`
- `video_data/`
- `results/`
- 仮想環境、Pythonキャッシュ、ログ、IDE設定

`git add -f`でこれらを強制追加しないでください。

一方、`config/excluded_trials.csv`は解析条件を再現するための設定ファイルなので、Gitで管理します。ここには氏名などの直接識別情報を記載しないでください。

## ディレクトリ構成

```text
Delicious_Detecter_Ver2/
├─ config/
│  └─ excluded_trials.csv            # 手動除外するtrialと理由（Git管理対象）
│
├─ data/                              # 実験データ（Git管理対象外）
│  ├─ labels.csv                      # trial、participant、飲料ラベル等
│  ├─ raw_facemesh/                   # P01～P06の正規化前CSV
│  │  ├─ P01_001.csv
│  │  └─ ...
│  └─ normalized_facemesh/            # 全フレーム正規化後CSV
│     ├─ P01_001.csv
│     └─ ...
│
├─ results/                           # 生成結果（Git管理対象外）
│  ├─ features.csv                    # 1行＝1試行、21顔特徴＋メタデータ
│  ├─ excluded_trials.csv             # 手動・自動除外の実行結果
│  └─ debug_signals/                  # 任意のデバッグ時系列・グラフ
│
├─ scripts/
│  ├─ validate_dataset.py             # labels/raw CSV検査
│  ├─ normalize_facemesh.py           # 全フレーム正規化
│  └─ extract_handcrafted_features.py # 21手作り特徴量生成
│
├─ raspberrypi/
│  ├─ pi_send_camera.py               # Raspberry Pi側の映像送信
│  └─ test_leds.py                     # LED動作確認
│
├─ face_landmarker.task               # MediaPipe Face Landmarkerモデル
├─ pc_receive_camera.py               # PC側の受信・FaceMesh表示
├─ video_record.py                    # 動画・FaceMesh CSV記録
├─ .gitignore
├─ LICENSE
└─ README.md
```

`data/`と`results/`はリポジトリに含まれないため、clone直後は自分で作成・配置してください。

## 動作環境

ローカルで確認した環境は以下です。

| ソフトウェア | バージョン |
|---|---:|
| Python | 3.11.9 |
| NumPy | 1.26.4 |
| pandas | 3.0.2 |
| OpenCV | 4.11.0.86 |
| MediaPipe | 0.10.33 |

解析スクリプトには主にNumPyとpandasを使用します。撮影処理にはOpenCVとMediaPipeが必要です。デバッググラフを保存する場合はMatplotlibも必要です。

Windows PowerShellでの例：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy pandas opencv-python mediapipe matplotlib
```

Raspberry Pi側では、環境に応じて次のライブラリも必要です。

- `picamera2`
- `RPi.GPIO`
- `board`
- `neopixel`

Raspberry Pi用パッケージはWindows側の仮想環境へインストールする必要はありません。

## データの準備

### labels.csv

`data/labels.csv`を作成します。現在の処理で使用する主な列は以下です。

```text
trial_id
participant_id
trial_order
sample_code
drink_name
drink_label
delicious
like
sweet
bitter
stimulus
surprise
easy_to_drink
facemesh_csv
```

飲料ラベルは次の対応です。

```text
water           = 0
sparkling_water = 1
```

trial IDとファイル名は、原則として次の形式にします。

```text
trial_id : P01_001
CSV      : P01_001.csv
```

### raw FaceMesh CSV

正規化前CSVをすべて`data/raw_facemesh/`へ配置します。

```text
data/raw_facemesh/P01_001.csv
...
data/raw_facemesh/P06_010.csv
```

raw CSVは1ファイル＝1試行、1行＝1フレームです。主な列は次のとおりです。

```text
video_name, rec_frame, global_frame, timestamp_ms, elapsed_sec, facemesh_ok
lm0_x_norm, lm0_y_norm, lm0_z_norm, lm0_x_px, lm0_y_px
...
lm477_x_norm, lm477_y_norm, lm477_z_norm, lm477_x_px, lm477_y_px
```

## 処理フロー

```text
data/labels.csv
        +
data/raw_facemesh/*.csv
        │
        ▼
scripts/normalize_facemesh.py
        │
        ▼
data/normalized_facemesh/*.csv
        +
config/excluded_trials.csv
        │
        ▼
scripts/extract_handcrafted_features.py
        │
        ├─ results/features.csv
        └─ results/excluded_trials.csv
```

## 1. rawデータの検査

```powershell
python scripts/validate_dataset.py --frames 180
```

このスクリプトは指定された先頭フレーム数が存在するか、必要列、FaceMesh欠損、ラベル構成などを検査します。

現在の解析本体は可変長の全フレームを使用します。したがって、この`--frames 180`は固定長解析の必須条件ではなく、rawデータ品質を確認するための目安です。手動除外対象の`P02_009`は109フレームのため、ここではフレーム不足として表示されます。

## 2. 全フレームの正規化

初回実行：

```powershell
python scripts/normalize_facemesh.py
```

既存の正規化CSVを再生成する場合：

```powershell
python scripts/normalize_facemesh.py --overwrite
```

各FaceMesh検出成功フレームに対して次を行います。

1. 鼻根ランドマーク168を原点にする
2. `(33, 133)`の中点と`(362, 263)`の中点を目中心として使用する
3. 目中心を結ぶ線が水平になるよう回転する
4. 眼間距離で割ってスケールを正規化する
5. y軸を上向き正にする
6. 全478点の`x_local`、`y_local`を保存する

`facemesh_ok != 1`のフレームは正規化CSVへ出力しません。`frame_index`には元CSVの0始まり行番号を保存するため、途中の未検出フレームをスキップした場合は欠番が残ります。

互換・検証目的で先頭N行に制限する場合だけ、次を使用します。

```powershell
python scripts/normalize_facemesh.py --frames 180 --overwrite
```

## 3. 手動除外の設定

`config/excluded_trials.csv`に、解析から除外する試行を記載します。

```csv
trial_id,exclude,reason,notes
P02_009,1,low_effective_fps,Approximately 10.8 FPS; substantially lower than the other trials.
```

新しい除外例：

```csv
P03_005,1,manual_quality_rejection,Describe the reason here.
```

除外を一時的に解除する場合は`exclude`を`0`にします。

```text
除外する   : 1 / true / yes
除外しない : 0 / false / no
```

有効な除外行では`reason`を空欄にできません。

## 4. 手作り特徴量の生成

```powershell
python scripts/extract_handcrafted_features.py
```

正規化CSVに含まれる可変長の全フレームを使用し、1試行から21個の特徴量を生成します。フレーム数を180へ切り詰めたり、paddingや時間リサンプリングを行ったりはしません。

### 21個の顔特徴量

目：

```text
eye_open_mean
eye_open_std
eye_open_min
eye_open_range
eye_motion_mean
blink_count
blink_closed_ratio
```

口の開き：

```text
mouth_open_mean
mouth_open_std
mouth_open_max
mouth_open_range
mouth_open_motion
```

口角：

```text
mouth_corner_mean
mouth_corner_std
mouth_corner_range
mouth_corner_motion
mouth_corner_asymmetry_mean
```

眉と目の距離：

```text
brow_eye_mean
brow_eye_std
brow_eye_range
brow_eye_motion
```

瞬目判定の暫定設定は次のとおりです。

```text
blink_threshold = median(eye_open) × 0.80
最小連続フレーム数 = 2
```

`motion`系特徴は現在、時間微分ではなくフレーム間差分の平均です。

```text
mean(abs(diff(signal)))
```

実効FPS差の影響を受ける可能性があるため、モデル評価時に再検討してください。

### デバッグ出力

特徴量計算の中間時系列を保存する場合：

```powershell
python scripts/extract_handcrafted_features.py --save-debug-signals
```

グラフも保存する場合：

```powershell
python scripts/extract_handcrafted_features.py `
  --save-debug-signals `
  --save-debug-plots
```

出力先は`results/debug_signals/`です。

## 除外処理

### 手動除外

`config/excluded_trials.csv`で指定した試行は特徴量計算前に除外されます。実行結果では`exclusion_source = manual_config`として記録されます。

### 自動除外

以下の場合は、プログラム全体を停止せず、その試行だけを除外して`results/excluded_trials.csv`へ記録します。

| reason | 条件 |
|---|---|
| `file_not_found` | 対応する正規化CSVがない |
| `read_error` | CSVを読み込めない |
| `missing_columns` | 必要なframe/landmark列がない |
| `trial_id_mismatch` | CSV内のtrial IDが期待値と異なる |
| `invalid_frame_index` | frame indexに非数値や空欄がある |
| `duplicate_frame_index` | frame indexが重複している |
| `insufficient_frames` | 有効フレームが2未満 |
| `invalid_coordinates` | 補間後も必要座標にNaN/Infが残る |
| `invalid_signal` | 顔動作時系列を計算できない |
| `invalid_features` | 21特徴にNaN/Infが発生する |
| `processing_error` | その他の予期しないエラー |

FaceMesh未検出はフレーム単位でスキップされるため、それだけで試行全体を除外するわけではありません。

## features.csv

`results/features.csv`は1行＝1試行です。列順は次のとおりです。

```text
trial_id
participant_id
drink_name
drink_label
21 facial features
delicious
like
sweet
bitter
stimulus
surprise
easy_to_drink
```

アンケート項目、被験者ID、飲料名、飲料ラベルは顔特徴量の計算には使用せず、メタデータとしてのみ付与します。今後分類コードを実装する際も、アンケート項目を分類入力へ混入させないでください。

## 現在のローカルデータで確認済みの結果

2026年8月25日時点のローカルデータでは、以下を確認しています。実験データ自体はGitHubに含まれないため、clone先のデータによって結果は変わります。

```text
labels trials     : 60
normalized trials : 60
processed trials  : 59
manual exclusions : 1 (P02_009)
automatic excludes: 0
water trials      : 29
sparkling trials  : 30
facial features   : 21
features.csv      : 59 rows × 32 columns
feature NaN/Inf   : 0
```

`P01_010`はFaceMesh未検出1フレームをスキップし、200 rawフレームから199 normalizedフレームを生成しています。

## 今後の予定

- ExtraTrees／Random Forestによる二値分類
- 被験者単位のLOSO評価
- Accuracy、Precision、Recall、F1、ROC-AUC、Confusion Matrixの保存
- 方法B用80ランドマーク時系列の生成
- 1D-CNN
- 方法Aと方法Bで共通の試行・除外設定・LOSO foldを使用
- 実効FPS差とmotion特徴の時間基準化の検討

## GitHubへpushする前の確認

```powershell
git status --short
git check-ignore -v data/labels.csv results/features.csv
git diff --check
```

追加対象の例：

```powershell
git add .gitignore README.md config scripts
```

`git status`で、動画、顔座標、アンケート、`results/features.csv`が追加対象に含まれていないことを必ず確認してください。

## ライセンス

[MIT License](LICENSE)
