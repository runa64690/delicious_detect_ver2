# Delicious Detector Ver2

Face Landmarkerで取得した顔ランドマークの時系列から、飲用時の顔動作を解析するプロジェクトです。現在は、水（`water = 0`）と炭酸水（`sparkling_water = 1`）の二値分類に向けたデータ収集、正規化、手作り特徴量生成、LOSO（Leave-One-Subject-Out）評価までを実装しています。

方法A（21個の手作り特徴量＋木モデル）は実行可能です。方法B（ランドマーク時系列＋1D-CNN）は今後実装予定です。

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
- Dummy、ExtraTrees、Random Forestによる二値分類
- 被験者単位の6-fold LOSO評価とデータリーク検査
- fold別・全体評価、予測、混同行列、特徴量重要度、実行条件の保存
- 全検証済み試行による推論用Random Forestの最終学習・保存・再読込検証
- Enter開始の10秒収集とRandom Forestによるリアルタイム判定

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
│  ├─ excluded_trials.csv            # 手動除外するtrialと理由（Git管理対象）
│  └─ loso_experiment.json           # LOSOモデルと評価条件
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
│  ├─ debug_signals/                  # 任意のデバッグ時系列・グラフ
│  ├─ ml/<run-name>/                  # LOSO予測・評価・混同行列等
│  └─ models/<model-id>/              # 保存済み最終モデルと再現情報
│
├─ scripts/
│  ├─ feature_schema.py               # 分類に使う21特徴の共通定義
│  ├─ validate_dataset.py             # labels/raw CSV検査
│  ├─ normalize_facemesh.py           # 全フレーム正規化
│  ├─ extract_handcrafted_features.py # 21手作り特徴量生成
│  ├─ train_loso_classifier.py        # 方法AのLOSO学習・評価
│  ├─ train_final_random_forest.py    # 推論用Random Forestの最終学習・保存
│  ├─ realtime_predictors.py          # 交換可能なリアルタイムモデル層
│  └─ realtime_pipeline.py            # 1試行の品質検査・正規化・推論・ログ
│
├─ tests/
│  ├─ test_loso_classifier.py         # 分割・除外・再現性等のテスト
│  ├─ test_final_random_forest.py     # モデル保存・再読込のテスト
│  └─ test_realtime_inference.py      # 推論・品質条件・ログのテスト
│
├─ raspberrypi/
│  ├─ pi_send_camera.py               # Raspberry Pi側の映像送信
│  └─ test_leds.py                     # LED動作確認
│
├─ face_landmarker.task               # MediaPipe Face Landmarkerモデル
├─ pc_receive_camera.py               # PC側の受信・FaceMesh表示
├─ realtime_classify.py               # Enter開始の水・炭酸水リアルタイム判定
├─ video_record.py                    # 動画・FaceMesh CSV記録
├─ .gitignore
├─ requirements.txt
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
| scikit-learn | 1.9.0 |
| joblib | 1.6.0 |
| Matplotlib | 3.10.8 |
| OpenCV | 4.11.0.86 |
| MediaPipe | 0.10.33 |

解析スクリプトには主にNumPyとpandasを使用します。撮影処理にはOpenCVとMediaPipeが必要です。デバッググラフを保存する場合はMatplotlibも必要です。

Windows PowerShellでの例：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
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

アンケート項目、被験者ID、飲料名、飲料ラベルは顔特徴量の計算には使用せず、メタデータとしてのみ付与します。分類コードは`feature_schema.py`で明示した21列だけを入力し、アンケート項目やIDが混入した場合でもモデル入力には使いません。

## 5. 方法AのLOSO学習・評価

標準設定は`config/loso_experiment.json`にあります。各foldで1被験者だけをtest、残りの被験者をtrainにし、全被験者が1回ずつtestになる6-fold LOSOです。ハイパーパラメータ探索はまだ行わず、固定条件のDummy、ExtraTrees、Random Forestを同じfoldで比較します。

```powershell
python scripts/train_loso_classifier.py --run-name initial_loso_v1
```

同名の結果を意図的に作り直す場合だけ`--overwrite`を付けます。

```powershell
python scripts/train_loso_classifier.py --run-name initial_loso_v1 --overwrite
```

主評価指標は6 foldのBalanced Accuracyの単純平均です。Accuracy、Precision、Recall、F1、Macro F1、ROC-AUCも保存します。主な出力は次のとおりです。

| 出力 | 内容 |
|---|---|
| `predictions_<model>.csv` | 各試行の正解、予測、クラス確率 |
| `fold_metrics_<model>.csv` | 被験者別の試行数と評価指標 |
| `summary_metrics.csv` | 各指標のfold平均・標準偏差と全予測の集計値 |
| `model_comparison.csv` | 3モデルの主要指標比較 |
| `confusion_matrix_<model>.csv/.png` | 全test予測を集約した混同行列 |
| `feature_importance_*.csv` | 木モデルのfold別・要約特徴量重要度 |
| `fold_assignments.csv` | 各試行をtestにしたfold |
| `dataset_audit.csv` | 被験者別の水・炭酸水試行数 |
| `run_manifest.json` | バージョン、Git状態、入力hash、特徴列、除外、fold条件 |

入力検査では、手動除外trialが`features.csv`へ残っていないこと、特徴量が有限値であること、各被験者に両クラスがあることを確認します。同一被験者がtrainとtestへ入った場合はエラーにします。

テストは次のコマンドで実行できます。

```powershell
python -m unittest discover -s tests -v
```

## 6. 推論用Random Forestの最終学習・保存

LOSOで採用候補をRandom Forestに決めた後、検証済みの全59試行で推論用モデルを学習します。

```powershell
python scripts/train_final_random_forest.py --model-id random_forest_v1
```

同じIDのモデルを意図的に再学習する場合だけ`--overwrite`を付けます。

```powershell
python scripts/train_final_random_forest.py `
  --model-id random_forest_v1 `
  --overwrite
```

出力先は`results/models/random_forest_v1/`です。

| 出力 | 内容 |
|---|---|
| `model.joblib` | 59試行すべてで学習したRandom Forest本体 |
| `metadata.json` | クラス対応、21特徴の順序、パラメータ、バージョン、モデルhash |
| `training_summary.json` | 学習試行数、被験者・ラベル内訳、除外、入力ファイルhash、再読込検証結果 |

保存直後にモデルを再読込し、保存前後で予測ラベルが一致すること、確率が浮動小数点許容差内で一致すること、特徴量名・順序とクラスが一致することを検査します。

このモデルは将来の新規試行を推論するためのものです。全59試行を学習に使うため、このモデル自身を同じ59試行で評価した値は汎化性能になりません。性能の報告には前節のLOSO結果を使用します。`results/`はGit管理対象外であり、学習済みモデルもGitHubには掲載されません。

## 7. Raspberry Piカメラのリアルタイム判定

`realtime_classify.py`は通常時にはラズパイ映像をストリーミング表示し、Enterキーを押した後の10秒間だけFaceMesh座標をメモリへ収集します。収集終了後に、学習時と同じ正規化、21特徴量生成、保存済みRandom Forestによる推論を行います。推論処理は別スレッドで行うため、処理中も映像受信・表示を継続します。

PC側を先に起動します。通常は映像接続元のRaspberry Pi IPをIR・LED制御にも自動使用するため、`--raspi-ip`の指定は不要です。

```powershell
python realtime_classify.py `
  --show-facemesh
```

映像接続元とIR・LED制御先が異なる特殊な構成の場合だけ、`--raspi-ip`を明示します。

```powershell
python realtime_classify.py `
  --raspi-ip 192.168.0.139 `
  --show-facemesh
```

`pc_receive_camera.py`や`video_record.py`も同じ映像ポート`9999`を使うため、同時には起動せず、リアルタイム判定時は`realtime_classify.py`だけを起動してください。また、事前に`results/models/random_forest_v1/model.joblib`が作成済みである必要があります。

続いてRaspberry Pi側で`raspberrypi/pi_send_camera.py`を実行します。同ファイルの`LAPTOP_IP`にはPCのIPアドレスを設定してください。

操作キーは次のとおりです。

| キー | 動作 |
|---|---|
| Enter | 10秒間の1試行収集を開始 |
| `F` | FaceMesh点表示のON/OFF |
| Space | 赤外線照明のON/OFF |
| `Q` | 終了 |

### フルカラーLEDリング

Raspberry Pi側の`pi_send_camera.py`は、`test_leds.py`と同じNeoPixel設定でLEDリングを制御します。

```text
データ端子 : GPIO18 (board.D18)
LED数      : 12
色順       : GRB
明るさ     : 15%
```

| 状態 | LED表示 |
|---|---|
| FaceMeshが顔を認識 | 全体を緑 |
| 顔を未認識 | 消灯 |
| 正規化・特徴量計算・推論中 | リング全体を回る虹色 |
| 水と判定 | 3秒間青、その後は顔認識状態へ戻る |
| 炭酸水と判定 | 3秒間赤、その後は顔認識状態へ戻る |
| 品質不足・計算エラー | 顔認識状態へ戻る |

LED状態はPCから制御ポート`10000`へ送ります。制御接続が一時的に切れても映像送信は継続し、Pi側は次のPC接続を再度待ち受けます。PC側もLED命令の送信失敗だけで映像処理を停止しません。

Raspberry Piには`board`と`neopixel`が必要です。環境によってはGPIO18のNeoPixel制御にroot権限が必要です。

```bash
sudo python raspberrypi/pi_send_camera.py
```

LEDリングには適切な5V電源を使用し、Raspberry PiとGNDを共通にしてください。LEDリング全体をGPIOピンから給電しないでください。

学習時の撮影手順へ合わせ、終了3秒前から`Release cup from mouth`を表示します。標準の品質条件は、FaceMesh有効180フレーム以上かつ成功率70%以上です。満たさない場合は水・炭酸水を無理に出さず、`NO RESULT`として記録します。

```text
収集前       : 映像・FaceMeshを常時表示
Enter        : 10秒収集開始
収集中       : 残り時間、有効FaceMesh数を表示
収集終了     : 正規化・特徴量計算・推論
判定成功     : WATERまたはSPARKLING_WATERと両クラスのscoreを表示
品質不足     : NO RESULTを表示
```

結果は`results/realtime_predictions.jsonl`へ1試行1行で保存します。成功試行には予測、両クラスのscore、21特徴量、フレーム数、FaceMesh成功率、実効FPS、処理時間が含まれます。失敗試行にも除外理由とフレーム数を保存します。モデル出力scoreは、現時点では校正済みの正答確率ではありません。

収集したraw・normalized座標CSVも残す場合は、次のように指定します。

```powershell
python realtime_classify.py `
  --save-trials-dir results/realtime_trials
```

`realtime_predictors.py`の`TrialPredictor`は、正規化済み478点の1試行を受け取る共通インターフェースです。現在の`HandcraftedRandomForestPredictor`はそこから21特徴量を生成します。将来、座標時系列モデルを追加するときは、新しいpredictorが同じ正規化済み試行を受け取って`180×80×2`等へ整形するように実装します。カメラ受信、収集キー、品質検査、ログ処理は変更せず、`load_predictor()`へ新しいモデル種別を追加して切り替えられます。

現在のラズパイ送信コードとの互換性のため、映像受信にはPython pickleを使用しています。信頼できる実験用LAN内だけで使用し、外部ネットワークへ公開しないでください。将来の運用版では長さ付きJPEGバイト列へ置き換える予定です。

## 現在のローカルデータで確認済みの結果

2026年8月31日時点のローカルデータでは、以下を確認しています。実験データ自体はGitHubに含まれないため、clone先のデータによって結果は変わります。

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

固定条件での初回LOSO結果は次のとおりです。主指標はfold平均Balanced Accuracyです。

| モデル | fold平均 Balanced Accuracy | fold標準偏差 | fold平均 Accuracy | fold平均 F1 | fold平均 ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.5583 | 0.1429 | 0.5611 | 0.5475 | 0.6267 |
| ExtraTrees | 0.5250 | 0.1666 | 0.5278 | 0.5415 | 0.6117 |
| Dummy | 0.5000 | 0.0000 | 0.4907 | 0.5556 | 0.5000 |

Random Forestがこの条件では最上位ですが、59試行・6被験者と小規模で、被験者別Balanced Accuracyは0.40～0.80とばらついています。現段階では性能確立ではなく、評価パイプラインの初期動作確認として扱います。モデル選択や特徴選択を行う場合は、test被験者の情報を使わないnested LOSO等が必要です。

推論用`random_forest_v1`は同じ59試行すべてで最終学習済みです。1000本の決定木、21特徴、`random_state = 42`を使用し、保存後の再読込検証にも成功しています。

## 今後の予定

- 方法B用80ランドマーク時系列の生成
- 1D-CNN
- 方法Aと方法Bで共通の試行・除外設定・LOSO foldを使用
- 保存済みRandom Forestによる1試行オフライン推論
- Raspberry Pi実機でのリアルタイム処理速度・通信安定性の検証
- ハイパーパラメータ・特徴選択を行う場合のnested LOSO
- 実効FPS差とmotion特徴の時間基準化の検討

## GitHubへpushする前の確認

```powershell
git status --short
git check-ignore -v data/labels.csv results/features.csv
git diff --check
```

追加対象の例：

```powershell
git add .gitignore README.md requirements.txt config scripts tests
```

`git status`で、動画、顔座標、アンケート、`results/features.csv`が追加対象に含まれていないことを必ず確認してください。

## ライセンス

[MIT License](LICENSE)
