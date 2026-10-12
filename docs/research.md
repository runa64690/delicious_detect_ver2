# データ準備・学習・評価

[READMEへ戻る](../README.md)

コマンドはリポジトリのルートで実行してください。Windowsでは先に`.\.venv\Scripts\Activate.ps1`で仮想環境を有効にします。セットアップは[README](../README.md)を参照してください。実験データと学習済みモデルは公開リポジトリに含みません。

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

現在の解析本体は可変長の全フレームを使用します。したがって、この`--frames 180`は固定長解析の必須条件ではなく、rawデータ品質を確認するための目安です。

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

`config/excluded_trials.example.csv`を`config/excluded_trials.csv`へコピーし、解析から除外する試行を記載します。既存の設定がある場合は上書きしないでください。実際の試行IDと除外理由はローカルだけで管理します。以下は書式例です。

```csv
trial_id,exclude,reason,notes
EXAMPLE_001,1,low_effective_fps,Describe the quality issue here.
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

LOSOで採用候補をRandom Forestに決めた後、検証済みの全試行で推論用モデルを学習します。

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
| `model.joblib` | 全試行で学習したRandom Forest本体 |
| `metadata.json` | クラス対応、21特徴の順序、パラメータ、バージョン、モデルhash |
| `training_summary.json` | 学習試行数、被験者・ラベル内訳、除外、入力ファイルhash、再読込検証結果 |

保存直後にモデルを再読込し、保存前後で予測ラベルが一致すること、確率が浮動小数点許容差内で一致すること、特徴量名・順序とクラスが一致することを検査します。

このモデルは将来の新規試行を推論するためのものです。全試行を学習に使うため、このモデル自身を同じ学習用試行で評価した値は汎化性能になりません。性能の報告には前節のLOSO結果を使用します。`results/`はGit管理対象外であり、学習済みモデルもGitHubには掲載されません。


## 追加の評価

[LOSO置換検定とグラフ作成](permutation_test_loso.md)を参照してください。
