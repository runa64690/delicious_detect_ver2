# LOSO交差検証の置換検定と論文用グラフ

`scripts/permutation_test_loso.py` は既存の21特徴量・被験者単位LOSOを使い、ラベル置換ごとに全foldを再学習します。既存のスコア一覧だけをシャッフルする処理ではありません。入力は交差検証に使用した `features.csv` と設定ファイルです。追加の依存パッケージは不要です。

## 実行

プロジェクトのルートで実行します。過去の実験を検定する場合は、その実験の設定スナップショットと同一の特徴量・除外設定を使用してください。現在の設定と保存済み設定が異なる場合、再計算した実測スコアも異なります。

```powershell
.\.venv\Scripts\python.exe scripts/permutation_test_loso.py `
  --config results/ml/initial_loso_v1/config_snapshot.json `
  --n-permutations 9999 --seed 42 --jobs 4 `
  --output results/ml/permutation_loso_paper
```

既定は全モデル、設定の `primary_metric`（現在はBalanced Accuracy）、999回です。上記は9,999回で最小p値が0.0001になります。回数は結果を見る前に決め、有意になるまで追加する運用は避けてください。計算量は「モデル数 × 被験者数 ×（置換回数 + 1）」回の学習で、1000本の木を使用する設定では長時間かかります。`--jobs` は置換を並列化し、各モデル内部の並列数は1に固定します。

動作確認は別フォルダで実行します（この結果は論文には使用しません）。

```powershell
.\.venv\Scripts\python.exe scripts/permutation_test_loso.py `
  --n-permutations 3 --jobs 2 --output results/ml/permutation_loso_smoke
```

中断後は同じコマンドに `--resume` を追加します。各置換結果をCSVへ逐次保存し、保存済みの最後の行の次から再開します。モデル・データ・設定・コード・ライブラリの変更を検出すると再開を拒否します。計算途中の行が破損している場合も黙って使用せず停止します。既存の非空フォルダへの新規実行は拒否します。

図だけ再生成する場合は、元データや再学習は不要です。

```powershell
.\.venv\Scripts\python.exe scripts/permutation_test_loso.py `
  --output results/ml/permutation_loso_paper --plot-only --width-mm 85 --dpi 600
```

既定は横幅180 mm、最大3パネル横並びです。85 mmなど150 mm未満では縦に配置します。図中の文字は英文で、PDFはTrueTypeフォントを埋め込み、SVGは編集可能な文字を保持します。投稿先の図幅・フォント・形式指定に合わせて再出力してください。

## 検定の定義と前提

- 統計量：各被験者をtestにしたfoldスコアの**単純平均**。試行数で重み付けしたpooledスコアとは異なります。`--metric` で変更できますが、主指標は事前に決めてください。
- 帰無分布：被験者内で試行ラベルをランダム置換し、各被験者のクラス数を保ちます。1回の置換で全データのラベルを一度だけ入れ替え、その同じラベルを全foldの学習・評価に使用します。置換は復元抽出で、同一置換や元ラベルと同じ並びも除外しません。
- p値：高いスコアほど良いという片側検定です。`p = (C + 1) / (B + 1)`、Cは実測値以上の置換スコア数、Bは置換回数。同点を含み、浮動小数点の丸め差を許容し、p=0とは報告しません。
- 多重性：その実行に含めた全モデル・単一指標を1つの検定ファミリーとしてHolm補正します。`p_raw` と `p_holm` を両方保存します。複数の指標や別実行で比較したモデルまでは補正されません。図はモデル間の優劣の差を検定するものではありません。
- 交換可能性：帰無仮説の下で、指定ブロック内の試行ラベルを交換できることが前提です。実験順序、セッション、対応ペアなどに制約がある場合、単純な被験者内置換は適切でないことがあります。必要な列を入力CSVに追加し、例として `--block-columns session_id pair_id` を指定すると「被験者 × セッション × ペア」内に制限できます。ブロック列はモデル特徴量にはなりません。時系列相関や割付方法によっては、実験計画に合わせた別の置換方式が必要です。
- 特徴量定義、除外、モデル、ハイパーパラメータは固定済みとします。このコードはデータを見て行った特徴選択・モデル探索を再現しません。それらを行った場合は、選択処理を学習fold内かつ各置換内で繰り返す評価設計が必要です。既存LOSO結果を見てRandom Forestだけを選んだ場合、その単独の未補正p値を事前指定モデルの確証的検定として報告しないでください。
- 帰無分布の広がりは汎化性能の信頼区間ではありません。有意差の有無だけで実用性や新規被験者への性能確立を主張しないでください。

モデル名は `dummy extra_trees random_forest`、対応指標は `accuracy balanced_accuracy precision recall f1 macro_f1 roc_auc` です。モデルの乱数種を固定し、置換番号から独立した乱数系列を生成するため、並列数と中断・再開によらず同じ置換を再現できます。

## 出力

| ファイル | 内容 |
|---|---|
| `permutation_test.pdf` / `.svg` / `.png` | 帰無分布・実測スコア・p値。PNGは既定600 dpi |
| `permutation_summary.csv` | 実測値、帰無分布平均・標準偏差、超過数、未補正・Holm補正p値 |
| `permutation_scores.csv` | 全置換の番号とモデル別スコア。再開用チェックポイント |
| `permutation_manifest.json` | 設定、データ・コードのSHA-256、乱数種、ライブラリ版、完了状態 |
| `observed_fold_metrics_*.csv` | 実測ラベルのfold別スコア |
| `observed_predictions_*.csv` | 実測ラベルのout-of-fold予測 |
| `dataset_audit.csv` / `fold_assignments.csv` | クラス数とLOSO分割の監査 |
| `permutation_blocks.csv` | 試行ごとの交換可能性ブロック |
| `figure_caption.txt` | 実際の試行数・被験者数・置換回数を含む英文キャプション案と前提 |

検定の方法は [scikit-learnのpermutation_test_score](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.permutation_test_score.html) と同じラベル置換・上側p値の定義に基づきます。手法の引用候補：Ojala, M. & Garriga, G. C. (2010). [Permutation Tests for Studying Classifier Performance](https://www.jmlr.org/papers/v11/ojala10a.html). *Journal of Machine Learning Research*, 11, 1833–1863。
