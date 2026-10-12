# GitHub公開前の確認

[READMEへ戻る](../README.md)

## 公開するもの・手元に残すもの

ソースコード、テスト、依存パッケージ定義、説明資料、評価条件、空の除外設定ひな形、公開用に選んだ写真を公開します。

以下はローカルに残し、Gitの追跡対象から外します。

- `data/`、`facemesh_csv/`などの顔座標・アンケート・参加者情報
- `video_data/`とその他の録画ファイル
- `results/`の予測・履歴・特徴量・評価出力・学習済みモデル
- `config/excluded_trials.csv`の実際の除外試行と理由（公開版は`config/excluded_trials.example.csv`）
- `face_landmarker.task`などの取得したモデル
- 仮想環境、キャッシュ、ログ、環境変数、鍵、端末固有の設定

`git rm --cached`による除外はローカルファイルを削除しません。Git上では削除として記録されます。`.gitignore`の対象を`git add -f`で追加しないでください。

## このリポジトリの過去履歴について

**2026-10-12の確認で、過去のコミットに`data/facemesh_landmarks.csv`が存在しています。現在のファイル一覧から除外しても、その履歴を含めて公開すると過去のCSVを取得できます。**

確認コマンド：

```powershell
git log --all --oneline -- data/facemesh_landmarks.csv
git log --all --format= --name-only
```

過去の除外設定、モデル、端末のIPアドレスも履歴に残ります。この整備では既存履歴を書き換えていません。現在のブランチをそのままpushするだけでは、過去データの公開対策は完了しません。

### 公開用に履歴を引き継がない場合

開発履歴は非公開で保管し、確認済みの最新ソースだけを新しい公開用リポジトリの最初のコミットにします。公開範囲を確認した変更をローカルでコミットした後、`git archive --format=zip --output=../delicious-detector-public.zip HEAD`で追跡対象だけを書き出せます。

出力ZIPを開いて不要なファイルがないことを確認し、別のフォルダへ展開して新しくGitを初期化します。元フォルダの`.git`や実験データはコピーしません。新しい公開先をREADMEのclone URLにも反映してください。元リポジトリの公開設定を切り替える方法では古い履歴が公開されるため、この方法とは異なります。

### 同じリポジトリ・履歴を使う場合

履歴をバックアップしたうえで、全ブランチ・タグに含まれる実験データなどを調査し、履歴から取り除く作業が別途必要です。コミットIDの変更とリモートへの強制更新は共同作業者にも影響します。通常の削除コミットでは代替できません。

既に公開済みの場合は、新しいリポジトリを作るだけで既存の公開履歴や取得済みコピーが消えるわけではありません。

## 今回の変更を確認する

```powershell
git status --short
git diff --check
git diff --cached --name-status
git ls-files -ci --exclude-standard
git check-ignore -v data/labels.csv results/features.csv video_data/example.mp4 face_landmarker.task config/excluded_trials.csv
```

`git ls-files -ci --exclude-standard`は、除外ルールに一致するのに追跡が残っているファイルの検査です。何も表示されない状態にします。これだけでは過去履歴や秘密情報の全件検査にはなりません。

変更内容を確認して追加する例：

```powershell
git add .gitignore README.md requirements.txt requirements-dev.txt LICENSE config/loso_experiment.json config/excluded_trials.example.csv docs raspberrypi scripts tests web web_app.py realtime_classify.py pc_receive_camera.py video_record.py
git diff --cached --stat
git diff --cached --check
git diff --cached
```

追加済みの削除も含め、データ・動画・モデルが追加されていないことを確認してください。写真を追加する場合は[写真のガイド](images/README.md)も参照してください。
