# デモの操作・通信・保存仕様

[READMEへ戻る](../README.md)

最初に[READMEのセットアップ](../README.md#セットアップ)を完了してください。コマンドはリポジトリのルートで実行します。IPアドレスは説明用の例です。

## PCウィンドウで操作する

`realtime_classify.py`は通常時にはラズパイ映像をストリーミング表示し、Enterキーを押した後の10秒間だけFaceMesh座標をメモリへ収集します。収集終了後に、学習時と同じ正規化、21特徴量生成、保存済みRandom Forestによる推論を行います。推論処理は別スレッドで行うため、処理中も映像受信・表示を継続します。

PC側を先に起動します。通常は映像接続元のRaspberry Pi IPをIR・LED制御にも自動使用するため、`--raspi-ip`の指定は不要です。

```powershell
python realtime_classify.py `
  --show-facemesh
```

映像接続元とIR・LED制御先が異なる特殊な構成の場合だけ、`--raspi-ip`を明示します。

```powershell
python realtime_classify.py `
  --raspi-ip 192.168.1.30 `
  --show-facemesh
```

`pc_receive_camera.py`とWeb版も映像ポート`9999`を使うため、同時には起動しないでください。収集用の`video_record.py`はポート`50000`を使います。収集時はPi側にも`--port 50000`を指定してください。また、事前に`results/models/random_forest_v1/model.joblib`が作成済みである必要があります。

続いてRaspberry Pi側で`sudo python raspberrypi/pi_send_camera.py --pc-host 192.168.1.20`を実行します。IPアドレスはPCのIPv4アドレスに置き換えてください。

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
sudo python raspberrypi/pi_send_camera.py --pc-host 192.168.1.20
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

## Androidから操作する

同じWi-Fiに接続したAndroidのChromeで、ライブ映像、1回ごとの飲料判定、録画、履歴、LED・赤外線照明を操作できます。PCは起動したまま使用します。Androidへのアプリのインストールは不要で、PCブラウザでも同じ画面を利用できます。

### 初回の準備

PCのプロジェクトディレクトリで依存パッケージを更新します。

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

録画用FFmpegは、PATH上の実行ファイルを優先し、見つからない場合は`imageio-ffmpeg`同梱版を使います。特定の実行ファイルを使う場合は`--ffmpeg C:\tools\ffmpeg.exe`を指定します。起動時にH.264エンコード、学習済みモデル、Face Landmarker、保存先への書き込みを確認します。

Pi側には更新した`raspberrypi/pi_send_camera.py`と`raspberrypi/led_control.py`の**両方**を同じディレクトリに配置してください。GPIO・カメラ用の依存パッケージは従来と同じです。

### 起動手順

1. PCで`ipconfig`を実行し、Wi-FiアダプターのIPv4アドレスを確認します（例：`192.168.1.20`）。
2. PCでWebサーバーを起動します。

```powershell
.\.venv\Scripts\python.exe web_app.py
```

3. PiでPCのIPv4アドレスを指定して起動します。

```bash
sudo python raspberrypi/pi_send_camera.py --pc-host 192.168.1.20
```

4. AndroidのChromeで`http://192.168.1.20:8000`を開きます。PC自身からは`http://127.0.0.1:8000`で開けます。

Webサーバーと既存の`realtime_classify.py`は同時起動できません。同じ映像受信ポートを使うため、先に起動したものがカメラを使用します。Webサーバーは必ず1プロセスで動作させてください。終了はPCのターミナルで`Ctrl+C`、Piでも`Ctrl+C`です。

| 通信 | 接続先・既定ポート |
|---|---|
| Android → PC | TCP 8000（操作画面・映像・動画ダウンロード） |
| Pi → PC | TCP 9999（カメラ映像） |
| PC → Pi | TCP 10000（LED・IR制御） |

Windowsで接続できない場合は、実験用Wi-Fiをプライベートネットワークに設定し、Windows Defenderファイアウォールの受信規則でPCのTCP 8000と9999を許可します。設定例は、管理者PowerShellで次を実行します。この操作はサーバーから自動実行しません。

```powershell
New-NetFirewallRule -DisplayName "Delicious Detector LAN" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8000,9999 -Profile Private -RemoteAddress LocalSubnet
```

ゲストWi-Fiの端末間通信禁止（AP isolation）が有効だと接続できません。Piのファイアウォールを使用している場合は、PCからTCP 10000への接続も許可します。インターネットへのポート転送は不要です。認証のない実験用LAN向け機能であり、既存のpickle映像通信も使うため、信頼できる端末だけが接続するLAN内で使用してください。

### 画面の使い方

- **判定を開始**：10秒間の映像を自動保存し、既存と同じ品質条件・モデルで判定します。収集中と推論中の重複開始は受け付けません。終了3秒前にコップを口から離す案内を表示します。
- **任意録画を開始／停止**：判定とは独立した動画を作ります。判定と重なっても、手動録画と判定用動画は別ファイルです。
- **FaceMesh**：ライブ映像への点表示を切り替えます。保存動画には点や判定文字を焼き込みません。
- **履歴**：判定結果・両クラスのスコア・品質・失敗理由・動画の保存状態を確認します。動画の「再生」「ダウンロード」と、履歴のCSV／JSONL出力を利用できます。CSVはExcelで開きやすいUTF-8 BOM付きです。スコアは正答率ではありません。
- **LED**：自動モードでは従来の状態表示、手動モードでは指定色を維持します。「LEDに適用」で設定を送り、「消灯」は手動の黒に設定します。自動へ戻す場合も「LEDに適用」を押してください。明るさは実際の出力0～15%の範囲です。再起動時は自動・15%・IR OFFで始まります。
- **制御状態**：「送信中」は応答待ち、「適用済み」はPi側の書き込み完了、「制御未接続」は切断・タイムアウト・デバイスエラーです。IRは明示的なON/OFFなので再送で反転しません。

ブラウザを閉じたりAndroidがスリープしたりしても、PCで進行中の判定・録画は続きます。再度開くと現在の状態を表示します。任意録画は明示的に停止してください。操作画面は複数端末で共有され、主に1台から操作することを想定しています。

### 保存と切断時の動作

| 保存内容 | 既定の保存先 |
|---|---|
| 加工前の動画（H.264/MP4、音声なし、30fps） | `video_data/realtime/video_<ID>.mp4` |
| 判定結果（既存形式に`video_id`を追加） | `results/realtime_predictions.jsonl` |
| 録画と試行の開始・終了・中断記録 | `results/realtime_journal.jsonl` |
| 動画エンコードのエラー情報 | 動画と同じ場所の`.ffmpeg.log` |

ライブ表示は640×400・最大10fpsですが、判定入力を10fpsに制限するものではありません。録画は受信解像度を保持し、実時間に合わせて録画用フレームだけ補完・間引きします。受信／推論／録画／配信を分離し、遅い閲覧端末が判定を止めない構成です。

Piの映像が切れた場合は収集中の試行を中断として記録し、受信済みの動画を終了します。収集完了後の推論は継続します。PC・Piは映像と制御を再接続し、次の判定は利用者が開始します。制御接続が切れるとPiのLED・IRは消灯します。再接続後、PCで指定されている設定を再適用します。

録画のエンコード失敗やバッファ不足は判定結果と別に履歴へ表示します。判定ログ自体の書き込みに失敗した場合は保存エラーを表示し、新規開始を止めます。保存先を修復してサーバーを再起動してください。録画開始には空き容量100MB以上が必要です。

正常終了では進行中の録画を閉じます。強制終了や停電で残った未完了の試行・録画は次回起動時に中断扱いになります。`.partial.mp4`は未完成ファイルであり、再生・ダウンロード一覧には公開しません。保存ファイルは自動削除せず、実験データとしてGit管理対象外のディレクトリに置きます。

保存先・ポートを変更する例：

```powershell
.\.venv\Scripts\python.exe web_app.py `
  --web-port 8000 --port 9999 --control-port 10000 `
  --video-dir video_data/realtime `
  --log results/realtime_predictions.jsonl `
  --journal results/realtime_journal.jsonl
```

従来のPCウィンドウで操作する場合は`python realtime_classify.py`を使います。キー操作は従来どおりです。`--record-video`を付けるとこちらでも判定区間の自動録画を有効にできます。両方の起動方法で同じ収集・推論処理を使います。

### APIと検証

API仕様はサーバー起動中の`http://<PCのIP>:8000/docs`から参照できます。書き込みリクエストは`Content-Type: application/json`が必須です。

| メソッド・パス | 用途 |
|---|---|
| `GET /api/state` | 接続、収集、録画、判定、LED状態 |
| `GET /stream.mjpg` | 最新フレームのMJPEG配信 |
| `POST /api/trials` | 1試行開始（202、重複・未接続は409） |
| `POST /api/recordings` / `POST /api/recordings/stop` | 任意録画の開始／停止 |
| `PUT /api/led` | `{ "mode": "manual", "rgb": [0, 0, 255], "brightness": 0.15 }` |
| `PUT /api/ir` / `PUT /api/view` | `{ "enabled": true }`でIR／FaceMesh設定 |
| `GET /api/history?offset=0&limit=50` | 既存ログを含む履歴 |
| `GET /api/videos/<ID>?download=true` | 完成済み動画のダウンロード（Range対応） |
| `GET /api/logs/csv` / `GET /api/logs/jsonl` | 動画情報を含めた履歴の出力 |

Piの制御ポートは、従来の改行区切りLED命令に加え、ID付きJSONの`SET_STATE`と成功／失敗応答に対応します。旧命令には応答を返さないため、元の`pc_receive_camera.py`とも併用できます（同時起動はしません）。

自動テスト：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Android・Pi実機での確認項目：

1. Chromeで映像が表示され、FaceMeshを切り替えられる。通常の実験用Wi-Fiで映像遅延1秒以内を目標に、実際の動きと見比べる。
2. 判定開始後10秒で結果または品質不足が表示され、動画が約10秒で再生・ダウンロードできる。
3. 任意録画中にも判定でき、独立した2本の動画が保存される。
4. ページ再読み込み・Androidのスリープ後もPCで処理が続き、状態が復元される。
5. LEDの手動色・明るさ・消灯・自動復帰、IR、制御応答表示を確認する。
6. Wi-Fi切断・Pi再起動後、試行が中断として残り、再接続後に新しい判定を開始できる。
7. CSV／JSONL出力と、サーバー再起動後の過去履歴を確認する。
