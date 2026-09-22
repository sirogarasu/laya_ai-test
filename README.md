# Laya ローカル GUI

Laya の多言語モデルを NVIDIA GPU で動かし、ブラウザから問い合わせ文を判定します。WSL 2 上の Docker Compose と、コンテナから利用できる NVIDIA GPU が必要です。

## 起動

```sh
docker compose up --build
```

モデルの読み込みが終わったら、ホストのブラウザで <http://localhost:17860> を開きます。文章を入力して「判定」を押すと、担当部署、返金希望スコア、詳細 JSON が表示されます。画面はローカルホストにだけ公開されます。終了するときはターミナルで Ctrl+C を押します。

初回は Hugging Face からモデルをダウンロードします。モデルは `laya_models` ボリュームに保存され、再起動後も再利用されます。通信が途中で切れた場合は最大4回再試行します。

担当部署は `billing`、`technical`、`sales` の3種類です。返金希望スコアや信頼度はモデルの出力であり、重要な処理を自動実行する前に結果を確認してください。判定項目は `docker/app.py` の `SCHEMA` で変更できます。

## 補足

`docker/example.py` は以前の1件だけ判定するサンプルとして残しています。記事の高速な推論値は Apple Silicon の MLX を使った結果で、この構成の CUDA 推論とは実行環境が異なります。

参考: [Laya 公式リポジトリ](https://github.com/NandhaKishorM/laya)
