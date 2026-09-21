# Laya Docker 環境

記事で紹介された Laya の判断モデルを、WSL 2 上の NVIDIA GPU を使い、PyTorch の CUDA 対応版で動かします。MLX は使用しません。

## 実行

NVIDIA GPU をコンテナに公開できる Docker Compose が使える WSL 2 ホストで、このディレクトリから実行します。`docker/` に Dockerfile とコンテナ内で実行するサンプルを置いています。

```sh
docker compose up --build
```

初回は Hugging Face から多言語モデルをダウンロードします。モデルは `laya_models` ボリュームに保存され、コンテナを作り直しても再利用されます。サンプルは日本語の問い合わせを分類します。

サンプルを編集した後に再実行する場合は、同じコマンドでイメージを再ビルドしてください。

## 注意

記事の高速な推論値は Apple Silicon の MLX を使った結果です。この構成は RTX 4070 Ti SUPER の CUDA 推論です。MLX の測定値とは実行環境が異なります。GPU をコンテナに公開できない場合、サンプルはエラーを出して終了します。Docker Desktop を使う場合は WSL 2 バックエンドと NVIDIA の Windows ドライバーが必要です。

参考: [Laya 公式リポジトリ](https://github.com/NandhaKishorM/laya)、[紹介記事](https://zenn.dev/mizchi/articles/laya-mlx-60fps)
