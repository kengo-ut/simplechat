# 課題の実行手順
- `backend`に移動
```bash
cd backend
```
- `.envrc`を作成し、ngrokのトークンを設定します 
```
export NGROK_TOKEN=<your_ngrok_token>
```
- `.envrc`を読み込みます
```bash
direnv allow
```
- 依存関係をインストールします
```bash
uv sync
```
- `backend/app.py`をローカル (Google Colab)で実行し、推論APIを立ち上げます
```bash
uv run python app.py
```
- 出力された公開URLをコピーし、NGROK_URLという環境変数に渡してデプロイします (詳しくは[README-UT20250423.md](README-UT20250423.md)を参照してください)
```bash
NGROK_URL=<https://your-ngrok-url.ngrok.url> cdk deploy
```
