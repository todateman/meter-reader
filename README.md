# メーター数値読み取りアプリ

アナログメーターと7セグメントデジタルメーターの画像から数値を自動で読み取るWebアプリケーションです。  
Claude APIのビジョン機能を活用して高精度な画像認識を実現しています。

## 機能

- **アナログメーター読み取り**: 針の位置から数値を自動読み取り
- **7セグメントデジタルメーター読み取り**: デジタル表示の数値を自動読み取り
- **ドラッグ&ドロップ対応**: 簡単に画像をアップロード
- **レスポンシブデザイン**: PC・タブレット・スマートフォン対応
- **詳細情報表示**: 読み取り結果の信頼度や詳細情報を表示

## スクリーンショット

```
[メイン画面]
- メータータイプ選択
- 画像アップロードエリア
- 解析結果表示
```

## 技術スタック

- **バックエンド**: Python 3.8+, Flask
- **画像認識**: Claude API (Anthropic)
- **フロントエンド**: HTML5, CSS3, JavaScript (Vanilla)
- **画像処理**: Pillow

## 必要要件

- Python 3.8以上
- Anthropic APIキー（[Anthropic Console](https://console.anthropic.com/)で取得）
- インターネット接続（Claude API呼び出しのため）

## セットアップ

### 1. リポジトリのクローン

```bash
cd meter-reader
```

### 2. 仮想環境の作成（推奨）

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# または
.venv\Scripts\activate  # Windows
```

### 3. 依存パッケージのインストール

```bash
uv pip install -r requirements.txt
```

### 4. 環境変数の設定

`.env.example`をコピーして`.env`ファイルを作成し、APIキーを設定します。

```bash
cp .env.example .env
```

`.env`ファイルを編集:

```env
ANTHROPIC_API_KEY=your_actual_api_key_here
FLASK_ENV=development
FLASK_DEBUG=True
MAX_CONTENT_LENGTH=10485760
UPLOAD_FOLDER=static/uploads
ALLOWED_EXTENSIONS=jpg,jpeg,png

# Claude APIモデル設定
# 推奨: claude-sonnet-4-5 (精度と速度のバランス)
# 高精度: claude-opus-4-5-20251101 (高コスト)
# 高速: claude-haiku-4-5 (精度は劣る)
CLAUDE_MODEL=claude-sonnet-4-5
CLAUDE_MAX_TOKENS=1024
CLAUDE_TIMEOUT=30
```

**モデルの選択について:**
- `claude-sonnet-4-5`: 精度と速度のバランスが良い（推奨）
- `claude-opus-4-5-20251101`: 最高精度だが高コスト（精度重視の場合）
- `claude-haiku-4-5`: 高速・低コストだが精度は劣る（テスト用）

### 5. アプリケーションの起動

```bash
python app.py
```

ブラウザで http://localhost:5000 にアクセスしてください。

## 使い方

1. **メータータイプを選択**: アナログメーターまたは7セグメントデジタルメーターを選択
2. **画像をアップロード**:
   - ドラッグ&ドロップで画像をアップロード
   - またはクリックしてファイル選択
3. **解析開始**: 「解析開始」ボタンをクリック
4. **結果確認**: 読み取った数値と詳細情報が表示されます

### 対応画像形式

- JPG/JPEG
- PNG
- 最大ファイルサイズ: 10MB
- 推奨解像度: 50x50ピクセル以上

### ヒント

- **アナログメーター**: メーター全体が写るように撮影してください
- **デジタルメーター**: 数字がはっきり見えるように撮影してください
- **照明**: 明るい場所で、反射や影がない状態で撮影すると精度が向上します
- **角度**: 正面から撮影するのが最適です

## API仕様

### POST /api/analyze

メーター画像を解析します。

**リクエスト:**

```http
POST /api/analyze
Content-Type: multipart/form-data

image: <画像ファイル>
meter_type: "analog" または "digital_7segment"
```

**レスポンス（成功時）:**

```json
{
  "success": true,
  "data": {
    "value": 123.45,
    "unit": "kWh",
    "confidence": "high",
    "meter_type": "analog",
    "details": {
      "scale_range": "0-200",
      "needle_position": "約62%の位置",
      "notes": "メーターは良好な状態です"
    }
  },
  "timestamp": "2026-01-28T10:30:00Z"
}
```

**レスポンス（エラー時）:**

```json
{
  "success": false,
  "error": {
    "code": "NO_METER_DETECTED",
    "message": "メーターを検出できませんでした",
    "details": "画像が不鮮明か、メーターが写っていない可能性があります"
  }
}
```

### GET /api/health

ヘルスチェックエンドポイント。

**レスポンス:**

```json
{
  "status": "ok",
  "timestamp": "2026-01-28T10:30:00Z"
}
```

## エラーコード

| コード | 説明 | 対処法 |
|--------|------|--------|
| `NO_FILE` | ファイルが選択されていない | 画像ファイルを選択してください |
| `INVALID_FILE_TYPE` | 非対応のファイル形式 | JPGまたはPNG形式の画像を使用してください |
| `FILE_TOO_LARGE` | ファイルサイズ超過 | 10MB以下の画像を使用してください |
| `NO_METER_DETECTED` | メーター未検出 | メーター全体が写っている画像を使用してください |
| `UNCLEAR_IMAGE` | 画像不鮮明 | 明るい場所で、ピントを合わせて撮影してください |
| `API_ERROR` | Claude APIエラー | しばらく待ってから再度お試しください |
| `RATE_LIMIT` | APIレート制限 | 少し待ってから再度お試しください |
| `NETWORK_ERROR` | ネットワークエラー | インターネット接続を確認してください |

## プロジェクト構造

```
meter-reader/
├── app.py                      # Flask メインアプリケーション
├── config.py                   # 設定ファイル
├── requirements.txt            # Python依存パッケージ
├── .env.example               # 環境変数テンプレート
├── .gitignore                 # Git除外設定
├── README.md                  # このファイル
├── utils/
│   ├── __init__.py
│   ├── claude_client.py       # Claude API クライアント
│   └── image_processor.py     # 画像処理ユーティリティ
├── static/
│   ├── css/
│   │   └── style.css          # スタイルシート
│   ├── js/
│   │   └── app.js             # フロントエンドロジック
│   └── uploads/               # アップロード画像一時保存
└── templates/
    └── index.html             # メインUIテンプレート
```

## 開発

### デバッグモードで実行

```bash
export FLASK_ENV=development
export FLASK_DEBUG=True
python app.py
```

### 本番環境デプロイ

本番環境では Gunicorn などの WSGI サーバーを使用することを推奨します。

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

### Docker での実行

Dockerfileを作成して実行することもできます（例）:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

```bash
docker build -t meter-reader .
docker run -p 5000:5000 --env-file .env meter-reader
```

## トラブルシューティング

### Q: "ANTHROPIC_API_KEY が設定されていません" というエラーが出る

A: `.env`ファイルを作成し、有効なAPIキーを設定してください。

### Q: 画像アップロード時に "413 Request Entity Too Large" エラーが出る

A: 画像サイズを10MB以下に縮小してください。

### Q: "メーターを検出できませんでした" と表示される

A: 以下を確認してください:
- メーター全体が写っているか
- 画像が鮮明か
- 照明が適切か
- 正面から撮影しているか

### Q: 解析に時間がかかる

A: Claude APIの呼び出しには数秒かかる場合があります。ネットワーク接続が安定していることを確認してください。

## セキュリティ

- APIキーは `.env` ファイルで管理し、決してコミットしないでください
- アップロードされた画像は解析後すぐに削除されます
- 本番環境では HTTPS を使用してください
- 適切なレートリミットを設定してください

## ライセンス

MIT License

## 作者

Created with Claude Code

## 貢献

Issue や Pull Request を歓迎します。

## 更新履歴

### v1.0.0 (2026-01-28)
- 初回リリース
- アナログメーター読み取り機能
- 7セグメントデジタルメーター読み取り機能
- レスポンシブWebUI

## 今後の拡張予定

- [ ] 複数画像の一括処理
- [ ] 解析履歴の保存・表示
- [ ] CSV/JSONエクスポート機能
- [ ] カメラからの直接撮影（モバイル）
- [ ] より詳細な信頼度スコア
- [ ] 多言語対応（英語など）
- [ ] テーマ切り替え（ダークモード）
