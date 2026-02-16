# メーター数値読み取りアプリ

アナログメーターと7セグメントデジタルメーターの画像から数値を自動で読み取るWebアプリケーションです。
OpenCVによるローカル画像処理とClaude APIのビジョン機能を組み合わせて高精度な読み取りを実現しています。

## 機能

- **アナログメーター読み取り**: OpenCVで針の角度を検出し、Claude APIでスケールを読み取って数値を算出
- **7セグメントデジタルメーター読み取り**: Claude APIのビジョン機能でデジタル表示の数値を自動読み取り
- **ドラッグ&ドロップ対応**: 簡単に画像をアップロード
- **レスポンシブデザイン**: PC・タブレット・スマートフォン対応
- **詳細情報表示**: 読み取り結果の信頼度や詳細情報を表示

## アナログメーター読み取りの仕組み

本アプリはアナログメーターの読み取りに2段階のハイブリッド方式を採用しています。

```text
画像 → [OpenCV] 針の角度検出 → position_ratio算出
                                       ↓
画像 → [Claude API] スケール読み取り → scale_min, scale_max, unit
                                       ↓
            [サーバー側計算] value = scale_min + (scale_max - scale_min) × position_ratio
```

1. **OpenCV（ローカル処理）**: ハフ変換で円と針を検出し、針の角度からスケール上の位置比率（position_ratio）を算出
2. **Claude API（スケール読み取り）**: 画像からスケールの最小値・最大値・単位のみを読み取り
3. **サーバー側計算**: `値 = 最小値 + (最大値 - 最小値) × position_ratio` で最終値を算出

この方式により、Claude APIの計算ミスや目視による上書きを排除し、安定した読み取り精度を実現しています。

## サンプル画像

`sample/` ディレクトリにテスト用画像が含まれています:

| ファイル | メータータイプ | 正解値 |
| -------- | -------------- | ------ |
| analog1.jpg | アナログ（COMPOUND圧力計） | -0.078 MPa |
| analog2.jpg | アナログ（COMPOUND圧力計） | 0.46 MPa |
| analog3.jpg | アナログ（スピードメーター） | 124 km/h |
| digital1.jpg | 7セグメントデジタル | - |

## 技術スタック

- **バックエンド**: Python 3.8+, Flask
- **画像認識**: Claude API (Anthropic)
- **画像処理**: OpenCV (opencv-python-headless), NumPy
- **フロントエンド**: HTML5, CSS3, JavaScript (Vanilla)

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
pip install -r requirements.txt
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
CLAUDE_MODEL=claude-sonnet-4-5
CLAUDE_MAX_TOKENS=1024
CLAUDE_TIMEOUT=30
```

### 5. アプリケーションの起動

```bash
python app.py
```

ブラウザで <http://localhost:5000> にアクセスしてください。

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

- **アナログメーター**: メーター全体が写るように撮影してください。複数のメーターが写っている場合、画像中心に最も近いものが解析対象になります
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
    "value": -0.078,
    "unit": "MPa",
    "confidence": "high",
    "meter_type": "analog",
    "details": {
      "scale_range": "-0.1~0.1",
      "needle_position": "スケール全体の11.1%の位置",
      "notes": "計算: -0.1 + (0.1 - -0.1) × 0.1108 = -0.078"
    }
  },
  "timestamp": "2026-02-17T10:30:00Z"
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
  "timestamp": "2026-02-17T10:30:00Z"
}
```

## エラーコード

| コード | 説明 | 対処法 |
| ------ | ---- | ------ |
| `NO_FILE` | ファイルが選択されていない | 画像ファイルを選択してください |
| `INVALID_FILE_TYPE` | 非対応のファイル形式 | JPGまたはPNG形式の画像を使用してください |
| `FILE_TOO_LARGE` | ファイルサイズ超過 | 10MB以下の画像を使用してください |
| `NO_METER_DETECTED` | メーター未検出 | メーター全体が写っている画像を使用してください |
| `API_ERROR` | Claude APIエラー | しばらく待ってから再度お試しください |
| `RATE_LIMIT` | APIレート制限 | 少し待ってから再度お試しください |

## プロジェクト構造

```text
meter-reader/
├── app.py                      # Flask メインアプリケーション
├── config.py                   # 設定ファイル
├── requirements.txt            # Python依存パッケージ
├── .env.example               # 環境変数テンプレート
├── .gitignore                 # Git除外設定
├── README.md                  # このファイル
├── utils/
│   ├── __init__.py
│   ├── claude_client.py       # Claude API クライアント（スケール読み取り+値計算）
│   ├── needle_detector.py     # OpenCV 針検出（角度・位置比率算出）
│   └── image_processor.py     # 画像処理ユーティリティ
├── sample/                    # テスト用サンプル画像
│   ├── analog.jpg
│   ├── analog2.jpg
│   ├── analog3.jpg
│   └── digital.jpg
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
gunicorn -w 4 -b 0.0.0.0:8000 app:app
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

### Q: アナログメーターの読み取り値がずれる

A: OpenCVの針検出は260°スイープを前提としています。ゲージの種類によってスイープ角度が異なるため、±2%程度の誤差が生じる場合があります。

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

## 更新履歴

### v1.1.0 (2026-02-17)

- OpenCVによるアナログメーター針検出機能を追加（NeedleDetector）
- ハフ変換による円検出・針検出、角度からposition_ratio算出
- 画像中心に最も近い円を優先する検出アルゴリズム
- Claudeにはスケール読み取りのみを担当させ、値計算はサーバー側で実行
- Claudeの目視による計算値の上書きを排除し、安定した精度を実現

### v1.0.0 (2026-01-28)

- 初回リリース
- アナログメーター読み取り機能
- 7セグメントデジタルメーター読み取り機能
- レスポンシブWebUI
