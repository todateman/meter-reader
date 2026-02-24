# メーター数値読み取りアプリ

アナログメーターと7セグメントデジタルメーターの画像から数値を自動で読み取るWebアプリケーションです。
OpenCVによるローカル画像処理とAWS Bedrock（Claudeモデル）のビジョン機能を組み合わせて高精度な読み取りを実現しています。

## 機能

- **アナログメーター読み取り**: YOLOセグメンテーション（有効時）+ OpenCVで針の角度を検出し、Bedrock上のClaudeでスケールを読み取って数値を算出
- **7セグメントデジタルメーター読み取り**: Bedrock上のClaudeビジョン機能でデジタル表示の数値を自動読み取り
- **ドラッグ&ドロップ対応**: 簡単に画像をアップロード
- **レスポンシブデザイン**: PC・タブレット・スマートフォン対応
- **詳細情報表示**: 読み取り結果の信頼度や詳細情報を表示

## アナログメーター読み取りの仕組み

本アプリはアナログメーターの読み取りに2段階のハイブリッド方式を採用しています。

```text
画像 → [OpenCV] 針の角度検出 → position_ratio算出
                                       ↓
画像 → [AWS Bedrock (Claude)] スケール読み取り → scale_min, scale_max, unit
                                       ↓
            [サーバー側計算] value = scale_min + (scale_max - scale_min) × position_ratio
```

1. **YOLO + OpenCV（ローカル処理）**: YOLOセグメンテーションでメーター領域（およびモデルに含まれる場合は針）を補助検出し、OpenCVで針の角度からスケール上の位置比率（position_ratio）を算出
2. **AWS Bedrock（スケール読み取り）**: 画像からスケールの最小値・最大値・単位のみを読み取り
3. **サーバー側計算**: `値 = 最小値 + (最大値 - 最小値) × position_ratio` で最終値を算出

この方式により、モデル側の計算ミスや目視による上書きを排除し、安定した読み取り精度を実現しています。

## サンプル画像

`sample/` ディレクトリにテスト用画像が含まれています:

| ファイル | メータータイプ | 正解値 | 最小値 | 最大値 |
| ------- | ------------- | ------ | ----- | ----- |
| analog1.jpg | アナログ（COMPOUND圧力計） | -0.078 MPa | -0.1 MPa | 0.1 MPa |
| analog2.jpg | アナログ（COMPOUND圧力計） | 0.46 MPa | -0.1 MPa | 1.5 MPa |
| analog3.jpg | アナログ（スピードメーター） | 124 km/h | 0 km/h | 160 km/h |
| analog4.jpg | アナログ（COMPOUND圧力計） | -8.0 kg/cm2 | 0 kg/cm2 | 10 kg/cm2 |
| analog5.jpg | アナログ（COMPOUND圧力計） | -4.5 kg/cm2 | 0 kg/cm2 | 10 kg/cm2 |
| digital1.jpg | 7セグメントデジタル | -2.66 MPa | - | - |

## 技術スタック

- **バックエンド**: Python 3.8+, Flask
- **画像認識**: AWS Bedrock（Claude）
- **画像処理**: OpenCV (opencv-python-headless), NumPy
- **フロントエンド**: HTML5, CSS3, JavaScript (Vanilla)

## 必要要件

- Python 3.8以上
- AWSアカウント（Bedrock利用権限付き）
- AWS CLI（認証情報設定済み）
- インターネット接続（Bedrock API呼び出しのため）

## セットアップ

### 1. リポジトリのクローン

```bash
cd meter-reader
```

### 2. 仮想環境の作成（推奨）

#### 2.1. uv（推奨）

```bash
uv venv
```

#### 2.2. venv

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# または
.venv\Scripts\activate  # Windows
```

### 3. 依存パッケージのインストール

#### 3.1 uv（推奨）

```bash
uv sync  # pyproject.toml と uv.lock をプロジェクト環境に同期
# または
uv pip install -r requirements.txt
```

#### 3.2. venv

```bash
pip install -r requirements.txt
```

### 4. AWS CLIのセットアップ

AWS Bedrockを使うため、先にAWS CLIを設定します。

```bash
# Linux でAWS CLIをインストール（未導入の場合）
sudo yum remove awscli

# Windows でAWS CLIをインストール（未導入の場合）
msiexec.exe /i https://awscli.amazonaws.com/AWSCLIV2.msi

# AWS CLIのインストール確認（バージョン確認）
aws --version

# リージョンを設定（例では東京リージョン）
aws configure set region ap-northeast-1

# AWS認証情報をプロファイルに追加する
# ロールに合わせて使い分けられるように、ロールをプロファイル名にしておくとよい
aws configure --profile hoge

  # 対話形式で各パラメータを入力する
  AWS Access Key ID [None]: 
  AWS Secret Access Key [None]: 
  AWS Session Token [None]: 
  Default region name [None]: ap-northeast-1
  Default output format [None]: json

# 設定確認
aws configure list
```

認証情報確認:

```bash
aws sts get-caller-identity
```

`Unable to locate credentials` が表示される場合は、`aws configure` または `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` を設定してください。

### 5. 環境変数の設定

`.env.example`をコピーして`.env`ファイルを作成します。

```bash
cp .env.example .env
```

`.env`ファイルを編集:

```env
FLASK_ENV=development
FLASK_DEBUG=True
MAX_CONTENT_LENGTH=10485760
UPLOAD_FOLDER=static/uploads
ALLOWED_EXTENSIONS=jpg,jpeg,png

# AWS Bedrock設定
# AWS CLIで設定したリージョンと同じ値を推奨
BEDROCK_REGION=ap-northeast-1

# Bedrock ClaudeモデルID(2026.02.24時点の最新モデル)
# 推奨: anthropic.claude-sonnet-4-5-20250929-v1:0
# 高精度: anthropic.claude-opus-4-6-v1
# 高速: anthropic.claude-haiku-4-5-20251001-v1:0
BEDROCK_MODEL_ID=anthropic.claude-sonnet-4-5-20250929-v1:0
# モデルによっては推論プロファイル必須（IDまたはARN）
# 推論プロファイル一覧：https://docs.aws.amazon.com/ja_jp/bedrock/latest/userguide/inference-profiles-support.html
# 推奨: jp.anthropic.claude-sonnet-4-5-20250929-v1:0
# 高速: jp.anthropic.claude-haiku-4-5-20251001-v1:0
BEDROCK_INFERENCE_PROFILE_ID=jp.anthropic.claude-sonnet-4-5-20250929-v1:0
CLAUDE_MAX_TOKENS=1024
BEDROCK_TIMEOUT=30

# 企業ネットワーク配下で SSL: CERTIFICATE_VERIFY_FAILED が出る場合に設定
# 参考：https://docs.aws.amazon.com/ja_jp/cli/v1/userguide/cli-chap-troubleshooting.html#tshoot-certificate-verify-failed, https://qiita.com/tyskJ/items/728fe1c2a73abd43cb40, https://qiita.com/satoushina/items/56831655a141ec80917d
# 例: C:/certs/corporate-root-ca.pem
BEDROCK_SSL_VERIFY=false
BEDROCK_CA_BUNDLE=

# aws configure で作成済みプロファイルを使う場合（推奨）
AWS_PROFILE=default

# AWS認証情報（aws configure済みなら不要）
# AWS_ACCESS_KEY_ID=
# AWS_SECRET_ACCESS_KEY=
# AWS_SESSION_TOKEN=

# 解析モード
# OPENCV: OpenCVのみ実行（アナログの針位置のみ）
# CLAUDE: Bedrock APIのみ実行
# BOTH: OpenCV + Bedrock API（通常モード）
DEBUG_MODE=BOTH
```

`AWS_PROFILE` を指定すると、アクセスキーを `.env` に直接書かずに、`aws configure` で保存済みの認証情報を利用できます。  

`ValidationException: ... with on-demand throughput isn’t supported` が出る場合は、`BEDROCK_INFERENCE_PROFILE_ID` に対象モデルを含む推論プロファイルのIDまたはARNを設定してください。  

`SSL: CERTIFICATE_VERIFY_FAILED` が発生する場合は、`BEDROCK_CA_BUNDLE` に社内ルートCA証明書（`.pem` もしくは `.cer`）を指定してください。  
どうしても一時回避が必要な場合のみ `BEDROCK_SSL_VERIFY=False` を使えますが、セキュリティ上は非推奨です。

### DEBUG_MODE の選択肢

- `OPENCV`: OpenCVのみ実行（アナログメーターの針位置検出のみ。スケール読み取り・最終値計算は行いません）
- `CLAUDE`: Bedrock APIのみ実行（画像目視で値を直接読み取り）
- `BOTH`: OpenCVとBedrock APIを両方実行（通常モード）

### YOLOセグメンテーション設定

- `YOLO_SEGMENTATION_ENABLED=True`: YOLOセグメンテーションを有効化（推奨）
- `YOLO_MODEL_PATH`: カスタム学習済みセグメンテーションモデル（`.pt`）を指定可能
- `YOLO_CONF_THRESHOLD` / `YOLO_IOU_THRESHOLD`: 検出のしきい値
- YOLOで対象が検出できない場合は自動的にOpenCV検出へフォールバック

### 6. アプリケーションの起動

#### 6.1. uv（推奨）

```bash
uv run app.py
```

#### 6.2. venv

`.venv`の仮想環境内で

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
| `API_ERROR` | Bedrock APIエラー | しばらく待ってから再度お試しください |
| `RATE_LIMIT` | APIレート制限 | 少し待ってから再度お試しください |

## プロジェクト構造

```text
meter-reader/
├── app.py                      # Flask メインアプリケーション
├── config.py                   # 設定ファイル
├── requirements.txt            # Python依存パッケージ
├── pyproject.toml              # uvの設定ファイル
├── uv.lock                     # uvのプロジェクト依存関係ファイル
├── .env.example               # 環境変数テンプレート
├── .gitignore                 # Git除外設定
├── README.md                  # このファイル
├── utils/
│   ├── __init__.py
│   ├── claude_client.py       # Bedrock API クライアント（スケール読み取り+値計算）
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

### Q: "Unable to locate credentials" というエラーが出る

A: AWS認証情報が未設定です。`aws configure` または `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` を設定してください。

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

A: Bedrock APIの呼び出しには数秒かかる場合があります。ネットワーク接続が安定していることを確認してください。

## セキュリティ

- AWS認証情報（アクセスキー、シークレットキー）は `.env` や環境変数で管理し、決してコミットしないでください
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
