import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Flask アプリケーション設定"""

    # Flask基本設定
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    DEBUG = os.getenv('FLASK_DEBUG', 'True') == 'True'

    # ファイルアップロード設定
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', 10485760))  # 10MB
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'static/uploads')
    ALLOWED_EXTENSIONS = set(os.getenv('ALLOWED_EXTENSIONS', 'jpg,jpeg,png').split(','))

    # Bedrock設定
    BEDROCK_REGION = os.getenv('BEDROCK_REGION', os.getenv('AWS_REGION', 'ap-northeast-1'))
    AWS_PROFILE = os.getenv('AWS_PROFILE', os.getenv('AWS_DEFAULT_PROFILE', '')).strip() or None

    # BedrockのClaudeモデルID
    # 推奨: anthropic.claude-sonnet-4-5-20250929-v1:0
    # 高精度: anthropic.claude-opus-4-1-20250805-v1:0
    # 高速: anthropic.claude-haiku-4-5-20251001-v1:0
    BEDROCK_MODEL_ID = os.getenv(
        'BEDROCK_MODEL_ID',
        os.getenv('CLAUDE_MODEL', 'anthropic.claude-sonnet-4-5-20250929-v1:0')
    )
    BEDROCK_INFERENCE_PROFILE_ID = os.getenv('BEDROCK_INFERENCE_PROFILE_ID', '').strip() or None
    CLAUDE_MAX_TOKENS = int(os.getenv('CLAUDE_MAX_TOKENS', '1024'))
    BEDROCK_TIMEOUT = int(os.getenv('BEDROCK_TIMEOUT', os.getenv('CLAUDE_TIMEOUT', '30')))  # seconds
    BEDROCK_SSL_VERIFY = os.getenv('BEDROCK_SSL_VERIFY', 'True').strip().lower() not in {'0', 'false', 'no'}
    BEDROCK_CA_BUNDLE = os.getenv('BEDROCK_CA_BUNDLE', '').strip() or None

    _debug_mode = os.getenv('DEBUG_MODE', 'BOTH').strip().upper()
    DEBUG_MODE = _debug_mode if _debug_mode in {'OPENCV', 'CLAUDE', 'BOTH'} else 'BOTH'

    # YOLOセグメンテーション設定（アナログメーター前処理）
    YOLO_SEGMENTATION_ENABLED = os.getenv('YOLO_SEGMENTATION_ENABLED', 'True') == 'True'
    YOLO_MODEL_PATH = os.getenv('YOLO_MODEL_PATH', '').strip() or None
    YOLO_CONF_THRESHOLD = float(os.getenv('YOLO_CONF_THRESHOLD', '0.25'))
    YOLO_IOU_THRESHOLD = float(os.getenv('YOLO_IOU_THRESHOLD', '0.45'))

    @staticmethod
    def init_app(app):
        """アプリケーション初期化時の処理"""
        # アップロードフォルダが存在しない場合は作成
        upload_folder = os.path.join(app.root_path, Config.UPLOAD_FOLDER)
        os.makedirs(upload_folder, exist_ok=True)
