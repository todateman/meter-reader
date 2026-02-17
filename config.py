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

    # Claude API設定
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

    # Claude APIモデル設定
    # デフォルトは claude-sonnet-4-5 (精度と速度のバランスが良い)
    # その他の選択肢:
    #   - claude-opus-4-5: 最高精度（高コスト）
    #   - claude-haiku-4-5: 高速・低コスト（精度は劣る）
    CLAUDE_MODEL = os.getenv('CLAUDE_MODEL', 'claude-sonnet-4-5')
    CLAUDE_MAX_TOKENS = int(os.getenv('CLAUDE_MAX_TOKENS', '1024'))
    CLAUDE_TIMEOUT = int(os.getenv('CLAUDE_TIMEOUT', '30'))  # seconds

    _debug_mode = os.getenv('DEBUG_MODE', 'BOTH').strip().upper()
    DEBUG_MODE = _debug_mode if _debug_mode in {'OPENCV', 'CLAUDE', 'BOTH'} else 'BOTH'

    @staticmethod
    def init_app(app):
        """アプリケーション初期化時の処理"""
        # アップロードフォルダが存在しない場合は作成
        upload_folder = os.path.join(app.root_path, Config.UPLOAD_FOLDER)
        os.makedirs(upload_folder, exist_ok=True)
