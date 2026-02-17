import os
from dotenv import load_dotenv

load_dotenv()


def _normalize_tesseract_cmd() -> str | None:
    """.envのTESSERACT_CMDを実行可能な形式に正規化する"""
    raw = os.getenv('TESSERACT_CMD')
    if not raw:
        return None

    path = raw.strip().strip('"').strip("'")
    if not path:
        return None

    path = os.path.expandvars(os.path.expanduser(path))
    path = os.path.normpath(path)

    # フォルダ指定の場合は実行ファイル名を補完
    if os.path.isdir(path):
        path = os.path.join(path, 'tesseract.exe')

    return path


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

    # Tesseract OCR設定（未設定時はPATH上のtesseractを使用）
    TESSERACT_CMD = _normalize_tesseract_cmd()

    @staticmethod
    def init_app(app):
        """アプリケーション初期化時の処理"""
        # アップロードフォルダが存在しない場合は作成
        upload_folder = os.path.join(app.root_path, Config.UPLOAD_FOLDER)
        os.makedirs(upload_folder, exist_ok=True)
