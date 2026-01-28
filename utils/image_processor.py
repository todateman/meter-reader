import os
import io
from pathlib import Path
from typing import Optional, Tuple
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
from PIL import Image


class ImageProcessorException(Exception):
    """画像処理エラー用のカスタム例外"""

    def __init__(self, code: str, message: str, details: Optional[str] = None):
        self.code = code
        self.message = message
        self.details = details
        super().__init__(self.message)


class ImageProcessor:
    """画像処理とバリデーションを行うクラス"""

    def __init__(self, allowed_extensions: set = None, max_size: int = 10485760):
        """
        ImageProcessorの初期化

        Args:
            allowed_extensions: 許可する拡張子のセット
            max_size: 最大ファイルサイズ（バイト）
        """
        self.allowed_extensions = allowed_extensions or {'jpg', 'jpeg', 'png'}
        self.max_size = max_size
        self.max_dimension = 4096  # 最大画像サイズ（幅・高さ）

    def validate_and_save(
        self,
        file: FileStorage,
        upload_folder: str
    ) -> Tuple[str, str]:
        """
        画像ファイルをバリデーションして保存

        Args:
            file: アップロードされたファイル
            upload_folder: 保存先フォルダ

        Returns:
            (保存したファイルのパス, オリジナルのファイル名)

        Raises:
            ImageProcessorException: バリデーションエラー時
        """
        # ファイルの存在チェック
        if not file or not file.filename:
            raise ImageProcessorException(
                'NO_FILE',
                'ファイルが選択されていません',
                'ファイルを選択してアップロードしてください'
            )

        # ファイル名のバリデーション
        filename = secure_filename(file.filename)
        if not filename:
            raise ImageProcessorException(
                'INVALID_FILENAME',
                'ファイル名が不正です',
                '有効なファイル名を使用してください'
            )

        # 拡張子のチェック
        if not self._allowed_file(filename):
            raise ImageProcessorException(
                'INVALID_FILE_TYPE',
                'サポートされていないファイル形式です',
                f'対応形式: {", ".join(self.allowed_extensions).upper()}'
            )

        # ファイルを一時的にメモリに読み込む
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        # ファイルサイズのチェック
        if file_size > self.max_size:
            max_mb = self.max_size / (1024 * 1024)
            raise ImageProcessorException(
                'FILE_TOO_LARGE',
                f'ファイルサイズが大きすぎます',
                f'最大サイズ: {max_mb:.1f}MB'
            )

        # MIMEタイプの検証（Pillowを使用）
        file_data = file.read()
        file.seek(0)  # リセット

        try:
            # Pillowで画像を開いて形式を確認
            img = Image.open(io.BytesIO(file_data))
            image_format = img.format
            if image_format not in ['JPEG', 'PNG']:
                raise ImageProcessorException(
                    'INVALID_FILE_TYPE',
                    'ファイルの内容が画像形式ではありません',
                    f'正しいJPEGまたはPNG画像を使用してください（検出された形式: {image_format}）'
                )
        except Exception as e:
            if isinstance(e, ImageProcessorException):
                raise
            raise ImageProcessorException(
                'INVALID_FILE_TYPE',
                'ファイルの内容が画像形式ではありません',
                '正しいJPEGまたはPNG画像を使用してください'
            )

        # ユニークなファイル名を生成
        import uuid
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(upload_folder, unique_filename)

        # ファイルを保存
        try:
            file.save(filepath)
        except Exception as e:
            raise ImageProcessorException(
                'SAVE_ERROR',
                'ファイルの保存に失敗しました',
                str(e)
            )

        # 画像の検証とリサイズ（必要な場合）
        try:
            self._validate_and_resize_image(filepath)
        except Exception as e:
            # エラーが発生した場合はファイルを削除
            if os.path.exists(filepath):
                os.remove(filepath)
            raise ImageProcessorException(
                'IMAGE_VALIDATION_ERROR',
                '画像の検証に失敗しました',
                str(e)
            )

        return filepath, filename

    def _allowed_file(self, filename: str) -> bool:
        """
        ファイル拡張子が許可されているかチェック

        Args:
            filename: ファイル名

        Returns:
            許可されている場合True
        """
        return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in self.allowed_extensions

    def _validate_and_resize_image(self, filepath: str) -> None:
        """
        画像を検証し、必要に応じてリサイズ

        Args:
            filepath: 画像ファイルのパス

        Raises:
            Exception: 画像の検証・処理エラー時
        """
        try:
            with Image.open(filepath) as img:
                # 画像形式の確認
                if img.format not in ['JPEG', 'PNG']:
                    raise ValueError(f'サポートされていない画像形式: {img.format}')

                # 画像サイズのチェック
                width, height = img.size

                # 最小サイズのチェック（小さすぎる画像は解析に適さない）
                if width < 50 or height < 50:
                    raise ValueError('画像が小さすぎます（最小: 50x50ピクセル）')

                # 大きすぎる画像はリサイズ
                if width > self.max_dimension or height > self.max_dimension:
                    # アスペクト比を保持してリサイズ
                    img.thumbnail((self.max_dimension, self.max_dimension), Image.Resampling.LANCZOS)

                    # リサイズした画像を保存
                    if img.format == 'JPEG':
                        img.save(filepath, 'JPEG', quality=95, optimize=True)
                    else:
                        img.save(filepath, 'PNG', optimize=True)

        except Exception as e:
            raise Exception(f'画像の検証・処理中にエラーが発生しました: {str(e)}')

    def cleanup_file(self, filepath: str) -> None:
        """
        アップロードされたファイルを削除

        Args:
            filepath: 削除するファイルのパス
        """
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception:
            # クリーンアップのエラーは無視（ログに記録すべきだが、ここでは省略）
            pass

    def cleanup_old_files(self, upload_folder: str, max_age_hours: int = 24) -> int:
        """
        古いアップロードファイルを削除

        Args:
            upload_folder: アップロードフォルダ
            max_age_hours: 削除対象とする経過時間（時間単位）

        Returns:
            削除したファイル数
        """
        import time
        deleted_count = 0
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600

        try:
            for filename in os.listdir(upload_folder):
                filepath = os.path.join(upload_folder, filename)

                # .gitkeepは削除しない
                if filename == '.gitkeep':
                    continue

                # ファイルの最終更新時刻をチェック
                if os.path.isfile(filepath):
                    file_age = current_time - os.path.getmtime(filepath)
                    if file_age > max_age_seconds:
                        os.remove(filepath)
                        deleted_count += 1
        except Exception:
            # クリーンアップのエラーは無視
            pass

        return deleted_count
