from flask import Flask, render_template, request, jsonify
from werkzeug.exceptions import RequestEntityTooLarge
from datetime import datetime
import os

from config import Config
from utils.claude_client import ClaudeVisionClient, MeterReaderException
from utils.image_processor import ImageProcessor, ImageProcessorException


app = Flask(__name__)
app.config.from_object(Config)
Config.init_app(app)

# クライアントとプロセッサの初期化
claude_client = ClaudeVisionClient(
    model=app.config['BEDROCK_MODEL_ID'],
    max_tokens=app.config['CLAUDE_MAX_TOKENS'],
    timeout=app.config['BEDROCK_TIMEOUT'],
    region_name=app.config['BEDROCK_REGION'],
    debug_mode=app.config['DEBUG_MODE'],
    yolo_enabled=app.config['YOLO_SEGMENTATION_ENABLED'],
    yolo_model_path=app.config['YOLO_MODEL_PATH'],
    yolo_conf_threshold=app.config['YOLO_CONF_THRESHOLD'],
    yolo_iou_threshold=app.config['YOLO_IOU_THRESHOLD'],
)

image_processor = ImageProcessor(
    allowed_extensions=app.config['ALLOWED_EXTENSIONS'],
    max_size=app.config['MAX_CONTENT_LENGTH']
)


@app.route('/')
def index():
    """メインページを表示"""
    return render_template('index.html', debug_mode=app.config['DEBUG_MODE'])


@app.route('/api/health', methods=['GET'])
def health():
    """ヘルスチェックエンドポイント"""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    メーター画像を解析するAPIエンドポイント

    Request:
        - image: 画像ファイル（multipart/form-data）
        - meter_type: メータータイプ（'analog' または 'digital_7segment'）

    Response:
        成功時:
        {
            "success": true,
            "data": {
                "value": 数値,
                "unit": "単位",
                "confidence": "信頼度",
                "details": {...},
                "meter_type": "メータータイプ"
            },
            "timestamp": "ISO形式のタイムスタンプ"
        }

        エラー時:
        {
            "success": false,
            "error": {
                "code": "エラーコード",
                "message": "エラーメッセージ",
                "details": "詳細情報"
            }
        }
    """
    filepath = None

    try:
        # リクエストのバリデーション
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': {
                    'code': 'NO_FILE',
                    'message': '画像ファイルが送信されていません',
                    'details': 'imageフィールドにファイルをアップロードしてください'
                }
            }), 400

        file = request.files['image']
        meter_type = request.form.get('meter_type', 'analog')

        # メータータイプのバリデーション
        if meter_type not in ['analog', 'digital_7segment']:
            return jsonify({
                'success': False,
                'error': {
                    'code': 'INVALID_METER_TYPE',
                    'message': '無効なメータータイプです',
                    'details': 'meter_typeは"analog"または"digital_7segment"を指定してください'
                }
            }), 400

        # 画像のバリデーションと保存
        try:
            filepath, original_filename = image_processor.validate_and_save(
                file,
                app.config['UPLOAD_FOLDER']
            )
        except ImageProcessorException as e:
            return jsonify({
                'success': False,
                'error': {
                    'code': e.code,
                    'message': e.message,
                    'details': e.details
                }
            }), 400

        # Claude APIでメーター解析
        try:
            result = claude_client.analyze_meter(filepath, meter_type)

            # レスポンスの構築
            response_data = {
                'value': result.get('value'),
                'unit': result.get('unit', ''),
                'confidence': result.get('confidence', 'medium'),
                'meter_type': meter_type,
                'analysis_mode': app.config['DEBUG_MODE'],
                'details': {}
            }

            # メータータイプごとの詳細情報
            if meter_type == 'analog':
                response_data['details'] = {
                    'scale_range': result.get('scale_range', ''),
                    'needle_position': result.get('needle_position', ''),
                    'notes': result.get('notes', ''),
                    'debug_image_base64': result.get('debug_image_base64', None),
                    'debug_selection_reason': result.get('debug_selection_reason', '')
                }
            else:  # digital_7segment
                response_data['details'] = {
                    'decimal_places': result.get('decimal_places', 0),
                    'segment_status': result.get('segment_status', '正常'),
                    'notes': result.get('notes', '')
                }

            return jsonify({
                'success': True,
                'data': response_data,
                'timestamp': datetime.now().isoformat()
            })

        except MeterReaderException as e:
            return jsonify({
                'success': False,
                'error': {
                    'code': e.code,
                    'message': e.message,
                    'details': e.details
                }
            }), 400

    except Exception as e:
        # 予期しないエラー
        app.logger.error(f'Unexpected error in /api/analyze: {str(e)}')
        return jsonify({
            'success': False,
            'error': {
                'code': 'INTERNAL_ERROR',
                'message': '内部サーバーエラーが発生しました',
                'details': str(e) if app.config['DEBUG'] else '詳細情報は利用できません'
            }
        }), 500

    finally:
        # アップロードされたファイルのクリーンアップ
        if filepath and os.path.exists(filepath):
            try:
                # 解析後は即座に削除（必要に応じて保持期間を設定可能）
                image_processor.cleanup_file(filepath)
            except Exception as e:
                app.logger.error(f'Failed to cleanup file {filepath}: {str(e)}')


@app.errorhandler(413)
def request_entity_too_large(error):
    """ファイルサイズ超過エラーハンドラ"""
    max_mb = app.config['MAX_CONTENT_LENGTH'] / (1024 * 1024)
    return jsonify({
        'success': False,
        'error': {
            'code': 'FILE_TOO_LARGE',
            'message': 'ファイルサイズが大きすぎます',
            'details': f'最大サイズ: {max_mb:.1f}MB'
        }
    }), 413


@app.errorhandler(404)
def not_found(error):
    """404エラーハンドラ"""
    return jsonify({
        'success': False,
        'error': {
            'code': 'NOT_FOUND',
            'message': 'リクエストされたエンドポイントが見つかりません',
            'details': str(error)
        }
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """500エラーハンドラ"""
    app.logger.error(f'Internal server error: {str(error)}')
    return jsonify({
        'success': False,
        'error': {
            'code': 'INTERNAL_ERROR',
            'message': '内部サーバーエラーが発生しました',
            'details': '詳細情報はログを確認してください'
        }
    }), 500


# 定期的な古いファイルのクリーンアップ（起動時に実行）
@app.before_request
def cleanup_old_uploads():
    """古いアップロードファイルを定期的にクリーンアップ"""
    # 最初のリクエスト時のみ実行（簡易的な実装）
    if not hasattr(app, 'cleanup_done'):
        deleted = image_processor.cleanup_old_files(
            app.config['UPLOAD_FOLDER'],
            max_age_hours=1  # 1時間以上古いファイルを削除
        )
        if deleted > 0:
            app.logger.info(f'Cleaned up {deleted} old upload files')
        app.cleanup_done = True


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=app.config['DEBUG']
    )
