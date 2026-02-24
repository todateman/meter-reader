import base64
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError
from botocore.exceptions import SSLError as BotoSSLError

from utils.needle_detector import NeedleDetector


class MeterReaderException(Exception):
    """メーター読み取りエラー用のカスタム例外"""

    def __init__(self, code: str, message: str, details: Optional[str] = None):
        self.code = code
        self.message = message
        self.details = details
        super().__init__(self.message)


class ClaudeVisionClient:
    """Bedrock経由でClaudeのビジョン機能を使用した画像解析クライアント"""

    VALID_DEBUG_MODES = {'OPENCV', 'CLAUDE', 'BOTH'}

    def __init__(
        self,
        model: str = 'anthropic.claude-sonnet-4-5-20250929-v1:0',
        inference_profile_id: Optional[str] = None,
        max_tokens: int = 1024,
        timeout: int = 30,
        region_name: str = 'ap-northeast-1',
        aws_profile: Optional[str] = None,
        ssl_verify: bool = True,
        ca_bundle: Optional[str] = None,
        debug_mode: str = 'BOTH',
        yolo_enabled: bool = True,
        yolo_model_path: Optional[str] = None,
        yolo_conf_threshold: float = 0.25,
        yolo_iou_threshold: float = 0.45,
    ):
        """ClaudeVisionClientの初期化"""
        self.debug_mode = (debug_mode or 'BOTH').upper()
        if self.debug_mode not in self.VALID_DEBUG_MODES:
            self.debug_mode = 'BOTH'

        self.client = None
        if self.debug_mode in {'CLAUDE', 'BOTH'}:
            verify_option: Any = ssl_verify
            if ca_bundle:
                ca_bundle_path = Path(ca_bundle)
                if not ca_bundle_path.exists():
                    raise MeterReaderException(
                        'INVALID_CA_BUNDLE',
                        '指定されたCA証明書ファイルが見つかりません',
                        f'BEDROCK_CA_BUNDLE={ca_bundle}',
                    )
                verify_option = str(ca_bundle_path)

            session_kwargs: Dict[str, Any] = {}
            if aws_profile:
                session_kwargs['profile_name'] = aws_profile

            session = boto3.Session(**session_kwargs)
            self.client = session.client(
                service_name='bedrock-runtime',
                region_name=region_name,
                config=BotoConfig(
                    connect_timeout=timeout,
                    read_timeout=timeout,
                    retries={'max_attempts': 3, 'mode': 'standard'},
                ),
                verify=verify_option,
            )

        self.model = model
        self.inference_profile_id = (inference_profile_id or '').strip() or None
        self.max_tokens = max_tokens
        self.needle_detector = NeedleDetector(
            yolo_enabled=yolo_enabled,
            yolo_model_path=yolo_model_path,
            yolo_conf_threshold=yolo_conf_threshold,
            yolo_iou_threshold=yolo_iou_threshold,
        )

    def analyze_meter(self, image_path: str, meter_type: str) -> Dict[str, Any]:
        """メーター画像を解析"""
        if meter_type == 'analog':
            return self.analyze_analog_meter(image_path)
        if meter_type == 'digital_7segment':
            if self.debug_mode == 'OPENCV':
                raise MeterReaderException(
                    'INVALID_DEBUG_MODE',
                    '現在のDEBUG_MODEではデジタルメーター解析を実行できません',
                    'digital_7segmentはBedrock APIが必要です。DEBUG_MODEをCLAUDEまたはBOTHに変更してください',
                )
            return self.analyze_digital_meter(image_path)

        raise MeterReaderException(
            'INVALID_METER_TYPE',
            f'無効なメータータイプ: {meter_type}',
            '指定できるタイプは "analog" または "digital_7segment" です',
        )

    def analyze_analog_meter(self, image_path: str) -> Dict[str, Any]:
        """アナログメーターをDEBUG_MODEに応じて解析"""
        if self.debug_mode == 'OPENCV':
            return self._analyze_analog_opencv_only(image_path)
        if self.debug_mode == 'CLAUDE':
            return self._analyze_analog_claude_only(image_path)
        return self._analyze_analog_both(image_path)

    def _analyze_analog_both(self, image_path: str) -> Dict[str, Any]:
        """アナログメーターをローカルOpenCV + Bedrock APIで解析（通常モード）"""
        opencv_result = self.needle_detector.detect(image_path)
        debug_image_base64 = self._encode_debug_image(opencv_result.get('debug_image_path'))
        debug_selection_reason = opencv_result.get('debug_selection_reason')

        prompt = self._build_analog_prompt(opencv_result)
        scale_info = self._call_api(image_path, prompt)

        if opencv_result.get('success') and 'scale_min' in scale_info and 'scale_max' in scale_info:
            ratio = opencv_result['position_ratio']
            scale_min = float(scale_info['scale_min'])
            scale_max = float(scale_info['scale_max'])
            calculated_value = scale_min + (scale_max - scale_min) * ratio

            decimal_places = max(
                len(str(scale_min).split('.')[-1]) if '.' in str(scale_min) else 0,
                len(str(scale_max).split('.')[-1]) if '.' in str(scale_max) else 0,
            )
            calculated_value = round(calculated_value, decimal_places + 2)

            return {
                'value': calculated_value,
                'unit': scale_info.get('unit', ''),
                'scale_range': f'{scale_min}~{scale_max}',
                'needle_position': f"スケール全体の{opencv_result['position_percent']}%の位置",
                'confidence': scale_info.get('confidence', 'medium'),
                'notes': f'計算: {scale_min} + ({scale_max} - {scale_min}) × {ratio} = {calculated_value}',
                'debug_image_base64': debug_image_base64,
                'debug_selection_reason': debug_selection_reason,
            }

        scale_info['debug_image_base64'] = debug_image_base64
        scale_info['debug_selection_reason'] = debug_selection_reason
        return scale_info

    def _analyze_analog_opencv_only(self, image_path: str) -> Dict[str, Any]:
        """アナログメーターをOpenCVのみで解析（デバッグ用）"""
        opencv_result = self.needle_detector.detect(image_path)

        if not opencv_result.get('success'):
            raise MeterReaderException(
                'NO_METER_DETECTED',
                'OpenCVで針を検出できませんでした',
                opencv_result.get('error', '針検出に失敗しました'),
            )

        return {
            'value': None,
            'unit': '',
            'scale_range': '',
            'needle_position': f"スケール全体の{opencv_result['position_percent']}%の位置",
            'confidence': 'medium',
            'notes': 'DEBUG_MODE=OPENCV: OpenCVのみ実行（スケール読み取り・最終値計算は未実施）',
            'debug_image_base64': self._encode_debug_image(opencv_result.get('debug_image_path')),
            'debug_selection_reason': opencv_result.get('debug_selection_reason'),
        }

    def _analyze_analog_claude_only(self, image_path: str) -> Dict[str, Any]:
        """アナログメーターをBedrock APIのみで解析（デバッグ用）"""
        prompt = self._build_analog_claude_only_prompt()
        result = self._call_api(image_path, prompt)

        scale_min = result.get('scale_min')
        scale_max = result.get('scale_max')
        scale_range = ''
        if scale_min is not None and scale_max is not None:
            scale_range = f'{scale_min}~{scale_max}'

        return {
            'value': result.get('value'),
            'unit': result.get('unit', ''),
            'scale_range': scale_range,
            'needle_position': result.get('needle_position', ''),
            'confidence': result.get('confidence', 'medium'),
            'notes': result.get('notes', 'DEBUG_MODE=CLAUDE: Bedrock APIのみ実行'),
            'debug_image_base64': None,
            'debug_selection_reason': 'DEBUG_MODE=CLAUDE: OpenCV針検出は実行していません',
        }

    def _encode_debug_image(self, debug_image_path: Optional[str]) -> Optional[str]:
        """デバッグ画像をbase64化して返す"""
        if not debug_image_path:
            return None

        try:
            with open(debug_image_path, 'rb') as file:
                return base64.standard_b64encode(file.read()).decode('utf-8')
        except Exception:
            return None

    def analyze_digital_meter(self, image_path: str) -> Dict[str, Any]:
        """7セグメントデジタルメーターを解析"""
        prompt = self._build_digital_prompt()
        return self._call_api(image_path, prompt)

    def _call_api(self, image_path: str, prompt: str) -> Dict[str, Any]:
        """Bedrock APIを呼び出して画像を解析"""
        try:
            if self.client is None:
                raise MeterReaderException(
                    'API_DISABLED',
                    'Bedrock APIクライアントが初期化されていません',
                    'DEBUG_MODE=OPENCVではBedrock API呼び出しは無効です',
                )

            with open(image_path, 'rb') as file:
                image_data = file.read()

            image_format = self._get_image_format(image_path)

            message = self.client.converse(
                modelId=self.inference_profile_id or self.model,
                inferenceConfig={
                    'maxTokens': self.max_tokens,
                },
                messages=[
                    {
                        'role': 'user',
                        'content': [
                            {
                                'image': {
                                    'format': image_format,
                                    'source': {
                                        'bytes': image_data,
                                    },
                                }
                            },
                            {
                                'text': prompt,
                            },
                        ],
                    }
                ],
            )

            return self._parse_response(message)

        except ClientError as error:
            error_code = error.response.get('Error', {}).get('Code', 'Unknown')
            error_message = error.response.get('Error', {}).get('Message', '')
            if error_code in {'ThrottlingException', 'TooManyRequestsException'}:
                raise MeterReaderException(
                    'RATE_LIMIT',
                    'APIレート制限に達しました',
                    'しばらく待ってから再度お試しください',
                )

            if error_code == 'ValidationException' and 'inference profile' in error_message.lower():
                raise MeterReaderException(
                    'INFERENCE_PROFILE_REQUIRED',
                    'このモデルはオンデマンド呼び出しに対応していません',
                    'BEDROCK_INFERENCE_PROFILE_ID に推論プロファイルIDまたはARNを設定してください。'
                    f' 現在のBEDROCK_MODEL_ID={self.model}',
                )

            raise MeterReaderException(
                'API_ERROR',
                'Bedrock API エラーが発生しました',
                f'{error_code}: {error_message or error}',
            )
        except FileNotFoundError:
            raise MeterReaderException(
                'FILE_NOT_FOUND',
                '画像ファイルが見つかりません',
                f'パス: {image_path}',
            )
        except BotoSSLError as error:
            raise MeterReaderException(
                'SSL_ERROR',
                'SSL証明書の検証に失敗しました',
                'BEDROCK_CA_BUNDLE に社内ルートCAのpemファイルを指定してください。'
                '一時的に検証を無効化する場合は BEDROCK_SSL_VERIFY=False を設定できます（非推奨）。'
                f' 詳細: {error}',
            )
        except MeterReaderException:
            raise
        except Exception as error:
            raise MeterReaderException(
                'UNKNOWN_ERROR',
                '予期しないエラーが発生しました',
                str(error),
            )

    def _build_analog_prompt(self, opencv_result: Dict[str, Any]) -> str:
        """アナログメーター用プロンプトを構築（スケール読み取り専用）"""
        return """あなたはアナログメーター読み取りの専門家です。
画像からメーターのスケール情報を読み取ってください。

## 読み取り手順
1. メーターのスケール最小値（左下端の数字）を読み取る
   - 負の値の場合はマイナス記号を含める（例: -0.1）
2. メーターのスケール最大値（右下端の数字）を読み取る
3. 単位を読み取る（MPa, kPa, km/h, ℃ など）

## 注意事項
- COMPOUND（複合）ゲージの場合、左下が負の最小値、右下が正の最大値です
- 左右対称のスケール（例: 左に0.1、右に0.1）の場合、左側は負（-0.1）です

## 出力形式
必ず以下のJSON形式で返してください（他のテキストは含めない）:

```json
{
  "scale_min": 最小値（数値）, 
  "scale_max": 最大値（数値）, 
  "unit": "単位文字列", 
  "confidence": "high/medium/low", 
  "notes": "スケールの説明"
}
```"""

    def _build_analog_claude_only_prompt(self) -> str:
        """アナログメーター用プロンプトを構築（Claude単独読み取り）"""
        return """あなたはアナログメーター読み取りの専門家です。
画像を目視して、メーター値を直接読み取ってください。

## 読み取り対象
1. 現在の指示値（value）
2. 単位（unit）
3. スケール最小値（scale_min）
4. スケール最大値（scale_max）
5. 針位置の説明（needle_position）

## 注意事項
- COMPOUNDゲージは左側が負値の可能性があります
- 値が曖昧な場合は confidence を medium か low にしてください

## 出力形式
必ず以下のJSON形式で返してください（他のテキストは含めない）:

```json
{
    "value": 数値,
    "unit": "単位文字列",
    "scale_min": 最小値（数値）, 
    "scale_max": 最大値（数値）, 
    "needle_position": "針位置の説明",
    "confidence": "high/medium/low",
    "notes": "読み取り根拠の簡潔な説明"
}
```"""

    def _build_digital_prompt(self) -> str:
        """7セグメントデジタルメーター用プロンプトを構築"""
        return """あなたは7セグメント・LED・LCDディスプレイ読み取りの専門家です。以下の手順で画像内のデジタルメーターを正確に読み取ってください。

## 読み取り手順

### ステップ1: ディスプレイの特定
1. デジタル表示部分を特定
2. 表示タイプを識別（7セグメント、LED、LCD、ドットマトリクスなど）
3. 桁数を確認
4. 小数点やコロンの位置を確認

### ステップ2: 各桁の数字を読み取り
1. 左から右へ順番に各桁を読み取る
2. 各数字（0-9）を正確に識別
   - 8と0、6と5、1と7などの混同に注意
   - セグメントの点灯パターンで判断
3. 小数点の位置を正確に記録
4. 先頭の0は省略せずに記録（表示されている場合）

### ステップ3: 特殊文字・単位の確認
1. 単位記号を確認（kWh, V, A, ℃, MPa など）
2. マイナス記号の有無を確認
3. その他の記号（%, °, :など）を確認

### ステップ4: 検証
1. 読み取った数値が妥当な範囲か確認
2. セグメントの欠けや異常がないか確認
3. 反射や影による誤認識がないか確認

## 注意事項
- 反射光や影を数字と間違えない
- 消えているセグメントと点灯していないセグメントを区別
- 斜めからの撮影による歪みを考慮
- 複数の表示部がある場合は主要な数値を読み取る
- 不明確な桁がある場合は confidence を "medium" または "low" に設定

## 出力形式
必ず以下のJSON形式で返してください（他のテキストは含めない）:

```json
{
  "value": 数値（数値型、小数点含む、例: 123.45）, 
  "unit": "単位文字列（例: kWh, V, A, ℃など。不明な場合は空文字列）", 
  "decimal_places": 小数点以下の桁数（整数型、例: 123.45なら2）, 
  "confidence": "high/medium/low（high: 全桁明確、medium: 一部不鮮明、low: 判読困難）", 
  "segment_status": "正常/異常の説明（例: 正常、2桁目のセグメント一部欠け、など）", 
  "notes": "追加の観察事項（例: 反射あり、先頭に0表示、マイナス記号ありなど）"
}
```

エラーの場合:
```json
{
  "error": "エラーの詳細説明（デジタル表示が写っていない、画像が不鮮明など）", 
  "confidence": "low"
}
```"""

    def _parse_response(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Bedrock APIのレスポンスからJSONを抽出してパース"""
        try:
            content = message.get('output', {}).get('message', {}).get('content', [])
            text_blocks = [item.get('text', '') for item in content if isinstance(item, dict) and 'text' in item]
            response_text = '\n'.join(text_blocks).strip()

            if not response_text:
                raise MeterReaderException(
                    'PARSE_ERROR',
                    'レスポンスにテキストが含まれていません',
                    str(message),
                )

            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
            if json_match:
                json_text = json_match.group(1)
            else:
                json_match = re.search(r'\{[\s\S]*\}', response_text)
                if not json_match:
                    raise MeterReaderException(
                        'PARSE_ERROR',
                        'レスポンスからJSONを抽出できませんでした',
                        response_text,
                    )
                json_text = json_match.group(0)

            data = json.loads(json_text)

            if 'error' in data:
                raise MeterReaderException(
                    'NO_METER_DETECTED',
                    'メーターを読み取れませんでした',
                    data.get('error', '不明なエラー'),
                )

            return data

        except json.JSONDecodeError as error:
            raise MeterReaderException(
                'PARSE_ERROR',
                'JSONのパースに失敗しました',
                str(error),
            )
        except (IndexError, AttributeError, TypeError) as error:
            raise MeterReaderException(
                'PARSE_ERROR',
                'レスポンスの形式が不正です',
                str(error),
            )

    def _get_image_format(self, image_path: str) -> str:
        """ファイル拡張子からBedrock用の画像フォーマットを取得"""
        ext = Path(image_path).suffix.lower()
        mapping = {
            '.jpg': 'jpeg',
            '.jpeg': 'jpeg',
            '.png': 'png',
        }
        return mapping.get(ext, 'jpeg')
