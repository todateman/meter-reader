import anthropic
import base64
import json
import re
from typing import Dict, Any, Optional
from pathlib import Path


class MeterReaderException(Exception):
    """メーター読み取りエラー用のカスタム例外"""

    def __init__(self, code: str, message: str, details: Optional[str] = None):
        self.code = code
        self.message = message
        self.details = details
        super().__init__(self.message)


class ClaudeVisionClient:
    """Claude APIのビジョン機能を使用した画像解析クライアント"""

    def __init__(self, api_key: str, model: str = 'claude-opus-4-5-20251101', max_tokens: int = 1024):
        """
        ClaudeVisionClientの初期化

        Args:
            api_key: Anthropic API キー
            model: 使用するClaudeモデル
            max_tokens: 最大トークン数
        """
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY が設定されていません")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def analyze_meter(self, image_path: str, meter_type: str) -> Dict[str, Any]:
        """
        メーター画像を解析

        Args:
            image_path: 画像ファイルのパス
            meter_type: メータータイプ ('analog' or 'digital_7segment')

        Returns:
            解析結果の辞書

        Raises:
            MeterReaderException: 解析エラー時
        """
        if meter_type == 'analog':
            return self.analyze_analog_meter(image_path)
        elif meter_type == 'digital_7segment':
            return self.analyze_digital_meter(image_path)
        else:
            raise MeterReaderException(
                'INVALID_METER_TYPE',
                f'無効なメータータイプ: {meter_type}',
                '指定できるタイプは "analog" または "digital_7segment" です'
            )

    def analyze_analog_meter(self, image_path: str) -> Dict[str, Any]:
        """アナログメーターを解析"""
        prompt = self._build_analog_prompt()
        return self._call_api(image_path, prompt)

    def analyze_digital_meter(self, image_path: str) -> Dict[str, Any]:
        """7セグメントデジタルメーターを解析"""
        prompt = self._build_digital_prompt()
        return self._call_api(image_path, prompt)

    def _call_api(self, image_path: str, prompt: str) -> Dict[str, Any]:
        """
        Claude APIを呼び出して画像を解析

        Args:
            image_path: 画像ファイルのパス
            prompt: 解析用プロンプト

        Returns:
            解析結果の辞書

        Raises:
            MeterReaderException: API呼び出しエラー時
        """
        try:
            # 画像ファイルを読み込んでbase64エンコード
            with open(image_path, 'rb') as f:
                image_data = base64.standard_b64encode(f.read()).decode('utf-8')

            # 画像のメディアタイプを判定
            media_type = self._get_media_type(image_path)

            # Claude APIにリクエスト
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_data,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ],
                    }
                ],
            )

            # レスポンスをパース
            return self._parse_response(message)

        except anthropic.RateLimitError as e:
            raise MeterReaderException(
                'RATE_LIMIT',
                'APIレート制限に達しました',
                'しばらく待ってから再度お試しください'
            )
        except anthropic.APIError as e:
            raise MeterReaderException(
                'API_ERROR',
                'Claude API エラーが発生しました',
                str(e)
            )
        except FileNotFoundError:
            raise MeterReaderException(
                'FILE_NOT_FOUND',
                '画像ファイルが見つかりません',
                f'パス: {image_path}'
            )
        except Exception as e:
            raise MeterReaderException(
                'UNKNOWN_ERROR',
                '予期しないエラーが発生しました',
                str(e)
            )

    def _build_analog_prompt(self) -> str:
        """アナログメーター用プロンプトを構築"""
        return """あなたはアナログメーター読み取りの専門家です。以下の手順で画像内のアナログメーターを正確に読み取ってください。

## 読み取り手順

### ステップ1: メーターの構造を理解
1. メーターの文字盤全体を観察
2. 目盛りの配置を確認（円形、半円形、扇形など）
3. 主要目盛り（大きな数字）と補助目盛り（小さな線）を識別
4. スケールの最小値と最大値を読み取る

### ステップ2: 針の位置を正確に特定
1. 針の先端が指している位置を慎重に観察
2. 針が2つの目盛りの間にある場合は、その中間位置を推定
3. 目盛り間の距離を等分に考えて補間計算
4. 針の影や反射に惑わされないよう注意

### ステップ3: 数値を計算
1. 針が指している主要目盛りを特定
2. 主要目盛り間の補助目盛りの数を数える
3. 針の位置から補間して正確な値を計算
   例: 目盛りが0, 10, 20で針が0と10の中間なら5
   例: 目盛りが0, 100, 200で針が100と200の3/4位置なら175
4. 小数点以下の精度も考慮（目盛りの密度に応じて）

### ステップ4: 検証
1. 計算した値がスケール範囲内にあることを確認
2. 針の角度と値の関係が妥当か確認
3. 単位が表示されている場合は記録

## 注意事項
- 針の影や光の反射を針本体と間違えない
- パースペクティブ（角度）による歪みを考慮
- 複数の針がある場合は、最も太い針または主要な針を読み取る
- 不明確な場合は confidence を "medium" または "low" に設定

## 出力形式
必ず以下のJSON形式で返してください（他のテキストは含めない）:

```json
{
  "value": 数値（数値型、小数点含む）,
  "unit": "単位文字列（例: kWh, MPa, ℃など。不明な場合は空文字列）",
  "scale_range": "最小値-最大値（例: 0-100）",
  "needle_position": "針の位置の詳細説明（例: スケールの約62%の位置、100と200の中間より少し上）",
  "confidence": "high/medium/low（high: 針が明確で目盛りも読みやすい、medium: やや不鮮明、low: 判読困難）",
  "notes": "追加の観察事項（例: 反射あり、針が目盛りの間など）"
}
```

エラーの場合:
```json
{
  "error": "エラーの詳細説明（メーターが写っていない、画像が不鮮明など）",
  "confidence": "low"
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

    def _parse_response(self, message) -> Dict[str, Any]:
        """
        Claude APIのレスポンスからJSONを抽出してパース

        Args:
            message: Claude APIからのメッセージオブジェクト

        Returns:
            パースされた辞書

        Raises:
            MeterReaderException: パースエラー時
        """
        try:
            # レスポンステキストを取得
            response_text = message.content[0].text

            # JSONブロックを抽出（```json ... ``` または {...} の形式）
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
            if json_match:
                json_text = json_match.group(1)
            else:
                # JSONブロックがない場合は、{ } で囲まれた部分を抽出
                json_match = re.search(r'\{[\s\S]*\}', response_text)
                if json_match:
                    json_text = json_match.group(0)
                else:
                    raise MeterReaderException(
                        'PARSE_ERROR',
                        'レスポンスからJSONを抽出できませんでした',
                        response_text
                    )

            # JSONをパース
            data = json.loads(json_text)

            # エラーレスポンスの場合
            if 'error' in data:
                raise MeterReaderException(
                    'NO_METER_DETECTED',
                    'メーターを読み取れませんでした',
                    data.get('error', '不明なエラー')
                )

            return data

        except json.JSONDecodeError as e:
            raise MeterReaderException(
                'PARSE_ERROR',
                'JSONのパースに失敗しました',
                str(e)
            )
        except (IndexError, AttributeError) as e:
            raise MeterReaderException(
                'PARSE_ERROR',
                'レスポンスの形式が不正です',
                str(e)
            )

    def _get_media_type(self, image_path: str) -> str:
        """
        ファイル拡張子からメディアタイプを取得

        Args:
            image_path: 画像ファイルのパス

        Returns:
            メディアタイプ文字列
        """
        ext = Path(image_path).suffix.lower()
        mapping = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png'
        }
        return mapping.get(ext, 'image/jpeg')
