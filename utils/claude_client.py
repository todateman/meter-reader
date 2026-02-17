import re
from typing import Dict, Any, Optional
from pathlib import Path

import cv2
import numpy as np
import pytesseract

from utils.needle_detector import NeedleDetector


class MeterReaderException(Exception):
    """メーター読み取りエラー用のカスタム例外"""

    def __init__(self, code: str, message: str, details: Optional[str] = None):
        self.code = code
        self.message = message
        self.details = details
        super().__init__(self.message)


class TesseractVisionClient:
    """Tesseract OCRを使用したメーター画像解析クライアント"""

    def __init__(self, tesseract_cmd: Optional[str] = None):
        """
        TesseractVisionClientの初期化

        Args:
            tesseract_cmd: tesseract実行ファイルパス（未指定ならPATH上のtesseractを使用）
        """
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        self.needle_detector = NeedleDetector()

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
        """アナログメーターをローカルOpenCV + Tesseract OCRで解析"""
        self._ensure_ocr_available()

        # ローカルでOpenCV針検出を実行
        opencv_result = self.needle_detector.detect(image_path)

        # OCRでスケール情報を読み取り
        scale_info = self._read_analog_scale_with_ocr(
            image_path,
            opencv_result.get("crop_path") if isinstance(opencv_result, dict) else None,
            opencv_result
        )

        # サーバー側で値を計算
        if opencv_result.get("success") and "scale_min" in scale_info and "scale_max" in scale_info:
            ratio = opencv_result["position_ratio"]
            scale_min = float(scale_info["scale_min"])
            scale_max = float(scale_info["scale_max"])
            calculated_value = scale_min + (scale_max - scale_min) * ratio

            # 小数点以下の桁数をスケールに合わせて丸め
            decimal_places = max(
                len(str(scale_min).split('.')[-1]) if '.' in str(scale_min) else 0,
                len(str(scale_max).split('.')[-1]) if '.' in str(scale_max) else 0,
            )
            calculated_value = round(calculated_value, decimal_places + 2)

            return {
                "value": calculated_value,
                "unit": scale_info.get("unit", ""),
                "scale_range": f"{scale_min}~{scale_max}",
                "needle_position": f"スケール全体の{opencv_result['position_percent']}%の位置",
                "confidence": scale_info.get("confidence", "medium"),
                "notes": f"計算: {scale_min} + ({scale_max} - {scale_min}) × {ratio} = {calculated_value}"
            }

        # OpenCV検出失敗時はOCR結果を返す（フォールバック）
        if scale_info.get("error"):
            raise MeterReaderException(
                'NO_METER_DETECTED',
                'メーターを読み取れませんでした',
                scale_info.get("error")
            )
        return scale_info

    def analyze_digital_meter(self, image_path: str) -> Dict[str, Any]:
        """7セグメントデジタルメーターをTesseract OCRで解析"""
        self._ensure_ocr_available()
        return self._read_digital_with_ocr(image_path)

    def _ensure_ocr_available(self) -> None:
        """Tesseract OCRエンジンが利用可能かチェック"""
        try:
            _ = pytesseract.get_tesseract_version()
        except Exception as e:
            raise MeterReaderException(
                'OCR_ENGINE_NOT_AVAILABLE',
                'Tesseract OCRエンジンを利用できません',
                str(e)
            )

    def _read_analog_scale_with_ocr(
        self,
        image_path: str,
        crop_path: Optional[str],
        opencv_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """OCRでアナログメーターの最小値・最大値・単位を推定"""
        image = cv2.imread(image_path)
        if image is None:
            raise MeterReaderException(
                'FILE_NOT_FOUND',
                '画像ファイルが見つかりません',
                f'パス: {image_path}'
            )

        text_regions = self._extract_analog_text_regions(image)
        if crop_path:
            crop_img = cv2.imread(crop_path)
            if crop_img is not None:
                text_regions["tip_crop"] = crop_img

        tokens = []
        for region in text_regions.values():
            tokens.extend(self._collect_ocr_tokens(region, digit_focused=False))

        left_numbers = self._extract_numeric_candidates_from_tokens(
            self._collect_ocr_tokens(text_regions.get("left_bottom", image), digit_focused=False)
        )
        right_numbers = self._extract_numeric_candidates_from_tokens(
            self._collect_ocr_tokens(text_regions.get("right_bottom", image), digit_focused=False)
        )
        all_numbers = self._extract_numeric_candidates_from_tokens(tokens)
        numbers = sorted(set(left_numbers + right_numbers + all_numbers))
        merged_text = "\n".join(token["text"] for token in tokens if token.get("text", "")).strip()

        if len(numbers) < 2:
            return {
                "error": "スケールの最小値・最大値をOCRで判定できませんでした",
                "confidence": "low",
                "notes": "画像の傾き・反射・ぼけを減らして再撮影してください"
            }

        unit = self._extract_unit(merged_text)
        position_ratio = 0.5
        if isinstance(opencv_result, dict):
            position_ratio = float(opencv_result.get("position_ratio", 0.5))

        scale_min, scale_max = self._select_scale_bounds(
            left_numbers,
            right_numbers,
            all_numbers,
            unit,
            position_ratio
        )

        confidence = "high" if len(numbers) >= 4 else "medium"

        return {
            "scale_min": scale_min,
            "scale_max": scale_max,
            "unit": unit,
            "confidence": confidence,
            "notes": f"OCR抽出候補数: {len(numbers)} / 文字候補数: {len(tokens)}"
        }

    def _read_digital_with_ocr(self, image_path: str) -> Dict[str, Any]:
        """OCRでデジタルメーターの数値を読み取り"""
        image = cv2.imread(image_path)
        if image is None:
            raise MeterReaderException(
                'FILE_NOT_FOUND',
                '画像ファイルが見つかりません',
                f'パス: {image_path}'
            )

        all_text = "\n".join(token["text"] for token in self._collect_ocr_tokens(image, digit_focused=False))
        unit = self._extract_unit(all_text)

        # 7セグ専用デコーダを優先
        seven_seg_result = self._decode_7segment_display(image)
        if seven_seg_result is not None:
            seven_seg_result["unit"] = unit
            return seven_seg_result

        roi_images = [image]
        display_roi = self._detect_digital_display_roi(image)
        if display_roi is not None:
            roi_images.insert(0, display_roi)

        token_candidates = []
        raw_texts = []
        for roi in roi_images:
            token_candidates.extend(self._collect_ocr_tokens(roi, digit_focused=True))
            raw_texts.extend(self._ocr_text_variants(roi, digit_focused=True))

        candidates = []
        for token in token_candidates:
            candidates.extend(re.findall(r'-?\d+(?:\.\d+)?', token["text"]))
        candidates.extend(self._extract_numeric_from_raw_texts(raw_texts))
        candidates = self._filter_numeric_candidates_by_unit(candidates, unit)

        best = self._select_best_numeric_token(candidates, unit)
        if best is None:
            raise MeterReaderException(
                'NO_METER_DETECTED',
                'デジタル表示の数値を読み取れませんでした',
                '表示部が鮮明に写るように再撮影してください'
            )

        try:
            value = float(best)
        except ValueError as e:
            raise MeterReaderException(
                'PARSE_ERROR',
                'OCR結果の数値変換に失敗しました',
                str(e)
            )

        decimal_places = len(best.split('.')[1]) if '.' in best else 0
        confidence = "high" if ('.' in best or len(best.replace('.', '').replace('-', '')) >= 3) else "medium"

        return {
            "value": value,
            "unit": unit,
            "decimal_places": decimal_places,
            "confidence": confidence,
            "segment_status": "OCR判定",
            "notes": f"Tesseract OCRによる読み取り（候補数: {len(candidates)}）"
        }

    def _extract_analog_text_regions(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """アナログメーター画像から数値が存在しやすい領域を切り出す"""
        regions: Dict[str, np.ndarray] = {"full": image}
        height, width = image.shape[:2]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=100,
            param1=100,
            param2=40,
            minRadius=max(40, min(width, height) // 6),
            maxRadius=max(80, min(width, height) // 2),
        )

        if circles is None:
            return regions

        circles = np.round(circles[0, :]).astype(int)
        img_cx, img_cy = width // 2, height // 2
        cx, cy, radius = min(
            circles,
            key=lambda c: (c[0] - img_cx) ** 2 + (c[1] - img_cy) ** 2
        )

        # 外周リング（目盛数字が並ぶ領域）
        ring_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(ring_mask, (cx, cy), int(radius * 0.95), 255, -1)
        cv2.circle(ring_mask, (cx, cy), int(radius * 0.5), 0, -1)
        ring = cv2.bitwise_and(image, image, mask=ring_mask)

        x1 = max(0, cx - int(radius * 1.05))
        y1 = max(0, cy - int(radius * 1.05))
        x2 = min(width, cx + int(radius * 1.05))
        y2 = min(height, cy + int(radius * 1.05))
        ring_crop = ring[y1:y2, x1:x2]
        if ring_crop.size > 0:
            regions["ring"] = ring_crop

        # 左下と右下（最小値・最大値が出やすい領域）
        left_bottom = image[
            max(0, cy - int(radius * 0.1)):min(height, cy + int(radius * 1.0)),
            max(0, cx - int(radius * 1.1)):max(1, cx + int(radius * 0.05))
        ]
        right_bottom = image[
            max(0, cy - int(radius * 0.1)):min(height, cy + int(radius * 1.0)),
            min(width - 1, cx - int(radius * 0.05)):min(width, cx + int(radius * 1.1))
        ]
        if left_bottom.size > 0:
            regions["left_bottom"] = left_bottom
        if right_bottom.size > 0:
            regions["right_bottom"] = right_bottom

        return regions

    def _select_scale_bounds(
        self,
        left_numbers: list[float],
        right_numbers: list[float],
        all_numbers: list[float],
        unit: str,
        position_ratio: float
    ) -> tuple[float, float]:
        """左右領域のOCR結果を優先して最小値・最大値を推定"""
        unit_l = (unit or "").lower()

        if unit_l in {"mpa", "kpa", "pa", "bar", "a"}:
            left_small = [v for v in left_numbers if -10.0 <= v <= 10.0]
            right_small = [v for v in right_numbers if 0.0 < v <= 10.0]
            all_small = [v for v in all_numbers if -10.0 <= v <= 10.0]

            if left_small and right_small and max(right_small) > min(left_small):
                return min(left_small), max(right_small)

            if right_small:
                max_v = max(right_small)
                if position_ratio <= 0.2 and max_v <= 1.0:
                    return -max_v, max_v
                return 0.0, max_v

            if len(all_small) >= 2:
                return min(all_small), max(all_small)

        if unit_l in {"km/h", "kmh"}:
            speed_max_candidates = [v for v in all_numbers if 20.0 <= v <= 260.0]
            if speed_max_candidates:
                return 0.0, max(speed_max_candidates)

        if all_numbers:
            high_candidates = [v for v in all_numbers if v >= 20.0]
            if high_candidates:
                max_v = max(high_candidates)
                positives = sorted(set(v for v in all_numbers if v >= 0.0))
                diffs = [
                    positives[i + 1] - positives[i]
                    for i in range(len(positives) - 1)
                    if 5.0 <= (positives[i + 1] - positives[i]) <= 60.0
                ]
                if diffs and position_ratio >= 0.6:
                    step = float(np.median(diffs))
                    if max_v >= 100.0 and step < 10.0:
                        step = 20.0
                    estimated_max = max_v + step
                    if estimated_max <= 260.0:
                        return 0.0, estimated_max
                return 0.0, max_v

        if left_numbers and right_numbers:
            left_min = min(left_numbers)
            right_max = max(right_numbers)
            if right_max > left_min:
                return left_min, right_max

        if len(all_numbers) >= 2:
            sorted_vals = sorted(all_numbers)
            if len(sorted_vals) >= 4:
                q1 = sorted_vals[len(sorted_vals) // 4]
                q3 = sorted_vals[(len(sorted_vals) * 3) // 4]
                filtered = [v for v in sorted_vals if q1 - 100 <= v <= q3 + 100]
                if len(filtered) >= 2:
                    return min(filtered), max(filtered)
            return min(sorted_vals), max(sorted_vals)

        if left_numbers:
            v = left_numbers[0]
            return v, v + 1.0
        if right_numbers:
            v = right_numbers[0]
            return 0.0, v

        return 0.0, 1.0

    def _detect_digital_display_roi(self, image: np.ndarray) -> Optional[np.ndarray]:
        """7セグ表示領域らしい矩形をOpenCVで抽出"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)

        # 明背景・暗背景の双方を想定
        th1 = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 9
        )
        th2 = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 21, 9
        )

        height, width = gray.shape[:2]
        min_area = max(500, int(width * height * 0.02))
        kernels = [
            cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5)),
            cv2.getStructuringElement(cv2.MORPH_RECT, (31, 7))
        ]

        best_rect = None
        best_score = -1.0

        for base in (th1, th2):
            for kernel in kernels:
                merged = cv2.morphologyEx(base, cv2.MORPH_CLOSE, kernel, iterations=2)
                contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    area = w * h
                    if area < min_area:
                        continue
                    aspect = w / max(h, 1)
                    if aspect < 1.3 or aspect > 12.0:
                        continue

                    roi = gray[y:y + h, x:x + w]
                    if roi.size == 0:
                        continue

                    edge_score = cv2.Canny(roi, 50, 150).mean()
                    center_bias = 1.0 - (abs((x + w / 2) - width / 2) / (width / 2 + 1e-6))
                    score = edge_score * 0.7 + center_bias * 30.0
                    if score > best_score:
                        best_score = score
                        best_rect = (x, y, w, h)

        if best_rect is None:
            return None

        x, y, w, h = best_rect
        pad_x = int(w * 0.08)
        pad_y = int(h * 0.2)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(width, x + w + pad_x)
        y2 = min(height, y + h + pad_y)
        roi = image[y1:y2, x1:x2]
        return roi if roi.size > 0 else None

    def _decode_7segment_display(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """輪郭→セグメント判定で7セグ数字を復元（小数点・符号含む）"""
        roi = self._detect_digital_display_roi(image)
        if roi is None:
            roi = image

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        enlarged = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        blur = cv2.GaussianBlur(enlarged, (3, 3), 0)

        _, otsu_bin = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_inv = cv2.bitwise_not(otsu_bin)

        bin_candidates = [
            cv2.adaptiveThreshold(
                blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 31, 9
            ),
            cv2.adaptiveThreshold(
                blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 31, 9
            ),
            otsu_inv,
            otsu_bin
        ]

        best_decoded = None
        best_score = -1
        for binary in bin_candidates:
            cleaned = cv2.morphologyEx(
                binary,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                iterations=1
            )
            decoded = self._decode_7segment_from_binary(cleaned)
            if decoded is None:
                continue
            score = decoded.get("decoded_digits", 0) * 10 - decoded.get("unknown_digits", 0) * 7
            if decoded.get("has_decimal", False):
                score += 3
            if score > best_score:
                best_score = score
                best_decoded = decoded

        if best_decoded is None:
            return None

        if int(best_decoded.get("decoded_digits", 0)) < 2:
            return None

        value_str = best_decoded.get("value_str", "")
        try:
            value = float(value_str)
        except ValueError:
            return None

        decimal_places = len(value_str.split('.')[1]) if '.' in value_str else 0
        decoded_digits = int(best_decoded.get("decoded_digits", 0))
        unknown_digits = int(best_decoded.get("unknown_digits", 0))
        confidence = "high" if decoded_digits >= 3 and unknown_digits == 0 else "medium"

        return {
            "value": value,
            "unit": "",
            "decimal_places": decimal_places,
            "confidence": confidence,
            "segment_status": "7セグ判定",
            "notes": f"7セグデコーダ（桁数: {decoded_digits}, 未確定: {unknown_digits}, 小数点: {best_decoded.get('has_decimal', False)}）"
        }

    def _decode_7segment_from_binary(self, binary: np.ndarray) -> Optional[Dict[str, Any]]:
        """二値画像から7セグ桁を抽出して数値文字列に復元"""
        boxes = self._find_7segment_digit_boxes(binary)
        if not boxes:
            return None

        decoded_chars: list[str] = []
        unknown_digits = 0
        for box in boxes:
            digit = self._decode_single_7segment_digit(binary, box)
            if digit is None:
                unknown_digits += 1
                continue
            decoded_chars.append(str(digit))

        if not decoded_chars:
            return None

        decimal_indexes = self._detect_decimal_positions(binary, boxes)
        has_decimal = len(decimal_indexes) > 0

        value_parts: list[str] = []
        for idx, ch in enumerate(decoded_chars):
            value_parts.append(ch)
            if idx in decimal_indexes:
                value_parts.append('.')

        value_str = ''.join(value_parts)
        if value_str.endswith('.'):
            value_str = value_str[:-1]
        if not value_str:
            return None

        if self._detect_negative_sign(binary, boxes):
            value_str = '-' + value_str

        return {
            "value_str": value_str,
            "decoded_digits": len(decoded_chars),
            "unknown_digits": unknown_digits,
            "has_decimal": has_decimal
        }

    def _find_7segment_digit_boxes(self, binary: np.ndarray) -> list[tuple[int, int, int, int]]:
        """7セグ桁の候補矩形を抽出"""
        h, w = binary.shape[:2]
        joined = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 9)),
            iterations=1
        )
        contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes: list[tuple[int, int, int, int]] = []
        min_area = max(80, int(w * h * 0.002))
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            area = bw * bh
            if area < min_area:
                continue
            if bh < h * 0.28:
                continue
            aspect = bw / max(bh, 1)
            if aspect < 0.12 or aspect > 0.95:
                continue
            boxes.append((x, y, bw, bh))

        boxes.sort(key=lambda b: b[0])
        if len(boxes) <= 1:
            projection_boxes = self._find_digit_boxes_by_projection(binary)
            if len(projection_boxes) > len(boxes):
                return projection_boxes
            return boxes

        merged: list[tuple[int, int, int, int]] = []
        for box in boxes:
            if not merged:
                merged.append(box)
                continue
            px, py, pw, ph = merged[-1]
            x, y, bw, bh = box
            gap = x - (px + pw)
            if gap <= max(3, int(min(pw, bw) * 0.08)) and abs(y - py) < max(ph, bh) * 0.35:
                nx = px
                ny = min(py, y)
                nw = max(px + pw, x + bw) - nx
                nh = max(py + ph, y + bh) - ny
                merged[-1] = (nx, ny, nw, nh)
            else:
                merged.append(box)

        if len(merged) <= 1:
            projection_boxes = self._find_digit_boxes_by_projection(binary)
            if len(projection_boxes) > len(merged):
                return projection_boxes

        return merged

    def _find_digit_boxes_by_projection(self, binary: np.ndarray) -> list[tuple[int, int, int, int]]:
        """列方向投影で桁の横位置を推定（輪郭抽出のフォールバック）"""
        h, w = binary.shape[:2]
        col_sum = np.sum(binary > 0, axis=0).astype(np.float32)
        if col_sum.max() <= 0:
            return []

        window = np.ones(9, dtype=np.float32) / 9.0
        smooth = np.convolve(col_sum, window, mode='same')
        threshold = max(3.0, float(smooth.max() * 0.12))
        active = smooth > threshold

        runs: list[tuple[int, int]] = []
        start = None
        for idx, is_on in enumerate(active):
            if is_on and start is None:
                start = idx
            elif not is_on and start is not None:
                runs.append((start, idx - 1))
                start = None
        if start is not None:
            runs.append((start, len(active) - 1))

        min_width = max(8, int(w * 0.03))
        boxes: list[tuple[int, int, int, int]] = []
        for x1, x2 in runs:
            if (x2 - x1 + 1) < min_width:
                continue
            strip = binary[:, x1:x2 + 1]
            row_sum = np.sum(strip > 0, axis=1)
            rows = np.where(row_sum > max(2, row_sum.max() * 0.15))[0]
            if rows.size == 0:
                continue
            y1, y2 = int(rows.min()), int(rows.max())
            bh = y2 - y1 + 1
            bw = x2 - x1 + 1
            if bh < h * 0.25:
                continue
            boxes.append((x1, y1, bw, bh))

        return boxes

    def _decode_single_7segment_digit(self, binary: np.ndarray, box: tuple[int, int, int, int]) -> Optional[int]:
        """1桁の7セグ点灯状態から数字を復元"""
        x, y, w, h = box
        roi = binary[y:y + h, x:x + w]
        if roi.size == 0:
            return None

        seg_rects = [
            (int(w * 0.20), int(h * 0.04), int(w * 0.80), int(h * 0.18)),
            (int(w * 0.75), int(h * 0.16), int(w * 0.96), int(h * 0.47)),
            (int(w * 0.75), int(h * 0.54), int(w * 0.96), int(h * 0.87)),
            (int(w * 0.20), int(h * 0.82), int(w * 0.80), int(h * 0.96)),
            (int(w * 0.04), int(h * 0.54), int(w * 0.25), int(h * 0.87)),
            (int(w * 0.04), int(h * 0.16), int(w * 0.25), int(h * 0.47)),
            (int(w * 0.20), int(h * 0.44), int(w * 0.80), int(h * 0.58)),
        ]

        states: list[int] = []
        for x1, y1, x2, y2 in seg_rects:
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(x1 + 1, min(w, x2))
            y2 = max(y1 + 1, min(h, y2))
            seg = roi[y1:y2, x1:x2]
            if seg.size == 0:
                states.append(0)
                continue
            on_ratio = float(cv2.countNonZero(seg)) / float(seg.size)
            states.append(1 if on_ratio >= 0.22 else 0)

        pattern = tuple(states)
        segment_map = {
            (1, 1, 1, 1, 1, 1, 0): 0,
            (0, 1, 1, 0, 0, 0, 0): 1,
            (1, 1, 0, 1, 1, 0, 1): 2,
            (1, 1, 1, 1, 0, 0, 1): 3,
            (0, 1, 1, 0, 0, 1, 1): 4,
            (1, 0, 1, 1, 0, 1, 1): 5,
            (1, 0, 1, 1, 1, 1, 1): 6,
            (1, 1, 1, 0, 0, 0, 0): 7,
            (1, 1, 1, 1, 1, 1, 1): 8,
            (1, 1, 1, 1, 0, 1, 1): 9,
        }
        if pattern in segment_map:
            return segment_map[pattern]

        # ハミング距離1以内で救済
        best_digit = None
        best_distance = 8
        for ref_pattern, ref_digit in segment_map.items():
            distance = sum(1 for a, b in zip(pattern, ref_pattern) if a != b)
            if distance < best_distance:
                best_distance = distance
                best_digit = ref_digit

        if best_digit is not None and best_distance <= 1:
            return best_digit
        return None

    def _detect_decimal_positions(self, binary: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> set[int]:
        """桁間の小数点位置を検出（どの桁の直後かを返す）"""
        decimal_after: set[int] = set()
        if len(boxes) < 2:
            return decimal_after

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area < 6 or area > 500:
                continue
            aspect = w / max(h, 1)
            if aspect < 0.5 or aspect > 1.8:
                continue

            cx = x + w / 2.0
            cy = y + h / 2.0
            for idx in range(len(boxes) - 1):
                x1, y1, w1, h1 = boxes[idx]
                x2, y2, w2, h2 = boxes[idx + 1]
                right_edge = x1 + w1
                left_next = x2
                if right_edge <= cx <= left_next + max(3, w2 * 0.2):
                    if cy >= y1 + h1 * 0.62:
                        decimal_after.add(idx)

        return decimal_after

    def _detect_negative_sign(self, binary: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> bool:
        """先頭付近の横棒から負号を判定"""
        if not boxes:
            return False

        first_x, first_y, first_w, first_h = boxes[0]
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if x >= first_x:
                continue
            if w < max(6, first_w * 0.25):
                continue
            if h > first_h * 0.25:
                continue
            if y + h < first_y + first_h * 0.35 or y > first_y + first_h * 0.70:
                continue
            if w / max(h, 1) >= 1.8:
                return True
        return False

    def _collect_ocr_tokens(self, image: np.ndarray, digit_focused: bool) -> list[Dict[str, Any]]:
        """複数の前処理・PSMでOCRを実行し、信頼度付きトークンを返す"""
        if image.size == 0:
            return []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        enlarged = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        denoised = cv2.bilateralFilter(enlarged, 7, 50, 50)

        _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        inv = cv2.bitwise_not(otsu)
        adapt = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 9
        )

        variants = [enlarged, otsu, inv, adapt]
        psm_modes = [7, 8, 6, 11] if digit_focused else [6, 11]

        results: list[Dict[str, Any]] = []
        seen = set()
        for variant in variants:
            for psm in psm_modes:
                config = f'--oem 3 --psm {psm}'
                if digit_focused:
                    config += ' -c tessedit_char_whitelist=0123456789.-'
                else:
                    config += ' -c tessedit_char_whitelist=0123456789.-/%kPaMPVHzCWhbarm'

                data = pytesseract.image_to_data(variant, config=config, output_type=pytesseract.Output.DICT)
                n = len(data.get("text", []))
                for i in range(n):
                    text = (data["text"][i] or "").strip()
                    if not text:
                        continue
                    try:
                        conf = float(data["conf"][i])
                    except Exception:
                        conf = -1.0
                    if conf < 20:
                        continue

                    key = (text, int(data["left"][i]), int(data["top"][i]), psm)
                    if key in seen:
                        continue
                    seen.add(key)

                    results.append({
                        "text": text,
                        "conf": conf,
                        "left": int(data["left"][i]),
                        "top": int(data["top"][i]),
                        "width": int(data["width"][i]),
                        "height": int(data["height"][i]),
                        "psm": psm,
                    })

        return results

    def _extract_numeric_candidates_from_tokens(self, tokens: list[Dict[str, Any]]) -> list[float]:
        """OCRトークンから数値候補を抽出"""
        numbers = []
        for token in tokens:
            text = token.get("text", "")
            conf = float(token.get("conf", 0))
            if conf < 25:
                continue

            for raw in re.findall(r'-?\d+(?:\.\d+)?', text):
                try:
                    value = float(raw)
                except ValueError:
                    continue

                if abs(value) > 1000:
                    continue
                numbers.append(value)

        return sorted(set(numbers))

    def _extract_numeric_from_raw_texts(self, texts: list[str]) -> list[str]:
        """OCR生テキストから数値候補を再構成して抽出"""
        candidates: list[str] = []
        for text in texts:
            cleaned = text.replace(' ', '').replace('\n', '')
            cleaned = cleaned.replace('O', '0').replace('o', '0').replace(',', '.')
            cleaned = re.sub(r'[^0-9.\-]', '', cleaned)
            if not cleaned:
                continue

            if '-' in cleaned and not cleaned.startswith('-'):
                cleaned = '-' + cleaned.replace('-', '')

            if cleaned.count('.') > 1:
                first = cleaned.find('.')
                cleaned = cleaned[:first + 1] + cleaned[first + 1:].replace('.', '')

            for token in re.findall(r'-?\d+(?:\.\d+)?|-?\d+\.', cleaned):
                if token.endswith('.'):
                    token = token + '0'
                candidates.append(token)

            if '.' in text and re.fullmatch(r'-?\d{2,4}', cleaned):
                sign = '-' if cleaned.startswith('-') else ''
                digits = cleaned[1:] if sign else cleaned
                if len(digits) >= 2:
                    candidates.append(f"{sign}{digits[0]}.{digits[1:]}")

        return candidates

    def _filter_numeric_candidates_by_unit(self, candidates: list[str], unit: str) -> list[str]:
        """単位に応じて数値候補を絞り込む"""
        if not candidates:
            return []

        unit_l = (unit or "").lower()
        values = []
        for candidate in candidates:
            try:
                values.append((candidate, float(candidate)))
            except ValueError:
                continue

        if not values:
            return candidates

        if unit_l in {"mpa", "kpa", "pa", "bar"}:
            narrowed = [c for c, v in values if abs(v) <= 10.0]
            recovered = []
            for _, value in values:
                abs_v = abs(value)
                if 10.0 < abs_v <= 999.0:
                    for divider in (10.0, 100.0):
                        shifted = value / divider
                        if abs(shifted) <= 10.0:
                            recovered.append(f"{shifted:.3f}".rstrip('0').rstrip('.'))
            narrowed.extend(recovered)
            narrowed = sorted(set(narrowed))
            if narrowed:
                return narrowed

        if unit_l in {"km/h", "kmh"}:
            narrowed = [c for c, v in values if 0.0 <= v <= 300.0]
            if narrowed:
                return narrowed

        return [c for c, _ in values]

    def _ocr_text_variants(self, image: np.ndarray, digit_focused: bool) -> list[str]:
        """前処理を変えながらOCRを複数回実行し、候補テキストを返す"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)

        _, th_otsu = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th_inv = cv2.bitwise_not(th_otsu)
        th_adapt = cv2.adaptiveThreshold(
            resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 9
        )
        th_adapt_inv = cv2.adaptiveThreshold(
            resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 9
        )

        texts = []
        psm_modes = [6, 7, 8, 11, 13] if digit_focused else [6, 11]
        for img in (resized, th_otsu, th_inv, th_adapt, th_adapt_inv):
            for psm in psm_modes:
                config = f'--oem 3 --psm {psm}'
                if digit_focused:
                    config += ' -c tessedit_char_whitelist=0123456789.-'
                text = pytesseract.image_to_string(img, config=config)
                if text:
                    texts.append(text)

        return texts

    def _extract_numeric_candidates(self, text: str) -> list[float]:
        """OCRテキストから数値候補を抽出"""
        raw = re.findall(r'-?\d+(?:\.\d+)?', text)
        numbers = []
        for token in raw:
            try:
                value = float(token)
                if abs(value) <= 100000:
                    numbers.append(value)
            except ValueError:
                continue

        unique_sorted = sorted(set(numbers))
        return unique_sorted

    def _select_best_numeric_token(self, tokens: list[str], unit: str = "") -> Optional[str]:
        """デジタル表示向けに最も妥当な数値トークンを選択"""
        if not tokens:
            return None

        normalized = []
        for token in tokens:
            token = token.strip()
            if token.endswith('.'):
                token = token + '0'
            if token.count('.') > 1:
                continue
            if token.count('-') > 1:
                continue
            if '-' in token and not token.startswith('-'):
                continue
            try:
                value = float(token)
            except ValueError:
                continue
            if abs(value) > 999:
                continue
            normalized.append(token)

        if not normalized:
            return None

        unit_l = (unit or "").lower()

        def score(token: str) -> tuple[int, int, int, int, int]:
            digits = len(token.replace('-', '').replace('.', ''))
            has_decimal = 1 if '.' in token else 0
            has_sign = 1 if token.startswith('-') else 0
            decimal_places = len(token.split('.')[1]) if '.' in token else 0

            magnitude_score = 0
            try:
                value = float(token)
            except ValueError:
                value = 0.0

            if unit_l in {"mpa", "kpa", "pa", "bar"}:
                abs_v = abs(value)
                if 1.0 <= abs_v <= 10.0:
                    magnitude_score = 3
                elif 0.5 <= abs_v < 1.0:
                    magnitude_score = 2
                elif 0.1 <= abs_v < 0.5:
                    magnitude_score = 1

            return magnitude_score, has_decimal, -abs(decimal_places - 2), has_sign, -abs(digits - 4)

        return max(normalized, key=score)

    def _extract_unit(self, text: str) -> str:
        """OCRテキストから単位を抽出"""
        normalized = text.replace(' ', '').replace('\n', '')
        unit_patterns = [
            r'kWh', r'MPa', r'kPa', r'Pa', r'bar',
            r'km/h', r'kmh', r'V', r'Hz', r'%', r'℃', r'°C'
        ]
        for pattern in unit_patterns:
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            if match:
                value = match.group(0)
                if value.lower() == 'kmh':
                    return 'km/h'
                return value
        return ''

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


# 後方互換: 既存importを壊さないためのエイリアス
ClaudeVisionClient = TesseractVisionClient
