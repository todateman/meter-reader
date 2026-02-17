import cv2
import numpy as np
import math
import os
import tempfile
import uuid
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image, ImageDraw, ImageFont


class NeedleDetector:
    """OpenCVを使用してアナログメーターの針を検出し、角度を計算する"""

    def __init__(self, target_width: int = 640):
        """
        Args:
            target_width: リサイズ後の横幅（px）
        """
        self.target_width = target_width

    def detect(self, image_path: str) -> Dict[str, Any]:
        """
        画像からメーターの針を検出し、角度を計算する

        Args:
            image_path: 画像ファイルのパス

        Returns:
            検出結果の辞書
        """
        img = cv2.imread(image_path)
        if img is None:
            return {"success": False, "error": "画像を読み込めませんでした"}

        # リサイズ
        img = self._resize(img)
        height, width = img.shape[:2]

        # メーター領域（円）の検出
        center, radius = self._detect_circle(img)
        if center is None:
            center = (width // 2, height // 2)
            radius = min(width, height) // 3

        cx, cy = center

        # 針の検出
        needle_info = self._detect_needle(img, cx, cy, radius)

        if needle_info is None:
            return {
                "success": False,
                "error": "針を検出できませんでした",
                "center": [cx, cy],
                "radius": radius
            }

        needle_angle = needle_info["needle_angle"]
        tip_x, tip_y = needle_info["tip"]
        line = needle_info["line"]
        candidate_lines = needle_info.get("candidate_lines", [])
        selection_reason = needle_info.get("selection_reason", "")

        # 角度を時計の位置に変換
        # 座標系: 0°=6時, 90°=9時, 180°=12時, 270°=3時
        clock_hour = ((needle_angle / 30.0) + 6) % 12
        clock_h = int(clock_hour)
        clock_m = int((clock_hour - clock_h) * 60)
        if clock_h == 0:
            clock_h = 12
        clock_position = f"{clock_h}:{clock_m:02d}"

        # 260°スイープ（標準）でposition_ratioを計算
        min_angle = 50.0  # 180 - 260/2
        max_angle = 310.0  # 180 + 260/2
        sweep = max_angle - min_angle
        position_ratio = (needle_angle - min_angle) / sweep
        position_ratio = max(0.0, min(1.0, position_ratio))

        # 針先端周辺をクロップ（スケール目盛りとの交差部分を拡大）
        crop_path = self._crop_tip_area(img, cx, cy, radius, tip_x, tip_y)
        debug_image_path = self._save_debug_image(
            img,
            cx,
            cy,
            radius,
            tip_x,
            tip_y,
            line,
            candidate_lines,
            selection_reason,
        )

        return {
            "success": True,
            "needle_angle": round(float(needle_angle), 1),
            "clock_position": clock_position,
            "position_ratio": round(float(position_ratio), 4),
            "position_percent": round(float(position_ratio * 100), 1),
            "center": [int(cx), int(cy)],
            "radius": int(radius),
            "tip": [int(tip_x), int(tip_y)],
            "needle_line": [int(v) for v in line],
            "crop_path": crop_path,
            "debug_image_path": debug_image_path,
            "debug_selection_reason": selection_reason
        }

    def _save_debug_image(
        self,
        img: np.ndarray,
        cx: int,
        cy: int,
        radius: int,
        tip_x: int,
        tip_y: int,
        line: Tuple,
        candidate_lines: List[Dict[str, Any]],
        selection_reason: str,
    ) -> Optional[str]:
        """針検出結果のデバッグ画像を保存"""
        try:
            debug_img = img.copy()
            x1, y1, x2, y2 = line

            for candidate in candidate_lines:
                lx1, ly1, lx2, ly2 = candidate["line"]
                if candidate.get("primary_eligible"):
                    color = (0, 165, 255)
                    thickness = 2
                else:
                    color = (120, 120, 120)
                    thickness = 1
                cv2.line(debug_img, (lx1, ly1), (lx2, ly2), color, thickness)

            cv2.circle(debug_img, (cx, cy), radius, (255, 0, 0), 2)
            cv2.circle(debug_img, (cx, cy), 5, (0, 255, 255), -1)
            cv2.line(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.circle(debug_img, (tip_x, tip_y), 6, (0, 0, 255), -1)

            cv2.putText(
                debug_img,
                "needle",
                (tip_x + 8, tip_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            debug_img = self._draw_debug_text_with_pillow(debug_img, selection_reason)

            file_name = f"meter_debug_{uuid.uuid4().hex}.jpg"
            debug_path = os.path.join(tempfile.gettempdir(), file_name)
            cv2.imwrite(debug_path, debug_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            return debug_path
        except Exception:
            return None

    def _draw_debug_text_with_pillow(self, debug_img: np.ndarray, selection_reason: str) -> np.ndarray:
        """凡例・選定理由をPillowで描画（日本語対応）"""
        try:
            font = self._get_japanese_font(22)
            small_font = self._get_japanese_font(20)

            pil_img = Image.fromarray(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)

            legend_lines = [
                ("中心: 黄色", (255, 255, 0)),
                ("候補線: グレー", (160, 160, 160)),
                ("一次候補(中心近傍かつ外周到達): オレンジ", (255, 165, 0)),
                ("採用線: 緑", (0, 255, 0)),
            ]

            max_width = max(220, debug_img.shape[1] - 24)
            wrapped = self._wrap_text(selection_reason, max_chars=max_width // 18) if selection_reason else []

            panel_x = 8
            panel_y = 8
            panel_width = min(debug_img.shape[1] - 16, max_width)
            panel_height = 4 + (len(legend_lines) * 30) + (len(wrapped[:4]) * 28) + 16

            panel = Image.new("RGBA", (panel_width, panel_height), (0, 0, 0, 135))
            pil_img_rgba = pil_img.convert("RGBA")
            pil_img_rgba.alpha_composite(panel, dest=(panel_x, panel_y))
            pil_img = pil_img_rgba.convert("RGB")
            draw = ImageDraw.Draw(pil_img)

            y = 12
            x = 12
            for text, color in legend_lines:
                draw.text((x + 1, y + 1), text, fill=(0, 0, 0), font=font)
                draw.text((x, y), text, fill=color, font=font)
                y += 30

            if wrapped:
                for line_text in wrapped[:4]:
                    draw.text((x + 1, y + 1), line_text, fill=(0, 0, 0), font=small_font)
                    draw.text((x, y), line_text, fill=(255, 255, 255), font=small_font)
                    y += 28

            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            return debug_img

    def _get_japanese_font(self, size: int) -> ImageFont.ImageFont:
        """日本語描画用フォントを取得（環境に応じてフォールバック）"""
        font_candidates = []
        windir = os.environ.get('WINDIR', r'C:\Windows')
        font_dir = os.path.join(windir, 'Fonts')
        font_candidates.extend([
            os.path.join(font_dir, 'YuGothM.ttc'),
            os.path.join(font_dir, 'YuGothR.ttc'),
            os.path.join(font_dir, 'meiryo.ttc'),
            os.path.join(font_dir, 'msgothic.ttc'),
            os.path.join(font_dir, 'msyh.ttc'),
            '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
            '/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc',
            '/System/Library/Fonts/Hiragino Sans GB.ttc',
        ])

        for font_path in font_candidates:
            if os.path.exists(font_path):
                try:
                    return ImageFont.truetype(font_path, size=size)
                except Exception:
                    continue

        return ImageFont.load_default()

    def _wrap_text(self, text: str, max_chars: int) -> List[str]:
        """描画用にテキストを簡易折り返し"""
        if max_chars <= 0:
            return [text]
        if len(text) <= max_chars:
            return [text]

        if ' ' not in text:
            return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

        words = text.split(' ')
        lines = []
        current = []
        current_len = 0
        for word in words:
            if current_len + len(word) + (1 if current else 0) > max_chars:
                lines.append(' '.join(current))
                current = [word]
                current_len = len(word)
            else:
                current.append(word)
                current_len += len(word) + (1 if current_len > 0 else 0)
        if current:
            lines.append(' '.join(current))
        return lines

    def _crop_tip_area(self, img: np.ndarray, cx: int, cy: int,
                       radius: int, tip_x: int, tip_y: int) -> Optional[str]:
        """針先端とスケール目盛りの交差部分をクロップして保存"""
        try:
            h, w = img.shape[:2]
            # 針の方向に沿ってスケール目盛りとの交差点付近をクロップ
            # 中心からtipへの方向の、radius*0.7の位置を中心にする
            dx = tip_x - cx
            dy = tip_y - cy
            dist = math.sqrt(dx * dx + dy * dy)
            if dist == 0:
                return None
            # スケール目盛り付近（radius*0.7）を中心に
            scale_x = int(cx + dx / dist * radius * 0.7)
            scale_y = int(cy + dy / dist * radius * 0.7)

            crop_size = int(radius * 0.35)
            x1 = max(0, scale_x - crop_size)
            y1 = max(0, scale_y - crop_size)
            x2 = min(w, scale_x + crop_size)
            y2 = min(h, scale_y + crop_size)

            cropped = img[y1:y2, x1:x2]
            if cropped.size == 0:
                return None

            # 3倍に拡大
            cropped = cv2.resize(cropped, (cropped.shape[1] * 3, cropped.shape[0] * 3),
                                 interpolation=cv2.INTER_CUBIC)

            crop_path = os.path.join(tempfile.gettempdir(), 'meter_tip_crop.jpg')
            cv2.imwrite(crop_path, cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
            return crop_path
        except Exception:
            return None

    def _resize(self, img: np.ndarray) -> np.ndarray:
        """横幅を指定サイズにリサイズ"""
        height, width = img.shape[:2]
        scale = self.target_width / width
        return cv2.resize(img, (self.target_width, int(height * scale)))

    def _detect_circle(self, img: np.ndarray) -> Tuple[Optional[Tuple[int, int]], Optional[int]]:
        """ハフ変換でメーターの円を検出（画像中心に近く大きい円を優先）"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        height, width = img.shape[:2]
        img_cx, img_cy = width // 2, height // 2
        min_dim = min(width, height)

        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=max(60, min_dim // 8),
            param1=100, param2=40,
            minRadius=max(30, int(min_dim * 0.08)),
            maxRadius=int(min_dim * 0.48)
        )

        # 候補が少ない場合は閾値を緩めて再試行
        if circles is None:
            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT,
                dp=1.2, minDist=max(50, min_dim // 10),
                param1=90, param2=30,
                minRadius=max(24, int(min_dim * 0.06)),
                maxRadius=int(min_dim * 0.5)
            )

        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)
            edges = cv2.Canny(blurred, 70, 140)

            candidates = []
            for x, y, r in circles:
                # フレーム内に十分収まる円を優先
                if x - r < 2 or y - r < 2 or x + r >= width - 2 or y + r >= height - 2:
                    continue
                center_dist = math.sqrt((x - img_cx) ** 2 + (y - img_cy) ** 2)
                ring_score = self._circle_edge_score(edges, x, y, r)
                candidates.append((x, y, r, center_dist, ring_score))

            if not candidates:
                return None, None

            # 円周エッジ成立度が低い候補は除外（不完全な疑似円を抑制）
            strong_candidates = [c for c in candidates if c[4] >= 0.09]
            if strong_candidates:
                candidates = strong_candidates

            # 最終選択: 画像中心への近さを最優先、同率なら円周エッジ成立度を優先
            best = min(candidates, key=lambda c: (c[3], -c[4]))
            return (int(best[0]), int(best[1])), int(best[2])

        return None, None

    def _circle_edge_score(self, edges: np.ndarray, x: int, y: int, r: int) -> float:
        """円周付近のエッジ被覆率を計算"""
        try:
            ring_mask = np.zeros(edges.shape, dtype=np.uint8)
            thickness = max(2, int(r * 0.04))
            cv2.circle(ring_mask, (x, y), r, 255, thickness)

            ring_pixels = edges[ring_mask > 0]
            if ring_pixels.size == 0:
                return 0.0
            return float(np.count_nonzero(ring_pixels) / ring_pixels.size)
        except Exception:
            return 0.0

    def _detect_needle(self, img: np.ndarray, cx: int, cy: int, radius: int) -> Optional[Dict[str, Any]]:
        """
        ハフ変換で針を検出し、角度を計算する

        Returns:
            検出情報の辞書 または None
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # メーター領域のマスク
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask, (cx, cy), int(radius * 0.85), 255, -1)
        # 中心の小さい領域を除外（文字や固定部分を避ける）
        cv2.circle(mask, (cx, cy), int(radius * 0.1), 0, -1)
        masked = cv2.bitwise_and(gray, gray, mask=mask)

        # 二値化
        _, thresh = cv2.threshold(masked, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # メーター領域外を除外
        thresh = cv2.bitwise_and(thresh, thresh, mask=mask)

        # エッジ検出
        edges = cv2.Canny(thresh, 50, 150)

        # ハフ変換で直線検出
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=20,
            minLineLength=int(radius * 0.18),
            maxLineGap=int(radius * 0.14)
        )

        if lines is None:
            return None

        # 針候補の選別
        # 1) 中心近傍から外側へ最も長く伸びる線を優先
        # 2) 候補がない場合は従来の中心近接優先にフォールバック
        best_line = None
        best_span = -1.0
        best_dist = float('inf')
        best_far_dist = -1.0
        selected_primary = False

        fallback_line = None
        fallback_score = float('inf')
        fallback_metrics = None
        candidate_lines: List[Dict[str, Any]] = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx, dy = x2 - x1, y2 - y1
            line_len = math.sqrt(dx * dx + dy * dy)

            if line_len < radius * 0.18:
                continue

            # 中心点から直線への距離
            dist = abs(dy * cx - dx * cy + x2 * y1 - y2 * x1) / line_len

            # 中心に十分近い直線のみ
            if dist > radius * 0.22:
                continue

            # 中心からの距離（両端）
            d1 = math.sqrt((x1 - cx) ** 2 + (y1 - cy) ** 2)
            d2 = math.sqrt((x2 - cx) ** 2 + (y2 - cy) ** 2)
            near_dist = min(d1, d2)
            far_dist = max(d1, d2)
            outward_span = far_dist - near_dist

            # フォールバック用: 中心への距離が小さく、長い線を優先
            this_fallback_score = dist / line_len
            if this_fallback_score < fallback_score:
                fallback_score = this_fallback_score
                fallback_line = (x1, y1, x2, y2)
                fallback_metrics = {
                    "dist": dist,
                    "line_len": line_len,
                    "near_dist": near_dist,
                    "far_dist": far_dist,
                    "outward_span": outward_span,
                    "score": this_fallback_score,
                }

            # 優先条件:
            # - 片端が中心近傍にある
            # - もう片端が外周方向まで十分伸びている
            primary_eligible = near_dist <= radius * 0.25 and far_dist >= radius * 0.5
            candidate_lines.append({
                "line": (int(x1), int(y1), int(x2), int(y2)),
                "dist": float(dist),
                "line_len": float(line_len),
                "near_dist": float(near_dist),
                "far_dist": float(far_dist),
                "outward_span": float(outward_span),
                "primary_eligible": primary_eligible,
                "fallback_score": float(this_fallback_score),
            })

            if not primary_eligible:
                continue

            # 中心から外側への伸長量を最優先、同率なら中心に近い線を優先
            if outward_span > best_span or (math.isclose(outward_span, best_span) and dist < best_dist):
                best_span = outward_span
                best_dist = dist
                best_far_dist = far_dist
                best_line = (x1, y1, x2, y2)
                selected_primary = True

        if best_line is None:
            best_line = fallback_line
            selected_primary = False
        if best_line is None:
            return None

        candidate_lines.sort(
            key=lambda c: (
                1 if c["primary_eligible"] else 0,
                c["outward_span"],
                c["far_dist"],
                -c["dist"],
            ),
            reverse=True,
        )
        candidate_lines = candidate_lines[:20]

        x1, y1, x2, y2 = best_line

        # 針の先端を特定（中心から遠い方）
        d1 = math.sqrt((x1 - cx) ** 2 + (y1 - cy) ** 2)
        d2 = math.sqrt((x2 - cx) ** 2 + (y2 - cy) ** 2)
        if d1 > d2:
            tip_x, tip_y = x1, y1
        else:
            tip_x, tip_y = x2, y2

        # 角度計算: 真下を0°、時計回りを正
        angle_rad = math.atan2(-(tip_x - cx), (tip_y - cy))
        angle_deg = math.degrees(angle_rad)
        if angle_deg < 0:
            angle_deg += 360

        if selected_primary:
            selection_reason = (
                "選定理由: 中心近傍(near<=0.25R)かつ外周到達(far>=0.5R)の候補から、"
                f"外向き伸長量(outward_span)最大を採用。selected span={best_span:.1f}, "
                f"far={best_far_dist:.1f}, dist={best_dist:.1f}"
            )
        elif fallback_metrics:
            selection_reason = (
                "選定理由: primary条件を満たす候補が不足したため、"
                "フォールバック(dist/line_len最小)を採用。"
                f"score={fallback_metrics['score']:.4f}, span={fallback_metrics['outward_span']:.1f}, "
                f"far={fallback_metrics['far_dist']:.1f}, near={fallback_metrics['near_dist']:.1f}"
            )
        else:
            selection_reason = "選定理由: 候補線の中からフォールバック基準で採用"

        return {
            "needle_angle": float(angle_deg),
            "tip": (int(tip_x), int(tip_y)),
            "line": (int(x1), int(y1), int(x2), int(y2)),
            "candidate_lines": candidate_lines,
            "selection_reason": selection_reason,
        }
