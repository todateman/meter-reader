import cv2
import numpy as np
import math
from typing import Dict, Any, Optional, Tuple


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

        needle_angle, tip_x, tip_y, line = needle_info

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
            "crop_path": crop_path
        }

    def _crop_tip_area(self, img: np.ndarray, cx: int, cy: int,
                       radius: int, tip_x: int, tip_y: int) -> Optional[str]:
        """針先端とスケール目盛りの交差部分をクロップして保存"""
        import tempfile
        import os

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

        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=100,
            param1=100, param2=50,
            minRadius=80, maxRadius=300
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)
            # 小さすぎる円を除外（画像幅の1/6以上）
            min_radius = width // 6
            candidates = [c for c in circles if c[2] >= min_radius]
            if not candidates:
                candidates = list(circles)
            # 画像中心に最も近い円を選択
            best = min(candidates, key=lambda c: math.sqrt(
                (c[0] - img_cx) ** 2 + (c[1] - img_cy) ** 2
            ))
            return (int(best[0]), int(best[1])), int(best[2])

        return None, None

    def _detect_needle(self, img: np.ndarray, cx: int, cy: int, radius: int) -> Optional[Tuple[float, int, int, Tuple]]:
        """
        ハフ変換で針を検出し、角度を計算する

        Returns:
            (角度, 先端x, 先端y, (x1,y1,x2,y2)) または None
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
            threshold=30,
            minLineLength=int(radius * 0.25),
            maxLineGap=10
        )

        if lines is None:
            return None

        # 針候補の選別
        best_line = None
        best_score = float('inf')

        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx, dy = x2 - x1, y2 - y1
            line_len = math.sqrt(dx * dx + dy * dy)

            if line_len < radius * 0.25:
                continue

            # 中心点から直線への距離
            dist = abs(dy * cx - dx * cy + x2 * y1 - y2 * x1) / line_len

            # 中心に十分近い直線のみ
            if dist > radius * 0.2:
                continue

            # スコア: 中心への距離が小さく、長い線を優先
            score = dist / line_len
            if score < best_score:
                best_score = score
                best_line = (x1, y1, x2, y2)

        if best_line is None:
            return None

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

        return angle_deg, tip_x, tip_y, best_line
