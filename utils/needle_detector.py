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

        # 標準的なアナログメーターの位置比率を計算
        # 座標系: 真下=0°、反時計回りに増加（左=90°, 上=180°, 右=270°）
        # 一般的な工業用ゲージは約260°のスイープ
        # スケール最小値 ≈ 50°（約7:40の位置）
        # スケール最大値 ≈ 310°（約4:20の位置）
        min_angle = 50.0
        max_angle = 310.0
        sweep = max_angle - min_angle  # 260°

        # 針の角度がスイープ範囲内の何%にあるか
        position_ratio = (needle_angle - min_angle) / sweep
        position_ratio = max(0.0, min(1.0, position_ratio))

        return {
            "success": True,
            "needle_angle": round(float(needle_angle), 1),
            "position_ratio": round(float(position_ratio), 4),
            "position_percent": round(float(position_ratio * 100), 1),
            "center": [int(cx), int(cy)],
            "radius": int(radius),
            "tip": [int(tip_x), int(tip_y)],
            "needle_line": [int(v) for v in line]
        }

    def _resize(self, img: np.ndarray) -> np.ndarray:
        """横幅を指定サイズにリサイズ"""
        height, width = img.shape[:2]
        scale = self.target_width / width
        return cv2.resize(img, (self.target_width, int(height * scale)))

    def _detect_circle(self, img: np.ndarray) -> Tuple[Optional[Tuple[int, int]], Optional[int]]:
        """ハフ変換でメーターの円を検出"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=100,
            param1=100, param2=50,
            minRadius=80, maxRadius=300
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)
            best = max(circles, key=lambda c: c[2])
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
