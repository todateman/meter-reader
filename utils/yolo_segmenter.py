from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


class YoloSegmenter:
    """YOLOセグメンテーションによるメーター領域・針領域の検出ヘルパー"""

    _cached_model: Any = None
    _cached_model_key: Optional[Tuple[str, str]] = None

    def __init__(
        self,
        enabled: bool = True,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu",
    ):
        self.enabled = bool(enabled)
        self.model_path = (model_path or "").strip() or "yolov8n-seg.pt"
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.device = device

    def is_available(self) -> bool:
        return self.enabled and self._load_model() is not None

    def detect_meter_circle(self, img: np.ndarray) -> Optional[Dict[str, Any]]:
        """YOLOセグメンテーションでメーター（時計/ゲージ）領域を円近似して返す"""
        prediction = self._predict(img)
        if prediction is None:
            return None

        names = prediction.names if hasattr(prediction, "names") else {}
        masks = self._extract_masks(prediction)
        classes = self._extract_classes(prediction)
        confs = self._extract_confidences(prediction)

        if masks is None or classes is None or confs is None:
            return None

        meter_keywords = ("clock", "gauge", "meter", "dial", "pressure")
        candidates = []

        for idx in range(min(len(masks), len(classes), len(confs))):
            class_id = int(classes[idx])
            class_name = str(names.get(class_id, "")).lower()
            is_meter = any(keyword in class_name for keyword in meter_keywords)

            if not is_meter:
                continue

            mask = self._to_binary_mask(masks[idx], img.shape[:2])
            circle = self._mask_to_circle(mask)
            if circle is None:
                continue

            cx, cy, radius = circle
            area = float(np.count_nonzero(mask))
            candidates.append(
                {
                    "center": (int(cx), int(cy)),
                    "radius": int(radius),
                    "confidence": float(confs[idx]),
                    "area": area,
                    "class_name": class_name,
                    "mask": mask,
                }
            )

        if not candidates:
            return None

        best = max(candidates, key=lambda c: (c["confidence"], c["area"], c["radius"]))
        best["source"] = "yolo"
        return best

    def detect_needle_tip(self, img: np.ndarray, cx: int, cy: int) -> Optional[Tuple[int, int]]:
        """YOLOセグメンテーションで針クラスが存在する場合のみ先端候補を返す"""
        prediction = self._predict(img)
        if prediction is None:
            return None

        names = prediction.names if hasattr(prediction, "names") else {}
        masks = self._extract_masks(prediction)
        classes = self._extract_classes(prediction)

        if masks is None or classes is None:
            return None

        needle_keywords = ("needle", "pointer", "hand")
        tip_candidates = []

        for idx in range(min(len(masks), len(classes))):
            class_id = int(classes[idx])
            class_name = str(names.get(class_id, "")).lower()
            if not any(keyword in class_name for keyword in needle_keywords):
                continue

            mask = self._to_binary_mask(masks[idx], img.shape[:2])
            ys, xs = np.where(mask > 0)
            if xs.size == 0:
                continue

            d2 = (xs - cx) ** 2 + (ys - cy) ** 2
            farthest = int(np.argmax(d2))
            tip_candidates.append((int(xs[farthest]), int(ys[farthest]), float(d2[farthest])))

        if not tip_candidates:
            return None

        best = max(tip_candidates, key=lambda t: t[2])
        return best[0], best[1]

    def _predict(self, img: np.ndarray):
        model = self._load_model()
        if model is None:
            return None

        try:
            results = model.predict(
                source=img,
                task="segment",
                verbose=False,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                retina_masks=True,
            )
            if not results:
                return None
            return results[0]
        except Exception:
            return None

    def _load_model(self):
        if not self.enabled:
            return None

        model_key = (self.model_path, self.device)
        if self.__class__._cached_model is not None and self.__class__._cached_model_key == model_key:
            return self.__class__._cached_model

        try:
            from ultralytics import YOLO

            model = YOLO(self.model_path)
            self.__class__._cached_model = model
            self.__class__._cached_model_key = model_key
            return model
        except Exception:
            return None

    def _extract_masks(self, prediction):
        if not hasattr(prediction, "masks") or prediction.masks is None:
            return None
        if not hasattr(prediction.masks, "data") or prediction.masks.data is None:
            return None
        return prediction.masks.data.cpu().numpy()

    def _extract_classes(self, prediction):
        if not hasattr(prediction, "boxes") or prediction.boxes is None:
            return None
        if not hasattr(prediction.boxes, "cls") or prediction.boxes.cls is None:
            return None
        return prediction.boxes.cls.cpu().numpy()

    def _extract_confidences(self, prediction):
        if not hasattr(prediction, "boxes") or prediction.boxes is None:
            return None
        if not hasattr(prediction.boxes, "conf") or prediction.boxes.conf is None:
            return None
        return prediction.boxes.conf.cpu().numpy()

    def _to_binary_mask(self, mask_data: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        mask = (mask_data > 0.5).astype(np.uint8)
        if mask.shape != shape:
            mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        return mask

    def _mask_to_circle(self, mask: np.ndarray) -> Optional[Tuple[int, int, int]]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 30:
            return None

        (x, y), radius = cv2.minEnclosingCircle(contour)
        if radius < 8:
            return None

        return int(x), int(y), int(radius)
