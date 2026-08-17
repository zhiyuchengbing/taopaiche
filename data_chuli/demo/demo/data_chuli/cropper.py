import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
import hyperlpr3 as lpr3
from typing import Tuple


_DEFAULT_MODEL_PATH = r"D:\project\\data_chuli\\demo\demo\\data_chuli\\data\cheliang_detect\\20260321\\best.pt"


class VehicleCropper:
    def __init__(self, classes=None, conf_thresh=0.2, mask_plates=True, model_name=_DEFAULT_MODEL_PATH,
                 center_weight=0.3, min_area_ratio=0.002):
        """
        conf_thresh: 车辆检测置信度阈值, 默认0.2. 2026-08-17 换 v3 模型后置信度大幅抬升
                     (未见帧平均0.975、角落误检0), 从 0.1 调回 0.2, 滤掉低置信噪声框;
                     原 0.1 是旧模型暗光难例真车置信度仅 0.05~0.2 时的权宜, 若回退旧模型需再降.
        center_weight: 选框时中心距离的折扣系数(0~1). 0=纯按面积取最大框, 越大越偏向画面中心.
                      默认0.3: 面积优先, 同时给贴边小框打折, 兼顾"取真车"与"避误检".
        min_area_ratio: 面积小于画面该比例的候选直接排除, 过滤噪声小框.
        """
        self.vehicle_classes = classes if classes is not None else [0]
        self.conf_thresh = conf_thresh
        self.mask_plates = mask_plates
        self.center_weight = center_weight
        self.min_area_ratio = min_area_ratio
        self.det_model = YOLO(model_name)
        self.catcher = lpr3.LicensePlateCatcher()

    def _to_bgr(self, pil_img: Image.Image):
        arr = np.array(pil_img.convert("RGB"))
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    def _to_pil(self, bgr_img: np.ndarray) -> Image.Image:
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def process_pil(self, pil_img: Image.Image) -> Tuple[Image.Image, bool]:
        img = self._to_bgr(pil_img)
        det_res = self.det_model.predict(source=img, classes=self.vehicle_classes, conf=self.conf_thresh, verbose=False)[0]
        boxes = det_res.boxes
        if boxes is None or len(boxes) == 0:
            return pil_img, False
        xyxy = boxes.xyxy.cpu().numpy()
        if xyxy.size == 0:
            return pil_img, False
        H, W = img.shape[:2]
        # 过滤面积过小的噪声框
        x1s, y1s, x2s, y2s = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
        areas = (x2s - x1s) * (y2s - y1s)
        valid = areas >= self.min_area_ratio * float(W * H)
        if not valid.any():
            valid[0] = True  # 全部过小则退回用第一个
        xyxy = xyxy[valid]
        x1s, y1s, x2s, y2s = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
        areas = (x2s - x1s) * (y2s - y1s)
        # 归一化中心距离(相对画面半宽/半高, 0=正中心, 1=画面边缘)
        cx0, cy0 = W / 2.0, H / 2.0
        centers_x = (x1s + x2s) / 2.0
        centers_y = (y1s + y2s) / 2.0
        norm_d = np.sqrt(((centers_x - cx0) / cx0) ** 2 + ((centers_y - cy0) / cy0) ** 2)
        # 面积优先, 中心距离打折: 越大越靠近中心的框胜出
        score = areas * (1.0 - self.center_weight * np.clip(norm_d, 0.0, 1.0))
        idx = int(np.argmax(score))
        x1, y1, x2, y2 = xyxy[idx]
        x1 = max(0, min(W - 1, int(x1)))
        y1 = max(0, min(H - 1, int(y1)))
        x2 = max(0, min(W, int(x2)))
        y2 = max(0, min(H, int(y2)))
        if x2 <= x1 or y2 <= y1:
            return pil_img, False
        crop = img[y1:y2, x1:x2].copy()
        if self.mask_plates:
            result = self.catcher(crop)
            if result and len(result) > 0:
                masked = crop.copy()
                h, w = masked.shape[:2]
                for item in result:
                    bbox = item[3] if len(item) > 3 else None
                    if bbox is None or len(bbox) != 4:
                        continue
                    px1, py1, px2, py2 = map(int, bbox)
                    px1 = max(0, min(w - 1, px1))
                    py1 = max(0, min(h - 1, py1))
                    px2 = max(0, min(w, px2))
                    py2 = max(0, min(h, py2))
                    if px2 > px1 and py2 > py1:
                        cv2.rectangle(masked, (px1, py1), (px2, py2), (0, 0, 0), thickness=-1)
                crop = masked
        return self._to_pil(crop), True
