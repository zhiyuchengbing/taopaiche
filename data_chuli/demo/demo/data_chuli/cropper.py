import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
import hyperlpr3 as lpr3


_DEFAULT_MODEL_PATH = r"D:\project\\data_chuli\\demo\demo\\data_chuli\\data\cheliang_detect\\20260321\\best.pt"


class VehicleCropper:
    def __init__(self, classes=None, conf_thresh=0.5, mask_plates=True, model_name=_DEFAULT_MODEL_PATH):
        self.vehicle_classes = classes if classes is not None else [0]
        self.conf_thresh = conf_thresh
        self.mask_plates = mask_plates
        self.det_model = YOLO(model_name)
        self.catcher = lpr3.LicensePlateCatcher()

    def _to_bgr(self, pil_img: Image.Image):
        arr = np.array(pil_img.convert("RGB"))
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    def _to_pil(self, bgr_img: np.ndarray) -> Image.Image:
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def process_pil(self, pil_img: Image.Image) -> Image.Image:
        img = self._to_bgr(pil_img)
        det_res = self.det_model.predict(source=img, classes=self.vehicle_classes, conf=self.conf_thresh, verbose=False)[0]
        boxes = det_res.boxes
        if boxes is None or len(boxes) == 0:
            return pil_img
        xyxy = boxes.xyxy.cpu().numpy()
        if xyxy.size == 0:
            return pil_img
        H, W = img.shape[:2]
        cx0 = W / 2.0
        cy0 = H / 2.0
        centers_x = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
        centers_y = (xyxy[:, 1] + xyxy[:, 3]) / 2.0
        d2 = (centers_x - cx0) ** 2 + (centers_y - cy0) ** 2
        idx = int(np.argmin(d2))
        x1, y1, x2, y2 = xyxy[idx]
        x1 = max(0, min(W - 1, int(x1)))
        y1 = max(0, min(H - 1, int(y1)))
        x2 = max(0, min(W, int(x2)))
        y2 = max(0, min(H, int(y2)))
        if x2 <= x1 or y2 <= y1:
            return pil_img
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
        return self._to_pil(crop)
