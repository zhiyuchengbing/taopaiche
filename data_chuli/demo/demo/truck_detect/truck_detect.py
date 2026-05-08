
import os
import sys
import argparse
from typing import List, Optional, Tuple
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
import hyperlpr3 as lpr3


_DEFAULT_MODEL_PATH = r"D:\data2\truck\split\runs\truck_train\weights\best.pt"
_LEGACY_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "yolo26m.pt")


class VehicleCropper:
    def __init__(self, classes=None, conf_thresh=0.5, mask_plates=True, model_name=None):
        self.conf_thresh = conf_thresh
        self.mask_plates = mask_plates
        if model_name is None:
            model_name = (
                _DEFAULT_MODEL_PATH
                if os.path.exists(_DEFAULT_MODEL_PATH)
                else (_LEGACY_MODEL_PATH if os.path.exists(_LEGACY_MODEL_PATH) else "yolo26m.pt")
            )
        self.det_model = YOLO(model_name)
        if classes is None:
            self.vehicle_classes = self._infer_vehicle_classes()
        else:
            self.vehicle_classes = classes
        self.catcher = lpr3.LicensePlateCatcher()

    def _infer_vehicle_classes(self) -> Optional[List[int]]:
        names = getattr(self.det_model, "names", None)
        if not names:
            return None

        if isinstance(names, dict):
            items = list(names.items())
        else:
            try:
                items = list(enumerate(list(names)))
            except Exception:
                return None

        target = {"car", "motorcycle", "bus", "truck", "train"}
        matched: List[int] = []
        for k, v in items:
            try:
                name = str(v).strip().lower()
            except Exception:
                continue
            if name in target:
                try:
                    matched.append(int(k))
                except Exception:
                    continue

        if matched:
            matched.sort()
            return matched
        return None

    def _detect_boxes(self, bgr_img: np.ndarray):
        if self.vehicle_classes is None:
            det_res = self.det_model.predict(
                source=bgr_img,
                conf=self.conf_thresh,
                verbose=False,
            )[0]
        else:
            det_res = self.det_model.predict(
                source=bgr_img,
                classes=self.vehicle_classes,
                conf=self.conf_thresh,
                verbose=False,
            )[0]
        boxes = det_res.boxes
        if boxes is None or len(boxes) == 0:
            return None, None
        xyxy = boxes.xyxy
        conf = getattr(boxes, "conf", None)
        if xyxy is None:
            return None, None
        xyxy_np = xyxy.cpu().numpy()
        conf_np = conf.cpu().numpy() if conf is not None else None
        if xyxy_np.size == 0:
            return None, None
        return xyxy_np, conf_np

    def _pick_center_box_index(self, xyxy: np.ndarray, img_shape: Tuple[int, int]) -> int:
        if xyxy is None or xyxy.size == 0:
            return 0
        H, W = img_shape
        cx0 = W / 2.0
        cy0 = H / 2.0
        centers_x = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
        centers_y = (xyxy[:, 1] + xyxy[:, 3]) / 2.0
        d2 = (centers_x - cx0) ** 2 + (centers_y - cy0) ** 2
        idx = int(np.argmin(d2))
        return idx

    def _to_bgr(self, pil_img: Image.Image):
        arr = np.array(pil_img.convert("RGB"))
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    def _to_pil(self, bgr_img: np.ndarray) -> Image.Image:
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def process_pil(self, pil_img: Image.Image) -> Image.Image:
        img = self._to_bgr(pil_img)
        xyxy, _ = self._detect_boxes(img)
        if xyxy is None:
            return pil_img
        H, W = img.shape[:2]
        idx = self._pick_center_box_index(xyxy, (H, W))
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

    def visualize_detections_pil(self, pil_img: Image.Image) -> Image.Image:
        img = self._to_bgr(pil_img)
        xyxy, conf = self._detect_boxes(img)
        if xyxy is None:
            return pil_img

        out = img.copy()
        H, W = out.shape[:2]

        if self.mask_plates:
            for i in range(xyxy.shape[0]):
                x1, y1, x2, y2 = xyxy[i]
                x1 = max(0, min(W - 1, int(x1)))
                y1 = max(0, min(H - 1, int(y1)))
                x2 = max(0, min(W, int(x2)))
                y2 = max(0, min(H, int(y2)))
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = out[y1:y2, x1:x2]
                result = self.catcher(crop)
                if not result:
                    continue
                h, w = crop.shape[:2]
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
                        cv2.rectangle(crop, (px1, py1), (px2, py2), (0, 0, 0), thickness=-1)

        for i in range(xyxy.shape[0]):
            x1, y1, x2, y2 = xyxy[i]
            x1 = max(0, min(W - 1, int(x1)))
            y1 = max(0, min(H - 1, int(y1)))
            x2 = max(0, min(W, int(x2)))
            y2 = max(0, min(H, int(y2)))
            if x2 <= x1 or y2 <= y1:
                continue
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), thickness=2)
            score = float(conf[i]) if conf is not None and i < len(conf) else None
            label = f"{i}" if score is None else f"{i}:{score:.2f}"
            cv2.putText(out, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)

        return self._to_pil(out)


def _resize_to_height(bgr: np.ndarray, h: int) -> np.ndarray:
    if bgr is None:
        return bgr
    ch, cw = bgr.shape[:2]
    if ch == h:
        return bgr
    scale = h / float(ch)
    nw = max(1, int(round(cw * scale)))
    return cv2.resize(bgr, (nw, h), interpolation=cv2.INTER_AREA)


def _run_gui() -> int:
    try:
        from PySide6.QtWidgets import (
            QApplication,
            QWidget,
            QLabel,
            QPushButton,
            QFileDialog,
            QHBoxLayout,
            QVBoxLayout,
            QLineEdit,
            QDoubleSpinBox,
            QCheckBox,
        )
        from PySide6.QtGui import QPixmap, QImage
        from PySide6.QtCore import Qt
    except Exception as e:
        raise RuntimeError("PySide6 not installed. Please install PySide6 or run CLI mode.") from e

    def pil_to_pixmap(pil_img: Image.Image) -> QPixmap:
        img = pil_img.convert("RGB")
        arr = np.array(img)
        h, w = arr.shape[:2]
        qimg = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg.copy())

    class MainWindow(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("车辆裁切可视化")

            self._pm_left = None
            self._pm_right = None

            self.path_edit = QLineEdit()
            self.path_edit.setPlaceholderText("选择一张图片")
            self.btn_open = QPushButton("选择图片")

            self.model_edit = QLineEdit(
                _DEFAULT_MODEL_PATH
                if os.path.exists(_DEFAULT_MODEL_PATH)
                else (_LEGACY_MODEL_PATH if os.path.exists(_LEGACY_MODEL_PATH) else "yolo26m.pt")
            )
            self.model_edit.setPlaceholderText("YOLO模型路径或名称")

            self.conf_spin = QDoubleSpinBox()
            self.conf_spin.setDecimals(2)
            self.conf_spin.setSingleStep(0.05)
            self.conf_spin.setRange(0.0, 1.0)
            self.conf_spin.setValue(0.50)

            self.mask_cb = QCheckBox("车牌打码")
            self.mask_cb.setChecked(True)

            self.box_mode_cb = QCheckBox("只显示车辆框(不裁切)")
            self.box_mode_cb.setChecked(True)

            self.btn_run = QPushButton("裁切")

            top_row = QHBoxLayout()
            top_row.addWidget(self.path_edit, 1)
            top_row.addWidget(self.btn_open)
            top_row.addWidget(QLabel("conf"))
            top_row.addWidget(self.conf_spin)
            top_row.addWidget(self.mask_cb)
            top_row.addWidget(self.box_mode_cb)

            model_row = QHBoxLayout()
            model_row.addWidget(QLabel("model"))
            model_row.addWidget(self.model_edit, 1)
            model_row.addWidget(self.btn_run)

            self.img_left = QLabel()
            self.img_right = QLabel()
            self.img_left.setAlignment(Qt.AlignCenter)
            self.img_right.setAlignment(Qt.AlignCenter)
            self.img_left.setMinimumHeight(360)
            self.img_right.setMinimumHeight(360)

            imgs = QHBoxLayout()
            imgs.addWidget(self.img_left, 1)
            imgs.addWidget(self.img_right, 1)

            root = QVBoxLayout()
            root.addLayout(top_row)
            root.addLayout(model_row)
            root.addLayout(imgs, 1)
            self.setLayout(root)

            self.btn_open.clicked.connect(self.on_open)
            self.btn_run.clicked.connect(self.on_run)
            self.box_mode_cb.toggled.connect(self._on_mode_changed)
            self.mask_cb.toggled.connect(self.on_run)

            self._on_mode_changed()

        def _on_mode_changed(self):
            self.btn_run.setText("检测" if self.box_mode_cb.isChecked() else "裁切")
            self.on_run()

        def on_open(self):
            path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.jpg *.jpeg *.png *.bmp *.webp)")
            if not path:
                return
            self.path_edit.setText(path)
            self.on_run()

        def on_run(self):
            path = self.path_edit.text().strip()
            if not path or not os.path.exists(path):
                return

            img_pil = Image.open(path)
            cropper = VehicleCropper(
                conf_thresh=float(self.conf_spin.value()),
                mask_plates=bool(self.mask_cb.isChecked()),
                model_name=self.model_edit.text().strip()
                or (
                    _DEFAULT_MODEL_PATH
                    if os.path.exists(_DEFAULT_MODEL_PATH)
                    else (_LEGACY_MODEL_PATH if os.path.exists(_LEGACY_MODEL_PATH) else "yolo26m.pt")
                ),
            )
            if self.box_mode_cb.isChecked():
                right_pil = cropper.visualize_detections_pil(img_pil)
            else:
                right_pil = cropper.process_pil(img_pil)

            pm1 = pil_to_pixmap(img_pil)
            pm2 = pil_to_pixmap(right_pil)

            self._pm_left = pm1
            self._pm_right = pm2
            self._render_pixmaps()

        def _render_pixmaps(self):
            if self._pm_left is not None:
                self.img_left.setPixmap(self._pm_left.scaled(self.img_left.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            if self._pm_right is not None:
                self.img_right.setPixmap(self._pm_right.scaled(self.img_right.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        def resizeEvent(self, event):
            self._render_pixmaps()
            return super().resizeEvent(event)

    app = QApplication([])
    w = MainWindow()
    w.resize(1100, 520)
    w.show()
    return app.exec()


def main() -> int:
    parser = argparse.ArgumentParser(description="Vehicle cropping visualization")
    parser.add_argument("image", help="input image path")
    parser.add_argument(
        "--model",
        default=(
            _DEFAULT_MODEL_PATH
            if os.path.exists(_DEFAULT_MODEL_PATH)
            else (_LEGACY_MODEL_PATH if os.path.exists(_LEGACY_MODEL_PATH) else "yolo26m.pt")
        ),
        help="YOLO model name/path for vehicle detection",
    )
    parser.add_argument("--conf", type=float, default=0.5, help="confidence threshold")
    parser.add_argument("--no-mask", action="store_true", help="disable license plate masking")
    parser.add_argument("--mode", choices=["boxes", "crop"], default="boxes", help="visualize mode: boxes=draw all vehicles, crop=crop center vehicle")
    args = parser.parse_args()

    image_path = args.image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"image not found: {image_path}")

    img_pil = Image.open(image_path)
    cropper = VehicleCropper(conf_thresh=float(args.conf), mask_plates=not args.no_mask, model_name=args.model)
    if args.mode == "crop":
        right_pil = cropper.process_pil(img_pil)
        title = "Vehicle Cropper (Left: Original, Right: Cropped)"
    else:
        right_pil = cropper.visualize_detections_pil(img_pil)
        title = "Vehicle Detector (Left: Original, Right: Boxes)"

    bgr_src = cropper._to_bgr(img_pil)
    bgr_crop = cropper._to_bgr(right_pil)
    h = max(bgr_src.shape[0], bgr_crop.shape[0])
    bgr_src = _resize_to_height(bgr_src, h)
    bgr_crop = _resize_to_height(bgr_crop, h)
    gap = np.full((h, 12, 3), 245, dtype=np.uint8)
    canvas = np.hstack([bgr_src, gap, bgr_crop])

    cv2.imshow(title, canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 1:
        raise SystemExit(_run_gui())
    raise SystemExit(main())
