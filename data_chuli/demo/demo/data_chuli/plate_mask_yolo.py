import os
import sys
import argparse
from typing import List, Optional, Tuple

import numpy as np
import cv2
from ultralytics import YOLO


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


_DEFAULT_CROP_WEIGHTS = r"D:\project\data_chuli\demo\demo\data_chuli\data\cheliang_detect\20260122\best.pt"
_DEFAULT_PLATE_WEIGHTS = r"D:\project\data_chuli\demo\demo\data_chuli\data\chepai_detect\l_pt\best.pt"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in _IMAGE_EXTS


def _iter_images(input_path: str, recursive: bool) -> List[str]:
    input_abs = os.path.abspath(input_path)
    if os.path.isfile(input_abs):
        if not _is_image(input_abs):
            raise ValueError(f"not an image: {input_abs}")
        return [input_abs]

    if not os.path.isdir(input_abs):
        raise FileNotFoundError(f"input not found: {input_abs}")

    out: List[str] = []
    if recursive:
        for dirpath, _dirnames, filenames in os.walk(input_abs):
            for fn in filenames:
                p = os.path.join(dirpath, fn)
                if _is_image(p):
                    out.append(p)
    else:
        for fn in os.listdir(input_abs):
            p = os.path.join(input_abs, fn)
            if os.path.isfile(p) and _is_image(p):
                out.append(p)

    out.sort()
    return out


def _infer_plate_classes(model: YOLO) -> Optional[List[int]]:
    names = getattr(model, "names", None)
    if not names:
        return None

    items: List[Tuple[int, str]]
    if isinstance(names, dict):
        items = []
        for k, v in names.items():
            try:
                items.append((int(k), str(v)))
            except Exception:
                continue
    else:
        try:
            items = [(int(i), str(n)) for i, n in enumerate(list(names))]
        except Exception:
            return None

    if len(items) == 1:
        return [items[0][0]]

    keywords = {"plate", "license", "licence", "lp", "carplate", "licenseplate", "车牌"}
    matched: List[int] = []
    for k, v in items:
        name = str(v).strip().lower().replace(" ", "")
        if name in keywords or any(kw in name for kw in keywords):
            matched.append(int(k))

    if matched:
        matched.sort()
        return matched
    return None


def _pick_center_box_index(xyxy, img_wh: Tuple[int, int]) -> int:
    w, h = img_wh
    cx0 = w / 2.0
    cy0 = h / 2.0
    centers_x = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
    centers_y = (xyxy[:, 1] + xyxy[:, 3]) / 2.0
    d2 = (centers_x - cx0) ** 2 + (centers_y - cy0) ** 2
    return int(np.argmin(d2))


def _infer_vehicle_classes(model: YOLO) -> Optional[List[int]]:
    names = getattr(model, "names", None)
    if not names:
        return None

    items: List[Tuple[int, str]]
    if isinstance(names, dict):
        items = []
        for k, v in names.items():
            try:
                items.append((int(k), str(v)))
            except Exception:
                continue
    else:
        try:
            items = [(int(i), str(n)) for i, n in enumerate(list(names))]
        except Exception:
            return None

    if len(items) == 1:
        return [items[0][0]]

    target = {"car", "motorcycle", "bus", "truck", "train", "vehicle", "车辆"}
    matched: List[int] = []
    for k, v in items:
        name = str(v).strip().lower().replace(" ", "")
        if name in target:
            matched.append(int(k))

    if matched:
        matched.sort()
        return matched
    return None


def _mask_boxes(
    bgr: "cv2.Mat",
    boxes_xyxy,
    *,
    mode: str,
    blur_ksize: int,
) -> None:
    h, w = bgr.shape[:2]
    for i in range(boxes_xyxy.shape[0]):
        x1, y1, x2, y2 = boxes_xyxy[i]
        x1 = max(0, min(w - 1, int(x1)))
        y1 = max(0, min(h - 1, int(y1)))
        x2 = max(0, min(w, int(x2)))
        y2 = max(0, min(h, int(y2)))
        if x2 <= x1 or y2 <= y1:
            continue

        if mode == "black":
            cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)
        elif mode == "blur":
            roi = bgr[y1:y2, x1:x2]
            k = int(blur_ksize)
            if k % 2 == 0:
                k += 1
            k = max(3, k)
            roi_blur = cv2.GaussianBlur(roi, (k, k), 0)
            bgr[y1:y2, x1:x2] = roi_blur
        else:
            raise ValueError(f"unknown mode: {mode}")


def _resize_to_height(bgr: np.ndarray, h: int) -> np.ndarray:
    if bgr is None:
        return bgr
    ch, cw = bgr.shape[:2]
    if ch == h:
        return bgr
    scale = h / float(ch)
    nw = max(1, int(round(cw * scale)))
    return cv2.resize(bgr, (nw, h), interpolation=cv2.INTER_AREA)


class PlateMasker:
    def __init__(
        self,
        *,
        crop_weights: str = _DEFAULT_CROP_WEIGHTS,
        plate_weights: str = _DEFAULT_PLATE_WEIGHTS,
        plate_classes: Optional[List[int]] = None,
        crop_classes: Optional[List[int]] = None,
        crop_conf: float = 0.25,
        crop_iou: float = 0.45,
        plate_conf: float = 0.25,
        plate_iou: float = 0.45,
        mode: str = "black",
        blur_ksize: int = 31,
    ):
        if not os.path.exists(crop_weights):
            raise FileNotFoundError(f"crop weights not found: {crop_weights}")
        if not os.path.exists(plate_weights):
            raise FileNotFoundError(f"plate weights not found: {plate_weights}")

        self.crop_model = YOLO(crop_weights)
        self.plate_model = YOLO(plate_weights)

        self.crop_classes = crop_classes if crop_classes is not None else _infer_vehicle_classes(self.crop_model)
        self.plate_classes = plate_classes if plate_classes is not None else [0]

        self.crop_conf = float(crop_conf)
        self.crop_iou = float(crop_iou)
        self.plate_conf = float(plate_conf)
        self.plate_iou = float(plate_iou)
        self.mode = str(mode)
        self.blur_ksize = int(blur_ksize)

    def _crop_vehicle(self, bgr: np.ndarray) -> np.ndarray:
        if self.crop_classes is None:
            crop_res = self.crop_model.predict(source=bgr, conf=self.crop_conf, iou=self.crop_iou, verbose=False)[0]
        else:
            crop_res = self.crop_model.predict(
                source=bgr,
                classes=self.crop_classes,
                conf=self.crop_conf,
                iou=self.crop_iou,
                verbose=False,
            )[0]

        boxes = getattr(crop_res, "boxes", None)
        if boxes is None or len(boxes) == 0 or boxes.xyxy is None:
            return bgr

        xyxy = boxes.xyxy.cpu().numpy()
        if xyxy.size == 0:
            return bgr

        h0, w0 = bgr.shape[:2]
        j = _pick_center_box_index(xyxy, (w0, h0))
        x1, y1, x2, y2 = xyxy[j]
        x1 = max(0, min(w0 - 1, int(x1)))
        y1 = max(0, min(h0 - 1, int(y1)))
        x2 = max(0, min(w0, int(x2)))
        y2 = max(0, min(h0, int(y2)))
        if x2 <= x1 or y2 <= y1:
            return bgr
        return bgr[y1:y2, x1:x2].copy()

    def _mask_plate_inplace(self, bgr: np.ndarray) -> None:
        if self.plate_classes is None:
            res = self.plate_model.predict(source=bgr, conf=self.plate_conf, iou=self.plate_iou, verbose=False)[0]
        else:
            res = self.plate_model.predict(
                source=bgr,
                classes=self.plate_classes,
                conf=self.plate_conf,
                iou=self.plate_iou,
                verbose=False,
            )[0]

        boxes = getattr(res, "boxes", None)
        if boxes is None or len(boxes) == 0 or boxes.xyxy is None:
            return
        xyxy = boxes.xyxy.cpu().numpy()
        if xyxy.size == 0:
            return
        _mask_boxes(bgr, xyxy, mode=self.mode, blur_ksize=self.blur_ksize)

    def process(self, image_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"image not found: {image_path}")
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise ValueError(f"failed to read image: {image_path}")

        cropped = self._crop_vehicle(bgr)
        masked = cropped.copy()
        self._mask_plate_inplace(masked)
        return bgr, cropped, masked

    def visualize(
        self,
        image_path: str,
        *,
        window_title: str = "Plate Mask (crop -> seg mask)",
        max_window_w: int = 1600,
        max_window_h: int = 900,
    ) -> None:
        src, crop, masked = self.process(image_path)
        h = max(src.shape[0], crop.shape[0], masked.shape[0])
        src_r = _resize_to_height(src, h)
        crop_r = _resize_to_height(crop, h)
        masked_r = _resize_to_height(masked, h)
        gap = np.full((h, 12, 3), 245, dtype=np.uint8)
        canvas = np.hstack([src_r, gap, crop_r, gap, masked_r])

        ch, cw = canvas.shape[:2]
        scale = min(max_window_w / float(cw), max_window_h / float(ch), 1.0)
        if scale < 1.0:
            nw = max(1, int(round(cw * scale)))
            nh = max(1, int(round(ch * scale)))
            canvas = cv2.resize(canvas, (nw, nh), interpolation=cv2.INTER_AREA)

        cv2.imshow(window_title, canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def _mask_with_segmentation(
    bgr: "cv2.Mat",
    masks_data,
    *,
    mode: str,
    blur_ksize: int,
) -> bool:
    if masks_data is None:
        return False

    try:
        masks_np = masks_data.cpu().numpy()
    except Exception:
        try:
            masks_np = masks_data.numpy()
        except Exception:
            return False

    if masks_np is None or getattr(masks_np, "size", 0) == 0:
        return False

    if masks_np.ndim == 2:
        masks_np = masks_np[None, ...]
    if masks_np.ndim != 3:
        return False

    h, w = bgr.shape[:2]
    mh, mw = masks_np.shape[1], masks_np.shape[2]
    if (mh, mw) != (h, w):
        resized = []
        for i in range(masks_np.shape[0]):
            m = masks_np[i]
            m = cv2.resize(m.astype("float32"), (w, h), interpolation=cv2.INTER_NEAREST)
            resized.append(m)
        masks_np = np.stack(resized, axis=0)

    union = (masks_np > 0.5).any(axis=0)
    if not union.any():
        return False

    if mode == "black":
        bgr[union] = (0, 0, 0)
        return True

    if mode == "blur":
        k = int(blur_ksize)
        if k % 2 == 0:
            k += 1
        k = max(3, k)
        blurred = cv2.GaussianBlur(bgr, (k, k), 0)
        bgr[union] = blurred[union]
        return True

    raise ValueError(f"unknown mode: {mode}")


def mask_plates(
    input_path: str,
    output_dir: str,
    *,
    weights: str,
    crop_weights: Optional[str],
    conf: float,
    iou: float,
    classes: Optional[List[int]],
    crop_conf: float,
    crop_iou: float,
    crop_classes: Optional[List[int]],
    recursive: bool,
    preserve_dirs: bool,
    mode: str,
    blur_ksize: int,
) -> int:
    in_abs = os.path.abspath(input_path)
    out_abs = os.path.abspath(output_dir)
    _ensure_dir(out_abs)

    images = _iter_images(in_abs, recursive=recursive)
    if not images:
        return 0

    if not weights:
        raise ValueError("weights is empty")
    if not os.path.exists(weights):
        raise FileNotFoundError(f"weights not found: {weights}")

    plate_model = YOLO(weights)
    if classes is None:
        classes = [0]

    crop_model = None
    if crop_weights:
        if not os.path.exists(crop_weights):
            raise FileNotFoundError(f"crop weights not found: {crop_weights}")
        crop_model = YOLO(crop_weights)
        if crop_classes is None:
            crop_classes = _infer_vehicle_classes(crop_model)

    for idx, img_path in enumerate(images, start=1):
        bgr = cv2.imread(img_path)
        if bgr is None:
            continue

        work = bgr
        if crop_model is not None:
            if crop_classes is None:
                crop_res = crop_model.predict(source=bgr, conf=crop_conf, iou=crop_iou, verbose=False)[0]
            else:
                crop_res = crop_model.predict(source=bgr, classes=crop_classes, conf=crop_conf, iou=crop_iou, verbose=False)[0]

            crop_boxes = getattr(crop_res, "boxes", None)
            if crop_boxes is not None and len(crop_boxes) > 0 and crop_boxes.xyxy is not None:
                xyxy0 = crop_boxes.xyxy.cpu().numpy()
                if xyxy0.size != 0:
                    h0, w0 = bgr.shape[:2]
                    j = _pick_center_box_index(xyxy0, (w0, h0))
                    x1, y1, x2, y2 = xyxy0[j]
                    x1 = max(0, min(w0 - 1, int(x1)))
                    y1 = max(0, min(h0 - 1, int(y1)))
                    x2 = max(0, min(w0, int(x2)))
                    y2 = max(0, min(h0, int(y2)))
                    if x2 > x1 and y2 > y1:
                        work = bgr[y1:y2, x1:x2].copy()

        if classes is None:
            res = plate_model.predict(source=work, conf=conf, iou=iou, verbose=False)[0]
        else:
            res = plate_model.predict(source=work, classes=classes, conf=conf, iou=iou, verbose=False)[0]

        boxes = res.boxes
        if boxes is not None and len(boxes) > 0 and boxes.xyxy is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            if xyxy.size != 0:
                _mask_boxes(work, xyxy, mode=mode, blur_ksize=blur_ksize)

        if preserve_dirs and os.path.isdir(in_abs):
            rel = os.path.relpath(img_path, in_abs)
            dst = os.path.join(out_abs, rel)
            _ensure_dir(os.path.dirname(dst))
        else:
            dst = os.path.join(out_abs, os.path.basename(img_path))

        cv2.imwrite(dst, work)
        if idx % 20 == 0 or idx == len(images):
            print(f"{idx}/{len(images)} -> {dst}")

    return len(images)


def _parse_int_list(s: str) -> List[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]


def main() -> int:
    parser = argparse.ArgumentParser(description="Mask license plates using YOLOv8 detections")
    parser.add_argument("input", help="image path or folder")
    parser.add_argument("--out-dir", required=True, help="output directory")
    parser.add_argument("--weights", required=True, help="YOLO weights path")
    parser.add_argument("--crop-weights", default=_DEFAULT_CROP_WEIGHTS, help="vehicle detection weights (crop first)")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--classes", type=_parse_int_list, default=None, help="class ids, e.g. 0 or 0,1")
    parser.add_argument("--crop-conf", type=float, default=0.25, help="vehicle crop confidence threshold")
    parser.add_argument("--crop-iou", type=float, default=0.45, help="vehicle crop NMS IoU threshold")
    parser.add_argument("--crop-classes", type=_parse_int_list, default=None, help="vehicle class ids, e.g. 0")
    parser.add_argument("--recursive", action="store_true", help="process subfolders when input is a directory")
    parser.add_argument("--preserve-dirs", action="store_true", help="preserve input folder structure in out-dir")
    parser.add_argument("--mode", choices=["black", "blur"], default="black", help="mask style")
    parser.add_argument("--blur-ksize", type=int, default=31, help="blur kernel size when mode=blur")
    args = parser.parse_args()

    n = mask_plates(
        args.input,
        args.out_dir,
        weights=args.weights,
        crop_weights=(args.crop_weights.strip() if str(args.crop_weights).strip() else None),
        conf=float(args.conf),
        iou=float(args.iou),
        classes=args.classes,
        crop_conf=float(args.crop_conf),
        crop_iou=float(args.crop_iou),
        crop_classes=args.crop_classes,
        recursive=bool(args.recursive),
        preserve_dirs=bool(args.preserve_dirs),
        mode=str(args.mode),
        blur_ksize=int(args.blur_ksize),
    )
    print(f"done, processed: {n}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 1:
        DEMO_IMAGE_PATH = r"D:\data2\交付文件\异常图片\异常图片\7.png"
        PlateMasker().visualize(DEMO_IMAGE_PATH)
        raise SystemExit(0)
    raise SystemExit(main())
