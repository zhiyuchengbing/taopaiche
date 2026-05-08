import os
import sys
import shutil
from typing import List, Optional, Tuple
import xml.etree.ElementTree as ET

import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO


_DEFAULT_MODEL_PATH = r"D:\data2\truck\split\runs\truck_train\weights\best.pt"
_LEGACY_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "yolo26m.pt")
_DEFAULT_OUT_IMG_DIR = r"D:\data2\truck\data"
_DEFAULT_OUT_XML_DIR = r"D:\data2\truck\label"
_DEFAULT_LABEL_NAME = "truck"
_DEFAULT_VEHICLE_CLASSES = [2, 3, 5, 6, 7]

_DEFAULT_INPUT_PATH = r"D:\data2\truck\input"

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _sanitize_filename(name: str) -> str:
    # Keep it simple for Windows: allow letters/digits/._- and replace others with '_'
    out = []
    for ch in name:
        if ch.isalnum() or ch in {".", "_", "-"}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def _make_unique_path(dst_dir: str, filename: str) -> str:
    base, ext = os.path.splitext(filename)
    candidate = os.path.join(dst_dir, filename)
    if not os.path.exists(candidate):
        return candidate
    for i in range(1, 1000000):
        cand = os.path.join(dst_dir, f"{base}__{i}{ext}")
        if not os.path.exists(cand):
            return cand
    raise RuntimeError(f"failed to find unique name for: {filename}")


def _flat_export_name(src_abs: str, input_root: Optional[str]) -> str:
    base = os.path.basename(src_abs)
    if not input_root:
        return _sanitize_filename(base)
    rel = os.path.relpath(src_abs, os.path.abspath(input_root))
    rel_no_ext, ext = os.path.splitext(rel)
    # Encode subfolders into the filename to avoid collisions.
    encoded = _sanitize_filename(rel_no_ext.replace("\\", "__").replace("/", "__"))
    return f"{encoded}{ext}"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _read_image_size_and_depth(image_path: str) -> Tuple[int, int, int]:
    pil = Image.open(image_path)
    pil = pil.convert("RGB")
    w, h = pil.size
    depth = 3
    return w, h, depth


def _to_voc_xml(
    *,
    folder: str,
    filename: str,
    full_image_path: str,
    width: int,
    height: int,
    depth: int,
    objects: List[Tuple[str, int, int, int, int]],
) -> ET.Element:
    annotation = ET.Element("annotation")

    ET.SubElement(annotation, "folder").text = folder
    ET.SubElement(annotation, "filename").text = filename

    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "Unknown"

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(depth)

    ET.SubElement(annotation, "segmented").text = "0"

    # LabelImg usually accepts either absolute or relative path. It's not mandatory,
    # but helps some tools.
    ET.SubElement(annotation, "path").text = full_image_path

    for (name, xmin, ymin, xmax, ymax) in objects:
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = name
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"

        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(int(xmin))
        ET.SubElement(bndbox, "ymin").text = str(int(ymin))
        ET.SubElement(bndbox, "xmax").text = str(int(xmax))
        ET.SubElement(bndbox, "ymax").text = str(int(ymax))

    return annotation


def _indent_xml(elem: ET.Element, level: int = 0) -> None:
    # In-place pretty print
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            _indent_xml(child, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def _infer_vehicle_classes_from_model(model: YOLO) -> Optional[List[int]]:
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

    target = {"car", "motorcycle", "bus", "truck", "train"}
    matched: List[int] = []
    for k, v in items:
        name = str(v).strip().lower()
        if name in target:
            matched.append(int(k))

    if matched:
        matched.sort()
        return matched
    return None


def detect_vehicle_boxes(
    image_path: str,
    *,
    model_path: Optional[str] = None,
    model: Optional[YOLO] = None,
    conf: float = 0.15,
    classes: Optional[List[int]] = None,
) -> List[Tuple[int, int, int, int]]:
    model_path = model_path or (
        _DEFAULT_MODEL_PATH
        if os.path.exists(_DEFAULT_MODEL_PATH)
        else (_LEGACY_MODEL_PATH if os.path.exists(_LEGACY_MODEL_PATH) else "yolo26m.pt")
    )
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"failed to read image: {image_path}")

    if model is None:
        model = YOLO(model_path)

    if classes is None:
        inferred = _infer_vehicle_classes_from_model(model)
        if inferred is None:
            res = model.predict(source=img_bgr, classes=_DEFAULT_VEHICLE_CLASSES, conf=conf, verbose=False)[0]
        else:
            res = model.predict(source=img_bgr, classes=inferred, conf=conf, verbose=False)[0]
    else:
        res = model.predict(source=img_bgr, classes=classes, conf=conf, verbose=False)[0]
    boxes = res.boxes
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy
    if xyxy is None:
        return []

    xyxy_np = xyxy.cpu().numpy()
    if xyxy_np.size == 0:
        return []

    h, w = img_bgr.shape[:2]
    out: List[Tuple[int, int, int, int]] = []
    for i in range(xyxy_np.shape[0]):
        x1, y1, x2, y2 = xyxy_np[i]
        x1 = max(0, min(w - 1, int(x1)))
        y1 = max(0, min(h - 1, int(y1)))
        x2 = max(0, min(w, int(x2)))
        y2 = max(0, min(h, int(y2)))
        if x2 <= x1 or y2 <= y1:
            continue
        out.append((x1, y1, x2, y2))
    return out


def export_one(
    image_path: str,
    out_img_dir: str = _DEFAULT_OUT_IMG_DIR,
    out_xml_dir: str = _DEFAULT_OUT_XML_DIR,
    label_name: str = _DEFAULT_LABEL_NAME,
    model_path: Optional[str] = None,
    model: Optional[YOLO] = None,
    conf: float = 0.15,
    input_root: Optional[str] = None,
) -> Tuple[str, str, int]:
    _ensure_dir(out_img_dir)
    _ensure_dir(out_xml_dir)

    src_abs = os.path.abspath(image_path)
    if not os.path.exists(src_abs):
        raise FileNotFoundError(f"image not found: {src_abs}")

    flat_img_name = _flat_export_name(src_abs, input_root)
    dst_img_path = _make_unique_path(out_img_dir, flat_img_name)
    flat_xml_name = os.path.splitext(os.path.basename(dst_img_path))[0] + ".xml"
    dst_xml_path = _make_unique_path(out_xml_dir, flat_xml_name)

    shutil.copy2(src_abs, dst_img_path)

    boxes = detect_vehicle_boxes(src_abs, model_path=model_path, model=model, conf=conf)
    w, h, depth = _read_image_size_and_depth(dst_img_path)

    objects = [(label_name, x1, y1, x2, y2) for (x1, y1, x2, y2) in boxes]
    annotation = _to_voc_xml(
        folder=os.path.basename(os.path.normpath(out_img_dir)),
        filename=os.path.basename(dst_img_path),
        full_image_path=dst_img_path,
        width=w,
        height=h,
        depth=depth,
        objects=objects,
    )
    _indent_xml(annotation)

    tree = ET.ElementTree(annotation)
    tree.write(dst_xml_path, encoding="utf-8", xml_declaration=True)

    return dst_img_path, dst_xml_path, len(objects)


def _iter_images_recursively(root_dir: str):
    root_abs = os.path.abspath(root_dir)
    for dirpath, _dirnames, filenames in os.walk(root_abs):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext not in _IMAGE_EXTS:
                continue
            yield os.path.join(dirpath, fn)


def export_folder(
    input_dir: str,
    out_img_dir: str = _DEFAULT_OUT_IMG_DIR,
    out_xml_dir: str = _DEFAULT_OUT_XML_DIR,
    label_name: str = _DEFAULT_LABEL_NAME,
    model_path: Optional[str] = None,
    conf: float = 0.15,
    progress_cb=None,
) -> Tuple[int, int]:
    input_abs = os.path.abspath(input_dir)
    if not os.path.isdir(input_abs):
        raise NotADirectoryError(f"input_dir is not a directory: {input_abs}")

    _ensure_dir(out_img_dir)
    _ensure_dir(out_xml_dir)

    model_path = model_path or (
        _DEFAULT_MODEL_PATH
        if os.path.exists(_DEFAULT_MODEL_PATH)
        else (_LEGACY_MODEL_PATH if os.path.exists(_LEGACY_MODEL_PATH) else "yolo26m.pt")
    )
    model = YOLO(model_path)

    processed = 0
    total_objects = 0
    for i, img_path in enumerate(_iter_images_recursively(input_abs), start=1):
        dst_img, dst_xml, n = export_one(
            img_path,
            out_img_dir=out_img_dir,
            out_xml_dir=out_xml_dir,
            label_name=label_name,
            model_path=model_path,
            model=model,
            conf=conf,
            input_root=input_abs,
        )
        processed += 1
        total_objects += n
        if progress_cb is not None:
            progress_cb(processed, total_objects, dst_img, dst_xml)

    return processed, total_objects


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
            QMessageBox,
        )
    except Exception as e:
        raise RuntimeError("PySide6 not installed. Please install PySide6 or run CLI mode.") from e

    class MainWindow(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("导出LabelImg XML")

            self.path_edit = QLineEdit()
            self.path_edit.setPlaceholderText("选择图片或文件夹")
            self.btn_open = QPushButton("选择图片")
            self.btn_open_dir = QPushButton("选择文件夹")

            self.conf_spin = QDoubleSpinBox()
            self.conf_spin.setDecimals(2)
            self.conf_spin.setSingleStep(0.05)
            self.conf_spin.setRange(0.0, 1.0)
            self.conf_spin.setValue(0.15)

            self.btn_export = QPushButton("导出XML")
            self.status = QLabel("")

            top = QHBoxLayout()
            top.addWidget(self.path_edit, 1)
            top.addWidget(self.btn_open)
            top.addWidget(self.btn_open_dir)

            mid = QHBoxLayout()
            mid.addWidget(QLabel("conf"))
            mid.addWidget(self.conf_spin)
            mid.addStretch(1)
            mid.addWidget(self.btn_export)

            root = QVBoxLayout()
            root.addLayout(top)
            root.addLayout(mid)
            root.addWidget(self.status)
            self.setLayout(root)

            self.btn_open.clicked.connect(self.on_open)
            self.btn_open_dir.clicked.connect(self.on_open_dir)
            self.btn_export.clicked.connect(self.on_export)

        def on_open(self):
            path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.jpg *.jpeg *.png *.bmp *.webp)")
            if not path:
                return
            self.path_edit.setText(path)

        def on_open_dir(self):
            path = QFileDialog.getExistingDirectory(self, "选择文件夹", "")
            if not path:
                return
            self.path_edit.setText(path)

        def on_export(self):
            path = self.path_edit.text().strip()
            if not path or not os.path.exists(path):
                QMessageBox.warning(self, "错误", "图片路径不存在")
                return
            try:
                if os.path.isdir(path):
                    def cb(processed, total_objects, last_img, last_xml):
                        self.status.setText(
                            f"已处理: {processed} 张 | 累计框: {total_objects} | 最后: {os.path.basename(last_img)}"
                        )
                        QApplication.processEvents()

                    processed, total_objects = export_folder(
                        path,
                        out_img_dir=_DEFAULT_OUT_IMG_DIR,
                        out_xml_dir=_DEFAULT_OUT_XML_DIR,
                        label_name=_DEFAULT_LABEL_NAME,
                        model_path=None,
                        conf=float(self.conf_spin.value()),
                        progress_cb=cb,
                    )
                    self.status.setText(f"完成: {processed} 张 | 累计框: {total_objects} | 输出: {_DEFAULT_OUT_XML_DIR}")
                else:
                    dst_img, dst_xml, n = export_one(
                        path,
                        out_img_dir=_DEFAULT_OUT_IMG_DIR,
                        out_xml_dir=_DEFAULT_OUT_XML_DIR,
                        label_name=_DEFAULT_LABEL_NAME,
                        model_path=None,
                        conf=float(self.conf_spin.value()),
                    )
                    self.status.setText(f"已导出: {os.path.basename(dst_img)} | 目标框: {n} | XML: {dst_xml}")
            except Exception as e:
                QMessageBox.critical(self, "导出失败", str(e))
                return

    app = QApplication([])
    w = MainWindow()
    w.resize(820, 120)
    w.show()
    return app.exec()


def main() -> int:
    input_path = _DEFAULT_INPUT_PATH
    model_path = (
        _DEFAULT_MODEL_PATH
        if os.path.exists(_DEFAULT_MODEL_PATH)
        else (_LEGACY_MODEL_PATH if os.path.exists(_LEGACY_MODEL_PATH) else "yolo26m.pt")
    )
    conf = 0.15
    out_img_dir = _DEFAULT_OUT_IMG_DIR
    out_xml_dir = _DEFAULT_OUT_XML_DIR
    label_name = _DEFAULT_LABEL_NAME

    if not input_path:
        return _run_gui()

    if os.path.isdir(input_path):
        processed, total_objects = export_folder(
            input_path,
            out_img_dir=out_img_dir,
            out_xml_dir=out_xml_dir,
            label_name=label_name,
            model_path=model_path,
            conf=float(conf),
        )
        print(f"processed images: {processed}")
        print(f"total objects:    {total_objects}")
        return 0

    dst_img, dst_xml, n = export_one(
        input_path,
        out_img_dir=out_img_dir,
        out_xml_dir=out_xml_dir,
        label_name=label_name,
        model_path=model_path,
        conf=float(conf),
    )
    print(f"exported image: {dst_img}")
    print(f"exported xml:   {dst_xml}")
    print(f"objects:        {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
