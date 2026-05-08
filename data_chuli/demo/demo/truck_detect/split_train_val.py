import os
import argparse
import random
import shutil
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET
from PIL import Image


_DEFAULT_IMAGES_DIR = r"D:\data2\套牌车数据集\01"
_DEFAULT_LABELS_DIR = r"D:\data2\套牌车数据集\01_label"
_DEFAULT_OUT_DIR = r"D:\data2\套牌车数据集\output0321加"

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _is_image_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in _IMAGE_EXTS


def _list_images(images_dir: str) -> List[str]:
    images_dir = os.path.abspath(images_dir)
    if not os.path.isdir(images_dir):
        raise NotADirectoryError(f"images_dir is not a directory: {images_dir}")

    out: List[str] = []
    for name in os.listdir(images_dir):
        p = os.path.join(images_dir, name)
        if os.path.isfile(p) and _is_image_file(p):
            out.append(p)
    return out


def _find_label_for_image(image_path: str, labels_dir: str) -> Optional[str]:
    stem = os.path.splitext(os.path.basename(image_path))[0]
    xml_path = os.path.join(labels_dir, f"{stem}.xml")
    if os.path.exists(xml_path) and os.path.isfile(xml_path):
        return xml_path
    return None


def _get_voc_objects(root: ET.Element) -> List[ET.Element]:
    return list(root.findall("object"))


def _parse_bbox_area(obj: ET.Element) -> Optional[Tuple[int, int, int, int, int]]:
    bnd = obj.find("bndbox")
    if bnd is None:
        return None

    def _read(tag: str) -> Optional[int]:
        node = bnd.find(tag)
        if node is None or node.text is None:
            return None
        try:
            return int(float(node.text.strip()))
        except Exception:
            return None

    xmin = _read("xmin")
    ymin = _read("ymin")
    xmax = _read("xmax")
    ymax = _read("ymax")
    if xmin is None or ymin is None or xmax is None or ymax is None:
        return None
    if xmax <= xmin or ymax <= ymin:
        return None
    area = (xmax - xmin) * (ymax - ymin)
    return xmin, ymin, xmax, ymax, area


def _analyze_xml_boxes(xml_path: str) -> Tuple[int, int]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objs = _get_voc_objects(root)
    valid = 0
    for obj in objs:
        if _parse_bbox_area(obj) is not None:
            valid += 1
    return len(objs), valid


def _read_size_from_xml(root: ET.Element) -> Optional[Tuple[int, int]]:
    size = root.find("size")
    if size is None:
        return None
    w = size.findtext("width")
    h = size.findtext("height")
    if w is None or h is None:
        return None
    try:
        return int(float(w)), int(float(h))
    except Exception:
        return None


def _largest_box_from_xml(root: ET.Element) -> Optional[Tuple[int, int, int, int]]:
    objs = _get_voc_objects(root)
    best = None
    best_area = -1
    for obj in objs:
        parsed = _parse_bbox_area(obj)
        if parsed is None:
            continue
        x1, y1, x2, y2, area = parsed
        if area > best_area:
            best_area = area
            best = (x1, y1, x2, y2)
    return best


def _write_yolo_txt_from_xml(src_xml: str, dst_txt: str, image_path: Optional[str] = None) -> Tuple[int, int]:
    tree = ET.parse(src_xml)
    root = tree.getroot()
    size = _read_size_from_xml(root)
    if size is None and image_path is not None and os.path.exists(image_path):
        img = Image.open(image_path).convert("RGB")
        size = img.size

    if size is None:
        raise RuntimeError(f"missing image size in xml and image not found: {src_xml}")

    w, h = size
    box = _largest_box_from_xml(root)

    _ensure_dir(os.path.dirname(dst_txt))

    if box is None:
        with open(dst_txt, "w", encoding="utf-8") as f:
            f.write("")
        return 0, 0

    x1, y1, x2, y2 = box
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w, int(x2)))
    y2 = max(0, min(h, int(y2)))
    bw = max(0, x2 - x1)
    bh = max(0, y2 - y1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0

    line = f"0 {cx / w:.6f} {cy / h:.6f} {bw / w:.6f} {bh / h:.6f}"
    with open(dst_txt, "w", encoding="utf-8") as f:
        f.write(line + "\n")
    return 1, 1


def _write_xml_keep_largest_box(src_xml: str, dst_xml: str) -> Tuple[int, int]:
    tree = ET.parse(src_xml)
    root = tree.getroot()
    objs = _get_voc_objects(root)

    candidates: List[Tuple[int, ET.Element]] = []
    for obj in objs:
        parsed = _parse_bbox_area(obj)
        if parsed is None:
            continue
        area = parsed[4]
        candidates.append((area, obj))

    if not candidates:
        return len(objs), 0

    candidates.sort(key=lambda x: x[0], reverse=True)
    keep_obj = candidates[0][1]

    for obj in objs:
        if obj is keep_obj:
            continue
        root.remove(obj)

    _ensure_dir(os.path.dirname(dst_xml))
    tree.write(dst_xml, encoding="utf-8", xml_declaration=True)
    return len(objs), 1


def _pair_images_and_labels(images_dir: str, labels_dir: str) -> Tuple[List[Tuple[str, str]], Dict[str, int]]:
    labels_dir = os.path.abspath(labels_dir)
    if not os.path.isdir(labels_dir):
        raise NotADirectoryError(f"labels_dir is not a directory: {labels_dir}")

    images = _list_images(images_dir)

    pairs: List[Tuple[str, str]] = []
    stats = {
        "images_total": len(images),
        "pairs": 0,
        "missing_xml": 0,
        "invalid_xml": 0,
        "empty_xml": 0,
        "multi_xml": 0,
    }

    for img in images:
        xml = _find_label_for_image(img, labels_dir)
        if xml is None:
            stats["missing_xml"] += 1
            continue

        try:
            total_objs, valid_objs = _analyze_xml_boxes(xml)
        except Exception:
            stats["invalid_xml"] += 1
            continue

        if valid_objs <= 0:
            stats["empty_xml"] += 1

        if valid_objs > 1:
            stats["multi_xml"] += 1

        pairs.append((img, xml))

    stats["pairs"] = len(pairs)
    return pairs, stats


def _split_indices(n: int, train_ratio: float) -> Tuple[List[int], List[int]]:
    idxs = list(range(n))
    train_n = int(round(n * train_ratio))
    train_n = max(0, min(n, train_n))
    train_idx = idxs[:train_n]
    val_idx = idxs[train_n:]
    return train_idx, val_idx


def _copy_or_move(src: str, dst: str, move: bool) -> None:
    _ensure_dir(os.path.dirname(dst))
    if move:
        shutil.move(src, dst)
    else:
        shutil.copy2(src, dst)


def split_dataset(
    images_dir: str = _DEFAULT_IMAGES_DIR,
    labels_dir: str = _DEFAULT_LABELS_DIR,
    out_dir: str = _DEFAULT_OUT_DIR,
    train_ratio: float = 0.8,
    seed: int = 42,
    move: bool = False,
) -> Dict[str, int]:
    pairs, stats = _pair_images_and_labels(images_dir, labels_dir)

    if len(pairs) == 0:
        raise RuntimeError("no valid image/xml pairs found")

    random.Random(seed).shuffle(pairs)

    train_idx, val_idx = _split_indices(len(pairs), train_ratio=train_ratio)

    train_img_dir = os.path.join(out_dir, "images", "train")
    train_lab_dir = os.path.join(out_dir, "labels", "train")
    val_img_dir = os.path.join(out_dir, "images", "val")
    val_lab_dir = os.path.join(out_dir, "labels", "val")

    _ensure_dir(train_img_dir)
    _ensure_dir(train_lab_dir)
    _ensure_dir(val_img_dir)
    _ensure_dir(val_lab_dir)

    copied = {
        "train": 0,
        "val": 0,
    }

    checked = {
        "xml_fixed": 0,
        "xml_skipped": 0,
    }

    for i in train_idx:
        img, xml = pairs[i]
        img_dst = os.path.join(train_img_dir, os.path.basename(img))
        stem = os.path.splitext(os.path.basename(img))[0]
        txt_dst = os.path.join(train_lab_dir, stem + ".txt")
        try:
            total_objs, valid_objs = _analyze_xml_boxes(xml)
        except Exception:
            checked["xml_skipped"] += 1
            continue

        try:
            _write_yolo_txt_from_xml(xml, txt_dst, image_path=img)
        except Exception:
            checked["xml_skipped"] += 1
            continue

        if valid_objs > 1:
            checked["xml_fixed"] += 1

        _copy_or_move(img, img_dst, move=move)
        copied["train"] += 1

    for i in val_idx:
        img, xml = pairs[i]
        img_dst = os.path.join(val_img_dir, os.path.basename(img))
        stem = os.path.splitext(os.path.basename(img))[0]
        txt_dst = os.path.join(val_lab_dir, stem + ".txt")
        try:
            total_objs, valid_objs = _analyze_xml_boxes(xml)
        except Exception:
            checked["xml_skipped"] += 1
            continue

        try:
            _write_yolo_txt_from_xml(xml, txt_dst, image_path=img)
        except Exception:
            checked["xml_skipped"] += 1
            continue

        if valid_objs > 1:
            checked["xml_fixed"] += 1

        _copy_or_move(img, img_dst, move=move)
        copied["val"] += 1

    return {
        **stats,
        "train": copied["train"],
        "val": copied["val"],
        "xml_fixed": checked["xml_fixed"],
        "xml_skipped": checked["xml_skipped"],
        "seed": seed,
        "moved": int(bool(move)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Split LabelImg (image+xml) dataset into train/val")
    parser.add_argument("--images-dir", default=_DEFAULT_IMAGES_DIR, help="source images dir")
    parser.add_argument("--labels-dir", default=_DEFAULT_LABELS_DIR, help="source xml labels dir")
    parser.add_argument("--out-dir", default=_DEFAULT_OUT_DIR, help="output split dir")
    parser.add_argument("--train", type=float, default=0.8, help="train ratio, e.g. 0.8")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--move", action="store_true", help="move instead of copy")
    args = parser.parse_args()

    stats = split_dataset(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        out_dir=args.out_dir,
        train_ratio=float(args.train),
        seed=int(args.seed),
        move=bool(args.move),
    )

    print("split done")
    print(f"images_total: {stats['images_total']}")
    print(f"pairs:        {stats['pairs']}")
    print(f"missing_xml:  {stats['missing_xml']}")
    print(f"invalid_xml:  {stats['invalid_xml']}")
    print(f"empty_xml:    {stats['empty_xml']}")
    print(f"multi_xml:    {stats['multi_xml']}")
    print(f"train:        {stats['train']}")
    print(f"val:          {stats['val']}")
    print(f"xml_fixed:    {stats['xml_fixed']}")
    print(f"xml_skipped:  {stats['xml_skipped']}")
    print(f"out_dir:      {os.path.abspath(args.out_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
