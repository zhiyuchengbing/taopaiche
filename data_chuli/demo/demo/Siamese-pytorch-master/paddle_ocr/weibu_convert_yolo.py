# -*- coding: utf-8 -*-
"""
将 labelme Pascal VOC XML 标注 (放大号 + 车挂号) 转换为 ultralytics YOLO 训练格式 (v2 双类)。
- 类别: 0 = fangdahao (放大号, ar<3.0), 1 = chegua (车挂号, ar>=3.0)
- 按图分层划分 train/val (8:2): 同一图的全部框不跨集, 且两类框在两边都有分布
- 干扰抑制: 纯数字/危险品区域未标注 = 天然背景负样本
- 输出: D:\data2\weibu_yolo_data_dual\ (全新, 不污染旧数据集)
"""
import os
import re
import shutil
import random
import xml.etree.ElementTree as ET

IMAGE_DIR = r"D:\data2\weibu_vehicle_crop\image"
LABEL_DIR = r"D:\data2\weibu_vehicle_crop\label"
OUT_DIR = r"D:\data2\weibu_yolo_data_dual"

CLASSES = ["fangdahao", "chegua"]  # 0: 放大号, 1: 车挂号
AR_THRESHOLD = 3.0


def normalize_plate(text):
    if not text:
        return ""
    t = str(text).upper().strip()
    t = re.sub(r"[\s\-_./·:：]+", "", t)
    t = re.sub(r"[^\u4e00-\u9fffA-Z0-9]", "", t)
    return t


def main():
    random.seed(42)
    xml_files = sorted(f for f in os.listdir(LABEL_DIR) if f.endswith(".xml"))

    # 收集标注
    records = []  # (img_name, img_path, list_of_(x1,y1,x2,y2,cls))
    skipped = 0
    for xml_name in xml_files:
        tree = ET.parse(os.path.join(LABEL_DIR, xml_name))
        root = tree.getroot()
        img_name = root.find("filename").text
        size = root.find("size")
        w = int(size.find("width").text)
        h = int(size.find("height").text)
        img_path = os.path.join(IMAGE_DIR, img_name)
        if not os.path.exists(img_path):
            print(f"跳过: 图片不存在 {img_name}")
            skipped += 1
            continue
        boxes = []
        for obj in root.findall("object"):
            label = normalize_plate(obj.find("name").text)
            bb = obj.find("bndbox")
            x1 = int(bb.find("xmin").text)
            y1 = int(bb.find("ymin").text)
            x2 = int(bb.find("xmax").text)
            y2 = int(bb.find("ymax").text)
            # 过滤掉明显过小的框 (可能是误标)
            if (x2 - x1) * (y2 - y1) < 800:
                print(f"  跳过小框: {label} ({x2-x1}x{y2-y1}) in {img_name}")
                continue
            # 双类划分: 宽高比 < 3.0 视为放大号(窄), 否则车挂号(宽)
            ar = (x2 - x1) / (y2 - y1) if y2 > y1 else 0
            cls = 0 if ar < AR_THRESHOLD else 1
            boxes.append((x1, y1, x2, y2, cls, w, h))
        records.append((img_name, img_path, boxes))

    # 按图分层划分: 先按图中是否有宽/窄框分组, 再各自8:2
    # 保证两类框在 train/val 两边都有分布, 同图框不跨集
    has_both = [r for r in records if any(b[4] == 0 for b in r[2]) and any(b[4] == 1 for b in r[2])]
    has_narrow_only = [r for r in records if any(b[4] == 0 for b in r[2]) and not any(b[4] == 1 for b in r[2])]
    has_wide_only = [r for r in records if any(b[4] == 1 for b in r[2]) and not any(b[4] == 0 for b in r[2])]
    empty = [r for r in records if not r[2]]

    val_records, train_records = [], []
    for group in [has_both, has_narrow_only, has_wide_only]:
        random.shuffle(group)
        n_val = max(1, int(len(group) * 0.2)) if group else 0
        val_records += group[:n_val]
        train_records += group[n_val:]

    print(f"总图片: {len(records)}, 训练: {len(train_records)}, 验证: {len(val_records)}, "
          f"跳过: {skipped}, 空框: {len(empty)}")
    print(f"分层: 双类图={len(has_both)}, 仅窄(放大号)={len(has_narrow_only)}, 仅宽(车挂号)={len(has_wide_only)}")

    # 创建目录
    for split in ["train", "val"]:
        os.makedirs(os.path.join(OUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, "labels", split), exist_ok=True)

    # 写文件
    n_boxes_train = 0
    n_boxes_val = 0
    cls_train = {0: 0, 1: 0}
    cls_val = {0: 0, 1: 0}
    for split, recs in [("train", train_records), ("val", val_records)]:
        for img_name, img_path, boxes in recs:
            base = os.path.splitext(img_name)[0]
            shutil.copy2(img_path, os.path.join(OUT_DIR, "images", split, img_name))
            label_path = os.path.join(OUT_DIR, "labels", split, base + ".txt")
            with open(label_path, "w", encoding="utf-8") as f:
                for x1, y1, x2, y2, cls, w, h in boxes:
                    cx = (x1 + x2) / 2.0 / w
                    cy = (y1 + y2) / 2.0 / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                    if split == "train":
                        n_boxes_train += 1
                        cls_train[cls] += 1
                    else:
                        n_boxes_val += 1
                        cls_val[cls] += 1

    print(f"训练框数: {n_boxes_train} (放大号{cls_train[0]} 车挂号{cls_train[1]}), "
          f"验证框数: {n_boxes_val} (放大号{cls_val[0]} 车挂号{cls_val[1]})")

    # data.yaml
    data_yaml = f"""path: {OUT_DIR}
train: images/train
val: images/val

names:
  0: fangdahao
  1: chegua
"""
    with open(os.path.join(OUT_DIR, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(data_yaml)

    print(f"完成。输出: {OUT_DIR}")


if __name__ == "__main__":
    main()
