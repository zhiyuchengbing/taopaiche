# -*- coding: utf-8 -*-
"""从合并后的 data(图)+label(VOC XML) 重建分层 train/val split 到 split_v2。

- 复用 split_train_val.py 的配对与"每图保留最大框"VOC->YOLO 转换逻辑。
- 剔除：缺 XML、空标注(0 个有效框)、转换失败的条目。
- 按磅点分层 85/15：磅点从文件名解析（新命名 {p}_1_… 取首段；旧命名 2025__11__{d}__selected__{p}_1_… 取 selected 后首段）。
- 输出 D:\data2\truck\split_v2\images\{train,val} + labels\{train,val} + data.yaml。
- 旧 D:\data2\truck\split 不动（回退）。
"""
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from split_train_val import (  # noqa: E402
    _pair_images_and_labels,
    _analyze_xml_boxes,
    _write_yolo_txt_from_xml,
    _ensure_dir,
)

IMAGES_DIR = r"D:\data2\truck\data"
LABELS_DIR = r"D:\data2\truck\label"
OUT_DIR = r"D:\data2\truck\split_v2"
TRAIN_RATIO = 0.85
SEED = 20260813


def parse_point(fname: str) -> str:
    """从文件名解析磅点。返回 str 或 'unknown'。"""
    parts = fname.split("__")
    if "selected" in parts:
        idx = parts.index("selected")
        if idx + 1 < len(parts):
            first = parts[idx + 1].split("_")[0]
            if first.isdigit():
                return first
        return "unknown"
    first = fname.split("_")[0]
    return first if first.isdigit() else "unknown"


def main():
    pairs, stats = _pair_images_and_labels(IMAGES_DIR, LABELS_DIR)
    print("=== 配对统计 ===")
    for k in ("images_total", "pairs", "missing_xml", "invalid_xml", "empty_xml", "multi_xml"):
        print(f"  {k}: {stats[k]}")

    # 剔除空标注
    valid = []
    for img, xml in pairs:
        try:
            _, v = _analyze_xml_boxes(xml)
        except Exception:
            continue
        if v <= 0:
            continue
        valid.append((img, xml))
    print(f"剔除空标注/无效后有效对: {len(valid)}")

    # 按磅点分层切分
    buckets = defaultdict(list)
    for img, xml in valid:
        b = parse_point(os.path.basename(img))
        buckets[b].append((img, xml))

    rng = random.Random(SEED)
    train_pairs, val_pairs = [], []
    print("=== 磅点分层 ===")
    for b in sorted(buckets):
        items = buckets[b]
        rng.shuffle(items)
        n = len(items)
        nt = int(round(n * TRAIN_RATIO))
        if nt >= n:
            nt = n - 1 if n > 1 else 0  # 保证每桶 val 至少 1(若可能)
        train_pairs.extend(items[:nt])
        val_pairs.extend(items[nt:])
        print(f"  磅点 {b}: 总数 {n} -> train {nt} / val {n - nt}")

    t_img = os.path.join(OUT_DIR, "images", "train")
    t_lab = os.path.join(OUT_DIR, "labels", "train")
    v_img = os.path.join(OUT_DIR, "images", "val")
    v_lab = os.path.join(OUT_DIR, "labels", "val")
    for d in (t_img, t_lab, v_img, v_lab):
        _ensure_dir(d)

    def write_set(pairs, img_dir, lab_dir):
        import shutil
        ok = skipped = 0
        for img, xml in pairs:
            stem = os.path.splitext(os.path.basename(img))[0]
            txt = os.path.join(lab_dir, stem + ".txt")
            try:
                _write_yolo_txt_from_xml(xml, txt, image_path=img)
            except Exception:
                skipped += 1
                continue
            shutil.copy2(img, os.path.join(img_dir, os.path.basename(img)))
            ok += 1
        return ok, skipped

    nt_ok, nt_skip = write_set(train_pairs, t_img, t_lab)
    nv_ok, nv_skip = write_set(val_pairs, v_img, v_lab)
    print(f"\ntrain 写入: {nt_ok} (跳过 {nt_skip}) | val 写入: {nv_ok} (跳过 {nv_skip})")

    # data.yaml
    yaml_path = os.path.join(OUT_DIR, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {OUT_DIR.replace(os.sep, '/')}\n")
        f.write("train: images/train\nval: images/val\nnames:\n  0: truck\n")
    print(f"data.yaml -> {yaml_path}")


if __name__ == "__main__":
    main()
