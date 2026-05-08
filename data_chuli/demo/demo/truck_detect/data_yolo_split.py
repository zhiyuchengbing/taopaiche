import os
import random
import shutil

# ====== 配置路径 ======
IMAGES_DIR = r"D:\data2\套牌车数据集\01"
LABELS_DIR = r"D:\data2\套牌车数据集\01_label"
OUT_DIR = r"D:\data2\套牌车数据集\output0321加"

TRAIN_RATIO = 0.8
SEED = 42

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image(file):
    return os.path.splitext(file)[1].lower() in IMAGE_EXTS


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def collect_pairs(images_dir, labels_dir):
    pairs = []

    for file in os.listdir(images_dir):
        if not is_image(file):
            continue

        img_path = os.path.join(images_dir, file)
        name = os.path.splitext(file)[0]
        label_path = os.path.join(labels_dir, name + ".txt")

        if os.path.exists(label_path):
            pairs.append((img_path, label_path))
        else:
            print(f"⚠️ 缺少标签: {file}")

    return pairs


def split_dataset():
    pairs = collect_pairs(IMAGES_DIR, LABELS_DIR)

    if len(pairs) == 0:
        raise RuntimeError("没有找到有效数据")

    random.seed(SEED)
    random.shuffle(pairs)

    train_size = int(len(pairs) * TRAIN_RATIO)

    train_pairs = pairs[:train_size]
    val_pairs = pairs[train_size:]

    # ====== 目标目录 ======
    train_img_dir = os.path.join(OUT_DIR, "images", "train")
    train_lab_dir = os.path.join(OUT_DIR, "labels", "train")
    val_img_dir = os.path.join(OUT_DIR, "images", "val")
    val_lab_dir = os.path.join(OUT_DIR, "labels", "val")

    for d in [train_img_dir, train_lab_dir, val_img_dir, val_lab_dir]:
        ensure_dir(d)

    # ====== 拷贝 ======
    def copy_data(pairs, img_dir, lab_dir):
        for img, lab in pairs:
            shutil.copy2(img, os.path.join(img_dir, os.path.basename(img)))
            shutil.copy2(lab, os.path.join(lab_dir, os.path.basename(lab)))

    copy_data(train_pairs, train_img_dir, train_lab_dir)
    copy_data(val_pairs, val_img_dir, val_lab_dir)

    print("✅ 划分完成")
    print(f"总数: {len(pairs)}")
    print(f"训练集: {len(train_pairs)}")
    print(f"验证集: {len(val_pairs)}")


if __name__ == "__main__":
    split_dataset()