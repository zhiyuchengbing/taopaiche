from PIL import Image
from pathlib import Path

# 批量处理：遍历 images 根目录，裁出“前四幅”中的第三幅，并保存到 out/<父文件夹>/<文件名>_tile3.jpg
SRC_ROOT = Path(r"f:\汽车衡数据集\images")
OUT_ROOT = SRC_ROOT.parent / "out"

# 固定前四幅布局的单块尺寸
TW, TH = 2560, 1440


def crop_third_tile(img: Image.Image) -> Image.Image:
    # 第三幅 = 第2行第1列 -> (x1,y1,x2,y2) = (0, TH, TW, 2*TH)
    return img.crop((0, TH, TW, 2 * TH))


def process_image(src_path: Path) -> None:
    try:
        out_dir = OUT_ROOT / src_path.parent.name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{src_path.stem}_tile3.jpg"
        if out_path.exists():
            print(f"Skip (exists): {out_path}")
            return
        with Image.open(src_path) as img:
            W, H = img.size
            if W < TW or H < 2 * TH:
                print(f"Skip (too small): {src_path} -> {W}x{H}")
                return
            crop = crop_third_tile(img)
            crop.save(out_path, quality=95)
            print(f"Saved: {out_path}")
    except Exception as e:
        print(f"Error: {src_path} -> {e}")


def main() -> None:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [p for p in SRC_ROOT.rglob("*") if p.suffix.lower() in exts]
    print(f"Found {len(files)} images under {SRC_ROOT}")
    pending = []
    for p in files:
        out_dir = OUT_ROOT / p.parent.name
        out_path = out_dir / f"{p.stem}_tile3.jpg"
        if not out_path.exists():
            pending.append(p)
    print(f"Pending {len(pending)} images to process")
    for p in pending:
        process_image(p)


if __name__ == "__main__":
    main()