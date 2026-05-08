import os
import shutil

def collect_images(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    count = 0

    # 递归遍历
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file in ["original1.jpg", "original2.jpg"]:
                src_path = os.path.join(root, file)


                parent_folder = os.path.basename(root)
                new_name = f"{parent_folder}_{file}"

                dst_path = os.path.join(dst_dir, new_name)

                shutil.copy2(src_path, dst_path)
                count += 1
                print(f"Copied: {src_path} -> {dst_path}")

    print(f"\n总共拷贝 {count} 张图片")


if __name__ == "__main__":
    src_dir = r"D:\project\data_chuli\demo\demo\Siamese-pytorch-master\exports\export_20260321_112931"   # 源目录
    dst_dir = r"D:\data2\套牌车数据集\01"   # 目标目录

    collect_images(src_dir, dst_dir)