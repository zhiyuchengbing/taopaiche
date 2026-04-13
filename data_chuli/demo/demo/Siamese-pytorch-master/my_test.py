import os
import shutil
from PIL import Image
from siamese import Siamese

model = Siamese()
num = 0
num_faults = 0

def process_images(src_dir, dst_dir, threshold, log_file_path):
    # 打开记录文件
    with open(log_file_path, 'w') as log_file:
        # 遍历所有的子文件夹
        for root, dirs, files in os.walk(src_dir):
            jpeg_files = [f for f in files if f.endswith('.jpeg')]
            if len(jpeg_files) < 2:
                continue  # 如果子文件夹中没有两张以上的图片，跳过

            # 获取相对路径
            relative_path = os.path.relpath(root, src_dir)
            target_dir = os.path.join(dst_dir, relative_path)

            # 两两配对
            for i in range(len(jpeg_files)):
                for j in range(i+1, len(jpeg_files)):
                    img1_path = os.path.join(root, jpeg_files[i])
                    img2_path = os.path.join(root, jpeg_files[j])

                    # 输出照片路径配对
                    # print(f"配对: {img1_path} 和 {img2_path}")

                    # 获取照片大小差异
                    probability = satisfies_threshold(img1_path, img2_path)


                    # 判断是否达到了阈值
                    if probability < threshold:
                        # 记录不满足阈值的配对
                        log_file.write(f"不满足阈值的配对: {img1_path} 和 {img2_path}, 相似度: {probability} \n")
                        print(f"不满足阈值的配对: {img1_path} 和 {img2_path}, 相似度: {probability} \n")

                        # 如果未达到阈值，则复制到目标文件夹
                        if not os.path.exists(target_dir):
                            os.makedirs(target_dir)

                        # 复制 img1
                        target_img1_path = os.path.join(target_dir, jpeg_files[i])
                        if not os.path.exists(target_img1_path):
                            shutil.copy2(img1_path, target_img1_path)

                        # 复制 img2
                        target_img2_path = os.path.join(target_dir, jpeg_files[j])
                        if not os.path.exists(target_img2_path):
                            shutil.copy2(img2_path, target_img2_path)


def satisfies_threshold(img1_path, img2_path):
    # 判断照片是否满足阈值
    image_1 = Image.open(img1_path)
    image_2 = Image.open(img2_path)
    probability = model.detect_image(image_1, image_2)
    return probability


src_folder = 'E:\photo'
dst_folder = r'E:\test'
size_threshold = 0.3  # 相似度阈值
log_file = r'E:\telog.txt'

process_images(src_folder, dst_folder, size_threshold, log_file)

