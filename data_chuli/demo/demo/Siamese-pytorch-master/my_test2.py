import os
import shutil
import random
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image
from siamese import Siamese
import matplotlib.ticker as mtick
import time

model = Siamese()

def process_images_with_varying_thresholds(src_dir, dst_dir, threshold):

    same_type_total = 0  # 同种类配对总次数
    same_type_below_threshold = 0  # 同种类小于阈值的次数


    # 遍历所有的子文件夹进行同种类配对
    for root, dirs, files in os.walk(src_dir):
        jpeg_files = [f for f in files if f.endswith('.jpeg')]
        if len(jpeg_files) < 2:
            continue  # 如果子文件夹中没有两张以上的图片，跳过

        # 获取相对路径
        # relative_path = os.path.relpath(root, src_dir)
        # target_dir = os.path.join(dst_dir, relative_path)

        # 两两配对同种类
        for i in range(len(jpeg_files)):
            for j in range(i+1, len(jpeg_files)):
                same_type_total += 1  # 记录同种类配对次数
                img1_path = os.path.join(root, jpeg_files[i])
                img2_path = os.path.join(root, jpeg_files[j])

                probability = satisfies_threshold(img1_path, img2_path)

                # 判断是否达到了阈值
                if probability < threshold:
                    same_type_below_threshold += 1  # 小于阈值的次数

                    # if not os.path.exists(target_dir):
                    #     os.makedirs(target_dir)
                    # shutil.copy2(img1_path, os.path.join(target_dir, jpeg_files[i]))
                    # shutil.copy2(img2_path, os.path.join(target_dir, jpeg_files[j]))

        if same_type_total >= 10000:
        #     print(f"同种类此轮配对次数:{same_type_total},阈值为:{threshold},小于阈值的次数:{same_type_below_threshold}")
        #     print("------------------------------------------------")
            break
    print(f"同种类此轮配对次数:{same_type_total},阈值为:{threshold},小于阈值的次数:{same_type_below_threshold}")
    return same_type_total, same_type_below_threshold

def satisfies_threshold(img1_path, img2_path):
    # 判断照片是否满足阈值
    image_1 = Image.open(img1_path)
    image_2 = Image.open(img2_path)
    probability = model.detect_image(image_1, image_2)
    return probability

def get_all_images_by_folder(src_dir):
    """按子文件夹分类收集所有jpeg文件的路径"""
    images_by_folder = {}
    for root, dirs, files in os.walk(src_dir):
        jpeg_files = [os.path.join(root, f) for f in files if f.endswith('.jpeg')]
        if jpeg_files:
            images_by_folder[root] = jpeg_files  # 将每个子文件夹的图片存入字典中
    return images_by_folder

def random_diff_folder_pairing(images_by_folder, threshold, diff_type_sum):
    """从不同子文件夹中随机选择配对，并记录配对次数和超过阈值的次数"""
    folder_keys = list(images_by_folder.keys())
    diff_type_total = 0
    diff_type_above_threshold = 0

    # 确保至少有两个不同子文件夹才能进行不同种类配对
    if len(folder_keys) < 2:
        return diff_type_total, diff_type_above_threshold

    while len(folder_keys) >= 2:
        # 随机选择两个不同的子文件夹
        folder1, folder2 = random.sample(folder_keys, 2)

        # 从两个不同的子文件夹中随机选择各一张图片
        img1_path = random.choice(images_by_folder[folder1])
        img2_path = random.choice(images_by_folder[folder2])

        diff_type_total += 1  # 记录不同种类配对次数
        size_difference = satisfies_threshold(img1_path, img2_path)

        if size_difference > threshold:
            diff_type_above_threshold += 1  # 记录大于阈值的次数

        # 从字典中移除空的文件夹，防止重复选择
        if len(images_by_folder[folder1]) == 1:
            del images_by_folder[folder1]
        else:
            images_by_folder[folder1].remove(img1_path)

        if len(images_by_folder[folder2]) == 1:
            del images_by_folder[folder2]
        else:
            images_by_folder[folder2].remove(img2_path)

        folder_keys = list(images_by_folder.keys())  # 更新文件夹列表

        if(diff_type_total == diff_type_sum):
            # print(f"不同种类的配对次数为:{diff_type_total}")
            # print("------------------------------------------------")
            break
    print(f"不同种类的配对次数为:{diff_type_total}")
    print("------------------------------------------------")

    return diff_type_total, diff_type_above_threshold

def frange(start, stop, step):
    """生成浮点数范围"""
    while start <= stop:
        yield start
        start += step

def set_chinese_font():
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体 SimHei 字体
    plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

def plot_graph(thresholds, same_type_rates, diff_type_rates):
    """绘制曲线图"""
    set_chinese_font()  # 调用设置中文字体的函数

    plt.plot(thresholds, same_type_rates, label='同种类小于阈值比率', marker='o')
    plt.plot(thresholds, diff_type_rates, label='不同种类大于阈值比率', marker='x')
    plt.xlabel('阈值')
    plt.ylabel('比率')
    plt.title('同种类和不同种类配对随阈值变化的比率')
    plt.legend()

    plt.grid()  # 显示网格
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))  # 将纵轴设置为百分比显示
    plt.tight_layout()

    plt.savefig("./my_test2.jpeg", dpi=300)
    time.sleep(2)
    plt.show()

if __name__ == '__main__':

    src_folder = 'E:\photo'
    dst_folder = '/path/to/destination_folder'
    diff_type_sum = 2000  # 不同种类配对总次数

    same_type_below_rates = []
    diff_type_above_rates = []

    thresholds = [round(i, 2) for i in frange(0.3, 0.65, 0.05)]
    for threshold in thresholds:
        # 从同子文件夹配对
        same_type_total, same_type_below_threshold = process_images_with_varying_thresholds(src_folder, dst_folder, threshold)

        # 从不同子文件夹配对
        all_images_by_folder = get_all_images_by_folder(src_folder)
        diff_type_total, diff_type_above_threshold = random_diff_folder_pairing(all_images_by_folder, threshold, diff_type_sum)

        # 计算同种类配对小于阈值的比率和不同种类配对大于阈值的比率
        same_type_below_rate = same_type_below_threshold / same_type_total if same_type_total > 0 else 0
        diff_type_above_rate = diff_type_above_threshold / diff_type_total if diff_type_total > 0 else 0

        same_type_below_rates.append(same_type_below_rate)
        diff_type_above_rates.append(diff_type_above_rate)


    # 绘制曲线图
    plot_graph(thresholds, same_type_below_rates, diff_type_above_rates)
    print(f"不同种类配对次数:{diff_type_total},同种类配对次数:{same_type_total}")
