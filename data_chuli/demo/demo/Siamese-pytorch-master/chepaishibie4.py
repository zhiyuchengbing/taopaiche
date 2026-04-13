import os
import cv2
import csv
import shutil
import hyperlpr3 as lpr3

# ===============================
# 参数设置
# ===============================
image_folder = r"E:\\套牌车识别项目\demo\\output1\\川ADP799"             # 输入图片文件夹路径
output_csv = os.path.join(image_folder, "plate_results.csv")  # 输出CSV文件
abnormal_folder = os.path.join("datasets", "yichang")         # 未识别图片存放路径

# ===============================
# 文件夹准备
# ===============================
os.makedirs(abnormal_folder, exist_ok=True)

# ===============================
# 初始化车牌识别器
# ===============================
catcher = lpr3.LicensePlateCatcher()

# ===============================
# 准备CSV输出
# ===============================
total = 0
recognized = 0
unrecognized = 0

with open(output_csv, mode='w', newline='', encoding='utf-8-sig') as file:
    writer = csv.writer(file)
    writer.writerow(["license_plate", "image_path"])  # 表头

    # 遍历所有图片
    for root, dirs, files in os.walk(image_folder):
        for name in files:
            if name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                total += 1
                image_path = os.path.join(root, name)
                image = cv2.imread(image_path)

                if image is None:
                    print(f"[跳过] 无法读取图像: {image_path}")
                    continue

                # 车牌识别
                result = catcher(image)

                # result 格式: [['桂BX6016', np.float32(0.9958), 1, [x1,y1,x2,y2]]]
                if result and len(result) > 0:
                    plate_text = result[0][0]
                    recognized += 1
                    print(f"[识别成功] {name} -> {plate_text}")
                    writer.writerow([plate_text, image_path])
                else:
                    unrecognized += 1
                    print(f"[未识别] {name}")
                    writer.writerow(["未识别", image_path])

                    # 复制未识别图片到异常文件夹
                    dst_path = os.path.join(abnormal_folder, name)
                    shutil.copy(image_path, dst_path)

print("\n===========================================")
print(f"✅ 批量识别完成，共处理 {total} 张图片")
print(f"✅ 识别成功: {recognized} 张")
print(f"⚠️ 未识别: {unrecognized} 张（已保存到 {abnormal_folder}）")
print(f"📄 识别结果已保存至: {os.path.abspath(output_csv)}")
print("===========================================")
