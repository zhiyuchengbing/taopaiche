import os
import cv2
import numpy as np
from ultralytics import YOLO

# ===============================
# 参数设置
# ===============================
input_folder = r"datasets/20250926"       # 输入图片文件夹路径
output_folder = r"datasets/truck_only"    # 输出结果文件夹路径
os.makedirs(output_folder, exist_ok=True)

# 加载 YOLOv8 分割模型（COCO 预训练）
model = YOLO('yolov8s-seg.pt')  # 也可改为 yolov8n-seg.pt、yolov8m-seg.pt 等

# ===============================77777777777777777777777777
# 主处理逻辑
# ===============================
for filename in os.listdir(input_folder):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    image_path = os.path.join(input_folder, filename)
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ 无法读取图像: {filename}")
        continue

    # 仅检测卡车（COCO 类别 ID = 7）
    results = model.predict(source=img, classes=[7], conf=0.5, verbose=False)
    result = results[0]

    if result.masks is None or len(result.masks.data) == 0:
        print(f"❌ 未检测到卡车: {filename}")
        continue

    # ===============================
    # 合并多个卡车 mask
    # ===============================
    combined_mask = np.zeros(result.masks.data[0].shape, dtype=np.uint8)
    for mask in result.masks.data:
        m = mask.cpu().numpy()
        combined_mask = np.logical_or(combined_mask, m)

    # 转换为 uint8 格式
    combined_mask = (combined_mask * 255).astype(np.uint8)

    # ⚠️ 调整 mask 尺寸与原图一致
    combined_mask = cv2.resize(combined_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # ===============================
    # 生成只包含卡车的图像
    # ===============================
    masked_img = cv2.bitwise_and(img, img, mask=combined_mask)

    # ===============================
    # 自动裁剪卡车区域（去除多余背景）
    # ===============================
    coords = cv2.findNonZero(combined_mask)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        cropped_img = masked_img[y:y+h, x:x+w]
    else:
        cropped_img = masked_img

    # ===============================
    # 保存结果
    # ===============================
    save_path = os.path.join(output_folder, filename)
    cv2.imwrite(save_path, cropped_img)
    print(f"卡车剪裁完成: {filename} -> {save_path}")

print("\所有图片处理完成！")
