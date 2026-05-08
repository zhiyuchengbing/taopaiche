import os
import cv2
import numpy as np
from ultralytics import YOLO

# ===============================
# 参数设置
# ===============================
input_folder = r"datasets/20250926"       # 输入图片文件夹路径
output_root = r"output1"    # 输出根目录：output1/<父文件夹>/<图片>
os.makedirs(output_root, exist_ok=True)


model = YOLO('yolov8s.pt')  # 可改为 yolov8n.pt、yolov8m.pt 等

# ===============================77777777777777777777777777
# 主处理逻辑
# ===============================
vehicle_classes = [2, 3, 5, 6, 7]  # car, motorcycle, bus, train, truck
valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')

for root, _, files in os.walk(input_folder):
    for filename in files:
        if not filename.lower().endswith(valid_exts):
            continue

        image_path = os.path.join(root, filename)
        img = cv2.imread(image_path)
        if img is None:
            print(f"⚠️ 无法读取图像: {image_path}")
            continue

        results = model.predict(source=img, classes=vehicle_classes, conf=0.5, verbose=False)   #模型预测
        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            print(f"未检测到车辆: {image_path}")
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        if xyxy.size == 0:
            print(f"未检测到车辆: {image_path}")
            continue

        # 选择面积最大的框
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        idx = int(np.argmax(areas))
        x1, y1, x2, y2 = xyxy[idx]

        # 坐标取整并钳制到图像范围
        H, W = img.shape[:2]
        x1 = max(0, min(W - 1, int(x1)))
        y1 = max(0, min(H - 1, int(y1)))
        x2 = max(0, min(W, int(x2)))
        y2 = max(0, min(H, int(y2)))
        if x2 <= x1 or y2 <= y1:
            print(f"⚠️ 无效框，跳过: {image_path}")
            continue

        cropped_img = img[y1:y2, x1:x2]

        parent_name = os.path.basename(os.path.dirname(image_path))
        out_dir = os.path.join(output_root, parent_name)
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, filename)

        cv2.imwrite(save_path, cropped_img)
        print(f"完成: {image_path} -> {save_path}")

print("所有图片处理完成！")
