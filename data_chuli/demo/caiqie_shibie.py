

import os
import cv2
import numpy as np
from ultralytics import YOLO
import hyperlpr3 as lpr3

# ===============================
# 配置
# ===============================
# 输入根目录（会递归遍历）
input_root = r"out"
# 输出根目录：output1/<原父文件夹>/<图片名>（文件名不变）
output_root = r"output1"

# 仅检测卡车（COCO 类别ID = 7），若要所有车辆可改为 [2,3,5,6,7]
vehicle_class_truck =[2,3,5,6,7]

# 其他参数
valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
conf_thresh = 0.5  # 目标检测置信度阈值

os.makedirs(output_root, exist_ok=True)

# ===============================
# 工具函数（兼容中文路径）
# ===============================
def imread_unicode(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def imwrite_unicode(path: str, image):
    ext = os.path.splitext(path)[1]
    if not ext:
        ext = ".jpg"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        return False
    buf.tofile(path)
    return True

# ===============================
# 模型初始化
# ===============================
# YOLOv8 检测模型（非分割）
model_det = YOLO("yolov8s.pt")  # 可换 yolov8n.pt / yolov8m.pt
# 车牌识别器
catcher = lpr3.LicensePlateCatcher()

# ===============================
# 主流程：卡车检测裁切 -> 车牌打码 -> 保存
# ===============================
for root, _, files in os.walk(input_root):
    for name in files:
        if not name.lower().endswith(valid_exts):
            continue

        src_path = os.path.join(root, name)
        # 输出目录：output1/<原父文件夹>/
        parent_name = os.path.basename(os.path.dirname(src_path))
        out_dir = os.path.join(output_root, parent_name)
        os.makedirs(out_dir, exist_ok=True)
        # 输出文件名与原图一致（保存最终“裁切+打码”结果）
        save_path = os.path.join(out_dir, name)

        # 断点续跑：已存在则跳过
        if os.path.exists(save_path):
            print(f"Skip (exists): {save_path}")
            continue

        # 读入原图
        img = imread_unicode(src_path)
        if img is None:
            print(f"无法读取: {src_path}")
            continue

        # 1) 卡车检测：取最大框
        det_res = model_det.predict(
            source=img, classes=vehicle_class_truck, conf=conf_thresh, verbose=False
        )[0]
        boxes = det_res.boxes
        if boxes is None or len(boxes) == 0:
            print(f"未检测到卡车: {src_path}")
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        if xyxy.size == 0:
            print(f"未检测到卡车: {src_path}")
            continue

        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        idx = int(np.argmax(areas))
        x1, y1, x2, y2 = xyxy[idx]

        H, W = img.shape[:2]
        x1 = max(0, min(W - 1, int(x1)))
        y1 = max(0, min(H - 1, int(y1)))
        x2 = max(0, min(W, int(x2)))
        y2 = max(0, min(H, int(y2)))
        if x2 <= x1 or y2 <= y1:
            print(f"无效卡车框: {src_path}")
            continue

        # 2) 裁切卡车区域
        crop = img[y1:y2, x1:x2].copy()

        # 3) 在裁切图上进行车牌识别并打码（黑色遮挡）
        result = catcher(crop)
        if result and len(result) > 0:
            masked = crop.copy()
            h, w = masked.shape[:2]
            for item in result:
                # HyperLPR3 result: [plate_text, conf, type, [x1,y1,x2,y2]]
                bbox = item[3] if len(item) > 3 else None
                if bbox is None or len(bbox) != 4:
                    continue
                px1, py1, px2, py2 = map(int, bbox)
                px1 = max(0, min(w - 1, px1))
                py1 = max(0, min(h - 1, py1))
                px2 = max(0, min(w, px2))
                py2 = max(0, min(h, py2))
                if px2 > px1 and py2 > py1:
                    cv2.rectangle(masked, (px1, py1), (px2, py2), (0, 0, 0), thickness=-1)
            out_img = masked
        else:
            # 未识别到车牌，直接使用裁切图
            out_img = crop

        # 4) 保存到 output1/<父文件夹>/<图片名>
        if imwrite_unicode(save_path, out_img):
            print(f"完成: {src_path} -> {save_path}")
        else:
            print(f"保存失败: {save_path}")