import os
import cv2
import shutil
import numpy as np
import hyperlpr3 as lpr3
# 车牌识别
# ===============================
# 参数设置（单张图片）
# ===============================
image_path = r".\out\\川ADP799\\20230728101208_tile3.jpg"   # 单张图片路径
abnormal_folder = os.path.join("datasets", "yichang")  # 未识别图片存放路径

# 规范化路径，避免分隔符问题
image_path = os.path.normpath(image_path)

# ===============================
# 文件夹准备
# ===============================
os.makedirs(abnormal_folder, exist_ok=True)

# ===============================
# 初始化车牌识别器
# ===============================
catcher = lpr3.LicensePlateCatcher()

# ===============================
# 单图识别：只打印结果
# ===============================
if not os.path.exists(image_path):
    raise FileNotFoundError(f"图片不存在: {image_path}")

# Windows 下 OpenCV 对含中文路径可能返回 None，改用 fromfile+imdecode
def imread_unicode(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

image = imread_unicode(image_path)
if image is None:
    raise RuntimeError(f"无法读取图像(Unicode路径): {os.path.abspath(image_path)}")

result = catcher(image)
# result 示例: [['桂BX6016', np.float32(0.9958), 1, [x1,y1,x2,y2]]]
if result and len(result) > 0:
    print(f"[识别成功] {os.path.basename(image_path)}")
    # 打印完整结果结构
    print("完整结果:", result)

    # 逐条打印车牌、置信度、位置，并对车牌区域进行遮挡（黑色）
    masked = image.copy()
    H, W = masked.shape[:2]
    for i, item in enumerate(result):
        try:
            plate_text = item[0]
            conf = float(item[1]) if len(item) > 1 else None
            bbox = item[3] if len(item) > 3 else None  # [x1,y1,x2,y2]
            print(f"  #{i+1}: plate={plate_text}, conf={conf}, bbox={bbox}")

            if bbox is not None and len(bbox) == 4:
                x1, y1, x2, y2 = map(int, bbox)
                x1 = max(0, min(W - 1, x1))
                y1 = max(0, min(H - 1, y1))
                x2 = max(0, min(W, x2))
                y2 = max(0, min(H, y2))
                if x2 > x1 and y2 > y1:
                    # 使用黑色矩形填充遮挡车牌区域
                    cv2.rectangle(masked, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)
        except Exception as e:
            print(f"  #{i+1}: 无法解析条目 {item} -> {e}")

    # 保存遮挡后的图片（与原图同目录，文件名加 _masked）- 兼容中文路径
    in_dir = os.path.dirname(image_path)
    name, ext = os.path.splitext(os.path.basename(image_path))
    ext = ext if ext else ".jpg"
    save_path = os.path.join(in_dir, f"{name}_masked{ext}")
    ok, buf = cv2.imencode(ext, masked)
    if ok:
        buf.tofile(save_path)
        print(f"[已保存遮挡图] {save_path}")
    else:
        print("⚠️ 保存遮挡图失败：imencode 返回 False")
else:
    print(f"[未识别] {os.path.basename(image_path)}")
    os.makedirs(abnormal_folder, exist_ok=True)
    dst_path = os.path.join(abnormal_folder, os.path.basename(image_path))
    shutil.copy(image_path, dst_path)
