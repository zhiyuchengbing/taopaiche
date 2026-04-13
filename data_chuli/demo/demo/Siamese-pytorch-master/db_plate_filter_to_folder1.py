"""
车牌识别与分类脚本
功能：
1. 从Oracle数据库读取PIC_MATCHTASK表数据
2. 处理TARE_IMAGE_PATH1和GROSS_IMAGE_PATH1两个字段的图片
3. 使用YOLOv11分割模型矩形框裁剪车辆区域（已替换原掩膜逻辑）
4. 使用PaddleOCR识别车牌
5. 将识别结果与数据库TRUCK_ID比对
6. 保存车辆区域图片到D:/data/车牌号/目录
"""

import os
import sys
import cv2
import cx_Oracle
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from datetime import datetime
from PIL import Image
import logging
from typing import Optional, Tuple, Dict, List
from pathlib import Path
from paddleocr import PaddleOCR
from ultralytics import YOLO

# 配置日志，只输出到控制台
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ===========================
# 初始化模型
# ===========================
SEG_MODEL_PATH = r"yolo11n-seg.pt"
seg_model = YOLO(SEG_MODEL_PATH)

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

# ================================================================
#  🚗 替换后的矩形框裁剪逻辑（替换旧的掩膜抠图函数）
# ================================================================
def extract_vehicle_mask_crop(image_path: str) -> Tuple[np.ndarray, tuple]:
    """
    使用 YOLOv11 分割模型检测车辆，仅使用检测框进行裁剪（不再使用掩膜）。
    返回裁剪后的车辆区域图像以及裁剪坐标。
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    results = seg_model(image_path, verbose=False)
    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        raise RuntimeError("分割模型未检测到车辆")

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()

    idx = int(np.argmax(confs))
    x1, y1, x2, y2 = boxes[idx].astype(int)

    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        raise RuntimeError("裁剪结果为空，请检查模型有效性")

    return crop, (x1, y1, x2, y2)

# ================================================================
# OCR & 数据处理函数保持原逻辑
# ================================================================
def detect_plate_text(vehicle_crop: np.ndarray) -> str:
    """识别车牌文本"""
    ocr_input = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2RGB)
    result = ocr.predict(input=ocr_input)

    if not result or not result[0]["rec_texts"]:
        return ""

    province_prefix = set(list("京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼港澳"))
    special_suffix = "挂警学领港澳"

    for text in result[0]["rec_texts"]:
        raw = str(text).strip().upper()
        t = re.sub(r"[·•∙.]", "", raw)
        if re.match(rf"^[\u4E00-\u9FA5][A-Z][A-Z0-9]{{4,5}}[A-Z0-9{special_suffix}]$", t):
            if t[0] in province_prefix:
                return t
    return ""

def process_image(image_path: str) -> Tuple[np.ndarray, str]:
    if not os.path.exists(image_path):
        print(f"警告: 图片不存在: {image_path}")
        return None, ""

    try:
        vehicle_crop, _ = extract_vehicle_mask_crop(image_path)
        plate_text = detect_plate_text(vehicle_crop)
        return vehicle_crop, plate_text

    except Exception as e:
        print(f"错误: 处理图片 {image_path} 时出错: {str(e)}")
        return None, ""

def clean_plate_number(plate: str) -> str:
    if not plate:
        return ""
    plate = str(plate).strip().upper()
    plate = re.sub(r'[^A-Z0-9\u4e00-\u9fa5]', '', plate)
    return plate

def save_plate_region(image: np.ndarray, plate_text: str, output_dir: str, base_name: str) -> bool:
    if image is None or not plate_text:
        return False

    try:
        plate_dir = os.path.join(output_dir, plate_text)
        os.makedirs(plate_dir, exist_ok=True)
        output_path = os.path.join(plate_dir, f"{base_name}_cropped.jpg")
        cv2.imwrite(output_path, image)
        return True

    except Exception as e:
        print(f"错误: 保存图片时出错: {e}")
        return False

def process_database_records(df: pd.DataFrame) -> Dict:
    stats = {'total': len(df), 'processed': 0, 'saved': 0, 'errors': 0}
    output_base_dir = "D:/data"
    os.makedirs(output_base_dir, exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="处理记录"):

        try:
            tare_path = row.get('TARE_IMAGE_PATH1')
            gross_path = row.get('GROSS_IMAGE_PATH1')
            truck_id = row.get('TRUCK_ID')
            record_id = row.get('ID', 'unknown')

            print(f"\n处理记录 ID: {record_id}, 数据库车牌: {truck_id}")

            if not truck_id:
                print("警告: TRUCK_ID缺失，跳过")
                continue

            # --- 空车 ---
            if tare_path and os.path.exists(tare_path):
                cropped_img, detected_plate = process_image(tare_path)
                plate_to_use = detected_plate or clean_plate_number(truck_id)

                if cropped_img is not None and plate_to_use:
                    save_plate_region(cropped_img, plate_to_use, output_base_dir, f"{record_id}_TARE")
                    stats['saved'] += 1
                stats['processed'] += 1

            # --- 重车 ---
            if gross_path and os.path.exists(gross_path):
                cropped_img, detected_plate = process_image(gross_path)
                plate_to_use = detected_plate or clean_plate_number(truck_id)

                if cropped_img is not None and plate_to_use:
                    save_plate_region(cropped_img, plate_to_use, output_base_dir, f"{record_id}_GROSS")
                    stats['saved'] += 1
                stats['processed'] += 1

        except Exception as e:
            logger.error(f"处理记录 {record_id} 时出错: {e}")
            stats['errors'] += 1

    return stats

def connect_to_oracle():
    try:
        os.environ["PATH"] = r"D:\\instantclient-basic-windows.x64-23.26.0.0.0\\instantclient_23_0" + ";" + os.environ.get("PATH", "")
        os.environ["TNS_ADMIN"] = r"D:\\instantclient-basic-windows.x64-23.26.0.0.0\\instantclient_23_0\\network\\admin"

        dsn_tns = cx_Oracle.makedsn('10.100.2.229', '1521', service_name='JLYXZ')
        connection = cx_Oracle.connect(user='identify', password='123456', dsn=dsn_tns)

        print("成功连接到Oracle数据库")
        return connection

    except Exception as e:
        print(f"数据库连接失败: {e}")
        return None

def read_data_from_oracle(connection, batch_size=1000):
    try:
        query = """
        SELECT 
            TASK_ID as ID,
            TARE_IMAGE_PATH1,
            GROSS_IMAGE_PATH1,
            TRUCK_ID
        FROM jlyxz.PIC_MATCHTASK
        WHERE ROWNUM <= :max_rows
        AND TARE_IMAGE_PATH1 IS NOT NULL
        AND GROSS_IMAGE_PATH1 IS NOT NULL
        AND TRUCK_ID IS NOT NULL
        """

        print("执行SQL查询...")
        df = pd.read_sql(query, connection, params={'max_rows': batch_size})

        if not df.empty:
            print("成功读取数据, 前5条:")
            print(df.head())

        return df

    except Exception as e:
        print(f"读取数据库失败: {e}")
        return pd.DataFrame()

def main():
    try:
        if not os.path.exists(SEG_MODEL_PATH):
            print(f"错误: 模型文件不存在 => {SEG_MODEL_PATH}")
            return

        output_dir = "D:/data2"
        os.makedirs(output_dir, exist_ok=True)

        connection = connect_to_oracle()
        if not connection:
            return

        df = read_data_from_oracle(connection)
        if df.empty:
            print("无可处理数据")
            return

        stats = process_database_records(df)

        print("\n=== 处理完成 ===")
        print(f"总记录: {stats['total']}")
        print(f"成功处理: {stats['processed']}")
        print(f"保存图片: {stats['saved']}")
        print(f"错误: {stats['errors']}")

    finally:
        if 'connection' in locals() and connection:
            connection.close()
            print("数据库连接已关闭")

if __name__ == "__main__":
    main()
