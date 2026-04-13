"""
车牌识别与分类脚本
功能：
1. 从Oracle数据库读取PIC_MATCHTASK表数据
2. 处理TARE_IMAGE_PATH1和GROSS_IMAGE_PATH1两个字段的图片
3. 使用YOLOv11分割模型裁剪车辆区域
4. 使用PaddleOCR识别车牌
5. 将识别结果与数据库TRUCK_ID比对
6. 保存车辆区域图片到D:/data/车牌号/目录

输出目录结构：
D:/data/
    ├── 粤A12345/
    │   ├── 1001_TARE_cropped.jpg
    │   └── 1001_GROSS_cropped.jpg
    └── 川B88888/
        └── ...
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
    handlers=[logging.StreamHandler()]  # 只保留控制台输出
)
logger = logging.getLogger(__name__)

# 初始化模型
SEG_MODEL_PATH = r"yolo11n-seg.pt"  # yolov11 分割模型权重
seg_model = YOLO(SEG_MODEL_PATH)
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

def extract_vehicle_mask_crop(image_path: str) -> Tuple[np.ndarray, tuple]:
    """
    使用 yolov11seg 进行车辆分割，按掩膜抠出面积最大的车辆并做最小外接矩形裁剪。
    返回抠图后的车辆图像和裁剪框坐标。
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    results = seg_model(image_path, verbose=False)
    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        raise RuntimeError("分割模型未检测到车辆")
    if result.masks is None or len(result.masks) == 0:
        raise RuntimeError("分割模型未返回掩膜")

    # 选择面积最大的掩膜
    masks = result.masks.data.cpu().numpy()  # [N, H, W]
    areas = masks.sum(axis=(1, 2))
    largest_idx = int(np.argmax(areas))
    mask = masks[largest_idx]

    # 将掩膜尺寸缩放到与原图一致
    h, w = image.shape[:2]
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0.5).astype(np.uint8)  # 0/1

    # 找到掩膜的外接矩形用于裁剪
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise RuntimeError("掩膜为空")
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    # 掩膜应用到原图（保持原分辨率，背景设为黑）
    masked = cv2.bitwise_and(image, image, mask=mask)
    crop = masked[y1: y2 + 1, x1: x2 + 1]

    if crop.size == 0:
        raise RuntimeError("掩膜裁剪结果为空")
    return crop, (x1, y1, x2, y2)

def detect_plate_text(vehicle_crop: np.ndarray) -> str:
    """识别车牌文本"""
    ocr_input = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2RGB)
    result = ocr.predict(input=ocr_input)
    
    if not result or not result[0]["rec_texts"]:
        return ""

    # 常见省份简称集合
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
    """
    处理单张图片：
    1. 使用YOLOv11分割车辆区域
    2. 使用PaddleOCR识别车牌
    3. 返回裁剪后的车辆图像和识别结果
    
    Returns:
        tuple: (cropped_image, plate_text) 或 (None, "") 如果处理失败
    """
    if not os.path.exists(image_path):
        print(f"警告: 图片不存在: {image_path}")
        return None, ""
    
    try:
        # 提取车辆区域
        vehicle_crop, _ = extract_vehicle_mask_crop(image_path)
        
        # 识别车牌
        plate_text = detect_plate_text(vehicle_crop)
        
        return vehicle_crop, plate_text
        
    except Exception as e:
        print(f"错误: 处理图片 {image_path} 时出错: {str(e)}")
        return None, ""

def clean_plate_number(plate: str) -> str:
    """清洗车牌号：去空格、转大写、去除特殊字符"""
    if not plate:
        return ""
    # 去空格、转大写
    plate = str(plate).strip().upper()
    # 去除特殊字符
    plate = re.sub(r'[^A-Z0-9\u4e00-\u9fa5]', '', plate)
    return plate

def save_plate_region(image: np.ndarray, plate_text: str, output_dir: str, base_name: str) -> bool:
    """
    保存车牌区域图片
    
    Args:
        image: 原始图像
        plate_text: 识别出的车牌号
        output_dir: 输出目录
        base_name: 基础文件名（不包含扩展名）
        
    Returns:
        bool: 是否保存成功
    """
    if image is None or not plate_text:
        return False
        
    try:
        # 创建输出目录
        plate_dir = os.path.join(output_dir, plate_text)
        os.makedirs(plate_dir, exist_ok=True)
        
        # 保存裁剪后的车辆图片
        output_path = os.path.join(plate_dir, f"{base_name}_cropped.jpg")
        cv2.imwrite(output_path, image)
        
        return True
        
    except Exception as e:
        print(f"错误: 保存图片时出错: {e}")
        return False

def process_database_records(df: pd.DataFrame) -> Dict:
    """
    处理数据库记录
    
    Args:
        df: 包含数据库记录的DataFrame
        
    Returns:
        dict: 处理统计信息
    """
    stats = {
        'total': len(df),
        'processed': 0,
        'saved': 0,
        'errors': 0
    }
    
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
                print("警告: 未找到TRUCK_ID, 跳过此记录")
                continue
                
            # 处理空重车图片
            if tare_path and os.path.exists(tare_path):
                print(f"处理空重车图片: {tare_path}")
                cropped_img, detected_plate = process_image(tare_path)
                if cropped_img is not None:
                    # 优先使用识别出的车牌号，如果识别失败则使用数据库中的车牌号
                    plate_to_use = detected_plate if detected_plate else clean_plate_number(truck_id)
                    if plate_to_use:
                        # 如果识别到车牌但与数据库不一致，给出提示
                        if detected_plate and detected_plate != clean_plate_number(truck_id):
                            print(f"注意: 识别车牌 '{detected_plate}' 与数据库车牌 '{truck_id}' 不一致，将使用识别结果")
                        
                        save_plate_region(
                            cropped_img, 
                            plate_to_use,
                            output_base_dir,
                            f"{record_id}_TARE"
                        )
                        stats['saved'] += 1
                        print(f"成功保存空重车图片, 使用车牌: {plate_to_use}")
                    else:
                        print("警告: 未识别到有效车牌")
                    stats['processed'] += 1
                else:
                    print("警告: 空重车图片处理失败")
                    
            # 处理重车图片
            if gross_path and os.path.exists(gross_path):
                print(f"处理重车图片: {gross_path}")
                cropped_img, detected_plate = process_image(gross_path)
                if cropped_img is not None:
                    # 优先使用识别出的车牌号，如果识别失败则使用数据库中的车牌号
                    plate_to_use = detected_plate if detected_plate else clean_plate_number(truck_id)
                    if plate_to_use:
                        # 如果识别到车牌但与数据库不一致，给出提示
                        if detected_plate and detected_plate != clean_plate_number(truck_id):
                            print(f"注意: 识别车牌 '{detected_plate}' 与数据库车牌 '{truck_id}' 不一致，将使用识别结果")
                        
                        save_plate_region(
                            cropped_img, 
                            plate_to_use,
                            output_base_dir,
                            f"{record_id}_GROSS"
                        )
                        stats['saved'] += 1
                        print(f"成功保存重车图片, 使用车牌: {plate_to_use}")
                    else:
                        print("警告: 未识别到有效车牌")
                    stats['processed'] += 1
                else:
                    print("警告: 重车图片处理失败")
                    
        except Exception as e:
            error_msg = f"处理记录 {record_id} 时出错: {e}"
            print(error_msg)
            logger.error(error_msg, exc_info=True)
            stats['errors'] += 1
            
    return stats

def connect_to_oracle():
    """连接到Oracle数据库"""
    try:
        # 设置Oracle客户端路径
        os.environ["PATH"] = r"D:\\instantclient-basic-windows.x64-23.26.0.0.0\\instantclient_23_0" + ";" + os.environ.get("PATH", "")
        os.environ["TNS_ADMIN"] = r"D:\\instantclient-basic-windows.x64-23.26.0.0.0\\instantclient_23_0\\network\\admin"

        # 创建DSN
        dsn_tns = cx_Oracle.makedsn(
            '10.100.2.229',  # 数据库服务器地址
            '1521',          # 端口
            service_name='JLYXZ'  # 服务名
        )
        
        # 连接数据库
        connection = cx_Oracle.connect(
            user='identify',
            password='123456',
            dsn=dsn_tns
        )
        
        print("成功连接到Oracle数据库")
        return connection
    except Exception as e:
        print(f"错误: 连接Oracle数据库失败: {e}")
        return None

def read_data_from_oracle(connection, batch_size=1000):
    """从Oracle读取PIC_MATCHTASK表数据"""
    try:
        # 使用CSV文件中的列名构建查询
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
        
        print("正在执行SQL查询...")
        df = pd.read_sql(query, connection, params={'max_rows': batch_size})
        
        # 打印前几行数据用于调试
        if not df.empty:
            print("成功读取数据，前5条记录：")
            print(df.head())
        else:
            print("警告: 查询返回空结果")
            
        return df
    except Exception as e:
        print(f"错误: 从数据库读取数据失败: {e}")
        return pd.DataFrame()

def main():
    """主函数"""
    try:
        # 检查模型文件是否存在
        if not os.path.exists(SEG_MODEL_PATH):
            error_msg = f"错误: 模型文件 {SEG_MODEL_PATH} 不存在"
            print(error_msg)
            logger.error(error_msg)
            return
            
        # 创建输出目录
        output_dir = "D:/data2"
        os.makedirs(output_dir, exist_ok=True)
        print(f"输出目录: {output_dir}")
        
        # 连接到数据库
        print("正在连接数据库...")
        connection = connect_to_oracle()
        if not connection:
            return
            
        # 读取数据
        print("正在从数据库读取数据...")
        df = read_data_from_oracle(connection)
        if df.empty:
            print("警告: 没有读取到数据")
            return
            
        # 处理数据
        print(f"开始处理 {len(df)} 条记录...")
        stats = process_database_records(df)
        
        # 输出统计信息
        print("\n" + "="*50)
        print("处理完成，统计信息:")
        print(f"总记录数: {stats['total']}")
        print(f"成功处理图片数: {stats['processed']}")
        print(f"成功保存图片数: {stats['saved']}")
        print(f"错误数: {stats['errors']}")
        print("="*50)
        
    except Exception as e:
        error_msg = f"处理过程中出错: {e}"
        print(error_msg)
        logger.error(error_msg, exc_info=True)
    finally:
        if 'connection' in locals() and connection:
            connection.close()
            print("数据库连接已关闭")

if __name__ == "__main__":
    main()