"""
套牌车识别GUI界面
用途：实现一个图形用户界面，用于选择两张图片并进行套牌车识别
功能：
1. 提供两个图片选择按钮，让用户选择要比较的两张车辆图片
2. 显示选中的图片预览
3. 调用Siamese模型进行图片相似度检测
4. 显示识别结果（是否为同一辆车或疑似套牌车辆）
技术栈：PySide6用于GUI设计，PIL用于图片处理
# 手动实现车头的检测    将车头区域拿出来 进行检测#
"""

import sys
import os
import re
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import cx_Oracle
import pandas as pd
from PIL import Image, ImageQt
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                               QMessageBox, QTextEdit, QDialog, QScrollArea,
                               QProgressDialog)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap, QImage
from typing import Tuple, Optional
from paddleocr import PaddleOCR
from ultralytics import YOLO

from siamese import Siamese
parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from data_chuli.cropper import VehicleCropper


class PlateRecognizer:
    """车牌识别器，用于从车辆图片中识别车牌号"""
    
    def __init__(self, seg_model_path: str = r"D:\project\yolo11n-seg.pt"):
        """
        初始化车牌识别器
        
        参数:
            seg_model_path: 分割模型路径
        """
        self.seg_model = YOLO(seg_model_path)
        # 车头/车尾检测模型，用于在车辆区域中进一步裁剪车头
        self.headtail_model = YOLO(r"D:\data2\runs\detect\train\weights\best.pt")
        self.ocr = PaddleOCR(

        )
        # 车牌格式相关配置
        self.province_prefix = set("京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼港澳")
        self.special_suffix = "挂警学领港澳"

    def extract_vehicle_mask_crop(self, image_path: str) -> np.ndarray:
        """提取车辆区域"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"无法读取图像: {image_path}")

            results = self.seg_model(image_path, verbose=False)
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
            crop = masked[y1:y2 + 1, x1:x2 + 1]

            if crop.size == 0:
                raise RuntimeError("掩膜裁剪结果为空")
            return crop
            
        except Exception as e:
            raise RuntimeError(f"提取车辆区域失败: {str(e)}")

    def is_valid_plate(self, text: str) -> bool:
        """验证车牌格式是否正确"""
        text = str(text).strip().upper()
        text = re.sub(r"[·•∙.]", "", text)
        pattern = rf"^[\u4E00-\u9FA5][A-Z][A-Z0-9]{{4,5}}[A-Z0-9{self.special_suffix}]$"
        return bool(re.match(pattern, text)) and text[0] in self.province_prefix

    def recognize_plate(self, image_path: str) -> Tuple[bool, Optional[str], str]:
        """
        识别图片中的车牌号
        
        返回:
            tuple: (是否成功, 车牌号, 错误信息)
        """
        try:
            # 读取图片
            image = cv2.imread(image_path)
            if image is None:
                return False, None, f"无法读取图片: {image_path}"
                
            # 使用YOLO分割模型获取车辆区域
            try:
                vehicle_crop = self.extract_vehicle_mask_crop(image_path)
            except Exception as e:
                print(f"车辆分割失败: {str(e)}")
                # 如果分割失败，使用原图进行识别
                vehicle_crop = image

            # 在车辆区域上进一步裁剪车头区域供 OCR 使用
            try:
                head_crop = self._crop_head_from_vehicle_bgr(vehicle_crop)
            except Exception as e:
                print(f"车头裁剪失败: {str(e)}")
                head_crop = vehicle_crop

            # 转换颜色空间供PaddleOCR使用（在车头区域上做 OCR）
            ocr_input = cv2.cvtColor(head_crop, cv2.COLOR_BGR2RGB)
            
            # 使用PaddleOCR进行车牌识别
            result = self.ocr.predict(input=ocr_input)
            
            # 解析识别结果
            if result is not None and len(result) > 0:
                # 获取所有识别到的文本
                texts = [line["rec_texts"] for line in result][0]
                
                # 查找符合车牌格式的文本
                for text in texts:
                    # 车牌常见分隔符（如中点"·"）需去除后再匹配
                    raw = str(text).strip().upper()
                    t = re.sub(r"[·•∙.]", "", raw)
                    # 允许末位出现特殊标识
                    if re.match(rf"^[\u4E00-\u9FA5][A-Z][A-Z0-9]{{4,5}}[A-Z0-9{self.special_suffix}]$", t):
                        if t[0] in self.province_prefix:  # 第一位必须是省份简称汉字
                            return True, t, ""
            
            return False, None, "未找到符合格式的车牌号"
            
        except Exception as e:
            import traceback
            print(f"识别过程中出错: {str(e)}\n{traceback.format_exc()}")
            return False, None, f"识别过程中出错: {str(e)}"

    def _crop_head_from_vehicle_bgr(self, vehicle_bgr: np.ndarray) -> np.ndarray:
        """在车辆 BGR 图上，使用 head/tail YOLO 模型进一步裁出车头区域。

        如果未检测到车头或裁剪失败，则回退为输入的整车图。
        """
        if vehicle_bgr is None or vehicle_bgr.size == 0:
            return vehicle_bgr

        results = self.headtail_model(vehicle_bgr, conf=0.25, verbose=False)
        if not results:
            return vehicle_bgr

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return vehicle_bgr

        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()

        # 选择类别为0（车头）的检测框，按置信度最高选一框
        best_idx = None
        best_score = -1.0
        for i, (cls_id, score) in enumerate(zip(classes, scores)):
            if int(cls_id) != 0:
                continue
            if float(score) > best_score:
                best_score = float(score)
                best_idx = i

        if best_idx is None:
            return vehicle_bgr

        x1, y1, x2, y2 = boxes[int(best_idx)]
        h, w = vehicle_bgr.shape[:2]
        x1 = max(0, min(int(x1), w - 1))
        x2 = max(0, min(int(x2), w))
        y1 = max(0, min(int(y1), h - 1))
        y2 = max(0, min(int(y2), h))

        if x2 <= x1 or y2 <= y1:
            return vehicle_bgr

        head_bgr = vehicle_bgr[y1:y2, x1:x2].copy()
        if head_bgr.size == 0:
            return vehicle_bgr
        return head_bgr


class PlateVerificationWorker(QThread):
    """用于在后台线程中执行车牌验证"""
    progress_updated = Signal(int, int)  # current, total
    verification_complete = Signal(list)  # list of valid pairs
    
    def __init__(self, pairs: list, recognizer: PlateRecognizer):
        super().__init__()
        self.pairs = pairs
        self.recognizer = recognizer
        self.is_running = True
    
    def run(self):
        valid_pairs = []
        total = len(self.pairs)
        
        for i, pair in enumerate(self.pairs):
            if not self.is_running:
                return
                
            self.progress_updated.emit(i + 1, total)
            
            try:
                # 识别两张图片的车牌
                success1, plate1, _ = self.recognizer.recognize_plate(pair['curr_path'])
                success2, plate2, _ = self.recognizer.recognize_plate(pair['prev_path'])
                
                # 打印对比结果
                print(f"\n对比结果 - 图片对 {i+1}/{total}:")
                print(f"当前图片: {pair['curr_path']}")
                print(f"  识别结果: {plate1 if success1 else '识别失败'}")
                print(f"历史图片: {pair['prev_path']}")
                print(f"  识别结果: {plate2 if success2 else '识别失败'}")
                
                # 如果两张图片都识别成功且车牌号不同，则移除该记录
                if success1 and success2 and plate1 != plate2:
                    print(f"❌ 车牌不匹配: '{plate1}' 和 '{plate2}'，移除该记录")
                else:
                    # 其他所有情况都保留记录
                    reason = ""
                    if not success1 or not success2:
                        reason = "至少有一张图片识别失败"
                    else:
                        reason = f"车牌号相同: '{plate1}'",
                    print(f"✅ {reason}，保留该记录")
                    valid_pairs.append(pair)
                    
            except Exception as e:
                print(f"验证过程中出错: {e}")
                # 如果出现错误，默认保留该记录以避免误删
                valid_pairs.append(pair)
        
        self.verification_complete.emit(valid_pairs)
    
    def stop(self):
        self.is_running = False


def connect_to_oracle():
    try:
        os.environ["PATH"] = r"D:\\instantclient-basic-windows.x64-23.26.0.0.0\\instantclient_23_0" + ";" + os.environ.get("PATH", "")
        os.environ["TNS_ADMIN"] = r"D:\\instantclient-basic-windows.x64-23.26.0.0.0\\instantclient_23_0\\network\\admin"

        dsn_tns = cx_Oracle.makedsn(
            '10.100.2.229',
            '1521',
            service_name='JLYXZ'
        )

        connection = cx_Oracle.connect(
            user='identify',
            password='123456',
            dsn=dsn_tns
        )
        print("成功连接到Oracle数据库")
        return connection
    except Exception as e:
        print(f"连接数据库时出错: {e}")
        return None


def read_pic_matchtask_by_gross_time(connection):
    try:
        query = """
        SELECT *
        FROM jlyxz.PIC_MATCHTASK
        WHERE GROSS_WEIGH_TIME IS NOT NULL
        ORDER BY GROSS_WEIGH_TIME ASC
        """
        df = pd.read_sql(query, connection)
        return df
    except Exception as e:
        print(f"读取PIC_MATCHTASK表数据时出错: {e}")
        return None


class CarPlateRecognitionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = Siamese(model_path=r"D:\project\data_chuli\demo\demo\Siamese-pytorch-master\logs\head\1211\best_epoch_weights.pth")
        self.cropper = VehicleCropper()
        self.plate_recognizer = PlateRecognizer()
        # 车头/车尾检测模型，用于从整车裁切图中再裁出车头区域
        self.headtail_model = YOLO(r"D:\data2\runs\detect\train\weights\best.pt")
        self.image1_path = None
        self.image2_path = None
        self.suspicious_pairs = []
        self.current_pair_index = -1
        self.verification_worker = None
        self.progress_dialog = None
        self.init_ui()
    
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle('套牌车识别系统')
        self.setGeometry(100, 100, 800, 600)
        
        # 创建中心窗口
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 标题和按钮行
        header_layout = QHBoxLayout()
        
        # 标题
        title_label = QLabel('套牌车识别系统')
        title_label.setStyleSheet('font-size: 24px; font-weight: bold; color: #2c3e50;')
        title_label.setAlignment(Qt.AlignCenter)
        
        # 添加一个水平伸缩项，使标题居中
        header_layout.addStretch()
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        # 添加复核按钮
        self.verify_btn = QPushButton('复核车牌')
        self.verify_btn.setStyleSheet('''
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        ''')
        self.verify_btn.clicked.connect(self.verify_plates)
        self.verify_btn.setEnabled(False)  # 默认禁用，当有疑似记录时启用
        
        # 将复核按钮添加到标题行
        header_layout.addWidget(self.verify_btn)
        
        # 将标题行添加到主布局
        main_layout.addLayout(header_layout)
        
        # 图片选择区域
        images_layout = QHBoxLayout()
        
        # 左侧图片区域
        left_layout = QVBoxLayout()
        self.select_image1_btn = QPushButton('选择第一张图片')
        self.select_image1_btn.clicked.connect(self.select_image1)
        self.select_image1_btn.setStyleSheet('''
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        ''')
        left_layout.addWidget(self.select_image1_btn)
        
        self.image1_label = QLabel('未选择图片')
        self.image1_label.setAlignment(Qt.AlignCenter)
        self.image1_label.setMinimumSize(300, 200)
        self.image1_label.setStyleSheet('border: 2px dashed #bdc3c7; background-color: #ecf0f1;')
        self.image1_label.setCursor(Qt.PointingHandCursor)
        left_layout.addWidget(self.image1_label)
        
        # 右侧图片区域
        right_layout = QVBoxLayout()
        self.select_image2_btn = QPushButton('选择第二张图片')
        self.select_image2_btn.clicked.connect(self.select_image2)
        self.select_image2_btn.setStyleSheet('''
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        ''')
        right_layout.addWidget(self.select_image2_btn)
        
        self.image2_label = QLabel('未选择图片')
        self.image2_label.setAlignment(Qt.AlignCenter)
        self.image2_label.setMinimumSize(300, 200)
        self.image2_label.setStyleSheet('border: 2px dashed #bdc3c7; background-color: #ecf0f1;')
        self.image2_label.setCursor(Qt.PointingHandCursor)
        right_layout.addWidget(self.image2_label)
        
        images_layout.addLayout(left_layout)
        images_layout.addLayout(right_layout)
        main_layout.addLayout(images_layout)

        # 点击图片使用系统默认看图工具打开原图
        self.image1_label.mousePressEvent = lambda event: self.open_in_system_viewer(self.image1_path)
        self.image2_label.mousePressEvent = lambda event: self.open_in_system_viewer(self.image2_path)
        
        # 识别按钮
        self.predict_btn = QPushButton('开始识别')
        self.predict_btn.clicked.connect(self.predict_similarity)
        self.predict_btn.setStyleSheet('''
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                padding: 15px;
                border-radius: 5px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #229954;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        ''')
        self.predict_btn.setEnabled(False)
        main_layout.addWidget(self.predict_btn)
        
        # 结果显示区域
        result_layout = QVBoxLayout()
        result_label = QLabel('识别结果：')
        result_label.setStyleSheet('font-size: 16px; font-weight: bold;')
        result_layout.addWidget(result_label)
        
        self.result_text = QTextEdit()
        self.result_text.setMaximumHeight(120)
        self.result_text.setStyleSheet('''
            QTextEdit {
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
            }
        ''')
        self.result_text.setReadOnly(True)
        result_layout.addWidget(self.result_text)

        # 批量检测按钮
        self.batch_btn = QPushButton('从数据库批量检测套牌车')
        self.batch_btn.clicked.connect(self.run_batch_check_from_gui)
        self.batch_btn.setStyleSheet('''
            QPushButton {
                background-color: #8e44ad;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #7d3c98;
            }
        ''')
        result_layout.addWidget(self.batch_btn)

        # 从CSV加载历史疑似结果按钮
        self.load_csv_btn = QPushButton('从CSV加载疑似结果并预览')
        self.load_csv_btn.clicked.connect(self.load_suspicious_from_csv)
        self.load_csv_btn.setStyleSheet('''
            QPushButton {
                background-color: #16a085;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #138d75;
            }
        ''')
        result_layout.addWidget(self.load_csv_btn)

        # 疑似图片对浏览按钮
        nav_layout = QHBoxLayout()
        self.prev_pair_btn = QPushButton('上一对疑似图片')
        self.next_pair_btn = QPushButton('下一对疑似图片')
        self.prev_pair_btn.clicked.connect(self.show_prev_suspicious_pair)
        self.next_pair_btn.clicked.connect(self.show_next_suspicious_pair)
        self.prev_pair_btn.setEnabled(False)
        self.next_pair_btn.setEnabled(False)
        nav_layout.addWidget(self.prev_pair_btn)
        nav_layout.addWidget(self.next_pair_btn)
        result_layout.addLayout(nav_layout)
        
        main_layout.addLayout(result_layout)
    
    def select_image1(self):
        """选择第一张图片"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, '选择第一张图片', '', 
            '图片文件 (*.jpg *.jpeg *.png *.bmp *.gif)'
        )
        if file_path:
            self.image1_path = file_path
            self.display_image(file_path, self.image1_label)
            self.check_ready_to_predict()
    
    def select_image2(self):
        """选择第二张图片"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, '选择第二张图片', '', 
            '图片文件 (*.jpg *.jpeg *.png *.bmp *.gif)'
        )
        if file_path:
            self.image2_path = file_path
            self.display_image(file_path, self.image2_label)
            self.check_ready_to_predict()
    
    def display_image(self, image_path, label):
        """在标签中显示图片"""
        try:
            pixmap = QPixmap(image_path)
            # 缩放图片以适应标签大小
            scaled_pixmap = pixmap.scaled(
                label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            label.setPixmap(scaled_pixmap)
        except Exception as e:
            QMessageBox.warning(self, '错误', f'无法加载图片: {str(e)}')
    
    def check_ready_to_predict(self):
        """检查是否可以进行识别"""
        if self.image1_path and self.image2_path:
            self.predict_btn.setEnabled(True)
        else:
            self.predict_btn.setEnabled(False)
    
    def predict_similarity(self):
        """进行图片相似度识别"""
        try:
            # 加载图片
            image1 = Image.open(self.image1_path)
            image2 = Image.open(self.image2_path)
            
            # 先进行整车裁切预处理
            image1 = self.cropper.process_pil(image1)
            image2 = self.cropper.process_pil(image2)

            # 在整车裁切结果上进一步裁切车头区域
            image1 = self._crop_head_from_vehicle_pil(image1)
            image2 = self._crop_head_from_vehicle_pil(image2)

            # 在车头图上对车牌区域进行涂黑处理
            image1 = self._mask_plate_region(image1)
            image2 = self._mask_plate_region(image2)

            # 进行识别
            probability = self.model.detect_image(image1, image2)
            # 将返回的 Tensor 转为 float，避免格式化时报错
            probability = probability.item() if hasattr(probability, 'item') else float(probability)
            
            # 显示结果
            if probability > 0.3:
                result = f'✓ 为同一辆车\n相似度概率: {probability:.4f}'
                self.result_text.setStyleSheet('''
                    QTextEdit {
                        border: 2px solid #27ae60;
                        border-radius: 5px;
                        padding: 10px;
                        font-size: 14px;
                        background-color: #d5f4e6;
                    }
                ''')
            else:
                result = f'⚠ 疑似为套牌车辆\n相似度概率: {probability:.4f}'
                self.result_text.setStyleSheet('''
                    QTextEdit {
                        border: 2px solid #e74c3c;
                        border-radius: 5px;
                        padding: 10px;
                        font-size: 14px;
                        background-color: #fadbd8;
                    }
                ''')
            
            self.result_text.setText(result)
            
        except Exception as e:
            QMessageBox.critical(self, '错误', f'识别过程中出现错误: {str(e)}')

    def show_original_image(self, image_path):
        """弹出对话框显示原图（目前未在点击中使用，保留备用）"""
        if not image_path or not isinstance(image_path, str):
            return
        path = image_path.strip()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, '提示', '原图文件不存在或路径为空。')
            return

        try:
            dialog = QDialog(self)
            dialog.setWindowTitle(os.path.basename(path))
            dialog.resize(800, 600)

            scroll = QScrollArea(dialog)
            scroll.setWidgetResizable(True)

            img_label = QLabel()
            pix = QPixmap(path)
            img_label.setPixmap(pix)
            img_label.setAlignment(Qt.AlignCenter)

            scroll.setWidget(img_label)

            layout = QVBoxLayout(dialog)
            layout.addWidget(scroll)

            dialog.setLayout(layout)
            dialog.exec()
        except Exception as e:
            QMessageBox.warning(self, '错误', f'打开原图时出错: {str(e)}')

    def open_in_system_viewer(self, image_path):
        """使用系统默认看图工具打开图片文件"""
        if not image_path or not isinstance(image_path, str):
            return
        path = image_path.strip()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, '提示', '图片文件不存在或路径为空。')
            return
        try:
            os.startfile(path)
        except Exception as e:
            QMessageBox.warning(self, '错误', f'调用系统查看图片失败: {str(e)}')

    def _compare_two_images(self, path1, path2):
        """用于批量检测的单次图片对比，返回相似度或None"""
        if not path1 or not path2:
            return None
        if not isinstance(path1, str) or not isinstance(path2, str):
            return None
        path1 = path1.strip()
        path2 = path2.strip()
        if not path1 or not path2:
            return None
        if (not os.path.exists(path1)) or (not os.path.exists(path2)):
            return None
        try:
            img1 = Image.open(path1)
            img2 = Image.open(path2)
            img1 = self.cropper.process_pil(img1)
            img2 = self.cropper.process_pil(img2)

            # 在整车裁切结果上进一步裁切车头区域
            img1 = self._crop_head_from_vehicle_pil(img1)
            img2 = self._crop_head_from_vehicle_pil(img2)

            # 在车头图上对车牌区域进行涂黑处理
            img1 = self._mask_plate_region(img1)
            img2 = self._mask_plate_region(img2)
            prob = self.model.detect_image(img1, img2)
            prob = prob.item() if hasattr(prob, 'item') else float(prob)
            return prob
        except Exception as e:
            print(f"批量比对时加载或识别图片出错: {e}")
            return None

    def _crop_head_from_vehicle_pil(self, vehicle_image: Image.Image) -> Image.Image:
        """在已经裁好的整车 PIL 图上，使用 head/tail YOLO 模型进一步裁出车头区域。

        如果未检测到车头或裁剪失败，则回退为输入的整车图，保证流程不中断。
        """
        try:
            if vehicle_image is None:
                return vehicle_image

            # PIL -> BGR
            rgb = vehicle_image.convert("RGB")
            img_np = np.array(rgb)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            results = self.headtail_model(img_bgr, conf=0.25, verbose=False)
            if not results:
                return vehicle_image

            r = results[0]
            if r.boxes is None or len(r.boxes) == 0:
                return vehicle_image

            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()

            # 只选择类别为0（车头）的检测框，按置信度最高选一框
            best_idx = None
            best_score = -1.0
            for i, (cls_id, score) in enumerate(zip(classes, scores)):
                if int(cls_id) != 0:
                    continue
                if float(score) > best_score:
                    best_score = float(score)
                    best_idx = i

            if best_idx is None:
                return vehicle_image

            x1, y1, x2, y2 = boxes[int(best_idx)]
            h, w = img_bgr.shape[:2]
            x1 = max(0, min(int(x1), w - 1))
            x2 = max(0, min(int(x2), w))
            y1 = max(0, min(int(y1), h - 1))
            y2 = max(0, min(int(y2), h))

            if x2 <= x1 or y2 <= y1:
                return vehicle_image

            head_bgr = img_bgr[y1:y2, x1:x2].copy()
            if head_bgr.size == 0:
                return vehicle_image

            head_rgb = cv2.cvtColor(head_bgr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(head_rgb)
        except Exception:
            # 任意异常直接回退整车图，避免影响上层逻辑
            return vehicle_image

    def _mask_plate_region(self, head_image: Image.Image) -> Image.Image:
        """在给 Siamese 对比前，对车头图中的车牌区域进行涂黑处理。

        使用 PlateRecognizer 的 PaddleOCR 检测车牌文本框，找到格式合法的车牌后，
        在对应区域画一个黑色矩形。检测失败则返回原图。
        """
        try:
            if head_image is None:
                return head_image

            # PIL -> RGB -> BGR (PaddleOCR 习惯使用 BGR)
            rgb = head_image.convert("RGB")
            img_np = np.array(rgb)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # 使用现有 PlateRecognizer 的 OCR 进行检测
            ocr = self.plate_recognizer.ocr
            result = ocr.ocr(img_bgr, cls=False)

            if not result or not result[0]:
                return head_image

            from math import inf
            best_box = None
            # 遍历所有识别结果，找到第一个/最佳合法车牌
            for line in result[0]:
                box = line[0]
                text = line[1][0]
                conf = float(line[1][1]) if line[1][1] is not None else 0.0

                # 先做一个简单置信度过滤
                if conf < 0.7:
                    continue

                # 复用 PlateRecognizer 的车牌格式校验逻辑
                if not self.plate_recognizer.is_valid_plate(text):
                    continue

                best_box = box
                break

            if best_box is None:
                return head_image

            # best_box 为 4 个点 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            xs = [p[0] for p in best_box]
            ys = [p[1] for p in best_box]
            x1, x2 = int(max(0, min(xs))), int(max(xs))
            y1, y2 = int(max(0, min(ys))), int(max(ys))

            h, w = img_bgr.shape[:2]
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h))

            if x2 <= x1 or y2 <= y1:
                return head_image

            # 在车牌区域画黑色矩形实现打码
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)

            # 转回 PIL
            masked_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(masked_rgb)
        except Exception:
            # 任意异常直接返回原始车头图，避免影响主流程
            return head_image

    def update_verify_button_state(self):
        """根据是否有疑似记录更新复核按钮状态"""
        has_pairs = len(self.suspicious_pairs) > 0
        self.verify_btn.setEnabled(has_pairs)
        
        # 更新按钮文本
        if has_pairs:
            self.verify_btn.setText(f'复核车牌 ({len(self.suspicious_pairs)}条)')
        else:
            self.verify_btn.setText('复核车牌')
            
    def run_batch_check_from_gui(self):
        """从GUI触发：读取数据库，批量检测套牌车并导出CSV"""
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            '选择疑似套牌车结果CSV保存路径',
            f"suspected_fake_plate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            'CSV 文件 (*.csv)'
        )
        if not output_path:
            return

        self.result_text.setText('开始从数据库读取数据并批量检测，请稍候...')
        QApplication.processEvents()

        conn = connect_to_oracle()
        if conn is None:
            QMessageBox.critical(self, '错误', '无法连接到Oracle数据库，请检查配置。')
            return

        try:
            df = read_pic_matchtask_by_gross_time(conn)
        finally:
            conn.close()

        if df is None or df.empty:
            QMessageBox.information(self, '结果', '未从数据库中读取到任何PIC_MATCHTASK数据。')
            self.result_text.setText('数据库中没有可用的数据。')
            return

        required_cols = ['TASK_ID', 'TRUCK_ID', 'GROSS_WEIGH_TIME', 'TARE_IMAGE_PATH1', 'GROSS_IMAGE_PATH1']
        for col in required_cols:
            if col not in df.columns:
                QMessageBox.critical(self, '错误', f'数据表缺少必要字段: {col}')
                self.result_text.setText(f'数据表缺少必要字段: {col}')
                return

        df = df[df['GROSS_WEIGH_TIME'].notna()].copy()
        if df.empty:
            QMessageBox.information(self, '结果', '没有包含GROSS_WEIGH_TIME的数据记录。')
            self.result_text.setText('没有包含GROSS_WEIGH_TIME的数据记录。')
            return

        df = df.sort_values('GROSS_WEIGH_TIME')

        last_record_by_plate = {}
        suspicious_rows = []
        self.suspicious_pairs = []
        self.current_pair_index = -1

        for idx, row in df.iterrows():
            try:
                current_task_id = row['TASK_ID']
            except Exception:
                current_task_id = None

            if current_task_id is not None:
                self.result_text.setText(f'当前处理 TASK_ID: {current_task_id}')
                QApplication.processEvents()

            plate = row['TRUCK_ID']
            if plate is None:
                continue
            plate_str = str(plate).strip()
            if not plate_str:
                continue

            prev_row = last_record_by_plate.get(plate_str)
            if prev_row is not None:
                suspicious = False

                curr_tare = row.get('TARE_IMAGE_PATH1')
                prev_tare = prev_row.get('TARE_IMAGE_PATH1')
                tare_prob = self._compare_two_images(curr_tare, prev_tare)
                if tare_prob is not None and tare_prob <= 0.3:
                    suspicious = True
                    self.suspicious_pairs.append({
                        'type': 'tare',
                        'task_id': row['TASK_ID'],
                        'prev_task_id': prev_row.get('TASK_ID'),
                        'truck_id': plate_str,
                        'curr_path': curr_tare,
                        'prev_path': prev_tare,
                        'probability': tare_prob,
                    })

                curr_gross = row.get('GROSS_IMAGE_PATH1')
                prev_gross = prev_row.get('GROSS_IMAGE_PATH1')
                gross_prob = self._compare_two_images(curr_gross, prev_gross)
                if gross_prob is not None and gross_prob <= 0.3:
                    suspicious = True
                    self.suspicious_pairs.append({
                        'type': 'gross',
                        'task_id': row['TASK_ID'],
                        'prev_task_id': prev_row.get('TASK_ID'),
                        'truck_id': plate_str,
                        'curr_path': curr_gross,
                        'prev_path': prev_gross,
                        'probability': gross_prob,
                    })

                if suspicious:
                    suspicious_rows.append(row)

            last_record_by_plate[plate_str] = row

        if suspicious_rows:
            # 只导出任务号和两张图片路径（当前图片路径、历史图片路径）
            export_rows = []
            for pair in self.suspicious_pairs:
                export_rows.append({
                    'TASK_ID': pair.get('task_id'),
                    'CURR_IMAGE_PATH': pair.get('curr_path'),
                    'PREV_IMAGE_PATH': pair.get('prev_path'),
                })

            out_df = pd.DataFrame(export_rows)
            try:
                out_df.to_csv(output_path, index=False, encoding='utf-8-sig')
                msg = f'检测完成，共发现 {len(suspicious_rows)} 条疑似套牌车记录，疑似图片对 {len(self.suspicious_pairs)} 对，已保存到:\n{output_path}'
                QMessageBox.information(self, '检测完成', msg)
                self.result_text.setText(msg)

                if self.suspicious_pairs:
                    self.current_pair_index = 0
                    self.prev_pair_btn.setEnabled(True)
                    self.next_pair_btn.setEnabled(True)
                    self.show_current_suspicious_pair()
                self.update_verify_button_state()  # 更新按钮状态
            except Exception as e:
                QMessageBox.critical(self, '错误', f'保存CSV文件时出错: {e}')
                self.result_text.setText(f'保存CSV文件时出错: {e}')
        else:
            QMessageBox.information(self, '检测完成', '检测完成，未发现疑似套牌车记录。')
            self.result_text.setText('检测完成，未发现疑似套牌车记录。')

    def show_current_suspicious_pair(self):
        """根据 current_pair_index 在左右图片区域显示一对疑似图片"""
        if not self.suspicious_pairs or self.current_pair_index < 0:
            return
        pair = self.suspicious_pairs[self.current_pair_index]
        curr_path = pair.get('curr_path')
        prev_path = pair.get('prev_path')

        if curr_path and os.path.exists(curr_path):
            self.image1_path = curr_path
            self.display_image(curr_path, self.image1_label)
        if prev_path and os.path.exists(prev_path):
            self.image2_path = prev_path
            self.display_image(prev_path, self.image2_label)

        info = (
            f"当前第 {self.current_pair_index + 1} 对 / 共 {len(self.suspicious_pairs)} 对\n"
            f"类型: {'皮重' if pair.get('type') == 'tare' else '毛重'}\n"
            f"车牌: {pair.get('truck_id')}\n"
            f"当前TASK_ID: {pair.get('task_id')}, 对比TASK_ID: {pair.get('prev_task_id')}\n"
            f"相似度概率: {pair.get('probability'):.4f}"
        )
        self.result_text.setText(info)

    def show_prev_suspicious_pair(self):
        """切换到上一对疑似图片"""
        if not self.suspicious_pairs:
            return
        self.current_pair_index = (self.current_pair_index - 1) % len(self.suspicious_pairs)
        self.show_current_suspicious_pair()

    def show_next_suspicious_pair(self):
        """切换到下一对疑似图片"""
        if not self.suspicious_pairs:
            return
        self.current_pair_index = (self.current_pair_index + 1) % len(self.suspicious_pairs)
        self.show_current_suspicious_pair()

    def verify_plates(self):
        """开始车牌复核过程"""
        if not self.suspicious_pairs:
            QMessageBox.information(self, '提示', '没有需要复核的记录')
            return
            
        # 创建进度对话框
        self.progress_dialog = QProgressDialog("正在复核车牌...", "取消", 0, len(self.suspicious_pairs), self)
        self.progress_dialog.setWindowTitle("复核进度")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.canceled.connect(self.cancel_verification)
        
        # 创建并启动工作线程
        self.verification_worker = PlateVerificationWorker(self.suspicious_pairs, self.plate_recognizer)
        self.verification_worker.progress_updated.connect(self.update_verification_progress)
        self.verification_worker.verification_complete.connect(self.verification_completed)
        self.verification_worker.finished.connect(self.verification_worker.deleteLater)
        
        self.verification_worker.start()
        self.progress_dialog.show()
    
    def cancel_verification(self):
        """取消车牌复核"""
        if self.verification_worker and self.verification_worker.isRunning():
            self.verification_worker.stop()
            self.verification_worker.wait()
        if self.progress_dialog:
            self.progress_dialog.close()
    
    def update_verification_progress(self, current: int, total: int):
        """更新复核进度"""
        if self.progress_dialog:
            self.progress_dialog.setMaximum(total)
            self.progress_dialog.setValue(current)
    
    def verification_completed(self, valid_pairs: list):
        """车牌复核完成处理"""
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
        
        removed_count = len(self.suspicious_pairs) - len(valid_pairs)
        self.suspicious_pairs = valid_pairs
        self.current_pair_index = -1 if not self.suspicious_pairs else 0
        
        # 更新显示
        if self.suspicious_pairs:
            self.show_current_suspicious_pair()
        else:
            # 清空显示
            self.image1_label.clear()
            self.image2_label.clear()
            self.result_text.clear()
        
        # 更新按钮状态
        self.update_verify_button_state()
        
        QMessageBox.information(self, '复核完成', 
            f'复核完成，移除了{removed_count}条记录，剩余{len(self.suspicious_pairs)}条记录')
    
    def load_suspicious_from_csv(self):
        """从历史CSV文件加载疑似记录，并基于当前DB数据重建图片对进行预览"""
        csv_path, _ = QFileDialog.getOpenFileName(
            self,
            '选择疑似套牌车CSV文件',
            '',
            'CSV 文件 (*.csv)'
        )
        if not csv_path:
            return

        try:
            csv_df = pd.read_csv(csv_path)
        except Exception as e:
            QMessageBox.critical(self, '错误', f'读取CSV文件失败: {e}')
            return

        if 'TASK_ID' not in csv_df.columns or 'TRUCK_ID' not in csv_df.columns:
            QMessageBox.critical(self, '错误', 'CSV 文件中缺少 TASK_ID 或 TRUCK_ID 列。')
            return

        suspicious_task_ids = set(csv_df['TASK_ID'].dropna().tolist())
        if not suspicious_task_ids:
            QMessageBox.information(self, '提示', 'CSV 中没有有效的 TASK_ID。')
            return

        self.result_text.setText('正在根据CSV中的TASK_ID从数据库重建疑似图片对，请稍候...')
        QApplication.processEvents()

        conn = connect_to_oracle()
        if conn is None:
            QMessageBox.critical(self, '错误', '无法连接到Oracle数据库，请检查配置。')
            return

        try:
            df = read_pic_matchtask_by_gross_time(conn)
        finally:
            conn.close()

        if df is None or df.empty:
            QMessageBox.information(self, '结果', '未从数据库中读取到任何PIC_MATCHTASK数据。')
            self.result_text.setText('数据库中没有可用的数据。')
            return

        required_cols = ['TASK_ID', 'TRUCK_ID', 'GROSS_WEIGH_TIME', 'TARE_IMAGE_PATH1', 'GROSS_IMAGE_PATH1']
        for col in required_cols:
            if col not in df.columns:
                QMessageBox.critical(self, '错误', f'数据库表缺少必要字段: {col}')
                self.result_text.setText(f'数据库表缺少必要字段: {col}')
                return

        df = df[df['GROSS_WEIGH_TIME'].notna()].copy()
        if df.empty:
            QMessageBox.information(self, '结果', '没有包含GROSS_WEIGH_TIME的数据记录。')
            self.result_text.setText('没有包含GROSS_WEIGH_TIME的数据记录。')
            return

        df = df.sort_values('GROSS_WEIGH_TIME')

        last_record_by_plate = {}
        self.suspicious_pairs = []
        self.current_pair_index = -1

        for idx, row in df.iterrows():
            task_id = row['TASK_ID']
            plate = row['TRUCK_ID']
            if plate is None or task_id is None:
                continue
            plate_str = str(plate).strip()
            if not plate_str:
                continue

            prev_row = last_record_by_plate.get(plate_str)
            if prev_row is not None and task_id in suspicious_task_ids:
                curr_tare = row.get('TARE_IMAGE_PATH1')
                prev_tare = prev_row.get('TARE_IMAGE_PATH1')
                if curr_tare and prev_tare:
                    self.suspicious_pairs.append({
                        'type': 'tare',
                        'task_id': task_id,
                        'prev_task_id': prev_row.get('TASK_ID'),
                        'truck_id': plate_str,
                        'curr_path': curr_tare,
                        'prev_path': prev_tare,
                        'probability': 0.0,
                    })

                curr_gross = row.get('GROSS_IMAGE_PATH1')
                prev_gross = prev_row.get('GROSS_IMAGE_PATH1')
                if curr_gross and prev_gross:
                    self.suspicious_pairs.append({
                        'type': 'gross',
                        'task_id': task_id,
                        'prev_task_id': prev_row.get('TASK_ID'),
                        'truck_id': plate_str,
                        'curr_path': curr_gross,
                        'prev_path': prev_gross,
                        'probability': 0.0,
                    })

            last_record_by_plate[plate_str] = row

        if not self.suspicious_pairs:
            QMessageBox.information(self, '提示', '未找到任何有效的疑似图片对。')
            self.result_text.setText('未找到任何有效的疑似图片对。')
            self.update_verify_button_state()  # 更新按钮状态
            return
            
        # 更新按钮状态并显示第一对图片
        self.update_verify_button_state()
        self.current_pair_index = 0
        self.show_current_suspicious_pair()

        self.current_pair_index = 0
        self.prev_pair_btn.setEnabled(True)
        self.next_pair_btn.setEnabled(True)
        self.show_current_suspicious_pair()


def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用程序样式
    app.setStyleSheet('''
        QMainWindow {
            background-color: #f8f9fa;
        }
    ''')
    
    window = CarPlateRecognitionGUI()
    window.show()
    
    sys.exit(app.exec())

 
if __name__ == '__main__':
    main()
