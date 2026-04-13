"""
套牌车识别GUI界面
用途：实现一个图形用户界面，用于选择两张图片并进行套牌车识别
功能：
1. 提供两个图片选择按钮，让用户选择要比较的两张车辆图片
2. 显示选中的图片预览
3. 调用Siamese模型进行图片相似度检测
4. 显示识别结果（是否为同一辆车或疑似套牌车辆）
技术栈：PySide6用于GUI设计，PIL用于图片处理
"""

import sys
import os
import cv2
import re
import numpy as np
from datetime import datetime
import cx_Oracle
import pandas as pd
from PIL import Image, ImageOps
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                               QMessageBox, QTextEdit, QDialog, QScrollArea)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage
from paddleocr import PaddleOCR

from siamese import Siamese
parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from data_chuli.cropper import VehicleCropper


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


class PlateRecognizer:
    """车牌识别类，封装PaddleOCR的使用"""
    def __init__(self):
        self.ocr = PaddleOCR(

        )
        
        # 常见省份简称集合
        self.province_prefix = set(list("京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼港澳"))
        self.special_suffix = "挂警学领港澳"
    
    def detect_plate_text(self, image_path):
        """识别车牌文本"""
        try:
            # 使用PIL读取图片并转换为numpy数组
            image = Image.open(image_path).convert('RGB')
            
            # 将图片调整为合适大小，提高识别准确率
            image = self._preprocess_image(image)
            image_np = np.array(image)
            
            # 将RGB转换为BGR（PaddleOCR需要BGR格式）
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            # 使用PaddleOCR进行车牌识别
            result = self.ocr.ocr(image_bgr, cls=False)
            
            if not result or not result[0]:
                return ""

            for line in result[0]:
                text = line[1][0]  # 获取识别文本
                confidence = line[1][1]  # 获取置信度
                
                # 只处理置信度大于0.7的结果
                if confidence < 0.7:  
                    continue
                    
                # 清理和验证车牌格式
                raw = str(text).strip().upper()
                cleaned = re.sub(r"[·•∙. ]", "", raw)
                
                # 匹配标准车牌格式：1位汉字 + 1位字母 + 5位字母数字 + 可选1位特殊字符
                if re.match(rf"^[\u4E00-\u9FA5][A-Z][A-Z0-9]{{4,5}}[A-Z0-9{self.special_suffix}]?$", cleaned):
                    if cleaned[0] in self.province_prefix:
                        return cleaned
            
            return ""
            
        except Exception as e:
            print(f"Error detecting plate: {e}")
            return ""
    
    def _preprocess_image(self, image, target_height=64):
        """预处理图片，调整大小并增强对比度"""
        # 调整大小，保持宽高比
        width, height = image.size
        new_width = int(width * (target_height / height))
        image = image.resize((new_width, target_height), Image.LANCZOS)
        
        # 转换为灰度图
        gray = image.convert('L')
        
        # 直方图均衡化
        equalized = ImageOps.equalize(gray)
        
        # 转回RGB（PaddleOCR需要3通道）
        return equalized.convert('RGB')


class CarPlateRecognitionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = Siamese()
        self.cropper = VehicleCropper()
        self.plate_recognizer = PlateRecognizer()
        self.image1_path = None
        self.image2_path = None
        self.suspicious_pairs = []
        self.current_pair_index = -1
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
        
        # 标题
        title_label = QLabel('套牌车识别系统')
        title_label.setStyleSheet('font-size: 24px; font-weight: bold; color: #2c3e50;')
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
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
        """进行图片相似度识别并检查车牌是否一致"""
        if not (self.image1_path and self.image2_path):
            QMessageBox.warning(self, '警告', '请先选择两张图片')
            return
            
        # 清空之前的结果
        self.result_text.clear()
        self.statusBar().showMessage('正在处理，请稍候...')
        QApplication.processEvents()
        
        try:
            result_info = []
            
            # 1. 先进行车辆相似度分析
            result_info.append('=== 车辆相似度分析 ===')
            
            # 加载并处理图片
            self.result_text.append('正在加载并处理图片...')
            QApplication.processEvents()
            
            try:
                # 加载图片
                image1 = Image.open(self.image1_path)
                image2 = Image.open(self.image2_path)
                
                # 裁切预处理
                image1 = self.cropper.process_pil(image1)
                image2 = self.cropper.process_pil(image2)
                
                # 计算相似度
                self.result_text.append('正在计算车辆相似度...')
                QApplication.processEvents()
                
                probability = self.model.detect_image(image1, image2)
                probability = probability.item() if hasattr(probability, 'item') else float(probability)
                
                # 显示相似度结果
                if probability > 0.3:
                    # 相似度高，直接判定为同一辆车，不需要进行车牌比对，也不显示在结果中
                    self.result_text.setPlainText('✅ 两车为同一辆车，无需进行套牌车检测')
                    self.result_text.setStyleSheet('''
                        QTextEdit {
                            border: 2px solid #27ae60;
                            border-radius: 5px;
                            padding: 10px;
                            font-size: 14px;
                            background-color: #d5f4e6;
                        }
                    ''')
                    self.statusBar().showMessage('处理完成')
                    return
                else:
                    result_info.append(f'⚠️ 车辆相似度: {probability:.2%} (相似度低，可能不是同一辆车)')
                    
                    # 相似度低，进行车牌识别和比对
                    self.result_text.append('正在识别车牌...')
                    QApplication.processEvents()
                    
                    plate1 = self.plate_recognizer.detect_plate_text(self.image1_path)
                    plate2 = self.plate_recognizer.detect_plate_text(self.image2_path)
                    
                    # 显示车牌识别结果
                    result_info.append('')
                    result_info.append('=== 车牌识别结果 ===')
                    result_info.append(f'图片1车牌: {plate1 if plate1 else "未识别到有效车牌"}')
                    result_info.append(f'图片2车牌: {plate2 if plate2 else "未识别到有效车牌"}')
                    
                    # 综合分析
                    result_info.append('')
                    result_info.append('=== 综合分析结果 ===')
                    
                    if plate1 and plate2:
                        if plate1 == plate2:
                            result_info.append('⚠️ 警告: 发现套牌车嫌疑！')
                            result_info.append('判断依据:')
                            result_info.append('1. 车辆外观相似度低')
                            result_info.append(f'2. 两车车牌号相同: {plate1}')
                            result_style = '''
                                QTextEdit {
                                    border: 2px solid #e74c3c;
                                    border-radius: 5px;
                                    padding: 10px;
                                    font-size: 14px;
                                    background-color: #fadbd8;
                                    font-weight: bold;
                                }
                            '''
                        else:
                            # 两车车牌不同，不是套牌车，不显示在结果中
                            self.result_text.setPlainText('✅ 两车不是套牌车')
                            self.result_text.setStyleSheet('''
                                QTextEdit {
                                    border: 2px solid #3498db;
                                    border-radius: 5px;
                                    padding: 10px;
                                    font-size: 14px;
                                    background-color: #e3f2fd;
                                }
                            ''')
                            self.statusBar().showMessage('处理完成')
                            return
                    else:
                        result_info.append('ℹ️ 提示: 未识别到有效车牌，无法确认是否为套牌车')
                        result_info.append('建议: 请人工检查这两张图片')
                        result_style = '''
                            QTextEdit {
                                border: 2px solid #f39c12;
                                border-radius: 5px;
                                padding: 10px;
                                font-size: 14px;
                                background-color: #fef9e7;
                            }
                        '''
                
                # 只显示套牌车嫌疑或车牌识别失败的情况
                if '发现套牌车嫌疑' in '\n'.join(result_info) or '未识别到有效车牌' in '\n'.join(result_info):
                    self.result_text.setPlainText('\n'.join(result_info))
                    self.result_text.setStyleSheet(result_style)
                else:
                    # 其他情况（如两车不同）不显示结果
                    self.result_text.clear()
                    self.statusBar().showMessage('处理完成')
                
            except Exception as e:
                self.result_text.append('\n❌ 分析过程中出现错误')
                self.result_text.append(f'错误信息: {str(e)}')
                raise
            
            self.statusBar().showMessage('处理完成')
            
        except Exception as e:
            self.statusBar().showMessage('处理出错')
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
            prob = self.model.detect_image(img1, img2)
            prob = prob.item() if hasattr(prob, 'item') else float(prob)
            return prob
        except Exception as e:
            print(f"批量比对时加载或识别图片出错: {e}")
            return None

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
            out_df = pd.DataFrame(suspicious_rows)
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
            QMessageBox.information(self, '结果', '根据该CSV和当前数据库记录，未能重建任何疑似图片对。')
            self.result_text.setText('未能重建任何疑似图片对。')
            return

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
