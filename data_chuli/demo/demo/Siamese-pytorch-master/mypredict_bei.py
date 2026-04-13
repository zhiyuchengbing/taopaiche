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
import numpy as np
from PIL import Image
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                               QMessageBox, QTextEdit)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap

from siamese import Siamese
parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from data_chuli.cropper import VehicleCropper


class CarPlateRecognitionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = Siamese()
        self.cropper = VehicleCropper()
        self.image1_path = None
        self.image2_path = None
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
        right_layout.addWidget(self.image2_label)
        
        images_layout.addLayout(left_layout)
        images_layout.addLayout(right_layout)
        main_layout.addLayout(images_layout)
        
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
        self.result_text.setMaximumHeight(100)
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
            
            # 先进行裁切预处理
            image1 = self.cropper.process_pil(image1)
            image2 = self.cropper.process_pil(image2)

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
