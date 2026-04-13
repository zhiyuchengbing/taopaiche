import os
import sys
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QTextEdit, QProgressBar,
    QTableWidget, QTableWidgetItem, QHeaderView
)
from PySide6.QtCore import Qt, QThread, Signal

import pandas as pd

# 便于导入上级目录模块
PARENT_DIR = os.path.dirname(os.path.dirname(__file__))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from detect_clone_plates import detect_from_csv  # type: ignore


class Worker(QThread):
    progress = Signal(int)
    finished = Signal(pd.DataFrame, str)
    failed = Signal(str)

    def __init__(self, csv_path: str, threshold: float):
        super().__init__()
        self.csv_path = csv_path
        self.threshold = threshold

    def run(self):
        try:
            df, out_path = detect_from_csv(Path(self.csv_path), threshold=self.threshold)
            self.finished.emit(df, str(out_path))
        except Exception as e:
            self.failed.emit(str(e))


class CloneCheckerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("套牌车检测批处理 - GUI")
        self.setGeometry(100, 100, 1000, 700)

        self.csv_path: str = str(Path(PARENT_DIR) / "data_chuli" / "data" / "匹配数据.csv")
        self.threshold: float = 0.3
        self.worker: Worker | None = None

        self._build_ui()

    def _build_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # CSV选择
        row1 = QHBoxLayout()
        self.csv_label = QLabel(f"CSV: {self.csv_path}")
        btn_choose_csv = QPushButton("选择CSV文件")
        btn_choose_csv.clicked.connect(self.choose_csv)
        row1.addWidget(self.csv_label)
        row1.addWidget(btn_choose_csv)
        layout.addLayout(row1)

        # 阈值与开始
        row2 = QHBoxLayout()
        self.threshold_label = QLabel(f"阈值: {self.threshold}")
        btn_start = QPushButton("开始检测")
        btn_start.clicked.connect(self.start_detection)
        row2.addWidget(self.threshold_label)
        row2.addWidget(btn_start)
        layout.addLayout(row2)

        # 进度与结果信息
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # 不确定时长，设置为忙等待
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.info = QTextEdit()
        self.info.setReadOnly(True)
        self.info.setMaximumHeight(120)
        layout.addWidget(self.info)

        # 结果表
        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels([
            "任务ID", "车号", "当前时间", "当前图片", "参考任务ID",
            "参考时间", "参考图片", "相似度", "判定"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table)

    def choose_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择CSV", str(Path(PARENT_DIR) / "data_chuli"), "CSV 文件 (*.csv)")
        if path:
            self.csv_path = path
            self.csv_label.setText(f"CSV: {self.csv_path}")

    def start_detection(self):
        self.info.clear()
        self.progress.setVisible(True)
        self.worker = Worker(self.csv_path, self.threshold)
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.worker.start()

    def on_finished(self, df: pd.DataFrame, out_path: str):
        self.progress.setVisible(False)
        self.info.append(f"完成。结果保存到: {out_path}")
        self.render_table(df)

    def on_failed(self, msg: str):
        self.progress.setVisible(False)
        QMessageBox.critical(self, "错误", msg)

    def render_table(self, df: pd.DataFrame):
        cols = ["任务ID", "车号", "当前时间", "当前图片", "参考任务ID", "参考时间", "参考图片", "相似度", "判定"]
        show_cols = [c for c in cols if c in df.columns]
        self.table.setColumnCount(len(show_cols))
        self.table.setHorizontalHeaderLabels(show_cols)
        self.table.setRowCount(len(df))
        for r, (_, row) in enumerate(df.iterrows()):
            for c, col in enumerate(show_cols):
                val = row.get(col)
                self.table.setItem(r, c, QTableWidgetItem("" if pd.isna(val) else str(val)))
        self.table.resizeRowsToContents()


def main():
    app = QApplication(sys.argv)
    win = CloneCheckerGUI()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
