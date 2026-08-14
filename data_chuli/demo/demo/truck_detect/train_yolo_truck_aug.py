# -*- coding: utf-8 -*-
"""场景增强后的车辆检测重训脚本（split_v2 + 离线增强 + 在线参数微调）。

在线增强参数在 ultralytics 默认基础上适度调整：
  hsv_h 0.015->0.03, hsv_v 0.4->0.5, translate 0.1->0.2, degrees 0->2, erasing 0.4->0.2
保留 mosaic=1.0 / close_mosaic=10 / fliplr=0.5 / scale=0.5。
"""
import os
from ultralytics import YOLO

DATA_YAML = r"D:\data2\truck\split_v2\data.yaml"
WEIGHTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "yolov8s.pt")

EPOCHS = 200
IMGSZ = 640
BATCH = 16
DEVICE = "0"
WORKERS = 2  # 机器无页面文件，workers=8 会因 cuBLAS 加载触发 WinError1455，降到 2
PROJECT_DIR = r"D:\data2\truck\split_v2\runs"
RUN_NAME = "truck_train0813"


def main() -> int:
    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(f"dataset yaml not found: {DATA_YAML}")

    weights = WEIGHTS if os.path.exists(WEIGHTS) else r"D:\project\data_chuli\demo\yolov8s.pt"

    model = YOLO(weights)
    model.train(
        data=DATA_YAML,
        epochs=int(EPOCHS),
        imgsz=int(IMGSZ),
        batch=int(BATCH),
        device=str(DEVICE),
        workers=int(WORKERS),
        project=str(PROJECT_DIR),
        name=str(RUN_NAME),
        hsv_h=0.03,
        hsv_s=0.7,
        hsv_v=0.5,
        translate=0.2,
        degrees=2.0,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        close_mosaic=10,
        erasing=0.2,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
