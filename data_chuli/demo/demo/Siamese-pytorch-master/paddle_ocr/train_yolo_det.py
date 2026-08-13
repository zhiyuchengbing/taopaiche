# -*- coding: utf-8 -*-
"""
训练 YOLO 检测模型 (v2 双类): 放大号 + 车挂号。
数据: D:\data2\weibu_yolo_data_dual (510 train / 127 val, 双类分层)
GPU: RTX 5060 Ti 16GB
权重: 本地 yolo11n.pt (避免网络下载)
"""
import os
import sys
sys.stdout.reconfigure(encoding="utf-8")
from ultralytics import YOLO


def main():
    DATA_YAML = r"D:\data2\weibu_yolo_data_dual\data.yaml"
    OUT_DIR = r"D:\data2\weibu_yolo_runs"
    WEIGHT = r"D:\project\data_chuli\demo\demo\Siamese-pytorch-master\paddle_ocr\yolo11n.pt"

    print(f"[1] 权重: {WEIGHT} 存在={os.path.exists(WEIGHT)}", flush=True)
    print(f"[2] 数据yaml: {DATA_YAML} 存在={os.path.exists(DATA_YAML)}", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    model = YOLO(WEIGHT)
    print(f"[3] 模型加载完成: {type(model).__name__}", flush=True)

    model.train(
        data=DATA_YAML,
        epochs=150,
        imgsz=640,
        batch=16,
        device=0,
        workers=2,
        project=OUT_DIR,
        name="det_dual",
        patience=40,
        lr0=0.01,
        cos_lr=True,
        augment=True,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        seed=42,
        verbose=True,
    )

    print("训练完成", flush=True)
    best = os.path.join(OUT_DIR, "det_dual", "weights", "best.pt")
    print("best.pt:", best, flush=True)


if __name__ == "__main__":
    main()
