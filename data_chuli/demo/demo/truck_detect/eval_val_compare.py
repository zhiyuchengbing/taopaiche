# -*- coding: utf-8 -*-
"""旧模型 vs 新模型 在 split_v2 真实 val（不增强）上的 mAP 对比。

计划 Step 4.1：新 val 上 mAP50/mAP50-95 与旧 0.995 对比。
"""
from ultralytics import YOLO

OLD = r"D:\project\data_chuli\demo\demo\data_chuli\data\cheliang_detect\20260321\best.pt"
NEW = r"D:\data2\truck\split_v2\runs\truck_train08134\weights\best.pt"
DATA = r"D:\data2\truck\split_v2\data.yaml"


def main():
    for tag, path in (("OLD", OLD), ("NEW", NEW)):
        m = YOLO(path)
        r = m.val(data=DATA, imgsz=640, conf=0.001, iou=0.6, verbose=False, device="0", workers=2)
        print(f"{tag}: mAP50={r.box.map50:.4f} mAP50-95={r.box.map:.4f} P={r.box.mp:.4f} R={r.box.mr:.4f}")


if __name__ == "__main__":
    main()
