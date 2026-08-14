# -*- coding: utf-8 -*-
"""硬样本召回对比：旧生产模型 vs 新训练模型，在夜间/傍晚/远距真实抓拍上对比 vehicle_detected。

- 从 17/22/53 报警抓拍目录挑"未进训练集"的夜间(18:00-06:00)样本，随机抽样 N 张。
- 两个模型 conf=0.2 预测，统计有车检出率、平均置信、平均框占比(远距代理)。
- 输出对比报告到 split_v2/eval_hard_report.txt。
"""
import os
import random
import time

import cv2
import numpy as np
from ultralytics import YOLO

OLD_MODEL = r"D:\project\data_chuli\demo\demo\data_chuli\data\cheliang_detect\20260321\best.pt"
NEW_MODEL = r"D:\data2\truck\split_v2\runs\truck_train08134\weights\best.pt"
CONF = 0.2
N = 120
SEED = 20260813
TRAIN_IMG = r"D:\data2\truck\split_v2\images\train"
VAL_IMG = r"D:\data2\truck\split_v2\images\val"
OUT_TXT = r"D:\data2\truck\split_v2\eval_hard_report.txt"

SRC_ROOTS = [
    r"D:\AlarmCaptures\17\1\2026\08",
    r"D:\AlarmCaptures\22\1\2026\08",
    r"D:\AlarmCaptures\53\1\2026\08",
    r"D:\AlarmCaptures\53\1\2026\05",
    r"D:\AlarmCaptures\17\1\2026\07",
    r"D:\AlarmCaptures\22\1\2026\07",
]


def collect_hard_images():
    """收集文件名带夜间时段(18-23, 0-6点)的抓拍，且不在训练/验证集。"""
    used = set()
    for d in (TRAIN_IMG, VAL_IMG):
        for f in os.listdir(d):
            used.add(os.path.splitext(f)[0])
    cands = []
    for root in SRC_ROOTS:
        if not os.path.isdir(root):
            continue
        for day in os.listdir(root):
            sel = os.path.join(root, day, "selected")
            if not os.path.isdir(sel):
                continue
            for f in os.listdir(sel):
                if not f.lower().endswith(".jpg"):
                    continue
                stem = os.path.splitext(f)[0]
                if stem in used:
                    continue
                # 文件名时间: {p}_1_{yyyymmdd}_{HHMMSS}_CH01_...
                try:
                    time_part = f.split("_")[3]
                    hh = int(time_part[:2])
                except Exception:
                    continue
                if hh >= 18 or hh < 6:
                    cands.append(os.path.join(sel, f))
    rng = random.Random(SEED)
    rng.shuffle(cands)
    return cands[:N], len(cands)


def run_model(model, paths):
    """返回 [(detected: bool, conf: float, area_frac: float), ...]"""
    out = []
    for p in paths:
        res = model.predict(source=p, conf=CONF, classes=[0], verbose=False)[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            out.append((False, 0.0, 0.0))
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        H, W = res.orig_shape[:2]
        # 取最大框
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        idx = int(np.argmax(areas))
        area_frac = areas[idx] / float(W * H)
        out.append((True, float(confs[idx]), float(area_frac)))
    return out


def main():
    paths, total_cand = collect_hard_images()
    lines = [f"硬样本候选总数: {total_cand} | 抽样: {len(paths)}"]
    print("\n".join(lines))
    if not paths:
        open(OUT_TXT, "w", encoding="utf-8").write("无候选硬样本\n")
        return

    print("加载旧模型...")
    old = YOLO(OLD_MODEL)
    print("加载新模型...")
    new = YOLO(NEW_MODEL)

    t0 = time.time()
    old_res = run_model(old, paths)
    new_res = run_model(new, paths)
    print(f"推理完成 用时 {time.time()-t0:.1f}s")

    def summarize(res):
        det = sum(1 for d, _, _ in res if d)
        confs = [c for _, c, _ in res if c > 0]
        areas = [a for _, _, a in res if a > 0]
        return det, (sum(confs) / len(confs) if confs else 0), (sum(areas) / len(areas) if areas else 0)

    o_det, o_conf, o_area = summarize(old_res)
    n_det, n_conf, n_area = summarize(new_res)

    lines += [
        f"=== 对比 (conf={CONF}) ===",
        f"旧模型: 检出 {o_det}/{len(paths)} ({o_det/len(paths)*100:.1f}%) | 平均置信 {o_conf:.3f} | 平均框占比 {o_area:.3f}",
        f"新模型: 检出 {n_det}/{len(paths)} ({n_det/len(paths)*100:.1f}%) | 平均置信 {n_conf:.3f} | 平均框占比 {n_area:.3f}",
        "",
        "=== 逐样本 (路径, 旧det, 旧conf, 新det, 新conf) ===",
    ]
    for p, o, n in zip(paths, old_res, new_res):
        lines.append(f"{p} | 旧:{int(o[0])},{o[1]:.3f} | 新:{int(n[0])},{n[1]:.3f}")

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"报告 -> {OUT_TXT}")
    print("\n".join(lines[1:7]))


if __name__ == "__main__":
    main()
