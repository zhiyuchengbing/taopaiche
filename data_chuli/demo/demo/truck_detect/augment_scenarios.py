# -*- coding: utf-8 -*-
"""离线场景数据增强：为 split_v2 训练集每图生成 K 张场景增强变体。

- 12 类场景模块（cv2/numpy/PIL），纯实现，无新依赖。
- 光度类场景框不变；几何类场景（距离远/近）框随动。
- 输出写回 split_v2/images/train + labels/train，命名 {basename}__aug{n}。
- 生成 augmented_preview/ 场景蒙太奇供人工验收。
"""
import os
import random
import time

import cv2
import numpy as np

TRAIN_IMG = r"D:\data2\truck\split_v2\images\train"
TRAIN_LAB = r"D:\data2\truck\split_v2\labels\train"
PREVIEW_DIR = r"D:\data2\truck\split_v2\augmented_preview"
K = 3                      # 每图增强变体数
SEED = 20260813
JPEG_Q = 90
LOG = r"D:\data2\truck\augment_progress.log"
PREVIEW_SOURCES = 2        # 蒙太奇取几张源图

rng = random.Random(SEED)


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------- 基础工具 ----------
def _clip_box(box, W, H):
    x1, y1, x2, y2 = box
    x1 = max(0, min(W - 1, int(round(x1))))
    y1 = max(0, min(H - 1, int(round(y1))))
    x2 = max(0, min(W, int(round(x2))))
    y2 = max(0, min(H, int(round(y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _box_to_norm(box, W, H):
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return f"0 {cx/W:.6f} {cy/H:.6f} {bw/W:.6f} {bh/H:.6f}\n"


def _read_norm_box(txt_path):
    with open(txt_path, encoding="utf-8") as f:
        line = f.readline().strip()
    if not line:
        return None
    parts = line.split()
    if len(parts) < 5:
        return None
    cx, cy, bw, bh = map(float, parts[1:5])
    return cx, cy, bw, bh


# ---------- 光度场景：框不变 ----------
def _apply_lighting(img, brightness, gamma, cast, contrast, noise_sigma):
    out = img.astype(np.float32) * brightness
    out[..., 0] *= cast[0]
    out[..., 1] *= cast[1]
    out[..., 2] *= cast[2]
    mean = out.mean()
    out = (out - mean) * contrast + mean
    out = np.clip(out, 0, 255)
    out = 255.0 * np.power(np.clip(out, 0, 255) / 255.0, gamma)
    out = np.clip(out, 0, 255)
    if noise_sigma > 0:
        out += np.random.normal(0, noise_sigma, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def time_dawn(img, box):
    return _apply_lighting(img, rng.uniform(0.45, 0.6), rng.uniform(1.12, 1.22),
                           (1.2, 0.92, 0.8), rng.uniform(0.75, 0.85), rng.uniform(7, 11))


def time_morning(img, box):
    return _apply_lighting(img, rng.uniform(1.03, 1.12), rng.uniform(0.97, 1.0),
                           (0.96, 1.0, 1.05), rng.uniform(1.05, 1.15), 0)


def time_noon(img, box):
    return _apply_lighting(img, rng.uniform(1.2, 1.35), rng.uniform(0.88, 0.95),
                           (1.0, 1.0, 1.0), rng.uniform(1.25, 1.45), 0)


def time_afternoon(img, box):
    return _apply_lighting(img, rng.uniform(1.05, 1.15), rng.uniform(0.95, 1.0),
                           (0.85, 1.0, 1.18), rng.uniform(1.0, 1.1), 0)


def time_dusk(img, box):
    return _apply_lighting(img, rng.uniform(0.6, 0.72), rng.uniform(1.08, 1.18),
                           (0.8, 0.9, 1.25), rng.uniform(0.85, 0.95), rng.uniform(5, 9))


def time_night(img, box):
    return _apply_lighting(img, rng.uniform(0.4, 0.52), rng.uniform(1.15, 1.25),
                           (1.15, 0.85, 0.65), rng.uniform(0.75, 0.85), rng.uniform(9, 14))


def add_rain(img, box):
    H, W = img.shape[:2]
    mask = np.zeros_like(img, dtype=np.float32)
    n = int(55 + rng.uniform(-12, 12))
    for _ in range(n):
        x = rng.randint(0, W - 1)
        y = rng.randint(0, H - 1)
        L = rng.randint(15, 45)
        ang = np.deg2rad(70 + rng.uniform(-6, 6))
        dx = L * np.cos(ang)
        dy = L * np.sin(ang)
        x2 = min(W - 1, int(x + dx))
        y2 = min(H - 1, int(y + dy))
        cv2.line(mask, (x, y), (x2, y2), (215, 215, 225), 1)
    mask = cv2.GaussianBlur(mask, (0, 0), 1.2)
    out = img.astype(np.float32) * 0.9 + mask * 0.65
    return np.clip(out, 0, 255).astype(np.uint8)


def add_fog(img, box):
    t = rng.uniform(0.25, 0.55)
    fog = np.full_like(img, 205, dtype=np.float32)
    out = img.astype(np.float32) * (1 - t) + fog * t
    out = np.clip(out, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= (1 - 0.35 * t)
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)


def night_streetlight_bloom(img, box):
    """夜间压暗 + 路灯暖白聚光直接照射车体表面（局部泛白）+ 外围淡暖晕。"""
    base = _apply_lighting(img, rng.uniform(0.42, 0.5), rng.uniform(1.15, 1.25),
                           (0.72, 0.9, 1.1), 0.82, rng.uniform(8, 12))
    H, W = base.shape[:2]
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    bw = max(1, box[2] - box[0])
    bh = max(1, box[3] - box[1])
    yy, xx = np.mgrid[0:H, 0:W]
    # 车体表面强聚光斑：椭圆高斯，锚定车体中心，纯白（R=G=B），峰近白
    sx, sy = 0.45 * bw, 0.55 * bh
    body = np.exp(-((xx - cx) ** 2) / (2 * sx * sx) - ((yy - cy) ** 2) / (2 * sy * sy))
    peak = rng.uniform(0.88, 1.0)
    m3 = np.stack([body * peak, body * peak, body * peak], -1)
    out = base.astype(np.float32) + m3 * (255.0 - base.astype(np.float32))
    # 外围淡白晕
    halo = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (1.5 * max(bw, bh)) ** 2))
    halo3 = np.stack([halo * 0.13, halo * 0.13, halo * 0.13], -1)
    out += halo3 * (255.0 - base.astype(np.float32))
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------- 几何场景：框随动 ----------
def distance_near(img, box):
    H, W = img.shape[:2]
    s = rng.uniform(1.05, 1.3)
    Ws = int(round(W * s))
    Hs = int(round(H * s))
    big = cv2.resize(img, (Ws, Hs), interpolation=cv2.INTER_LINEAR)
    ox = rng.randint(0, Ws - W) if Ws - W > 0 else 0
    oy = rng.randint(0, Hs - H) if Hs - H > 0 else 0
    crop = big[oy:oy + H, ox:ox + W]
    nb = (box[0] * s - ox, box[1] * s - oy, box[2] * s - ox, box[3] * s - oy)
    return crop, nb


# ---------- 额外建议项 ----------
def motion_blur(img, box):
    size = rng.randint(11, 21)
    ang = rng.uniform(-10, 10)  # 车主要水平运动
    M = cv2.getRotationMatrix2D((size // 2, size // 2), ang, 1.0)
    kernel = np.zeros((size, size), np.float32)
    kernel[size // 2, :] = 1
    kernel = cv2.warpAffine(kernel, M, (size, size))
    kernel /= kernel.sum()
    return cv2.filter2D(img, -1, kernel)


def sensor_noise(img, box):
    sigma = rng.uniform(4, 10)
    g = np.random.normal(0, sigma, img.shape)
    p = np.sqrt(img.astype(np.float32)) * np.random.normal(0, sigma * 0.5, img.shape)
    return np.clip(img.astype(np.float32) + g + p, 0, 255).astype(np.uint8)


def shadow_bars(img, box):
    """5 根横向矩形暗条（长度 0.25×图宽，高度 4%~10% 图高），随机分布在整图各处。"""
    out = img.copy().astype(np.float32)
    H, W = out.shape[:2]
    bar_w = 0.25 * W
    for _ in range(5):
        bar_h = H * rng.uniform(0.04, 0.10)
        x = rng.uniform(0.0, max(1.0, W - bar_w))
        y = rng.uniform(0.0, max(1.0, H - bar_h))  # 保证条完整可见，不贴边截断
        alpha = rng.uniform(0.35, 0.6)
        x1, x2 = int(x), min(W, int(x + bar_w))
        y1, y2 = int(y), min(H, int(y + bar_h))
        if x2 <= x1 or y2 <= y1:
            continue
        out[y1:y2, x1:x2] *= (1 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def wb_cast(img, box):
    kind = rng.choice(["blue", "orange", "green"])
    if kind == "blue":
        c = (1.15, 0.95, 0.8)
    elif kind == "orange":
        c = (0.85, 0.95, 1.2)
    else:
        c = (0.9, 1.15, 0.9)
    out = img.astype(np.float32) * np.array([c[0], c[1], c[2]])
    return np.clip(out, 0, 255).astype(np.uint8)


def exposure_contrast(img, box):
    gamma = rng.uniform(0.55, 1.5)
    contrast = rng.uniform(0.7, 1.5)
    out = 255.0 * np.power(np.clip(img.astype(np.float32), 0, 255) / 255.0, gamma)
    mean = out.mean()
    out = (out - mean) * contrast + mean
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------- 场景池(带权) ----------
SCENARIOS = [
    (time_dawn, "凌晨", 1.5, False, "dawn"), (time_morning, "上午", 1.5, False, "morning"),
    (time_noon, "中午", 1.5, False, "noon"), (time_afternoon, "下午", 1.5, False, "afternoon"),
    (time_dusk, "傍晚", 1.5, False, "dusk"), (time_night, "夜晚", 1.5, False, "night"),
    (add_rain, "雨", 1.2, False, "rain"), (add_fog, "雾", 1.3, False, "fog"),
    (night_streetlight_bloom, "路灯泛白", 1.3, False, "streetlight_bloom"),
    (distance_near, "距离近", 1.5, True, "distance_near"),
    (motion_blur, "运动模糊", 1.0, False, "motion_blur"), (sensor_noise, "传感器噪声", 0.8, False, "sensor_noise"),
    (shadow_bars, "阴影条", 1.2, False, "shadow_bars"),
    (wb_cast, "白平衡偏色", 0.8, False, "wb_cast"), (exposure_contrast, "曝光对比度", 0.8, False, "exposure"),
]
WEIGHTS = [w for _, _, w, _, _ in SCENARIOS]


def _sample_scenarios(n):
    """带权不放回抽 n 个不同场景索引。"""
    idxs = list(range(len(SCENARIOS)))
    out = []
    for _ in range(n):
        if not idxs:
            break
        ws = [WEIGHTS[i] for i in idxs]
        total = sum(ws)
        r = rng.random() * total
        acc = 0.0
        pick = idxs[-1]
        for i, w in zip(idxs, ws):
            acc += w
            if r <= acc:
                pick = i
                break
        out.append(pick)
        idxs.remove(pick)
    return out


def _augment_one(img_bgr, box, fn, geometric):
    if geometric:
        new_img, nb = fn(img_bgr, box)
        nb = _clip_box(nb, new_img.shape[1], new_img.shape[0])
        if nb is None:
            return None, None
        return new_img, nb
    return fn(img_bgr, box), box


def main(preview_only=False):
    open(LOG, "w", encoding="utf-8").close()
    os.makedirs(PREVIEW_DIR, exist_ok=True)

    imgs = sorted(f for f in os.listdir(TRAIN_IMG) if f.lower().endswith(".jpg"))
    log(f"训练图: {len(imgs)} | 每图 {K} 变体 -> 预计新增 {len(imgs)*K} | 场景数: {len(SCENARIOS)}")
    if not imgs:
        return

    # 预览源图：挑框占比最大的前几张小图，便于观察增强效果
    scored = []
    for f in imgs[:300]:
        stem = os.path.splitext(f)[0]
        nb = _read_norm_box(os.path.join(TRAIN_LAB, stem + ".txt"))
        if nb:
            scored.append((nb[2] * nb[3], f))
    scored.sort(reverse=True)
    preview_srcs = [f for _, f in scored[:PREVIEW_SOURCES]] or imgs[:PREVIEW_SOURCES]
    preview_frames = {}
    for i, fn in enumerate(preview_srcs):
        stem = os.path.splitext(fn)[0]
        lbl = os.path.join(TRAIN_LAB, stem + ".txt")
        nb = _read_norm_box(lbl)
        if nb is None:
            continue
        img = cv2.imread(os.path.join(TRAIN_IMG, fn))
        H, W = img.shape[:2]
        cx, cy, bw, bh = nb
        box = _clip_box((cx - bw / 2 * W, cy - bh / 2 * H, cx + bw / 2 * W, cy + bh / 2 * H), W, H)
        frames = {"原图": img.copy()}
        # 每个场景画一个代表变体（固定种子序号以稳定）
        for j, (fnc, name, w, geo, _en_) in enumerate(SCENARIOS):
            for _rep in range(5):  # 试几次拿到有效结果
                aug, nb2 = _augment_one(img, box, fnc, geo)
                if aug is not None:
                    frames[name] = aug
                    break
        preview_frames[i] = frames

    # 生成蒙太奇
    from PIL import Image, ImageDraw, ImageFont
    _font = None
    for fp in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\simhei.ttf"):
        if os.path.exists(fp):
            try:
                _font = ImageFont.truetype(fp, 20)
                break
            except Exception:
                _font = None
    _en = {s[1]: s[4] for s in SCENARIOS}
    for i, frames in preview_frames.items():
        names = list(frames.keys())
        cols = 4
        rows = (len(names) + cols - 1) // cols
        th = 160
        tw = 300
        sheet = np.full((rows * th, cols * tw, 3), 18, np.uint8)
        for idx, name in enumerate(names):
            r, c = divmod(idx, cols)
            img = cv2.resize(frames[name], (tw - 8, th - 30))
            sheet[r * th + 4:r * th + 4 + img.shape[0], c * tw + 4:c * tw + 4 + img.shape[1]] = img
        pil = Image.fromarray(cv2.cvtColor(sheet, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        for idx, name in enumerate(names):
            r, c = divmod(idx, cols)
            lbl = name if _font is not None else _en.get(name, name)
            draw.text((c * tw + 6, r * th + th - 26), lbl, fill=(255, 255, 0),
                      font=_font if _font is not None else ImageFont.load_default())
        out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        p = os.path.join(PREVIEW_DIR, f"preview_src{i}.jpg")
        cv2.imwrite(p, out, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        log(f"预览蒙太奇 -> {p}")
        # 逐场景小图（1280 宽），供验收页展示
        cell_dir = os.path.join(PREVIEW_DIR, "cells", f"src{i}")
        os.makedirs(cell_dir, exist_ok=True)
        for name, frame in frames.items():
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (1280, int(1280 * h / w)))
            # cv2.imwrite 在 Windows 下对中文路径不可靠(GBK 编码/0x5C 字节问题)，
            # 用 ASCII 英文名落盘，中文标签由蒙太奇(PIL)与验收页负责。
            en = "source" if name == "原图" else _en.get(name, "cell%d" % len(frames))
            cv2.imwrite(os.path.join(cell_dir, f"{en}.jpg"), small, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        log(f"预览小图 -> {cell_dir}")

    if preview_only:
        log("预览模式：未进行全量增强，等待确认")
        return

    # 主增强循环
    t0 = time.time()
    done = written = geo_cnt = 0
    for fn in imgs:
        stem = os.path.splitext(fn)[0]
        lbl = os.path.join(TRAIN_LAB, stem + ".txt")
        nb = _read_norm_box(lbl)
        img = cv2.imread(os.path.join(TRAIN_IMG, fn))
        if img is None or nb is None:
            done += 1
            continue
        H, W = img.shape[:2]
        cx, cy, bw, bh = nb
        box = _clip_box((cx - bw / 2 * W, cy - bh / 2 * H, cx + bw / 2 * W, cy + bh / 2 * H), W, H)
        if box is None:
            done += 1
            continue
        picks = _sample_scenarios(K)
        for n_idx, s_idx in enumerate(picks):
            fnc, name, w, geo, _en_ = SCENARIOS[s_idx]
            aug, nb2 = _augment_one(img, box, fnc, geo)
            if aug is None:
                continue
            out_name = f"{stem}__aug{n_idx+1}.jpg"
            cv2.imwrite(os.path.join(TRAIN_IMG, out_name), aug, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_Q])
            with open(os.path.join(TRAIN_LAB, os.path.splitext(out_name)[0] + ".txt"), "w", encoding="utf-8") as f:
                f.write(_box_to_norm(nb2, aug.shape[1], aug.shape[0]))
            written += 1
            geo_cnt += 1 if geo else 0
        done += 1
        if done % 200 == 0:
            log(f"进度 {done}/{len(imgs)} 已写 {written} 用时 {time.time()-t0:.1f}s")

    log(f"完成: 处理 {done} 图 | 写入增强 {written} 张 (几何 {geo_cnt}) | 用时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview-only", action="store_true", help="只生成预览蒙太奇/小图，不进行全量增强")
    args = ap.parse_args()
    main(preview_only=args.preview_only)
