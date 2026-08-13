# -*- coding: utf-8 -*-
"""CharReader — 尾部视角车挂号/放大号字符检测+比对.

封装方案B 两阶段字符检测管线，支持车挂号(48类 conf=0.70)和放大号(49类 conf=0.90)。

用法:
    reader = CharReader()
    result = reader.compare_pair("path3.jpg", "path4.jpg")
    # result["verdict"] ∈ {"一致", "不一致", "无法判断", "作废"}
"""

import os
import sys
import json
import numpy as np
import cv2


# ── 模型路径 (引用 D:\data2\weibu_zifu 成品权重) ──────────────────────
# 2026-08-12 结构重构后路径更新: 车挂号->che_gua_hao/, 放大号->fang_da_hao/.
# 放大号权重已替换: 检测器=单类找框(yolo11n_fd_char_s1), 分类器=50类含鄂(yolo11n_fd_char_cls2).
_GUA_BOX_W = r"D:\data2\weibu_zifu\yolo_det\weights\best.pt"
_CHAR_DET_W = r"D:\data2\weibu_zifu\che_gua_hao\char_det_train\yolo11n_char_v2\weights\best.pt"
_CHAR_CLS_W = r"D:\data2\weibu_zifu\che_gua_hao\char_cls_train\yolo11n_char_cls2\weights\best.pt"
_FD_DET_W = r"D:\data2\weibu_zifu\fang_da_hao\fang_da_hao_det_train\yolo11n_fd_char\weights\best.pt"
_FD_CLS_W = r"D:\data2\weibu_zifu\fang_da_hao\fang_da_hao_cls_train\yolo11n_fd_char_cls\weights\best.pt"

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CLS_NAMES_JSON = os.path.join(_THIS_DIR, "cls_names.json")
_FD_CLS_NAMES_JSON = os.path.join(_THIS_DIR, "fd_cls_names.json")

# ── 参数 ──────────────────────────────────────────────────────────────
GUA_CONF_LINE = 0.70        # 车挂号取信线
FD_CONF_LINE = 0.90         # 放大号取信线
MIN_CHARS = 3               # 一图可读下限
DET_CHAR_CONF = 0.25        # 字符检测 conf
DET_CHAR_IOU = 0.45         # 字符检测 iou
DET_CHAR_IMSZ = 512         # 字符检测 imgsz
CLS_IMSZ = 96               # 分类器输入尺寸
GUA_BOX_CONF = 0.20         # 车挂号/放大号 box 检测 conf
GUA_CLS_ID = 0              # yolo_det: 车挂号窄框
FD_CLS_ID = 1               # yolo_det: 放大号宽框


# ── 辅助函数 ──────────────────────────────────────────────────────────

def _imread(path):
    """中文路径安全读取."""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _is_han(c):
    return ord(c) > 127


def _crop_square(img, xyxy, pad_frac=0.10, imgsz=96):
    """按框裁剪 → 扩 pad_frac → 长边 pad 正方 → resize imgsz. 返回 imgsz x imgsz BGR."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = xyxy
    pw, ph = (x2 - x1) * pad_frac, (y2 - y1) * pad_frac
    x1 = max(0, int(x1 - pw))
    y1 = max(0, int(y1 - ph))
    x2 = min(W, int(x2 + pw))
    y2 = min(H, int(y2 + ph))
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    h, w = crop.shape[:2]
    side = max(h, w)
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    canvas[:h, :w] = crop
    return cv2.resize(canvas, (imgsz, imgsz), interpolation=cv2.INTER_CUBIC)


def _reading_order(boxes):
    """boxes: [(x1,y1,x2,y2),...] → 读序下标列表.
    y-center 聚类 1-2 行 → 行内 x 左→右 → x 离群滤除."""
    if not boxes:
        return []
    cents = np.array([((b[1] + b[3]) / 2, (b[0] + b[2]) / 2) for b in boxes])
    hs = np.array([b[3] - b[1] for b in boxes])
    order = np.argsort(cents[:, 0])
    ys = cents[order, 0]
    gap = np.diff(ys)
    h_med = float(np.median(hs)) if len(hs) else 0.0
    if len(order) > 1 and gap.max() > 0.5 * h_med:
        split_at = int(np.argmax(gap)) + 1
        row_groups = [list(order[:split_at]), list(order[split_at:])]
    else:
        row_groups = [sorted(order.tolist(), key=lambda i: cents[i, 1])]
    rows = []
    for row in row_groups:
        row = list(row)
        xs = np.array([cents[i, 1] for i in row])
        if len(row) > 3:
            o = np.argsort(xs)
            gaps = np.diff(xs[o])
            med_gap = float(np.median(gaps)) if len(gaps) else 0.0
            if med_gap > 0:
                mx = int(np.argmax(gaps))
                if gaps[mx] > 2.5 * med_gap:
                    keep_pos = o[mx + 1:] if mx < len(o) // 2 else o[:mx + 1]
                    row = [row[int(i)] for i in keep_pos]
        row.sort(key=lambda i: cents[i, 1])
        rows.extend(row)
    return rows


def _align_seq(seqA, seqB):
    """编辑距离对齐 → [(ca, cb), ...], 缺口用 None."""
    n, m = len(seqA), len(seqB)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if seqA[i - 1][0] == seqB[j - 1][0] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    pairs = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if seqA[i - 1][0] == seqB[j - 1][0] else 1):
            pairs.append((seqA[i - 1], seqB[j - 1])); i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            pairs.append((seqA[i - 1], None)); i -= 1
        else:
            pairs.append((None, seqB[j - 1])); j -= 1
    pairs.reverse()
    return pairs


def normalize_seq(seq, conf_line=None):
    """读序规范化 (保留可靠字符对):
    1. 厂/内 辅助字调整到序列最前, 保持"厂内"顺序
    2. 位置语法修正: 省字(含桂)只能在辅助字之后的省字位; "挂"只能末尾;
       省字位的"挂"=桂误识别→转桂; 末尾"桂"=挂误识别→转挂; 中间省字剔除
    3. 一张牌最多一个省字 + 最多一个"挂", 多余剔除
    """
    seq = list(seq)
    aux = [(c, cf) for c, cf in seq if c in ("厂", "内")]
    rest = [(c, cf) for c, cf in seq if c not in ("厂", "内")]
    aux.sort(key=lambda x: 0 if x[0] == "厂" else 1)
    ordered = aux + rest
    n = len(ordered)
    prov_idx = len(aux)
    out = []
    kept_gua = 0
    for i, (c, cf) in enumerate(ordered):
        last = (i == n - 1)
        if c in ("厂", "内"):
            out.append((c, cf))
        elif c == "挂":
            if last and kept_gua == 0:
                out.append((c, cf)); kept_gua += 1
            elif i == prov_idx:
                out.append(("桂", cf))
            else:
                pass
        elif c == "桂":
            if i == prov_idx:
                out.append((c, cf))
            elif last:
                out.append(("挂", cf))
            else:
                pass
        elif _is_han(c):
            if i == prov_idx:
                out.append((c, cf))
            else:
                pass
        else:
            out.append((c, cf))
    return out


def compare(seqA, seqB, conf_line):
    """方案B 比对: 按位对齐 → R/M/U → 判定.
    Returns: {"verdict": "一致"|"不一致"|"无法判断", "R": int, "M": int, "U": int}
    """
    if not seqA or not seqB:
        return {"verdict": "作废", "R": 0, "M": 0, "U": 0}
    pairs = _align_seq(seqA, seqB)
    R = M = U = 0
    for pairA, pairB in pairs:
        ca = fa = cb = fb = None
        if pairA is not None:
            ca, fa = pairA
        if pairB is not None:
            cb, fb = pairB
        if ca is None or cb is None:
            U += 1
        else:
            a_reliable = fa >= conf_line
            b_reliable = fb >= conf_line
            if a_reliable and b_reliable:
                R += 1
                if ca != cb:
                    M += 1
            else:
                U += 1
    if R < 4:
        verdict = "无法判断"
    elif M >= 2:
        verdict = "不一致"
    elif M == 0:
        verdict = "一致"
    elif M == 1 and U > 2:
        verdict = "无法判断"
    else:
        verdict = "一致"
    return {"verdict": verdict, "R": R, "M": M, "U": U}


def analyze_plate(det, cls_model, names, img, conf_line):
    """单图字符检测→分类→读序.
    det: 字符检测 YOLO, cls_model: 字符分类 YOLO, names: 类名列表.
    Returns: [(char, conf), ...] 按读序排列."""
    res = det.predict(img, conf=DET_CHAR_CONF, iou=DET_CHAR_IOU, verbose=False, imgsz=DET_CHAR_IMSZ)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return []
    boxes = res.boxes.xyxy.cpu().numpy().tolist()
    order = _reading_order(boxes)
    ordered_boxes = [boxes[i] for i in order]
    crops = []
    for b in ordered_boxes:
        c = _crop_square(img, b, imgsz=CLS_IMSZ)
        crops.append(c if c is not None else np.zeros((CLS_IMSZ, CLS_IMSZ, 3), dtype=np.uint8))
    preds = cls_model.predict(crops, imgsz=CLS_IMSZ, verbose=False)
    seq = []
    for r in preds:
        c = names[r.probs.top1]
        seq.append((c, float(r.probs.top1conf)))
    return seq


# ── CharReader ────────────────────────────────────────────────────────

class CharReader:
    """尾部视角车挂号/放大号字符读取 + 比对.

    模型懒加载: 首次调用时自动加载 YOLO 模型 (耗时 ~10s 加载 5 个模型).
    线程安全: 依赖 GIL, 不保证多线程并发安全 (Flask threaded=True 每次请求串行调用).
    """

    def __init__(self):
        self._models = {}
        self._cropper = None
        self._gua_names = None
        self._fd_names = None

    # ── 懒加载 ──────────────────────────────────────────────────────

    def _load_model(self, key, path):
        if key not in self._models:
            from ultralytics import YOLO
            self._models[key] = YOLO(path)
        return self._models[key]

    @property
    def gua_box(self):
        return self._load_model("gua_box", _GUA_BOX_W)

    @property
    def char_det(self):
        return self._load_model("char_det", _CHAR_DET_W)

    @property
    def char_cls(self):
        return self._load_model("char_cls", _CHAR_CLS_W)

    @property
    def fd_det(self):
        return self._load_model("fd_det", _FD_DET_W)

    @property
    def fd_cls(self):
        return self._load_model("fd_cls", _FD_CLS_W)

    @property
    def cropper(self):
        if self._cropper is None:
            from chewei_detect.chewei_detect import VehicleCropper as TailViewCropper
            self._cropper = TailViewCropper()
        return self._cropper

    @property
    def gua_names(self):
        if self._gua_names is None:
            with open(_CLS_NAMES_JSON, encoding="utf-8") as f:
                id2char = json.load(f)
            self._gua_names = [id2char[f"{i:02d}"] for i in range(48)]
        return self._gua_names

    @property
    def fd_names(self):
        if self._fd_names is None:
            with open(_FD_CLS_NAMES_JSON, encoding="utf-8") as f:
                id2char = json.load(f)
            self._fd_names = [id2char[f"{i:02d}"] for i in range(50)]
        return self._fd_names

    # ── 预热 ────────────────────────────────────────────────────────

    def warmup(self):
        """触发所有模型加载 (避免首次请求等待)."""
        _ = self.gua_box
        _ = self.char_det
        _ = self.char_cls
        _ = self.fd_det
        _ = self.fd_cls
        _ = self.cropper
        _ = self.gua_names
        _ = self.fd_names

    # ── 单图读牌 ────────────────────────────────────────────────────

    def _read_plate(self, img_bgr, plate_type, conf_line):
        """从车辆 crop 图读取指定类型号牌字符序列.

        Args:
            img_bgr: 车辆裁剪图 (BGR numpy array)
            plate_type: "chegua" | "fangdahao"
            conf_line: 取信线

        Returns:
            {"status": "OK"|"无框"|"不可读", "seq": [(char,conf)], "boxes": [...]}
        """
        cls_id = GUA_CLS_ID if plate_type == "chegua" else FD_CLS_ID
        det_model = self.char_det if plate_type == "chegua" else self.fd_det
        cls_model = self.char_cls if plate_type == "chegua" else self.fd_cls
        names = self.gua_names if plate_type == "chegua" else self.fd_names

        r = self.gua_box.predict(img_bgr, conf=GUA_BOX_CONF, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return {"status": "无框", "seq": [], "boxes": [], "char_n": 0}

        best_seq, best_score, best_boxes = None, -1, None
        for b, c, s in zip(r.boxes.xyxy.cpu().numpy(),
                           r.boxes.cls.cpu().numpy().astype(int),
                           r.boxes.conf.cpu().numpy()):
            if c != cls_id:
                continue
            x1, y1, x2, y2 = [int(v) for v in b]
            crop = img_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            seq = normalize_seq(analyze_plate(det_model, cls_model, names, crop, conf_line))
            score = sum(1 for _c, _f in seq if _f >= conf_line)
            if score > best_score or (score == best_score and len(seq) > len(best_seq or [])):
                best_score = score
                best_seq = seq
                best_boxes = [(x1, y1, x2, y2)]

        if best_seq is None or len(best_seq) < MIN_CHARS:
            return {"status": "不可读", "seq": best_seq or [], "boxes": best_boxes or [],
                    "char_n": len(best_seq) if best_seq else 0}

        return {"status": "OK", "seq": best_seq, "boxes": best_boxes,
                "char_n": len(best_seq)}

    # ── 单图双牌读取 ────────────────────────────────────────────────

    def _read_both(self, img_bgr):
        """同时读取车挂号+放大号.
        Returns: {"chegua": {...}, "fangdahao": {...}}
        """
        return {
            "chegua": self._read_plate(img_bgr, "chegua", GUA_CONF_LINE),
            "fangdahao": self._read_plate(img_bgr, "fangdahao", FD_CONF_LINE),
        }

    # ── 两图比对主入口 ──────────────────────────────────────────────

    def compare_pair(self, path3, path4, pre_cropped=None):
        """两图比对 — 分阶段: 车挂号优先, 车挂号无法确定再放大号; 跨类兜底保留.

        阶段1: 只读车挂号(两侧)并比对; 车挂号 一致/不一致 即定案, 不再读放大号.
        阶段2: 车挂号无法确定(无框/不可读/单侧/比对R<4) → 再读放大号比对;
               放大号同类优先, 单侧可用时保留跨类(车挂号vs放大号)兜底.
        检测兜底: 都无→作废; 单边→直接用.

        pre_cropped: 可选 (bgr3, bgr4) — 上游已用同一 VehicleCropper 裁好的
            车辆图, 传入则跳过内部二次裁剪(提速)。

        Returns: dict with verdict, R, M, U, seqs, boxes, etc.
        """
        result = {
            "verdict": "作废",
            "plate_type_used": None,
            "R": 0, "M": 0, "U": 0,
            "p3_seq": [], "p4_seq": [],
            "p3_status": "", "p4_status": "",
            "p3_boxes": [], "p4_boxes": [],
            "error": None,
        }
        # 每图每类型字段默认值 (任何提前返回路径都有值, 供详情页展示/画框)
        for _img_key in ("3", "4"):
            for _typ in ("chegua", "fangdahao"):
                result[f"p{_img_key}_{_typ}_seq"] = []
                result[f"p{_img_key}_{_typ}_status"] = "未检测"
                result[f"p{_img_key}_{_typ}_boxes"] = []

        # 1. 读取 + 车辆裁剪 (可复用上游裁剪, 跳过二次裁剪)
        if pre_cropped is not None:
            vcrop3, vcrop4 = pre_cropped
            if vcrop3 is None or vcrop4 is None or getattr(vcrop3, "size", 0) == 0 or getattr(vcrop4, "size", 0) == 0:
                result["error"] = "pre_cropped 包含空图"
                return result
            vbox3 = vbox4 = (0, 0, 1, 1)  # 占位: 上游已裁到车辆
        else:
            img3 = _imread(path3)
            img4 = _imread(path4)
            if img3 is None:
                result["error"] = "path3 读图失败"
                return result
            if img4 is None:
                result["error"] = "path4 读图失败"
                return result

            try:
                vcrop3, vbox3 = self.cropper.crop_image(path3)
                vcrop4, vbox4 = self.cropper.crop_image(path4)
            except Exception as e:
                result["error"] = f"车辆裁剪失败: {e}"
                return result

        if vbox3 is None and vbox4 is None:
            result["verdict"] = "作废"
            result["p3_status"] = "无车辆"
            result["p4_status"] = "无车辆"
            for _img_key in ("3", "4"):
                for _typ in ("chegua", "fangdahao"):
                    result[f"p{_img_key}_{_typ}_status"] = "无车辆"
            return result

        def _empty_plate(status="无车辆"):
            return {"status": status, "seq": [], "boxes": [], "char_n": 0}

        # 阶段1: 只读车挂号 (车挂号 一致/不一致 即定案, 不再读放大号)
        c3 = self._read_plate(vcrop3, "chegua", GUA_CONF_LINE) if vbox3 is not None else _empty_plate()
        c4 = self._read_plate(vcrop4, "chegua", GUA_CONF_LINE) if vbox4 is not None else _empty_plate()
        for _img_key, _rd in (("3", c3), ("4", c4)):
            result[f"p{_img_key}_chegua_seq"] = _rd["seq"]
            result[f"p{_img_key}_chegua_status"] = _rd["status"]
            result[f"p{_img_key}_chegua_boxes"] = _rd["boxes"]

        c3_ok = c3["status"] == "OK"
        c4_ok = c4["status"] == "OK"

        if c3_ok and c4_ok:
            # 两边车挂号都可读 → 直接比对
            cmp = compare(c3["seq"], c4["seq"], GUA_CONF_LINE)
            if cmp["verdict"] in ("一致", "不一致"):
                # 车挂号定案, 放大号字段保持"未检测"
                result["verdict"] = cmp["verdict"]
                result["plate_type_used"] = "chegua"
                result["p3_status"] = "chegua_OK"
                result["p4_status"] = "chegua_OK"
                result["p3_seq"] = c3["seq"]
                result["p4_seq"] = c4["seq"]
                result["p3_boxes"] = c3["boxes"]
                result["p4_boxes"] = c4["boxes"]
                result["R"] = cmp["R"]
                result["M"] = cmp["M"]
                result["U"] = cmp["U"]
                return result
            # 车挂号比对"无法判断"(R<4) → 落到放大号

        # 阶段2: 车挂号无法确定, 再读放大号
        f3 = self._read_plate(vcrop3, "fangdahao", FD_CONF_LINE) if vbox3 is not None else _empty_plate()
        f4 = self._read_plate(vcrop4, "fangdahao", FD_CONF_LINE) if vbox4 is not None else _empty_plate()
        for _img_key, _rd in (("3", f3), ("4", f4)):
            result[f"p{_img_key}_fangdahao_seq"] = _rd["seq"]
            result[f"p{_img_key}_fangdahao_status"] = _rd["status"]
            result[f"p{_img_key}_fangdahao_boxes"] = _rd["boxes"]

        f3_ok = f3["status"] == "OK"
        f4_ok = f4["status"] == "OK"

        # 3. 匹配选择 (放大号同类优先 → 跨类兜底 → 单侧/作废)
        seqA = seqB = None
        conf_line = GUA_CONF_LINE

        if f3_ok and f4_ok:
            # 两边都有放大号 → 放大号比对
            seqA, seqB = f3["seq"], f4["seq"]
            result["plate_type_used"] = "fangdahao"
            result["p3_status"] = "fd_OK"
            result["p4_status"] = "fd_OK"
            result["p3_boxes"] = f3["boxes"]
            result["p4_boxes"] = f4["boxes"]
            conf_line = FD_CONF_LINE
        elif c3_ok and f4_ok:
            # 交叉兜底: p3车挂号 p4放大号
            seqA, seqB = c3["seq"], f4["seq"]
            result["plate_type_used"] = "cross_chegua_fd"
            result["p3_status"] = "chegua_OK"
            result["p4_status"] = "fd_OK"
            result["p3_boxes"] = c3["boxes"]
            result["p4_boxes"] = f4["boxes"]
            conf_line = min(GUA_CONF_LINE, FD_CONF_LINE)
        elif f3_ok and c4_ok:
            # 交叉兜底: p3放大号 p4车挂号
            seqA, seqB = f3["seq"], c4["seq"]
            result["plate_type_used"] = "cross_fd_chegua"
            result["p3_status"] = "fd_OK"
            result["p4_status"] = "chegua_OK"
            result["p3_boxes"] = f3["boxes"]
            result["p4_boxes"] = c4["boxes"]
            conf_line = min(GUA_CONF_LINE, FD_CONF_LINE)
        elif c3_ok and not c4_ok and not f4_ok:
            # 只有p3有车挂号, p4都无 → 无法比对
            result["p3_status"] = "chegua_OK"
            result["p4_status"] = "p4_no_plate"
            result["verdict"] = "作废"
            return result
        elif c4_ok and not c3_ok and not f3_ok:
            result["p4_status"] = "chegua_OK"
            result["p3_status"] = "p3_no_plate"
            result["verdict"] = "作废"
            return result
        elif f3_ok and not c4_ok and not f4_ok:
            result["p3_status"] = "fd_OK"
            result["p4_status"] = "p4_no_plate"
            result["verdict"] = "作废"
            return result
        elif f4_ok and not c3_ok and not f3_ok:
            result["p4_status"] = "fd_OK"
            result["p3_status"] = "p3_no_plate"
            result["verdict"] = "作废"
            return result
        else:
            # 两边都有框但都不可读/或组合不支持
            result["verdict"] = "作废"
            result["p3_status"] = f"c={c3['status']}_f={f3['status']}"
            result["p4_status"] = f"c={c4['status']}_f={f4['status']}"
            return result

        result["p3_seq"] = seqA
        result["p4_seq"] = seqB

        # 4. 方案B 比对
        cmp = compare(seqA, seqB, conf_line)
        result["verdict"] = cmp["verdict"]
        result["R"] = cmp["R"]
        result["M"] = cmp["M"]
        result["U"] = cmp["U"]

        return result


# ── 便捷函数 ──────────────────────────────────────────────────────────

def fmt_seq(seq, conf_line=0.70):
    """格式化字符序列为可读字符串, 低置信字符加 ?."""
    if not seq:
        return ""
    chars = []
    for c, cf in seq:
        if cf is not None and cf < conf_line:
            chars.append(f"{c}?")
        else:
            chars.append(c)
    return "".join(chars)
