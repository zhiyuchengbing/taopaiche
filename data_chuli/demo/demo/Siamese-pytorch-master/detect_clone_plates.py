import argparse
import os
import re
import sys
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd
from PIL import Image

# 保证可以导入到 data_chuli 下的工具
PARENT_DIR = os.path.dirname(os.path.dirname(__file__))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

# 复用项目内模型与裁剪
from siamese import Siamese
from data_chuli.cropper import VehicleCropper


TIME_COL_TARE = "皮重过磅时间"
TIME_COL_GROSS = "毛重过磅时间"
PLATE_COL = "车号"
IMG_COL = "过皮部位1图片URL"


def try_read_csv(csv_path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "gbk", "utf-8"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(csv_path, encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"无法读取CSV，请检查编码/路径。最后错误: {last_err}")


def parse_time_from_filename(path_str: str) -> Optional[pd.Timestamp]:
    """
    尝试从文件名解析时间，匹配常见模式：..._YYYYMMDD_HHMMSS_...
    例如：22_1_20251111_145406_CH01_0039_XXXXXXXX.jpg
    """
    if not isinstance(path_str, str):
        return None
    m = re.search(r"(20\d{6})[_-](\d{6})", os.path.basename(path_str))
    if not m:
        # 兼容路径中包含日期时间片段的情况
        m = re.search(r"(20\d{6}).*?(\d{6})", path_str)
    if not m:
        return None
    yyyymmdd, hhmmss = m.group(1), m.group(2)
    try:
        return pd.to_datetime(f"{yyyymmdd} {hhmmss}", format="%Y%m%d %H%M%S", errors="coerce")
    except Exception:
        return None


def parse_event_time(row: pd.Series) -> Optional[pd.Timestamp]:
    # 1) 皮重
    t1 = pd.to_datetime(row.get(TIME_COL_TARE), errors="coerce") if TIME_COL_TARE in row else pd.NaT
    if pd.notna(t1):
        return t1
    # 2) 毛重
    t2 = pd.to_datetime(row.get(TIME_COL_GROSS), errors="coerce") if TIME_COL_GROSS in row else pd.NaT
    if pd.notna(t2):
        return t2
    # 3) 文件名兜底
    return parse_time_from_filename(row.get(IMG_COL))


def load_and_prepare_image(img_path: str, cropper: VehicleCropper) -> Optional[Image.Image]:
    if not isinstance(img_path, str) or not img_path:
        return None
    if not os.path.exists(img_path):
        return None
    try:
        img = Image.open(img_path)
        img = cropper.process_pil(img)
        return img
    except Exception:
        return None


def detect_similarity(model: Siamese, img1: Image.Image, img2: Image.Image) -> Optional[float]:
    try:
        prob = model.detect_image(img1, img2)
        if hasattr(prob, "item"):
            prob = prob.item()
        return float(prob)
    except Exception:
        return None


def detect_from_csv(csv_path: Path, threshold: float = 0.3) -> Tuple[pd.DataFrame, Path]:
    df = try_read_csv(csv_path)

    # 基础列检查
    for col in [PLATE_COL, IMG_COL]:
        if col not in df.columns:
            raise KeyError(f"缺少必要列: {col}。当前列: {list(df.columns)}")

    # 提取时间
    df = df.copy()
    df["__event_time"] = df.apply(parse_event_time, axis=1)

    # 初始化模型与裁剪器
    model = Siamese()
    cropper = VehicleCropper()

    results: List[dict] = []

    # 提取可能存在的任务ID列名（兼容不同表头）
    candidate_task_cols = ["任务ID", "任务Id", "id", "ID", "任务编号"]
    task_col = next((c for c in candidate_task_cols if c in df.columns), None)

    # 按车号分组
    grouped = df.groupby(PLATE_COL, dropna=False)

    for plate, g in grouped:
        # 排序
        g_sorted = g.sort_values("__event_time", kind="mergesort")  # 保持稳定排序

        # 累积一个历史列表：(idx, time, img_path, prepared_img)
        history: List[Tuple[int, pd.Timestamp, str, Optional[Image.Image]]] = []

        for idx, row in g_sorted.iterrows():
            current_img_path = row.get(IMG_COL)
            current_time = row.get("__event_time")
            task_id_val = row.get(task_col) if task_col else None

            if not isinstance(current_img_path, str) or not current_img_path or not os.path.exists(current_img_path):
                results.append({
                    "任务ID": task_id_val,
                    "车号": plate,
                    "当前时间": current_time,
                    "当前图片": current_img_path,
                    "参考任务ID": None,
                    "参考时间": None,
                    "参考图片": None,
                    "相似度": None,
                    "判定": "不可判定",
                    "备注": "无当前图或文件不存在"
                })
                # 不将无图记录加入历史可比集合，但仍然保留时间线
                continue

            # 寻找最近一条历史（时间在当前之前）的有图记录
            # 从history末尾向前找最近有prepared_img的记录
            ref_idx, ref_time, ref_path, ref_img = None, None, None, None
            for h_idx in range(len(history) - 1, -1, -1):
                h_i, h_t, h_p, h_img = history[h_idx]
                if pd.notna(current_time) and pd.notna(h_t) and h_t is not None and (h_t < current_time):
                    if h_img is not None:
                        ref_idx, ref_time, ref_path, ref_img = h_i, h_t, h_p, h_img
                        break
                # 若当前时间缺失，只要历史有图则也可作为参考（但备注说明）
                elif pd.isna(current_time) and h_img is not None:
                    ref_idx, ref_time, ref_path, ref_img = h_i, h_t, h_p, h_img
                    break

            # 准备当前图片（放在寻找参考之后，避免无谓开销）
            curr_img_prepared = load_and_prepare_image(current_img_path, cropper)
            # 将当前加入历史以供后续使用（即使准备失败，也记录路径和时间）
            history.append((idx, current_time, current_img_path, curr_img_prepared))

            if curr_img_prepared is None:
                results.append({
                    "任务ID": task_id_val,
                    "车号": plate,
                    "当前时间": current_time,
                    "当前图片": current_img_path,
                    "参考任务ID": None,
                    "参考时间": None,
                    "参考图片": None,
                    "相似度": None,
                    "判定": "不可判定",
                    "备注": "当前图像读取或预处理失败"
                })
                continue

            if ref_img is None:
                results.append({
                    "任务ID": task_id_val,
                    "车号": plate,
                    "当前时间": current_time,
                    "当前图片": current_img_path,
                    "参考任务ID": None,
                    "参考时间": None,
                    "参考图片": None,
                    "相似度": None,
                    "判定": "新来车",
                    "备注": "无可用历史参考图"
                })
                continue

            # 做相似度
            prob = detect_similarity(model, curr_img_prepared, ref_img)
            if prob is None:
                results.append({
                    "任务ID": task_id_val,
                    "车号": plate,
                    "当前时间": current_time,
                    "当前图片": current_img_path,
                    "参考任务ID": None,
                    "参考时间": ref_time,
                    "参考图片": ref_path,
                    "相似度": None,
                    "判定": "不可判定",
                    "备注": "模型推理失败"
                })
                continue

            decision = "正常" if prob >= threshold else "疑似套牌"
            results.append({
                "任务ID": task_id_val,
                "车号": plate,
                "当前时间": current_time,
                "当前图片": current_img_path,
                "参考任务ID": None,  # 无法稳定找到参考任务ID，除非我们缓存它
                "参考时间": ref_time,
                "参考图片": ref_path,
                "相似度": prob,
                "判定": decision,
                "备注": None
            })

    res_df = pd.DataFrame(results)

    # 输出路径与保存
    out_path = csv_path.parent / "clone_check_report.csv"
    res_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    # 控制台摘要
    total = len(res_df)
    cnt_normal = (res_df["判定"] == "正常").sum()
    cnt_clone = (res_df["判定"] == "疑似套牌").sum()
    cnt_new = (res_df["判定"] == "新来车").sum()
    cnt_na = (res_df["判定"] == "不可判定").sum()

    print(f"总记录: {total}")
    print(f"正常: {cnt_normal}  疑似套牌: {cnt_clone}  新来车: {cnt_new}  不可判定: {cnt_na}")
    print(f"结果已保存: {out_path}")

    return res_df, out_path


def main():
    parser = argparse.ArgumentParser(description="按同车牌向过去最近一趟进行图像比对，判断疑似套牌")
    parser.add_argument("--csv", type=str, default=str(Path(PARENT_DIR) / "data_chuli" / "data" / "匹配数据.csv"), help="输入CSV路径")
    parser.add_argument("--threshold", type=float, default=0.3, help="相似度阈值，>=阈值视为同车（正常）")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV不存在: {csv_path}")

    detect_from_csv(csv_path, threshold=args.threshold)


if __name__ == "__main__":
    main()
