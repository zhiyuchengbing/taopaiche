import os
import sys
import threading
import urllib.parse
import base64
import io
import json
import time
import datetime
import uuid
import shutil
import csv
import zipfile
import tempfile
import re
import random
from collections import deque
from typing import Optional, Tuple, Dict, Any, List

import requests

import numpy as np
import cv2
from PIL import Image
from flask import Flask, jsonify, request, render_template, send_file, send_from_directory
from ultralytics import YOLO

from siamese import Siamese
from data_tran.image_resolver import ImagePathResolver
from qwen_vl.predict_ai import VehicleCheck
from qwen_vl.predict_ai_shijiao2 import TailVehicleCheck
from chewei_detect.chewei_detect import VehicleCropper as TailViewCropper
from plate_char_det import CharReader
from plate_char_det.char_reader import fmt_seq as _fmt_char_seq
from plate_char_det.char_reader import GUA_CONF_LINE as _GUA_CONF_LINE

parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from data_chuli.cropper import VehicleCropper as MainVehicleCropper

app = Flask(__name__)

_INIT_LOCK = threading.Lock()
_PIPELINE_LOCK = threading.Lock()
_INITIALIZED = False

_CROPPER: Optional[MainVehicleCropper] = None
_CROPPER_UNMASKED: Optional[MainVehicleCropper] = None
_HEAD_MODEL: Optional[Siamese] = None
_TAIL_MODEL: Optional[Siamese] = None
_HEADTAIL_MODEL: Optional[YOLO] = None
_TAIL_VIEW_CROPPER: Optional[TailViewCropper] = None
_IMAGE_RESOLVER: Optional[ImagePathResolver] = None
_AI_CHECKER: Optional[VehicleCheck] = None
_AI_TAIL_CHECKER: Optional[TailVehicleCheck] = None
_CHAR_READER: Optional[CharReader] = None

_DEFAULT_HEAD_THRESHOLD = float(os.environ.get("HEAD_THRESHOLD_DEFAULT", "0.8"))
_DEFAULT_TAIL_THRESHOLD = float(os.environ.get("TAIL_THRESHOLD_DEFAULT", "0.8"))
_DEFAULT_TAIL_CHAR_THRESHOLD = float(os.environ.get("TAIL_CHAR_THRESHOLD_DEFAULT", "0.85"))
_DEFAULT_TAIL_SIM_CHANGE_LOW = float(os.environ.get("TAIL_SIM_CHANGE_LOW_DEFAULT", "0.25"))
_DIRECT_FAKE_PLATE_HEAD_THRESHOLD = float(os.environ.get("DIRECT_FAKE_PLATE_HEAD_THRESHOLD", "0.1"))
_THRESHOLDS_FILE = os.path.join(os.path.dirname(__file__), "thresholds.json")
_THRESHOLD_LOCK = threading.Lock()
_HEAD_THRESHOLD: float = _DEFAULT_HEAD_THRESHOLD
_TAIL_THRESHOLD: float = _DEFAULT_TAIL_THRESHOLD
_TAIL_CHAR_THRESHOLD: float = _DEFAULT_TAIL_CHAR_THRESHOLD
_TAIL_SIM_CHANGE_LOW: float = _DEFAULT_TAIL_SIM_CHANGE_LOW

# 特殊号牌白名单: 字符比对判定"一致"但实为换挂的极少数号牌 (如 20260813_211147_3ce87a60 桂BA852).
# 命中条件: 两侧去挂/厂/内后的车挂号序列完全相同且等于名单内号牌(不允许含?未知字符) → 作废字符判定.
_CHAR_CHANGE_WHITELIST: set = {"桂BA852"}

# 评估运行状态（后台线程 + 前端轮询）
_EVAL_STATE: Dict[str, Any] = {
    "running": False,
    "total": 0,
    "processed": 0,
    "success": 0,
    "failed": 0,
    "current_index": 0,
    "current_sample": "",
    "message": "",
    "errors": [],
    "results": [],
    "started_at": None,
    "finished_at": None,
    "run_id": None,
    "metrics": None,
    "avg_lat_ms": None,
    "per_category": None,
}
_EVAL_STATE_LOCK = threading.Lock()


def _validate_threshold_value(name: str, value: Any) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")
    return threshold


def _save_threshold_settings() -> None:
    payload = {
        "head_threshold": _HEAD_THRESHOLD,
        "tail_threshold": _TAIL_THRESHOLD,
        "tail_char_threshold": _TAIL_CHAR_THRESHOLD,
        "tail_sim_change_low": _TAIL_SIM_CHANGE_LOW,
    }
    with open(_THRESHOLDS_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _load_threshold_settings() -> None:
    global _HEAD_THRESHOLD, _TAIL_THRESHOLD, _TAIL_CHAR_THRESHOLD, _TAIL_SIM_CHANGE_LOW

    if not os.path.exists(_THRESHOLDS_FILE):

        return

    try:
        with open(_THRESHOLDS_FILE, "r", encoding="utf-8") as f:
            payload = json.load(f)
        _HEAD_THRESHOLD = _validate_threshold_value(
            "head_threshold",
            payload.get("head_threshold", _DEFAULT_HEAD_THRESHOLD),
        )
        _TAIL_THRESHOLD = _validate_threshold_value(
            "tail_threshold",
            payload.get("tail_threshold", _DEFAULT_TAIL_THRESHOLD),
        )
        _TAIL_CHAR_THRESHOLD = _validate_threshold_value(
            "tail_char_threshold",
            payload.get("tail_char_threshold", _DEFAULT_TAIL_CHAR_THRESHOLD),
        )
        _TAIL_SIM_CHANGE_LOW = _validate_threshold_value(
            "tail_sim_change_low",
            payload.get("tail_sim_change_low", _DEFAULT_TAIL_SIM_CHANGE_LOW),
        )
    except Exception as e:
        print(f"[thresholds] failed to load {_THRESHOLDS_FILE}: {e}")
        _HEAD_THRESHOLD = _DEFAULT_HEAD_THRESHOLD
        _TAIL_THRESHOLD = _DEFAULT_TAIL_THRESHOLD
        _TAIL_CHAR_THRESHOLD = _DEFAULT_TAIL_CHAR_THRESHOLD
        _TAIL_SIM_CHANGE_LOW = _DEFAULT_TAIL_SIM_CHANGE_LOW


_load_threshold_settings()


class _MetricsStore:
    def __init__(self, *, log_dir: str, retention_days: int = 90, recent_max: int = 300) -> None:
        self._lock = threading.Lock()
        self._log_dir = log_dir
        self._retention_days = int(retention_days)
        self._recent_max = int(recent_max)

        self._service_start_ts = time.time()
        self._loaded_history = False

        self._totals: Dict[str, int] = {
            "requests": 0,
            "ok": 0,
            "errors": 0,
            "http_400": 0,
            "http_500": 0,
        }
        self._by_endpoint: Dict[str, Dict[str, Any]] = {}
        self._case_type: Dict[str, int] = {}
        self._recent = deque(maxlen=self._recent_max)
        self._hourly: Dict[str, Dict[str, Any]] = {}
        self._last_cleanup_ts = 0.0

        os.makedirs(self._log_dir, exist_ok=True)

        # 图片存储目录
        self._images_dir = os.path.join(self._log_dir, "images")
        os.makedirs(self._images_dir, exist_ok=True)

        # 受保护记录列表文件
        self._protected_file = os.path.join(self._log_dir, "protected_records.json")
        self._protected_records: set = self._load_protected_records()

    def _now_iso(self) -> str:
        dt = datetime.datetime.now().astimezone()
        return dt.isoformat(timespec="milliseconds")

    def _date_key(self, dt: datetime.datetime) -> str:
        return dt.strftime("%Y%m%d")

    def _hour_key(self, dt: datetime.datetime) -> str:
        return dt.strftime("%Y-%m-%d %H:00")

    def _log_path_for_dt(self, dt: datetime.datetime) -> str:
        fn = f"stats_{self._date_key(dt)}.jsonl"
        return os.path.join(self._log_dir, fn)

    def _load_protected_records(self) -> set:
        """加载受保护的记录ID列表"""
        try:
            if os.path.exists(self._protected_file):
                with open(self._protected_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return set(data.get("protected", []))
        except Exception:
            pass
        return set()

    def _save_protected_records(self) -> None:
        """保存受保护的记录ID列表"""
        try:
            with open(self._protected_file, "w", encoding="utf-8") as f:
                json.dump({"protected": list(self._protected_records)}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _cleanup_old_files(self) -> None:
        now = time.time()
        if now - self._last_cleanup_ts < 600:
            return
        self._last_cleanup_ts = now

        try:
            cutoff = datetime.datetime.now().date() - datetime.timedelta(days=self._retention_days)

            # 清理旧的 jsonl 文件
            for name in os.listdir(self._log_dir):
                if not name.startswith("stats_") or not name.endswith(".jsonl"):
                    continue
                date_part = name[len("stats_"): len("stats_") + 8]
                try:
                    d = datetime.datetime.strptime(date_part, "%Y%m%d").date()
                except Exception:
                    continue
                if d < cutoff:
                    try:
                        os.remove(os.path.join(self._log_dir, name))
                    except Exception:
                        pass

            # 清理旧的图片文件夹
            if os.path.exists(self._images_dir):
                for date_folder in os.listdir(self._images_dir):
                    try:
                        d = datetime.datetime.strptime(date_folder, "%Y%m%d").date()
                    except Exception:
                        continue
                    if d < cutoff:
                        date_path = os.path.join(self._images_dir, date_folder)
                        if os.path.isdir(date_path):
                            # 遍历该日期下的所有记录
                            for record_folder in os.listdir(date_path):
                                record_path = os.path.join(date_path, record_folder)
                                if not os.path.isdir(record_path):
                                    continue

                                # 读取记录元数据
                                meta_file = os.path.join(record_path, "meta.json")
                                try:
                                    with open(meta_file, "r", encoding="utf-8") as f:
                                        meta = json.load(f)

                                    record_id = meta.get("record_id", "")
                                    case_type = meta.get("case_type", "")

                                    # 判断是否可以删除
                                    can_delete = False
                                    if case_type == "normal":
                                        # 正常车辆直接删除
                                        can_delete = True
                                    elif case_type in ["fake_plate", "change_trailer"]:
                                        # 套牌/换挂车检查保护标记
                                        if record_id not in self._protected_records:
                                            can_delete = True
                                    else:
                                        # 其他类型也删除
                                        can_delete = True

                                    if can_delete:
                                        shutil.rmtree(record_path, ignore_errors=True)
                                except Exception:
                                    # 如果无法读取元数据，也删除
                                    shutil.rmtree(record_path, ignore_errors=True)

                            # 如果日期文件夹为空，删除它
                            try:
                                if not os.listdir(date_path):
                                    os.rmdir(date_path)
                            except Exception:
                                pass
        except Exception:
            return

    def _percentile(self, values: list, p: float) -> Optional[float]:
        if not values:
            return None
        if p <= 0:
            return float(min(values))
        if p >= 100:
            return float(max(values))
        s = sorted(values)
        k = int(round((p / 100.0) * (len(s) - 1)))
        k = max(0, min(len(s) - 1, k))
        return float(s[k])

    def _apply_event(self, ev: Dict[str, Any]) -> None:
        endpoint = str(ev.get("endpoint") or "")
        ok = bool(ev.get("ok"))
        http_status = int(ev.get("http_status") or 0)
        case_type = str(ev.get("case_type") or "")
        lat_ms = ev.get("lat_ms")

        self._totals["requests"] += 1
        if ok:
            self._totals["ok"] += 1
        else:
            self._totals["errors"] += 1
        if http_status == 400:
            self._totals["http_400"] += 1
        if http_status >= 500:
            self._totals["http_500"] += 1

        if case_type:
            self._case_type[case_type] = int(self._case_type.get(case_type, 0)) + 1

        ep = self._by_endpoint.get(endpoint)
        if ep is None:
            ep = {"requests": 0, "ok": 0, "errors": 0, "lat_ms": deque(maxlen=3000), "http_400": 0, "http_500": 0}
            self._by_endpoint[endpoint] = ep
        ep["requests"] += 1
        if ok:
            ep["ok"] += 1
        else:
            ep["errors"] += 1
        if http_status == 400:
            ep["http_400"] += 1
        if http_status >= 500:
            ep["http_500"] += 1
        if isinstance(lat_ms, (int, float)):
            ep["lat_ms"].append(float(lat_ms))

        ts = str(ev.get("ts") or "")
        try:
            dt = datetime.datetime.fromisoformat(ts)
        except Exception:
            dt = datetime.datetime.now().astimezone()
        hour_key = self._hour_key(dt)
        hb = self._hourly.get(hour_key)
        if hb is None:
            hb = {"requests": 0, "errors": 0, "case_type": {}}
            self._hourly[hour_key] = hb
        hb["requests"] += 1
        if not ok:
            hb["errors"] += 1
        if case_type:
            ctd = hb["case_type"]
            ctd[case_type] = int(ctd.get(case_type, 0)) + 1

        self._recent.appendleft(ev)

    def _ensure_history_loaded(self) -> None:
        if self._loaded_history:
            return
        with self._lock:
            if self._loaded_history:
                return
            cutoff = datetime.datetime.now().date() - datetime.timedelta(days=self._retention_days)
            try:
                for name in sorted(os.listdir(self._log_dir)):
                    if not name.startswith("stats_") or not name.endswith(".jsonl"):
                        continue
                    date_part = name[len("stats_"): len("stats_") + 8]
                    try:
                        d = datetime.datetime.strptime(date_part, "%Y%m%d").date()
                    except Exception:
                        continue
                    if d < cutoff:
                        continue
                    path = os.path.join(self._log_dir, name)
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    ev = json.loads(line)
                                except Exception:
                                    continue
                                self._apply_event(ev)
                    except Exception:
                        continue
            finally:
                self._loaded_history = True

    def record(self, ev: Dict[str, Any]) -> None:
        self._ensure_history_loaded()
        dt = datetime.datetime.now().astimezone()
        ev = dict(ev)
        ev.setdefault("ts", self._now_iso())

        with self._lock:
            self._apply_event(ev)
            try:
                path = self._log_path_for_dt(dt)
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(ev, ensure_ascii=False) + "\n")
            except Exception:
                pass
            self._cleanup_old_files()

    def snapshot(self) -> Dict[str, Any]:
        self._ensure_history_loaded()
        with self._lock:
            by_endpoint_out: Dict[str, Any] = {}
            for ep, v in self._by_endpoint.items():
                lat_list = list(v.get("lat_ms") or [])
                by_endpoint_out[ep] = {
                    "requests": int(v.get("requests", 0)),
                    "ok": int(v.get("ok", 0)),
                    "errors": int(v.get("errors", 0)),
                    "http_400": int(v.get("http_400", 0)),
                    "http_500": int(v.get("http_500", 0)),
                    "lat_avg_ms": (sum(lat_list) / len(lat_list)) if lat_list else None,
                    "lat_p95_ms": self._percentile(lat_list, 95),
                }

            return {
                "service_start_ts": self._service_start_ts,
                "totals": dict(self._totals),
                "case_type": dict(self._case_type),
                "by_endpoint": by_endpoint_out,
                "recent": list(self._recent),
            }

    def recent(self, n: int = 200) -> Dict[str, Any]:
        self._ensure_history_loaded()
        with self._lock:
            return {"recent": list(self._recent)[: max(0, int(n))]}

    def summary(self, *, days: int = 7) -> Dict[str, Any]:
        self._ensure_history_loaded()
        with self._lock:
            cutoff = datetime.datetime.now().astimezone() - datetime.timedelta(days=int(days))
            out = []
            for k in sorted(self._hourly.keys()):
                try:
                    dt = datetime.datetime.strptime(k, "%Y-%m-%d %H:00").replace(
                        tzinfo=datetime.datetime.now().astimezone().tzinfo)
                except Exception:
                    continue
                if dt < cutoff:
                    continue
                hb = self._hourly[k]
                out.append({
                    "hour": k,
                    "requests": int(hb.get("requests", 0)),
                    "errors": int(hb.get("errors", 0)),
                    "case_type": dict(hb.get("case_type", {})),
                })
            return {"hours": out, "days": int(days)}

    def reset(self) -> Dict[str, Any]:
        """
        重置统计数据，从当前时间重新开始监控

        Returns:
            重置后的状态信息
        """
        with self._lock:
            # 重置服务启动时间
            old_start_ts = self._service_start_ts
            self._service_start_ts = time.time()

            # 重置计数器
            old_totals = dict(self._totals)
            self._totals = {
                "requests": 0,
                "ok": 0,
                "errors": 0,
                "http_400": 0,
                "http_500": 0,
            }

            # 清空分类统计
            old_case_type = dict(self._case_type)
            self._case_type = {}

            # 清空端点统计
            old_by_endpoint = dict(self._by_endpoint)
            self._by_endpoint = {}

            # 清空最近记录
            old_recent_len = len(self._recent)
            self._recent.clear()

            # 清空小时统计
            old_hourly_len = len(self._hourly)
            self._hourly = {}

            return {
                "success": True,
                "message": "统计已重置",
                "old_service_start": datetime.datetime.fromtimestamp(old_start_ts).isoformat(),
                "new_service_start": datetime.datetime.fromtimestamp(self._service_start_ts).isoformat(),
                "cleared_totals": old_totals,
                "cleared_case_types": old_case_type,
                "cleared_endpoints": list(old_by_endpoint.keys()),
                "cleared_recent_count": old_recent_len,
                "cleared_hourly_count": old_hourly_len,
            }

    def save_images(self, record_id: str, previews: Dict[str, str], meta: Dict[str, Any],
                    original_images: Optional[Dict[str, str]] = None) -> Optional[str]:
        """
        保存预览图和原始图到磁盘

        Args:
            record_id: 记录唯一ID
            previews: 包含6张处理后图片的data URL字典
            meta: 记录元数据
            original_images: 包含2张原始图片的data URL字典（可选）

        Returns:
            图片目录路径，失败返回None
        """
        try:
            dt = datetime.datetime.now()
            date_folder = self._date_key(dt)

            # 创建日期文件夹
            date_path = os.path.join(self._images_dir, date_folder)
            os.makedirs(date_path, exist_ok=True)

            # 创建记录文件夹
            record_path = os.path.join(date_path, record_id)
            os.makedirs(record_path, exist_ok=True)

            # 保存6张处理后的图片 (未遮挡车辆裁剪图不再保存)
            for key in ["vehicle1", "vehicle2", "head1", "head2", "tail1", "tail2"]:
                data_url = previews.get(key, "")
                if not data_url or not data_url.startswith("data:image/"):
                    continue

                try:
                    # 解析 data URL
                    header, encoded = data_url.split(",", 1)
                    img_data = base64.b64decode(encoded)

                    # 保存图片
                    img_path = os.path.join(record_path, f"{key}.jpg")
                    with open(img_path, "wb") as f:
                        f.write(img_data)
                except Exception:
                    continue

            # 保存原始图片和尾部视角裁切图（如果提供）
            if original_images:
                def _save_img(key: str, data_url: str) -> None:
                    if not data_url or not data_url.startswith("data:image/"):
                        return
                    try:
                        header, encoded = data_url.split(",", 1)
                        img_data = base64.b64decode(encoded)
                        img_path = os.path.join(record_path, f"{key}.jpg")
                        with open(img_path, "wb") as f:
                            f.write(img_data)
                    except Exception:
                        pass

                for key in ["original1", "original2", "original3", "original4"]:
                    _save_img(key, original_images.get(key, ""))

                # 尾部视角: 检测到车挂号/放大号只保存框选图(boxed), 无框时才保存原裁剪图
                for idx in ("3", "4"):
                    boxed_url = original_images.get(f"tail_view_crop{idx}_boxed", "")
                    if boxed_url.startswith("data:image/"):
                        _save_img(f"tail_view_crop{idx}_boxed", boxed_url)
                    else:
                        _save_img(f"tail_view_crop{idx}", original_images.get(f"tail_view_crop{idx}", ""))

            # 保存元数据
            meta_path = os.path.join(record_path, "meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            return record_path
        except Exception:
            return None

    def query_records(
            self,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            case_type: Optional[str] = None,
            time_filter: Optional[str] = None,
            review_filter: Optional[str] = None,
            judge_mode: Optional[str] = None,
            limit: int = 50,
            offset: int = 0
    ) -> Dict[str, Any]:
        """
        查询记录列表

        Args:
            start_date: 开始日期 YYYY-MM-DD
            end_date: 结束日期 YYYY-MM-DD
            case_type: 类型筛选 normal/fake_plate/change_trailer/all
            time_filter: 耗时筛选 lt3/3to60/60to150/gt150/all
            review_filter: 复核筛选 reviewed/unreviewed/all
            judge_mode: 判定模式筛选 头部车辆裁剪/尾部字符检测/ai判断/阈值兜底/all
            limit: 返回条数
            offset: 偏移量

        Returns:
            包含记录列表和总数的字典
        """
        self._ensure_history_loaded()

        try:
            # 解析日期范围
            if start_date:
                start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
            else:
                start_dt = datetime.datetime.now().date() - datetime.timedelta(days=7)

            if end_date:
                end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
            else:
                end_dt = datetime.datetime.now().date()

            # 收集所有符合条件的记录
            records = []
            current_date = start_dt
            while current_date <= end_dt:
                date_key = current_date.strftime("%Y%m%d")
                log_path = os.path.join(self._log_dir, f"stats_{date_key}.jsonl")

                if os.path.exists(log_path):
                    with open(log_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                record = json.loads(line)

                                # 筛选条件 - 不显示已删除记录
                                if record.get("deleted", False):
                                    continue

                                if case_type and case_type != "all":
                                    if record.get("case_type") != case_type:
                                        continue

                                # 判定模式筛选
                                if judge_mode and judge_mode != "all":
                                    if _derive_judge_mode(record) != judge_mode:
                                        continue

                                # 耗时筛选
                                if time_filter:
                                    lat_ms = record.get("lat_ms")
                                    if lat_ms is not None:
                                        try:
                                            lat_val = float(lat_ms)
                                            if time_filter == "lt3" and lat_val/1000 >= 3:
                                                continue
                                            elif time_filter == "3to60" and (lat_val/1000 < 3 or lat_val/1000 >= 60):
                                                continue
                                            elif time_filter == "60to150" and (lat_val/1000 < 60 or lat_val/1000 >= 150):
                                                continue
                                            elif time_filter == "gt150" and lat_val/1000 <= 150:
                                                continue
                                        except (ValueError, TypeError):
                                            continue

                                # 复核筛选
                                if review_filter:
                                    reviewed = record.get("reviewed", False)
                                    if review_filter == "reviewed" and not reviewed:
                                        continue
                                    elif review_filter == "unreviewed" and reviewed:
                                        continue

                                # 只保留有 record_id 的记录（有图片的）
                                if "record_id" in record:
                                    records.append(record)
                            except Exception:
                                continue

                current_date += datetime.timedelta(days=1)

            # 按时间倒序排序
            records.sort(key=lambda x: x.get("ts", ""), reverse=True)

            # 分页
            total = len(records)
            records = records[offset:offset + limit]

            return {
                "records": records,
                "total": total,
                "limit": limit,
                "offset": offset
            }
        except Exception as e:
            return {
                "records": [],
                "total": 0,
                "limit": limit,
                "offset": offset,
                "error": str(e)
            }

    def get_record(self, record_id: str) -> Optional[Dict[str, Any]]:
        """获取单条记录详情"""
        try:
            # 从 record_id 中提取日期
            date_part = record_id.split("_")[0]
            log_path = os.path.join(self._log_dir, f"stats_{date_part}.jsonl")

            if not os.path.exists(log_path):
                return None

            with open(log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if record.get("record_id") == record_id:
                            return record
                    except Exception:
                        continue

            return None
        except Exception:
            return None

    def delete_record(self, record_id: str, hard_delete: bool = False) -> Tuple[bool, str]:
        """
        删除记录

        Args:
            record_id: 记录ID
            hard_delete: 是否硬删除（彻底删除文件）

        Returns:
            (成功, 消息)
        """
        try:
            # 获取记录
            record = self.get_record(record_id)
            if not record:
                return False, "记录不存在"

            # 检查是否允许删除
            case_type = record.get("case_type", "")
            if case_type == "normal":
                return False, "正常车辆记录由系统自动清理，无需手动删除"

            if case_type not in ["fake_plate", "change_trailer"]:
                return False, f"不支持删除类型: {case_type}"

            if hard_delete:
                # 硬删除：删除图片文件夹
                image_dir = record.get("image_dir", "")
                if image_dir and os.path.exists(image_dir):
                    shutil.rmtree(image_dir, ignore_errors=True)

                # 从 jsonl 中删除（标记为已删除）
                date_part = record_id.split("_")[0]
                log_path = os.path.join(self._log_dir, f"stats_{date_part}.jsonl")

                if os.path.exists(log_path):
                    lines = []
                    with open(log_path, "r", encoding="utf-8") as f:
                        for line in f:
                            try:
                                r = json.loads(line.strip())
                                if r.get("record_id") != record_id:
                                    lines.append(line)
                            except Exception:
                                lines.append(line)

                    with open(log_path, "w", encoding="utf-8") as f:
                        f.writelines(lines)

                # 从保护列表中移除
                if record_id in self._protected_records:
                    self._protected_records.remove(record_id)
                    self._save_protected_records()

                return True, "记录已彻底删除"
            else:
                # 软删除：只标记
                date_part = record_id.split("_")[0]
                log_path = os.path.join(self._log_dir, f"stats_{date_part}.jsonl")

                if os.path.exists(log_path):
                    lines = []
                    with open(log_path, "r", encoding="utf-8") as f:
                        for line in f:
                            try:
                                r = json.loads(line.strip())
                                if r.get("record_id") == record_id:
                                    r["deleted"] = True
                                    lines.append(json.dumps(r, ensure_ascii=False) + "\n")
                                else:
                                    lines.append(line)
                            except Exception:
                                lines.append(line)

                    with open(log_path, "w", encoding="utf-8") as f:
                        f.writelines(lines)

                return True, "记录已标记为删除"
        except Exception as e:
            return False, f"删除失败: {str(e)}"

    def protect_record(self, record_id: str, protected: bool, note: str = "") -> Tuple[bool, str]:
        """
        设置记录保护状态

        Args:
            record_id: 记录ID
            protected: 是否保护
            note: 备注信息

        Returns:
            (成功, 消息)
        """
        try:
            # 获取记录
            record = self.get_record(record_id)
            if not record:
                return False, "记录不存在"

            # 更新保护状态
            date_part = record_id.split("_")[0]
            log_path = os.path.join(self._log_dir, f"stats_{date_part}.jsonl")

            if os.path.exists(log_path):
                lines = []
                with open(log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            r = json.loads(line.strip())
                            if r.get("record_id") == record_id:
                                r["protected"] = protected
                                if note:
                                    r["note"] = note
                                lines.append(json.dumps(r, ensure_ascii=False) + "\n")
                            else:
                                lines.append(line)
                        except Exception:
                            lines.append(line)

                with open(log_path, "w", encoding="utf-8") as f:
                    f.writelines(lines)

            # 更新保护列表
            if protected:
                self._protected_records.add(record_id)
            else:
                self._protected_records.discard(record_id)
            self._save_protected_records()

            # 更新元数据文件
            image_dir = record.get("image_dir", "")
            if image_dir and os.path.exists(image_dir):
                meta_file = os.path.join(image_dir, "meta.json")
                if os.path.exists(meta_file):
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    meta["protected"] = protected
                    if note:
                        meta["note"] = note
                    with open(meta_file, "w", encoding="utf-8") as f:
                        json.dump(meta, f, ensure_ascii=False, indent=2)

            return True, f"已{'设置保护' if protected else '取消保护'}"
        except Exception as e:
            return False, f"操作失败: {str(e)}"

    def review_record(self, record_id: str, reviewed_case_type: str, review_reason: str,
                      reviewed_by: str, review_confidence: str = "medium") -> Tuple[bool, str]:
        """
        提交复核结果

        Args:
            record_id: 记录ID
            reviewed_case_type: 复核后的类型
            review_reason: 复核理由
            reviewed_by: 复核人员
            review_confidence: 置信度 high/medium/low

        Returns:
            (成功, 消息)
        """
        try:
            # 获取记录
            record = self.get_record(record_id)
            if not record:
                return False, "记录不存在"

            # 验证复核类型
            valid_types = ["normal", "fake_plate", "change_trailer"]
            if reviewed_case_type not in valid_types:
                return False, f"无效的复核类型: {reviewed_case_type}"

            # 准备复核信息
            review_data = {
                "reviewed": True,
                "reviewed_at": datetime.datetime.now().isoformat(timespec="milliseconds"),
                "reviewed_by": reviewed_by,
                "reviewed_case_type": reviewed_case_type,
                "review_reason": review_reason,
                "review_confidence": review_confidence
            }

            # 保存复核历史
            review_history = record.get("review_history", [])
            review_history.append(review_data.copy())

            # 更新记录
            date_part = record_id.split("_")[0]
            log_path = os.path.join(self._log_dir, f"stats_{date_part}.jsonl")

            if os.path.exists(log_path):
                lines = []
                with open(log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            r = json.loads(line.strip())
                            if r.get("record_id") == record_id:
                                r.update(review_data)
                                r["review_history"] = review_history
                                lines.append(json.dumps(r, ensure_ascii=False) + "\n")
                            else:
                                lines.append(line)
                        except Exception:
                            lines.append(line)

                with open(log_path, "w", encoding="utf-8") as f:
                    f.writelines(lines)

            # 更新元数据文件
            image_dir = record.get("image_dir", "")
            if image_dir and os.path.exists(image_dir):
                meta_file = os.path.join(image_dir, "meta.json")
                if os.path.exists(meta_file):
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    meta.update(review_data)
                    meta["review_history"] = review_history
                    with open(meta_file, "w", encoding="utf-8") as f:
                        json.dump(meta, f, ensure_ascii=False, indent=2)

            return True, "复核结果已保存"
        except Exception as e:
            return False, f"操作失败: {str(e)}"

    def revoke_review(self, record_id: str) -> Tuple[bool, str]:
        """
        撤销复核

        Args:
            record_id: 记录ID

        Returns:
            (成功, 消息)
        """
        try:
            # 获取记录
            record = self.get_record(record_id)
            if not record:
                return False, "记录不存在"

            if not record.get("reviewed", False):
                return False, "该记录未复核"

            # 移除复核字段
            date_part = record_id.split("_")[0]
            log_path = os.path.join(self._log_dir, f"stats_{date_part}.jsonl")

            if os.path.exists(log_path):
                lines = []
                with open(log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            r = json.loads(line.strip())
                            if r.get("record_id") == record_id:
                                r["reviewed"] = False
                                r.pop("reviewed_at", None)
                                r.pop("reviewed_by", None)
                                r.pop("reviewed_case_type", None)
                                r.pop("review_reason", None)
                                r.pop("review_confidence", None)
                                lines.append(json.dumps(r, ensure_ascii=False) + "\n")
                            else:
                                lines.append(line)
                        except Exception:
                            lines.append(line)

                with open(log_path, "w", encoding="utf-8") as f:
                    f.writelines(lines)

            # 更新元数据文件
            image_dir = record.get("image_dir", "")
            if image_dir and os.path.exists(image_dir):
                meta_file = os.path.join(image_dir, "meta.json")
                if os.path.exists(meta_file):
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    meta["reviewed"] = False
                    meta.pop("reviewed_at", None)
                    meta.pop("reviewed_by", None)
                    meta.pop("reviewed_case_type", None)
                    meta.pop("review_reason", None)
                    meta.pop("review_confidence", None)
                    with open(meta_file, "w", encoding="utf-8") as f:
                        json.dump(meta, f, ensure_ascii=False, indent=2)

            return True, "已撤销复核"
        except Exception as e:
            return False, f"操作失败: {str(e)}"

    def get_review_stats(self) -> Dict[str, Any]:
        """获取复核统计"""
        try:
            stats = {
                "total_records": 0,
                "reviewed_count": 0,
                "review_rate": 0.0,
                "accuracy": {
                    "confirmed": 0,
                    "corrected": 0
                },
                "by_type": {}
            }

            # 遍历所有记录
            cutoff = datetime.datetime.now().date() - datetime.timedelta(days=self._retention_days)

            for name in os.listdir(self._log_dir):
                if not name.startswith("stats_") or not name.endswith(".jsonl"):
                    continue

                date_part = name[len("stats_"):len("stats_") + 8]
                try:
                    d = datetime.datetime.strptime(date_part, "%Y%m%d").date()
                except Exception:
                    continue

                if d < cutoff:
                    continue

                log_path = os.path.join(self._log_dir, name)
                with open(log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                            if "record_id" not in record:
                                continue

                            case_type = record.get("case_type", "")
                            if not case_type or case_type == "abnormal":
                                continue

                            stats["total_records"] += 1

                            # 初始化类型统计
                            if case_type not in stats["by_type"]:
                                stats["by_type"][case_type] = {
                                    "total": 0,
                                    "reviewed": 0,
                                    "confirmed": 0,
                                    "corrected": 0,
                                    "corrections": {}
                                }

                            stats["by_type"][case_type]["total"] += 1

                            # 复核统计
                            if record.get("reviewed", False):
                                stats["reviewed_count"] += 1
                                stats["by_type"][case_type]["reviewed"] += 1

                                reviewed_type = record.get("reviewed_case_type", "")
                                if reviewed_type == case_type:
                                    # 确认
                                    stats["accuracy"]["confirmed"] += 1
                                    stats["by_type"][case_type]["confirmed"] += 1
                                else:
                                    # 修正
                                    stats["accuracy"]["corrected"] += 1
                                    stats["by_type"][case_type]["corrected"] += 1

                                    # 记录修正流向
                                    if reviewed_type not in stats["by_type"][case_type]["corrections"]:
                                        stats["by_type"][case_type]["corrections"][reviewed_type] = 0
                                    stats["by_type"][case_type]["corrections"][reviewed_type] += 1
                        except Exception:
                            continue

            # 计算复核率
            if stats["total_records"] > 0:
                stats["review_rate"] = stats["reviewed_count"] / stats["total_records"]

            return stats
        except Exception as e:
            return {"error": str(e)}


_METRICS = _MetricsStore(
    log_dir=os.path.join(os.path.dirname(__file__), "stats_logs"),
    retention_days=90,
    recent_max=300,
)


class RecordExporter:
    """记录导出器"""

    def __init__(self, metrics_store: _MetricsStore, export_base_dir: str = None):
        self.metrics = metrics_store
        if export_base_dir is None:
            export_base_dir = os.path.join(os.path.dirname(__file__), "exports")
        self.export_base_dir = export_base_dir
        os.makedirs(self.export_base_dir, exist_ok=True)

    def export_single(
            self,
            record_id: str,
            export_path: Optional[str] = None,
            image_types: Optional[List[str]] = None
    ) -> Tuple[bool, str, Optional[str]]:
        """
        导出单条记录

        Args:
            record_id: 记录ID
            export_path: 导出路径（可选）
            image_types: 要导出的图片类型列表，如 ["original1", "original2", "vehicle1", "vehicle2", "head1", "head2", "tail1", "tail2"]
                        如果为None，则导出所有图片

        Returns:
            (成功, 消息, 导出路径)
        """
        try:
            # 获取记录
            record = self.metrics.get_record(record_id)
            if not record:
                return False, "记录不存在", None

            # 获取图片目录
            image_dir = record.get("image_dir", "")
            if not image_dir or not os.path.exists(image_dir):
                return False, "图片目录不存在", None

            # 确定导出路径
            if export_path is None:
                export_path = self.export_base_dir

            # 创建导出文件夹
            case_type = record.get("case_type", "unknown")
            folder_name = f"{record_id}_{case_type}"
            export_folder = os.path.join(export_path, folder_name)
            os.makedirs(export_folder, exist_ok=True)

            # 确定要导出的图片类型
            if image_types is None:
                # 默认导出所有图片
                image_types = ["original1", "original2", "original3", "original4",
                               "tail_view_crop3", "tail_view_crop4",
                               "tail_view_crop3_boxed", "tail_view_crop4_boxed",
                               "vehicle1", "vehicle2",
                               "head1", "head2", "tail1", "tail2"]

            normalized: List[str] = []
            for it in image_types:
                if isinstance(it, dict):
                    it = it.get("value")
                if not it:
                    continue
                if not isinstance(it, str):
                    continue
                it = os.path.basename(it.strip())
                if not it:
                    continue
                if it.lower().endswith(".jpg"):
                    normalized.append(it[:-4])
                else:
                    normalized.append(it)
            image_types = normalized

            required_processed = {"vehicle1", "vehicle2", "head1", "head2", "tail1", "tail2"}
            if not required_processed.intersection(set(image_types)):
                image_types.extend(sorted(required_processed))

            copied_files: List[str] = []

            want = set(image_types)
            all_jpgs: List[str] = []
            try:
                for fn in os.listdir(image_dir):
                    if not isinstance(fn, str):
                        continue
                    if fn.lower().endswith(".jpg"):
                        all_jpgs.append(fn)
            except Exception:
                all_jpgs = []

            selected_jpgs: List[str] = []
            for fn in all_jpgs:
                base, ext = os.path.splitext(fn)
                if base in want:
                    selected_jpgs.append(fn)

            if not selected_jpgs:
                selected_jpgs = all_jpgs

            for fn in selected_jpgs:
                src_path = os.path.join(image_dir, fn)
                if not os.path.exists(src_path):
                    continue
                dst_path = os.path.join(export_folder, fn)
                try:
                    shutil.copy2(src_path, dst_path)
                    copied_files.append(fn)
                except Exception:
                    continue

            meta_src = os.path.join(image_dir, "meta.json")
            if os.path.exists(meta_src):
                try:
                    shutil.copy2(meta_src, os.path.join(export_folder, "meta.json"))
                    copied_files.append("meta.json")
                except Exception:
                    pass

            # 生成信息文件
            info_path = os.path.join(export_folder, "info.txt")
            with open(info_path, "w", encoding="utf-8") as f:
                f.write(f"记录ID: {record_id}\n")
                f.write(f"时间: {record.get('ts', '')}\n")
                f.write(f"系统判定: {record.get('case_type', '')}\n")
                f.write(f"车头相似度: {record.get('head_prob', 'N/A')}\n")
                f.write(f"车尾相似度: {record.get('tail_prob', 'N/A')}\n")
                f.write(f"输入路径1: {record.get('input_path1', '')}\n")
                f.write(f"输入路径2: {record.get('input_path2', '')}\n")
                f.write(f"输入路径3: {record.get('input_path3', '')}\n")
                f.write(f"输入路径4: {record.get('input_path4', '')}\n")
                f.write(f"输入模式: {record.get('input_mode', '')}\n")
                f.write(f"尾部AI模式: {record.get('tail_ai_mode', '')}\n")
                f.write(f"原方案结果: {record.get('stage1_case_type', '')}\n")
                f.write(f"3/4视角优先判定: {record.get('tail_second_check_result', '')}\n")
                f.write(f"车头AI依据: {record.get('ai_head_reason', '')}\n")
                f.write(f"主视角尾部依据: {record.get('ai_tail_reason', '')}\n")
                f.write(f"尾牌编号一致性: {record.get('tail_number_consistency', '')}\n")
                f.write(f"尾牌结构一致性: {record.get('tail_structure_consistency', '')}\n")

                # 如果有复核信息
                if record.get('reviewed'):
                    f.write(f"\n--- 复核信息 ---\n")
                    f.write(f"复核结果: {record.get('reviewed_case_type', '')}\n")
                    f.write(f"复核人员: {record.get('reviewed_by', '')}\n")
                    f.write(f"复核时间: {record.get('reviewed_at', '')}\n")
                    f.write(f"复核理由: {record.get('review_reason', '')}\n")

                f.write(f"\n导出时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                exported_count = len([x for x in copied_files if isinstance(x, str) and x.lower().endswith('.jpg')])
                f.write(f"导出图片数: {exported_count}\n")
                f.write(f"导出文件: {', '.join(copied_files)}\n")

            exported_count = len([x for x in copied_files if isinstance(x, str) and x.lower().endswith('.jpg')])
            if exported_count == 0:
                return False, "导出文件夹已创建，但未找到可导出的图片（请确认记录图片目录存在且包含 .jpg 文件）", export_folder
            return True, f"成功导出 {exported_count} 张图片: {', '.join([x for x in copied_files if isinstance(x, str) and x.lower().endswith('.jpg')])}", export_folder
        except Exception as e:
            return False, f"导出失败: {str(e)}", None

    def export_batch(
            self,
            record_ids: List[str],
            export_path: Optional[str] = None,
            group_by: str = "case_type",
            image_types: Optional[List[str]] = None,
            include_summary: bool = True
    ) -> Tuple[bool, str, Optional[str]]:
        """
        批量导出记录

        Args:
            record_ids: 记录ID列表
            export_path: 导出路径
            group_by: 分组方式 ("case_type" 或 "none")
            image_types: 要导出的图片类型
            include_summary: 是否生成汇总文件

        Returns:
            (成功, 消息, 导出路径)
        """
        try:
            if not record_ids:
                return False, "没有要导出的记录", None

            # 创建导出任务文件夹
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            task_folder = f"export_{timestamp}"
            if export_path is None:
                export_path = self.export_base_dir
            export_folder = os.path.join(export_path, task_folder)
            os.makedirs(export_folder, exist_ok=True)

            # 导出记录
            results = []
            for record_id in record_ids:
                record = self.metrics.get_record(record_id)
                if not record:
                    results.append({
                        "record_id": record_id,
                        "success": False,
                        "message": "记录不存在"
                    })
                    continue

                # 确定子文件夹
                if group_by == "case_type":
                    case_type = record.get("case_type", "unknown")
                    sub_folder = os.path.join(export_folder, case_type)
                else:
                    sub_folder = export_folder

                os.makedirs(sub_folder, exist_ok=True)

                # 导出单条记录
                success, message, _ = self.export_single(
                    record_id,
                    sub_folder,
                    image_types
                )

                results.append({
                    "record_id": record_id,
                    "success": success,
                    "message": message,
                    "case_type": record.get("case_type", ""),
                    "head_prob": record.get("head_prob"),
                    "tail_prob": record.get("tail_prob"),
                    "ts": record.get("ts", "")
                })

            # 生成汇总文件
            if include_summary:
                self._generate_summary_csv(results, export_folder)
                self._generate_export_log(results, export_folder, image_types)

            success_count = sum(1 for r in results if r["success"])
            return True, f"成功导出 {success_count}/{len(record_ids)} 条记录", export_folder
        except Exception as e:
            return False, f"批量导出失败: {str(e)}", None

    def _generate_summary_csv(self, results: List[Dict], export_folder: str):
        """生成汇总CSV文件"""
        try:
            csv_path = os.path.join(export_folder, "export_summary.csv")
            with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
                import csv
                writer = csv.writer(f)
                writer.writerow([
                    "记录ID", "时间", "系统判定", "车头相似度", "车尾相似度",
                    "导出状态", "备注"
                ])

                for r in results:
                    writer.writerow([
                        r.get("record_id", ""),
                        r.get("ts", ""),
                        r.get("case_type", ""),
                        r.get("head_prob", ""),
                        r.get("tail_prob", ""),
                        "成功" if r.get("success") else "失败",
                        r.get("message", "")
                    ])
        except Exception:
            pass

    def _generate_export_log(self, results: List[Dict], export_folder: str, image_types: Optional[List[str]]):
        """生成导出日志"""
        try:
            log_path = os.path.join(export_folder, "export_log.txt")
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("=" * 60 + "\n")
                f.write("导出日志\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"导出时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"总记录数: {len(results)}\n")
                f.write(f"成功: {sum(1 for r in results if r['success'])}\n")
                f.write(f"失败: {sum(1 for r in results if not r['success'])}\n")

                if image_types:
                    f.write(f"\n导出图片类型: {', '.join(image_types)}\n")
                else:
                    f.write(f"\n导出图片类型: 全部\n")

                f.write("\n" + "=" * 60 + "\n")
                f.write("详细结果\n")
                f.write("=" * 60 + "\n\n")

                for r in results:
                    status = "✓" if r["success"] else "✗"
                    f.write(f"{status} {r['record_id']} - {r['message']}\n")
        except Exception:
            pass


_EXPORTER = RecordExporter(_METRICS)


class RecordExporterLegacy:
    """记录导出器"""

    def __init__(self, export_base_dir: str = None):
        if export_base_dir is None:
            export_base_dir = os.path.join(os.path.dirname(__file__), "exports")
        self.export_base_dir = export_base_dir
        os.makedirs(self.export_base_dir, exist_ok=True)

    def export_single(self, record_id: str, export_path: str = None,
                      include_meta: bool = True) -> Tuple[bool, str, Optional[str]]:
        """
        导出单条记录

        Args:
            record_id: 记录ID
            export_path: 导出路径（可选）
            include_meta: 是否包含元数据文件

        Returns:
            (成功, 消息, 导出路径)
        """
        try:
            # 获取记录
            record = _METRICS.get_record(record_id)
            if not record:
                return False, "记录不存在", None

            # 获取图片目录
            image_dir = record.get("image_dir", "")
            if not image_dir or not os.path.exists(image_dir):
                return False, "图片目录不存在", None

            # 确定导出路径
            if export_path is None:
                export_path = self.export_base_dir

            case_type = record.get("case_type", "unknown")
            folder_name = f"{record_id}_{case_type}"
            target_dir = os.path.join(export_path, folder_name)
            os.makedirs(target_dir, exist_ok=True)

            # 复制图片
            image_files = ["vehicle1.jpg", "vehicle2.jpg", "head1.jpg",
                           "head2.jpg", "tail1.jpg", "tail2.jpg"]
            copied_count = 0

            for img_name in image_files:
                src = os.path.join(image_dir, img_name)
                if os.path.exists(src):
                    dst = os.path.join(target_dir, img_name)
                    shutil.copy2(src, dst)
                    copied_count += 1

            # 生成元数据文件
            if include_meta:
                info_path = os.path.join(target_dir, "info.txt")
                with open(info_path, "w", encoding="utf-8") as f:
                    f.write(f"记录ID: {record_id}\n")
                    f.write(f"检测时间: {record.get('ts', '')}\n")
                    f.write(f"系统判定: {case_type}\n")
                    f.write(f"车头相似度: {record.get('head_prob', 'N/A')}\n")
                    f.write(f"车尾相似度: {record.get('tail_prob', 'N/A')}\n")
                    f.write(f"输入路径1: {record.get('input_path1', '')}\n")
                    f.write(f"输入路径2: {record.get('input_path2', '')}\n")
                    f.write(f"输入路径3: {record.get('input_path3', '')}\n")
                    f.write(f"输入路径4: {record.get('input_path4', '')}\n")
                    f.write(f"输入模式: {record.get('input_mode', '')}\n")
                    f.write(f"尾部AI模式: {record.get('tail_ai_mode', '')}\n")
                    f.write(f"原方案结果: {record.get('stage1_case_type', '')}\n")
                    f.write(f"3/4视角优先判定: {record.get('tail_second_check_result', '')}\n")
                    f.write(f"车头AI依据: {record.get('ai_head_reason', '')}\n")
                    f.write(f"主视角尾部依据: {record.get('ai_tail_reason', '')}\n")
                    f.write(f"尾牌编号一致性: {record.get('tail_number_consistency', '')}\n")
                    f.write(f"尾牌结构一致性: {record.get('tail_structure_consistency', '')}\n")

                    # 复核信息
                    if record.get("reviewed", False):
                        f.write(f"\n--- 复核信息 ---\n")
                        f.write(f"复核结果: {record.get('reviewed_case_type', '')}\n")
                        f.write(f"复核人员: {record.get('reviewed_by', '')}\n")
                        f.write(f"复核时间: {record.get('reviewed_at', '')}\n")
                        f.write(f"复核理由: {record.get('review_reason', '')}\n")

                    if record.get("note"):
                        f.write(f"\n备注: {record.get('note')}\n")

            return True, f"已导出 {copied_count} 个文件", target_dir
        except Exception as e:
            return False, f"导出失败: {str(e)}", None

    def export_batch(self, record_ids: List[str], export_path: str = None,
                     group_by: str = "case_type", image_types: Optional[List[str]] = None,
                     include_summary: bool = True) -> Tuple[bool, str, Optional[str]]:
        """
        批量导出记录

        Args:
            record_ids: 记录ID列表
            export_path: 导出路径（可选）
            group_by: 分组方式 case_type/date/none
            image_types: 要导出的图片类型列表（可选，None表示导出全部）
            include_summary: 是否生成汇总文件

        Returns:
            (成功, 消息, 导出路径)
        """
        try:
            if not record_ids:
                return False, "没有要导出的记录", None

            # 创建导出任务文件夹
            if export_path is None:
                export_path = self.export_base_dir

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            task_folder = f"export_{timestamp}"
            task_dir = os.path.join(export_path, task_folder)
            os.makedirs(task_dir, exist_ok=True)

            # 导出记录
            exported_records = []
            failed_records = []

            for record_id in record_ids:
                try:
                    record = _METRICS.get_record(record_id)
                    if not record:
                        failed_records.append({"record_id": record_id, "error": "记录不存在"})
                        continue

                    # 确定目标目录
                    if group_by == "case_type":
                        case_type = record.get("case_type", "unknown")
                        target_base = os.path.join(task_dir, case_type)
                    elif group_by == "date":
                        date_str = record_id.split("_")[0]
                        target_base = os.path.join(task_dir, date_str)
                    else:
                        target_base = task_dir

                    os.makedirs(target_base, exist_ok=True)

                    # 创建记录文件夹
                    record_folder = os.path.join(target_base, record_id)
                    os.makedirs(record_folder, exist_ok=True)

                    image_dir = record.get("image_dir", "")
                    has_saved_images = image_dir and os.path.exists(image_dir)
                    copied_images = 0

                    if has_saved_images:
                        # 方案1: 从已保存的图片目录复制（新记录）
                        if image_types is None:
                            # 导出全部图片
                            image_files = ["vehicle1.jpg", "vehicle2.jpg", "head1.jpg",
                                           "head2.jpg", "tail1.jpg", "tail2.jpg",
                                           "original1.jpg", "original2.jpg",
                                           "original3.jpg", "original4.jpg",
                                           "tail_view_crop3.jpg", "tail_view_crop4.jpg"]
                        else:
                            # 根据指定类型导出
                            image_files = []
                            if "vehicle" in image_types:
                                image_files.extend(["vehicle1.jpg", "vehicle2.jpg"])
                            if "head" in image_types:
                                image_files.extend(["head1.jpg", "head2.jpg"])
                            if "tail" in image_types:
                                image_files.extend(["tail1.jpg", "tail2.jpg"])
                            if "original" in image_types:
                                image_files.extend([
                                    "original1.jpg", "original2.jpg",
                                    "original3.jpg", "original4.jpg",
                                    "tail_view_crop3.jpg", "tail_view_crop4.jpg",
                                ])

                        for img_name in image_files:
                            src = os.path.join(image_dir, img_name)
                            if os.path.exists(src):
                                dst = os.path.join(record_folder, img_name)
                                shutil.copy2(src, dst)
                                copied_images += 1

                        # 复制元数据
                        meta_src = os.path.join(image_dir, "meta.json")
                        if os.path.exists(meta_src):
                            meta_dst = os.path.join(record_folder, "meta.json")
                            shutil.copy2(meta_src, meta_dst)

                    # 无论新旧记录，都尝试复制原始图片（如果还存在的话）
                    input_path1 = record.get("input_path1", "")
                    input_path2 = record.get("input_path2", "")
                    input_path3 = record.get("input_path3", "")
                    input_path4 = record.get("input_path4", "")

                    if input_path1 and os.path.exists(input_path1):
                        # 如果已经有original1.jpg就不重复复制
                        original1_path = os.path.join(record_folder, "original1.jpg")
                        if not os.path.exists(original1_path):
                            try:
                                shutil.copy2(input_path1, original1_path)
                                copied_images += 1
                            except Exception:
                                pass

                    if input_path2 and os.path.exists(input_path2):
                        # 如果已经有original2.jpg就不重复复制
                        original2_path = os.path.join(record_folder, "original2.jpg")
                        if not os.path.exists(original2_path):
                            try:
                                shutil.copy2(input_path2, original2_path)
                                copied_images += 1
                            except Exception:
                                pass

                    # 生成info.txt
                    info_path = os.path.join(record_folder, "info.txt")
                    with open(info_path, "w", encoding="utf-8") as f:
                        f.write(f"记录ID: {record_id}\n")
                        f.write(f"检测时间: {record.get('ts', '')}\n")
                        f.write(f"系统判定: {record.get('case_type', '')}\n")
                        f.write(f"车头相似度: {record.get('head_prob', 'N/A')}\n")
                        f.write(f"车尾相似度: {record.get('tail_prob', 'N/A')}\n")
                        f.write(f"输入路径1: {input_path1}\n")
                        f.write(f"输入路径2: {input_path2}\n")
                        f.write(f"输入路径3: {input_path3}\n")
                        f.write(f"输入路径4: {input_path4}\n")
                        f.write(f"输入模式: {record.get('input_mode', '')}\n")
                        f.write(f"尾部AI模式: {record.get('tail_ai_mode', '')}\n")
                        f.write(f"原方案结果: {record.get('stage1_case_type', '')}\n")
                        f.write(f"3/4视角优先判定: {record.get('tail_second_check_result', '')}\n")
                        f.write(f"车头AI依据: {record.get('ai_head_reason', '')}\n")
                        f.write(f"主视角尾部依据: {record.get('ai_tail_reason', '')}\n")
                        f.write(f"尾牌编号一致性: {record.get('tail_number_consistency', '')}\n")
                        f.write(f"尾牌结构一致性: {record.get('tail_structure_consistency', '')}\n")
                        f.write(f"导出图片数: {copied_images}\n")

                        # 复核信息
                        if record.get("reviewed", False):
                            f.write(f"\n--- 复核信息 ---\n")
                            f.write(f"复核结果: {record.get('reviewed_case_type', '')}\n")
                            f.write(f"复核人员: {record.get('reviewed_by', '')}\n")
                            f.write(f"复核时间: {record.get('reviewed_at', '')}\n")
                            f.write(f"复核理由: {record.get('review_reason', '')}\n")

                        if record.get("note"):
                            f.write(f"\n备注: {record.get('note')}\n")

                    if copied_images == 0:
                        failed_records.append({"record_id": record_id, "error": "没有找到任何图片文件"})
                    else:
                        exported_records.append(record)

                except Exception as e:
                    failed_records.append({"record_id": record_id, "error": str(e)})

            # 生成汇总文件
            if include_summary and exported_records:
                self._generate_summary_csv(exported_records, task_dir)
                self._generate_export_log(exported_records, failed_records, task_dir)

            # 生成结果消息
            msg = f"成功导出 {len(exported_records)} 条记录"
            if failed_records:
                msg += f"，失败 {len(failed_records)} 条"

            return True, msg, task_dir
        except Exception as e:
            return False, f"批量导出失败: {str(e)}", None

    def export_by_filter(self, start_date: str = None, end_date: str = None,
                         case_types: List[str] = None, export_path: str = None) -> Tuple[bool, str, Optional[str]]:
        """
        按条件导出

        Args:
            start_date: 开始日期
            end_date: 结束日期
            case_types: 类型列表
            export_path: 导出路径

        Returns:
            (成功, 消息, 导出路径)
        """
        try:
            # 查询符合条件的记录
            result = _METRICS.query_records(
                start_date=start_date,
                end_date=end_date,
                case_type=None,
                include_deleted=False,
                limit=10000,
                offset=0
            )

            records = result.get("records", [])

            # 按类型筛选
            if case_types:
                records = [r for r in records if r.get("case_type") in case_types]

            if not records:
                return False, "没有符合条件的记录", None

            # 提取记录ID
            record_ids = [r.get("record_id") for r in records if r.get("record_id")]

            # 批量导出
            return self.export_batch(record_ids, export_path, group_by="case_type", include_summary=True)
        except Exception as e:
            return False, f"按条件导出失败: {str(e)}", None

    def _generate_summary_csv(self, records: List[Dict], output_dir: str):
        """生成汇总CSV"""
        try:
            csv_path = os.path.join(output_dir, "export_summary.csv")
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "记录ID", "检测时间", "系统判定", "车头相似度", "车尾相似度",
                    "是否复核", "复核结果", "复核人员",
                    "输入路径1", "输入路径2", "输入路径3", "输入路径4", "输入模式", "尾部AI模式"
                ])

                for record in records:
                    writer.writerow([
                        record.get("record_id", ""),
                        record.get("ts", ""),
                        record.get("case_type", ""),
                        record.get("head_prob", ""),
                        record.get("tail_prob", ""),
                        "是" if record.get("reviewed", False) else "否",
                        record.get("reviewed_case_type", ""),
                        record.get("reviewed_by", ""),
                        record.get("input_path1", ""),
                        record.get("input_path2", ""),
                        record.get("input_path3", ""),
                        record.get("input_path4", ""),
                        record.get("input_mode", ""),
                        record.get("tail_ai_mode", ""),
                    ])
        except Exception:
            pass

    def _generate_export_log(self, exported: List[Dict], failed: List[Dict], output_dir: str):
        """生成导出日志"""
        try:
            log_path = os.path.join(output_dir, "export_log.txt")
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"导出时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"成功导出: {len(exported)} 条\n")
                f.write(f"导出失败: {len(failed)} 条\n\n")

                if failed:
                    f.write("--- 失败记录 ---\n")
                    for item in failed:
                        f.write(f"记录ID: {item['record_id']}, 错误: {item['error']}\n")
        except Exception:
            pass


_EXPORTER_LEGACY = RecordExporterLegacy()


def _record_metric(
        *,
        endpoint: str,
        source: str,
        http_status: int,
        ok: bool,
        case_type: Optional[str],
        head_prob: Optional[float],
        tail_prob: Optional[float],
        lat_ms: float,
        stage_ms: Optional[Dict[str, float]] = None,
        error: str = "",
        previews: Optional[Dict[str, str]] = None,
        original_images: Optional[Dict[str, str]] = None,
        input_path1: str = "",
        input_path2: str = "",
        input_path3: str = "",
        input_path4: str = "",
        input_mode: str = "",
        ai_judge_used: bool = False,
        head_ai_used: bool = False,
        ai_head_result: Optional[str] = None,
        ai_tail_result: Optional[str] = None,
        ai_head_reason: Optional[str] = None,
        ai_tail_reason: Optional[str] = None,
        ai_ms: Optional[float] = None,
        char_ms: Optional[float] = None,
        tail_ai_mode: str = "",
        stage1_case_type: str = "",
        tail_second_check_used: bool = False,
        tail_second_check_result: str = "",
        tail_second_check_reason: str = "",
        tail_number_consistency: Optional[str] = None,
        tail_structure_consistency: Optional[str] = None,
        diff_analyzed_part: Optional[str] = None,
        ai_diff_ms: Optional[float] = None,
        head_ai_display_text: Optional[str] = None,
        tail34_ai_display_text: Optional[str] = None,
        main_tail_ai_display_text: Optional[str] = None,
        final_diff_summary: Optional[str] = None,
        crop_status: Optional[Dict[str, Any]] = None,
        char_compare_used: bool = False,
        char_compare_verdict: str = "",
        char_compare_plate_type: str = "",
        char_compare_R: Optional[int] = None,
        char_compare_M: Optional[int] = None,
        char_compare_U: Optional[int] = None,
        char_compare_p3_seq: str = "",
        char_compare_p4_seq: str = "",
        char_chegua3_seq: str = "",
        char_chegua4_seq: str = "",
        char_fangdahao3_seq: str = "",
        char_fangdahao4_seq: str = "",
        char_p3_chegua_status: str = "",
        char_p3_fangdahao_status: str = "",
        char_p4_chegua_status: str = "",
        char_p4_fangdahao_status: str = "",
) -> Optional[str]:
    """
    记录指标并保存图片

    Args:
        original_images: 包含原始图片的字典 {"original1": data_url, "original2": data_url}
        diff_analyzed_part: 分析的部位
        ai_diff_ms: 差异分析耗时

    Returns:
        record_id if images saved, else None
    """
    record_id = None
    image_dir = None

    # 如果有预览图，保存它们
    if previews and case_type and case_type != "abnormal":
        dt = datetime.datetime.now()
        timestamp = dt.strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        record_id = f"{timestamp}_{unique_id}"

        meta = {
            "record_id": record_id,
            "ts": dt.isoformat(timespec="milliseconds"),
            "case_type": case_type or "",
            "head_prob": head_prob,
            "tail_prob": tail_prob,
            "input_path1": input_path1,
            "input_path2": input_path2,
            "input_path3": input_path3,
            "input_path4": input_path4,
            "input_mode": input_mode,
            "ai_judge_used": bool(ai_judge_used),
            "head_ai_used": bool(head_ai_used),
            "ai_head_result": ai_head_result,
            "ai_tail_result": ai_tail_result,
            "ai_head_reason": ai_head_reason,
            "ai_tail_reason": ai_tail_reason,
            "ai_ms": ai_ms,
            "char_ms": char_ms,
            "tail_ai_mode": tail_ai_mode,
            "stage1_case_type": stage1_case_type,
            "tail_second_check_used": bool(tail_second_check_used),
            "tail_second_check_result": tail_second_check_result,
            "tail_second_check_reason": tail_second_check_reason,
            "tail_number_consistency": tail_number_consistency,
            "tail_structure_consistency": tail_structure_consistency,
            "head_ai_display_text": head_ai_display_text,
            "tail34_ai_display_text": tail34_ai_display_text,
            "main_tail_ai_display_text": main_tail_ai_display_text,
            "final_diff_summary": final_diff_summary,
            "crop_status": crop_status,
            "char_compare_used": bool(char_compare_used),
            "char_compare_verdict": char_compare_verdict or "",
            "char_compare_plate_type": char_compare_plate_type or "",
            "char_compare_R": char_compare_R,
            "char_compare_M": char_compare_M,
            "char_compare_U": char_compare_U,
            "char_compare_p3_seq": char_compare_p3_seq or "",
            "char_compare_p4_seq": char_compare_p4_seq or "",
            "char_chegua3_seq": char_chegua3_seq or "",
            "char_chegua4_seq": char_chegua4_seq or "",
            "char_fangdahao3_seq": char_fangdahao3_seq or "",
            "char_fangdahao4_seq": char_fangdahao4_seq or "",
            "char_p3_chegua_status": char_p3_chegua_status or "",
            "char_p3_fangdahao_status": char_p3_fangdahao_status or "",
            "char_p4_chegua_status": char_p4_chegua_status or "",
            "char_p4_fangdahao_status": char_p4_fangdahao_status or "",
            "endpoint": endpoint,
            "source": source,
            "lat_ms": lat_ms,
            "protected": False,
            "deleted": False,
            "note": "",
        }

        # 添加差异分析信息到meta
        if diff_analyzed_part:
            meta["diff_analyzed_part"] = diff_analyzed_part
        if ai_diff_ms is not None:
            meta["ai_diff_ms"] = ai_diff_ms

        saved_path = _METRICS.save_images(record_id, previews, meta, original_images)
        if saved_path:
            image_dir = saved_path

    ev: Dict[str, Any] = {
        "endpoint": endpoint,
        "source": source,
        "ok": bool(ok),
        "http_status": int(http_status),
        "case_type": case_type or "",
        "head_prob": head_prob,
        "tail_prob": tail_prob,
        "lat_ms": float(lat_ms),
        "stage_ms": stage_ms or {},
        "error": error or "",
        "input_path1": input_path1,
        "input_path2": input_path2,
        "input_path3": input_path3,
        "input_path4": input_path4,
        "input_mode": input_mode,
        "ai_judge_used": bool(ai_judge_used),
        "head_ai_used": bool(head_ai_used),
        "ai_head_result": ai_head_result,
        "ai_tail_result": ai_tail_result,
        "ai_head_reason": ai_head_reason,
        "ai_tail_reason": ai_tail_reason,
        "ai_ms": ai_ms,
        "char_ms": char_ms,
        "tail_ai_mode": tail_ai_mode,
        "stage1_case_type": stage1_case_type,
        "tail_second_check_used": bool(tail_second_check_used),
        "tail_second_check_result": tail_second_check_result,
        "tail_second_check_reason": tail_second_check_reason,
        "tail_number_consistency": tail_number_consistency,
        "tail_structure_consistency": tail_structure_consistency,
        "head_ai_display_text": head_ai_display_text,
        "tail34_ai_display_text": tail34_ai_display_text,
        "main_tail_ai_display_text": main_tail_ai_display_text,
        "final_diff_summary": final_diff_summary,
        "crop_status": crop_status,
        "char_compare_used": bool(char_compare_used),
        "char_compare_verdict": char_compare_verdict or "",
        "char_compare_plate_type": char_compare_plate_type or "",
        "char_compare_R": char_compare_R,
        "char_compare_M": char_compare_M,
        "char_compare_U": char_compare_U,
        "char_compare_p3_seq": char_compare_p3_seq or "",
        "char_compare_p4_seq": char_compare_p4_seq or "",
        "char_chegua3_seq": char_chegua3_seq or "",
        "char_chegua4_seq": char_chegua4_seq or "",
        "char_fangdahao3_seq": char_fangdahao3_seq or "",
        "char_fangdahao4_seq": char_fangdahao4_seq or "",
        "char_p3_chegua_status": char_p3_chegua_status or "",
        "char_p3_fangdahao_status": char_p3_fangdahao_status or "",
        "char_p4_chegua_status": char_p4_chegua_status or "",
        "char_p4_fangdahao_status": char_p4_fangdahao_status or "",
    }

    if record_id:
        ev["record_id"] = record_id
        ev["image_dir"] = image_dir
        # 添加差异分析信息到日志
        if diff_analyzed_part:
            ev["diff_analyzed_part"] = diff_analyzed_part
        if ai_diff_ms is not None:
            ev["ai_diff_ms"] = ai_diff_ms

    _METRICS.record(ev)
    return record_id


def _is_http_url(s: str) -> bool:
    try:
        u = urllib.parse.urlparse(s)
        return u.scheme in {"http", "https"} and bool(u.netloc)
    except Exception:
        return False


def _get_allowed_base_dirs() -> Tuple[str, ...]:
    raw = os.environ.get("ALLOWED_BASE_DIRS", "").strip()
    if not raw:
        return tuple()
    parts = [p.strip() for p in raw.split(";") if p.strip()]
    return tuple(os.path.abspath(p) for p in parts)


def _is_path_allowed(path: str) -> bool:
    allowed = _get_allowed_base_dirs()
    if not allowed:
        return True
    try:
        abs_path = os.path.abspath(path)
        for base in allowed:
            if os.path.commonpath([abs_path, base]) == base:
                return True
        return False
    except Exception:
        return False


def _remote_fetch_enabled() -> bool:
    raw = str(os.environ.get("REMOTE_FETCH_ENABLED", "1")).strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _validate_image_path(p: Any) -> Tuple[bool, Optional[str]]:
    global _IMAGE_RESOLVER
    if not isinstance(p, str) or not p.strip():
        return False, "path must be a non-empty string"

    raw = p.strip()
    if _is_http_url(raw):
        if not _remote_fetch_enabled():
            raw_flag = str(os.environ.get("REMOTE_FETCH_ENABLED", "1")).strip()
            return False, f"remote fetch disabled: REMOTE_FETCH_ENABLED={raw_flag}"
        if _IMAGE_RESOLVER is None:
            _IMAGE_RESOLVER = ImagePathResolver()
        print(f"[predict] try remote fetch: {raw}")
        ok, local_path, err = _IMAGE_RESOLVER.fetch_to_local(raw)
        if not ok or not local_path:
            return False, f"remote fetch failed: {err}"
        abs_path = os.path.abspath(local_path)
        if not _is_path_allowed(abs_path):
            return False, "path not allowed"
        if not os.path.exists(abs_path) or not os.path.isfile(abs_path):
            return False, "file not found after remote fetch"
        ext = os.path.splitext(abs_path)[1].lower()
        if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            return False, "unsupported file extension"
        return True, abs_path

    if not os.path.isabs(raw):
        return False, "path must be absolute"
    abs_path = os.path.abspath(raw)
    if not _is_path_allowed(abs_path):
        return False, "path not allowed"

    if not os.path.isfile(abs_path):
        if os.path.exists(abs_path):
            return False, "path is not a file"
    ext = os.path.splitext(abs_path)[1].lower()
    if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        return False, "unsupported file extension"

    if not os.path.exists(abs_path):
        if _remote_fetch_enabled():
            if _IMAGE_RESOLVER is None:
                _IMAGE_RESOLVER = ImagePathResolver()
            print(f"[predict] local file missing, try remote fetch: {p}")
            ok, local_path, err = _IMAGE_RESOLVER.fetch_to_local(p)
            if ok and local_path:
                abs_path = os.path.abspath(local_path)
                if not _is_path_allowed(abs_path):
                    return False, "path not allowed"
                if os.path.exists(abs_path) and os.path.isfile(abs_path):
                    return True, abs_path
                return False, "file not found after remote fetch"
            return False, f"file not found (remote fetch failed: {err})"
        raw_flag = str(os.environ.get("REMOTE_FETCH_ENABLED", "1")).strip()
        return False, f"file not found (remote fetch disabled: REMOTE_FETCH_ENABLED={raw_flag})"

    return True, abs_path


class VehiclePairPredictor:
    def predict_from_paths(self, path1: str, path2: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        return _compute_head_tail_probs(path1, path2)

    def predict_from_pil(self, img1: Image.Image, img2: Image.Image) -> Tuple[
        Optional[float], Optional[float], Optional[str]]:
        return _compute_head_tail_probs_pil(img1, img2)

    def classify(self, head_prob: Optional[float], tail_prob: Optional[float]) -> str:
        return _classify_case(head_prob, tail_prob)


def _init_models() -> None:
    global _INITIALIZED, _CROPPER, _CROPPER_UNMASKED, _HEAD_MODEL, _TAIL_MODEL, _HEADTAIL_MODEL, _TAIL_VIEW_CROPPER, _IMAGE_RESOLVER, _AI_CHECKER, _AI_TAIL_CHECKER, _CHAR_READER
    if _INITIALIZED:
        return
    with _INIT_LOCK:
        if _INITIALIZED:
            return

        head_model_path = os.environ.get(
            "HEAD_MODEL_PATH",
            r"D:\project\data_chuli\demo\demo\Siamese-pytorch-master\logs\head\0505\best_epoch_weights.pth",
        )
        tail_model_path = os.environ.get(
            "TAIL_MODEL_PATH",
            r"D:\project\data_chuli\demo\demo\Siamese-pytorch-master\logs\weibu\0505\best_epoch_weights.pth",
        )
        headtail_model_path = os.environ.get(
            "HEADTAIL_MODEL_PATH",
            r"D:\data2\runs\detect\train\weights\best.pt",
        )

        _CROPPER = MainVehicleCropper()
        try:
            _CROPPER_UNMASKED = MainVehicleCropper(mask_plates=False)
        except Exception as e:
            _CROPPER_UNMASKED = None
            print(f"[predict] failed to initialize unmasked vehicle cropper: {e}")
        _HEAD_MODEL = Siamese(model_path=head_model_path)
        _TAIL_MODEL = Siamese(model_path=tail_model_path)
        _HEADTAIL_MODEL = YOLO(headtail_model_path)
        try:
            _TAIL_VIEW_CROPPER = TailViewCropper()
        except Exception as e:
            _TAIL_VIEW_CROPPER = None
            print(f"[predict] failed to initialize 3/4 tail-view cropper: {e}")
        if _IMAGE_RESOLVER is None:
            _IMAGE_RESOLVER = ImagePathResolver()

        # 初始化AI二次判断模型（延迟加载，仅在启用时初始化）
        ai_enabled = _ai_second_judge_enabled()
        if ai_enabled and _AI_CHECKER is None:
            ai_model_name = os.environ.get("AI_JUDGE_MODEL", "qwen3.5:9b")
            _AI_CHECKER = VehicleCheck(model_name=ai_model_name)
            print(f"[predict] AI二次判断已启用, 模型: {ai_model_name}")
        if ai_enabled and _AI_TAIL_CHECKER is None:
            tail_ai_model_name = os.environ.get("AI_TAIL_JUDGE_MODEL", os.environ.get("AI_JUDGE_MODEL", "qwen3.5:9b"))
            _AI_TAIL_CHECKER = TailVehicleCheck(model_name=tail_ai_model_name)
            print(f"[predict] 3/4视角车尾AI判断已启用, 模型: {tail_ai_model_name}")

        if _CHAR_READER is None:
            try:
                _CHAR_READER = CharReader()
                print("[predict] 方案B 字符检测已初始化 (模型懒加载)")
            except Exception as e:
                _CHAR_READER = None
                print(f"[predict] 方案B 字符检测初始化失败: {e}")

        _INITIALIZED = True


def _warmup_char_reader() -> None:
    """启动时预热方案B字符检测 (首次请求不再等 ~10s 模型加载).
    失败仅打印并保留懒加载兜底, 不影响服务启动."""
    global _CHAR_READER
    try:
        if _CHAR_READER is None:
            _CHAR_READER = CharReader()
        _CHAR_READER.warmup()
        print("[predict] 方案B 字符检测预热完成", flush=True)
    except Exception as e:
        print(f"[predict] 方案B 字符检测预热失败(将懒加载): {e}", flush=True)


def _ai_second_judge_enabled() -> bool:
    """检查AI二次判断是否启用"""
    raw = str(os.environ.get("AI_SECOND_JUDGE_ENABLED", "1")).strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _build_classification_result() -> Dict[str, Any]:
    return {
        "case_type": "abnormal",
        "ai_judge_used": False,
        "head_ai_used": False,
        "ai_head_result": None,
        # ai_tail_* 仅表示主视角车尾裁切图 AI 的结果，
        # 不应复用来承载 3/4 视角尾部 AI 的结论。
        "ai_tail_result": None,
        "ai_head_reason": None,
        "ai_tail_reason": None,
        "ai_ms": 0.0,
        "diff_analyzed_part": None,
        "ai_diff_ms": 0.0,
        "tail_ai_mode": "none",
        "stage1_case_type": None,
        # tail_second_check_* 仅表示 3/4 视角尾部 AI 的优先判定结果。
        "tail_second_check_used": False,
        "tail_second_check_result": None,
        "tail_second_check_reason": None,
        # main_tail_ai_used 为主视角车尾 AI 是否真正触发的唯一可信开关。
        "main_tail_ai_used": False,
        "tail_number_consistency": None,
        "tail_structure_consistency": None,
        "ocr_used": False,
        "ocr_match": None,
        "ocr_text1": None,
        "ocr_text2": None,
        "ocr_error": None,
        "head_ai_display_text": None,
        "tail34_ai_display_text": None,
        "main_tail_ai_display_text": None,
        "final_diff_summary": None,
        "crop_status": None,
        "head_ai_decision_source": None,
        "main_tail_ai_decision_source": None,
        # 方案B 字符检测字段
        "char_compare_used": False,
        "char_compare_verdict": None,
        "char_compare_plate_type": None,
        "char_compare_R": None,
        "char_compare_M": None,
        "char_compare_U": None,
        "char_compare_p3_seq": None,
        "char_compare_p4_seq": None,
        "char_whitelist_voided": False,
        "char_whitelist_void_reason": None,
        "timing_ms": {},
    }


def _compute_ai_ms(result: Dict[str, Any]) -> float:
    """AI判断耗时 = 真正进入 ollama 大模型的耗时之和（不含字符检测耗时）。"""
    t = result.get("timing_ms") or {}
    return round(sum(float(t.get(k) or 0.0) for k in ("head_ai_ms", "tail34_ai_ms", "main_tail_ai_ms")), 1)


def _normalize_head_display_label(label: Optional[str]) -> str:
    text = str(label or "").strip().lower()
    if text in {"normal", "正常"}:
        return "正常"
    if text in {"fake_plate", "套牌"}:
        return "套牌"
    return "无法判断"


def _normalize_tail_display_label(label: Optional[str]) -> str:
    text = str(label or "").strip().lower()
    if text in {"normal", "正常"}:
        return "正常"
    if text in {"change_trailer", "换挂"}:
        return "换挂"
    if text in {"undetermined", "无法判断", "无法判定", "unknown"}:
        return "无法判断"
    return "无法判断"


def _clean_reason_text(reason: Optional[str]) -> str:
    text = str(reason or "").strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text


def _build_label_reason_text(label: Optional[str], reason: Optional[str], *, part: str) -> str:
    pretty_label = _normalize_head_display_label(label) if part == "head" else _normalize_tail_display_label(label)
    clean_reason = _clean_reason_text(reason) or "未获得稳定结论"
    return f"{pretty_label}：{clean_reason}"


def _populate_ai_trace_texts(result: Dict[str, Any], head_prob: Optional[float]) -> Dict[str, Any]:
    head_ai_display_text = None
    tail34_ai_display_text = None
    main_tail_ai_display_text = None
    final_diff_summary = None

    if result.get("head_ai_used"):
        head_ai_display_text = _build_label_reason_text(
            result.get("ai_head_result"),
            result.get("ai_head_reason"),
            part="head",
        )

    if result.get("tail_second_check_used"):
        tail34_ai_display_text = _build_label_reason_text(
            result.get("tail_second_check_result"),
            result.get("tail_second_check_reason"),
            part="tail",
        )

    if result.get("main_tail_ai_used"):
        main_tail_ai_display_text = _build_label_reason_text(
            result.get("ai_tail_result"),
            result.get("ai_tail_reason"),
            part="tail",
        )

    case_type = str(result.get("case_type") or "")
    if case_type == "fake_plate":
        head_ai_used = bool(result.get("head_ai_used"))
        ai_head_result = str(result.get("ai_head_result") or "").strip().lower()

        if head_ai_used and ai_head_result == "fake_plate":
            full_reason = _clean_reason_text(result.get("ai_head_reason")) or "车头AI判定为套牌"
            final_diff_summary = f"套牌：{full_reason}"
        elif not head_ai_used and head_prob is not None and head_prob <= _HEAD_THRESHOLD:
            final_diff_summary = "套牌：车头相似度低于阈值，判定为套牌"
        elif result.get("diff_analyzed_part") == "vehicle_detection":
            _cs = result.get("crop_status") or {}
            _v1_txt = "有车" if _cs.get("vehicle1_detected") else "无车"
            _v2_txt = "有车" if _cs.get("vehicle2_detected") else "无车"
            final_diff_summary = f"图片1{{{_v1_txt}}}vs图片2{{{_v2_txt}}},判定为套牌"
        elif result.get("diff_analyzed_part") == "头部视角车辆裁剪":
            # 用户2026-08-17: 头视裁剪不对称(远车/车头过小未裁出的一侧视为无车),
            # 不区分图片1还是图片2, 差异总结统一写固定文案.
            final_diff_summary = "头部视角车辆检测中有车vs无车，直接判定为套牌"
    elif case_type == "change_trailer":
        if result.get("tail_ai_mode") in ("char_compare_change", "char_compare_change_direct"):
            # 尾部字符检测不一致直接判换挂: 固定格式
            p3 = str(result.get("char_compare_p3_seq") or "")
            p4 = str(result.get("char_compare_p4_seq") or "")
            final_diff_summary = f"车挂号/放大号字符检测结果{p3}vs{p4}，明显不一致，判定为换挂"
        elif main_tail_ai_display_text:
            full_reason = _clean_reason_text(result.get("ai_tail_reason"))
            final_diff_summary = f"换挂：{full_reason}" if full_reason else "换挂"
        elif tail34_ai_display_text:
            full_reason = _clean_reason_text(result.get("tail_second_check_reason"))
            final_diff_summary = f"换挂：{full_reason}" if full_reason else "换挂"
        else:
            # 兜底: 直接判换挂但无 display_text 的路径（如 sim_low_change_direct）
            full_reason = _clean_reason_text(result.get("ai_tail_reason"))
            final_diff_summary = f"换挂：{full_reason}" if full_reason else "换挂"
    elif case_type == "normal":
        final_diff_summary = None

    result["head_ai_display_text"] = head_ai_display_text
    result["tail34_ai_display_text"] = tail34_ai_display_text
    result["main_tail_ai_display_text"] = main_tail_ai_display_text
    result["final_diff_summary"] = final_diff_summary
    return result


def _save_pil_to_temp(pil_img: Image.Image, prefix: str = "crop") -> Optional[str]:
    """将PIL图片保存到临时文件，返回路径"""
    try:
        if pil_img is None:
            return None
        img = pil_img.convert("RGB")
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", prefix=prefix + "_", delete=False)
        img.save(tmp, format="JPEG", quality=95)
        tmp.close()
        return tmp.name
    except Exception:
        return None


def _save_upload_file_to_temp(file_storage: Any, prefix: str = "upload") -> Optional[str]:
    """将上传文件保存到临时文件，返回路径"""
    try:
        if file_storage is None:
            return None
        file_storage.stream.seek(0)
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", prefix=prefix + "_", delete=False)
        with open(tmp.name, "wb") as f:
            shutil.copyfileobj(file_storage.stream, f)
        file_storage.stream.seek(0)
        return tmp.name
    except Exception:
        return None


def _pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = pil_img.convert("RGB")
    arr = np.array(rgb)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _preview_max_size() -> int:
    try:
        return int(os.environ.get("PREVIEW_MAX_SIZE", "640"))
    except Exception:
        return 640


def _pil_to_jpeg_data_url(pil_img: Image.Image) -> str:
    img = pil_img
    if img is None:
        return ""
    img = img.convert("RGB")
    max_size = _preview_max_size()
    if max_size > 0:
        img = img.copy()
        img.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _pil_to_original_data_url(pil_img: Image.Image) -> str:
    """将PIL图片转换为原始大小的data URL（不缩放）"""
    img = pil_img
    if img is None:
        return ""
    img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)  # 使用更高质量
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _load_original_data_url_from_path(path: str) -> Optional[str]:
    try:
        if not path or not os.path.exists(path):
            return None
        with Image.open(path) as img:
            return _pil_to_original_data_url(img.copy())
    except Exception:
        return None


def _append_tail_original_images(
        original_images: Optional[Dict[str, str]],
        path3: Optional[str],
        path4: Optional[str]
) -> Optional[Dict[str, str]]:
    if original_images is None:
        return None

    merged = dict(original_images)
    original3 = _load_original_data_url_from_path(str(path3 or ""))
    original4 = _load_original_data_url_from_path(str(path4 or ""))
    if original3:
        merged["original3"] = original3
    if original4:
        merged["original4"] = original4
    return merged


def _crop_tail_view_image(path: str) -> Tuple[Optional[Image.Image], Optional[str]]:
    try:
        _init_models()
        if not path:
            return None, "tail view path missing"
        if _TAIL_VIEW_CROPPER is None:
            return None, "tail view cropper unavailable"

        cropped_bgr, _ = _TAIL_VIEW_CROPPER.crop_image(path)
        if cropped_bgr is None or getattr(cropped_bgr, "size", 0) == 0:
            return None, f"failed to crop tail view: {path}"
        return _bgr_to_pil(cropped_bgr), None
    except Exception as e:
        return None, str(e)


def _prepare_tail_view_assets(
        path3: Optional[str],
        path4: Optional[str],
) -> Tuple[Optional[Tuple[str, str]], Optional[Dict[str, str]], List[str], Optional[str], Optional[Dict[int, np.ndarray]]]:
    if not path3 or not path4:
        return None, None, [], None, None

    temp_files: List[str] = []
    merged: Dict[str, str] = {}
    ai_paths: List[str] = []
    bgr_crops: Dict[int, np.ndarray] = {}

    for idx, path in ((3, str(path3)), (4, str(path4))):
        cropped_pil, err = _crop_tail_view_image(path)
        if cropped_pil is None:
            return None, None, temp_files, err or f"failed to crop tail view {idx}", None

        merged[f"tail_view_crop{idx}"] = _pil_to_original_data_url(cropped_pil)
        # 保留 BGR 车辆图, 供方案B字符比对复用, 避免二次裁剪
        bgr_crops[idx] = np.ascontiguousarray(np.array(cropped_pil)[:, :, ::-1])

        temp_path = _save_pil_to_temp(cropped_pil, prefix=f"tail_view_{idx}")
        if temp_path:
            temp_files.append(temp_path)
            ai_paths.append(temp_path)
        else:
            ai_paths.append(path)

    return (ai_paths[0], ai_paths[1]), merged, temp_files, None, bgr_crops


# ── 方案B 字符检测辅助 ──────────────────────────────────────────────

def _run_char_compare(path3: str, path4: str, pre_cropped=None) -> Dict[str, Any]:
    """运行方案B 字符比对, 异常时返回作废结果 (不抛异常).

    pre_cropped: 可选 (bgr3, bgr4) 已裁剪车辆图, 传入则跳过内部二次裁剪.
    """
    try:
        if _CHAR_READER is None:
            return {"verdict": "作废", "error": "CharReader not initialized"}
        return _CHAR_READER.compare_pair(path3, path4, pre_cropped=pre_cropped)
    except Exception as e:
        return {"verdict": "作废", "error": str(e)}


def _run_char_compare_step(result: Dict[str, Any], char_compare_paths, tail_view_bgr) -> Optional[Dict[str, Any]]:
    """执行方案B 字符比对并填充 result 字段 (高相似短路与 tail_need_ai 共用).

    - 填充 timing_ms.char_compare_ms 与全部 char_compare_*/char_* 字段.
    - 尾部车辆裁剪图叠加 车挂号/放大号 yolo 框 → tail_view_crop*_boxed.
    - 返回 CharReader.compare_pair 原始结果 (含 verdict 等); 路径无效时返回 None.
    """
    if not char_compare_paths or not char_compare_paths[0] or not char_compare_paths[1]:
        return None
    _pre_cropped = None
    if tail_view_bgr is not None:
        _b3, _b4 = tail_view_bgr.get(3), tail_view_bgr.get(4)
        if _b3 is not None and _b4 is not None:
            _pre_cropped = (_b3, _b4)
    _t_char0 = time.perf_counter()
    char_result = _run_char_compare(char_compare_paths[0], char_compare_paths[1], pre_cropped=_pre_cropped)
    result["timing_ms"]["char_compare_ms"] = round((time.perf_counter() - _t_char0) * 1000.0, 1)
    result["char_compare_used"] = True
    result["char_compare_verdict"] = char_result.get("verdict")
    result["char_compare_plate_type"] = char_result.get("plate_type_used")
    result["char_compare_R"] = char_result.get("R")
    result["char_compare_M"] = char_result.get("M")
    result["char_compare_U"] = char_result.get("U")
    result["char_compare_p3_seq"] = _fmt_char_seq(char_result.get("p3_seq", []), conf_line=_GUA_CONF_LINE)
    result["char_compare_p4_seq"] = _fmt_char_seq(char_result.get("p4_seq", []), conf_line=_GUA_CONF_LINE)
    result["char_chegua3_seq"] = _fmt_char_seq(char_result.get("p3_chegua_seq", []), conf_line=_GUA_CONF_LINE)
    result["char_chegua4_seq"] = _fmt_char_seq(char_result.get("p4_chegua_seq", []), conf_line=_GUA_CONF_LINE)
    result["char_fangdahao3_seq"] = _fmt_char_seq(char_result.get("p3_fangdahao_seq", []), conf_line=0.90)
    result["char_fangdahao4_seq"] = _fmt_char_seq(char_result.get("p4_fangdahao_seq", []), conf_line=0.90)
    result["char_p3_chegua_status"] = char_result.get("p3_chegua_status")
    result["char_p3_fangdahao_status"] = char_result.get("p3_fangdahao_status")
    result["char_p4_chegua_status"] = char_result.get("p4_chegua_status")
    result["char_p4_fangdahao_status"] = char_result.get("p4_fangdahao_status")
    # 尾部车辆裁剪图叠加 车挂号/放大号 yolo 框 (供详情页展示)
    if tail_view_bgr is not None:
        _b3 = tail_view_bgr.get(3)
        _b4 = tail_view_bgr.get(4)
        if _b3 is not None and _b4 is not None:
            _boxed3 = _draw_plate_boxes(
                _b3,
                char_result.get("p3_chegua_boxes", []),
                char_result.get("p3_fangdahao_boxes", []),
            )
            _boxed4 = _draw_plate_boxes(
                _b4,
                char_result.get("p4_chegua_boxes", []),
                char_result.get("p4_fangdahao_boxes", []),
            )
            # 仅当实际检测到车挂号/放大号框时才生成 boxed; 无框时不保存框选图
            if char_result.get("p3_chegua_boxes") or char_result.get("p3_fangdahao_boxes"):
                result["tail_view_crop3_boxed"] = _bgr_to_data_url(_boxed3)
            if char_result.get("p4_chegua_boxes") or char_result.get("p4_fangdahao_boxes"):
                result["tail_view_crop4_boxed"] = _bgr_to_data_url(_boxed4)
    return char_result


def _apply_char_whitelist_void(result: Dict[str, Any]) -> bool:
    """白名单号牌命中 → 作废字符比对判定 (verdict 置为 无法判断, 交相似度分带).

    命中条件: 车挂号比对, 且两侧去挂/厂/内后的读序完全相同、不含未知字符(?), 且等于白名单号牌.
    例: 20260813_211147_3ce87a60 桂BA852 两侧完全一致但实为换挂 → 作废字符"一致", 靠低相似度直判换挂.
    """
    if result.get("char_compare_plate_type") != "chegua":
        return False
    p3 = (result.get("char_compare_p3_seq") or "").strip()
    p4 = (result.get("char_compare_p4_seq") or "").strip()
    if not p3 or not p4:
        return False
    # 未知字符数必须为0 (0.85取信线下两侧全部可靠), 否则不允许作废
    # 例: 5ab98815 两侧虽显示桂BA852但 U=2 → 不命中白名单 → 字符一致直判正常
    if int(result.get("char_compare_U") or 0) != 0:
        return False
    if "?" in p3 or "?" in p4:
        return False
    if p3 == p4 and p3 in _CHAR_CHANGE_WHITELIST:
        result["char_compare_verdict"] = "无法判断"
        result["char_whitelist_voided"] = True
        result["char_whitelist_void_reason"] = f"白名单号牌{p3}, 字符判定作废交相似度分带"
        print(f"[predict] char whitelist hit {p3}, verdict voided -> similarity band")
        return True
    return False


def _resolve_tail_ai_paths(result: Dict[str, Any], tail_original_paths, temp_files: List[str]) -> List[str]:
    """送尾部视角AI的图片路径: 检测到车挂号/放大号时优先用框选图(boxed), 无框才用原裁剪图.

    boxed 图以 data URL 存在 result 中, 此处解码写临时文件并登记到 temp_files 供 finally 清理.
    """
    out_paths = list(tail_original_paths)
    for i, idx in ((0, "3"), (1, "4")):
        data_url = result.get(f"tail_view_crop{idx}_boxed", "")
        if not data_url.startswith("data:image/"):
            continue
        try:
            header, encoded = data_url.split(",", 1)
            img_data = base64.b64decode(encoded)
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", prefix=f"tail_ai_{idx}_", delete=False)
            with open(tmp.name, "wb") as f:
                f.write(img_data)
            tmp.close()
            out_paths[i] = tmp.name
            temp_files.append(tmp.name)
        except Exception:
            pass
    return out_paths


def _bgr_to_data_url(bgr) -> str:
    """BGR 图像转 data URL (中文路径安全: cv2.imencode + base64)."""
    if bgr is None:
        return ""
    ok, buf = cv2.imencode(".jpg", bgr)
    if not ok:
        return ""
    import base64
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _draw_plate_boxes(bgr, chegua_boxes, fangdahao_boxes):
    """在图像上叠加 车挂号(绿)/放大号(蓝) yolo 框与标签, 返回新图(不改原图)."""
    if bgr is None:
        return bgr
    img = bgr.copy()
    for (x1, y1, x2, y2) in (chegua_boxes or []):
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 2)
        cv2.putText(img, "chegua", (int(x1), max(int(y1) - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)
    for (x1, y1, x2, y2) in (fangdahao_boxes or []):
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 140, 0), 2)
        cv2.putText(img, "fangdahao", (int(x1), max(int(y1) - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 140, 0), 1, cv2.LINE_AA)
    return img


def _char_metric_kwargs(ai_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """从 ai_result 提取字符检测字段, 用于 _record_metric 的 **kwargs 展开."""
    r = ai_result or {}
    return {
        "char_compare_used": bool(r.get("char_compare_used")),
        "char_compare_verdict": str(r.get("char_compare_verdict") or ""),
        "char_compare_plate_type": str(r.get("char_compare_plate_type") or ""),
        "char_compare_R": r.get("char_compare_R"),
        "char_compare_M": r.get("char_compare_M"),
        "char_compare_U": r.get("char_compare_U"),
        "char_compare_p3_seq": str(r.get("char_compare_p3_seq") or ""),
        "char_compare_p4_seq": str(r.get("char_compare_p4_seq") or ""),
        "char_chegua3_seq": str(r.get("char_chegua3_seq") or ""),
        "char_chegua4_seq": str(r.get("char_chegua4_seq") or ""),
        "char_fangdahao3_seq": str(r.get("char_fangdahao3_seq") or ""),
        "char_fangdahao4_seq": str(r.get("char_fangdahao4_seq") or ""),
        "char_p3_chegua_status": str(r.get("char_p3_chegua_status") or ""),
        "char_p3_fangdahao_status": str(r.get("char_p3_fangdahao_status") or ""),
        "char_p4_chegua_status": str(r.get("char_p4_chegua_status") or ""),
        "char_p4_fangdahao_status": str(r.get("char_p4_fangdahao_status") or ""),
        "char_ms": (r.get("timing_ms") or {}).get("char_compare_ms"),
    }


def _merge_boxed_tail_images(original_images, ai_result):
    """把尾部视角带 yolo 框的 data URL 并入 original_images (供落盘保存)."""
    if original_images is None:
        original_images = {}
    for _key, _url in (
        ("tail_view_crop3_boxed", (ai_result or {}).get("tail_view_crop3_boxed") or ""),
        ("tail_view_crop4_boxed", (ai_result or {}).get("tail_view_crop4_boxed") or ""),
    ):
        if _url:
            original_images[_key] = _url
    return original_images


def _build_char_hint(char_result: Dict[str, Any]) -> str:
    """构建字符检测摘要文本, 用于注入 AI prompt.
    char_result 为 result 字典 (含已格式化的 char_compare_* 字段)."""
    if not char_result or not char_result.get("char_compare_used"):
        return ""
    verdict = char_result.get("char_compare_verdict") or ""
    ptype = char_result.get("char_compare_plate_type") or ""
    R = char_result.get("char_compare_R")
    M = char_result.get("char_compare_M")
    U = char_result.get("char_compare_U")
    s3 = char_result.get("char_compare_p3_seq") or ""
    s4 = char_result.get("char_compare_p4_seq") or ""
    parts = [
        f"[字符检测辅助信息] 类型={ptype} 比对结果={verdict}",
        f"path3读到: {s3}",
        f"path4读到: {s4}",
        f"统计: R={R} M={M} U={U}",
    ]
    return "; ".join(parts)


_AI_QUALITY_TOO_POOR_MARKERS = ("图片质量太差", "ai无法判断", "AI无法判断")
_HEAD_CROP_NO_VEHICLE_MARKERS = (
    "裁切失败侧无目标车辆",
    "无目标车辆",
    "未见车辆",
    "无车",
    "空车道",
    "未拍到车辆",
    "不存在车辆",
)


def _pil_crop_succeeded(parent: Optional[Image.Image], child: Optional[Image.Image]) -> bool:
    if parent is None or child is None:
        return False
    if parent.size != child.size:
        return True
    parent_arr = np.array(parent.convert("RGB"))
    child_arr = np.array(child.convert("RGB"))
    return not np.array_equal(parent_arr, child_arr)


def _build_crop_status(
        img1: Image.Image,
        img2: Image.Image,
        v1: Image.Image,
        v2: Image.Image,
        h1: Image.Image,
        h2: Image.Image,
        t1: Image.Image,
        t2: Image.Image,
        vehicle1_detected: bool = False,
        vehicle2_detected: bool = False,
) -> Dict[str, Any]:
    vehicle1_ok = _pil_crop_succeeded(img1, v1)
    vehicle2_ok = _pil_crop_succeeded(img2, v2)
    head1_ok = _pil_crop_succeeded(v1, h1)
    head2_ok = _pil_crop_succeeded(v2, h2)
    main_tail1_ok = _pil_crop_succeeded(v1, t1)
    main_tail2_ok = _pil_crop_succeeded(v2, t2)
    return {
        "vehicle1_ok": vehicle1_ok,
        "vehicle2_ok": vehicle2_ok,
        "head1_ok": head1_ok,
        "head2_ok": head2_ok,
        "main_tail1_ok": main_tail1_ok,
        "main_tail2_ok": main_tail2_ok,
        "head_ai_pair_ok": head1_ok and head2_ok,
        "head_ai_asymmetric": head1_ok != head2_ok,
        "main_tail_ai_pair_ok": main_tail1_ok and main_tail2_ok,
        "main_tail_ai_asymmetric": main_tail1_ok != main_tail2_ok,
        "vehicle1_detected": vehicle1_detected,
        "vehicle2_detected": vehicle2_detected,
    }


def _reason_indicates_ai_quality_fallback(reason: Optional[str]) -> bool:
    text = str(reason or "")
    return any(marker in text for marker in _AI_QUALITY_TOO_POOR_MARKERS)


def _reason_indicates_head_crop_no_vehicle(reason: Optional[str]) -> bool:
    text = str(reason or "")
    return any(marker in text for marker in _HEAD_CROP_NO_VEHICLE_MARKERS)


def _head_similarity_fallback_label(head_prob: Optional[float], head_threshold: float) -> str:
    if head_prob is not None and head_prob > head_threshold:
        return "normal"
    return "fake_plate"


def _head_similarity_fallback_reason(head_prob: Optional[float], head_threshold: float, ai_label: str) -> str:
    prob_text = f"{head_prob:.4f}" if head_prob is not None else "未知"
    if ai_label == "normal":
        return (
            f"输入图片质量太差，AI无法判断，车头相似度{prob_text}高于阈值{head_threshold}，判定正常"
        )
    return (
        f"输入图片质量太差，AI无法判断，车头相似度{prob_text}低于或等于阈值{head_threshold}，判定套牌"
    )


def _resolve_head_ai_with_crop_guard(
        ai_payload: Dict[str, Any],
        head_prob: Optional[float],
        head_threshold: float,
) -> Tuple[str, str, str]:
    ai_head = str(ai_payload.get("label") or "").strip().lower()
    ai_head_reason = str(ai_payload.get("reason") or "").strip()

    if _reason_indicates_head_crop_no_vehicle(ai_head_reason):
        reason = ai_head_reason or "裁切失败侧无目标车辆，判定套牌"
        return "fake_plate", reason, "crop_no_vehicle"

    if ai_head in ("fake_plate", "normal"):
        return ai_head, ai_head_reason or "", "ai"

    if ai_head == "unknown" or _reason_indicates_ai_quality_fallback(ai_head_reason):
        fallback_label = _head_similarity_fallback_label(head_prob, head_threshold)
        fallback_reason = _head_similarity_fallback_reason(head_prob, head_threshold, fallback_label)
        return fallback_label, fallback_reason, "similarity_fallback"

    return "", "", "invalid"


def _resolve_main_tail_ai_with_crop_guard(
        ai_payload: Dict[str, Any],
        tail_prob: Optional[float],
        tail_threshold: float,
) -> Tuple[str, str, str]:
    ai_tail = str(ai_payload.get("label") or "").strip().lower()
    ai_tail_reason = str(ai_payload.get("reason") or "").strip()

    if ai_tail in ("change_trailer", "normal"):
        return ai_tail, ai_tail_reason or "", "ai"

    if ai_tail == "unknown" or _reason_indicates_ai_quality_fallback(ai_tail_reason):
        fallback_label = _main_tail_similarity_fallback_label(tail_prob, tail_threshold)
        prob_text = f"{tail_prob:.4f}" if tail_prob is not None else "未知"
        if fallback_label == "normal":
            fallback_reason = (
                f"输入图片质量太差，AI无法判断，车尾相似度{prob_text}高于阈值{tail_threshold}，判定正常"
            )
        else:
            fallback_reason = (
                f"输入图片质量太差，AI无法判断，车尾相似度{prob_text}低于或等于阈值{tail_threshold}，判定换挂"
            )
        return fallback_label, fallback_reason, "similarity_fallback"

    return "", "", "invalid"


def _crop_part_from_vehicle_pil(vehicle_image: Image.Image, cls_id: int) -> Image.Image:
    try:
        if vehicle_image is None:
            return vehicle_image
        if _HEADTAIL_MODEL is None:
            return vehicle_image

        bgr = _pil_to_bgr(vehicle_image)
        results = _HEADTAIL_MODEL(bgr, conf=0.25, verbose=False)
        if not results:
            return vehicle_image
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return vehicle_image

        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()

        best_idx = None
        best_score = -1.0
        for i, (c, s) in enumerate(zip(classes, scores)):
            if int(c) != int(cls_id):
                continue
            if float(s) > best_score:
                best_score = float(s)
                best_idx = i

        if best_idx is None:
            return vehicle_image

        x1, y1, x2, y2 = boxes[int(best_idx)]
        h, w = bgr.shape[:2]
        x1 = max(0, min(int(x1), w - 1))
        x2 = max(0, min(int(x2), w))
        y1 = max(0, min(int(y1), h - 1))
        y2 = max(0, min(int(y2), h))
        if x2 <= x1 or y2 <= y1:
            return vehicle_image

        crop = bgr[y1:y2, x1:x2].copy()
        if crop.size == 0:
            return vehicle_image
        return _bgr_to_pil(crop)
    except Exception:
        return vehicle_image


def _compute_head_tail_probs(path1: str, path2: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    try:
        _init_models()
        if _CROPPER is None or _HEAD_MODEL is None or _TAIL_MODEL is None:
            return None, None, "models not initialized"

        img1 = Image.open(path1)
        img2 = Image.open(path2)

        img1, _ = _CROPPER.process_pil(img1)
        img2, _ = _CROPPER.process_pil(img2)

        head1 = _crop_part_from_vehicle_pil(img1, cls_id=0)
        head2 = _crop_part_from_vehicle_pil(img2, cls_id=0)
        tail1 = _crop_part_from_vehicle_pil(img1, cls_id=1)
        tail2 = _crop_part_from_vehicle_pil(img2, cls_id=1)

        head_prob = _HEAD_MODEL.detect_image(head1, head2)
        tail_prob = _TAIL_MODEL.detect_image(tail1, tail2)

        if hasattr(head_prob, "item"):
            head_prob = head_prob.item()
        if hasattr(tail_prob, "item"):
            tail_prob = tail_prob.item()

        return float(head_prob), float(tail_prob), None
    except Exception as e:
        return None, None, str(e)


def _compute_probs_and_previews_pil(
        img1: Image.Image, img2: Image.Image
) -> Tuple[
    Optional[float], Optional[float], Optional[Dict[str, str]], Optional[Dict[str, str]],
    Optional[Dict[str, Image.Image]], Optional[Dict[str, Any]], Optional[str]
]:
    """
    计算概率并生成预览图和原始图

    Returns:
        (head_prob, tail_prob, previews, original_images, cropped_pils, crop_status, error)
        cropped_pils: {"h1": ..., "h2": ..., "t1": ..., "t2": ...} 裁切后的PIL图片，用于AI二次判断
    """
    try:
        _init_models()
        if _CROPPER is None or _HEAD_MODEL is None or _TAIL_MODEL is None:
            return None, None, None, None, None, None, "models not initialized"

        # 保存原始图片的data URL
        original_images: Dict[str, str] = {
            "original1": _pil_to_original_data_url(img1),
            "original2": _pil_to_original_data_url(img2),
        }

        v1, vehicle1_detected = _CROPPER.process_pil(img1)
        v2, vehicle2_detected = _CROPPER.process_pil(img2)

        h1 = _crop_part_from_vehicle_pil(v1, cls_id=0)
        h2 = _crop_part_from_vehicle_pil(v2, cls_id=0)
        t1 = _crop_part_from_vehicle_pil(v1, cls_id=1)
        t2 = _crop_part_from_vehicle_pil(v2, cls_id=1)

        # 2026-08-17 用户要求: 头视裁剪不对称(一边车头裁出/一边没裁出)时,
        # 相似度是无意义的"整车vs车头"垃圾值, 不再强行计算, 由判定层直接报套牌.
        crop_status = _build_crop_status(img1, img2, v1, v2, h1, h2, t1, t2, vehicle1_detected, vehicle2_detected)
        if crop_status.get("head_ai_asymmetric"):
            head_prob = None
            tail_prob = None
        else:
            head_prob = _HEAD_MODEL.detect_image(h1, h2)
            tail_prob = _TAIL_MODEL.detect_image(t1, t2)
            if hasattr(head_prob, "item"):
                head_prob = head_prob.item()
            if hasattr(tail_prob, "item"):
                tail_prob = tail_prob.item()

        previews: Dict[str, str] = {
            "vehicle1": _pil_to_jpeg_data_url(v1),
            "vehicle2": _pil_to_jpeg_data_url(v2),
            "head1": _pil_to_jpeg_data_url(h1),
            "head2": _pil_to_jpeg_data_url(h2),
            "tail1": _pil_to_jpeg_data_url(t1),
            "tail2": _pil_to_jpeg_data_url(t2),
        }

        # 未遮挡车牌版的车辆裁剪图 (供详情页展示)
        if _CROPPER_UNMASKED is not None:
            try:
                vu1, _ = _CROPPER_UNMASKED.process_pil(img1)
                vu2, _ = _CROPPER_UNMASKED.process_pil(img2)
                previews["vehicle1_unmasked"] = _pil_to_jpeg_data_url(vu1)
                previews["vehicle2_unmasked"] = _pil_to_jpeg_data_url(vu2)
            except Exception as e:
                print(f"[predict] failed to build unmasked vehicle crops: {e}")

        # 保留裁切后的PIL图片，用于AI二次判断
        cropped_pils: Dict[str, Image.Image] = {
            "h1": h1, "h2": h2, "t1": t1, "t2": t2,
        }
        if crop_status.get("head_ai_asymmetric") or crop_status.get("main_tail_ai_asymmetric"):
            print(f"[predict] crop_status: {crop_status}")

        return (
            None if head_prob is None else float(head_prob),
            None if tail_prob is None else float(tail_prob),
            previews, original_images, cropped_pils, crop_status, None,
        )
    except Exception as e:
        return None, None, None, None, None, None, str(e)


def _compute_head_tail_probs_pil(img1: Image.Image, img2: Image.Image) -> Tuple[
    Optional[float], Optional[float], Optional[str]]:
    try:
        _init_models()
        if _CROPPER is None or _HEAD_MODEL is None or _TAIL_MODEL is None:
            return None, None, "models not initialized"

        img1, _ = _CROPPER.process_pil(img1)
        img2, _ = _CROPPER.process_pil(img2)

        head1 = _crop_part_from_vehicle_pil(img1, cls_id=0)
        head2 = _crop_part_from_vehicle_pil(img2, cls_id=0)
        tail1 = _crop_part_from_vehicle_pil(img1, cls_id=1)
        tail2 = _crop_part_from_vehicle_pil(img2, cls_id=1)

        head_prob = _HEAD_MODEL.detect_image(head1, head2)
        tail_prob = _TAIL_MODEL.detect_image(tail1, tail2)

        if hasattr(head_prob, "item"):
            head_prob = head_prob.item()
        if hasattr(tail_prob, "item"):
            tail_prob = tail_prob.item()

        return float(head_prob), float(tail_prob), None
    except Exception as e:
        return None, None, str(e)


def _classify_case(head_prob: Optional[float], tail_prob: Optional[float]) -> str:
    if head_prob is None or tail_prob is None:
        return "abnormal"

    head_low_th = _HEAD_THRESHOLD
    tail_low_th = _TAIL_THRESHOLD

    if head_prob < head_low_th:
        return "fake_plate"
    if head_prob >= head_low_th and tail_prob <= tail_low_th:
        return "change_trailer"
    return "normal"


# 判定模式（顺序即展示顺序）：头部车辆裁剪 / 尾部字符检测 / ai判断 / 阈值兜底
JUDGE_MODES = ["头部车辆裁剪", "尾部字符检测", "ai判断", "阈值兜底"]


def _derive_judge_mode(r: dict) -> str:
    """按判定链路优先级，把一条记录归类到 4 种判定模式之一（仅依赖已落盘字段）。

    与 _classify_with_ai_second_judge_internal 的决策顺序保持一致：
      ① 头部车辆裁剪失败（车辆检测异常/头视裁剪不对称/无目标车辆）→ 直接套牌
      ② 尾部字符检测：字符一致→正常、不一致→换挂，直判
      ③ ai判断：车头AI判套牌、尾部视角车尾AI、头部视角车尾AI给出结论；
        AI无法确定回退阈值的不算 ai判断
      ④ 阈值兜底：其余全部（字符无法判断的相似度分带、AI回退阈值等）
    """
    if r.get("judge_mode"):  # 防御：未来若落盘则直接使用
        return r["judge_mode"]

    cs = r.get("crop_status") or {}
    diff = r.get("diff_analyzed_part") or ""
    tm = str(r.get("tail_ai_mode") or "")
    case_type = r.get("case_type") or ""
    _crop_no_veh = lambda s: any(m in s for m in _HEAD_CROP_NO_VEHICLE_MARKERS)
    _ai_fallback = lambda s: any(m in s for m in _AI_QUALITY_TOO_POOR_MARKERS)

    # ① 头部车辆裁剪：裁剪失败/车辆检测异常 → 直接套牌
    if diff in ("vehicle_detection", "头部视角车辆裁剪"):
        return "头部车辆裁剪"
    if case_type == "fake_plate":
        crop_failed = any(
            cs.get(k) is False
            for k in ("vehicle1_ok", "vehicle2_ok", "head1_ok", "head2_ok")
        )
        if crop_failed and not r.get("head_ai_used"):
            return "头部车辆裁剪"
        # 车头AI reason 含"无车/裁切失败"标记 → 头部裁剪
        if _crop_no_veh(str(r.get("ai_head_reason") or "")):
            return "头部车辆裁剪"

    # ② 尾部字符检测：字符一致/不一致直判
    if tm in ("char_compare_normal_direct", "char_compare_change_direct"):
        return "尾部字符检测"
    if r.get("char_compare_used") and r.get("char_compare_verdict") in ("一致", "不一致"):
        return "尾部字符检测"

    # ③ ai判断：AI 确实给出结论
    if r.get("head_ai_used") and r.get("ai_head_result") == "fake_plate":
        if not _crop_no_veh(str(r.get("ai_head_reason") or "")):
            return "ai判断"
    if r.get("tail_second_check_used"):
        if r.get("tail_second_check_result") in ("change_trailer", "normal"):
            return "ai判断"
    if tm in ("tail34_cropped_primary", "tail34_cropped_then_main", "main_tail_crop_only"):
        if not _ai_fallback(str(r.get("ai_tail_reason") or "")):
            return "ai判断"

    # ④ 阈值兜底（其余全部）
    return "阈值兜底"


def _classify_with_thresholds(
        head_prob: Optional[float],
        tail_prob: Optional[float],
        head_threshold: float,
        tail_threshold: float) -> str:
    """度量学习初判：镜像 _classify_case，但用显式阈值（评估口径用运行时刻的阈值）"""
    if head_prob is None or tail_prob is None:
        return "abnormal"
    if head_prob < head_threshold:
        return "fake_plate"
    if head_prob >= head_threshold and tail_prob <= tail_threshold:
        return "change_trailer"
    return "normal"


def _main_tail_similarity_fallback_label(tail_prob: Optional[float], tail_threshold: float) -> str:
    if tail_prob is not None and tail_prob > tail_threshold:
        return "normal"
    return "change_trailer"


def _apply_main_tail_similarity_fallback(
        tail_prob: Optional[float],
        tail_threshold: float,
) -> Tuple[str, str, str]:
    """头部视角车尾 AI 无有效结论时，按车尾相似度与阈值比较定案。"""
    ai_label = _main_tail_similarity_fallback_label(tail_prob, tail_threshold)
    if ai_label == "normal":
        prob_text = f"{tail_prob:.4f}" if tail_prob is not None else "未知"
        reason = (
            f"头部视角车尾AI无有效结论，车尾相似度{prob_text}高于阈值{tail_threshold}，判定正常"
        )
        return "same", ai_label, reason

    prob_text = f"{tail_prob:.4f}" if tail_prob is not None else "未知"
    reason = (
        f"头部视角车尾AI无有效结论，车尾相似度{prob_text}低于或等于阈值{tail_threshold}，判定换挂"
    )
    return "different", ai_label, reason


def _head_ai_cleared_normal(head_need_ai: bool, head_verdict: Optional[str], head_ai_used: bool) -> bool:
    return head_verdict == "normal" and (not head_need_ai or head_ai_used)


def _apply_tail34_h2_guard(ai_tail_payload: Dict[str, Any]) -> Dict[str, Any]:
    """GUI 二次校验：尾部视角车尾 AI 在号牌一致时不得输出换挂或无法判断。"""
    payload = dict(ai_tail_payload or {})
    label = str(payload.get("label") or "").strip()
    plate = str(payload.get("plate_or_number_consistency") or "").strip()
    pair_comparable = str(payload.get("pair_comparable") or "").strip()
    reason = str(payload.get("reason") or "").strip()

    if pair_comparable == "否":
        return payload

    plate_matched = plate == "一致"
    reason_indicates_plate_match = any(
        token in reason
        for token in ("号牌一致", "号牌相同", "车牌一致", "车牌相同", "编号一致")
    )
    should_force_normal = plate_matched or (label == "换挂" and reason_indicates_plate_match)
    if not should_force_normal:
        return payload

    guard_reason = "双侧挂车号牌/放大号关键位一致，GUI二次校验：结构比对结论无效，强制判定正常"
    original_label = label
    payload["label"] = "正常"
    payload["plate_or_number_consistency"] = "一致"
    payload["structure_consistency"] = "未检验"
    if reason:
        payload["reason"] = f"{guard_reason}（原判定：{reason}）"
    else:
        payload["reason"] = guard_reason
    if original_label != "正常":
        print(
            f"[predict] tail34 H2 GUI guard adjusted label: {original_label!r} -> '正常'"
        )
    return payload


def _classify_with_ai_second_judge_internal(
        head_prob: Optional[float],
        tail_prob: Optional[float],
        cropped_pils: Optional[Dict[str, Image.Image]] = None,
        tail_original_paths: Optional[Tuple[str, str]] = None,
        force_head_ai_recheck: bool = False,
        ocr_match: Optional[bool] = None,
        crop_status: Optional[Dict[str, Any]] = None,
        char_compare_paths: Optional[Tuple[str, str]] = None,
        tail_view_bgr: Optional[Dict[int, np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    两层鉴别分类：
    第一层：Siamese相似度快速筛选
      - 高于阈值：该部位直接视为正常
      - 低于阈值：该部位进入AI复核
    第二层：视觉大模型复核
      - 车头低阈值时，仅对 1/2 主视角裁切车头做一次AI判断
      - 车尾低阈值时，优先对 3/4 视角裁切图做AI判断
      - 若 3/4 视角已明确“正常”或“换挂”，直接结束，不再跑主视角尾部AI
      - 只有 3/4 视角无法明确时，才回退到 1/2 主视角裁切尾部做补充判断

    Args:
        head_prob: 车头相似度
        tail_prob: 车尾相似度
        cropped_pils: 裁切后的PIL图片 {"h1", "h2", "t1", "t2"}

    Returns:
        {
            "case_type": str,              # 最终分类结果
            "ai_judge_used": bool,         # 是否调用了AI二次判断
            "ai_head_result": str|None,    # AI车头判断结果
            "ai_tail_result": str|None,    # 主视角车尾裁切图 AI 判断结果
            "ai_head_reason": str|None,    # AI车头判断依据
            "ai_tail_reason": str|None,    # AI最终采用的车尾判断依据
            "ai_ms": float,                # AI判断耗时(ms)
            "diff_analyzed_part": str|None, # 分析的部位（head/tail/both）
            "ai_diff_ms": float,           # 差异分析耗时(ms)
            "tail_ai_mode": str,           # none / tail34_cropped_primary / tail34_cropped_then_main / main_tail_crop_only
            "stage1_case_type": str|None,  # 原方案最终结果
            "tail_second_check_used": bool,
            "tail_second_check_result": str|None, # 3/4视角尾部 AI 优先判定结果
            "tail_second_check_reason": str|None,
            "tail_number_consistency": str|None,
            "tail_structure_consistency": str|None,
        }
    """
    result: Dict[str, Any] = _build_classification_result()
    result["crop_status"] = crop_status

    # 车辆检测早期检查：如果一张有车一张没车，直接判定为套牌
    if crop_status:
        vehicle1_detected = crop_status.get("vehicle1_detected", False)
        vehicle2_detected = crop_status.get("vehicle2_detected", False)
        if vehicle1_detected != vehicle2_detected:
            _v1_txt = "有车" if vehicle1_detected else "无车"
            _v2_txt = "有车" if vehicle2_detected else "无车"
            _veh_diff_summary = f"图片1{{{_v1_txt}}}vs图片2{{{_v2_txt}}},判定为套牌"
            result["case_type"] = "fake_plate"
            result["diff_analyzed_part"] = "vehicle_detection"
            return _populate_ai_trace_texts(result, head_prob)

        # 2026-08-17 用户要求: 头视裁剪不对称(一边车头裁出/一边没裁出)直接报套牌,
        # 相似度已在 _compute_probs_and_previews_pil 跳过, 这里不进 AI/字符比对, 直接定案.
        if crop_status.get("head_ai_asymmetric"):
            result["case_type"] = "fake_plate"
            result["stage1_case_type"] = "fake_plate"
            result["diff_analyzed_part"] = "头部视角车辆裁剪"
            result["ai_judge_used"] = False
            print("[predict] head crop asymmetric -> direct fake_plate (头部视角车辆裁剪)")
            return _populate_ai_trace_texts(result, head_prob)

    if head_prob is None or tail_prob is None:
        return _populate_ai_trace_texts(result, head_prob)

    head_direct_normal_th = _HEAD_THRESHOLD
    tail_direct_normal_th = _TAIL_THRESHOLD

    head_need_ai = False
    tail_need_ai = False
    use_tail_original_ai = bool(tail_original_paths and tail_original_paths[0] and tail_original_paths[1])
    head_verdict: Optional[str] = "normal"
    tail_verdict: Optional[str] = "same"
    ai_head_reason: Optional[str] = None
    ai_tail_reason: Optional[str] = None
    tail_second_label: Optional[str] = None
    tail_second_reason: Optional[str] = None
    tail_number_consistency: Optional[str] = None
    tail_structure_consistency: Optional[str] = None
    ai_fallback_reason = "图片质量太差，AI无法判断，维持原结论"
    head_ai_invalid = False

    stage1_case_type = _classify_case(head_prob, tail_prob)
    result["stage1_case_type"] = stage1_case_type

    char_compare_paths_valid = bool(
        char_compare_paths and char_compare_paths[0] and char_compare_paths[1]
    )

    head_need_ai = force_head_ai_recheck or (head_prob <= head_direct_normal_th)
    tail_need_ai = tail_prob <= tail_direct_normal_th

    if head_need_ai:
        head_verdict = None

    if tail_need_ai:
        tail_verdict = None

    ai_enabled = _ai_second_judge_enabled()
    if not ai_enabled or _AI_CHECKER is None or cropped_pils is None:
        result["case_type"] = stage1_case_type
        return _populate_ai_trace_texts(result, head_prob)

    # 保存裁切图片到临时文件
    temp_files = []
    try:
        if head_need_ai:
            result["head_ai_used"] = True
            h1_path = _save_pil_to_temp(cropped_pils.get("h1"), prefix="head1")
            h2_path = _save_pil_to_temp(cropped_pils.get("h2"), prefix="head2")
            if h1_path:
                temp_files.append(h1_path)
            if h2_path:
                temp_files.append(h2_path)

            if h1_path and h2_path:
                print(
                    f"[predict] head similarity {head_prob:.4f} requires head AI recheck"
                )
                low_similarity_fallback_label = (
                    "fake_plate" if head_prob is not None and head_prob <= head_direct_normal_th else "normal"
                )
                _t_head_ai_0 = time.perf_counter()
                ai_head_payload = _AI_CHECKER.check_head_with_reason(
                    h1_path,
                    h2_path,
                    low_similarity_fallback_label=low_similarity_fallback_label,
                    crop_status=crop_status,
                )
                result["timing_ms"]["head_ai_ms"] = round((time.perf_counter() - _t_head_ai_0) * 1000.0, 1)
                head_label, ai_head_reason, head_decision_source = _resolve_head_ai_with_crop_guard(
                    ai_head_payload,
                    head_prob,
                    head_direct_normal_th,
                )
                if head_decision_source == "invalid":
                    result["ai_head_result"] = str(ai_head_payload.get("label") or "")
                    result["ai_head_reason"] = ai_head_reason or None
                    print(
                        f"[predict] head AI returned invalid result: "
                        f"{ai_head_payload.get('label')!r}, fallback to stage1"
                    )
                    head_ai_invalid = True
                    head_verdict = None
                else:
                    result["ai_head_result"] = head_label
                    result["ai_head_reason"] = ai_head_reason or None
                    result["head_ai_decision_source"] = head_decision_source
                    head_verdict = head_label
                    print(
                        f"[predict] head AI resolved via {head_decision_source}: "
                        f"{head_label} ({ai_head_reason})"
                    )
            else:
                print("[predict] failed to save head crops, fallback to stage1 result")
                head_ai_invalid = True
                head_verdict = None

        if head_verdict == "fake_plate":
            result["ai_judge_used"] = True
            result["ai_ms"] = _compute_ai_ms(result)
            result["case_type"] = "fake_plate"
            result["diff_analyzed_part"] = "head"
            result["ai_diff_ms"] = 0.0
            print("[predict] head AI concluded fake_plate, skipping all tail AI analysis")
            return _populate_ai_trace_texts(result, head_prob)

        # === 点①③ 字符检测先行 + 白名单作废 + 相似度分带 ===
        # 所有尾部视角车辆裁剪图统一做字符检测; 字符能明确判定一致/不一致即跳过相似度/AI直接定案.
        if char_compare_paths_valid:
            _run_char_compare_step(result, char_compare_paths, tail_view_bgr)
            _apply_char_whitelist_void(result)
            _char_verdict = result.get("char_compare_verdict")
            _plate_type = result.get("char_compare_plate_type")
            if _char_verdict == "一致":
                _reason = (
                    f"字符比对一致({_plate_type}) R={result.get('char_compare_R')} "
                    f"M={result.get('char_compare_M')}, 直接判定正常"
                )
                result["case_type"] = "normal"
                result["tail_ai_mode"] = "char_compare_normal_direct"
                result["ai_tail_reason"] = _reason
                result["ai_judge_used"] = True
                result["ai_ms"] = _compute_ai_ms(result)
                result["diff_analyzed_part"] = None
                result["ai_diff_ms"] = 0.0
                print("[predict] char compare -> 一致, direct normal")
                return _populate_ai_trace_texts(result, head_prob)
            if _char_verdict == "不一致":
                _reason = (
                    f"字符比对不一致({_plate_type}) R={result.get('char_compare_R')} "
                    f"M={result.get('char_compare_M')}, 直接判定换挂"
                )
                result["case_type"] = "change_trailer"
                result["tail_ai_mode"] = "char_compare_change_direct"
                result["ai_tail_reason"] = _reason
                result["ai_judge_used"] = True
                result["ai_ms"] = _compute_ai_ms(result)
                result["diff_analyzed_part"] = "tail"
                result["ai_diff_ms"] = 0.0
                print("[predict] char compare -> 不一致, direct change_trailer")
                return _populate_ai_trace_texts(result, head_prob)
            # 字符无法判断/作废/白名单命中 → 相似度分带 (点③: 无漏检前提下最小化AI进入)
            if head_verdict == "normal" and tail_prob is not None and tail_prob > tail_direct_normal_th:
                # 高相似带 → 正常 (跳过AI)
                _reason = (
                    f"字符无法判定，尾部相似度{tail_prob:.4f}高于阈值{tail_direct_normal_th}, 直接判定正常"
                )
                result["case_type"] = "normal"
                result["tail_ai_mode"] = "sim_high_normal_direct"
                result["ai_tail_reason"] = _reason
                result["ai_judge_used"] = False
                result["ai_ms"] = _compute_ai_ms(result)
                result["diff_analyzed_part"] = None
                result["ai_diff_ms"] = 0.0
                print(f"[predict] char undetermined, tail sim {tail_prob:.4f} > {tail_direct_normal_th}, direct normal")
                return _populate_ai_trace_texts(result, head_prob)
            if tail_prob is not None and tail_prob < _TAIL_SIM_CHANGE_LOW:
                # 低相似带 → 换挂 (跳过AI)
                _reason = (
                    f"字符无法判定，尾部相似度{tail_prob:.4f}低于阈值{_TAIL_SIM_CHANGE_LOW}, 直接判定换挂"
                )
                result["case_type"] = "change_trailer"
                result["tail_ai_mode"] = "sim_low_change_direct"
                result["ai_tail_reason"] = _reason
                result["ai_judge_used"] = True
                result["ai_ms"] = _compute_ai_ms(result)
                result["diff_analyzed_part"] = "tail"
                result["ai_diff_ms"] = 0.0
                print(f"[predict] char undetermined, tail sim {tail_prob:.4f} < {_TAIL_SIM_CHANGE_LOW}, direct change_trailer")
                return _populate_ai_trace_texts(result, head_prob)
            # 中间带 → 需尾部 AI 复核 (tail_verdict 保持 None)
            result["tail_ai_mode"] = None

        if tail_need_ai and tail_verdict is None and use_tail_original_ai and _AI_TAIL_CHECKER is not None:
            print("[predict] tail similarity is below threshold, running 3/4 cropped tail-view AI first")
            result["tail_second_check_used"] = True
            if not result.get("tail_ai_mode") or result["tail_ai_mode"] in ("char_agree_but_low_sim", "char_undetermined_fallback_to_ai"):
                result["tail_ai_mode"] = "tail34_cropped_primary"
            try:
                _t_tail34_0 = time.perf_counter()
                # 送尾部AI的图: 有车挂号/放大号框选时用框选图, 无框才用原裁剪图
                _ai_tail_input_paths = _resolve_tail_ai_paths(result, tail_original_paths, temp_files)
                ai_tail_payload = _apply_tail34_h2_guard(
                    _AI_TAIL_CHECKER.check_tail_on_original(
                        _ai_tail_input_paths[0],
                        _ai_tail_input_paths[1],
                        char_hint=_build_char_hint(result),
                    )
                )
                result["timing_ms"]["tail34_ai_ms"] = round((time.perf_counter() - _t_tail34_0) * 1000.0, 1)
                tail_second_label = str(ai_tail_payload.get("label") or "").strip()
                tail_second_reason = str(ai_tail_payload.get("reason") or "").strip()
                tail_number_consistency = str(ai_tail_payload.get("plate_or_number_consistency") or "").strip()
                tail_structure_consistency = str(ai_tail_payload.get("structure_consistency") or "").strip()
                result["tail_second_check_reason"] = tail_second_reason or None
                result["tail_number_consistency"] = tail_number_consistency or None
                result["tail_structure_consistency"] = tail_structure_consistency or None
                if tail_second_label == "换挂":
                    result["tail_second_check_result"] = "change_trailer"
                    ai_tail_reason = tail_second_reason
                    tail_verdict = "different"
                elif tail_second_label == "正常":
                    result["tail_second_check_result"] = "normal"
                    ai_tail_reason = tail_second_reason
                    tail_verdict = "same"
                elif tail_second_label == "无法判断":
                    result["tail_second_check_result"] = "undetermined"
                    result["tail_second_check_reason"] = tail_second_reason or "3/4视角尾部信息不足，回退主视角车尾裁切图继续判断"
                    print("[predict] 3/4 cropped tail-view AI reported insufficient tail evidence, fallback to main tail AI")
                    tail_verdict = None
                else:
                    print(f"[predict] 3/4 cropped tail-view AI returned invalid result: {tail_second_label!r}")
                    tail_verdict = None
            except Exception as e:
                print(f"[predict] 3/4 cropped tail-view AI failed: {e}")
                tail_verdict = None

        if tail_need_ai and tail_verdict is None:
            result["main_tail_ai_used"] = True
            t1_path = _save_pil_to_temp(cropped_pils.get("t1"), prefix="tail1")
            t2_path = _save_pil_to_temp(cropped_pils.get("t2"), prefix="tail2")
            if t1_path:
                temp_files.append(t1_path)
            if t2_path:
                temp_files.append(t2_path)

            if t1_path and t2_path:
                print(
                    f"[predict] 3/4 cropped tail-view AI could not decide, "
                    f"tail similarity {tail_prob:.4f} still requires main tail AI fallback"
                )
                main_tail_fallback_label = _main_tail_similarity_fallback_label(
                    tail_prob, tail_direct_normal_th
                )
                _t_main_tail_0 = time.perf_counter()
                ai_tail_payload = _AI_CHECKER.check_tail_with_reason(
                    t1_path,
                    t2_path,
                    low_similarity_fallback_label=main_tail_fallback_label,
                    crop_status=crop_status,
                )
                result["timing_ms"]["main_tail_ai_ms"] = round((time.perf_counter() - _t_main_tail_0) * 1000.0, 1)
                main_tail_label, ai_tail_reason, main_tail_decision_source = _resolve_main_tail_ai_with_crop_guard(
                    ai_tail_payload,
                    tail_prob,
                    tail_direct_normal_th,
                )
                result["ai_tail_result"] = main_tail_label or str(ai_tail_payload.get("label") or "")
                result["ai_tail_reason"] = ai_tail_reason or None
                result["tail_ai_mode"] = "tail34_cropped_then_main" if use_tail_original_ai else "main_tail_crop_only"
                if main_tail_decision_source == "invalid":
                    if _head_ai_cleared_normal(
                        head_need_ai, head_verdict, bool(result.get("head_ai_used"))
                    ):
                        tail_verdict, ai_tail_label, ai_tail_reason = _apply_main_tail_similarity_fallback(
                            tail_prob, tail_direct_normal_th
                        )
                        result["ai_tail_result"] = ai_tail_label
                        result["ai_tail_reason"] = ai_tail_reason
                        result["main_tail_ai_decision_source"] = "similarity_fallback"
                        print(
                            f"[predict] main tail AI returned invalid result: "
                            f"{ai_tail_payload.get('label')!r}, similarity fallback -> {ai_tail_label}"
                        )
                    else:
                        print(
                            f"[predict] main tail AI returned invalid result: "
                            f"{ai_tail_payload.get('label')!r}, fallback to stage1"
                        )
                        tail_verdict = None
                else:
                    result["main_tail_ai_decision_source"] = main_tail_decision_source
                    tail_verdict = "different" if main_tail_label == "change_trailer" else "same"
                    print(
                        f"[predict] main tail AI resolved via {main_tail_decision_source}: "
                        f"{main_tail_label} ({ai_tail_reason})"
                    )
            elif _head_ai_cleared_normal(
                head_need_ai, head_verdict, bool(result.get("head_ai_used"))
            ):
                print("[predict] failed to save main tail crops, fallback to tail similarity")
                tail_verdict, ai_tail_label, ai_tail_reason = _apply_main_tail_similarity_fallback(
                    tail_prob, tail_direct_normal_th
                )
                result["ai_tail_result"] = ai_tail_label
                result["ai_tail_reason"] = ai_tail_reason
            else:
                print("[predict] failed to save main tail crops, fallback to stage1 result")
                tail_verdict = None

    finally:
        # 清理临时文件
        for f in temp_files:
            try:
                os.remove(f)
            except Exception:
                pass

    result["ai_judge_used"] = True
    result["ai_ms"] = _compute_ai_ms(result)

    if (
        tail_need_ai
        and tail_verdict is None
        and _head_ai_cleared_normal(head_need_ai, head_verdict, bool(result.get("head_ai_used")))
    ):
        tail_verdict, ai_tail_label, ai_tail_reason = _apply_main_tail_similarity_fallback(
            tail_prob, tail_direct_normal_th
        )
        result["main_tail_ai_used"] = True
        result["ai_tail_result"] = ai_tail_label
        result["ai_tail_reason"] = ai_tail_reason
        print(f"[predict] tail AI inconclusive after head AI normal, similarity fallback -> {ai_tail_label}")

    ai_invalid = (head_need_ai and head_verdict is None) or (tail_need_ai and tail_verdict is None)

    # ---- 综合判定 ----
    if ai_invalid:
        result["case_type"] = stage1_case_type
        result["diff_analyzed_part"] = None
        return _populate_ai_trace_texts(result, head_prob)
    elif head_verdict == "fake_plate":
        result["case_type"] = "fake_plate"
    elif tail_verdict == "different":
        result["case_type"] = "change_trailer"
    else:
        result["case_type"] = "normal"

    if result["case_type"] == "normal":
        result["diff_analyzed_part"] = None
        result["ai_diff_ms"] = 0.0
        return _populate_ai_trace_texts(result, head_prob)

    analyzed_parts: List[str] = []

    if result["case_type"] == "fake_plate":
        if ai_head_reason:
            analyzed_parts.append("head")
        if tail_need_ai and ai_tail_reason:
            analyzed_parts.append("tail")
    elif result["case_type"] == "change_trailer":
        if ai_tail_reason:
            analyzed_parts.append("tail")
        if tail_second_reason and tail_second_reason != ai_tail_reason:
            if "tail" not in analyzed_parts:
                analyzed_parts.append("tail")

    if analyzed_parts:
        uniq_parts = []
        for part in analyzed_parts:
            if part not in uniq_parts:
                uniq_parts.append(part)
        result["diff_analyzed_part"] = "+".join(uniq_parts) if len(uniq_parts) > 1 else uniq_parts[0]
    else:
        result["diff_analyzed_part"] = None
    result["ai_diff_ms"] = 0.0

    return _populate_ai_trace_texts(result, head_prob)


def _classify_with_ai_second_judge(
        head_prob: Optional[float],
        tail_prob: Optional[float],
        cropped_pils: Optional[Dict[str, Image.Image]] = None,
        tail_original_paths: Optional[Tuple[str, str]] = None,
        crop_status: Optional[Dict[str, Any]] = None,
        char_compare_paths: Optional[Tuple[str, str]] = None,
        tail_view_bgr: Optional[Dict[int, np.ndarray]] = None,
) -> Dict[str, Any]:
    result = _build_classification_result()
    result["stage1_case_type"] = _classify_case(head_prob, tail_prob)

    # OCR已废弃，使用默认值以兼容历史数据
    ocr_result = {
        "ocr_used": False,
        "ocr_match": None,
        "ocr_text1": None,
        "ocr_text2": None,
        "ocr_error": None,
    }
    result.update(ocr_result)

    downstream = _classify_with_ai_second_judge_internal(
        head_prob,
        tail_prob,
        cropped_pils,
        tail_original_paths=tail_original_paths,
        force_head_ai_recheck=False,
        crop_status=crop_status,
        char_compare_paths=char_compare_paths,
        tail_view_bgr=tail_view_bgr,
    )
    downstream.update(ocr_result)
    return _populate_ai_trace_texts(downstream, head_prob)


def _append_ai_trace_fields(resp: Dict[str, Any], ai_result: Dict[str, Any]) -> Dict[str, Any]:
    resp["head_ai_used"] = ai_result.get("head_ai_used", False)
    if ai_result.get("head_ai_display_text") is not None:
        resp["head_ai_display_text"] = ai_result.get("head_ai_display_text")
    if ai_result.get("tail34_ai_display_text") is not None:
        resp["tail34_ai_display_text"] = ai_result.get("tail34_ai_display_text")
    if ai_result.get("main_tail_ai_display_text") is not None:
        resp["main_tail_ai_display_text"] = ai_result.get("main_tail_ai_display_text")
    if ai_result.get("final_diff_summary") is not None:
        resp["final_diff_summary"] = ai_result.get("final_diff_summary")
    if ai_result.get("crop_status") is not None:
        resp["crop_status"] = ai_result.get("crop_status")
    if ai_result.get("timing_ms"):
        resp["timing_ms"] = ai_result.get("timing_ms")
    return resp


@app.get("/")
def index() -> Any:
    return jsonify({
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_upload": "/predict_upload",
            "ui": "/ui",
            "records": "/records",
            "review_stats": "/review_stats",
            "dashboard": "/dashboard",
        }
    })


@app.get("/ui")
def ui() -> Any:
    return render_template("ui.html")


@app.get("/dashboard")
def dashboard() -> Any:
    return render_template("dashboard.html")


@app.get("/stats")
def stats() -> Any:
    return jsonify(_METRICS.snapshot())


@app.get("/stats/recent")
def stats_recent() -> Any:
    try:
        n = int(request.args.get("n", "200"))
    except Exception:
        n = 200
    return jsonify(_METRICS.recent(n=n))


@app.get("/stats/summary")
def stats_summary() -> Any:
    raw = str(request.args.get("days", "7")).strip()
    try:
        days = int(raw)
    except Exception:
        days = 7
    days = max(1, min(90, days))
    return jsonify(_METRICS.summary(days=days))


@app.post("/stats/reset")
def stats_reset() -> Any:
    """重置统计数据，从当前时间重新开始监控"""
    try:
        result = _METRICS.reset()
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


def _parse_bucket_edges(raw: str) -> List[float]:
    """解析 bucket_edges 参数: 逗号分隔秒, 如 '3,60,150'. 非法/空→缺省 [3,60,150]."""
    default = [3.0, 60.0, 150.0]
    if not raw:
        return default
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        return default
    try:
        edges = [float(p) for p in parts]
    except ValueError:
        return default
    edges = sorted(set(e for e in edges if e > 0))
    if not edges:
        return default
    return edges


def _latency_bucket_labels(edges: List[float]) -> List[str]:
    """由边界生成标签: edges=[3,60,150] → ['<3s','3-60s','60-150s','>150s']."""
    labels = [f"<{int(edges[0])}s"]
    for i in range(len(edges) - 1):
        labels.append(f"{int(edges[i])}-{int(edges[i + 1])}s")
    labels.append(f">{int(edges[-1])}s")
    return labels


def _bucket_label(lat_s: float, edges: List[float], labels: List[str]) -> str:
    """把秒数归类到对应区间标签."""
    for i, e in enumerate(edges):
        if lat_s < e:
            return labels[i]
    return labels[-1]


@app.get("/api/stats/range")
def api_stats_range() -> Any:
    """获取指定日期范围的统计数据"""
    try:
        start_date_str = request.args.get("start_date", "").strip()
        end_date_str = request.args.get("end_date", "").strip()
        bucket_edges = _parse_bucket_edges(request.args.get("bucket_edges", ""))
        bucket_labels = _latency_bucket_labels(bucket_edges)
        lat_buckets = {label: 0 for label in bucket_labels}

        if not start_date_str or not end_date_str:
            return jsonify({"error": "请提供开始日期和结束日期"}), 400
        
        # 解析日期
        try:
            start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d").date()
            end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d").date()
        except ValueError:
            return jsonify({"error": "日期格式错误，请使用 YYYY-MM-DD 格式"}), 400
        
        if start_date > end_date:
            return jsonify({"error": "开始日期不能晚于结束日期"}), 400
        
        # 统计数据
        summary = {
            "total_requests": 0,
            "avg_latency_ms": 0.0,
            "normal_count": 0,
            "abnormal_count": 0,
            "fake_plate_count": 0,
            "change_trailer_count": 0,
        }
        
        latency_analysis = {
            "总体": dict(lat_buckets),
            "正常": dict(lat_buckets),
            "换挂": dict(lat_buckets),
            "套牌": dict(lat_buckets),
        }

        # 判定模式分析：各模式请求量+判定结果、按最终判定分组的模式占比
        mode_breakdown = {
            m: {"total": 0, "normal": 0, "fake_plate": 0, "change_trailer": 0}
            for m in JUDGE_MODES
        }
        mode_pies = {k: {m: 0 for m in JUDGE_MODES} for k in ("总体", "正常", "换挂", "套牌")}

        total_latency = 0.0
        by_endpoint = {}
        
        # 遍历日期范围内的日志文件
        current_date = start_date
        while current_date <= end_date:
            date_key = current_date.strftime("%Y%m%d")
            log_path = os.path.join(_METRICS._log_dir, f"stats_{date_key}.jsonl")
            
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            record = json.loads(line.strip())
                            
                            # 统计汇总
                            lat_ms = record.get("lat_ms")
                            if lat_ms is not None:
                                summary["total_requests"] += 1
                                total_latency += lat_ms
                                
                                # 按耗时区间分类
                                lat_s = lat_ms / 1000.0
                                interval = _bucket_label(lat_s, bucket_edges, bucket_labels)
                                
                                # 总体统计
                                latency_analysis["总体"][interval] += 1
                                
                                # 按类型统计
                                case_type = record.get("case_type", "")
                                if case_type == "normal":
                                    latency_analysis["正常"][interval] += 1
                                    summary["normal_count"] += 1
                                elif case_type == "fake_plate":
                                    latency_analysis["套牌"][interval] += 1
                                    summary["fake_plate_count"] += 1
                                elif case_type == "change_trailer":
                                    latency_analysis["换挂"][interval] += 1
                                    summary["change_trailer_count"] += 1
                                elif case_type == "abnormal":
                                    summary["abnormal_count"] += 1

                                # 判定模式累计（仅统计有 lat_ms 的记录，保证各模式次数之和==总请求次数）
                                judge_mode = _derive_judge_mode(record)
                                mode_breakdown[judge_mode]["total"] += 1
                                mode_pies["总体"][judge_mode] += 1
                                if case_type == "normal":
                                    mode_breakdown[judge_mode]["normal"] += 1
                                    mode_pies["正常"][judge_mode] += 1
                                elif case_type == "fake_plate":
                                    mode_breakdown[judge_mode]["fake_plate"] += 1
                                    mode_pies["套牌"][judge_mode] += 1
                                elif case_type == "change_trailer":
                                    mode_breakdown[judge_mode]["change_trailer"] += 1
                                    mode_pies["换挂"][judge_mode] += 1
                            
                            # 统计端点信息
                            endpoint = record.get("endpoint", "")
                            if endpoint:
                                if endpoint not in by_endpoint:
                                    by_endpoint[endpoint] = {
                                        "requests": 0,
                                        "ok": 0,
                                        "errors": 0,
                                        "lat_ms": deque(maxlen=3000),
                                        "http_400": 0,
                                        "http_500": 0,
                                    }
                                ep = by_endpoint[endpoint]
                                ep["requests"] += 1
                                if record.get("ok"):
                                    ep["ok"] += 1
                                else:
                                    ep["errors"] += 1
                                if lat_ms is not None:
                                    ep["lat_ms"].append(lat_ms)
                                if record.get("http_status") == 400:
                                    ep["http_400"] += 1
                                if record.get("http_status", 0) >= 500:
                                    ep["http_500"] += 1
                        except (json.JSONDecodeError, KeyError):
                            continue
            
            current_date += datetime.timedelta(days=1)
        
        # 计算平均耗时
        if summary["total_requests"] > 0:
            summary["avg_latency_ms"] = total_latency / summary["total_requests"]
        
        # 计算端点统计
        for ep in by_endpoint.values():
            lat_list = list(ep["lat_ms"])
            if lat_list:
                ep["lat_avg_ms"] = sum(lat_list) / len(lat_list)
                ep["lat_p95_ms"] = _METRICS._percentile(lat_list, 95)
            else:
                ep["lat_avg_ms"] = 0
                ep["lat_p95_ms"] = 0
            del ep["lat_ms"]
        
        # 获取最近记录
        recent = _METRICS.recent(n=50)
        
        return jsonify({
            "summary": summary,
            "latency_analysis": latency_analysis,
            "judge_mode_analysis": {
                "mode_breakdown": mode_breakdown,
                "mode_pies": mode_pies,
            },
            "by_endpoint": by_endpoint,
            "recent": recent,
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/health")
def health() -> Any:
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict() -> Any:
    t0 = time.perf_counter()
    predictor = VehiclePairPredictor()
    payload = request.get_json(silent=True) or {}
    source = "path"
    path1_input = str(payload.get("path1") or "")
    path2_input = str(payload.get("path2") or "")
    path3_input = str(payload.get("path3") or "")
    path4_input = str(payload.get("path4") or "")
    has_tail_paths = bool(path3_input or path4_input)
    input_mode = "4_paths" if has_tail_paths else "2_paths"

    if any(_is_http_url(x) for x in (path1_input, path2_input, path3_input, path4_input) if x):
        source = "http"

    t_validate0 = time.perf_counter()
    ok1, p1 = _validate_image_path(payload.get("path1"))
    ok2, p2 = _validate_image_path(payload.get("path2"))
    ok3, p3 = True, ""
    ok4, p4 = True, ""
    if has_tail_paths and not (path3_input and path4_input):
        ok3, p3 = False, "path3 and path4 must both be provided"
        ok4, p4 = False, "path3 and path4 must both be provided"
    elif path3_input and path4_input:
        ok3, p3 = _validate_image_path(payload.get("path3"))
        ok4, p4 = _validate_image_path(payload.get("path4"))
    t_validate_ms = (time.perf_counter() - t_validate0) * 1000.0
    if not ok1:
        lat_ms = (time.perf_counter() - t0) * 1000.0
        _record_metric(
            endpoint="/predict",
            source=source,
            http_status=400,
            ok=False,
            case_type="abnormal",
            head_prob=None,
            tail_prob=None,
            lat_ms=lat_ms,
            stage_ms={"validate": t_validate_ms},
            error=f"path1 invalid: {p1}",
            input_path1=path1_input,
            input_path2=path2_input,
            input_path3=path3_input,
            input_path4=path4_input,
            input_mode=input_mode,
        )
        return jsonify({"ok": False, "error": f"path1 invalid: {p1}"}), 400
    if not ok2:
        lat_ms = (time.perf_counter() - t0) * 1000.0
        _record_metric(
            endpoint="/predict",
            source=source,
            http_status=400,
            ok=False,
            case_type="abnormal",
            head_prob=None,
            tail_prob=None,
            lat_ms=lat_ms,
            stage_ms={"validate": t_validate_ms},
            error=f"path2 invalid: {p2}",
            input_path1=path1_input,
            input_path2=path2_input,
            input_path3=path3_input,
            input_path4=path4_input,
            input_mode=input_mode,
        )
        return jsonify({"ok": False, "error": f"path2 invalid: {p2}"}), 400
    if not ok3:
        lat_ms = (time.perf_counter() - t0) * 1000.0
        _record_metric(
            endpoint="/predict",
            source=source,
            http_status=400,
            ok=False,
            case_type="abnormal",
            head_prob=None,
            tail_prob=None,
            lat_ms=lat_ms,
            stage_ms={"validate": t_validate_ms},
            error=f"path3 invalid: {p3}",
            input_path1=path1_input,
            input_path2=path2_input,
            input_path3=path3_input,
            input_path4=path4_input,
            input_mode=input_mode,
        )
        return jsonify({"ok": False, "error": f"path3 invalid: {p3}"}), 400
    if not ok4:
        lat_ms = (time.perf_counter() - t0) * 1000.0
        _record_metric(
            endpoint="/predict",
            source=source,
            http_status=400,
            ok=False,
            case_type="abnormal",
            head_prob=None,
            tail_prob=None,
            lat_ms=lat_ms,
            stage_ms={"validate": t_validate_ms},
            error=f"path4 invalid: {p4}",
            input_path1=path1_input,
            input_path2=path2_input,
            input_path3=path3_input,
            input_path4=path4_input,
            input_mode=input_mode,
        )
        return jsonify({"ok": False, "error": f"path4 invalid: {p4}"}), 400

    # 为了保存图片，需要生成预览图
    t_open_ms = 0.0
    previews = None
    original_images = None
    with _PIPELINE_LOCK:
        try:
            t_open0 = time.perf_counter()
            img1 = Image.open(p1)
            img2 = Image.open(p2)
            t_open_ms = (time.perf_counter() - t_open0) * 1000.0

            # 生成预览图和原始图（用于保存）
            t_preview0 = time.perf_counter()
            head_prob, tail_prob, previews, original_images, cropped_pils, crop_status, err = _compute_probs_and_previews_pil(img1, img2)
            t_preview_ms = (time.perf_counter() - t_preview0) * 1000.0

            # 计算耗时
            t_compute_ms = t_preview_ms
        except Exception as e:
            lat_ms = (time.perf_counter() - t0) * 1000.0
            _record_metric(
                endpoint="/predict",
                source=source,
                http_status=500,
                ok=False,
                case_type="abnormal",
                head_prob=None,
                tail_prob=None,
                lat_ms=lat_ms,
                stage_ms={"validate": t_validate_ms},
                error=f"processing failed: {e}",
                input_path1=path1_input,
                input_path2=path2_input,
                input_path3=path3_input,
                input_path4=path4_input,
                input_mode=input_mode,
            )
            return jsonify({"ok": False, "error": f"processing failed: {e}"}), 500

        tail_view_temp_paths: List[str] = []
        tail_ai_paths = None
        tail_view_bgr = None
        if path3_input and path4_input:
            original_images = _append_tail_original_images(original_images, p3, p4)
            tail_ai_paths, tail_view_images, tail_view_temp_paths, tail_view_err, tail_view_bgr = _prepare_tail_view_assets(p3, p4)
            if tail_view_images:
                if original_images is None:
                    original_images = {}
                original_images.update(tail_view_images)
                if previews is None:
                    previews = {}
                if tail_view_images.get("tail_view_crop3"):
                    previews["original3"] = tail_view_images.get("tail_view_crop3")
                if tail_view_images.get("tail_view_crop4"):
                    previews["original4"] = tail_view_images.get("tail_view_crop4")
            if tail_view_err:
                print(f"[predict] failed to prepare cropped 3/4 tail views: {tail_view_err}")

        # 两层鉴别分类（含AI二次判断）
        ai_result = _classify_with_ai_second_judge(
            head_prob,
            tail_prob,
            cropped_pils,
            tail_original_paths=tail_ai_paths,
            crop_status=crop_status,
            char_compare_paths=((p3, p4) if p3 and p4 else None),
            tail_view_bgr=tail_view_bgr,
        )
        case_type = ai_result["case_type"]

    lat_ms = (time.perf_counter() - t0) * 1000.0

    resp: Dict[str, Any] = {
        "ok": case_type != "abnormal",
        "case_type": case_type,
        "head_prob": head_prob,
        "tail_prob": tail_prob,
        "input_mode": input_mode,
        "tail_ai_mode": ai_result.get("tail_ai_mode", "none"),
        "stage1_case_type": ai_result.get("stage1_case_type"),
        "tail_second_check_used": ai_result.get("tail_second_check_used", False),
        "tail_second_check_result": ai_result.get("tail_second_check_result"),
        "tail_second_check_reason": ai_result.get("tail_second_check_reason"),
        "char_compare_used": ai_result.get("char_compare_used", False),
        "char_compare_verdict": ai_result.get("char_compare_verdict"),
        "char_compare_plate_type": ai_result.get("char_compare_plate_type"),
        "char_compare_R": ai_result.get("char_compare_R"),
        "char_compare_M": ai_result.get("char_compare_M"),
        "char_compare_U": ai_result.get("char_compare_U"),
        "char_compare_p3_seq": ai_result.get("char_compare_p3_seq"),
        "char_compare_p4_seq": ai_result.get("char_compare_p4_seq"),
        "char_chegua3_seq": ai_result.get("char_chegua3_seq"),
        "char_chegua4_seq": ai_result.get("char_chegua4_seq"),
        "char_fangdahao3_seq": ai_result.get("char_fangdahao3_seq"),
        "char_fangdahao4_seq": ai_result.get("char_fangdahao4_seq"),
        "char_p3_chegua_status": ai_result.get("char_p3_chegua_status"),
        "char_p3_fangdahao_status": ai_result.get("char_p3_fangdahao_status"),
        "char_p4_chegua_status": ai_result.get("char_p4_chegua_status"),
        "char_p4_fangdahao_status": ai_result.get("char_p4_fangdahao_status"),
        "lat_ms": round(lat_ms, 1),
    }
    _append_ai_trace_fields(resp, ai_result)
    # 字符检测耗时（仅进入字符比对时有值）
    if ai_result.get("char_compare_used"):
        resp["char_ms"] = round((ai_result.get("timing_ms") or {}).get("char_compare_ms") or 0.0, 1)
    if ai_result["ai_judge_used"]:
        resp["ai_judge_used"] = True
        resp["ai_head_result"] = ai_result["ai_head_result"]
        resp["ai_tail_result"] = ai_result["ai_tail_result"]
        resp["ai_head_reason"] = ai_result.get("ai_head_reason")
        resp["ai_tail_reason"] = ai_result.get("ai_tail_reason")
        resp["ai_ms"] = round(ai_result["ai_ms"], 1)
    if ai_result.get("tail_number_consistency") is not None:
        resp["tail_number_consistency"] = ai_result.get("tail_number_consistency")
    if ai_result.get("tail_structure_consistency") is not None:
        resp["tail_structure_consistency"] = ai_result.get("tail_structure_consistency")
    # 添加细粒度差异分析结果（仅异常车辆）
    if ai_result.get("diff_analyzed_part") is not None:
        resp["diff_analyzed_part"] = ai_result.get("diff_analyzed_part")
    if ai_result.get("ai_diff_ms") is not None:
        resp["ai_diff_ms"] = round(ai_result.get("ai_diff_ms", 0.0), 1)
    if err:
        resp["error"] = err

    # 尾部视角带框图并入 original_images, 供落盘保存
    _merge_boxed_tail_images(original_images, ai_result)

    # 保存图片并记录
    record_id = _record_metric(
        endpoint="/predict",
        source=source,
        http_status=200,
        ok=case_type != "abnormal",
        case_type=case_type,
        head_prob=head_prob,
        tail_prob=tail_prob,
        lat_ms=lat_ms,
        stage_ms={"validate": t_validate_ms, "open": t_open_ms, "compute": t_compute_ms, "ai_judge": ai_result["ai_ms"]},
        error=str(err or ""),
        previews=previews,
        original_images=original_images,
        input_path1=path1_input,
        input_path2=path2_input,
        input_path3=path3_input,
        input_path4=path4_input,
        input_mode=input_mode,
        ai_judge_used=bool(ai_result.get("ai_judge_used")),
        head_ai_used=bool(ai_result.get("head_ai_used")),
        ai_head_result=ai_result.get("ai_head_result"),
        ai_tail_result=ai_result.get("ai_tail_result"),
        ai_head_reason=ai_result.get("ai_head_reason"),
        ai_tail_reason=ai_result.get("ai_tail_reason"),
        ai_ms=ai_result.get("ai_ms"),
        tail_ai_mode=ai_result.get("tail_ai_mode", "none"),
        stage1_case_type=str(ai_result.get("stage1_case_type") or ""),
        tail_second_check_used=bool(ai_result.get("tail_second_check_used")),
        tail_second_check_result=str(ai_result.get("tail_second_check_result") or ""),
        tail_second_check_reason=str(ai_result.get("tail_second_check_reason") or ""),
        tail_number_consistency=ai_result.get("tail_number_consistency"),
        tail_structure_consistency=ai_result.get("tail_structure_consistency"),
        diff_analyzed_part=ai_result.get("diff_analyzed_part"),
        ai_diff_ms=ai_result.get("ai_diff_ms"),
        head_ai_display_text=ai_result.get("head_ai_display_text"),
        tail34_ai_display_text=ai_result.get("tail34_ai_display_text"),
        main_tail_ai_display_text=ai_result.get("main_tail_ai_display_text"),
        final_diff_summary=ai_result.get("final_diff_summary"),
        crop_status=ai_result.get("crop_status"),
        **_char_metric_kwargs(ai_result),
    )

    if record_id:
        resp["record_id"] = record_id

    for temp_path in tail_view_temp_paths:
        try:
            os.remove(temp_path)
        except Exception:
            pass

    return jsonify(resp)


@app.post("/predict_preview")
def predict_preview() -> Any:
    t0 = time.perf_counter()
    predictor = VehiclePairPredictor()
    payload = request.get_json(silent=True) or {}
    source = "path"
    path1_input = str(payload.get("path1") or "")
    path2_input = str(payload.get("path2") or "")
    path3_input = str(payload.get("path3") or "")
    path4_input = str(payload.get("path4") or "")
    has_tail_paths = bool(path3_input or path4_input)
    input_mode = "4_paths" if has_tail_paths else "2_paths"

    if any(_is_http_url(x) for x in (path1_input, path2_input, path3_input, path4_input) if x):
        source = "http"

    t_validate0 = time.perf_counter()
    ok1, p1 = _validate_image_path(payload.get("path1"))
    ok2, p2 = _validate_image_path(payload.get("path2"))
    ok3, p3 = True, ""
    ok4, p4 = True, ""
    if has_tail_paths and not (path3_input and path4_input):
        ok3, p3 = False, "path3 and path4 must both be provided"
        ok4, p4 = False, "path3 and path4 must both be provided"
    elif path3_input and path4_input:
        ok3, p3 = _validate_image_path(payload.get("path3"))
        ok4, p4 = _validate_image_path(payload.get("path4"))
    t_validate_ms = (time.perf_counter() - t_validate0) * 1000.0
    if not ok1:
        lat_ms = (time.perf_counter() - t0) * 1000.0
        _record_metric(
            endpoint="/predict_preview",
            source=source,
            http_status=400,
            ok=False,
            case_type="abnormal",
            head_prob=None,
            tail_prob=None,
            lat_ms=lat_ms,
            stage_ms={"validate": t_validate_ms},
            error=f"path1 invalid: {p1}",
            input_path1=path1_input,
            input_path2=path2_input,
            input_path3=path3_input,
            input_path4=path4_input,
            input_mode=input_mode,
        )
        return jsonify({"ok": False, "error": f"path1 invalid: {p1}"}), 400
    if not ok2:
        lat_ms = (time.perf_counter() - t0) * 1000.0
        _record_metric(
            endpoint="/predict_preview",
            source=source,
            http_status=400,
            ok=False,
            case_type="abnormal",
            head_prob=None,
            tail_prob=None,
            lat_ms=lat_ms,
            stage_ms={"validate": t_validate_ms},
            error=f"path2 invalid: {p2}",
            input_path1=path1_input,
            input_path2=path2_input,
            input_path3=path3_input,
            input_path4=path4_input,
            input_mode=input_mode,
        )
        return jsonify({"ok": False, "error": f"path2 invalid: {p2}"}), 400
    if not ok3:
        lat_ms = (time.perf_counter() - t0) * 1000.0
        _record_metric(
            endpoint="/predict_preview",
            source=source,
            http_status=400,
            ok=False,
            case_type="abnormal",
            head_prob=None,
            tail_prob=None,
            lat_ms=lat_ms,
            stage_ms={"validate": t_validate_ms},
            error=f"path3 invalid: {p3}",
            input_path1=path1_input,
            input_path2=path2_input,
            input_path3=path3_input,
            input_path4=path4_input,
            input_mode=input_mode,
        )
        return jsonify({"ok": False, "error": f"path3 invalid: {p3}"}), 400
    if not ok4:
        lat_ms = (time.perf_counter() - t0) * 1000.0
        _record_metric(
            endpoint="/predict_preview",
            source=source,
            http_status=400,
            ok=False,
            case_type="abnormal",
            head_prob=None,
            tail_prob=None,
            lat_ms=lat_ms,
            stage_ms={"validate": t_validate_ms},
            error=f"path4 invalid: {p4}",
            input_path1=path1_input,
            input_path2=path2_input,
            input_path3=path3_input,
            input_path4=path4_input,
            input_mode=input_mode,
        )
        return jsonify({"ok": False, "error": f"path4 invalid: {p4}"}), 400

    t_open_ms = 0.0
    with _PIPELINE_LOCK:
        try:
            t_open0 = time.perf_counter()
            img1 = Image.open(p1)
            img2 = Image.open(p2)
            t_open_ms = (time.perf_counter() - t_open0) * 1000.0
        except Exception as e:
            lat_ms = (time.perf_counter() - t0) * 1000.0
            _record_metric(
                endpoint="/predict_preview",
                source=source,
                http_status=400,
                ok=False,
                case_type="abnormal",
                head_prob=None,
                tail_prob=None,
                lat_ms=lat_ms,
                stage_ms={"validate": t_validate_ms},
                error=f"failed to open images: {e}",
                input_path1=path1_input,
                input_path2=path2_input,
                input_path3=path3_input,
                input_path4=path4_input,
                input_mode=input_mode,
            )
            return jsonify({"ok": False, "error": f"failed to open images: {e}"}), 400

        t_compute0 = time.perf_counter()
        head_prob, tail_prob, previews, original_images, cropped_pils, crop_status, err = _compute_probs_and_previews_pil(img1, img2)
        t_compute_ms = (time.perf_counter() - t_compute0) * 1000.0

        tail_view_temp_paths: List[str] = []
        tail_ai_paths = None
        tail_view_bgr = None
        if path3_input and path4_input:
            original_images = _append_tail_original_images(original_images, p3, p4)
            tail_ai_paths, tail_view_images, tail_view_temp_paths, tail_view_err, tail_view_bgr = _prepare_tail_view_assets(p3, p4)
            if tail_view_images:
                if original_images is None:
                    original_images = {}
                original_images.update(tail_view_images)
                if previews is None:
                    previews = {}
                if tail_view_images.get("tail_view_crop3"):
                    previews["original3"] = tail_view_images.get("tail_view_crop3")
                if tail_view_images.get("tail_view_crop4"):
                    previews["original4"] = tail_view_images.get("tail_view_crop4")
            if tail_view_err:
                print(f"[predict] failed to prepare cropped 3/4 tail views: {tail_view_err}")

        # 两层鉴别分类（含AI二次判断）
        ai_result = _classify_with_ai_second_judge(
            head_prob,
            tail_prob,
            cropped_pils,
            tail_original_paths=tail_ai_paths,
            crop_status=crop_status,
            char_compare_paths=((p3, p4) if p3 and p4 else None),
            tail_view_bgr=tail_view_bgr,
        )
        case_type = ai_result["case_type"]

    resp: Dict[str, Any] = {
        "ok": case_type != "abnormal",
        "case_type": case_type,
        "head_prob": head_prob,
        "tail_prob": tail_prob,
        "previews": previews or {},
        "input_mode": input_mode,
        "tail_ai_mode": ai_result.get("tail_ai_mode", "none"),
        "char_compare_used": ai_result.get("char_compare_used", False),
        "char_compare_verdict": ai_result.get("char_compare_verdict"),
        "char_compare_plate_type": ai_result.get("char_compare_plate_type"),
        "char_compare_R": ai_result.get("char_compare_R"),
        "char_compare_M": ai_result.get("char_compare_M"),
        "char_compare_U": ai_result.get("char_compare_U"),
        "char_compare_p3_seq": ai_result.get("char_compare_p3_seq"),
        "char_compare_p4_seq": ai_result.get("char_compare_p4_seq"),
        "char_chegua3_seq": ai_result.get("char_chegua3_seq"),
        "char_chegua4_seq": ai_result.get("char_chegua4_seq"),
        "char_fangdahao3_seq": ai_result.get("char_fangdahao3_seq"),
        "char_fangdahao4_seq": ai_result.get("char_fangdahao4_seq"),
        "char_p3_chegua_status": ai_result.get("char_p3_chegua_status"),
        "char_p3_fangdahao_status": ai_result.get("char_p3_fangdahao_status"),
        "char_p4_chegua_status": ai_result.get("char_p4_chegua_status"),
        "char_p4_fangdahao_status": ai_result.get("char_p4_fangdahao_status"),
        "stage1_case_type": ai_result.get("stage1_case_type"),
        "tail_second_check_used": ai_result.get("tail_second_check_used", False),
        "tail_second_check_result": ai_result.get("tail_second_check_result"),
        "tail_second_check_reason": ai_result.get("tail_second_check_reason"),
    }
    _append_ai_trace_fields(resp, ai_result)
    # 字符检测耗时（仅进入字符比对时有值）
    if ai_result.get("char_compare_used"):
        resp["char_ms"] = round((ai_result.get("timing_ms") or {}).get("char_compare_ms") or 0.0, 1)
    if ai_result["ai_judge_used"]:
        resp["ai_judge_used"] = True
        resp["ai_head_result"] = ai_result["ai_head_result"]
        resp["ai_tail_result"] = ai_result["ai_tail_result"]
        resp["ai_head_reason"] = ai_result.get("ai_head_reason")
        resp["ai_tail_reason"] = ai_result.get("ai_tail_reason")
        resp["ai_ms"] = round(ai_result["ai_ms"], 1)
    if ai_result.get("tail_number_consistency") is not None:
        resp["tail_number_consistency"] = ai_result.get("tail_number_consistency")
    if ai_result.get("tail_structure_consistency") is not None:
        resp["tail_structure_consistency"] = ai_result.get("tail_structure_consistency")
    # 添加细粒度差异分析结果（仅异常车辆）
    if ai_result.get("diff_analyzed_part") is not None:
        resp["diff_analyzed_part"] = ai_result.get("diff_analyzed_part")
    if ai_result.get("ai_diff_ms") is not None:
        resp["ai_diff_ms"] = round(ai_result.get("ai_diff_ms", 0.0), 1)
    if err:
        resp["error"] = err
    lat_ms = (time.perf_counter() - t0) * 1000.0
    resp["lat_ms"] = round(lat_ms, 1)

    # 尾部视角带框图并入 original_images, 供落盘保存
    _merge_boxed_tail_images(original_images, ai_result)

    # 4图预览: boxed 尾图注入 resp.previews 供前端显示
    _pv = resp.get("previews") or {}
    if ai_result.get("tail_view_crop3_boxed") or ai_result.get("tail_view_crop4_boxed"):
        _pv["tail_view_crop3_boxed"] = ai_result.get("tail_view_crop3_boxed")
        _pv["tail_view_crop4_boxed"] = ai_result.get("tail_view_crop4_boxed")
        resp["previews"] = _pv

    # 保存图片并记录
    record_id = _record_metric(
        endpoint="/predict_preview",
        source=source,
        http_status=200,
        ok=case_type != "abnormal",
        case_type=case_type,
        head_prob=head_prob,
        tail_prob=tail_prob,
        lat_ms=lat_ms,
        stage_ms={"validate": t_validate_ms, "open": t_open_ms, "compute": t_compute_ms, "ai_judge": ai_result["ai_ms"]},
        error=str(err or ""),
        previews=previews,
        original_images=original_images,
        input_path1=path1_input,
        input_path2=path2_input,
        input_path3=path3_input,
        input_path4=path4_input,
        input_mode=input_mode,
        ai_judge_used=bool(ai_result.get("ai_judge_used")),
        head_ai_used=bool(ai_result.get("head_ai_used")),
        ai_head_result=ai_result.get("ai_head_result"),
        ai_tail_result=ai_result.get("ai_tail_result"),
        ai_head_reason=ai_result.get("ai_head_reason"),
        ai_tail_reason=ai_result.get("ai_tail_reason"),
        ai_ms=ai_result.get("ai_ms"),
        tail_ai_mode=ai_result.get("tail_ai_mode", "none"),
        stage1_case_type=str(ai_result.get("stage1_case_type") or ""),
        tail_second_check_used=bool(ai_result.get("tail_second_check_used")),
        tail_second_check_result=str(ai_result.get("tail_second_check_result") or ""),
        tail_second_check_reason=str(ai_result.get("tail_second_check_reason") or ""),
        tail_number_consistency=ai_result.get("tail_number_consistency"),
        tail_structure_consistency=ai_result.get("tail_structure_consistency"),
        diff_analyzed_part=ai_result.get("diff_analyzed_part"),
        ai_diff_ms=ai_result.get("ai_diff_ms"),
        head_ai_display_text=ai_result.get("head_ai_display_text"),
        tail34_ai_display_text=ai_result.get("tail34_ai_display_text"),
        main_tail_ai_display_text=ai_result.get("main_tail_ai_display_text"),
        final_diff_summary=ai_result.get("final_diff_summary"),
        crop_status=ai_result.get("crop_status"),
        **_char_metric_kwargs(ai_result),
    )

    if record_id:
        resp["record_id"] = record_id

    for temp_path in tail_view_temp_paths:
        try:
            os.remove(temp_path)
        except Exception:
            pass

    return jsonify(resp)


@app.post("/predict_upload_preview")
def predict_upload_preview() -> Any:
    t0 = time.perf_counter()
    predictor = VehiclePairPredictor()
    f1 = request.files.get("file1")
    f2 = request.files.get("file2")
    f3 = request.files.get("file3")
    f4 = request.files.get("file4")
    has_tail_files = bool(f3 or f4)
    input_mode = "4_paths" if has_tail_files else "2_paths"
    if f1 is None:
        lat_ms = (time.perf_counter() - t0) * 1000.0
        _record_metric(
            endpoint="/predict_upload_preview",
            source="upload",
            http_status=400,
            ok=False,
            case_type="abnormal",
            head_prob=None,
            tail_prob=None,
            lat_ms=lat_ms,
            stage_ms={},
            error="file1 missing",
            input_path1=f1.filename if f1 else "",
            input_path2=f2.filename if f2 else "",
            input_path3=f3.filename if f3 else "",
            input_path4=f4.filename if f4 else "",
            input_mode=input_mode,
        )
        return jsonify({"ok": False, "error": "file1 missing"}), 400
    if f2 is None:
        lat_ms = (time.perf_counter() - t0) * 1000.0
        _record_metric(
            endpoint="/predict_upload_preview",
            source="upload",
            http_status=400,
            ok=False,
            case_type="abnormal",
            head_prob=None,
            tail_prob=None,
            lat_ms=lat_ms,
            stage_ms={},
            error="file2 missing",
            input_path1=f1.filename if f1 else "",
            input_path2=f2.filename if f2 else "",
            input_path3=f3.filename if f3 else "",
            input_path4=f4.filename if f4 else "",
            input_mode=input_mode,
        )
        return jsonify({"ok": False, "error": "file2 missing"}), 400
    if has_tail_files and not (f3 and f4):
        lat_ms = (time.perf_counter() - t0) * 1000.0
        _record_metric(
            endpoint="/predict_upload_preview",
            source="upload",
            http_status=400,
            ok=False,
            case_type="abnormal",
            head_prob=None,
            tail_prob=None,
            lat_ms=lat_ms,
            stage_ms={},
            error="file3 and file4 must both be provided",
            input_path1=f1.filename if f1 else "",
            input_path2=f2.filename if f2 else "",
            input_path3=f3.filename if f3 else "",
            input_path4=f4.filename if f4 else "",
            input_mode=input_mode,
        )
        return jsonify({"ok": False, "error": "file3 and file4 must both be provided"}), 400

    t_open_ms = 0.0
    temp_tail_paths: List[str] = []
    with _PIPELINE_LOCK:
        try:
            t_open0 = time.perf_counter()
            img1 = Image.open(f1.stream)
            img2 = Image.open(f2.stream)
            t_open_ms = (time.perf_counter() - t_open0) * 1000.0
        except Exception as e:
            lat_ms = (time.perf_counter() - t0) * 1000.0
            _record_metric(
                endpoint="/predict_upload_preview",
                source="upload",
                http_status=400,
                ok=False,
                case_type="abnormal",
                head_prob=None,
                tail_prob=None,
                lat_ms=lat_ms,
                stage_ms={},
                error=f"failed to open images: {e}",
                input_path1=f1.filename if f1 else "",
                input_path2=f2.filename if f2 else "",
                input_path3=f3.filename if f3 else "",
                input_path4=f4.filename if f4 else "",
                input_mode=input_mode,
            )
            return jsonify({"ok": False, "error": f"failed to open images: {e}"}), 400

        t_compute0 = time.perf_counter()
        head_prob, tail_prob, previews, original_images, cropped_pils, crop_status, err = _compute_probs_and_previews_pil(img1, img2)
        t_compute_ms = (time.perf_counter() - t_compute0) * 1000.0

        # 两层鉴别分类（含AI二次判断）
        tail_original_paths = None
        char_compare_paths = None
        tail_view_bgr = None
        if f3 and f4:
            p3 = _save_upload_file_to_temp(f3, prefix="upload_tail3")
            p4 = _save_upload_file_to_temp(f4, prefix="upload_tail4")
            if p3:
                temp_tail_paths.append(p3)
            if p4:
                temp_tail_paths.append(p4)
            if p3 and p4:
                char_compare_paths = (p3, p4)
                original_images = _append_tail_original_images(original_images, p3, p4)
                tail_original_paths, tail_view_images, tail_view_temp_paths, tail_view_err, tail_view_bgr = _prepare_tail_view_assets(p3, p4)
                temp_tail_paths.extend(tail_view_temp_paths)
                if tail_view_images:
                    if original_images is None:
                        original_images = {}
                    original_images.update(tail_view_images)
                    if previews is None:
                        previews = {}
                    if tail_view_images.get("tail_view_crop3"):
                        previews["original3"] = tail_view_images.get("tail_view_crop3")
                    if tail_view_images.get("tail_view_crop4"):
                        previews["original4"] = tail_view_images.get("tail_view_crop4")
                if tail_view_err:
                    print(f"[predict] failed to prepare cropped 3/4 tail views: {tail_view_err}")
        ai_result = _classify_with_ai_second_judge(
            head_prob,
            tail_prob,
            cropped_pils,
            tail_original_paths=tail_original_paths,
            crop_status=crop_status,
            char_compare_paths=char_compare_paths,
            tail_view_bgr=tail_view_bgr,
        )
        case_type = ai_result["case_type"]

    resp: Dict[str, Any] = {
        "ok": case_type != "abnormal",
        "case_type": case_type,
        "head_prob": head_prob,
        "tail_prob": tail_prob,
        "previews": previews or {},
        "input_mode": input_mode,
        "tail_ai_mode": ai_result.get("tail_ai_mode", "none"),
        "char_compare_used": ai_result.get("char_compare_used", False),
        "char_compare_verdict": ai_result.get("char_compare_verdict"),
        "char_compare_plate_type": ai_result.get("char_compare_plate_type"),
        "char_compare_R": ai_result.get("char_compare_R"),
        "char_compare_M": ai_result.get("char_compare_M"),
        "char_compare_U": ai_result.get("char_compare_U"),
        "char_compare_p3_seq": ai_result.get("char_compare_p3_seq"),
        "char_compare_p4_seq": ai_result.get("char_compare_p4_seq"),
        "char_chegua3_seq": ai_result.get("char_chegua3_seq"),
        "char_chegua4_seq": ai_result.get("char_chegua4_seq"),
        "char_fangdahao3_seq": ai_result.get("char_fangdahao3_seq"),
        "char_fangdahao4_seq": ai_result.get("char_fangdahao4_seq"),
        "char_p3_chegua_status": ai_result.get("char_p3_chegua_status"),
        "char_p3_fangdahao_status": ai_result.get("char_p3_fangdahao_status"),
        "char_p4_chegua_status": ai_result.get("char_p4_chegua_status"),
        "char_p4_fangdahao_status": ai_result.get("char_p4_fangdahao_status"),
        "stage1_case_type": ai_result.get("stage1_case_type"),
        "tail_second_check_used": ai_result.get("tail_second_check_used", False),
        "tail_second_check_result": ai_result.get("tail_second_check_result"),
        "tail_second_check_reason": ai_result.get("tail_second_check_reason"),
    }
    _append_ai_trace_fields(resp, ai_result)
    # 字符检测耗时（仅进入字符比对时有值）
    if ai_result.get("char_compare_used"):
        resp["char_ms"] = round((ai_result.get("timing_ms") or {}).get("char_compare_ms") or 0.0, 1)
    if ai_result["ai_judge_used"]:
        resp["ai_judge_used"] = True
        resp["ai_head_result"] = ai_result["ai_head_result"]
        resp["ai_tail_result"] = ai_result["ai_tail_result"]
        resp["ai_head_reason"] = ai_result.get("ai_head_reason")
        resp["ai_tail_reason"] = ai_result.get("ai_tail_reason")
        resp["ai_ms"] = round(ai_result["ai_ms"], 1)
    if ai_result.get("tail_number_consistency") is not None:
        resp["tail_number_consistency"] = ai_result.get("tail_number_consistency")
    if ai_result.get("tail_structure_consistency") is not None:
        resp["tail_structure_consistency"] = ai_result.get("tail_structure_consistency")
    if err:
        resp["error"] = err
    lat_ms = (time.perf_counter() - t0) * 1000.0
    resp["lat_ms"] = round(lat_ms, 1)

    # 尾部视角带框图并入 original_images, 供落盘保存
    _merge_boxed_tail_images(original_images, ai_result)

    # 4图预览: boxed 尾图注入 resp.previews 供前端显示
    _pv = resp.get("previews") or {}
    if ai_result.get("tail_view_crop3_boxed") or ai_result.get("tail_view_crop4_boxed"):
        _pv["tail_view_crop3_boxed"] = ai_result.get("tail_view_crop3_boxed")
        _pv["tail_view_crop4_boxed"] = ai_result.get("tail_view_crop4_boxed")
        resp["previews"] = _pv

    # 保存图片并记录
    file1_name = f1.filename if f1 else "unknown"
    file2_name = f2.filename if f2 else "unknown"
    file3_name = f3.filename if f3 else ""
    file4_name = f4.filename if f4 else ""

    # 添加细粒度差异分析结果到响应（仅异常车辆）
    if ai_result.get("diff_analyzed_part") is not None:
        resp["diff_analyzed_part"] = ai_result.get("diff_analyzed_part")
    if ai_result.get("ai_diff_ms") is not None:
        resp["ai_diff_ms"] = round(ai_result.get("ai_diff_ms", 0.0), 1)

    record_id = _record_metric(
        endpoint="/predict_upload_preview",
        source="upload",
        http_status=200,
        ok=case_type != "abnormal",
        case_type=case_type,
        head_prob=head_prob,
        tail_prob=tail_prob,
        lat_ms=lat_ms,
        stage_ms={"open": t_open_ms, "compute": t_compute_ms, "ai_judge": ai_result["ai_ms"]},
        error=str(err or ""),
        previews=previews,
        original_images=original_images,
        input_path1=file1_name,
        input_path2=file2_name,
        input_path3=file3_name,
        input_path4=file4_name,
        input_mode=input_mode,
        ai_judge_used=bool(ai_result.get("ai_judge_used")),
        head_ai_used=bool(ai_result.get("head_ai_used")),
        ai_head_result=ai_result.get("ai_head_result"),
        ai_tail_result=ai_result.get("ai_tail_result"),
        ai_head_reason=ai_result.get("ai_head_reason"),
        ai_tail_reason=ai_result.get("ai_tail_reason"),
        ai_ms=ai_result.get("ai_ms"),
        tail_ai_mode=ai_result.get("tail_ai_mode", "none"),
        stage1_case_type=str(ai_result.get("stage1_case_type") or ""),
        tail_second_check_used=bool(ai_result.get("tail_second_check_used")),
        tail_second_check_result=str(ai_result.get("tail_second_check_result") or ""),
        tail_second_check_reason=str(ai_result.get("tail_second_check_reason") or ""),
        tail_number_consistency=ai_result.get("tail_number_consistency"),
        tail_structure_consistency=ai_result.get("tail_structure_consistency"),
        diff_analyzed_part=ai_result.get("diff_analyzed_part"),
        ai_diff_ms=ai_result.get("ai_diff_ms"),
        head_ai_display_text=ai_result.get("head_ai_display_text"),
        tail34_ai_display_text=ai_result.get("tail34_ai_display_text"),
        main_tail_ai_display_text=ai_result.get("main_tail_ai_display_text"),
        final_diff_summary=ai_result.get("final_diff_summary"),
        crop_status=ai_result.get("crop_status"),
        **_char_metric_kwargs(ai_result),
    )

    if record_id:
        resp["record_id"] = record_id

    for temp_path in temp_tail_paths:
        try:
            os.remove(temp_path)
        except Exception:
            pass

    return jsonify(resp)


@app.post("/predict_upload")
def predict_upload() -> Any:
    t0 = time.perf_counter()
    predictor = VehiclePairPredictor()
    f1 = request.files.get("file1")
    f2 = request.files.get("file2")
    f3 = request.files.get("file3")
    f4 = request.files.get("file4")
    has_tail_files = bool(f3 or f4)
    input_mode = "4_paths" if has_tail_files else "2_paths"
    if f1 is None:
        lat_ms = (time.perf_counter() - t0) * 1000.0
        _record_metric(
            endpoint="/predict_upload",
            source="upload",
            http_status=400,
            ok=False,
            case_type="abnormal",
            head_prob=None,
            tail_prob=None,
            lat_ms=lat_ms,
            stage_ms={},
            error="file1 missing",
            input_path1=f1.filename if f1 else "",
            input_path2=f2.filename if f2 else "",
            input_path3=f3.filename if f3 else "",
            input_path4=f4.filename if f4 else "",
            input_mode=input_mode,
        )
        return jsonify({"ok": False, "error": "file1 missing"}), 400
    if f2 is None:
        lat_ms = (time.perf_counter() - t0) * 1000.0
        _record_metric(
            endpoint="/predict_upload",
            source="upload",
            http_status=400,
            ok=False,
            case_type="abnormal",
            head_prob=None,
            tail_prob=None,
            lat_ms=lat_ms,
            stage_ms={},
            error="file2 missing",
            input_path1=f1.filename if f1 else "",
            input_path2=f2.filename if f2 else "",
            input_path3=f3.filename if f3 else "",
            input_path4=f4.filename if f4 else "",
            input_mode=input_mode,
        )
        return jsonify({"ok": False, "error": "file2 missing"}), 400
    if has_tail_files and not (f3 and f4):
        lat_ms = (time.perf_counter() - t0) * 1000.0
        _record_metric(
            endpoint="/predict_upload",
            source="upload",
            http_status=400,
            ok=False,
            case_type="abnormal",
            head_prob=None,
            tail_prob=None,
            lat_ms=lat_ms,
            stage_ms={},
            error="file3 and file4 must both be provided",
            input_path1=f1.filename if f1 else "",
            input_path2=f2.filename if f2 else "",
            input_path3=f3.filename if f3 else "",
            input_path4=f4.filename if f4 else "",
            input_mode=input_mode,
        )
        return jsonify({"ok": False, "error": "file3 and file4 must both be provided"}), 400

    t_open_ms = 0.0
    previews = None
    original_images = None
    temp_tail_paths: List[str] = []
    with _PIPELINE_LOCK:
        try:
            t_open0 = time.perf_counter()
            img1 = Image.open(f1.stream)
            img2 = Image.open(f2.stream)
            t_open_ms = (time.perf_counter() - t_open0) * 1000.0

            # 生成预览图和原始图（用于保存）
            t_preview0 = time.perf_counter()
            head_prob, tail_prob, previews, original_images, cropped_pils, crop_status, err = _compute_probs_and_previews_pil(img1, img2)
            t_preview_ms = (time.perf_counter() - t_preview0) * 1000.0
        except Exception as e:
            lat_ms = (time.perf_counter() - t0) * 1000.0
            _record_metric(
                endpoint="/predict_upload",
                source="upload",
                http_status=400,
                ok=False,
                case_type="abnormal",
                head_prob=None,
                tail_prob=None,
                lat_ms=lat_ms,
                stage_ms={},
                error=f"failed to open images: {e}",
                input_path1=f1.filename if f1 else "",
                input_path2=f2.filename if f2 else "",
                input_path3=f3.filename if f3 else "",
                input_path4=f4.filename if f4 else "",
                input_mode=input_mode,
            )
            return jsonify({"ok": False, "error": f"failed to open images: {e}"}), 400

        t_compute_ms = (time.perf_counter() - t_open0) * 1000.0

        # 两层鉴别分类（含AI二次判断）
        tail_original_paths = None
        char_compare_paths = None
        tail_view_bgr = None
        if f3 and f4:
            p3 = _save_upload_file_to_temp(f3, prefix="upload_tail3")
            p4 = _save_upload_file_to_temp(f4, prefix="upload_tail4")
            if p3:
                temp_tail_paths.append(p3)
            if p4:
                temp_tail_paths.append(p4)
            if p3 and p4:
                char_compare_paths = (p3, p4)
                original_images = _append_tail_original_images(original_images, p3, p4)
                tail_original_paths, tail_view_images, tail_view_temp_paths, tail_view_err, tail_view_bgr = _prepare_tail_view_assets(p3, p4)
                temp_tail_paths.extend(tail_view_temp_paths)
                if tail_view_images:
                    if original_images is None:
                        original_images = {}
                    original_images.update(tail_view_images)
                    if previews is None:
                        previews = {}
                    if tail_view_images.get("tail_view_crop3"):
                        previews["original3"] = tail_view_images.get("tail_view_crop3")
                    if tail_view_images.get("tail_view_crop4"):
                        previews["original4"] = tail_view_images.get("tail_view_crop4")
                if tail_view_err:
                    print(f"[predict] failed to prepare cropped 3/4 tail views: {tail_view_err}")
        ai_result = _classify_with_ai_second_judge(
            head_prob,
            tail_prob,
            cropped_pils,
            tail_original_paths=tail_original_paths,
            crop_status=crop_status,
            char_compare_paths=char_compare_paths,
            tail_view_bgr=tail_view_bgr,
        )
        case_type = ai_result["case_type"]

    resp: Dict[str, Any] = {
        "ok": case_type != "abnormal",
        "case_type": case_type,
        "head_prob": head_prob,
        "tail_prob": tail_prob,
        "input_mode": input_mode,
        "tail_ai_mode": ai_result.get("tail_ai_mode", "none"),
        "stage1_case_type": ai_result.get("stage1_case_type"),
        "tail_second_check_used": ai_result.get("tail_second_check_used", False),
        "tail_second_check_result": ai_result.get("tail_second_check_result"),
        "tail_second_check_reason": ai_result.get("tail_second_check_reason"),
        "char_compare_used": ai_result.get("char_compare_used", False),
        "char_compare_verdict": ai_result.get("char_compare_verdict"),
        "char_compare_plate_type": ai_result.get("char_compare_plate_type"),
        "char_compare_R": ai_result.get("char_compare_R"),
        "char_compare_M": ai_result.get("char_compare_M"),
        "char_compare_U": ai_result.get("char_compare_U"),
        "char_compare_p3_seq": ai_result.get("char_compare_p3_seq"),
        "char_compare_p4_seq": ai_result.get("char_compare_p4_seq"),
        "char_chegua3_seq": ai_result.get("char_chegua3_seq"),
        "char_chegua4_seq": ai_result.get("char_chegua4_seq"),
        "char_fangdahao3_seq": ai_result.get("char_fangdahao3_seq"),
        "char_fangdahao4_seq": ai_result.get("char_fangdahao4_seq"),
        "char_p3_chegua_status": ai_result.get("char_p3_chegua_status"),
        "char_p3_fangdahao_status": ai_result.get("char_p3_fangdahao_status"),
        "char_p4_chegua_status": ai_result.get("char_p4_chegua_status"),
        "char_p4_fangdahao_status": ai_result.get("char_p4_fangdahao_status"),
    }
    _append_ai_trace_fields(resp, ai_result)
    # 字符检测耗时（仅进入字符比对时有值）
    if ai_result.get("char_compare_used"):
        resp["char_ms"] = round((ai_result.get("timing_ms") or {}).get("char_compare_ms") or 0.0, 1)
    if ai_result["ai_judge_used"]:
        resp["ai_judge_used"] = True
        resp["ai_head_result"] = ai_result["ai_head_result"]
        resp["ai_tail_result"] = ai_result["ai_tail_result"]
        resp["ai_head_reason"] = ai_result.get("ai_head_reason")
        resp["ai_tail_reason"] = ai_result.get("ai_tail_reason")
        resp["ai_ms"] = round(ai_result["ai_ms"], 1)
    if ai_result.get("tail_number_consistency") is not None:
        resp["tail_number_consistency"] = ai_result.get("tail_number_consistency")
    if ai_result.get("tail_structure_consistency") is not None:
        resp["tail_structure_consistency"] = ai_result.get("tail_structure_consistency")
    # 添加细粒度差异分析结果（仅异常车辆）
    if ai_result.get("diff_analyzed_part") is not None:
        resp["diff_analyzed_part"] = ai_result.get("diff_analyzed_part")
    if ai_result.get("ai_diff_ms") is not None:
        resp["ai_diff_ms"] = round(ai_result.get("ai_diff_ms", 0.0), 1)
    if err:
        resp["error"] = err
    lat_ms = (time.perf_counter() - t0) * 1000.0
    resp["lat_ms"] = round(lat_ms, 1)

    # 尾部视角带框图并入 original_images, 供落盘保存
    _merge_boxed_tail_images(original_images, ai_result)

    # 保存图片并记录
    file1_name = f1.filename if f1 else "unknown"
    file2_name = f2.filename if f2 else "unknown"
    file3_name = f3.filename if f3 else ""
    file4_name = f4.filename if f4 else ""

    record_id = _record_metric(
        endpoint="/predict_upload",
        source="upload",
        http_status=200,
        ok=case_type != "abnormal",
        case_type=case_type,
        head_prob=head_prob,
        tail_prob=tail_prob,
        lat_ms=lat_ms,
        stage_ms={"open": t_open_ms, "compute": t_compute_ms, "ai_judge": ai_result["ai_ms"]},
        error=str(err or ""),
        previews=previews,
        original_images=original_images,
        input_path1=file1_name,
        input_path2=file2_name,
        input_path3=file3_name,
        input_path4=file4_name,
        input_mode=input_mode,
        ai_judge_used=bool(ai_result.get("ai_judge_used")),
        head_ai_used=bool(ai_result.get("head_ai_used")),
        ai_head_result=ai_result.get("ai_head_result"),
        ai_tail_result=ai_result.get("ai_tail_result"),
        ai_head_reason=ai_result.get("ai_head_reason"),
        ai_tail_reason=ai_result.get("ai_tail_reason"),
        ai_ms=ai_result.get("ai_ms"),
        tail_ai_mode=ai_result.get("tail_ai_mode", "none"),
        stage1_case_type=str(ai_result.get("stage1_case_type") or ""),
        tail_second_check_used=bool(ai_result.get("tail_second_check_used")),
        tail_second_check_result=str(ai_result.get("tail_second_check_result") or ""),
        tail_second_check_reason=str(ai_result.get("tail_second_check_reason") or ""),
        tail_number_consistency=ai_result.get("tail_number_consistency"),
        tail_structure_consistency=ai_result.get("tail_structure_consistency"),
        diff_analyzed_part=ai_result.get("diff_analyzed_part"),
        ai_diff_ms=ai_result.get("ai_diff_ms"),
        head_ai_display_text=ai_result.get("head_ai_display_text"),
        tail34_ai_display_text=ai_result.get("tail34_ai_display_text"),
        main_tail_ai_display_text=ai_result.get("main_tail_ai_display_text"),
        final_diff_summary=ai_result.get("final_diff_summary"),
        crop_status=ai_result.get("crop_status"),
        **_char_metric_kwargs(ai_result),
    )

    if record_id:
        resp["record_id"] = record_id

    for temp_path in temp_tail_paths:
        try:
            os.remove(temp_path)
        except Exception:
            pass

    return jsonify(resp)


@app.get("/records")
def records_page() -> Any:
    """记录查询页面"""
    return render_template("records.html")


@app.get("/api/records")
def api_query_records() -> Any:
    """查询记录列表API"""
    try:
        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")
        case_type = request.args.get("case_type", "all")
        time_filter = request.args.get("time_filter", "all")
        review_filter = request.args.get("review_filter", "all")
        judge_mode = request.args.get("judge_mode", "all")
        limit = int(request.args.get("limit", "50"))
        offset = int(request.args.get("offset", "0"))

        result = _METRICS.query_records(
            start_date=start_date,
            end_date=end_date,
            case_type=case_type if case_type != "all" else None,
            time_filter=time_filter if time_filter != "all" else None,
            review_filter=review_filter if review_filter != "all" else None,
            judge_mode=judge_mode if judge_mode != "all" else None,
            limit=limit,
            offset=offset
        )

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "records": [], "total": 0}), 500


@app.get("/api/record/<record_id>")
def api_get_record(record_id: str) -> Any:
    """获取单条记录详情API"""
    try:
        record = _METRICS.get_record(record_id)
        if not record:
            return jsonify({"error": "记录不存在"}), 404

        return jsonify(record)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/api/record/<record_id>/image/<image_name>")
def api_get_image(record_id: str, image_name: str) -> Any:
    """获取记录的图片"""
    try:
        # 验证图片名称
        valid_names = [
            "original1.jpg", "original2.jpg", "original3.jpg", "original4.jpg",
            "tail_view_crop3.jpg", "tail_view_crop4.jpg",
            "tail_view_crop3_boxed.jpg", "tail_view_crop4_boxed.jpg",
            "vehicle1.jpg", "vehicle2.jpg",
            "vehicle1_unmasked.jpg", "vehicle2_unmasked.jpg",
            "head1.jpg", "head2.jpg", "tail1.jpg", "tail2.jpg",
        ]
        if image_name not in valid_names:
            return jsonify({"error": "无效的图片名称"}), 400

        # 获取记录
        record = _METRICS.get_record(record_id)
        if not record:
            return jsonify({"error": "记录不存在"}), 404

        # 获取图片路径
        image_dir = record.get("image_dir", "")
        if not image_dir or not os.path.exists(image_dir):
            return jsonify({"error": "图片目录不存在"}), 404

        image_path = os.path.join(image_dir, image_name)
        if not os.path.exists(image_path):
            # 历史记录没有新图时的回退: 未遮挡裁剪→遮挡裁剪, 带框图→原尾部裁剪
            fallback = {
                "vehicle1_unmasked.jpg": "vehicle1.jpg",
                "vehicle2_unmasked.jpg": "vehicle2.jpg",
                "tail_view_crop3_boxed.jpg": "tail_view_crop3.jpg",
                "tail_view_crop4_boxed.jpg": "tail_view_crop4.jpg",
            }.get(image_name)
            if fallback:
                image_path = os.path.join(image_dir, fallback)
            if not os.path.exists(image_path):
                return jsonify({"error": "图片不存在"}), 404

        return send_file(image_path, mimetype="image/jpeg")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.delete("/api/record/<record_id>")
def api_delete_record(record_id: str) -> Any:
    """删除记录API"""
    try:
        payload = request.get_json(silent=True) or {}
        hard_delete = payload.get("hard_delete", False)

        success, message = _METRICS.delete_record(record_id, hard_delete=hard_delete)

        if success:
            return jsonify({"ok": True, "message": message})
        else:
            return jsonify({"ok": False, "error": message}), 400
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/api/record/<record_id>/protect")
def api_protect_record(record_id: str) -> Any:
    """设置记录保护状态API"""
    try:
        payload = request.get_json(silent=True) or {}
        protected = payload.get("protected", False)
        note = payload.get("note", "")

        success, message = _METRICS.protect_record(record_id, protected, note)

        if success:
            return jsonify({"ok": True, "message": message})
        else:
            return jsonify({"ok": False, "error": message}), 400
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/api/records/batch_delete")
def api_batch_delete() -> Any:
    """批量删除记录API"""
    try:
        payload = request.get_json(silent=True) or {}
        record_ids = payload.get("record_ids", [])
        hard_delete = payload.get("hard_delete", False)

        if not isinstance(record_ids, list):
            return jsonify({"ok": False, "error": "record_ids 必须是数组"}), 400

        results = []
        for record_id in record_ids:
            success, message = _METRICS.delete_record(record_id, hard_delete=hard_delete)
            results.append({
                "record_id": record_id,
                "success": success,
                "message": message
            })

        return jsonify({"ok": True, "results": results})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/api/record/<record_id>/export")
def api_export_single(record_id: str) -> Any:
    """导出单条记录API"""
    try:
        payload = request.get_json(silent=True) or {}
        export_path = payload.get("export_path")
        image_types = payload.get("image_types")  # 可选的图片类型列表

        success, message, folder = _EXPORTER.export_single(
            record_id,
            export_path,
            image_types
        )

        if success:
            return jsonify({
                "ok": True,
                "message": message,
                "export_path": folder
            })
        else:
            return jsonify({"ok": False, "error": message}), 400
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/api/records/batch_export")
def api_batch_export() -> Any:
    """批量导出记录API"""
    try:
        payload = request.get_json(silent=True) or {}
        record_ids = payload.get("record_ids", [])
        export_path = payload.get("export_path")
        group_by = payload.get("group_by", "case_type")
        image_types = payload.get("image_types")
        include_summary = payload.get("include_summary", True)

        if not isinstance(record_ids, list):
            return jsonify({"ok": False, "error": "record_ids 必须是数组"}), 400

        success, message, folder = _EXPORTER.export_batch(
            record_ids,
            export_path,
            group_by,
            image_types,
            include_summary
        )

        if success:
            return jsonify({
                "ok": True,
                "message": message,
                "export_path": folder
            })
        else:
            return jsonify({"ok": False, "error": message}), 400
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.get("/api/export/image_types")
def api_get_image_types() -> Any:
    """获取可用的图片类型列表"""
    return jsonify({
        "image_types": [
            {"value": "original1", "label": "原图1", "group": "原始图片"},
            {"value": "original2", "label": "原图2", "group": "原始图片"},
            {"value": "original3", "label": "尾部视角原图3", "group": "尾部视角原图"},
            {"value": "original4", "label": "尾部视角原图4", "group": "尾部视角原图"},
            {"value": "tail_view_crop3", "label": "尾部视角裁切图3", "group": "尾部视角裁切图"},
            {"value": "tail_view_crop4", "label": "尾部视角裁切图4", "group": "尾部视角裁切图"},
            {"value": "vehicle1", "label": "车辆1（裁切）", "group": "裁切图片"},
            {"value": "vehicle2", "label": "车辆2（裁切）", "group": "裁切图片"},
            {"value": "head1", "label": "车头1", "group": "部件图片"},
            {"value": "head2", "label": "车头2", "group": "部件图片"},
            {"value": "tail1", "label": "车尾1", "group": "部件图片"},
            {"value": "tail2", "label": "车尾2", "group": "部件图片"},
        ],
        "presets": {
            "all": ["original1", "original2", "original3", "original4", "tail_view_crop3", "tail_view_crop4", "vehicle1", "vehicle2", "head1", "head2", "tail1", "tail2"],
            "original_only": ["original1", "original2", "original3", "original4", "tail_view_crop3", "tail_view_crop4"],
            "processed_only": ["vehicle1", "vehicle2", "head1", "head2", "tail1", "tail2"],
            "head_only": ["head1", "head2"],
            "tail_only": ["tail1", "tail2"],
            "parts_only": ["head1", "head2", "tail1", "tail2"],
        }
    })


@app.post("/api/record/<record_id>/review")
def api_review_record(record_id: str) -> Any:
    """提交复核结果API"""
    try:
        payload = request.get_json(silent=True) or {}
        reviewed_case_type = payload.get("reviewed_case_type", "")
        review_reason = payload.get("review_reason", "")
        reviewed_by = payload.get("reviewed_by", "")
        review_confidence = payload.get("review_confidence", "medium")

        if not reviewed_case_type:
            return jsonify({"ok": False, "error": "复核类型不能为空"}), 400

        if not reviewed_by:
            return jsonify({"ok": False, "error": "复核人员不能为空"}), 400

        success, message = _METRICS.review_record(
            record_id, reviewed_case_type, review_reason, reviewed_by, review_confidence
        )

        if success:
            # 返回更新后的记录
            record = _METRICS.get_record(record_id)
            return jsonify({"ok": True, "message": message, "record": record})
        else:
            return jsonify({"ok": False, "error": message}), 400
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.delete("/api/record/<record_id>/review")
def api_revoke_review(record_id: str) -> Any:
    """撤销复核API"""
    try:
        success, message = _METRICS.revoke_review(record_id)

        if success:
            return jsonify({"ok": True, "message": message})
        else:
            return jsonify({"ok": False, "error": message}), 400
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.get("/api/records/review_stats")
def api_review_stats() -> Any:
    """获取复核统计API"""
    try:
        stats = _METRICS.get_review_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/review_stats")
def review_stats_page() -> Any:
    """复核统计页面"""
    return render_template("review_stats.html")


@app.get("/dataset")
def dataset_page() -> Any:
    """评估数据集管理页面"""
    return render_template("dataset.html")


@app.get("/thresholds")
def get_thresholds() -> Any:
    return jsonify({
        "head_threshold": _HEAD_THRESHOLD,
        "tail_threshold": _TAIL_THRESHOLD,
        "tail_char_threshold": _TAIL_CHAR_THRESHOLD,
        "tail_sim_change_low": _TAIL_SIM_CHANGE_LOW,
    })


@app.post("/thresholds")
def set_thresholds() -> Any:
    global _HEAD_THRESHOLD, _TAIL_THRESHOLD, _TAIL_CHAR_THRESHOLD, _TAIL_SIM_CHANGE_LOW

    try:
        payload = request.get_json(silent=True) or {}
        head_threshold = payload.get("head_threshold", _HEAD_THRESHOLD)
        tail_threshold = payload.get("tail_threshold", _TAIL_THRESHOLD)
        tail_char_threshold = payload.get("tail_char_threshold", _TAIL_CHAR_THRESHOLD)
        tail_sim_change_low = payload.get("tail_sim_change_low", _TAIL_SIM_CHANGE_LOW)

        new_head_threshold = _validate_threshold_value("head_threshold", head_threshold)
        new_tail_threshold = _validate_threshold_value("tail_threshold", tail_threshold)
        new_tail_char_threshold = _validate_threshold_value("tail_char_threshold", tail_char_threshold)
        new_tail_sim_change_low = _validate_threshold_value("tail_sim_change_low", tail_sim_change_low)

        with _THRESHOLD_LOCK:
            _HEAD_THRESHOLD = new_head_threshold
            _TAIL_THRESHOLD = new_tail_threshold
            _TAIL_CHAR_THRESHOLD = new_tail_char_threshold
            _TAIL_SIM_CHANGE_LOW = new_tail_sim_change_low
            _save_threshold_settings()

        return jsonify({
            "ok": True,
            "message": "thresholds updated",
            "head_threshold": _HEAD_THRESHOLD,
            "tail_threshold": _TAIL_THRESHOLD,
            "tail_char_threshold": _TAIL_CHAR_THRESHOLD,
            "tail_sim_change_low": _TAIL_SIM_CHANGE_LOW,
        })
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"ok": False, "error": f"failed to update thresholds: {e}"}), 500


@app.post("/api/build_category_dataset")
def api_build_category_dataset() -> Any:
    """构造单个类别数据集"""
    try:
        payload = request.get_json(silent=True) or {}
        exports_path = payload.get("exports_path")
        dataset_path = payload.get("dataset_path")
        category = payload.get("category")
        
        if not exports_path or not dataset_path or not category:
            return jsonify({"ok": False, "error": "exports_path, dataset_path and category are required"}), 400
        
        if category not in ["normal", "fake_plate", "change_trailer"]:
            return jsonify({"ok": False, "error": "Invalid category"}), 400
        
        # 导入build_dataset模块
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        import build_dataset
        
        # 更新配置
        build_dataset.EXPORTS_DIR = exports_path
        build_dataset.DATASET_BASE_DIR = dataset_path
        
        # 执行单个类别构建
        build_dataset.build_single_category_dataset(category)
        
        return jsonify({"ok": True, "message": f"{category} dataset built successfully"})
    except Exception as e:
        return jsonify({"ok": False, "error": f"failed to build category dataset: {e}"}), 500


@app.post("/api/build_all_category_datasets")
def api_build_all_category_datasets() -> Any:
    """构造所有类别数据集"""
    try:
        payload = request.get_json(silent=True) or {}
        exports_path = payload.get("exports_path")
        dataset_path = payload.get("dataset_path")
        
        if not exports_path or not dataset_path:
            return jsonify({"ok": False, "error": "exports_path and dataset_path are required"}), 400
        
        # 导入build_dataset模块
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        import build_dataset
        
        # 更新配置
        build_dataset.EXPORTS_DIR = exports_path
        build_dataset.DATASET_BASE_DIR = dataset_path
        
        # 执行所有类别构建
        stats = build_dataset.build_all_category_datasets()
        
        return jsonify({"ok": True, "message": "All category datasets built successfully", "stats": stats})
    except Exception as e:
        return jsonify({"ok": False, "error": f"failed to build all category datasets: {e}"}), 500


@app.post("/api/build_eval_dataset")
def api_build_eval_dataset() -> Any:
    """构造评估数据集"""
    try:
        payload = request.get_json(silent=True) or {}
        exports_path = payload.get("exports_path")
        dataset_path = payload.get("dataset_path")
        total_samples = payload.get("total_samples", 500)
        distribution = payload.get("distribution", {})
        
        if not exports_path or not dataset_path:
            return jsonify({"ok": False, "error": "exports_path and dataset_path are required"}), 400
        
        # 导入build_dataset模块
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        import build_dataset
        
        # 更新配置
        build_dataset.EXPORTS_DIR = exports_path
        build_dataset.DATASET_BASE_DIR = dataset_path
        build_dataset.EVAL_TOTAL = total_samples
        build_dataset.EVAL_DISTRIBUTION = distribution
        
        # 执行评估数据集构建
        build_dataset.build_eval_dataset_only()
        
        # 获取评估数据集统计
        from pathlib import Path
        eval_dir = Path(dataset_path) / "eval_dataset"
        samples_dir = eval_dir / "samples"
        stats = {"total": 0, "normal": 0, "fake_plate": 0, "change_trailer": 0}
        
        if samples_dir.exists():
            dataset_json_path = eval_dir / "dataset.json"
            if dataset_json_path.exists():
                with open(dataset_json_path, "r", encoding="utf-8") as f:
                    dataset_json = json.load(f)
                stats["total"] = dataset_json.get("total_samples", 0)
                stats["normal"] = dataset_json.get("distribution", {}).get("normal", 0)
                stats["fake_plate"] = dataset_json.get("distribution", {}).get("fake_plate", 0)
                stats["change_trailer"] = dataset_json.get("distribution", {}).get("change_trailer", 0)
        
        return jsonify({"ok": True, "message": "Evaluation dataset built successfully", "stats": stats})
    except Exception as e:
        return jsonify({"ok": False, "error": f"failed to build eval dataset: {e}"}), 500


def _resolve_eval_dir(user_path: str) -> str:
    """兼容两种输入: 完整 eval_dataset 路径 或 旧 base_dir(其下含 eval_dataset/)."""
    p = str(user_path or EVAL_DATASET_DIR).rstrip("\\/")
    if os.path.isfile(os.path.join(p, "dataset.json")):
        return p
    if os.path.isfile(os.path.join(p, "eval_dataset", "dataset.json")):
        return os.path.join(p, "eval_dataset")
    return p


def _get_eval_dataset_distribution(eval_dir: str) -> Dict[str, Any]:
    """扫描 eval_dir/samples 下各样本 meta.json 的真值分布（总样本/正常/套牌/换挂）"""
    dist: Dict[str, Any] = {"total": 0, "normal": 0, "fake_plate": 0, "change_trailer": 0}
    samples_root = os.path.join(eval_dir, "samples")
    if not os.path.isdir(samples_root):
        return dist
    for name in sorted(os.listdir(samples_root)):
        meta_path = os.path.join(samples_root, name, "meta.json")
        if not os.path.isfile(meta_path):
            continue
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            case_type = (meta.get("ground_truth") or {}).get("case_type")
        except Exception:
            case_type = None
        dist["total"] += 1
        if case_type in dist:
            dist[case_type] += 1
    return dist


@app.get("/api/dataset_stats")
def api_dataset_stats() -> Any:
    """获取数据集统计信息"""
    try:
        # 从查询参数获取路径，如果没有则使用默认值
        dataset_base_dir = request.args.get("dataset_path", EVAL_DATASET_DIR)
        stats = {
            "normal": 0,
            "fake_plate": 0,
            "change_trailer": 0,
            "eval": 0
        }

        # 统计各类别数据集 (旧 base_dir 布局兼容: base/normal|fake_plate|change_trailer)
        categories = ["normal", "fake_plate", "change_trailer"]
        for category in categories:
            category_dir = os.path.join(dataset_base_dir, category)
            if os.path.exists(category_dir):
                # 统计sample_xxxx目录数量
                count = 0
                for item in os.listdir(category_dir):
                    item_path = os.path.join(category_dir, item)
                    if os.path.isdir(item_path) and item.startswith("sample_"):
                        count += 1
                stats[category] = count

        # 统计评估数据集 (兼容完整 eval_dataset 路径 或 旧 base_dir)
        eval_dir = _resolve_eval_dir(dataset_base_dir)
        if os.path.exists(eval_dir):
            samples_dir = os.path.join(eval_dir, "samples")
            if os.path.exists(samples_dir):
                # 统计sample_xxxx目录数量
                count = 0
                for item in os.listdir(samples_dir):
                    item_path = os.path.join(samples_dir, item)
                    if os.path.isdir(item_path) and item.startswith("sample_"):
                        count += 1
                stats["eval"] = count

        eval_dist = _get_eval_dataset_distribution(eval_dir)
        return jsonify({"ok": True, "stats": stats, "eval_distribution": eval_dist})
    except Exception as e:
        return jsonify({"ok": False, "error": f"failed to get stats: {e}"}), 500


# ==================== 数据集相关常量 ====================

EVAL_DATASET_DIR = r"D:\test_dataset\eval_dataset"
EVAL_RESULTS_DIR = r"D:\test_dataset\eval_results"
EVAL_CATEGORIES = ["normal", "fake_plate", "change_trailer"]


# ==================== 评估运行 ====================

def _update_eval_state(**kw: Any) -> None:
    with _EVAL_STATE_LOCK:
        for k, v in kw.items():
            _EVAL_STATE[k] = v


def _load_eval_dataset(eval_dir: str) -> Optional[Dict[str, Any]]:
    """读取 eval_dir/dataset.json"""
    dataset_json_path = os.path.join(eval_dir, "dataset.json")
    if not os.path.exists(dataset_json_path):
        return None
    with open(dataset_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_sample_meta(sample_dir: str) -> Dict[str, Any]:
    meta_path = os.path.join(sample_dir, "meta.json")
    if not os.path.exists(meta_path):
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _post_predict(base_url: str, payload: Dict[str, Any], timeout: int = 600) -> Tuple[int, Dict[str, Any]]:
    """自调用 /predict，优先用请求方的 host，失败后回退本机回环地址"""
    candidates = [base_url.rstrip("/") + "/predict"]
    try:
        u = urllib.parse.urlsplit(base_url)
        port = u.port or (443 if u.scheme == "https" else 80)
        alt = f"{u.scheme}://127.0.0.1:{port}/predict"
        if alt not in candidates:
            candidates.append(alt)
    except Exception:
        pass

    last_err: Optional[Exception] = None
    for url in candidates:
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            try:
                data = resp.json()
            except Exception:
                data = {}
            return resp.status_code, data
        except Exception as e:
            last_err = e
    return -1, {"ok": False, "error": f"request failed: {last_err}"}


def _map_char_verdict(verdict: Optional[str]) -> Optional[str]:
    """字符检测判定 → EVAL_CATEGORIES 口径.
    "一致"→normal, "不一致"→change_trailer, 无法判断/作废→None (未参与字符判定)."""
    if verdict == "一致":
        return "normal"
    if verdict == "不一致":
        return "change_trailer"
    return None


def _compute_eval_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """双口径指标：正确率 / 误报率 / 漏检率（二分类聚合口径）

    - 误报率 = 正常被判异常(套牌/换挂) / 正常样本总数
    - 漏检率 = 异常被判正常 / 异常样本总数
    stage1 口径用度量学习初判(stage1_case_type)，final 口径用 AI 终判(case_type)
    char_compare 口径：仅统计字符检测做出判定(一致/不一致)的样本，衡量字符检测本身准确度
    """
    def _calc(pred_key: str) -> Dict[str, Any]:
        total = len(results)
        correct = 0
        normal_total = 0
        normal_wrong = 0
        abnormal_total = 0
        abnormal_missed = 0
        for r in results:
            gt = r.get("ground_truth")
            pred = r.get(pred_key)
            if not gt:
                continue
            if pred == gt:
                correct += 1
            if gt == "normal":
                normal_total += 1
                if pred != "normal":
                    normal_wrong += 1
            else:
                abnormal_total += 1
                if pred == "normal":
                    abnormal_missed += 1
        return {
            "accuracy": round(correct / total, 4) if total else None,
            "fpr": round(normal_wrong / normal_total, 4) if normal_total else None,
            "fnr": round(abnormal_missed / abnormal_total, 4) if abnormal_total else None,
        }

    def _calc_char_compare() -> Dict[str, Any]:
        decided = []          # (gt, mapped_pred) 仅字符检测有明确结论的样本
        decided_count = 0
        undecided_count = 0
        for r in results:
            gt = r.get("ground_truth")
            verdict = r.get("char_compare_verdict")
            mapped = _map_char_verdict(verdict)
            if not gt:
                continue
            if mapped is None:
                undecided_count += 1
                continue
            decided_count += 1
            decided.append((gt, mapped))
        total_decided = len(decided)
        correct = sum(1 for gt, pred in decided if pred == gt)
        normal_total = sum(1 for gt, _ in decided if gt == "normal")
        normal_wrong = sum(1 for gt, pred in decided if gt == "normal" and pred != "normal")
        abnormal_total = sum(1 for gt, _ in decided if gt != "normal")
        abnormal_missed = sum(1 for gt, pred in decided if gt != "normal" and pred == "normal")
        return {
            "accuracy": round(correct / total_decided, 4) if total_decided else None,
            "fpr": round(normal_wrong / normal_total, 4) if normal_total else None,
            "fnr": round(abnormal_missed / abnormal_total, 4) if abnormal_total else None,
            "decided": decided_count,
            "undecided": undecided_count,
            "coverage": round(decided_count / (decided_count + undecided_count), 4)
            if (decided_count + undecided_count) else None,
        }

    return {
        "stage1": _calc("stage1_case_type"),
        "final": _calc("case_type"),
        "char_compare": _calc_char_compare(),
    }


def _scan_threshold_grid(
        results: List[Dict[str, Any]],
        head_threshold: float,
        tail_ths: List[float],
        tail_char_ths: List[float],
        recorded_tail_threshold: Optional[float] = None,
        recorded_tail_char_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """阈值网格扫描：用已记录 head_prob/tail_prob/char_compare_verdict 模拟不同阈值组合下的判定.

    额外输出每组合的 ai_count（进入AI复核的样本数）与 est_avg_s（估算平均耗时）：
      耗时基线用记录结果里 lat_ms 中位数分组（AI复核组 / 非AI组），
      参考点默认取 (tail_ths 末位, tail_char_ths 首位)，也可用 recorded_tail_threshold/
      recorded_tail_char_threshold 显式指定记录运行的实际阈值。
      AI组 = proxy_agree_low_sim / proxy_undetermined / proxy_no_prob；
      proxy_head 计入非AI（head AI 成本对所有 tail_th 恒定，比较时抵消）。

    模拟规则 (与 _classify_with_ai_second_judge_internal 的字符检测优先逻辑对齐):
      - head_prob < head_threshold                       → 用记录终判作代理 (head AI 结果未记录,
                                                            线上 head AI 可把低相似度清成 normal)
      - head_prob>=head_th 且 tail_prob>tail_th          → normal (直接通过, 不触发复核)
      - 进入尾部复核 (tail_prob<=tail_th):
          * 字符一致 且 tail_prob>tail_char_th           → normal (char_normal)
          * 字符不一致                                    → change_trailer (char_change)
          * 字符一致但相似度不足 / 字符无法判断 / 字符未运行 → 用记录终判作代理 (proxy_*)
      - head/tail 概率缺失                               → 用记录终判作代理

    注: 因 tail_ths 上限 0.98 >= 记录运行阈值, 需用字符证据的样本(tail_prob<=tail_th)
        在记录运行时同样被标记复核, char_compare_verdict 一定存在, 证据无缺口.
    指标口径与 _calc 一致: accuracy / fpr(正常判异常) / fnr(异常判正常).
    """

    def _simulate(r: Dict[str, Any], tail_th: float, tail_char_th: float) -> Tuple[str, str]:
        head_prob = r.get("head_prob")
        tail_prob = r.get("tail_prob")
        if head_prob is None or tail_prob is None:
            return r.get("case_type") or "abnormal", "proxy_no_prob"
        if head_prob < head_threshold:
            return r.get("case_type") or "abnormal", "proxy_head"
        if tail_prob > tail_th:
            return "normal", "direct_normal"
        if r.get("char_compare_used"):
            mapped = _map_char_verdict(r.get("char_compare_verdict"))
            if mapped == "normal":
                if tail_prob > tail_char_th:
                    return "normal", "char_normal"
                return r.get("case_type") or "abnormal", "proxy_agree_low_sim"
            if mapped == "change_trailer":
                return "change_trailer", "char_change"
        return r.get("case_type") or "abnormal", "proxy_undetermined"

    def _median(values: List[float]) -> Optional[float]:
        if not values:
            return None
        s = sorted(values)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0

    # 耗时基线：用记录阈值把样本分成 AI复核 / 非AI 两组，取中位耗时
    AI_SOURCES = {"proxy_agree_low_sim", "proxy_undetermined", "proxy_no_prob"}
    ref_tail = recorded_tail_threshold if recorded_tail_threshold is not None else (tail_ths[-1] if tail_ths else 0.98)
    ref_char = recorded_tail_char_threshold if recorded_tail_char_threshold is not None else (tail_char_ths[0] if tail_char_ths else 0.85)
    ai_lats: List[float] = []
    fast_lats: List[float] = []
    for r in results:
        if r.get("lat_ms") is None:
            continue
        src = _simulate(r, ref_tail, ref_char)[1]
        if src in AI_SOURCES:
            ai_lats.append(float(r["lat_ms"]))
        else:
            fast_lats.append(float(r["lat_ms"]))
    ai_est = _median(ai_lats) / 1000.0 if ai_lats else None
    fast_est = _median(fast_lats) / 1000.0 if fast_lats else None

    def _metrics(tail_th: float, tail_char_th: float) -> Dict[str, Any]:
        correct = 0
        total = 0
        normal_total = 0
        normal_wrong = 0
        abnormal_total = 0
        abnormal_missed = 0
        src_counts: Dict[str, int] = {}
        for r in results:
            gt = r.get("ground_truth")
            if not gt:
                continue
            pred, src = _simulate(r, tail_th, tail_char_th)
            src_counts[src] = src_counts.get(src, 0) + 1
            total += 1
            if pred == gt:
                correct += 1
            if gt == "normal":
                normal_total += 1
                if pred != "normal":
                    normal_wrong += 1
            else:
                abnormal_total += 1
                if pred == "normal":
                    abnormal_missed += 1
        ai_count = sum(src_counts.get(s, 0) for s in AI_SOURCES)
        est_avg_s = None
        if ai_est is not None and fast_est is not None and total:
            est_avg_s = round((ai_count * ai_est + (total - ai_count) * fast_est) / total, 2)
        return {
            "tail_threshold": tail_th,
            "tail_char_threshold": tail_char_th,
            "accuracy": round(correct / total, 4) if total else None,
            "correct": correct,
            "total": total,
            "fpr": round(normal_wrong / normal_total, 4) if normal_total else None,
            "normal_total": normal_total,
            "normal_wrong": normal_wrong,
            "fnr": round(abnormal_missed / abnormal_total, 4) if abnormal_total else None,
            "abnormal_total": abnormal_total,
            "abnormal_missed": abnormal_missed,
            "ai_count": ai_count,
            "est_avg_s": est_avg_s,
            "lat_ai_median_s": ai_est,
            "lat_fast_median_s": fast_est,
            "source_counts": src_counts,
        }

    grid = [_metrics(th, cth) for th in tail_ths for cth in tail_char_ths]
    return {"head_threshold": head_threshold, "grid": grid}


def _pick_best_threshold_combo(
        grid: List[Dict[str, Any]],
        max_fnr: float,
        prefer: str = "fpr",
) -> Dict[str, Any]:
    """从扫描网格中选最优组合.

    prefer="fpr" (计划默认): 在 fnr<=max_fnr 的组合里选 fpr 最低;
    平分时取 accuracy 更高, 再取 tail_threshold 更低(减少AI复核).
    prefer="accuracy": 忽略 fnr 约束, 取 accuracy 最高.
    无满足条件组合时返回 None.
    """
    candidates = [c for c in grid if c.get("fnr") is not None and c["fnr"] <= max_fnr]
    if not candidates:
        return None
    if prefer == "accuracy":
        return max(
            candidates,
            key=lambda c: (
                c["accuracy"] if c.get("accuracy") is not None else -1,
                -c["fpr"] if c.get("fpr") is not None else 1e9,
                -c["tail_threshold"],
            ),
        )
    return min(
        candidates,
        key=lambda c: (
            c["fpr"] if c.get("fpr") is not None else 1e9,
            -(c["accuracy"] if c.get("accuracy") is not None else -1),
            c["tail_threshold"],
        ),
    )


def _build_eval_summary(results: List[Dict[str, Any]], head_threshold: float, tail_threshold: float, dataset_path: str, run_id: Optional[str] = None) -> Dict[str, Any]:
    """汇总：总览 + 每类准确率 + 混淆矩阵 + 双口径指标 + 平均耗时"""
    total = len(results)
    ok_count = sum(1 for r in results if r.get("ok"))
    hit_count = sum(1 for r in results if r.get("hit"))
    stage1_hit_count = sum(1 for r in results if r.get("hit_stage1"))

    lat_values = [float(r["lat_ms"]) for r in results if r.get("lat_ms") is not None]
    avg_lat_ms = round(sum(lat_values) / len(lat_values), 1) if lat_values else None

    confusion: Dict[str, Dict[str, int]] = {
        gt: {pred: 0 for pred in EVAL_CATEGORIES}
        for gt in EVAL_CATEGORIES
    }
    per_category: Dict[str, Dict[str, Any]] = {}

    for gt in EVAL_CATEGORIES:
        gt_results = [r for r in results if r.get("ground_truth") == gt]
        hits = sum(1 for r in gt_results if r.get("hit"))
        stage1_hits = sum(1 for r in gt_results if r.get("hit_stage1"))
        per_category[gt] = {
            "count": len(gt_results),
            "hit": hits,
            "hit_stage1": stage1_hits,
            "accuracy": round(hits / len(gt_results), 4) if gt_results else None,
            "stage1_accuracy": round(stage1_hits / len(gt_results), 4) if gt_results else None,
        }
        for r in gt_results:
            pred = r.get("case_type")
            if pred in confusion[gt]:
                confusion[gt][pred] += 1

    return {
        "run_id": run_id,
        "dataset_path": dataset_path,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "head_threshold": head_threshold,
        "tail_threshold": tail_threshold,
        "total": total,
        "ok": ok_count,
        "abnormal": total - ok_count,
        "hit": hit_count,
        "hit_stage1": stage1_hit_count,
        "overall_accuracy": round(hit_count / total, 4) if total else None,
        "stage1_accuracy": round(stage1_hit_count / total, 4) if total else None,
        "avg_lat_ms": avg_lat_ms,
        "metrics": _compute_eval_metrics(results),
        "per_category": per_category,
        "confusion_matrix": confusion,
        "results": results,
    }


def _run_evaluation_background(eval_dir: str, results_path: str, base_url: str) -> None:
    """后台执行评估：逐组调用 /predict，把结果写入 runs/{run_id}/ 多轮目录"""
    errors: List[Dict[str, Any]] = []
    try:
        _update_eval_state(
            running=True, total=0, processed=0, success=0, failed=0,
            current_index=0, current_sample="", message="开始加载评估数据集...",
            errors=[], results=[], started_at=datetime.datetime.now().isoformat(timespec="seconds"),
            finished_at=None, run_id=None,
        )

        dataset = _load_eval_dataset(eval_dir)
        if not dataset:
            _update_eval_state(
                running=False, message="未找到 dataset.json",
                finished_at=datetime.datetime.now().isoformat(timespec="seconds"),
            )
            return

        samples = dataset.get("samples", [])
        total = len(samples)
        if total == 0:
            _update_eval_state(
                running=False, message="评估数据集为空",
                finished_at=datetime.datetime.now().isoformat(timespec="seconds"),
            )
            return

        # 随机打乱评估顺序（不重复），避免每次按固定顺序读取
        random.shuffle(samples)

        results_path = os.path.abspath(results_path)
        os.makedirs(results_path, exist_ok=True)

        head_threshold = _HEAD_THRESHOLD
        tail_threshold = _TAIL_THRESHOLD

        # 本轮 run 目录
        run_id = "run_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        runs_dir = _get_eval_runs_dir(results_path)
        run_dir = os.path.join(runs_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        _update_eval_state(run_id=run_id, message=f"共 {total} 组，开始评估...")

        done_results: List[Dict[str, Any]] = []
        success_count = 0
        failed_count = 0

        _update_eval_state(total=total)

        for idx, sample in enumerate(samples):
            sample_id = sample.get("sample_id", f"sample_{idx + 1:04d}")
            _update_eval_state(
                current_index=idx + 1, current_sample=sample_id, processed=idx + 1,
                success=success_count, failed=failed_count,
                message=f"正在评估 {sample_id} ({idx + 1}/{total})...",
            )

            sample_dir = os.path.join(eval_dir, "samples", sample_id)
            sample_meta = _load_sample_meta(sample_dir)
            image_paths = sample_meta.get("image_paths", {})
            ground_truth = sample_meta.get("ground_truth", {})

            # 相对路径转绝对路径，仅传存在的图
            payload: Dict[str, str] = {}
            for key in ("path1", "path2", "path3", "path4"):
                rel = image_paths.get(key)
                if rel:
                    abs_path = os.path.join(sample_dir, rel)
                    if os.path.isfile(abs_path):
                        payload[key] = abs_path

            if not payload.get("path1") or not payload.get("path2"):
                entry = {
                    "sample_id": sample_id,
                    "ok": False,
                    "case_type": "abnormal",
                    "ground_truth": ground_truth.get("case_type"),
                    "error": "样本主图缺失",
                    "hit": False,
                    "hit_stage1": False,
                    "stage1_case_type": None,
                    "lat_ms": None,
                    "record_id": None,
                }
                done_results.append(entry)
                failed_count += 1
                errors.append({"sample_id": sample_id, "error": "样本主图缺失"})
                continue

            status_code, resp_data = _post_predict(base_url, payload)

            record_id = resp_data.get("record_id")
            prediction = resp_data.get("case_type")
            is_ok = bool(resp_data.get("ok"))
            head_prob = resp_data.get("head_prob")
            tail_prob = resp_data.get("tail_prob")
            gt_type = ground_truth.get("case_type")
            hit = bool(gt_type) and is_ok and prediction == gt_type

            # 度量学习初判（用运行时刻阈值重算，保证两口径一致）
            stage1_type = _classify_with_thresholds(head_prob, tail_prob, head_threshold, tail_threshold)
            hit_stage1 = bool(gt_type) and is_ok and stage1_type == gt_type

            entry = {
                "sample_id": sample_id,
                "ok": is_ok,
                "case_type": prediction,
                "stage1_case_type": stage1_type,
                "head_prob": head_prob,
                "tail_prob": tail_prob,
                "record_id": record_id,
                "hit": hit,
                "hit_stage1": hit_stage1,
                "ground_truth": gt_type,
                "input_mode": resp_data.get("input_mode"),
                "http_status": status_code,
                "lat_ms": resp_data.get("lat_ms"),
                "char_ms": resp_data.get("char_ms"),
                "error": resp_data.get("error"),
                "final_diff_summary": resp_data.get("final_diff_summary"),
                "char_compare_used": resp_data.get("char_compare_used", False),
                "char_compare_verdict": resp_data.get("char_compare_verdict"),
                "char_compare_plate_type": resp_data.get("char_compare_plate_type"),
                "char_compare_R": resp_data.get("char_compare_R"),
                "char_compare_M": resp_data.get("char_compare_M"),
                "char_compare_U": resp_data.get("char_compare_U"),
                "char_compare_p3_seq": resp_data.get("char_compare_p3_seq"),
                "char_compare_p4_seq": resp_data.get("char_compare_p4_seq"),
                "timing_ms": resp_data.get("timing_ms"),
            }
            done_results.append(entry)

            if is_ok:
                success_count += 1
            else:
                failed_count += 1
                if resp_data.get("error"):
                    errors.append({"sample_id": sample_id, "error": resp_data.get("error")})

            # 写入本轮样本目录
            sample_result_dir = os.path.join(run_dir, "samples", sample_id)
            os.makedirs(sample_result_dir, exist_ok=True)

            with open(os.path.join(sample_result_dir, "eval_result.json"), "w", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False, indent=2)

            # 运行记录 meta.json（合并样本信息）
            run_meta: Dict[str, Any] = {}
            if record_id:
                rec = _METRICS.get_record(record_id)
                if rec:
                    run_meta = dict(rec)
            run_meta["sample_id"] = sample_id
            run_meta["ground_truth"] = ground_truth
            run_meta["hit"] = hit
            run_meta["hit_stage1"] = hit_stage1
            run_meta["stage1_case_type"] = stage1_type
            run_meta["eval_run_id"] = run_id
            run_meta["eval_run_at"] = datetime.datetime.now().isoformat(timespec="seconds")
            with open(os.path.join(sample_result_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(run_meta, f, ensure_ascii=False, indent=2)

        # 汇总（新格式：含双口径指标 + 平均耗时）
        summary = _build_eval_summary(done_results, head_threshold, tail_threshold, eval_dir, run_id=run_id)
        with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        with open(os.path.join(run_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(done_results, f, ensure_ascii=False, indent=2)

        _update_eval_state(
            running=False, success=success_count, failed=failed_count,
            processed=total, message="评估完成", errors=errors, results=done_results,
            finished_at=datetime.datetime.now().isoformat(timespec="seconds"),
            metrics=summary.get("metrics"),
            avg_lat_ms=summary.get("avg_lat_ms"),
            per_category=summary.get("per_category"),
        )
        _cleanup_old_eval_runs(results_path)
        print(f"[eval] {run_id} 评估完成，共 {total} 组，命中 {len([r for r in done_results if r.get('hit')])} 组，结果写入 {run_dir}", flush=True)
    except Exception as e:
        errors.append({"error": str(e)})
        _update_eval_state(
            running=False, message=f"评估异常: {e}", errors=errors,
            finished_at=datetime.datetime.now().isoformat(timespec="seconds"),
        )
        print(f"[eval] evaluation failed: {e}", flush=True)


def _get_eval_runs_dir(results_path: str) -> str:
    return os.path.join(results_path, "runs")


def _list_eval_runs(results_path: str) -> List[Dict[str, Any]]:
    """列出该结果目录下的所有评估运行（读各 run 的 summary.json）"""
    results_path = os.path.abspath(results_path)
    _migrate_legacy_eval_runs(results_path)
    runs_dir = _get_eval_runs_dir(results_path)
    runs: List[Dict[str, Any]] = []
    if os.path.isdir(runs_dir):
        for name in os.listdir(runs_dir):
            run_dir = os.path.join(runs_dir, name)
            summary_path = os.path.join(run_dir, "summary.json")
            if os.path.isdir(run_dir) and os.path.exists(summary_path):
                try:
                    with open(summary_path, "r", encoding="utf-8") as f:
                        summary = json.load(f)
                    summary["run_id"] = summary.get("run_id") or name
                    runs.append(summary)
                except Exception:
                    continue
    runs.sort(key=lambda x: x.get("run_id", ""), reverse=True)
    return runs


def _migrate_legacy_eval_runs(results_path: str) -> None:
    """旧版平铺结果(eval_results/sample_*/ + summary.json)封装成一个 run 并删除"""
    results_path = os.path.abspath(results_path)
    summary_path = os.path.join(results_path, "summary.json")
    if os.path.exists(_get_eval_runs_dir(results_path)) or not os.path.exists(summary_path):
        return
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            old = json.load(f)
    except Exception:
        return
    old_results = old.get("results") or []
    sample_dirs = [d for d in os.listdir(results_path)
                   if os.path.isdir(os.path.join(results_path, d)) and d.startswith("sample_")]
    if not old_results and not sample_dirs:
        return

    # run_id 由旧 summary 生成时间解析，失败则用文件 mtime
    run_id = None
    gen = old.get("generated_at") or ""
    try:
        dt = datetime.datetime.fromisoformat(gen.replace("Z", "+00:00"))
        run_id = "run_" + dt.strftime("%Y%m%d_%H%M%S")
    except Exception:
        try:
            dt = datetime.datetime.fromtimestamp(os.path.getmtime(summary_path))
            run_id = "run_" + dt.strftime("%Y%m%d_%H%M%S")
        except Exception:
            run_id = "run_legacy"

    run_dir = os.path.join(_get_eval_runs_dir(results_path), run_id)
    os.makedirs(os.path.join(run_dir, "samples"), exist_ok=True)

    head_threshold = float(old.get("head_threshold") or _DEFAULT_HEAD_THRESHOLD)
    tail_threshold = float(old.get("tail_threshold") or _DEFAULT_TAIL_THRESHOLD)
    migrated_results: List[Dict[str, Any]] = []
    for r in old_results:
        r = dict(r)
        gt = r.get("ground_truth")
        stage1 = _classify_with_thresholds(r.get("head_prob"), r.get("tail_prob"), head_threshold, tail_threshold)
        r.setdefault("stage1_case_type", stage1)
        r.setdefault("hit_stage1", bool(gt) and stage1 == gt)
        r.setdefault("lat_ms", None)
        r.setdefault("hit", bool(r.get("hit")))
        migrated_results.append(r)
        # 旧 sample 目录里的 eval_result.json / meta.json 原样搬入
        sid = r.get("sample_id")
        if sid:
            old_sdir = os.path.join(results_path, sid)
            new_sdir = os.path.join(run_dir, "samples", sid)
            os.makedirs(new_sdir, exist_ok=True)
            for fname in ("eval_result.json", "meta.json"):
                src = os.path.join(old_sdir, fname)
                dst = os.path.join(new_sdir, fname)
                if os.path.exists(src):
                    try:
                        shutil.copy2(src, dst)
                    except Exception:
                        pass

    summary = _build_eval_summary(migrated_results, head_threshold, tail_threshold, old.get("dataset_path") or "", run_id=run_id)
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(run_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(migrated_results, f, ensure_ascii=False, indent=2)

    # 删除旧平铺目录与旧 summary.json
    for d in sample_dirs:
        shutil.rmtree(os.path.join(results_path, d), ignore_errors=True)
    try:
        os.remove(summary_path)
    except Exception:
        pass
    print(f"[eval] 已迁移旧版平铺结果到 {run_id}", flush=True)


def _cleanup_old_eval_runs(results_path: str, days: int = 30) -> int:
    """删除早于 days 天的评估运行目录，返回删除数量"""
    results_path = os.path.abspath(results_path)
    runs_dir = _get_eval_runs_dir(results_path)
    if not os.path.isdir(runs_dir):
        return 0
    cutoff = datetime.datetime.now() - datetime.timedelta(days=days)
    removed = 0
    for name in os.listdir(runs_dir):
        run_dir = os.path.join(runs_dir, name)
        if not os.path.isdir(run_dir):
            continue
        dt = None
        ts = name[len("run_"):] if name.startswith("run_") else name
        try:
            dt = datetime.datetime.strptime(ts, "%Y%m%d_%H%M%S")
        except Exception:
            try:
                dt = datetime.datetime.fromtimestamp(os.path.getmtime(run_dir))
            except Exception:
                dt = None
        if dt is not None and dt < cutoff:
            shutil.rmtree(run_dir, ignore_errors=True)
            removed += 1
    if removed:
        print(f"[eval] 自动清理 {removed} 个超过 {days} 天的评估运行", flush=True)
    return removed


@app.get("/api/eval/runs")
def api_eval_runs() -> Any:
    """列出评估运行记录（对比表数据）"""
    try:
        results_path = request.args.get("results_path") or EVAL_RESULTS_DIR
        return jsonify({"ok": True, "results_path": os.path.abspath(results_path), "runs": _list_eval_runs(results_path)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.get("/api/eval/runs/<run_id>")
def api_eval_run_detail(run_id: str) -> Any:
    """单个评估运行详情（summary + results）"""
    try:
        results_path = request.args.get("results_path") or EVAL_RESULTS_DIR
        run_dir = os.path.join(_get_eval_runs_dir(os.path.abspath(results_path)), run_id)
        summary_path = os.path.join(run_dir, "summary.json")
        if not os.path.isdir(run_dir) or not os.path.exists(summary_path):
            return jsonify({"ok": False, "error": "评估运行不存在"}), 404
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        results_path_ = os.path.join(run_dir, "results.json")
        if os.path.exists(results_path_):
            with open(results_path_, "r", encoding="utf-8") as f:
                summary["results"] = json.load(f)
        return jsonify({"ok": True, "run": summary})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/api/eval/threshold_scan")
def api_eval_threshold_scan() -> Any:
    """阈值网格扫描：对已完成评估 run 的 results.json 重新模拟判定, 输出准确率/误报率/漏检率表格.

    body:
      run_id: str           (必填) 目标评估运行
      results_path: str     (可选) 结果根目录, 默认 EVAL_RESULTS_DIR
      head_threshold: float (可选) 默认取 run summary 的 head_threshold
      tail_ths: [float]     (可选) 尾部相似度阈值扫描集, 默认 [0.80..0.98]
      tail_char_ths: [float](可选) 字符一致放行阈值扫描集, 默认 [0.70..0.95]
      max_fnr: float        (可选) 选优时漏检率上限, 默认 0.10
      prefer: str           (可选) "fpr"|"accuracy", 默认 "fpr"
      recorded_tail_threshold / recorded_tail_char_threshold (可选):
        耗时估算参考点(记录运行实际阈值), 默认取当前生效阈值
    每网格行含 ai_count(进AI复核样本数) 与 est_avg_s(估算平均耗时,秒).
    """
    try:
        payload = request.get_json(silent=True) or {}
        run_id = payload.get("run_id")
        if not run_id:
            return jsonify({"ok": False, "error": "缺少 run_id"}), 400
        results_path = os.path.abspath(payload.get("results_path") or EVAL_RESULTS_DIR)
        run_dir = os.path.join(_get_eval_runs_dir(results_path), run_id)
        summary_path = os.path.join(run_dir, "summary.json")
        results_path_ = os.path.join(run_dir, "results.json")
        if not os.path.isdir(run_dir) or not os.path.exists(summary_path):
            return jsonify({"ok": False, "error": "评估运行不存在"}), 404
        if not os.path.exists(results_path_):
            return jsonify({"ok": False, "error": "该运行无 results.json（无样本级记录）"}), 400

        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        with open(results_path_, "r", encoding="utf-8") as f:
            results = json.load(f)

        head_threshold = float(payload.get("head_threshold") or summary.get("head_threshold") or _DEFAULT_HEAD_THRESHOLD)
        tail_ths = [float(x) for x in (payload.get("tail_ths") or [0.80, 0.82, 0.85, 0.88, 0.90, 0.92, 0.95, 0.98])]
        tail_char_ths = [float(x) for x in (payload.get("tail_char_ths") or [0.70, 0.75, 0.80, 0.85, 0.90, 0.95])]
        max_fnr = float(payload.get("max_fnr") if payload.get("max_fnr") is not None else 0.10)
        prefer = str(payload.get("prefer") or "fpr")
        # 耗时估算的参考点：记录运行实际阈值，默认取当前生效阈值
        recorded_tail = float(payload["recorded_tail_threshold"]) if payload.get("recorded_tail_threshold") is not None else _TAIL_THRESHOLD
        recorded_char = float(payload["recorded_tail_char_threshold"]) if payload.get("recorded_tail_char_threshold") is not None else _TAIL_CHAR_THRESHOLD

        scan = _scan_threshold_grid(results, head_threshold, tail_ths, tail_char_ths,
                                    recorded_tail_threshold=recorded_tail,
                                    recorded_tail_char_threshold=recorded_char)
        grid = scan["grid"]
        best_fpr = _pick_best_threshold_combo(grid, max_fnr, prefer="fpr")
        best_accuracy = _pick_best_threshold_combo(grid, max_fnr, prefer="accuracy")
        # 不受 fnr 约束的最优，始终给用户一个可参考候选
        best_fpr_unconstrained = _pick_best_threshold_combo(grid, 1.0, prefer="fpr")
        best_accuracy_unconstrained = _pick_best_threshold_combo(grid, 1.0, prefer="accuracy")

        return jsonify({
            "ok": True,
            "run_id": run_id,
            "sample_count": len(results),
            "head_threshold": head_threshold,
            "tail_ths": tail_ths,
            "tail_char_ths": tail_char_ths,
            "max_fnr": max_fnr,
            "recorded_tail_threshold": recorded_tail,
            "recorded_tail_char_threshold": recorded_char,
            "grid": grid,
            "best_fpr": best_fpr,
            "best_accuracy": best_accuracy,
            "best_fpr_unconstrained": best_fpr_unconstrained,
            "best_accuracy_unconstrained": best_accuracy_unconstrained,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.delete("/api/eval/runs/<run_id>")
def api_eval_run_delete(run_id: str) -> Any:
    """删除一个评估运行记录"""
    try:
        results_path = request.args.get("results_path") or EVAL_RESULTS_DIR
        run_dir = os.path.join(_get_eval_runs_dir(os.path.abspath(results_path)), run_id)
        if not os.path.isdir(run_dir):
            return jsonify({"ok": False, "error": "评估运行不存在"}), 404
        shutil.rmtree(run_dir, ignore_errors=True)
        return jsonify({"ok": True, "message": f"已删除 {run_id}"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/api/records/by_ids")
def api_records_by_ids() -> Any:
    """按 record_id 列表批量查询记录（用于评估页导出的 ID 回查）"""
    try:
        payload = request.get_json(silent=True) or {}
        record_ids = payload.get("record_ids", [])
        if not isinstance(record_ids, list):
            return jsonify({"ok": False, "error": "record_ids 必须是数组"}), 400
        ids = [str(x).strip() for x in record_ids if str(x).strip()]
        found: List[Dict[str, Any]] = []
        missing: List[str] = []
        for rid in ids:
            rec = _METRICS.get_record(rid)
            if rec and not rec.get("deleted"):
                found.append(rec)
            else:
                missing.append(rid)
        return jsonify({"ok": True, "records": found, "total": len(found), "requested": len(ids), "missing": missing})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/api/run_evaluation")
def api_run_evaluation() -> Any:
    """启动评估：后台逐组调用 /predict 并写入 eval_results"""
    try:
        payload = request.get_json(silent=True) or {}
        eval_dir = _resolve_eval_dir(payload.get("dataset_path") or EVAL_DATASET_DIR)
        results_path = str(payload.get("results_path") or EVAL_RESULTS_DIR).rstrip("\\/")

        with _EVAL_STATE_LOCK:
            if _EVAL_STATE.get("running"):
                return jsonify({"ok": False, "error": "评估已在运行中"}), 409
            _EVAL_STATE["running"] = True

        base_url = request.host_url
        threading.Thread(
            target=_run_evaluation_background,
            args=(eval_dir, results_path, base_url),
            daemon=True,
        ).start()
        return jsonify({"ok": True, "message": "评估已开始"})
    except Exception as e:
        with _EVAL_STATE_LOCK:
            _EVAL_STATE["running"] = False
        return jsonify({"ok": False, "error": f"failed to start evaluation: {e}"}), 500


@app.get("/api/eval_progress")
def api_eval_progress() -> Any:
    """获取评估进度"""
    with _EVAL_STATE_LOCK:
        state = dict(_EVAL_STATE)
        state["errors"] = list(_EVAL_STATE["errors"])
        state["results"] = list(_EVAL_STATE["results"])
    return jsonify({"ok": True, "state": state})


if __name__ == "__main__":
    try:
        _cleanup_old_eval_runs(EVAL_RESULTS_DIR)
    except Exception as e:
        print(f"[eval] 启动清理评估运行失败: {e}", flush=True)
    # 启动前预热方案B字符检测, 避免首次请求等待模型加载
    _warmup_char_reader()
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8001"))
    app.run(host=host, port=port, threaded=True)
