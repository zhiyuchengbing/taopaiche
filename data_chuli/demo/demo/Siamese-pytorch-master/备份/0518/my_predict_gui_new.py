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
from collections import deque
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import cv2
from PIL import Image
from flask import Flask, jsonify, request, render_template, send_file, send_from_directory
from ultralytics import YOLO

from siamese import Siamese
from data_tran.image_resolver import ImagePathResolver
from qwen_vl.predict_ai import VehicleCheck
from qwen_vl.predict_ai_shijiao2 import TailVehicleCheck
from paddle_ocr.ocr_detect import MaxBoxOCR
from chewei_detect.chewei_detect import VehicleCropper as TailViewCropper

parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from data_chuli.cropper import VehicleCropper as MainVehicleCropper

app = Flask(__name__)

_INIT_LOCK = threading.Lock()
_PIPELINE_LOCK = threading.Lock()
_INITIALIZED = False

_CROPPER: Optional[MainVehicleCropper] = None
_HEAD_MODEL: Optional[Siamese] = None
_TAIL_MODEL: Optional[Siamese] = None
_HEADTAIL_MODEL: Optional[YOLO] = None
_TAIL_VIEW_CROPPER: Optional[TailViewCropper] = None
_IMAGE_RESOLVER: Optional[ImagePathResolver] = None
_AI_CHECKER: Optional[VehicleCheck] = None
_AI_TAIL_CHECKER: Optional[TailVehicleCheck] = None
_OCR_CHECKER: Optional[MaxBoxOCR] = None
_OCR_LOCK = threading.Lock()

_DEFAULT_HEAD_THRESHOLD = float(os.environ.get("HEAD_THRESHOLD_DEFAULT", "0.8"))
_DEFAULT_TAIL_THRESHOLD = float(os.environ.get("TAIL_THRESHOLD_DEFAULT", "0.8"))
_DIRECT_FAKE_PLATE_HEAD_THRESHOLD = float(os.environ.get("DIRECT_FAKE_PLATE_HEAD_THRESHOLD", "0.1"))
_HEAD_OCR_MIN_AREA = float(os.environ.get("HEAD_OCR_MIN_AREA", "15000"))
_HEAD_OCR_AI_RECHECK_THRESHOLD = float(os.environ.get("HEAD_OCR_AI_RECHECK_THRESHOLD", "0.8"))
_THRESHOLDS_FILE = os.path.join(os.path.dirname(__file__), "thresholds.json")
_THRESHOLD_LOCK = threading.Lock()
_HEAD_THRESHOLD: float = _DEFAULT_HEAD_THRESHOLD
_TAIL_THRESHOLD: float = _DEFAULT_TAIL_THRESHOLD


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
    }
    with open(_THRESHOLDS_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _load_threshold_settings() -> None:
    global _HEAD_THRESHOLD, _TAIL_THRESHOLD

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
    except Exception as e:
        print(f"[thresholds] failed to load {_THRESHOLDS_FILE}: {e}")
        _HEAD_THRESHOLD = _DEFAULT_HEAD_THRESHOLD
        _TAIL_THRESHOLD = _DEFAULT_TAIL_THRESHOLD


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

            # 保存6张处理后的图片
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
                for key in [
                    "original1", "original2", "original3", "original4",
                    "tail_view_crop3", "tail_view_crop4",
                ]:
                    data_url = original_images.get(key, "")
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
            include_deleted: bool = False,
            limit: int = 50,
            offset: int = 0
    ) -> Dict[str, Any]:
        """
        查询记录列表

        Args:
            start_date: 开始日期 YYYY-MM-DD
            end_date: 结束日期 YYYY-MM-DD
            case_type: 类型筛选 normal/fake_plate/change_trailer/all
            include_deleted: 是否包含已删除记录
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

                                # 筛选条件
                                if not include_deleted and record.get("deleted", False):
                                    continue

                                if case_type and case_type != "all":
                                    if record.get("case_type") != case_type:
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
                               "tail_view_crop3", "tail_view_crop4", "vehicle1", "vehicle2",
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
        tail_ai_mode: str = "",
        stage1_case_type: str = "",
        tail_second_check_used: bool = False,
        tail_second_check_result: str = "",
        tail_second_check_reason: str = "",
        tail_number_consistency: Optional[str] = None,
        tail_structure_consistency: Optional[str] = None,
        ocr_used: bool = False,
        ocr_match: Optional[bool] = None,
        ocr_text1: Optional[str] = None,
        ocr_text2: Optional[str] = None,
        ocr_error: Optional[str] = None,
        diff_desc: Optional[str] = None,
        diff_analyzed_part: Optional[str] = None,
        ai_diff_ms: Optional[float] = None,
        head_ai_display_text: Optional[str] = None,
        tail34_ai_display_text: Optional[str] = None,
        main_tail_ai_display_text: Optional[str] = None,
        final_diff_summary: Optional[str] = None,
) -> Optional[str]:
    """
    记录指标并保存图片

    Args:
        original_images: 包含原始图片的字典 {"original1": data_url, "original2": data_url}
        diff_desc: AI细粒度差异分析描述
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
            "tail_ai_mode": tail_ai_mode,
            "stage1_case_type": stage1_case_type,
            "tail_second_check_used": bool(tail_second_check_used),
            "tail_second_check_result": tail_second_check_result,
            "tail_second_check_reason": tail_second_check_reason,
            "tail_number_consistency": tail_number_consistency,
            "tail_structure_consistency": tail_structure_consistency,
            "ocr_used": bool(ocr_used),
            "ocr_match": ocr_match,
            "ocr_text1": ocr_text1,
            "ocr_text2": ocr_text2,
            "ocr_error": ocr_error,
            "head_ai_display_text": head_ai_display_text,
            "tail34_ai_display_text": tail34_ai_display_text,
            "main_tail_ai_display_text": main_tail_ai_display_text,
            "final_diff_summary": final_diff_summary,
            "endpoint": endpoint,
            "source": source,
            "lat_ms": lat_ms,
            "protected": False,
            "deleted": False,
            "note": "",
        }

        # 添加差异分析信息到meta
        if diff_desc:
            meta["diff_desc"] = diff_desc
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
        "tail_ai_mode": tail_ai_mode,
        "stage1_case_type": stage1_case_type,
        "tail_second_check_used": bool(tail_second_check_used),
        "tail_second_check_result": tail_second_check_result,
        "tail_second_check_reason": tail_second_check_reason,
        "tail_number_consistency": tail_number_consistency,
        "tail_structure_consistency": tail_structure_consistency,
        "ocr_used": bool(ocr_used),
        "ocr_match": ocr_match,
        "ocr_text1": ocr_text1,
        "ocr_text2": ocr_text2,
        "ocr_error": ocr_error,
        "head_ai_display_text": head_ai_display_text,
        "tail34_ai_display_text": tail34_ai_display_text,
        "main_tail_ai_display_text": main_tail_ai_display_text,
        "final_diff_summary": final_diff_summary,
    }

    if record_id:
        ev["record_id"] = record_id
        ev["image_dir"] = image_dir
        # 添加差异分析信息到日志
        if diff_desc:
            ev["diff_desc"] = diff_desc
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
    global _INITIALIZED, _CROPPER, _HEAD_MODEL, _TAIL_MODEL, _HEADTAIL_MODEL, _TAIL_VIEW_CROPPER, _IMAGE_RESOLVER, _AI_CHECKER, _AI_TAIL_CHECKER
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

        _INITIALIZED = True


def _ai_second_judge_enabled() -> bool:
    """检查AI二次判断是否启用"""
    raw = str(os.environ.get("AI_SECOND_JUDGE_ENABLED", "1")).strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _head_ocr_enabled() -> bool:
    """检查车头OCR预比对是否启用"""
    raw = str(os.environ.get("HEAD_OCR_PRECHECK_ENABLED", "1")).strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _get_ocr_checker() -> Optional[MaxBoxOCR]:
    global _OCR_CHECKER

    if not _head_ocr_enabled():
        return None
    if _OCR_CHECKER is not None:
        return _OCR_CHECKER

    with _OCR_LOCK:
        if _OCR_CHECKER is not None:
            return _OCR_CHECKER
        try:
            _OCR_CHECKER = MaxBoxOCR()
            print("[predict] 车头OCR预比对已启用")
        except Exception as e:
            print(f"[predict] failed to initialize head OCR checker: {e}")
            _OCR_CHECKER = None
    return _OCR_CHECKER


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
        "diff_desc": None,
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
    }


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


def _shorten_reason_text(reason: Optional[str], limit: int = 80) -> str:
    text = _clean_reason_text(reason)
    if not text:
        return ""
    for sep in ("；", "。", "\n"):
        if sep in text:
            text = text.split(sep)[0].strip()
            break
    if len(text) > limit:
        text = text[:limit].rstrip("，,；;。.") + "..."
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
        ocr_match = result.get("ocr_match")
        text1 = str(result.get("ocr_text1") or "").strip() or "-"
        text2 = str(result.get("ocr_text2") or "").strip() or "-"

        if head_prob is not None and head_prob < _DIRECT_FAKE_PLATE_HEAD_THRESHOLD:
            final_diff_summary = "套牌：车头相似度过低，直接判定为套牌"
        elif (
            (not head_ai_used)
            and head_prob is not None
            and head_prob <= _HEAD_THRESHOLD
            and ocr_match is False
        ):
            final_diff_summary = f"套牌：车头相似度低于阈值，车头OCR为“{text1} / {text2}”，判定为套牌"
        elif head_ai_used and ai_head_result == "fake_plate":
            short_reason = _shorten_reason_text(result.get("ai_head_reason")) or "车头AI判定为套牌"
            final_diff_summary = f"套牌：{short_reason}"
        elif ocr_match is False:
            final_diff_summary = f"套牌：车头OCR不一致：'{text1}' vs '{text2}'"
    elif case_type == "change_trailer":
        if main_tail_ai_display_text:
            short_reason = _shorten_reason_text(result.get("ai_tail_reason"))
            final_diff_summary = f"换挂：{short_reason}" if short_reason else "换挂"
        elif tail34_ai_display_text:
            short_reason = _shorten_reason_text(result.get("tail_second_check_reason"))
            final_diff_summary = f"换挂：{short_reason}" if short_reason else "换挂"
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


def _run_head_ocr_precheck(cropped_pils: Optional[Dict[str, Image.Image]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "ocr_used": False,
        "ocr_match": None,
        "ocr_text1": None,
        "ocr_text2": None,
        "ocr_error": None,
    }

    checker = _get_ocr_checker()
    if checker is None:
        if _head_ocr_enabled():
            result["ocr_error"] = "ocr checker unavailable"
        return _populate_ai_trace_texts(result, head_prob)

    if not cropped_pils:
        result["ocr_error"] = "head crops missing"
        return _populate_ai_trace_texts(result, head_prob)

    h1 = cropped_pils.get("h1")
    h2 = cropped_pils.get("h2")
    if h1 is None or h2 is None:
        result["ocr_error"] = "head crops missing"
        return result

    p1 = _save_pil_to_temp(h1, prefix="ocr_head1")
    p2 = _save_pil_to_temp(h2, prefix="ocr_head2")
    temp_files = [p for p in [p1, p2] if p]
    try:
        if not p1 or not p2:
            result["ocr_error"] = "failed to save head crops for ocr"
            return result

        result["ocr_used"] = True
        ocr_result1 = checker.get_max_text(p1)
        ocr_result2 = checker.get_max_text(p2)
        print(ocr_result1)
        print(ocr_result2)
        area1 = float(ocr_result1.get("area") or 0.0) if isinstance(ocr_result1, dict) else 0.0
        area2 = float(ocr_result2.get("area") or 0.0) if isinstance(ocr_result2, dict) else 0.0
        if area1 <= _HEAD_OCR_MIN_AREA:
            ocr_result1 = {"text": "", "score": 0.0, "area": area1}
        if area2 <= _HEAD_OCR_MIN_AREA:
            ocr_result2 = {"text": "", "score": 0.0, "area": area2}
        compare_result = checker.compare_texts(ocr_result1, ocr_result2)
        text1 = str(compare_result.get("text1") or "").strip()
        text2 = str(compare_result.get("text2") or "").strip()
        area1 = ocr_result1.get("area") if isinstance(ocr_result1, dict) else None
        area2 = ocr_result2.get("area") if isinstance(ocr_result2, dict) else None
        score1 = ocr_result1.get("score") if isinstance(ocr_result1, dict) else None
        score2 = ocr_result2.get("score") if isinstance(ocr_result2, dict) else None

        print(
            "[predict][ocr] "
            f"text1='{text1}' area1={area1} score1={score1} "
            f"text2='{text2}' area2={area2} score2={score2} "
            f"match={compare_result.get('match')} "
            f"similarity={compare_result.get('similarity')} "
            f"reason={compare_result.get('reason')}"
        )

        if not text1 and not text2:
            result["ocr_match"] = None
            result["ocr_error"] = "ocr text empty"
        else:
            result["ocr_match"] = compare_result.get("match")
            result["ocr_error"] = None
        result["ocr_text1"] = text1 or None
        result["ocr_text2"] = text2 or None
    except Exception as e:
        result["ocr_error"] = str(e)
    finally:
        for temp_path in temp_files:
            try:
                os.remove(temp_path)
            except Exception:
                pass

    return result


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
) -> Tuple[Optional[Tuple[str, str]], Optional[Dict[str, str]], List[str], Optional[str]]:
    if not path3 or not path4:
        return None, None, [], None

    temp_files: List[str] = []
    merged: Dict[str, str] = {}
    ai_paths: List[str] = []

    for idx, path in ((3, str(path3)), (4, str(path4))):
        cropped_pil, err = _crop_tail_view_image(path)
        if cropped_pil is None:
            return None, None, temp_files, err or f"failed to crop tail view {idx}"

        merged[f"tail_view_crop{idx}"] = _pil_to_original_data_url(cropped_pil)

        temp_path = _save_pil_to_temp(cropped_pil, prefix=f"tail_view_{idx}")
        if temp_path:
            temp_files.append(temp_path)
            ai_paths.append(temp_path)
        else:
            ai_paths.append(path)

    return (ai_paths[0], ai_paths[1]), merged, temp_files, None


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

        img1 = _CROPPER.process_pil(img1)
        img2 = _CROPPER.process_pil(img2)

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
    Optional[Dict[str, Image.Image]], Optional[str]
]:
    """
    计算概率并生成预览图和原始图

    Returns:
        (head_prob, tail_prob, previews, original_images, cropped_pils, error)
        cropped_pils: {"h1": ..., "h2": ..., "t1": ..., "t2": ...} 裁切后的PIL图片，用于AI二次判断
    """
    try:
        _init_models()
        if _CROPPER is None or _HEAD_MODEL is None or _TAIL_MODEL is None:
            return None, None, None, None, None, "models not initialized"

        # 保存原始图片的data URL
        original_images: Dict[str, str] = {
            "original1": _pil_to_original_data_url(img1),
            "original2": _pil_to_original_data_url(img2),
        }

        v1 = _CROPPER.process_pil(img1)
        v2 = _CROPPER.process_pil(img2)

        h1 = _crop_part_from_vehicle_pil(v1, cls_id=0)
        h2 = _crop_part_from_vehicle_pil(v2, cls_id=0)
        t1 = _crop_part_from_vehicle_pil(v1, cls_id=1)
        t2 = _crop_part_from_vehicle_pil(v2, cls_id=1)

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

        # 保留裁切后的PIL图片，用于AI二次判断
        cropped_pils: Dict[str, Image.Image] = {
            "h1": h1, "h2": h2, "t1": t1, "t2": t2,
        }

        return float(head_prob), float(tail_prob), previews, original_images, cropped_pils, None
    except Exception as e:
        return None, None, None, None, None, str(e)


def _compute_head_tail_probs_pil(img1: Image.Image, img2: Image.Image) -> Tuple[
    Optional[float], Optional[float], Optional[str]]:
    try:
        _init_models()
        if _CROPPER is None or _HEAD_MODEL is None or _TAIL_MODEL is None:
            return None, None, "models not initialized"

        img1 = _CROPPER.process_pil(img1)
        img2 = _CROPPER.process_pil(img2)

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


def _classify_with_ai_second_judge(
        head_prob: Optional[float],
        tail_prob: Optional[float],
        cropped_pils: Optional[Dict[str, Image.Image]] = None,
        tail_original_paths: Optional[Tuple[str, str]] = None,
        force_head_ai_recheck: bool = False,
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
            "diff_desc": str|None,         # 差异描述（仅异常车辆有）
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

    if head_prob < _DIRECT_FAKE_PLATE_HEAD_THRESHOLD:
        result["case_type"] = "fake_plate"
        result["diff_desc"] = "车头相似度过低，直接判定为套牌"
        result["diff_analyzed_part"] = "head"
        print(
            f"[predict] head similarity {head_prob:.4f} is below direct fake-plate "
            f"threshold {_DIRECT_FAKE_PLATE_HEAD_THRESHOLD}, skipping all AI analysis"
        )
        return _populate_ai_trace_texts(result, head_prob)

    if (not force_head_ai_recheck) and head_prob > head_direct_normal_th and tail_prob > tail_direct_normal_th:
        result["case_type"] = "normal"
        return _populate_ai_trace_texts(result, head_prob)

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

    t_ai_start = time.perf_counter()

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
                ai_head_payload = _AI_CHECKER.check_head_with_reason(h1_path, h2_path)
                ai_head = str(ai_head_payload.get("label") or "")
                ai_head_reason = str(ai_head_payload.get("reason") or "").strip()
                if ai_head in ("fake_plate", "normal"):
                    result["ai_head_result"] = ai_head
                    result["ai_head_reason"] = ai_head_reason or None
                    head_verdict = ai_head
                elif ai_head == "unknown":
                    if head_prob is not None and head_prob < head_direct_normal_th:
                        head_verdict = "fake_plate"
                        fallback_reason = "输入图片质量太差，AI无法判断，车头相似度低于阈值，判断为套牌"
                    else:
                        head_verdict = "normal"
                        fallback_reason = "输入图片质量太差，AI无法判断，车头相似度大于阈值，判断为正常"
                    result["ai_head_result"] = head_verdict
                    result["ai_head_reason"] = fallback_reason
                    ai_head_reason = fallback_reason
                    print(
                        f"[predict] head AI result undetermined, "
                        f"fallback to stage1 by head similarity {head_prob:.4f} -> {head_verdict}"
                    )
                else:
                    result["ai_head_result"] = ai_head
                    result["ai_head_reason"] = ai_head_reason or None
                    print(f"[predict] head AI returned invalid result: {ai_head!r}, fallback to stage1")
                    head_ai_invalid = True
                    head_verdict = None
            else:
                print("[predict] failed to save head crops, fallback to stage1 result")
                head_ai_invalid = True
                head_verdict = None

        if head_verdict == "fake_plate":
            result["ai_judge_used"] = True
            result["ai_ms"] = (time.perf_counter() - t_ai_start) * 1000.0
            result["case_type"] = "fake_plate"
            result["diff_desc"] = ai_head_reason or "车头AI判定为套牌"
            result["diff_analyzed_part"] = "head"
            result["ai_diff_ms"] = 0.0
            print("[predict] head AI concluded fake_plate, skipping all tail AI analysis")
            return _populate_ai_trace_texts(result, head_prob)

        if tail_need_ai and use_tail_original_ai and _AI_TAIL_CHECKER is not None:
            print("[predict] tail similarity is below threshold, running 3/4 cropped tail-view AI first")
            result["tail_second_check_used"] = True
            result["tail_ai_mode"] = "tail34_cropped_primary"
            try:
                ai_tail_payload = _AI_TAIL_CHECKER.check_tail_on_original(
                    tail_original_paths[0],
                    tail_original_paths[1],
                )
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
                ai_tail_payload = _AI_CHECKER.check_tail_with_reason(t1_path, t2_path)
                ai_tail = str(ai_tail_payload.get("label") or "")
                ai_tail_reason = str(ai_tail_payload.get("reason") or "").strip()
                result["ai_tail_result"] = ai_tail
                result["ai_tail_reason"] = ai_tail_reason or None
                result["tail_ai_mode"] = "tail34_cropped_then_main" if use_tail_original_ai else "main_tail_crop_only"
                if ai_tail in ("change_trailer", "normal"):
                    tail_verdict = "different" if ai_tail == "change_trailer" else "same"
                else:
                    print(f"[predict] main tail AI returned invalid result: {ai_tail!r}, fallback to stage1")
                    tail_verdict = None
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
    result["ai_ms"] = (time.perf_counter() - t_ai_start) * 1000.0

    ai_invalid = (head_need_ai and head_verdict is None) or (tail_need_ai and tail_verdict is None)

    # ---- 综合判定 ----
    if ai_invalid:
        result["case_type"] = stage1_case_type
        result["diff_desc"] = ai_fallback_reason
        result["diff_analyzed_part"] = None
    elif head_verdict == "fake_plate":
        result["case_type"] = "fake_plate"
    elif tail_verdict == "different":
        result["case_type"] = "change_trailer"
    else:
        result["case_type"] = "normal"

    if result["case_type"] == "normal":
        result["diff_desc"] = None
        result["diff_analyzed_part"] = None
        result["ai_diff_ms"] = 0.0
        return _populate_ai_trace_texts(result, head_prob)

    diff_desc_list: List[str] = []
    analyzed_parts: List[str] = []

    if result["case_type"] == "fake_plate":
        if ai_head_reason:
            diff_desc_list.append(ai_head_reason)
            analyzed_parts.append("head")
        if tail_need_ai and ai_tail_reason:
            diff_desc_list.append(ai_tail_reason)
            analyzed_parts.append("tail")
    elif result["case_type"] == "change_trailer":
        if ai_tail_reason:
            diff_desc_list.append(ai_tail_reason)
            analyzed_parts.append("tail")
        if tail_second_reason and tail_second_reason != ai_tail_reason:
            diff_desc_list.append(tail_second_reason)
            if "tail" not in analyzed_parts:
                analyzed_parts.append("tail")

    if diff_desc_list:
        concise_desc = diff_desc_list[0]
        if len(diff_desc_list) > 1:
            concise_desc = f"{concise_desc}；另视角结论一致"
        result["diff_desc"] = concise_desc
    else:
        result["diff_desc"] = None
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


def _classify_with_head_ocr_precheck(
        head_prob: Optional[float],
        tail_prob: Optional[float],
        cropped_pils: Optional[Dict[str, Image.Image]] = None,
        tail_original_paths: Optional[Tuple[str, str]] = None,
) -> Dict[str, Any]:
    result = _build_classification_result()
    result["stage1_case_type"] = _classify_case(head_prob, tail_prob)

    ocr_result = _run_head_ocr_precheck(cropped_pils)
    result.update(ocr_result)

    if ocr_result.get("ocr_match") is False:
        text1 = ocr_result.get("ocr_text1") or ""
        text2 = ocr_result.get("ocr_text2") or ""
        if head_prob is not None and head_prob > _HEAD_OCR_AI_RECHECK_THRESHOLD:
            downstream = _classify_with_ai_second_judge(
                head_prob,
                tail_prob,
                cropped_pils,
                tail_original_paths=tail_original_paths,
                force_head_ai_recheck=True,
            )
            downstream.update(ocr_result)
            downstream["diff_desc"] = (
                f"车头OCR不一致，但车头相似度 {head_prob:.4f} 高于 {_HEAD_OCR_AI_RECHECK_THRESHOLD:.2f}，"
                f"已触发车头AI复核: '{text1}' vs '{text2}'"
            )
            downstream["diff_analyzed_part"] = "head"
            return _populate_ai_trace_texts(downstream, head_prob)

        result["case_type"] = "fake_plate"
        result["diff_desc"] = f"车头OCR不一致，判定为套牌: '{text1}' vs '{text2}'"
        result["diff_analyzed_part"] = "head"
        return _populate_ai_trace_texts(result, head_prob)

    downstream = _classify_with_ai_second_judge(
        head_prob,
        tail_prob,
        cropped_pils,
        tail_original_paths=tail_original_paths,
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
            head_prob, tail_prob, previews, original_images, cropped_pils, err = _compute_probs_and_previews_pil(img1, img2)
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
        if path3_input and path4_input:
            original_images = _append_tail_original_images(original_images, p3, p4)
            tail_ai_paths, tail_view_images, tail_view_temp_paths, tail_view_err = _prepare_tail_view_assets(p3, p4)
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
        ai_result = _classify_with_head_ocr_precheck(
            head_prob,
            tail_prob,
            cropped_pils,
            tail_original_paths=tail_ai_paths,
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
    }
    _append_ai_trace_fields(resp, ai_result)
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
    resp["ocr_used"] = ai_result.get("ocr_used", False)
    if ai_result.get("ocr_match") is not None:
        resp["ocr_match"] = ai_result.get("ocr_match")
    if ai_result.get("ocr_text1") is not None:
        resp["ocr_text1"] = ai_result.get("ocr_text1")
    if ai_result.get("ocr_text2") is not None:
        resp["ocr_text2"] = ai_result.get("ocr_text2")
    if ai_result.get("ocr_error"):
        resp["ocr_error"] = ai_result.get("ocr_error")
    # 添加细粒度差异分析结果（仅异常车辆）
    if ai_result.get("diff_desc"):
        resp["diff_desc"] = ai_result["diff_desc"]
        resp["diff_analyzed_part"] = ai_result.get("diff_analyzed_part")
        resp["ai_diff_ms"] = round(ai_result.get("ai_diff_ms", 0.0), 1)
    if err:
        resp["error"] = err
    lat_ms = (time.perf_counter() - t0) * 1000.0

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
        ocr_used=bool(ai_result.get("ocr_used")),
        ocr_match=ai_result.get("ocr_match"),
        ocr_text1=ai_result.get("ocr_text1"),
        ocr_text2=ai_result.get("ocr_text2"),
        ocr_error=ai_result.get("ocr_error"),
        diff_desc=ai_result.get("diff_desc"),
        diff_analyzed_part=ai_result.get("diff_analyzed_part"),
        ai_diff_ms=ai_result.get("ai_diff_ms"),
        head_ai_display_text=ai_result.get("head_ai_display_text"),
        tail34_ai_display_text=ai_result.get("tail34_ai_display_text"),
        main_tail_ai_display_text=ai_result.get("main_tail_ai_display_text"),
        final_diff_summary=ai_result.get("final_diff_summary"),
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
        head_prob, tail_prob, previews, original_images, cropped_pils, err = _compute_probs_and_previews_pil(img1, img2)
        t_compute_ms = (time.perf_counter() - t_compute0) * 1000.0

        tail_view_temp_paths: List[str] = []
        tail_ai_paths = None
        if path3_input and path4_input:
            original_images = _append_tail_original_images(original_images, p3, p4)
            tail_ai_paths, tail_view_images, tail_view_temp_paths, tail_view_err = _prepare_tail_view_assets(p3, p4)
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
        ai_result = _classify_with_head_ocr_precheck(
            head_prob,
            tail_prob,
            cropped_pils,
            tail_original_paths=tail_ai_paths,
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
        "stage1_case_type": ai_result.get("stage1_case_type"),
        "tail_second_check_used": ai_result.get("tail_second_check_used", False),
        "tail_second_check_result": ai_result.get("tail_second_check_result"),
        "tail_second_check_reason": ai_result.get("tail_second_check_reason"),
    }
    _append_ai_trace_fields(resp, ai_result)
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
    resp["ocr_used"] = ai_result.get("ocr_used", False)
    if ai_result.get("ocr_match") is not None:
        resp["ocr_match"] = ai_result.get("ocr_match")
    if ai_result.get("ocr_text1") is not None:
        resp["ocr_text1"] = ai_result.get("ocr_text1")
    if ai_result.get("ocr_text2") is not None:
        resp["ocr_text2"] = ai_result.get("ocr_text2")
    if ai_result.get("ocr_error"):
        resp["ocr_error"] = ai_result.get("ocr_error")
    # 添加细粒度差异分析结果（仅异常车辆）
    if ai_result.get("diff_desc"):
        resp["diff_desc"] = ai_result["diff_desc"]
        resp["diff_analyzed_part"] = ai_result.get("diff_analyzed_part")
        resp["ai_diff_ms"] = round(ai_result.get("ai_diff_ms", 0.0), 1)
    if err:
        resp["error"] = err
    lat_ms = (time.perf_counter() - t0) * 1000.0

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
        ocr_used=bool(ai_result.get("ocr_used")),
        ocr_match=ai_result.get("ocr_match"),
        ocr_text1=ai_result.get("ocr_text1"),
        ocr_text2=ai_result.get("ocr_text2"),
        ocr_error=ai_result.get("ocr_error"),
        diff_desc=ai_result.get("diff_desc"),
        diff_analyzed_part=ai_result.get("diff_analyzed_part"),
        ai_diff_ms=ai_result.get("ai_diff_ms"),
        head_ai_display_text=ai_result.get("head_ai_display_text"),
        tail34_ai_display_text=ai_result.get("tail34_ai_display_text"),
        main_tail_ai_display_text=ai_result.get("main_tail_ai_display_text"),
        final_diff_summary=ai_result.get("final_diff_summary"),
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
        head_prob, tail_prob, previews, original_images, cropped_pils, err = _compute_probs_and_previews_pil(img1, img2)
        t_compute_ms = (time.perf_counter() - t_compute0) * 1000.0

        # 两层鉴别分类（含AI二次判断）
        tail_original_paths = None
        if f3 and f4:
            p3 = _save_upload_file_to_temp(f3, prefix="upload_tail3")
            p4 = _save_upload_file_to_temp(f4, prefix="upload_tail4")
            if p3:
                temp_tail_paths.append(p3)
            if p4:
                temp_tail_paths.append(p4)
            if p3 and p4:
                original_images = _append_tail_original_images(original_images, p3, p4)
                tail_original_paths, tail_view_images, tail_view_temp_paths, tail_view_err = _prepare_tail_view_assets(p3, p4)
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
        ai_result = _classify_with_head_ocr_precheck(
            head_prob,
            tail_prob,
            cropped_pils,
            tail_original_paths=tail_original_paths,
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
        "stage1_case_type": ai_result.get("stage1_case_type"),
        "tail_second_check_used": ai_result.get("tail_second_check_used", False),
        "tail_second_check_result": ai_result.get("tail_second_check_result"),
        "tail_second_check_reason": ai_result.get("tail_second_check_reason"),
    }
    _append_ai_trace_fields(resp, ai_result)
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
    resp["ocr_used"] = ai_result.get("ocr_used", False)
    if ai_result.get("ocr_match") is not None:
        resp["ocr_match"] = ai_result.get("ocr_match")
    if ai_result.get("ocr_text1") is not None:
        resp["ocr_text1"] = ai_result.get("ocr_text1")
    if ai_result.get("ocr_text2") is not None:
        resp["ocr_text2"] = ai_result.get("ocr_text2")
    if ai_result.get("ocr_error"):
        resp["ocr_error"] = ai_result.get("ocr_error")
    if err:
        resp["error"] = err
    lat_ms = (time.perf_counter() - t0) * 1000.0

    # 保存图片并记录
    file1_name = f1.filename if f1 else "unknown"
    file2_name = f2.filename if f2 else "unknown"
    file3_name = f3.filename if f3 else ""
    file4_name = f4.filename if f4 else ""

    # 添加细粒度差异分析结果到响应（仅异常车辆）
    if ai_result.get("diff_desc"):
        resp["diff_desc"] = ai_result["diff_desc"]
        resp["diff_analyzed_part"] = ai_result.get("diff_analyzed_part")
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
        ocr_used=bool(ai_result.get("ocr_used")),
        ocr_match=ai_result.get("ocr_match"),
        ocr_text1=ai_result.get("ocr_text1"),
        ocr_text2=ai_result.get("ocr_text2"),
        ocr_error=ai_result.get("ocr_error"),
        diff_desc=ai_result.get("diff_desc"),
        diff_analyzed_part=ai_result.get("diff_analyzed_part"),
        ai_diff_ms=ai_result.get("ai_diff_ms"),
        head_ai_display_text=ai_result.get("head_ai_display_text"),
        tail34_ai_display_text=ai_result.get("tail34_ai_display_text"),
        main_tail_ai_display_text=ai_result.get("main_tail_ai_display_text"),
        final_diff_summary=ai_result.get("final_diff_summary"),
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
            head_prob, tail_prob, previews, original_images, cropped_pils, err = _compute_probs_and_previews_pil(img1, img2)
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
        if f3 and f4:
            p3 = _save_upload_file_to_temp(f3, prefix="upload_tail3")
            p4 = _save_upload_file_to_temp(f4, prefix="upload_tail4")
            if p3:
                temp_tail_paths.append(p3)
            if p4:
                temp_tail_paths.append(p4)
            if p3 and p4:
                original_images = _append_tail_original_images(original_images, p3, p4)
                tail_original_paths, tail_view_images, tail_view_temp_paths, tail_view_err = _prepare_tail_view_assets(p3, p4)
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
        ai_result = _classify_with_head_ocr_precheck(
            head_prob,
            tail_prob,
            cropped_pils,
            tail_original_paths=tail_original_paths,
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
    }
    _append_ai_trace_fields(resp, ai_result)
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
    resp["ocr_used"] = ai_result.get("ocr_used", False)
    if ai_result.get("ocr_match") is not None:
        resp["ocr_match"] = ai_result.get("ocr_match")
    if ai_result.get("ocr_text1") is not None:
        resp["ocr_text1"] = ai_result.get("ocr_text1")
    if ai_result.get("ocr_text2") is not None:
        resp["ocr_text2"] = ai_result.get("ocr_text2")
    if ai_result.get("ocr_error"):
        resp["ocr_error"] = ai_result.get("ocr_error")
    # 添加细粒度差异分析结果（仅异常车辆）
    if ai_result.get("diff_desc"):
        resp["diff_desc"] = ai_result["diff_desc"]
        resp["diff_analyzed_part"] = ai_result.get("diff_analyzed_part")
        resp["ai_diff_ms"] = round(ai_result.get("ai_diff_ms", 0.0), 1)
    if err:
        resp["error"] = err
    lat_ms = (time.perf_counter() - t0) * 1000.0

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
        ocr_used=bool(ai_result.get("ocr_used")),
        ocr_match=ai_result.get("ocr_match"),
        ocr_text1=ai_result.get("ocr_text1"),
        ocr_text2=ai_result.get("ocr_text2"),
        ocr_error=ai_result.get("ocr_error"),
        diff_desc=ai_result.get("diff_desc"),
        diff_analyzed_part=ai_result.get("diff_analyzed_part"),
        ai_diff_ms=ai_result.get("ai_diff_ms"),
        head_ai_display_text=ai_result.get("head_ai_display_text"),
        tail34_ai_display_text=ai_result.get("tail34_ai_display_text"),
        main_tail_ai_display_text=ai_result.get("main_tail_ai_display_text"),
        final_diff_summary=ai_result.get("final_diff_summary"),
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
        include_deleted = request.args.get("include_deleted", "false").lower() == "true"
        limit = int(request.args.get("limit", "50"))
        offset = int(request.args.get("offset", "0"))

        result = _METRICS.query_records(
            start_date=start_date,
            end_date=end_date,
            case_type=case_type if case_type != "all" else None,
            include_deleted=include_deleted,
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
            "vehicle1.jpg", "vehicle2.jpg", "head1.jpg", "head2.jpg", "tail1.jpg", "tail2.jpg",
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


@app.get("/thresholds")
def get_thresholds() -> Any:
    return jsonify({
        "head_threshold": _HEAD_THRESHOLD,
        "tail_threshold": _TAIL_THRESHOLD,
    })


@app.post("/thresholds")
def set_thresholds() -> Any:
    global _HEAD_THRESHOLD, _TAIL_THRESHOLD

    try:
        payload = request.get_json(silent=True) or {}
        head_threshold = payload.get("head_threshold", _HEAD_THRESHOLD)
        tail_threshold = payload.get("tail_threshold", _TAIL_THRESHOLD)

        new_head_threshold = _validate_threshold_value("head_threshold", head_threshold)
        new_tail_threshold = _validate_threshold_value("tail_threshold", tail_threshold)

        with _THRESHOLD_LOCK:
            _HEAD_THRESHOLD = new_head_threshold
            _TAIL_THRESHOLD = new_tail_threshold
            _save_threshold_settings()

        return jsonify({
            "ok": True,
            "message": "thresholds updated",
            "head_threshold": _HEAD_THRESHOLD,
            "tail_threshold": _TAIL_THRESHOLD,
        })
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"ok": False, "error": f"failed to update thresholds: {e}"}), 500


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8001"))
    app.run(host=host, port=port, threaded=True)
