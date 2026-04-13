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
import gc
from collections import deque
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import cv2
import torch
from PIL import Image
from flask import Flask, jsonify, request, render_template, send_file, send_from_directory
from ultralytics import YOLO

# 假设这些是你本地的自定义模块
from siamese import Siamese
from data_tran.image_resolver import ImagePathResolver
from qwen_vl.predict_ai import VehicleCheck

parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from data_chuli.cropper import VehicleCropper


app = Flask(__name__)

_INIT_LOCK = threading.Lock()
_INFER_LOCK = threading.Lock()
_REQUEST_LOCK = threading.Lock()
_INITIALIZED = False
_OOM_RESTART_LOCK = threading.Lock()
_OOM_RESTART_SCHEDULED = False

_CROPPER: Optional[VehicleCropper] = None
_HEAD_MODEL: Optional[Siamese] = None
_TAIL_MODEL: Optional[Siamese] = None
_HEADTAIL_MODEL: Optional[YOLO] = None
_IMAGE_RESOLVER: Optional[ImagePathResolver] = None

# 全局阈值变量，默认设置
_HEAD_THRESHOLD: float = 0.8
_TAIL_THRESHOLD: float = 0.8

# 复检进度跟踪
_RECHECK_LOCK = threading.Lock()
_RECHECK_STATUS = {
    "running": False,
    "started_at": None,
    "total": 0,
    "processed": 0,
    "success": 0,
    "failed": 0,
    "current_record": None,
    "error": None,
    "results": []
}


def _release_cuda_memory() -> None:
    try:
        gc.collect()
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _is_cuda_oom_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    if "out of memory" in msg and "cuda" in msg:
        return True
    if isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", tuple())):
        return True
    if hasattr(torch, "OutOfMemoryError") and isinstance(exc, torch.OutOfMemoryError):
        return True
    return False


def _oom_auto_restart_enabled() -> bool:
    raw = str(os.environ.get("OOM_AUTO_RESTART", "1")).strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _schedule_process_restart() -> None:
    global _OOM_RESTART_SCHEDULED
    with _OOM_RESTART_LOCK:
        if _OOM_RESTART_SCHEDULED:
            return
        _OOM_RESTART_SCHEDULED = True

    def _restart_worker() -> None:
        try:
            time.sleep(0.5)
            exe = sys.executable
            if exe and os.path.exists(exe):
                try:
                    os.spawnv(os.P_NOWAIT, exe, [exe] + sys.argv)
                except Exception as spawn_err:
                    print(f"[oom] spawn restart failed: {spawn_err}")
        finally:
            os._exit(86)

    threading.Thread(target=_restart_worker, daemon=True).start()


def _reset_models_for_oom_recovery() -> None:
    global _INITIALIZED, _CROPPER, _HEAD_MODEL, _TAIL_MODEL, _HEADTAIL_MODEL
    with _INIT_LOCK:
        for model_obj in (_HEAD_MODEL, _TAIL_MODEL):
            try:
                net = getattr(model_obj, "net", None)
                if net is not None and hasattr(net, "cpu"):
                    net.cpu()
            except Exception:
                pass
        try:
            if _HEADTAIL_MODEL is not None and hasattr(_HEADTAIL_MODEL, "to"):
                _HEADTAIL_MODEL.to("cpu")
        except Exception:
            pass

        _CROPPER = None
        _HEAD_MODEL = None
        _TAIL_MODEL = None
        _HEADTAIL_MODEL = None
        _INITIALIZED = False


def _init_models() -> None:
    """初始化模型的逻辑"""
    global _INITIALIZED, _CROPPER, _HEAD_MODEL, _TAIL_MODEL, _HEADTAIL_MODEL, _IMAGE_RESOLVER
    with _INIT_LOCK:
        if _INITIALIZED:
            return
        try:
            print("[init] 正在初始化模型...")
            # 在这里添加具体的模型初始化代码，例如：
            # _CROPPER = VehicleCropper(...)
            # _HEAD_MODEL = Siamese(...)
            # _TAIL_MODEL = Siamese(...)
            # _HEADTAIL_MODEL = YOLO(...)
            # _IMAGE_RESOLVER = ImagePathResolver(...)
            _INITIALIZED = True
            print("[init] 模型初始化完成")
        except Exception as e:
            print(f"[init] 初始化模型失败: {e}")
            raise


def _recover_from_cuda_oom(context: str, *, allow_restart: bool = False) -> None:
    print(f"[oom] detected at {context}, start recovery")
    _release_cuda_memory()
    try:
        _reset_models_for_oom_recovery()
    except Exception as reset_err:
        print(f"[oom] reset models failed: {reset_err}")
    _release_cuda_memory()
    try:
        _init_models()
    except Exception as init_err:
        print(f"[oom] re-init models failed: {init_err}")
    _release_cuda_memory()
    if allow_restart and _oom_auto_restart_enabled():
        print("[oom] recovery failed repeatedly, scheduling process restart")
        _schedule_process_restart()


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
                date_part = name[len("stats_") : len("stats_") + 8]
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
                    date_part = name[len("stats_") : len("stats_") + 8]
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
                    dt = datetime.datetime.strptime(k, "%Y-%m-%d %H:00").replace(tzinfo=datetime.datetime.now().astimezone().tzinfo)
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
    
    def save_images(self, record_id: str, previews: Dict[str, str], meta: Dict[str, Any], 
                    original_images: Optional[Dict[str, str]] = None) -> Optional[str]:
        """保存预览图和原始图到磁盘"""
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
            
            # 保存2张原始图片（如果提供）
            if original_images:
                for key in ["original1", "original2"]:
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
        """查询记录列表"""
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

                                # 如果AI复检成功，以AI复检结果作为最终真实结果
                                ai_recheck = record.get("ai_recheck", {}) or {}
                                if ai_recheck.get("attempted") and ai_recheck.get("success") and ai_recheck.get("ai_result"):
                                    record["case_type"] = str(ai_recheck.get("ai_result"))
                                
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
        """删除记录"""
        try:
            # 获取记录
            record = self.get_record(record_id)
            if not record:
                return False, "record not found"
            
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
                
                return True, "record deleted"

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
        """设置记录保护状态"""
        try:
            # 获取记录
            record = self.get_record(record_id)
            if not record:
                return False, "record not found"
            
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
            
            return True, ("protected" if protected else "unprotected")
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
                return False, "record not found"
            
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
                            
                            # 【补全被截断的部分】-----------------------------
                            if r.get("record_id") == record_id:
                                r.update(review_data)
                                r["review_history"] = review_history
                                r["case_type"] = reviewed_case_type  # 覆盖最终结论
                                lines.append(json.dumps(r, ensure_ascii=False) + "\n")
                            else:
                                lines.append(line)
                            # -----------------------------------------------
                        except Exception:
                            lines.append(line)
                
                with open(log_path, "w", encoding="utf-8") as f:
                    f.writelines(lines)

            # 同步更新图像目录下的元数据信息
            image_dir = record.get("image_dir", "")
            if image_dir and os.path.exists(image_dir):
                meta_file = os.path.join(image_dir, "meta.json")
                if os.path.exists(meta_file):
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    
                    meta.update(review_data)
                    meta["case_type"] = reviewed_case_type
                    
                    with open(meta_file, "w", encoding="utf-8") as f:
                        json.dump(meta, f, ensure_ascii=False, indent=2)

            return True, "复核结果已成功提交并更新"
        except Exception as e:
            return False, f"复核提交失败: {str(e)}"

# 初始化监控与统计实例 (假设存在 logs 目录)
metrics = _MetricsStore(log_dir="./logs", retention_days=30, recent_max=500)

# ==========================================
# 补全后续的 Flask 基础路由骨架
# ==========================================

@app.before_request
def initialize():
    """确保在第一次请求时加载模型"""
    if not _INITIALIZED:
        _init_models()

@app.route("/api/v1/predict", methods=["POST"])
def predict():
    """基础预测接口示意"""
    start_time = time.time()
    try:
        data = request.get_json() or {}
        # 你的推理逻辑...
        
        result = {"ok": True, "case_type": "normal", "confidence": 0.9}
        metrics.record({
            "endpoint": "/predict",
            "ok": True,
            "http_status": 200,
            "case_type": result.get("case_type"),
            "lat_ms": (time.time() - start_time) * 1000
        })
        return jsonify(result)
    except Exception as e:
        metrics.record({
            "endpoint": "/predict",
            "ok": False,
            "http_status": 500,
            "lat_ms": (time.time() - start_time) * 1000
        })
        return jsonify({"error": str(e)}), 500

@app.route("/api/v1/records", methods=["GET"])
def get_records():
    """获取所有记录"""
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")
    case_type = request.args.get("case_type")
    limit = int(request.args.get("limit", 50))
    offset = int(request.args.get("offset", 0))
    
    result = metrics.query_records(
        start_date=start_date,
        end_date=end_date,
        case_type=case_type,
        limit=limit,
        offset=offset
    )
    return jsonify(result)

@app.route("/api/v1/records/<record_id>/review", methods=["POST"])
def review_record_endpoint(record_id):
    """提交记录复核"""
    data = request.get_json() or {}
    case_type = data.get("reviewed_case_type")
    reason = data.get("review_reason", "")
    by = data.get("reviewed_by", "system")
    confidence = data.get("review_confidence", "medium")
    
    if not case_type:
        return jsonify({"error": "missing reviewed_case_type"}), 400
        
    success, msg = metrics.review_record(record_id, case_type, reason, by, confidence)
    if success:
        return jsonify({"ok": True, "message": msg})
    return jsonify({"error": msg}), 400

@app.route("/api/v1/stats", methods=["GET"])
def get_stats():
    """获取统计信息"""
    return jsonify(metrics.snapshot())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)