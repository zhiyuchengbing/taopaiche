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

# 鍏ㄥ眬闃堝€煎彉閲忥紝榛樿璁剧疆涓?.8
_HEAD_THRESHOLD: float = 0.8
_TAIL_THRESHOLD: float = 0.8

# 澶嶆杩涘害璺熻釜
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
        
        # 鍥剧墖瀛樺偍鐩綍
        self._images_dir = os.path.join(self._log_dir, "images")
        os.makedirs(self._images_dir, exist_ok=True)
        
        # 鍙椾繚鎶よ褰曞垪琛ㄦ枃浠?
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
        """鍔犺浇鍙椾繚鎶ょ殑璁板綍ID鍒楄〃"""
        try:
            if os.path.exists(self._protected_file):
                with open(self._protected_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return set(data.get("protected", []))
        except Exception:
            pass
        return set()
    
    def _save_protected_records(self) -> None:
        """淇濆瓨鍙椾繚鎶ょ殑璁板綍ID鍒楄〃"""
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
            
            # 娓呯悊鏃х殑 jsonl 鏂囦欢
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
            
            # 娓呯悊鏃х殑鍥剧墖鏂囦欢澶?
            if os.path.exists(self._images_dir):
                for date_folder in os.listdir(self._images_dir):
                    try:
                        d = datetime.datetime.strptime(date_folder, "%Y%m%d").date()
                    except Exception:
                        continue
                    if d < cutoff:
                        date_path = os.path.join(self._images_dir, date_folder)
                        if os.path.isdir(date_path):
                            # 閬嶅巻璇ユ棩鏈熶笅鐨勬墍鏈夎褰?
                            for record_folder in os.listdir(date_path):
                                record_path = os.path.join(date_path, record_folder)
                                if not os.path.isdir(record_path):
                                    continue
                                
                                # 璇诲彇璁板綍鍏冩暟鎹?
                                meta_file = os.path.join(record_path, "meta.json")
                                try:
                                    with open(meta_file, "r", encoding="utf-8") as f:
                                        meta = json.load(f)
                                    
                                    record_id = meta.get("record_id", "")
                                    case_type = meta.get("case_type", "")
                                    
                                    # 鍒ゆ柇鏄惁鍙互鍒犻櫎
                                    can_delete = False
                                    if case_type == "normal":
                                        # 姝ｅ父杞﹁締鐩存帴鍒犻櫎
                                        can_delete = True
                                    elif case_type in ["fake_plate", "change_trailer"]:
                                        # 濂楃墝/鎹㈡寕杞︽鏌ヤ繚鎶ゆ爣璁?
                                        if record_id not in self._protected_records:
                                            can_delete = True
                                    else:
                                        # 鍏朵粬绫诲瀷涔熷垹闄?
                                        can_delete = True
                                    
                                    if can_delete:
                                        shutil.rmtree(record_path, ignore_errors=True)
                                except Exception:
                                    # 濡傛灉鏃犳硶璇诲彇鍏冩暟鎹紝涔熷垹闄?
                                    shutil.rmtree(record_path, ignore_errors=True)
                            
                            # 濡傛灉鏃ユ湡鏂囦欢澶逛负绌猴紝鍒犻櫎瀹?
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
        """
        淇濆瓨棰勮鍥惧拰鍘熷鍥惧埌纾佺洏
        
        Args:
            record_id: 璁板綍鍞竴ID
            previews: 鍖呭惈6寮犲鐞嗗悗鍥剧墖鐨刣ata URL瀛楀吀
            meta: 璁板綍鍏冩暟鎹?
            original_images: 鍖呭惈2寮犲師濮嬪浘鐗囩殑data URL瀛楀吀锛堝彲閫夛級
        
        Returns:
            鍥剧墖鐩綍璺緞锛屽け璐ヨ繑鍥濶one
        """
        try:
            dt = datetime.datetime.now()
            date_folder = self._date_key(dt)
            
            # 鍒涘缓鏃ユ湡鏂囦欢澶?
            date_path = os.path.join(self._images_dir, date_folder)
            os.makedirs(date_path, exist_ok=True)
            
            # 鍒涘缓璁板綍鏂囦欢澶?
            record_path = os.path.join(date_path, record_id)
            os.makedirs(record_path, exist_ok=True)
            
            # 淇濆瓨6寮犲鐞嗗悗鐨勫浘鐗?
            for key in ["vehicle1", "vehicle2", "head1", "head2", "tail1", "tail2"]:
                data_url = previews.get(key, "")
                if not data_url or not data_url.startswith("data:image/"):
                    continue
                
                try:
                    # 瑙ｆ瀽 data URL
                    header, encoded = data_url.split(",", 1)
                    img_data = base64.b64decode(encoded)
                    
                    # 淇濆瓨鍥剧墖
                    img_path = os.path.join(record_path, f"{key}.jpg")
                    with open(img_path, "wb") as f:
                        f.write(img_data)
                except Exception:
                    continue
            
            # 淇濆瓨2寮犲師濮嬪浘鐗囷紙濡傛灉鎻愪緵锛?
            if original_images:
                for key in ["original1", "original2"]:
                    data_url = original_images.get(key, "")
                    if not data_url or not data_url.startswith("data:image/"):
                        continue
                    
                    try:
                        # 瑙ｆ瀽 data URL
                        header, encoded = data_url.split(",", 1)
                        img_data = base64.b64decode(encoded)
                        
                        # 淇濆瓨鍥剧墖
                        img_path = os.path.join(record_path, f"{key}.jpg")
                        with open(img_path, "wb") as f:
                            f.write(img_data)
                    except Exception:
                        continue
            
            # 淇濆瓨鍏冩暟鎹?
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
        鏌ヨ璁板綍鍒楄〃
        
        Args:
            start_date: 寮€濮嬫棩鏈?YYYY-MM-DD
            end_date: 缁撴潫鏃ユ湡 YYYY-MM-DD
            case_type: 绫诲瀷绛涢€?normal/fake_plate/change_trailer/all
            include_deleted: 鏄惁鍖呭惈宸插垹闄よ褰?
            limit: 杩斿洖鏉℃暟
            offset: 鍋忕Щ閲?
        
        Returns:
            鍖呭惈璁板綍鍒楄〃鍜屾€绘暟鐨勫瓧鍏?
        """
        self._ensure_history_loaded()
        
        try:
            # 瑙ｆ瀽鏃ユ湡鑼冨洿
            if start_date:
                start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
            else:
                start_dt = datetime.datetime.now().date() - datetime.timedelta(days=7)
            
            if end_date:
                end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
            else:
                end_dt = datetime.datetime.now().date()
            
            # 鏀堕泦鎵€鏈夌鍚堟潯浠剁殑璁板綍
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

                                # 濡傛灉AI澶嶆鎴愬姛锛屼互AI澶嶆缁撴灉浣滀负鏈€缁堢湡瀹炵粨鏋?
                                ai_recheck = record.get("ai_recheck", {}) or {}
                                if ai_recheck.get("attempted") and ai_recheck.get("success") and ai_recheck.get("ai_result"):
                                    record["case_type"] = str(ai_recheck.get("ai_result"))
                                
                                # 绛涢€夋潯浠?
                                if not include_deleted and record.get("deleted", False):
                                    continue
                                
                                if case_type and case_type != "all":
                                    if record.get("case_type") != case_type:
                                        continue
                                
                                # 鍙繚鐣欐湁 record_id 鐨勮褰曪紙鏈夊浘鐗囩殑锛?
                                if "record_id" in record:
                                    records.append(record)
                            except Exception:
                                continue
                
                current_date += datetime.timedelta(days=1)
            
            # 鎸夋椂闂村€掑簭鎺掑簭
            records.sort(key=lambda x: x.get("ts", ""), reverse=True)
            
            # 鍒嗛〉
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
        """鑾峰彇鍗曟潯璁板綍璇︽儏"""
        try:
            # 浠?record_id 涓彁鍙栨棩鏈?
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
        鍒犻櫎璁板綍
        
        Args:
            record_id: 璁板綍ID
            hard_delete: 鏄惁纭垹闄わ紙褰诲簳鍒犻櫎鏂囦欢锛?
        
        Returns:
            (鎴愬姛, 娑堟伅)
        """
        try:
            # 鑾峰彇璁板綍
            record = self.get_record(record_id)
            if not record:
                return False, "record not found"
            
            # 妫€鏌ユ槸鍚﹀厑璁稿垹闄?
            case_type = record.get("case_type", "")
            if case_type == "normal":
                return False, "姝ｅ父杞﹁締璁板綍鐢辩郴缁熻嚜鍔ㄦ竻鐞嗭紝鏃犻渶鎵嬪姩鍒犻櫎"
            
            if case_type not in ["fake_plate", "change_trailer"]:
                return False, f"涓嶆敮鎸佸垹闄ょ被鍨? {case_type}"
            
            if hard_delete:
                # 纭垹闄わ細鍒犻櫎鍥剧墖鏂囦欢澶?
                image_dir = record.get("image_dir", "")
                if image_dir and os.path.exists(image_dir):
                    shutil.rmtree(image_dir, ignore_errors=True)
                
                # 浠?jsonl 涓垹闄わ紙鏍囪涓哄凡鍒犻櫎锛?
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
                
                # 浠庝繚鎶ゅ垪琛ㄤ腑绉婚櫎
                if record_id in self._protected_records:
                    self._protected_records.remove(record_id)
                    self._save_protected_records()
                
                return True, "record deleted"
                # soft delete branch
                # 杞垹闄わ細鍙爣璁?
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
                
                return True, "璁板綍宸叉爣璁颁负鍒犻櫎"
        except Exception as e:
            return False, f"鍒犻櫎澶辫触: {str(e)}"
    
    def protect_record(self, record_id: str, protected: bool, note: str = "") -> Tuple[bool, str]:
        """
        璁剧疆璁板綍淇濇姢鐘舵€?
        
        Args:
            record_id: 璁板綍ID
            protected: 鏄惁淇濇姢
            note: 澶囨敞淇℃伅
        
        Returns:
            (鎴愬姛, 娑堟伅)
        """
        try:
            # 鑾峰彇璁板綍
            record = self.get_record(record_id)
            if not record:
                return False, "record not found"
            
            # 鏇存柊淇濇姢鐘舵€?
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
            
            # 鏇存柊淇濇姢鍒楄〃
            if protected:
                self._protected_records.add(record_id)
            else:
                self._protected_records.discard(record_id)
            self._save_protected_records()
            
            # 鏇存柊鍏冩暟鎹枃浠?
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
            return False, f"鎿嶄綔澶辫触: {str(e)}"
    
    def review_record(self, record_id: str, reviewed_case_type: str, review_reason: str, 
                     reviewed_by: str, review_confidence: str = "medium") -> Tuple[bool, str]:
        """
        鎻愪氦澶嶆牳缁撴灉
        
        Args:
            record_id: 璁板綍ID
            reviewed_case_type: 澶嶆牳鍚庣殑绫诲瀷
            review_reason: 澶嶆牳鐞嗙敱
            reviewed_by: 澶嶆牳浜哄憳
            review_confidence: 缃俊搴?high/medium/low
        
        Returns:
            (鎴愬姛, 娑堟伅)
        """
        try:
            # 鑾峰彇璁板綍
            record = self.get_record(record_id)
            if not record:
                return False, "record not found"
            
            # 楠岃瘉澶嶆牳绫诲瀷
            valid_types = ["normal", "fake_plate", "change_trailer"]
            if reviewed_case_type not in valid_types:
                return False, f"鏃犳晥鐨勫鏍哥被鍨? {reviewed_case_type}"
            
            # 鍑嗗澶嶆牳淇℃伅
            review_data = {
                "reviewed": True,
                "reviewed_at": datetime.datetime.now().isoformat(timespec="milliseconds"),
                "reviewed_by": reviewed_by,
                "reviewed_case_type": reviewed_case_type,
                "review_reason": review_reason,
                "review_confidence": review_confidence
            }
            
            # 淇濆瓨澶嶆牳鍘嗗彶
            review_history = record.get("review_history", [])
            review_history.append(review_data.copy())
            
            # 鏇存柊璁板綍
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
            
            # 鏇存柊鍏冩暟鎹枃浠?
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
            
            return True, "review saved"
        except Exception as e:
            return False, f"鎿嶄綔澶辫触: {str(e)}"
    
    def revoke_review(self, record_id: str) -> Tuple[bool, str]:
        """
        鎾ら攢澶嶆牳
        
        Args:
            record_id: 璁板綍ID
        
        Returns:
            (鎴愬姛, 娑堟伅)
        """
        try:
            # 鑾峰彇璁板綍
            record = self.get_record(record_id)
            if not record:
                return False, "record not found"
            
            if not record.get("reviewed", False):
                return False, "璇ヨ褰曟湭澶嶆牳"
            
            # 绉婚櫎澶嶆牳瀛楁
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
            
            # 鏇存柊鍏冩暟鎹枃浠?
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
            
            return True, "宸叉挙閿€澶嶆牳"
        except Exception as e:
            return False, f"鎿嶄綔澶辫触: {str(e)}"
    
    def get_review_stats(self) -> Dict[str, Any]:
        """鑾峰彇澶嶆牳缁熻"""
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
            
            # 閬嶅巻鎵€鏈夎褰?
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
                            
                            # 濡傛灉AI澶嶆鎴愬姛锛屼互AI澶嶆缁撴灉浣滀负鏈€缁堢湡瀹炵粨鏋?
                            ai_recheck = record.get("ai_recheck", {}) or {}
                            if ai_recheck.get("attempted") and ai_recheck.get("success") and ai_recheck.get("ai_result"):
                                record["case_type"] = str(ai_recheck.get("ai_result"))
                            
                            # ...
                            
                            # 杩囨护璁板綍
                            if case_type and record.get("case_type") != case_type:
                                continue
                            if reviewed is not None and record.get("reviewed") != reviewed:
                                continue
                            if protected is not None and record.get("protected") != protected:
                                continue
                            if start_date and datetime.datetime.strptime(record.get("ts", ""), "%Y-%m-%d %H:%M:%S").date() < start_date:
                                continue
                            if end_date and datetime.datetime.strptime(record.get("ts", ""), "%Y-%m-%d %H:%M:%S").date() > end_date:
                                continue
                            
                            # ...
                            
                            # 灏濊瘯璇诲彇鍏冩暟鎹枃浠惰ˉ鍏呬俊鎭?
                            try:
                                record_dir = record.get("image_dir")
                                if record_dir and os.path.exists(record_dir):
                                    meta_file = os.path.join(record_dir, "meta.json")
                                    if os.path.exists(meta_file):
                                        with open(meta_file, "r", encoding="utf-8") as mf:
                                            meta = json.load(mf)
                                            # 鍚堝苟鍏冩暟鎹紙涓嶈鐩栧凡鏈夊瓧娈碉級
                                            for k, v in meta.items():
                                                if k not in record:
                                                    record[k] = v
                            except Exception:
                                pass
                            
                            # ...
                            
                            # 杩斿洖璁板綍
                            records.append(record)
                        except Exception:
                            continue
            
            # 璁＄畻澶嶆牳鐜?
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
    """Record exporter."""
    
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
        瀵煎嚭鍗曟潯璁板綍
        
        Args:
            record_id: 璁板綍ID
            export_path: 瀵煎嚭璺緞锛堝彲閫夛級
            image_types: 瑕佸鍑虹殑鍥剧墖绫诲瀷鍒楄〃锛屽 ["original1", "original2", "vehicle1", "vehicle2", "head1", "head2", "tail1", "tail2"]
                        濡傛灉涓篘one锛屽垯瀵煎嚭鎵€鏈夊浘鐗?
        
        Returns:
            (鎴愬姛, 娑堟伅, 瀵煎嚭璺緞)
        """
        try:
            # 鑾峰彇璁板綍
            record = self.metrics.get_record(record_id)
            if not record:
                return False, "record not found", None
            
            # 鑾峰彇鍥剧墖鐩綍
            image_dir = record.get("image_dir", "")
            if not image_dir or not os.path.exists(image_dir):
                return False, "image dir not found", None
            
            # 纭畾瀵煎嚭璺緞
            if export_path is None:
                export_path = self.export_base_dir
            
            # 鍒涘缓瀵煎嚭鏂囦欢澶?
            case_type = record.get("case_type", "unknown")
            folder_name = f"{record_id}_{case_type}"
            export_folder = os.path.join(export_path, folder_name)
            os.makedirs(export_folder, exist_ok=True)
            
            # 纭畾瑕佸鍑虹殑鍥剧墖绫诲瀷
            if image_types is None:
                # 榛樿瀵煎嚭鎵€鏈夊浘鐗?
                image_types = ["original1", "original2", "vehicle1", "vehicle2", 
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
            
            # 鐢熸垚淇℃伅鏂囦欢
            info_path = os.path.join(export_folder, "info.txt")
            with open(info_path, "w", encoding="utf-8") as f:
                f.write(f"璁板綍ID: {record_id}\n")
                f.write(f"鏃堕棿: {record.get('ts', '')}\n")
                f.write(f"绯荤粺鍒ゅ畾: {record.get('case_type', '')}\n")
                f.write(f"杞﹀ご鐩镐技搴? {record.get('head_prob', 'N/A')}\n")
                f.write(f"杞﹀熬鐩镐技搴? {record.get('tail_prob', 'N/A')}\n")
                f.write(f"杈撳叆璺緞1: {record.get('input_path1', '')}\n")
                f.write(f"杈撳叆璺緞2: {record.get('input_path2', '')}\n")
                
                # 濡傛灉鏈夊鏍镐俊鎭?
                if record.get('reviewed'):
                    f.write(f"\n--- 澶嶆牳淇℃伅 ---\n")
                    f.write(f"澶嶆牳缁撴灉: {record.get('reviewed_case_type', '')}\n")
                    f.write(f"澶嶆牳浜哄憳: {record.get('reviewed_by', '')}\n")
                    f.write(f"澶嶆牳鏃堕棿: {record.get('reviewed_at', '')}\n")
                    f.write(f"澶嶆牳鐞嗙敱: {record.get('review_reason', '')}\n")
                
                f.write(f"\n瀵煎嚭鏃堕棿: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                exported_count = len([x for x in copied_files if isinstance(x, str) and x.lower().endswith('.jpg')])
                f.write(f"瀵煎嚭鍥剧墖鏁? {exported_count}\n")
                f.write(f"瀵煎嚭鏂囦欢: {', '.join(copied_files)}\n")
            
            exported_count = len([x for x in copied_files if isinstance(x, str) and x.lower().endswith('.jpg')])
            if exported_count == 0:
                return False, "export folder created but no .jpg files were found", export_folder
            return True, f"exported {exported_count} jpg(s): {', '.join([x for x in copied_files if isinstance(x, str) and x.lower().endswith('.jpg')])}", export_folder
        except Exception as e:
            return False, f"瀵煎嚭澶辫触: {str(e)}", None
    
    def export_batch(
        self,
        record_ids: List[str],
        export_path: Optional[str] = None,
        group_by: str = "case_type",
        image_types: Optional[List[str]] = None,
        include_summary: bool = True
    ) -> Tuple[bool, str, Optional[str]]:
        """
        鎵归噺瀵煎嚭璁板綍
        
        Args:
            record_ids: 璁板綍ID鍒楄〃
            export_path: 瀵煎嚭璺緞
            group_by: 鍒嗙粍鏂瑰紡 ("case_type" 鎴?"none")
            image_types: 瑕佸鍑虹殑鍥剧墖绫诲瀷
            include_summary: 鏄惁鐢熸垚姹囨€绘枃浠?
        
        Returns:
            (鎴愬姛, 娑堟伅, 瀵煎嚭璺緞)
        """
        try:
            if not record_ids:
                return False, "娌℃湁瑕佸鍑虹殑璁板綍", None
            
            # 鍒涘缓瀵煎嚭浠诲姟鏂囦欢澶?
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            task_folder = f"export_{timestamp}"
            if export_path is None:
                export_path = self.export_base_dir
            export_folder = os.path.join(export_path, task_folder)
            os.makedirs(export_folder, exist_ok=True)
            
            # 瀵煎嚭璁板綍
            results = []
            for record_id in record_ids:
                record = self.metrics.get_record(record_id)
                if not record:
                    results.append({
                        "record_id": record_id,
                        "success": False,
                        "message": "record not found",
                    })
                    continue
                
                # 纭畾瀛愭枃浠跺す
                if group_by == "case_type":
                    case_type = record.get("case_type", "unknown")
                    sub_folder = os.path.join(export_folder, case_type)
                else:
                    sub_folder = export_folder
                
                os.makedirs(sub_folder, exist_ok=True)
                
                # 瀵煎嚭鍗曟潯璁板綍
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
            
            # 鐢熸垚姹囨€绘枃浠?
            if include_summary:
                self._generate_summary_csv(results, export_folder)
                self._generate_export_log(results, export_folder, image_types)
            
            success_count = sum(1 for r in results if r["success"])
            return True, f"exported {success_count}/{len(record_ids)} record(s)", export_folder
        except Exception as e:
            return False, f"鎵归噺瀵煎嚭澶辫触: {str(e)}", None
    
    def _generate_summary_csv(self, results: List[Dict], export_folder: str):
        """鐢熸垚姹囨€籆SV鏂囦欢"""
        try:
            csv_path = os.path.join(export_folder, "export_summary.csv")
            import csv
            with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "record_id",
                    "ts",
                    "case_type",
                    "head_prob",
                    "tail_prob",
                    "success",
                    "message",
                ])
                for r in results:
                    writer.writerow([
                        r.get("record_id", ""),
                        r.get("ts", ""),
                        r.get("case_type", ""),
                        r.get("head_prob", ""),
                        r.get("tail_prob", ""),
                        bool(r.get("success")),
                        r.get("message", ""),
                    ])
        except Exception:
            pass
    
    def _generate_export_log(self, results: List[Dict], export_folder: str, image_types: Optional[List[str]]):
        """鐢熸垚瀵煎嚭鏃ュ織"""
        try:
            log_path = os.path.join(export_folder, "export_log.txt")
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("=" * 60 + "\n")
                f.write("瀵煎嚭鏃ュ織\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"瀵煎嚭鏃堕棿: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"鎬昏褰曟暟: {len(results)}\n")
                f.write(f"鎴愬姛: {sum(1 for r in results if r['success'])}\n")
                f.write(f"澶辫触: {sum(1 for r in results if not r['success'])}\n")
                
                if image_types:
                    f.write(f"\n瀵煎嚭鍥剧墖绫诲瀷: {', '.join(image_types)}\n")
                else:
                    f.write(f"\n瀵煎嚭鍥剧墖绫诲瀷: 鍏ㄩ儴\n")
                
                f.write("\n" + "=" * 60 + "\n")
                f.write("璇︾粏缁撴灉\n")
                f.write("=" * 60 + "\n\n")
                
                for r in results:
                    status = "OK" if r.get("success") else "FAIL"
                    f.write(f"{status} {r['record_id']} - {r['message']}\n")
        except Exception:
            pass


_EXPORTER = RecordExporter(_METRICS)


class RecordExporterLegacy:
    """Record exporter (legacy)."""
    
    def __init__(self, export_base_dir: str = None):
        if export_base_dir is None:
            export_base_dir = os.path.join(os.path.dirname(__file__), "exports")
        self.export_base_dir = export_base_dir
        os.makedirs(self.export_base_dir, exist_ok=True)
    
    def export_single(self, record_id: str, export_path: str = None, 
                     include_meta: bool = True) -> Tuple[bool, str, Optional[str]]:
        """
        瀵煎嚭鍗曟潯璁板綍
        
        Args:
            record_id: 璁板綍ID
            export_path: 瀵煎嚭璺緞锛堝彲閫夛級
            include_meta: 鏄惁鍖呭惈鍏冩暟鎹枃浠?
        
        Returns:
            (鎴愬姛, 娑堟伅, 瀵煎嚭璺緞)
        """
        try:
            # 鑾峰彇璁板綍
            record = _METRICS.get_record(record_id)
            if not record:
                return False, "record not found", None
            
            # 鑾峰彇鍥剧墖鐩綍
            image_dir = record.get("image_dir", "")
            if not image_dir or not os.path.exists(image_dir):
                return False, "image dir not found", None
            
            # 纭畾瀵煎嚭璺緞
            if export_path is None:
                export_path = self.export_base_dir
            
            case_type = record.get("case_type", "unknown")
            folder_name = f"{record_id}_{case_type}"
            target_dir = os.path.join(export_path, folder_name)
            os.makedirs(target_dir, exist_ok=True)
            
            # 澶嶅埗鍥剧墖
            image_files = ["vehicle1.jpg", "vehicle2.jpg", "head1.jpg", 
                          "head2.jpg", "tail1.jpg", "tail2.jpg"]
            copied_count = 0
            
            for img_name in image_files:
                src = os.path.join(image_dir, img_name)
                if os.path.exists(src):
                    dst = os.path.join(target_dir, img_name)
                    shutil.copy2(src, dst)
                    copied_count += 1
            
            # 鐢熸垚鍏冩暟鎹枃浠?
            if include_meta:
                info_path = os.path.join(target_dir, "info.txt")
                with open(info_path, "w", encoding="utf-8") as f:
                    f.write(f"璁板綍ID: {record_id}\n")
                    f.write(f"妫€娴嬫椂闂? {record.get('ts', '')}\n")
                    f.write(f"绯荤粺鍒ゅ畾: {case_type}\n")
                    f.write(f"杞﹀ご鐩镐技搴? {record.get('head_prob', 'N/A')}\n")
                    f.write(f"杞﹀熬鐩镐技搴? {record.get('tail_prob', 'N/A')}\n")
                    f.write(f"杈撳叆璺緞1: {record.get('input_path1', '')}\n")
                    f.write(f"杈撳叆璺緞2: {record.get('input_path2', '')}\n")
                    
                    # 澶嶆牳淇℃伅
                    if record.get("reviewed", False):
                        f.write(f"\n--- 澶嶆牳淇℃伅 ---\n")
                        f.write(f"澶嶆牳缁撴灉: {record.get('reviewed_case_type', '')}\n")
                        f.write(f"澶嶆牳浜哄憳: {record.get('reviewed_by', '')}\n")
                        f.write(f"澶嶆牳鏃堕棿: {record.get('reviewed_at', '')}\n")
                        f.write(f"澶嶆牳鐞嗙敱: {record.get('review_reason', '')}\n")
                    
                    if record.get("note"):
                        f.write(f"\n澶囨敞: {record.get('note')}\n")
            
            return True, f"exported {copied_count} file(s)", target_dir
        except Exception as e:
            return False, f"瀵煎嚭澶辫触: {str(e)}", None
    
    def export_batch(self, record_ids: List[str], export_path: str = None,
                    group_by: str = "case_type", image_types: Optional[List[str]] = None,
                    include_summary: bool = True) -> Tuple[bool, str, Optional[str]]:
        """
        鎵归噺瀵煎嚭璁板綍
        
        Args:
            record_ids: 璁板綍ID鍒楄〃
            export_path: 瀵煎嚭璺緞锛堝彲閫夛級
            group_by: 鍒嗙粍鏂瑰紡 case_type/date/none
            image_types: 瑕佸鍑虹殑鍥剧墖绫诲瀷鍒楄〃锛堝彲閫夛紝None琛ㄧず瀵煎嚭鍏ㄩ儴锛?
            include_summary: 鏄惁鐢熸垚姹囨€绘枃浠?
        
        Returns:
            (鎴愬姛, 娑堟伅, 瀵煎嚭璺緞)
        """
        try:
            if not record_ids:
                return False, "no records to export", None

            if export_path is None:
                export_path = self.export_base_dir

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            task_folder = f"export_{timestamp}"
            task_dir = os.path.join(export_path, task_folder)
            os.makedirs(task_dir, exist_ok=True)

            results: List[Dict[str, Any]] = []
            for rid in record_ids:
                rec = _METRICS.get_record(rid)
                if not rec:
                    results.append({"record_id": rid, "success": False, "message": "record not found"})
                    continue

                case_type = rec.get("case_type", "unknown")
                if group_by == "case_type":
                    sub_folder = os.path.join(task_dir, case_type)
                elif group_by == "date":
                    date_str = rid.split("_")[0]
                    sub_folder = os.path.join(task_dir, date_str)
                else:
                    sub_folder = task_dir
                os.makedirs(sub_folder, exist_ok=True)

                ok, msg, _ = self.export_single(rid, sub_folder, include_meta=True)
                results.append({"record_id": rid, "success": ok, "message": msg, "case_type": case_type})

            if include_summary:
                try:
                    import csv
                    summary_path = os.path.join(task_dir, "export_summary.csv")
                    with open(summary_path, "w", encoding="utf-8-sig", newline="") as f:
                        w = csv.writer(f)
                        w.writerow(["record_id", "case_type", "success", "message"])
                        for r in results:
                            w.writerow([r.get("record_id", ""), r.get("case_type", ""), bool(r.get("success")), r.get("message", "")])
                except Exception:
                    pass

            success_count = sum(1 for r in results if r.get("success"))
            return True, f"exported {success_count}/{len(results)} record(s)", task_dir
        except Exception as e:
            return False, f"鎵归噺瀵煎嚭澶辫触: {str(e)}", None
    
    def export_by_filter(self, start_date: str = None, end_date: str = None,
                        case_types: List[str] = None, export_path: str = None) -> Tuple[bool, str, Optional[str]]:
        """
        鎸夋潯浠跺鍑?
        
        Args:
            start_date: 寮€濮嬫棩鏈?
            end_date: 缁撴潫鏃ユ湡
            case_types: 绫诲瀷鍒楄〃
            export_path: 瀵煎嚭璺緞
        
        Returns:
            (鎴愬姛, 娑堟伅, 瀵煎嚭璺緞)
        """
        try:
            # 鏌ヨ绗﹀悎鏉′欢鐨勮褰?
            result = _METRICS.query_records(
                start_date=start_date,
                end_date=end_date,
                case_type=None,
                include_deleted=False,
                limit=10000,
                offset=0
            )
            
            records = result.get("records", [])
            
            # 鎸夌被鍨嬬瓫閫?
            if case_types:
                records = [r for r in records if r.get("case_type") in case_types]
            
            if not records:
                return False, "没有符合条件的记录", None
            
            # 鎻愬彇璁板綍ID
            record_ids = [r.get("record_id") for r in records if r.get("record_id")]
            
            # 鎵归噺瀵煎嚭
            return self.export_batch(record_ids, export_path, group_by="case_type", include_summary=True)
        except Exception as e:
            return False, f"鎸夋潯浠跺鍑哄け璐? {str(e)}", None
    
    def _generate_summary_csv(self, records: List[Dict], output_dir: str):
        """鐢熸垚姹囨€籆SV"""
        try:
            csv_path = os.path.join(output_dir, "export_summary.csv")
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "璁板綍ID", "妫€娴嬫椂闂?, "绯荤粺鍒ゅ畾", "杞﹀ご鐩镐技搴?, "杞﹀熬鐩镐技搴?,
                    "鏄惁澶嶆牳", "澶嶆牳缁撴灉", "澶嶆牳浜哄憳", "杈撳叆璺緞1", "杈撳叆璺緞2"
                ])
                
                for record in records:
                    writer.writerow([
                        record.get("record_id", ""),
                        record.get("ts", ""),
                        record.get("case_type", ""),
                        record.get("head_prob", ""),
                        record.get("tail_prob", ""),
                        "鏄? if record.get("reviewed", False) else "鍚?,
                        record.get("reviewed_case_type", ""),
                        record.get("reviewed_by", ""),
                        record.get("input_path1", ""),
                        record.get("input_path2", "")
                    ])
        except Exception:
            pass
    
    def _generate_export_log(self, exported: List[Dict], failed: List[Dict], output_dir: str):
        """鐢熸垚瀵煎嚭鏃ュ織"""
        try:
            log_path = os.path.join(output_dir, "export_log.txt")
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"瀵煎嚭鏃堕棿: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"鎴愬姛瀵煎嚭: {len(exported)} 鏉n")
                f.write(f"瀵煎嚭澶辫触: {len(failed)} 鏉n\n")
                
                if failed:
                    f.write("--- 澶辫触璁板綍 ---\n")
                    for item in failed:
                        f.write(f"璁板綍ID: {item['record_id']}, 閿欒: {item['error']}\n")
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
) -> Optional[str]:
    """
    璁板綍鎸囨爣骞朵繚瀛樺浘鐗?
    
    Args:
        original_images: 鍖呭惈鍘熷鍥剧墖鐨勫瓧鍏?{"original1": data_url, "original2": data_url}
    
    Returns:
        record_id if images saved, else None
    """
    record_id = None
    image_dir = None
    
    # 濡傛灉鏈夐瑙堝浘锛屼繚瀛樺畠浠?
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
            "endpoint": endpoint,
            "source": source,
            "lat_ms": lat_ms,
            "protected": False,
            "deleted": False,
            "note": "",
        }
        
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
    }
    
    if record_id:
        ev["record_id"] = record_id
        ev["image_dir"] = image_dir
        ev["input_path1"] = input_path1
        ev["input_path2"] = input_path2
    
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

    def predict_from_pil(self, img1: Image.Image, img2: Image.Image) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        return _compute_head_tail_probs_pil(img1, img2)

    def classify(self, head_prob: Optional[float], tail_prob: Optional[float]) -> str:
        return _classify_case(head_prob, tail_prob)
    
    def classify_with_ai(self, head_prob: Optional[float], tail_prob: Optional[float], 
                        vehicle1_img: Optional[Image.Image] = None, 
                        vehicle2_img: Optional[Image.Image] = None) -> str:
        return _classify_case_with_ai(head_prob, tail_prob, vehicle1_img, vehicle2_img)


def _init_models() -> None:
    global _INITIALIZED, _CROPPER, _HEAD_MODEL, _TAIL_MODEL, _HEADTAIL_MODEL, _IMAGE_RESOLVER
    if _INITIALIZED:
        return
    with _INIT_LOCK:
        if _INITIALIZED:
            return

        head_model_path = os.environ.get(
            "HEAD_MODEL_PATH",
            r"D:\project\data_chuli\demo\demo\Siamese-pytorch-master\logs\head\1211\best_epoch_weights.pth",
        )
        tail_model_path = os.environ.get(
            "TAIL_MODEL_PATH",
            r"D:\\project\data_chuli\demo\demo\Siamese-pytorch-master\\logs\\best_epoch_weights.pth",
        )
        headtail_model_path = os.environ.get(
            "HEADTAIL_MODEL_PATH",
            r"D:\data2\runs\detect\train\weights\best.pt",
        )

        _CROPPER = VehicleCropper()
        _HEAD_MODEL = Siamese(model_path=head_model_path)
        _TAIL_MODEL = Siamese(model_path=tail_model_path)
        _HEADTAIL_MODEL = YOLO(headtail_model_path)
        if _IMAGE_RESOLVER is None:
            _IMAGE_RESOLVER = ImagePathResolver()

        _INITIALIZED = True


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
    """灏哖IL鍥剧墖杞崲涓哄師濮嬪ぇ灏忕殑data URL锛堜笉缂╂斁锛?""
    img = pil_img
    if img is None:
        return ""
    img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)  # 浣跨敤鏇撮珮璐ㄩ噺
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _crop_part_from_vehicle_pil(vehicle_image: Image.Image, cls_id: int) -> Image.Image:
    try:
        if vehicle_image is None:
            return vehicle_image
        if _HEADTAIL_MODEL is None:
            return vehicle_image

        bgr = _pil_to_bgr(vehicle_image)
        with torch.inference_mode():
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


def _compute_head_tail_probs(path1: str, path2: str, _oom_retry: bool = True) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    img1 = img2 = None
    head1 = head2 = tail1 = tail2 = None
    head_prob = tail_prob = None
    try:
        _init_models()
        if _CROPPER is None or _HEAD_MODEL is None or _TAIL_MODEL is None:
            return None, None, "models not initialized"

        with _INFER_LOCK:
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
        if _is_cuda_oom_error(e):
            _recover_from_cuda_oom("compute_head_tail_probs")
            if _oom_retry:
                return _compute_head_tail_probs(path1, path2, _oom_retry=False)
            _recover_from_cuda_oom("compute_head_tail_probs.final", allow_restart=True)
            return None, None, "gpu busy, auto recovery in progress"
        return None, None, str(e)
    finally:
        del img1, img2, head1, head2, tail1, tail2, head_prob, tail_prob
        _release_cuda_memory()


def _compute_probs_and_previews_pil(
    img1: Image.Image, img2: Image.Image, _oom_retry: bool = True
) -> Tuple[Optional[float], Optional[float], Optional[Dict[str, str]], Optional[Dict[str, str]], Optional[Image.Image], Optional[Image.Image], Optional[str]]:
    """
    璁＄畻姒傜巼骞剁敓鎴愰瑙堝浘鍜屽師濮嬪浘
    
    Returns:
        (head_prob, tail_prob, previews, original_images, vehicle1_pil, vehicle2_pil, error)
    """
    v1 = v2 = None
    h1 = h2 = t1 = t2 = None
    head_prob = tail_prob = None
    previews = None
    original_images = None
    try:
        _init_models()
        if _CROPPER is None or _HEAD_MODEL is None or _TAIL_MODEL is None:
            return None, None, None, None, None, None, "models not initialized"

        # 淇濆瓨鍘熷鍥剧墖鐨刣ata URL
        with _INFER_LOCK:
            original_images = {
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

        previews = {
            "vehicle1": _pil_to_jpeg_data_url(v1),
            "vehicle2": _pil_to_jpeg_data_url(v2),
            "head1": _pil_to_jpeg_data_url(h1),
            "head2": _pil_to_jpeg_data_url(h2),
            "tail1": _pil_to_jpeg_data_url(t1),
            "tail2": _pil_to_jpeg_data_url(t2),
        }
        return float(head_prob), float(tail_prob), previews, original_images, v1, v2, None
    except Exception as e:
        if _is_cuda_oom_error(e):
            _recover_from_cuda_oom("compute_probs_and_previews_pil")
            if _oom_retry:
                return _compute_probs_and_previews_pil(img1, img2, _oom_retry=False)
            _recover_from_cuda_oom("compute_probs_and_previews_pil.final", allow_restart=True)
            return None, None, None, None, None, None, "gpu busy, auto recovery in progress"
        return None, None, None, None, None, None, str(e)
    finally:
        del h1, h2, t1, t2, head_prob, tail_prob
        _release_cuda_memory()


def _compute_head_tail_probs_pil(img1: Image.Image, img2: Image.Image, _oom_retry: bool = True) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    p1 = p2 = None
    head1 = head2 = tail1 = tail2 = None
    head_prob = tail_prob = None
    try:
        _init_models()
        if _CROPPER is None or _HEAD_MODEL is None or _TAIL_MODEL is None:
            return None, None, "models not initialized"

        with _INFER_LOCK:
            p1 = _CROPPER.process_pil(img1)
            p2 = _CROPPER.process_pil(img2)

            head1 = _crop_part_from_vehicle_pil(p1, cls_id=0)
            head2 = _crop_part_from_vehicle_pil(p2, cls_id=0)
            tail1 = _crop_part_from_vehicle_pil(p1, cls_id=1)
            tail2 = _crop_part_from_vehicle_pil(p2, cls_id=1)

            head_prob = _HEAD_MODEL.detect_image(head1, head2)
            tail_prob = _TAIL_MODEL.detect_image(tail1, tail2)

        if hasattr(head_prob, "item"):
            head_prob = head_prob.item()
        if hasattr(tail_prob, "item"):
            tail_prob = tail_prob.item()

        return float(head_prob), float(tail_prob), None
    except Exception as e:
        if _is_cuda_oom_error(e):
            _recover_from_cuda_oom("compute_head_tail_probs_pil")
            if _oom_retry:
                return _compute_head_tail_probs_pil(img1, img2, _oom_retry=False)
            _recover_from_cuda_oom("compute_head_tail_probs_pil.final", allow_restart=True)
            return None, None, "gpu busy, auto recovery in progress"
        return None, None, str(e)
    finally:
        del p1, p2, head1, head2, tail1, tail2, head_prob, tail_prob
        _release_cuda_memory()


def _ai_check_timeout_sec() -> float:
    raw = str(os.environ.get("AI_CHECK_TIMEOUT_SEC", "20")).strip()
    try:
        value = float(raw)
    except Exception:
        value = 20.0
    return max(1.0, value)


def _run_ai_check_with_timeout(vehicle1_path: str, vehicle2_path: str) -> str:
    result_holder: Dict[str, str] = {"result": "閺冪姵纭堕崚銈嗘焽"}
    error_holder: Dict[str, str] = {}

    def _worker() -> None:
        try:
            checker = VehicleCheck(model_name="qwen3.5:9b")
            result_holder["result"] = checker.check_vehicle(vehicle1_path, vehicle2_path)
        except Exception as e:
            error_holder["error"] = str(e)

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()
    worker.join(timeout=_ai_check_timeout_sec())
    if worker.is_alive():
        print("[ai] vision check timeout, fallback to primary result")
        return "閺冪姵纭堕崚銈嗘焽"
    if "error" in error_holder:
        raise RuntimeError(error_holder["error"])
    return result_holder.get("result", "閺冪姵纭堕崚銈嗘焽")


def _ai_vision_check(vehicle1_img: Image.Image, vehicle2_img: Image.Image) -> str:
    """
    浣跨敤AI瑙嗚妯″瀷杩涜浜屾鍒ゆ柇
    
    Args:
        vehicle1_img: 瑁佸垏鍚庣殑绗竴寮犺溅杈嗗浘鐗?
        vehicle2_img: 瑁佸垏鍚庣殑绗簩寮犺溅杈嗗浘鐗?
    
    Returns:
        鏈€缁堝垽鏂粨鏋? "濂楃墝", "鎹㈡寕", "姝ｅ父" 鎴?"鏃犳硶鍒ゆ柇"
    """
    try:
        # 闃插尽锛氶伩鍏嶈鎶?dataURL(str) 浼犺繘鏉ュ鑷?.save 鎶ラ敊
        if not hasattr(vehicle1_img, "save") or not hasattr(vehicle2_img, "save"):
            return "鏃犳硶鍒ゆ柇"

        # 鍒涘缓涓存椂鐩綍淇濆瓨鍥剧墖
        temp_dir = tempfile.mkdtemp(prefix="ai_check_")
        
        # 淇濆瓨瑁佸垏鍚庣殑杞﹁締鍥剧墖
        vehicle1_path = os.path.join(temp_dir, "vehicle1.jpg")
        vehicle2_path = os.path.join(temp_dir, "vehicle2.jpg")
        
        vehicle1_img.save(vehicle1_path, "JPEG", quality=85)
        vehicle2_img.save(vehicle2_path, "JPEG", quality=85)
        
        # 璋冪敤AI瑙嗚妯″瀷
        result = _run_ai_check_with_timeout(vehicle1_path, vehicle2_path)
        
        # 娓呯悊涓存椂鏂囦欢
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return result
        
    except Exception as e:
        # 娓呯悊涓存椂鏂囦欢
        try:
            if 'temp_dir' in locals():
                shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass
        
        print(f"AI瑙嗚妫€鏌ュけ璐? {e}")
        return "鏃犳硶鍒ゆ柇"


def _update_record_with_ai_result(record_id: str, ai_result: Optional[str], 
                                success: bool, reason: str = "") -> bool:
    """
    鏇存柊璁板綍鐨凙I澶嶆缁撴灉
    
    Args:
        record_id: 璁板綍ID
        ai_result: AI鍒ゆ柇缁撴灉
        success: 澶嶆鏄惁鎴愬姛
        reason: 澶辫触鍘熷洜
    
    Returns:
        鏄惁鏇存柊鎴愬姛
    """
    try:
        # 浠庤褰旾D涓彁鍙栨棩鏈?
        date_part = record_id.split("_")[0]
        
        # 鏌ユ壘璁板綍鐨勫浘鐗囩洰褰?
        images_dir = os.path.join(_METRICS._images_dir, date_part)
        if not os.path.exists(images_dir):
            return False
        
        record_dir = None
        for folder in os.listdir(images_dir):
            if folder == record_id:
                record_dir = os.path.join(images_dir, folder)
                break
        
        if not record_dir:
            return False
        
        # 璇诲彇鍏冩暟鎹枃浠?
        meta_file = os.path.join(record_dir, "meta.json")
        if not os.path.exists(meta_file):
            return False
        
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        
        # 鏇存柊鍏冩暟鎹?
        original_case_type = meta.get("case_type", "")
        
        # 娣诲姞澶嶆淇℃伅
        recheck_info = {
            "attempted": True,
            "success": success,
            "ai_result": ai_result,
            "reason": reason,
            "recheck_at": datetime.datetime.now().isoformat(timespec="milliseconds")
        }
        
        meta["ai_recheck"] = recheck_info
        
        # 濡傛灉澶嶆鎴愬姛锛屾洿鏂癱ase_type
        if success and ai_result:
            # AI缁撴灉宸茬粡鏄嫳鏂囷紝鐩存帴浣跨敤
            meta["case_type"] = ai_result
        
        # 淇濆瓨鍏冩暟鎹?
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        
        # 鏇存柊jsonl鏂囦欢涓殑璁板綍
        date_part = record_id.split("_")[0]
        log_path = os.path.join(_METRICS._log_dir, f"stats_{date_part}.jsonl")
        
        if os.path.exists(log_path):
            lines = []
            updated = False
            with open(log_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line.strip())
                        if r.get("record_id") == record_id:
                            print(f"鏇存柊璁板綍 {record_id}: 鍘焎ase_type={r.get('case_type')}, AI缁撴灉={ai_result}")
                            r["ai_recheck"] = recheck_info
                            if success and ai_result:
                                # AI缁撴灉宸茬粡鏄嫳鏂囷紝鐩存帴浣跨敤
                                r["case_type"] = ai_result
                                print(f"鏇存柊case_type: {original_case_type} -> {ai_result}")
                                updated = True
                        lines.append(json.dumps(r, ensure_ascii=False) + "\n")
                    except Exception as e:
                        print(f"瑙ｆ瀽JSONL琛屽け璐? {e}")
                        lines.append(line)
            
            if updated:
                with open(log_path, "w", encoding="utf-8") as f:
                    f.writelines(lines)
                print(f"JSONL鏂囦欢宸叉洿鏂? {log_path}")
            else:
                print(f"璁板綍鏈壘鍒版垨鏃犻渶鏇存柊: {record_id}")
        
        return True
        
    except Exception as e:
        print(f"鏇存柊璁板綍澶辫触: {e}")
        return False


def _get_abnormal_records_for_recheck() -> List[Dict[str, Any]]:
    """
    鑾峰彇闇€瑕佸妫€鐨勫紓甯歌褰?
    
    Returns:
        寮傚父璁板綍鍒楄〃
    """
    abnormal_records = []
    
    try:
        # 閬嶅巻鍥剧墖鐩綍
        if not os.path.exists(_METRICS._images_dir):
            return abnormal_records
        
        for date_folder in os.listdir(_METRICS._images_dir):
            date_path = os.path.join(_METRICS._images_dir, date_folder)
            if not os.path.isdir(date_path):
                continue
            
            for record_folder in os.listdir(date_path):
                record_path = os.path.join(date_path, record_folder)
                if not os.path.isdir(record_path):
                    continue
                
                # 妫€鏌ehicle1.jpg鍜寁ehicle2.jpg鏄惁瀛樺湪
                vehicle1_path = os.path.join(record_path, "vehicle1.jpg")
                vehicle2_path = os.path.join(record_path, "vehicle2.jpg")
                meta_path = os.path.join(record_path, "meta.json")
                
                if not all(os.path.exists(p) for p in [vehicle1_path, vehicle2_path, meta_path]):
                    continue
                
                # 璇诲彇鍏冩暟鎹?
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    
                    case_type = meta.get("case_type", "")
                    ai_recheck = meta.get("ai_recheck", {})
                    
                    # 妫€鏌ユ槸鍚﹀凡缁忔垚鍔熷妫€杩?
                    recheck_attempted = ai_recheck.get("attempted", False)
                    recheck_success = ai_recheck.get("success", False)
                    
                    # 鍙鐞嗗鐗屽拰鎹㈡寕锛屼笖鏈妫€鎴栧妫€澶辫触鐨勮褰?
                    if case_type in ["fake_plate", "change_trailer"] and (not recheck_attempted or not recheck_success):
                        abnormal_records.append({
                            "record_id": record_folder,
                            "case_type": case_type,
                            "vehicle1_path": vehicle1_path,
                            "vehicle2_path": vehicle2_path,
                            "meta_path": meta_path,
                            "meta": meta
                        })
                except Exception:
                    continue
        
        return abnormal_records
        
    except Exception as e:
        print(f"鑾峰彇寮傚父璁板綍澶辫触: {e}")
        return abnormal_records


def _batch_recheck_abnormal_records() -> None:
    """
    鎵归噺澶嶆寮傚父璁板綍锛堝悗鍙颁换鍔★級
    """
    global _RECHECK_STATUS
    
    try:
        # 鑾峰彇闇€瑕佸妫€鐨勮褰?
        abnormal_records = _get_abnormal_records_for_recheck()
        
        with _RECHECK_LOCK:
            _RECHECK_STATUS.update({
                "running": True,
                "started_at": datetime.datetime.now().isoformat(timespec="milliseconds"),
                "total": len(abnormal_records),
                "processed": 0,
                "success": 0,
                "failed": 0,
                "current_record": None,
                "error": None,
                "results": []
            })
        
        # 鎵归噺澶勭悊
        for i, record in enumerate(abnormal_records):
            try:
                with _RECHECK_LOCK:
                    _RECHECK_STATUS["processed"] = i + 1
                    _RECHECK_STATUS["current_record"] = record["record_id"]
                
                # 鎵цAI澶嶆
                ai_result = _ai_vision_check_from_paths(
                    record["vehicle1_path"], 
                    record["vehicle2_path"]
                )
                
                # 鐩存帴浣跨敤鑻辨枃缁撴灉
                success = ai_result in ["fake_plate", "change_trailer", "normal"]
                
                # 鏇存柊璁板綍
                updated = _update_record_with_ai_result(
                    record["record_id"], 
                    ai_result,  # 浣跨敤鑻辨枃缁撴灉
                    success, 
                    "" if success else "AI unknown"
                )
                
                # 璁板綍缁撴灉
                result_info = {
                    "record_id": record["record_id"],
                    "original_case_type": record["case_type"],
                    "ai_result": ai_result,
                    "success": success,
                    "updated": updated
                }
                
                with _RECHECK_LOCK:
                    _RECHECK_STATUS["results"].append(result_info)
                    if success:
                        _RECHECK_STATUS["success"] += 1
                    else:
                        _RECHECK_STATUS["failed"] += 1
                
                print(f"澶嶆璁板綍 {record['record_id']}: {record['case_type']} -> {ai_result} ({'鎴愬姛' if success else '澶辫触'})")
                
            except Exception as e:
                print(f"澶嶆璁板綍 {record['record_id']} 澶辫触: {e}")
                
                with _RECHECK_LOCK:
                    _RECHECK_STATUS["failed"] += 1
                    _RECHECK_STATUS["results"].append({
                        "record_id": record["record_id"],
                        "original_case_type": record["case_type"],
                        "ai_result": None,
                        "success": False,
                        "updated": False,
                        "error": str(e)
                    })
        
        # 瀹屾垚澶嶆
        with _RECHECK_LOCK:
            _RECHECK_STATUS["running"] = False
            _RECHECK_STATUS["current_record"] = None
        
        print(f"鎵归噺澶嶆瀹屾垚: 鎬昏 {_RECHECK_STATUS['total']}, 鎴愬姛 {_RECHECK_STATUS['success']}, 澶辫触 {_RECHECK_STATUS['failed']}")
        
    except Exception as e:
        with _RECHECK_LOCK:
            _RECHECK_STATUS["running"] = False
            _RECHECK_STATUS["error"] = str(e)
        print(f"鎵归噺澶嶆澶辫触: {e}")


def _ai_vision_check_from_paths(vehicle1_path: str, vehicle2_path: str) -> str:
    """
    浣跨敤AI瑙嗚妯″瀷杩涜浜屾鍒ゆ柇锛堜粠鏂囦欢璺緞璇诲彇锛?
    
    Args:
        vehicle1_path: 绗竴寮犺溅杈嗗浘鐗囪矾寰?
        vehicle2_path: 绗簩寮犺溅杈嗗浘鐗囪矾寰?
    
    Returns:
        鏈€缁堝垽鏂粨鏋? "fake_plate", "change_trailer", "normal" 鎴?"unknown"
    """
    try:
        # 楠岃瘉鏂囦欢鏄惁瀛樺湪
        if not os.path.exists(vehicle1_path) or not os.path.exists(vehicle2_path):
            return "unknown"
        
        # 璋冪敤AI瑙嗚妯″瀷
        result = _run_ai_check_with_timeout(vehicle1_path, vehicle2_path)
        
        # 灏嗕腑鏂囩粨鏋滆浆鎹负鑻辨枃
        result_mapping = {
            "濂楃墝": "fake_plate",
            "鎹㈡寕": "change_trailer", 
            "姝ｅ父": "normal",
            "鏃犳硶鍒ゆ柇": "unknown"
        }
        english_result = result_mapping.get(result, "unknown")
        
        return english_result
        
    except Exception as e:
        print(f"AI瑙嗚妫€鏌ュけ璐? {e}")
        return "unknown"


def _classify_case_with_ai(head_prob: Optional[float], tail_prob: Optional[float], 
                          vehicle1_img: Optional[Image.Image] = None, 
                          vehicle2_img: Optional[Image.Image] = None) -> str:
    """
    甯︽湁AI瑙嗚浜屾鍒ゆ柇鐨勫垎绫诲嚱鏁?
    
    Args:
        head_prob: 杞﹀ご鐩镐技搴︽鐜?
        tail_prob: 杞﹀熬鐩镐技搴︽鐜?
        vehicle1_img: 瑁佸垏鍚庣殑绗竴寮犺溅杈嗗浘鐗?
        vehicle2_img: 瑁佸垏鍚庣殑绗簩寮犺溅杈嗗浘鐗?
    
    Returns:
        鏈€缁堝垽鏂粨鏋?
    """
    if head_prob is None or tail_prob is None:
        return "abnormal"

    # 浣跨敤鍏ㄥ眬闃堝€煎彉閲?
    head_low_th = _HEAD_THRESHOLD
    head_high_th = float(os.environ.get("HEAD_HIGH_TH", "0.8"))  # 杞﹀ご楂樼浉浼煎害闃堝€?
    tail_low_th = _TAIL_THRESHOLD

    # 绗竴娆″垽鏂?
    if head_prob < head_low_th:
        first_result = "fake_plate"
    elif head_prob >= head_high_th and tail_prob <= tail_low_th:
        first_result = "change_trailer"
    else:
        first_result = "normal"
    
    # 濡傛灉绗竴娆″垽鏂负姝ｅ父锛岀洿鎺ヨ繑鍥?
    if first_result == "normal":
        return "normal"

    # 鏋佷綆鐩镐技搴︾洿鎺ュ畾鎬э紝涓嶈繘鍏I浜屾閴村埆锛堝浐瀹氶槇鍊?0.3锛?
    if first_result == "fake_plate" and head_prob < 0.3:
        return "fake_plate"
    if first_result == "change_trailer" and tail_prob < 0.3:
        return "change_trailer"
    
    # 濡傛灉绗竴娆″垽鏂负濂楃墝鎴栨崲鎸傦紝杩涜AI瑙嗚浜屾鍒ゆ柇
    if vehicle1_img is not None and vehicle2_img is not None:
        ai_result = _ai_vision_check(vehicle1_img, vehicle2_img)
        
        # 鏄犲皠AI缁撴灉鍒扮郴缁熺粨鏋?
        result_mapping = {
            "濂楃墝": "fake_plate",
            "鎹㈡寕": "change_trailer", 
            "姝ｅ父": "normal",
            "鏃犳硶鍒ゆ柇": first_result  # AI鏃犳硶鍒ゆ柇鏃朵娇鐢ㄧ涓€娆＄粨鏋?
        }
        
        final_result = result_mapping.get(ai_result, first_result)
        print(f"绗竴娆″垽鏂? {first_result}, AI鍒ゆ柇: {ai_result}, 鏈€缁堢粨鏋? {final_result}")
        return final_result
    
    # 濡傛灉娌℃湁鎻愪緵杞﹁締鍥剧墖锛岃繑鍥炵涓€娆″垽鏂粨鏋?
    return first_result


def _run_ai_check_with_timeout(vehicle1_path: str, vehicle2_path: str) -> Dict[str, Any]:
    result_holder: Dict[str, Any] = {
        "result": {
            "decision": "鏃犳硶鍒ゆ柇",
            "confidence": 0.0,
            "stable_same_features": [],
            "stable_diff_features": [],
            "interference_factors": [],
            "lighting_interference": False,
            "reason": "",
            "note": "",
            "raw_output": "",
        }
    }
    error_holder: Dict[str, str] = {}

    def _worker() -> None:
        try:
            checker = VehicleCheck(model_name="qwen3.5:9b")
            result_holder["result"] = checker.check_vehicle(vehicle1_path, vehicle2_path)
        except Exception as e:
            error_holder["error"] = str(e)

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()
    worker.join(timeout=_ai_check_timeout_sec())
    if worker.is_alive():
        print("[ai] vision check timeout, fallback to primary result")
        return {
            "decision": "鏃犳硶鍒ゆ柇",
            "confidence": 0.0,
            "stable_same_features": [],
            "stable_diff_features": [],
            "interference_factors": ["ai_timeout"],
            "lighting_interference": False,
            "reason": "ai timeout, fallback to first stage result",
            "note": "",
            "raw_output": "",
        }
    if "error" in error_holder:
        raise RuntimeError(error_holder["error"])

    result = result_holder.get("result")
    if isinstance(result, dict):
        return result
    return {
        "decision": "鏃犳硶鍒ゆ柇",
        "confidence": 0.0,
        "stable_same_features": [],
        "stable_diff_features": [],
        "interference_factors": ["ai_invalid_format"],
        "lighting_interference": False,
        "reason": "ai invalid format",
        "note": "",
        "raw_output": str(result or ""),
    }


def _normalize_ai_decision(ai_result: Any) -> Dict[str, Any]:
    if not isinstance(ai_result, dict):
        return {
            "decision": "unknown",
            "raw_decision": "unknown",
            "confidence": 0.0,
            "stable_same_features": [],
            "stable_diff_features": [],
            "interference_factors": [],
            "lighting_interference": False,
            "reason": "",
            "note": "",
        }

    raw_decision = str(ai_result.get("decision") or "").strip()
    decision_mapping = {
        "濂楃墝": "fake_plate",
        "鎹㈡寕": "change_trailer",
        "姝ｅ父": "normal",
        "unknown": "unknown",
        "fake_plate": "fake_plate",
        "change_trailer": "change_trailer",
        "normal": "normal",
        "unknown": "unknown",
    }

    try:
        confidence = float(ai_result.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    stable_same = ai_result.get("stable_same_features")
    stable_diff = ai_result.get("stable_diff_features")
    interference = ai_result.get("interference_factors")
    if not isinstance(stable_same, list):
        stable_same = []
    if not isinstance(stable_diff, list):
        stable_diff = []
    if not isinstance(interference, list):
        interference = []

    return {
        "decision": decision_mapping.get(raw_decision, "unknown"),
        "raw_decision": raw_decision or "unknown",
        "confidence": confidence,
        "stable_same_features": [str(x).strip() for x in stable_same if str(x).strip()],
        "stable_diff_features": [str(x).strip() for x in stable_diff if str(x).strip()],
        "interference_factors": [str(x).strip() for x in interference if str(x).strip()],
        "lighting_interference": bool(ai_result.get("lighting_interference", False)),
        "reason": str(ai_result.get("reason") or "").strip(),
        "note": str(ai_result.get("note") or "").strip(),
    }


def _should_accept_ai_result(first_result: str, ai_info: Dict[str, Any]) -> bool:
    ai_decision = str(ai_info.get("decision") or "unknown")
    confidence = float(ai_info.get("confidence", 0.0) or 0.0)
    stable_diff_count = len(ai_info.get("stable_diff_features") or [])
    lighting_interference = bool(ai_info.get("lighting_interference", False))
    interference_text = " ".join(ai_info.get("interference_factors") or []).lower()

    if lighting_interference:
        return False
    if any(keyword in interference_text for keyword in ["lamp", "night", "exposure", "glare", "brightness"]):
        return False
    if ai_decision == "unknown":
        return False
    if ai_decision == first_result:
        return confidence >= 0.45
    return confidence >= 0.72 and stable_diff_count >= 2


def _ai_vision_check(vehicle1_img: Image.Image, vehicle2_img: Image.Image) -> Dict[str, Any]:
    try:
        if not hasattr(vehicle1_img, "save") or not hasattr(vehicle2_img, "save"):
            return _normalize_ai_decision(None)

        temp_dir = tempfile.mkdtemp(prefix="ai_check_")
        vehicle1_path = os.path.join(temp_dir, "vehicle1.jpg")
        vehicle2_path = os.path.join(temp_dir, "vehicle2.jpg")

        vehicle1_img.save(vehicle1_path, "JPEG", quality=85)
        vehicle2_img.save(vehicle2_path, "JPEG", quality=85)
        result = _normalize_ai_decision(_run_ai_check_with_timeout(vehicle1_path, vehicle2_path))

        shutil.rmtree(temp_dir, ignore_errors=True)
        return result
    except Exception as e:
        try:
            if "temp_dir" in locals():
                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass

        print(f"AI vision check failed: {e}")
        fallback = _normalize_ai_decision(None)
        fallback["interference_factors"] = ["ai_check_failed"]
        fallback["reason"] = str(e)
        return fallback


def _ai_vision_check_from_paths(vehicle1_path: str, vehicle2_path: str) -> str:
    try:
        if not os.path.exists(vehicle1_path) or not os.path.exists(vehicle2_path):
            return "unknown"

        ai_info = _normalize_ai_decision(_run_ai_check_with_timeout(vehicle1_path, vehicle2_path))
        return str(ai_info.get("decision") or "unknown")
    except Exception as e:
        print(f"AI vision check failed: {e}")
        return "unknown"


def _classify_case_with_ai(head_prob: Optional[float], tail_prob: Optional[float],
                          vehicle1_img: Optional[Image.Image] = None,
                          vehicle2_img: Optional[Image.Image] = None) -> str:
    if head_prob is None or tail_prob is None:
        return "abnormal"

    head_low_th = _HEAD_THRESHOLD
    head_high_th = float(os.environ.get("HEAD_HIGH_TH", "0.8"))
    tail_low_th = _TAIL_THRESHOLD

    if head_prob < head_low_th:
        first_result = "fake_plate"
    elif head_prob >= head_high_th and tail_prob <= tail_low_th:
        first_result = "change_trailer"
    else:
        first_result = "normal"

    if first_result == "normal":
        return "normal"
    if first_result == "fake_plate" and head_prob < 0.3:
        return "fake_plate"
    if first_result == "change_trailer" and tail_prob < 0.3:
        return "change_trailer"

    if vehicle1_img is not None and vehicle2_img is not None:
        ai_info = _ai_vision_check(vehicle1_img, vehicle2_img)
        ai_decision = str(ai_info.get("decision") or "unknown")
        accepted = _should_accept_ai_result(first_result, ai_info)
        final_result = ai_decision if accepted else first_result
        print(
            f"绗竴娆″垽鏂? {first_result}, "
            f"AI鍒ゆ柇: {ai_info.get('raw_decision', ai_decision)}, "
            f"缃俊搴? {ai_info.get('confidence', 0.0):.2f}, "
            f"鐏厜骞叉壈: {ai_info.get('lighting_interference', False)}, "
            f"鏈€缁堢粨鏋? {final_result}, "
            f"AI閲囩敤: {accepted}"
        )
        return final_result

    return first_result


def _classify_case(head_prob: Optional[float], tail_prob: Optional[float]) -> str:
    if head_prob is None or tail_prob is None:
        return "abnormal"

    # 浣跨敤鍏ㄥ眬闃堝€煎彉閲?
    head_low_th = _HEAD_THRESHOLD
    head_high_th = float(os.environ.get("HEAD_HIGH_TH", "0.8"))  # 杞﹀ご楂樼浉浼煎害闃堝€?
    tail_low_th = _TAIL_THRESHOLD

    if head_prob < head_low_th:
        return "fake_plate"
    if head_prob >= head_high_th and tail_prob <= tail_low_th:
        return "change_trailer"
    return "normal"


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
    
    if _is_http_url(path1_input) or _is_http_url(path2_input):
        source = "http"

    t_validate0 = time.perf_counter()
    ok1, p1 = _validate_image_path(payload.get("path1"))
    ok2, p2 = _validate_image_path(payload.get("path2"))
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
        )
        return jsonify({"ok": False, "error": f"path2 invalid: {p2}"}), 400

    # 涓轰簡淇濆瓨鍥剧墖锛岄渶瑕佺敓鎴愰瑙堝浘
    t_open_ms = 0.0
    previews = None
    original_images = None
    try:
        t_open0 = time.perf_counter()
        img1 = Image.open(p1)
        img2 = Image.open(p2)
        t_open_ms = (time.perf_counter() - t_open0) * 1000.0
        
        # 鐢熸垚棰勮鍥惧拰鍘熷鍥撅紙鐢ㄤ簬淇濆瓨锛?
        with _REQUEST_LOCK:
            t_preview0 = time.perf_counter()
            head_prob, tail_prob, previews, original_images, vehicle1_pil, vehicle2_pil, err = _compute_probs_and_previews_pil(img1, img2)
            t_preview_ms = (time.perf_counter() - t_preview0) * 1000.0
        case_type = predictor.classify_with_ai(head_prob, tail_prob,
                                               vehicle1_img=vehicle1_pil,
                                               vehicle2_img=vehicle2_pil)
        
        # 璁＄畻鑰楁椂
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
        )
        return jsonify({"ok": False, "error": f"processing failed: {e}"}), 500

    resp: Dict[str, Any] = {
        "ok": case_type != "abnormal",
        "case_type": case_type,
        "head_prob": head_prob,
        "tail_prob": tail_prob,
    }
    if err:
        resp["error"] = err
    lat_ms = (time.perf_counter() - t0) * 1000.0
    
    # 淇濆瓨鍥剧墖骞惰褰?
    record_id = _record_metric(
        endpoint="/predict",
        source=source,
        http_status=200,
        ok=case_type != "abnormal",
        case_type=case_type,
        head_prob=head_prob,
        tail_prob=tail_prob,
        lat_ms=lat_ms,
        stage_ms={"validate": t_validate_ms, "open": t_open_ms, "compute": t_compute_ms},
        error=str(err or ""),
        previews=previews,
        original_images=original_images,
        input_path1=path1_input,
        input_path2=path2_input,
    )
    
    if record_id:
        resp["record_id"] = record_id
    
    return jsonify(resp)


@app.post("/predict_preview")
def predict_preview() -> Any:
    t0 = time.perf_counter()
    predictor = VehiclePairPredictor()
    payload = request.get_json(silent=True) or {}
    source = "path"
    path1_input = str(payload.get("path1") or "")
    path2_input = str(payload.get("path2") or "")
    
    if _is_http_url(path1_input) or _is_http_url(path2_input):
        source = "http"

    t_validate0 = time.perf_counter()
    ok1, p1 = _validate_image_path(payload.get("path1"))
    ok2, p2 = _validate_image_path(payload.get("path2"))
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
        )
        return jsonify({"ok": False, "error": f"path2 invalid: {p2}"}), 400

    t_open_ms = 0.0
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
        )
        return jsonify({"ok": False, "error": f"failed to open images: {e}"}), 400

    with _REQUEST_LOCK:
        t_compute0 = time.perf_counter()
        head_prob, tail_prob, previews, original_images, vehicle1_pil, vehicle2_pil, err = _compute_probs_and_previews_pil(img1, img2)
        t_compute_ms = (time.perf_counter() - t_compute0) * 1000.0
    case_type = predictor.classify_with_ai(head_prob, tail_prob,
                                           vehicle1_img=vehicle1_pil,
                                           vehicle2_img=vehicle2_pil)

    resp: Dict[str, Any] = {
        "ok": case_type != "abnormal",
        "case_type": case_type,
        "head_prob": head_prob,
        "tail_prob": tail_prob,
        "previews": previews or {},
    }
    if err:
        resp["error"] = err
    lat_ms = (time.perf_counter() - t0) * 1000.0
    
    # 淇濆瓨鍥剧墖骞惰褰?
    record_id = _record_metric(
        endpoint="/predict_preview",
        source=source,
        http_status=200,
        ok=case_type != "abnormal",
        case_type=case_type,
        head_prob=head_prob,
        tail_prob=tail_prob,
        lat_ms=lat_ms,
        stage_ms={"validate": t_validate_ms, "open": t_open_ms, "compute": t_compute_ms},
        error=str(err or ""),
        previews=previews,
        original_images=original_images,
        input_path1=path1_input,
        input_path2=path2_input,
    )
    
    if record_id:
        resp["record_id"] = record_id
    
    return jsonify(resp)


@app.post("/predict_upload_preview")
def predict_upload_preview() -> Any:
    t0 = time.perf_counter()
    predictor = VehiclePairPredictor()
    f1 = request.files.get("file1")
    f2 = request.files.get("file2")
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
        )
        return jsonify({"ok": False, "error": "file2 missing"}), 400

    t_open_ms = 0.0
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
        )
        return jsonify({"ok": False, "error": f"failed to open images: {e}"}), 400

    with _REQUEST_LOCK:
        t_compute0 = time.perf_counter()
        head_prob, tail_prob, previews, original_images, vehicle1_pil, vehicle2_pil, err = _compute_probs_and_previews_pil(img1, img2)
        t_compute_ms = (time.perf_counter() - t_compute0) * 1000.0
    case_type = predictor.classify_with_ai(head_prob, tail_prob,
                                           vehicle1_img=vehicle1_pil,
                                           vehicle2_img=vehicle2_pil)

    resp: Dict[str, Any] = {
        "ok": case_type != "abnormal",
        "case_type": case_type,
        "head_prob": head_prob,
        "tail_prob": tail_prob,
        "previews": previews or {},
    }
    if err:
        resp["error"] = err
    lat_ms = (time.perf_counter() - t0) * 1000.0
    
    # 淇濆瓨鍥剧墖骞惰褰?
    file1_name = f1.filename if f1 else "unknown"
    file2_name = f2.filename if f2 else "unknown"
    
    record_id = _record_metric(
        endpoint="/predict_upload_preview",
        source="upload",
        http_status=200,
        ok=case_type != "abnormal",
        case_type=case_type,
        head_prob=head_prob,
        tail_prob=tail_prob,
        lat_ms=lat_ms,
        stage_ms={"open": t_open_ms, "compute": t_compute_ms},
        error=str(err or ""),
        previews=previews,
        original_images=original_images,
        input_path1=file1_name,
        input_path2=file2_name,
    )
    
    if record_id:
        resp["record_id"] = record_id
    
    return jsonify(resp)


@app.post("/predict_upload")
def predict_upload() -> Any:
    t0 = time.perf_counter()
    predictor = VehiclePairPredictor()
    f1 = request.files.get("file1")
    f2 = request.files.get("file2")
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
        )
        return jsonify({"ok": False, "error": "file2 missing"}), 400

    t_open_ms = 0.0
    previews = None
    original_images = None
    try:
        t_open0 = time.perf_counter()
        img1 = Image.open(f1.stream)
        img2 = Image.open(f2.stream)
        t_open_ms = (time.perf_counter() - t_open0) * 1000.0
        
        # 鐢熸垚棰勮鍥惧拰鍘熷鍥撅紙鐢ㄤ簬淇濆瓨锛?
        with _REQUEST_LOCK:
            t_preview0 = time.perf_counter()
            head_prob, tail_prob, previews, original_images, vehicle1_pil, vehicle2_pil, err = _compute_probs_and_previews_pil(img1, img2)
            t_preview_ms = (time.perf_counter() - t_preview0) * 1000.0
        case_type = predictor.classify_with_ai(head_prob, tail_prob,
                                               vehicle1_img=vehicle1_pil,
                                               vehicle2_img=vehicle2_pil)
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
        )
        return jsonify({"ok": False, "error": f"failed to open images: {e}"}), 400

    t_compute_ms = (time.perf_counter() - t_open0) * 1000.0
    resp: Dict[str, Any] = {
        "ok": case_type != "abnormal",
        "case_type": case_type,
        "head_prob": head_prob,
        "tail_prob": tail_prob,
    }
    if err:
        resp["error"] = err
    lat_ms = (time.perf_counter() - t0) * 1000.0
    
    # 淇濆瓨鍥剧墖骞惰褰?
    file1_name = f1.filename if f1 else "unknown"
    file2_name = f2.filename if f2 else "unknown"
    
    record_id = _record_metric(
        endpoint="/predict_upload",
        source="upload",
        http_status=200,
        ok=case_type != "abnormal",
        case_type=case_type,
        head_prob=head_prob,
        tail_prob=tail_prob,
        lat_ms=lat_ms,
        stage_ms={"open": t_open_ms, "compute": t_compute_ms},
        error=str(err or ""),
        previews=previews,
        original_images=original_images,
        input_path1=file1_name,
        input_path2=file2_name,
    )
    
    if record_id:
        resp["record_id"] = record_id
    
    return jsonify(resp)


@app.get("/records")
def records_page() -> Any:
    """璁板綍鏌ヨ椤甸潰"""
    return render_template("records.html")


@app.get("/api/records")
def api_query_records() -> Any:
    """鏌ヨ璁板綍鍒楄〃API"""
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
    """鑾峰彇鍗曟潯璁板綍璇︽儏API"""
    try:
        record = _METRICS.get_record(record_id)
        if not record:
            return jsonify({"error": "璁板綍涓嶅瓨鍦?}), 404
        
        return jsonify(record)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/api/record/<record_id>/image/<image_name>")
def api_get_image(record_id: str, image_name: str) -> Any:
    """鑾峰彇璁板綍鐨勫浘鐗?""
    try:
        # 楠岃瘉鍥剧墖鍚嶇О
        valid_names = ["vehicle1.jpg", "vehicle2.jpg", "head1.jpg", "head2.jpg", "tail1.jpg", "tail2.jpg"]
        if image_name not in valid_names:
            return jsonify({"error": "鏃犳晥鐨勫浘鐗囧悕绉?}), 400
        
        # 鑾峰彇璁板綍
        record = _METRICS.get_record(record_id)
        if not record:
            return jsonify({"error": "璁板綍涓嶅瓨鍦?}), 404
        
        # 鑾峰彇鍥剧墖璺緞
        image_dir = record.get("image_dir", "")
        if not image_dir or not os.path.exists(image_dir):
            return jsonify({"error": "鍥剧墖鐩綍涓嶅瓨鍦?}), 404
        
        image_path = os.path.join(image_dir, image_name)
        if not os.path.exists(image_path):
            return jsonify({"error": "鍥剧墖涓嶅瓨鍦?}), 404
        
        return send_file(image_path, mimetype="image/jpeg")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.delete("/api/record/<record_id>")
def api_delete_record(record_id: str) -> Any:
    """鍒犻櫎璁板綍API"""
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
    """璁剧疆璁板綍淇濇姢鐘舵€丄PI"""
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
    """鎵归噺鍒犻櫎璁板綍API"""
    try:
        payload = request.get_json(silent=True) or {}
        record_ids = payload.get("record_ids", [])
        hard_delete = payload.get("hard_delete", False)
        
        if not isinstance(record_ids, list):
            return jsonify({"ok": False, "error": "record_ids 蹇呴』鏄暟缁?}), 400
        
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
    """瀵煎嚭鍗曟潯璁板綍API"""
    try:
        payload = request.get_json(silent=True) or {}
        export_path = payload.get("export_path")
        image_types = payload.get("image_types")  # 鍙€夌殑鍥剧墖绫诲瀷鍒楄〃
        
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
    """鎵归噺瀵煎嚭璁板綍API"""
    try:
        payload = request.get_json(silent=True) or {}
        record_ids = payload.get("record_ids", [])
        export_path = payload.get("export_path")
        group_by = payload.get("group_by", "case_type")
        image_types = payload.get("image_types")
        include_summary = payload.get("include_summary", True)
        
        if not isinstance(record_ids, list):
            return jsonify({"ok": False, "error": "record_ids 蹇呴』鏄暟缁?}), 400
        
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
    """鑾峰彇鍙敤鐨勫浘鐗囩被鍨嬪垪琛?""
    return jsonify({
        "image_types": [
            {"value": "original1", "label": "鍘熷浘1", "group": "鍘熷鍥剧墖"},
            {"value": "original2", "label": "鍘熷浘2", "group": "鍘熷鍥剧墖"},
            {"value": "vehicle1", "label": "杞﹁締1锛堣鍒囷級", "group": "瑁佸垏鍥剧墖"},
            {"value": "vehicle2", "label": "杞﹁締2锛堣鍒囷級", "group": "瑁佸垏鍥剧墖"},
            {"value": "head1", "label": "杞﹀ご1", "group": "閮ㄤ欢鍥剧墖"},
            {"value": "head2", "label": "杞﹀ご2", "group": "閮ㄤ欢鍥剧墖"},
            {"value": "tail1", "label": "杞﹀熬1", "group": "閮ㄤ欢鍥剧墖"},
            {"value": "tail2", "label": "杞﹀熬2", "group": "閮ㄤ欢鍥剧墖"}
        ],
        "presets": {
            "all": ["original1", "original2", "vehicle1", "vehicle2", "head1", "head2", "tail1", "tail2"],
            "original_only": ["original1", "original2"],
            "processed_only": ["vehicle1", "vehicle2", "head1", "head2", "tail1", "tail2"],
            "head_only": ["head1", "head2"],
            "tail_only": ["tail1", "tail2"],
            "parts_only": ["head1", "head2", "tail1", "tail2"]
        }
    })


@app.post("/api/record/<record_id>/review")
def api_review_record(record_id: str) -> Any:
    """鎻愪氦澶嶆牳缁撴灉API"""
    try:
        payload = request.get_json(silent=True) or {}
        reviewed_case_type = payload.get("reviewed_case_type", "")
        review_reason = payload.get("review_reason", "")
        reviewed_by = payload.get("reviewed_by", "")
        review_confidence = payload.get("review_confidence", "medium")
        
        if not reviewed_case_type:
            return jsonify({"ok": False, "error": "澶嶆牳绫诲瀷涓嶈兘涓虹┖"}), 400
        
        if not reviewed_by:
            return jsonify({"ok": False, "error": "澶嶆牳浜哄憳涓嶈兘涓虹┖"}), 400
        
        success, message = _METRICS.review_record(
            record_id, reviewed_case_type, review_reason, reviewed_by, review_confidence
        )
        
        if success:
            # 杩斿洖鏇存柊鍚庣殑璁板綍
            record = _METRICS.get_record(record_id)
            return jsonify({"ok": True, "message": message, "record": record})
        else:
            return jsonify({"ok": False, "error": message}), 400
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.delete("/api/record/<record_id>/review")
def api_revoke_review(record_id: str) -> Any:
    """鎾ら攢澶嶆牳API"""
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
    """鑾峰彇澶嶆牳缁熻API"""
    try:
        stats = _METRICS.get_review_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/review_stats")
def review_stats_page() -> Any:
    """澶嶆牳缁熻椤甸潰"""
    return render_template("review_stats.html")


@app.get("/thresholds")
def get_thresholds() -> Any:
    """鑾峰彇褰撳墠闃堝€艰缃?""
    return jsonify({
        "head_threshold": _HEAD_THRESHOLD,
        "tail_threshold": _TAIL_THRESHOLD
    })


@app.post("/thresholds")
def set_thresholds() -> Any:
    """璁剧疆闃堝€?""
    try:
        payload = request.get_json(silent=True) or {}
        head_threshold = payload.get("head_threshold")
        tail_threshold = payload.get("tail_threshold")
        
        global _HEAD_THRESHOLD, _TAIL_THRESHOLD
        
        if head_threshold is not None:
            try:
                _HEAD_THRESHOLD = float(head_threshold)
                if not (0.0 <= _HEAD_THRESHOLD <= 1.0):
                    return jsonify({"error": "杞﹀ご闃堝€煎繀椤诲湪0.0-1.0涔嬮棿"}), 400
            except (ValueError, TypeError):
                return jsonify({"error": "杞﹀ご闃堝€兼牸寮忛敊璇?}), 400
        
        if tail_threshold is not None:
            try:
                _TAIL_THRESHOLD = float(tail_threshold)
                if not (0.0 <= _TAIL_THRESHOLD <= 1.0):
                    return jsonify({"error": "杞﹀熬闃堝€煎繀椤诲湪0.0-1.0涔嬮棿"}), 400
            except (ValueError, TypeError):
                return jsonify({"error": "杞﹀熬闃堝€兼牸寮忛敊璇?}), 400
        
        return jsonify({
            "message": "闃堝€艰缃垚鍔?,
            "head_threshold": _HEAD_THRESHOLD,
            "tail_threshold": _TAIL_THRESHOLD
        })
    except Exception as e:
        return jsonify({"error": f"璁剧疆澶辫触: {str(e)}"}), 500


@app.post("/api/recheck/selected")
def api_recheck_selected() -> Any:
    """澶嶆閫変腑鐨勮褰?""
    try:
        payload = request.get_json(silent=True) or {}
        record_ids = payload.get("record_ids", [])
        
        if not isinstance(record_ids, list) or len(record_ids) == 0:
            return jsonify({
                "ok": False,
                "error": "璇锋彁渚涜澶嶆鐨勮褰旾D鍒楄〃"
            }), 400
        
        with _RECHECK_LOCK:
            if _RECHECK_STATUS["running"]:
                return jsonify({
                    "ok": False,
                    "error": "澶嶆浠诲姟姝ｅ湪杩愯涓?
                }), 400
        
        # 楠岃瘉璁板綍鏄惁瀛樺湪
        valid_records = []
        for record_id in record_ids:
            # 浠庤褰旾D涓彁鍙栨棩鏈?
            date_part = record_id.split("_")[0]
            record_dir = os.path.join(_METRICS._images_dir, date_part, record_id)
            
            if os.path.exists(record_dir):
                vehicle1_path = os.path.join(record_dir, "vehicle1.jpg")
                vehicle2_path = os.path.join(record_dir, "vehicle2.jpg")
                meta_path = os.path.join(record_dir, "meta.json")
                
                if all(os.path.exists(p) for p in [vehicle1_path, vehicle2_path, meta_path]):
                    # 璇诲彇鍏冩暟鎹?
                    try:
                        with open(meta_path, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                        
                        valid_records.append({
                            "record_id": record_id,
                            "case_type": meta.get("case_type", ""),
                            "vehicle1_path": vehicle1_path,
                            "vehicle2_path": vehicle2_path,
                            "meta_path": meta_path,
                            "meta": meta
                        })
                    except Exception:
                        continue
        
        if not valid_records:
            return jsonify({
                "ok": False,
                "error": "娌℃湁鎵惧埌鏈夋晥鐨勮褰?
            }), 400
        
        # 鍚姩鍚庡彴绾跨▼鎵ц閫変腑璁板綍澶嶆
        thread = threading.Thread(
            target=_recheck_selected_records, 
            args=(valid_records,), 
            daemon=True
        )
        thread.start()
        
        return jsonify({
            "ok": True,
            "message": f"宸插惎鍔?{len(valid_records)} 鏉¤褰曠殑澶嶆浠诲姟"
        })
        
    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500


def _recheck_selected_records(records: list) -> None:
    """澶嶆閫変腑鐨勮褰曪紙鍚庡彴浠诲姟锛?""
    global _RECHECK_STATUS
    
    try:
        with _RECHECK_LOCK:
            _RECHECK_STATUS.update({
                "running": True,
                "started_at": datetime.datetime.now().isoformat(timespec="milliseconds"),
                "total": len(records),
                "processed": 0,
                "success": 0,
                "failed": 0,
                "current_record": None,
                "error": None,
                "results": []
            })
        
        # 澶勭悊閫変腑鐨勮褰?
        for i, record in enumerate(records):
            try:
                with _RECHECK_LOCK:
                    _RECHECK_STATUS["processed"] = i + 1
                    _RECHECK_STATUS["current_record"] = record["record_id"]
                
                # 鎵цAI澶嶆
                ai_result = _ai_vision_check_from_paths(
                    record["vehicle1_path"], 
                    record["vehicle2_path"]
                )
                
                # 鐩存帴浣跨敤鑻辨枃缁撴灉
                success = ai_result in ["fake_plate", "change_trailer", "normal"]
                
                # 鏇存柊璁板綍
                updated = _update_record_with_ai_result(
                    record["record_id"], 
                    ai_result,  # 浣跨敤鑻辨枃缁撴灉
                    success, 
                    "" if success else "AI unknown"
                )
                
                # 璁板綍缁撴灉
                result_info = {
                    "record_id": record["record_id"],
                    "original_case_type": record["case_type"],
                    "ai_result": ai_result,
                    "success": success,
                    "updated": updated
                }
                
                with _RECHECK_LOCK:
                    _RECHECK_STATUS["results"].append(result_info)
                    if success:
                        _RECHECK_STATUS["success"] += 1
                    else:
                        _RECHECK_STATUS["failed"] += 1
                
                print(f"澶嶆璁板綍 {record['record_id']}: {record['case_type']} -> {ai_result} ({'鎴愬姛' if success else '澶辫触'})")
                
            except Exception as e:
                print(f"澶嶆璁板綍 {record['record_id']} 澶辫触: {e}")
                
                with _RECHECK_LOCK:
                    _RECHECK_STATUS["failed"] += 1
                    _RECHECK_STATUS["results"].append({
                        "record_id": record["record_id"],
                        "original_case_type": record["case_type"],
                        "ai_result": None,
                        "success": False,
                        "updated": False,
                        "error": str(e)
                    })
        
        # 瀹屾垚澶嶆
        with _RECHECK_LOCK:
            _RECHECK_STATUS["running"] = False
            _RECHECK_STATUS["current_record"] = None
        
        print(f"閫変腑璁板綍澶嶆瀹屾垚: 鎬昏 {_RECHECK_STATUS['total']}, 鎴愬姛 {_RECHECK_STATUS['success']}, 澶辫触 {_RECHECK_STATUS['failed']}")
        
    except Exception as e:
        with _RECHECK_LOCK:
            _RECHECK_STATUS["running"] = False
            _RECHECK_STATUS["error"] = str(e)
        print(f"閫変腑璁板綍澶嶆澶辫触: {e}")


@app.post("/api/recheck/start")
def api_start_recheck() -> Any:
    """鍚姩鎵归噺澶嶆"""
    try:
        with _RECHECK_LOCK:
            if _RECHECK_STATUS["running"]:
                return jsonify({
                    "ok": False,
                    "error": "澶嶆浠诲姟姝ｅ湪杩愯涓?
                }), 400
        
        # 鍚姩鍚庡彴绾跨▼鎵ц澶嶆
        thread = threading.Thread(target=_batch_recheck_abnormal_records, daemon=True)
        thread.start()
        
        return jsonify({
            "ok": True,
            "message": "鎵归噺澶嶆浠诲姟宸插惎鍔?
        })
        
    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500


@app.get("/api/recheck/status")
def api_recheck_status() -> Any:
    """鑾峰彇澶嶆鐘舵€?""
    try:
        with _RECHECK_LOCK:
            status = _RECHECK_STATUS.copy()
        
        # 璁＄畻杩涘害鐧惧垎姣?
        if status["total"] > 0:
            status["progress_percent"] = round((status["processed"] / status["total"]) * 100, 2)
        else:
            status["progress_percent"] = 0
        
        return jsonify(status)
        
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


@app.get("/api/recheck/results")
def api_recheck_results() -> Any:
    """鑾峰彇澶嶆缁撴灉缁熻"""
    try:
        with _RECHECK_LOCK:
            results = _RECHECK_STATUS.get("results", [])
        
        # 缁熻缁撴灉
        stats = {
            "total": len(results),
            "success_count": 0,
            "failed_count": 0,
            "corrected_count": 0,
            "details": []
        }
        
        for result in results:
            if result["success"]:
                stats["success_count"] += 1
                
                # 妫€鏌ユ槸鍚︾籂姝ｄ簡鍘熺粨鏋?
                if result["ai_result"] and result["original_case_type"] != result["ai_result"]:
                    stats["corrected_count"] += 1
            else:
                stats["failed_count"] += 1
            
            stats["details"].append(result)
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


@app.get("/api/recheck/failed")
def api_recheck_failed() -> Any:
    """鑾峰彇澶嶆澶辫触鐨勮褰曪紝鐢ㄤ簬浜哄伐澶勭悊"""
    try:
        # 鏌ヨ澶嶆澶辫触鐨勮褰?
        result = _METRICS.query_records(
            start_date=None,
            end_date=None,
            case_type="all",
            include_deleted=False,
            limit=100,
            offset=0
        )
        
        failed_records = []
        for record in result.get("records", []):
            ai_recheck = record.get("ai_recheck", {})
            if ai_recheck.get("attempted", False) and not ai_recheck.get("success", False):
                failed_records.append(record)
        
        return jsonify({
            "records": failed_records,
            "total": len(failed_records)
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "records": [],
            "total": 0
        }), 500


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8001"))
    app.run(host=host, port=port, threaded=True)
