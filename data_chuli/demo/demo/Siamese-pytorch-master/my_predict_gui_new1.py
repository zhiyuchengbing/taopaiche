import os
import sys
import threading
import urllib.parse
import base64
import io
from typing import Optional, Tuple, Dict, Any

import numpy as np
import cv2
from PIL import Image
from flask import Flask, jsonify, request, render_template
from ultralytics import YOLO

from siamese import Siamese
from data_tran.image_resolver import ImagePathResolver

parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from data_chuli.cropper import VehicleCropper


app = Flask(__name__)

_INIT_LOCK = threading.Lock()
_INFER_LOCK = threading.Lock()
_INITIALIZED = False

_CROPPER: Optional[VehicleCropper] = None
_HEAD_MODEL: Optional[Siamese] = None
_TAIL_MODEL: Optional[Siamese] = None
_HEADTAIL_MODEL: Optional[YOLO] = None
_IMAGE_RESOLVER: Optional[ImagePathResolver] = None


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
            r"D:\project\data_chuli\demo\demo\Siamese-pytorch-master\logs\weibu\1211\best_epoch_weights.pth",
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

        with _INFER_LOCK:
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
) -> Tuple[Optional[float], Optional[float], Optional[Dict[str, str]], Optional[str]]:
    try:
        _init_models()
        if _CROPPER is None or _HEAD_MODEL is None or _TAIL_MODEL is None:
            return None, None, None, "models not initialized"

        v1 = _CROPPER.process_pil(img1)
        v2 = _CROPPER.process_pil(img2)

        h1 = _crop_part_from_vehicle_pil(v1, cls_id=0)
        h2 = _crop_part_from_vehicle_pil(v2, cls_id=0)
        t1 = _crop_part_from_vehicle_pil(v1, cls_id=1)
        t2 = _crop_part_from_vehicle_pil(v2, cls_id=1)

        with _INFER_LOCK:
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
        return float(head_prob), float(tail_prob), previews, None
    except Exception as e:
        return None, None, None, str(e)


def _compute_head_tail_probs_pil(img1: Image.Image, img2: Image.Image) -> Tuple[Optional[float], Optional[float], Optional[str]]:
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

        with _INFER_LOCK:
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

    head_low_th = float(os.environ.get("HEAD_LOW_TH", "0.8"))
    head_same_th = float(os.environ.get("HEAD_SAME_TH", "0.3"))
    tail_low_th = float(os.environ.get("TAIL_LOW_TH", "0.3"))

    if head_prob < head_low_th:
        return "fake_plate"
    if head_prob > head_same_th and tail_prob <= tail_low_th:
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
        }
    })


@app.get("/ui")
def ui() -> Any:
    return render_template("ui.html")


@app.get("/health")
def health() -> Any:
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict() -> Any:
    predictor = VehiclePairPredictor()
    payload = request.get_json(silent=True) or {}
    ok1, p1 = _validate_image_path(payload.get("path1"))
    ok2, p2 = _validate_image_path(payload.get("path2"))
    if not ok1:
        return jsonify({"ok": False, "error": f"path1 invalid: {p1}"}), 400
    if not ok2:
        return jsonify({"ok": False, "error": f"path2 invalid: {p2}"}), 400

    head_prob, tail_prob, err = predictor.predict_from_paths(p1, p2)
    case_type = predictor.classify(head_prob, tail_prob)

    resp: Dict[str, Any] = {
        "ok": case_type != "abnormal",
        "case_type": case_type,
        "head_prob": head_prob,
        "tail_prob": tail_prob,
    }
    if err:
        resp["error"] = err
    return jsonify(resp)


@app.post("/predict_preview")
def predict_preview() -> Any:
    predictor = VehiclePairPredictor()
    payload = request.get_json(silent=True) or {}
    ok1, p1 = _validate_image_path(payload.get("path1"))
    ok2, p2 = _validate_image_path(payload.get("path2"))
    if not ok1:
        return jsonify({"ok": False, "error": f"path1 invalid: {p1}"}), 400
    if not ok2:
        return jsonify({"ok": False, "error": f"path2 invalid: {p2}"}), 400

    try:
        img1 = Image.open(p1)
        img2 = Image.open(p2)
    except Exception as e:
        return jsonify({"ok": False, "error": f"failed to open images: {e}"}), 400

    head_prob, tail_prob, previews, err = _compute_probs_and_previews_pil(img1, img2)
    case_type = predictor.classify(head_prob, tail_prob)

    resp: Dict[str, Any] = {
        "ok": case_type != "abnormal",
        "case_type": case_type,
        "head_prob": head_prob,
        "tail_prob": tail_prob,
        "previews": previews or {},
    }
    if err:
        resp["error"] = err
    return jsonify(resp)


@app.post("/predict_upload_preview")
def predict_upload_preview() -> Any:
    predictor = VehiclePairPredictor()
    f1 = request.files.get("file1")
    f2 = request.files.get("file2")
    if f1 is None:
        return jsonify({"ok": False, "error": "file1 missing"}), 400
    if f2 is None:
        return jsonify({"ok": False, "error": "file2 missing"}), 400

    try:
        img1 = Image.open(f1.stream)
        img2 = Image.open(f2.stream)
    except Exception as e:
        return jsonify({"ok": False, "error": f"failed to open images: {e}"}), 400

    head_prob, tail_prob, previews, err = _compute_probs_and_previews_pil(img1, img2)
    case_type = predictor.classify(head_prob, tail_prob)

    resp: Dict[str, Any] = {
        "ok": case_type != "abnormal",
        "case_type": case_type,
        "head_prob": head_prob,
        "tail_prob": tail_prob,
        "previews": previews or {},
    }
    if err:
        resp["error"] = err
    return jsonify(resp)


@app.post("/predict_upload")
def predict_upload() -> Any:
    predictor = VehiclePairPredictor()
    f1 = request.files.get("file1")
    f2 = request.files.get("file2")
    if f1 is None:
        return jsonify({"ok": False, "error": "file1 missing"}), 400
    if f2 is None:
        return jsonify({"ok": False, "error": "file2 missing"}), 400

    try:
        img1 = Image.open(f1.stream)
        img2 = Image.open(f2.stream)
    except Exception as e:
        return jsonify({"ok": False, "error": f"failed to open images: {e}"}), 400

    head_prob, tail_prob, err = predictor.predict_from_pil(img1, img2)
    case_type = predictor.classify(head_prob, tail_prob)

    resp: Dict[str, Any] = {
        "ok": case_type != "abnormal",
        "case_type": case_type,
        "head_prob": head_prob,
        "tail_prob": tail_prob,
    }
    if err:
        resp["error"] = err
    return jsonify(resp)


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8001"))
    app.run(host=host, port=port, threaded=True)
