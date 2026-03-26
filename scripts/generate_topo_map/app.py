import json
import os
import threading
import uuid
from typing import Dict, List

import numpy as np

from flask import Flask, jsonify, render_template, request

from inference_service import apply_filter, run_inference, load_session_meta
from row_generation import generate_rows
from topo_export import build_topological_yaml

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
IMAGE_ROOT = os.path.join(BASE_DIR, "images", "riseholme")
CACHE_ROOT = os.path.join(os.path.dirname(__file__), "cache")

MODEL_ROOTS = [
    os.path.join(BASE_DIR, "weights"),
    os.path.join(BASE_DIR, "scripts", "gaussian_heatmap_resnet", "gaussian_heatmap_hybrid", "results_hybrid"),
]

os.makedirs(CACHE_ROOT, exist_ok=True)

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "templates"))

JOBS = {}
JOBS_LOCK = threading.Lock()


def _create_job(job_type: str) -> str:
    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "type": job_type,
            "status": "running",
            "progress": 0,
            "message": "Starting",
            "result": None,
            "error": None,
        }
    return job_id


def _update_job(job_id: str, **kwargs) -> None:
    with JOBS_LOCK:
        if job_id not in JOBS:
            return
        if "result" in kwargs:
            kwargs["result"] = _normalize_for_json(kwargs["result"])
        JOBS[job_id].update(kwargs)


def _normalize_for_json(value):
    if isinstance(value, dict):
        return {key: _normalize_for_json(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _get_job(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        return dict(job) if job else None


def _list_image_folders() -> List[str]:
    folders = []
    if not os.path.isdir(IMAGE_ROOT):
        return folders

    for dirpath, _dirnames, filenames in os.walk(IMAGE_ROOT):
        if any(fname.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")) for fname in filenames):
            rel = os.path.relpath(dirpath, IMAGE_ROOT)
            if rel != ".":
                folders.append(rel)
    folders.sort()
    return folders


def _list_models() -> List[Dict[str, str]]:
    models = []
    for root in MODEL_ROOTS:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            for fname in filenames:
                if fname.lower().endswith((".pth", ".pt")):
                    abs_path = os.path.join(dirpath, fname)
                    label = os.path.relpath(abs_path, BASE_DIR)
                    models.append({"label": label, "path": abs_path})
    models.sort(key=lambda item: item["label"])
    return models


def _list_cached_sessions() -> List[Dict[str, str]]:
    sessions = []
    if not os.path.isdir(CACHE_ROOT):
        return sessions

    for filename in os.listdir(CACHE_ROOT):
        if not filename.endswith(".json"):
            continue
        session_id = filename[:-5]
        meta_path = os.path.join(CACHE_ROOT, filename)
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue
        created_at = meta.get("created_at", "")
        images = meta.get("images", [])
        label = f"{session_id} ({created_at})" if created_at else session_id
        sessions.append({
            "id": session_id,
            "label": label,
            "images": str(len(images)),
        })

    sessions.sort(key=lambda item: item["label"], reverse=True)
    return sessions


@app.route("/")
def index():
    return render_template(
        "index.html",
        mapbox_token=os.environ.get("MAPBOX_TOKEN"),
    )


@app.route("/api/folders", methods=["GET"])
def list_folders():
    return jsonify({"folders": _list_image_folders()})


@app.route("/api/models", methods=["GET"])
def list_models():
    return jsonify({"models": _list_models()})


@app.route("/api/sessions", methods=["GET"])
def list_sessions():
    return jsonify({"sessions": _list_cached_sessions()})


@app.route("/api/job/<job_id>", methods=["GET"])
def api_job_status(job_id: str):
    job = _get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/api/run_inference", methods=["POST"])
def api_run_inference():
    payload = request.get_json(force=True)
    folder = payload.get("folder")
    model_path = payload.get("model_path")
    image_size = payload.get("image_size", [1280, 960])
    confidence = float(payload.get("confidence", 0.4))
    cluster_radius = float(payload.get("cluster_radius", 1.5))
    cluster_algo = payload.get("cluster_algo", "dbscan")
    backbone = payload.get("backbone", "resnet101")

    if not folder:
        return jsonify({"error": "Folder is required"}), 400
    if not model_path:
        return jsonify({"error": "Model path is required"}), 400

    input_dir = os.path.join(IMAGE_ROOT, folder)
    job_id = _create_job("inference")

    def progress_cb(done, total, message):
        # Stage progress: Inference 0-80%, Filtering 80-90%, Clustering 90-98%, Complete 100%
        if "Filtered poles" in message or "Filtering poles" in message:
            percent = 85
        elif "Clustering poles" in message:
            percent = 92
        elif "Filtering complete" in message:
            percent = 98
        elif message and message.startswith("Processed"):
            percent = int((done / total) * 75) if total else 75
        else:
            percent = int((done / total) * 75) if total else 75
        _update_job(job_id, progress=percent, message=message)

    def worker():
        try:
            _update_job(job_id, progress=0, message="Running inference")
            session_id, poles_geojson, stats, vine_rows_geojson = run_inference(
                input_dir=input_dir,
                model_path=model_path,
                cache_root=CACHE_ROOT,
                image_size=(int(image_size[0]), int(image_size[1])),
                backbone=backbone,
                confidence_threshold=confidence,
                cluster_eps_m=cluster_radius,
                cluster_algo=cluster_algo,
                progress_callback=progress_cb,
            )
            _update_job(
                job_id,
                status="done",
                progress=100,
                message="Inference complete",
                result={"session_id": session_id, "poles": poles_geojson, "stats": stats, "vine_rows": vine_rows_geojson},
            )
        except Exception as exc:
            _update_job(job_id, status="error", error=str(exc), message="Inference failed")

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/apply_filter", methods=["POST"])
def api_apply_filter():
    payload = request.get_json(force=True)
    session_id = payload.get("session_id")
    rows_geojson = payload.get("rows")
    confidence = float(payload.get("confidence", 0.4))
    cluster_radius = float(payload.get("cluster_radius", 1.5))
    cluster_algo = payload.get("cluster_algo", "dbscan")
    filter_vine_rows = bool(payload.get("filter_vine_rows", True))

    if not session_id:
        return jsonify({"error": "Session id is required"}), 400

    job_id = _create_job("filter")

    def progress_cb(done, total, message):
        percent = int((done / total) * 100) if total else 100
        _update_job(job_id, progress=percent, message=message)

    def worker():
        try:
            _update_job(job_id, progress=0, message="Applying filter")
            result = apply_filter(
                cache_root=CACHE_ROOT,
                session_id=session_id,
                confidence_threshold=confidence,
                cluster_eps_m=cluster_radius,
                cluster_algo=cluster_algo,
                rows_geojson=rows_geojson,
                filter_by_vine_rows=filter_vine_rows,
                progress_callback=progress_cb,
            )
            _update_job(
                job_id,
                status="done",
                progress=100,
                message="Filter complete",
                result=result,
            )
        except Exception as exc:
            _update_job(job_id, status="error", error=str(exc), message="Filter failed")

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/generate_rows", methods=["POST"])
def api_generate_rows():
    payload = request.get_json(force=True)
    poles_geojson = payload.get("poles")
    session_id = payload.get("session_id")
    vine_rows_geojson = None
    
    if not poles_geojson:
        return jsonify({"error": "Poles GeoJSON is required"}), 400
    
    if session_id:
        try:
            meta = load_session_meta(CACHE_ROOT, session_id)
            vine_rows_geojson = meta.get("vine_rows_geojson")
        except Exception:
            pass

    job_id = _create_job("rows")

    def worker():
        try:
            _update_job(job_id, progress=10, message="Grouping poles")
            rows_geojson = generate_rows(poles_geojson, vine_rows_geojson=vine_rows_geojson)
            _update_job(
                job_id,
                status="done",
                progress=100,
                message="Rows generated",
                result={"rows": rows_geojson},
            )
        except Exception as exc:
            _update_job(job_id, status="error", error=str(exc), message="Row generation failed")

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/export_topo", methods=["POST"])
def api_export_topo():
    payload = request.get_json(force=True)
    poles_geojson = payload.get("poles")
    rows_geojson = payload.get("rows")
    map_name = payload.get("map_name", "vineyard")
    metric_map = payload.get("metric_map", "vineyard_metric")
    node_spacing_m = payload.get("node_spacing_m", 2.0)
    extend_distance_m = payload.get("extend_distance_m", 3.0)
    cross_row_distance_m = payload.get("cross_row_distance_m", 6.0)

    if not poles_geojson:
        return jsonify({"error": "Poles GeoJSON is required"}), 400
    if not rows_geojson:
        return jsonify({"error": "Rows GeoJSON is required"}), 400

    job_id = _create_job("export")

    def worker():
        try:
            _update_job(job_id, progress=20, message="Building topo map")
            topo_yaml, datum_yaml = build_topological_yaml(
                poles_geojson=poles_geojson,
                rows_geojson=rows_geojson,
                map_name=map_name,
                metric_map=metric_map,
                node_spacing_m=node_spacing_m,
                extend_distance_m=extend_distance_m,
                cross_row_distance_m=cross_row_distance_m,
            )
            _update_job(
                job_id,
                status="done",
                progress=100,
                message="Export ready",
                result={
                    "topo_yaml": topo_yaml,
                    "datum_yaml": datum_yaml,
                    "topo_filename": f"{map_name}_topological_map.tmap2.yaml",
                    "datum_filename": f"{map_name}_datum.yaml",
                },
            )
        except Exception as exc:
            _update_job(job_id, status="error", error=str(exc), message="Export failed")

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"job_id": job_id})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005, debug=True, threaded=True)
