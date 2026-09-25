"""Session-bound review endpoints for the local app."""
import hmac
import json
from pathlib import Path
import secrets

from flask import Blueprint, Response, abort, jsonify, render_template, request, send_file, session

from core.jobs import JobManager
from core.profiles import load_profile
from core.review import LIMITATIONS, preview_frame
from core.selection import validate_choices, validate_manual, match_face_selection


def register_review(app, base_dir: str, allowed_images: set, allowed_videos: set) -> None:
    manager = JobManager(Path(base_dir) / "jobs", str(Path(base_dir) / "models"))
    app.extensions["review_jobs"] = manager
    api = Blueprint("review", __name__)

    @api.before_request
    def guard():
        session.setdefault("owner", secrets.token_hex(24))
        session.setdefault("csrf", secrets.token_hex(24))
        if request.method in {"POST", "DELETE"}:
            if not hmac.compare_digest(request.headers.get("X-CSRF-Token", ""), session["csrf"]):
                abort(403)

    @api.after_request
    def private_response(response):
        response.headers["Cache-Control"] = "no-store, private"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    def get_job(key):
        job = manager.get(key, session["owner"])
        if job is None:
            abort(404)
        return job

    @api.get("/review")
    def page():
        return render_template("review.html", csrf=session["csrf"])

    @api.post("/api/jobs")
    def create():
        upload = request.files.get("file")
        if not upload or not upload.filename or "." not in upload.filename:
            return jsonify(error="Choose an image or video"), 400
        ext = upload.filename.rsplit(".", 1)[1].lower()
        if ext not in allowed_images | allowed_videos:
            return jsonify(error="Unsupported image or video type"), 400
        try:
            profile = load_profile(request.form.get("profile", "creator"), request.form.get("custom"))
            if request.form.get("faces_only") == "true":
                profile["classes"] = ["faces"]
            job = manager.create(session["owner"], upload, ext, ext in allowed_videos, profile)
            return jsonify(job.status()), 202
        except (ValueError, TypeError):
            return jsonify(error="Invalid profile or job limit reached. Check the profile and delete unused jobs."), 400

    @api.get("/api/jobs/<key>")
    def status(key):
        return jsonify(get_job(key).status())

    @api.get("/api/jobs/<key>/analysis")
    def analysis(key):
        job = get_job(key)
        with job.lock:
            if job.state != "ready":
                return jsonify(error="Analysis is not ready"), 409
            data = {k: v for k, v in job.analysis.items() if k != "frames"}
            return jsonify(data | {"profile": job.profile, "limitations": LIMITATIONS})

    @api.get("/api/jobs/<key>/frames/<int:index>")
    def frame(key, index):
        job = get_job(key)
        with job.lock:
            if job.state != "ready" or not 0 <= index < job.analysis["frame_count"]:
                abort(404)
            if request.args.get("boxes") == "1":
                return jsonify(job.analysis["frames"][index])
            try:
                return Response(preview_frame(str(job.source), job.analysis, index), mimetype="image/jpeg")
            except ValueError:
                return jsonify(error="Could not decode preview frame"), 422

    @api.get("/api/jobs/<key>/source")
    def source(key):
        job = get_job(key)
        with job.lock:
            if job.state != "ready" or not job.video:
                abort(404)
            # Stream the existing upload with HTTP range support, no new original copy.
            mime = "video/webm" if job.source.suffix == ".webm" else "video/mp4"
            return send_file(job.source, mimetype=mime, conditional=True)

    @api.post("/api/jobs/<key>/select-face")
    def select_face(key):
        job = get_job(key)
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error="Expected a face box and frame"), 400
        with job.lock:
            if job.state != "ready":
                return jsonify(error="Job is not ready"), 409
            try:
                index = payload.get("frame")
                region = validate_manual([{"box": payload.get("box"), "start": index, "end": index}],
                                         job.analysis["width"], job.analysis["height"], job.analysis["frame_count"])[0]
                track_id = match_face_selection(region["box"], job.analysis["frames"][index])
                positions = [{"frame": i, "box": record.get("raw_box") or record["box"]}
                             for i, records in enumerate(job.analysis["frames"])
                             for record in records if record["id"] == track_id]
                return jsonify(track=job.analysis["tracks"][track_id], positions=positions)
            except (ValueError, TypeError) as exc:
                return jsonify(error=str(exc) if isinstance(exc, ValueError) else "Invalid face box"), 400

    @api.post("/api/jobs/<key>/export")
    def export(key):
        job = get_job(key)
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error="Expected export choices"), 400
        with job.lock:
            if job.state != "ready":
                return jsonify(error="Job is not ready"), 409
            try:
                data = job.analysis
                choices = validate_choices(payload.get("choices", {}), data["tracks"], data["frame_count"])
                manual = validate_manual(payload.get("manual", []), data["width"], data["height"], data["frame_count"])
                audio = payload.get("audio", job.profile["audio"])
                if audio not in {"keep", "mute"}:
                    raise ValueError("Audio must be keep or mute")
                manager.start_export(job, choices, manual, audio)
                return jsonify(job.status()), 202
            except (ValueError, TypeError) as exc:
                return jsonify(error=str(exc) if isinstance(exc, ValueError) else "Invalid export choices"), 400

    @api.delete("/api/jobs/<key>")
    def delete(key):
        manager.delete(get_job(key))
        return jsonify(state="cancelling"), 202

    @api.get("/api/jobs/<key>/download")
    def download(key):
        job = get_job(key)
        with job.lock:
            if job.state != "complete" or not job.output:
                abort(404)
            return send_file(job.output, as_attachment=True, download_name=job.output.name)

    @api.get("/api/jobs/<key>/report")
    def report(key):
        job = get_job(key)
        with job.lock:
            if job.state != "complete" or job.report is None:
                abort(404)
            if request.args.get("format") == "html":
                return render_template("report.html", report=job.report)
            return Response(json.dumps(job.report, indent=2), mimetype="application/json",
                            headers={"Content-Disposition": 'attachment; filename="redaction-report.json"'})

    app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Strict")
    app.register_blueprint(api)
