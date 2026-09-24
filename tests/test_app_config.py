"""
Tests for project/app.py's entrypoint hardening:
  - debug mode and host default safely (off / localhost) and are env-overridable
  - a processing failure never leaks raw exception text to the client
"""

import importlib
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "project"))


def _reload_app(monkeypatch, **env):
    for key in ("FLASK_DEBUG", "FLASK_HOST", "FLASK_PORT"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    import app as app_module
    importlib.reload(app_module)
    return app_module


def test_debug_defaults_to_false(monkeypatch):
    app_module = _reload_app(monkeypatch)
    assert app_module.DEBUG is False


def test_host_defaults_to_localhost(monkeypatch):
    app_module = _reload_app(monkeypatch)
    assert app_module.HOST == "127.0.0.1"


def test_port_defaults_to_5000(monkeypatch):
    app_module = _reload_app(monkeypatch)
    assert app_module.PORT == 5000


def test_debug_and_host_are_env_overridable(monkeypatch):
    app_module = _reload_app(monkeypatch, FLASK_DEBUG="true", FLASK_HOST="0.0.0.0", FLASK_PORT="8080")
    assert app_module.DEBUG is True
    assert app_module.HOST == "0.0.0.0"
    assert app_module.PORT == 8080


def test_process_500_does_not_leak_exception_text(monkeypatch):
    app_module = _reload_app(monkeypatch)
    client = app_module.app.test_client()

    def boom(*args, **kwargs):
        raise RuntimeError("/etc/super/secret/internal-path leaked here")

    monkeypatch.setattr(app_module, "process_image", boom)

    data = {"image": (io.BytesIO(b"not a real image but has the right extension"), "test.jpg")}
    resp = client.post("/process", data=data, content_type="multipart/form-data")

    assert resp.status_code == 500
    assert b"/etc/super/secret" not in resp.data
    assert b"RuntimeError" not in resp.data
