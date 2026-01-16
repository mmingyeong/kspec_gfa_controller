# tests/test_gfa_environment.py
import json
from pathlib import Path

import pytest

import kspec_gfa_controller.gfa_environment as gfa_environment


# -------------------------
# Fakes
# -------------------------
class FakeLogger:
    def __init__(self):
        self.logs = []

    def info(self, m):
        self.logs.append(("info", str(m)))

    def debug(self, m):
        self.logs.append(("debug", str(m)))

    def warning(self, m):
        self.logs.append(("warning", str(m)))

    def error(self, m):
        self.logs.append(("error", str(m)))


class FakeController:
    """
    GFAEnvironment에서 사용하는 최소 API만 구현
    shutdown에서 close_camera만 사용함.
    """

    def __init__(self, config_path, logger):
        self.config_path = config_path
        self.logger = logger
        self.close_camera_calls = []

    def close_camera(self, camnum: int):
        self.close_camera_calls.append(camnum)


class FakeAstrometry:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger


class FakeGuider:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger


# -------------------------
# Helpers
# -------------------------
def _write_env_cams_json(path: Path):
    cfg = {
        "GfaController": {
            "Elements": {
                "Cameras": {
                    "Elements": {
                        "Cam1": {"IpAddress": "1.1.1.1", "Number": 1},
                        "Cam2": {"IpAddress": "2.2.2.2", "Number": 2},
                        "Cam3": {"IpAddress": "3.3.3.3", "Number": 3},
                        "Cam4": {"IpAddress": "4.4.4.4", "Number": 4},
                        "Cam5": {"IpAddress": "5.5.5.5", "Number": 5},
                        "Cam6": {"IpAddress": "6.6.6.6", "Number": 6},
                        "Cam7": {"IpAddress": "7.7.7.7", "Number": 7},
                        "CamX": {"IpAddress": "9.9.9.9"},  # Number 없음 -> skip
                        "Cam8": {"IpAddress": "8.8.8.8", "Number": 8},  # 범위 밖 -> 제외
                    }
                }
            }
        }
    }
    path.write_text(json.dumps(cfg), encoding="utf-8")


# -------------------------
# get_config_path()
# -------------------------
def test_get_config_path_success(tmp_path, monkeypatch):
    fake_logger = FakeLogger()
    monkeypatch.setattr(gfa_environment, "logger", fake_logger)

    # base_dir을 tmp_path로 유도하기 위해 __file__을 tmp 경로로 바꾼다
    fake_module_file = tmp_path / "gfa_environment.py"
    fake_module_file.write_text("# dummy", encoding="utf-8")
    monkeypatch.setattr(gfa_environment, "__file__", str(fake_module_file))

    # base_dir/relative_path로 만들어질 "실제 파일"을 준비
    rel = "whatever.json"
    target = tmp_path / rel
    target.write_text("{}", encoding="utf-8")

    out = gfa_environment.get_config_path(rel)
    assert out == str(target)


def test_get_config_path_missing_raises(tmp_path, monkeypatch):
    fake_logger = FakeLogger()
    monkeypatch.setattr(gfa_environment, "logger", fake_logger)

    fake_module_file = tmp_path / "gfa_environment.py"
    fake_module_file.write_text("# dummy", encoding="utf-8")
    monkeypatch.setattr(gfa_environment, "__file__", str(fake_module_file))

    rel = "etc/cams.json"  # 없도록 둔다

    with pytest.raises(FileNotFoundError):
        gfa_environment.get_config_path(rel)

    assert any(lvl == "error" for (lvl, msg) in fake_logger.logs)


# -------------------------
# get_camera_ids()
# -------------------------
def test_get_camera_ids_plate_and_finder(tmp_path, monkeypatch):
    fake_logger = FakeLogger()
    monkeypatch.setattr(gfa_environment, "logger", fake_logger)

    cfgp = tmp_path / "cams.json"
    _write_env_cams_json(cfgp)

    plate_ids = gfa_environment.get_camera_ids(str(cfgp), role="plate")
    finder_ids = gfa_environment.get_camera_ids(str(cfgp), role="finder")

    assert plate_ids == [1, 2, 3, 4, 5, 6]
    assert finder_ids == [7]


# -------------------------
# GFAEnvironment: plate
# -------------------------
def test_environment_plate_initializes_components(tmp_path, monkeypatch):
    fake_logger = FakeLogger()
    monkeypatch.setattr(gfa_environment, "logger", fake_logger)

    monkeypatch.setattr(gfa_environment, "GFAController", FakeController)
    monkeypatch.setattr(gfa_environment, "GFAAstrometry", FakeAstrometry)
    monkeypatch.setattr(gfa_environment, "GFAGuider", FakeGuider)

    cfgp = tmp_path / "cams.json"
    _write_env_cams_json(cfgp)
    astp = tmp_path / "ast.json"
    astp.write_text("{}", encoding="utf-8")

    env = gfa_environment.GFAEnvironment(
        gfa_config_path=str(cfgp),
        ast_config_path=str(astp),
        role="plate",
    )

    assert env.role == "plate"
    assert env.camera_ids == [1, 2, 3, 4, 5, 6]
    assert isinstance(env.controller, FakeController)
    assert isinstance(env.astrometry, FakeAstrometry)
    assert isinstance(env.guider, FakeGuider)


def test_environment_plate_shutdown_closes_each_camera(tmp_path, monkeypatch):
    fake_logger = FakeLogger()
    monkeypatch.setattr(gfa_environment, "logger", fake_logger)

    monkeypatch.setattr(gfa_environment, "GFAController", FakeController)
    monkeypatch.setattr(gfa_environment, "GFAAstrometry", FakeAstrometry)
    monkeypatch.setattr(gfa_environment, "GFAGuider", FakeGuider)

    cfgp = tmp_path / "cams.json"
    _write_env_cams_json(cfgp)
    astp = tmp_path / "ast.json"
    astp.write_text("{}", encoding="utf-8")

    env = gfa_environment.GFAEnvironment(
        gfa_config_path=str(cfgp),
        ast_config_path=str(astp),
        role="plate",
    )
    env.shutdown()

    assert env.controller.close_camera_calls == [1, 2, 3, 4, 5, 6]


# -------------------------
# GFAEnvironment: finder
# -------------------------
def test_environment_finder_initializes_components(tmp_path, monkeypatch):
    fake_logger = FakeLogger()
    monkeypatch.setattr(gfa_environment, "logger", fake_logger)

    monkeypatch.setattr(gfa_environment, "GFAController", FakeController)

    cfgp = tmp_path / "cams.json"
    _write_env_cams_json(cfgp)

    env = gfa_environment.GFAEnvironment(
        gfa_config_path=str(cfgp),
        ast_config_path=None,
        role="finder",
    )

    assert env.role == "finder"
    assert env.camera_ids == [7]
    assert isinstance(env.controller, FakeController)
    assert env.astrometry is None
    assert env.guider is None


def test_environment_finder_shutdown_closes_cam7(tmp_path, monkeypatch):
    fake_logger = FakeLogger()
    monkeypatch.setattr(gfa_environment, "logger", fake_logger)

    monkeypatch.setattr(gfa_environment, "GFAController", FakeController)

    cfgp = tmp_path / "cams.json"
    _write_env_cams_json(cfgp)

    env = gfa_environment.GFAEnvironment(
        gfa_config_path=str(cfgp),
        ast_config_path=None,
        role="finder",
    )
    env.shutdown()

    assert env.controller.close_camera_calls == [7]


# -------------------------
# create_environment()
# -------------------------
def test_create_environment_plate_and_finder(tmp_path, monkeypatch):
    fake_logger = FakeLogger()
    monkeypatch.setattr(gfa_environment, "logger", fake_logger)

    cams = tmp_path / "cams.json"
    _write_env_cams_json(cams)
    ast = tmp_path / "astrometry_params.json"
    ast.write_text("{}", encoding="utf-8")

    def fake_get_config_path(rel):
        if rel == "etc/cams.json":
            return str(cams)
        if rel == "etc/astrometry_params.json":
            return str(ast)
        raise AssertionError(f"Unexpected rel: {rel}")

    monkeypatch.setattr(gfa_environment, "get_config_path", fake_get_config_path)

    monkeypatch.setattr(gfa_environment, "GFAController", FakeController)
    monkeypatch.setattr(gfa_environment, "GFAAstrometry", FakeAstrometry)
    monkeypatch.setattr(gfa_environment, "GFAGuider", FakeGuider)

    env_plate = gfa_environment.create_environment(role="plate")
    assert env_plate.role == "plate"
    assert env_plate.astrometry is not None
    assert env_plate.guider is not None
    assert env_plate.camera_ids == [1, 2, 3, 4, 5, 6]

    env_finder = gfa_environment.create_environment(role="finder")
    assert env_finder.role == "finder"
    assert env_finder.astrometry is None
    assert env_finder.guider is None
    assert env_finder.camera_ids == [7]
