# tests/test_gfa_environment.py
import json
from pathlib import Path

import pytest
import gfa_environment


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
    gfa_environment.GFAEnvironment 에서 사용하는 최소 API만 구현
    """

    def __init__(self, config_path, logger):
        self.config_path = config_path
        self.logger = logger
        self.open_selected_calls = []
        self.open_camera_calls = []
        self.close_camera_calls = []

    def open_selected_cameras(self, camera_ids):
        self.open_selected_calls.append(list(camera_ids))

    def open_camera(self, camnum: int):
        self.open_camera_calls.append(camnum)

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
    """
    get_camera_ids가 읽을 cams.json 구조를 만들어줌.
    - plate: Number 1~6
    - finder: Number 7
    - Number 없는 카메라 / 범위 밖 카메라도 섞어서 필터링 동작 확인
    """
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
                        "Cam8": {
                            "IpAddress": "8.8.8.8",
                            "Number": 8,
                        },  # 범위 밖 -> plate/finder 둘 다 제외
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
    # logger는 모듈 전역이므로 테스트용으로 교체
    fake_logger = FakeLogger()
    monkeypatch.setattr(gfa_environment, "logger", fake_logger)

    f = tmp_path / "a.json"
    f.write_text("{}", encoding="utf-8")

    # relative_path는 base_dir 기준으로 join되는데,
    # 여기서는 함수가 "존재하면 반환"만 보면 되므로,
    # os.path.abspath(__file__) 기반 실제 base_dir을 우회하기 위해
    # os.path.join 결과를 "실제 존재하는 파일"로 유도하는 방식은 번거로움.
    # 대신 get_config_path 자체는 "isfile(full_path)"가 True이면 full_path 반환이므로
    # full_path 계산을 고정시키기 위해 os.path.abspath/dirname/join을 건드리기보다,
    # os.path.isfile과 os.path.join을 단순 패치해 테스트.
    monkeypatch.setattr(gfa_environment.os.path, "join", lambda a, b: str(f))
    monkeypatch.setattr(gfa_environment.os.path, "isfile", lambda p: True)

    out = gfa_environment.get_config_path("whatever.json")
    assert out == str(f)


def test_get_config_path_missing_raises(tmp_path, monkeypatch):
    fake_logger = FakeLogger()
    monkeypatch.setattr(gfa_environment, "logger", fake_logger)

    monkeypatch.setattr(
        gfa_environment.os.path, "join", lambda a, b: str(tmp_path / "nope.json")
    )
    monkeypatch.setattr(gfa_environment.os.path, "isfile", lambda p: False)

    with pytest.raises(FileNotFoundError):
        gfa_environment.get_config_path("etc/cams.json")

    # 에러 로그도 남는지(선택)
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
def test_environment_plate_initializes_and_opens_selected(tmp_path, monkeypatch):
    fake_logger = FakeLogger()
    monkeypatch.setattr(gfa_environment, "logger", fake_logger)

    # 의존 클래스 전부 fake로
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
    assert env.controller.open_selected_calls == [[1, 2, 3, 4, 5, 6]]
    assert isinstance(env.astrometry, FakeAstrometry)
    assert isinstance(env.guider, FakeGuider)


def test_environment_plate_missing_open_selected_method_raises(tmp_path, monkeypatch):
    fake_logger = FakeLogger()
    monkeypatch.setattr(gfa_environment, "logger", fake_logger)

    class BadController(FakeController):
        # open_selected_cameras 제거(또는 아예 없는 클래스로)
        def __getattribute__(self, name):
            if name == "open_selected_cameras":
                raise AttributeError("nope")
            return super().__getattribute__(name)

    monkeypatch.setattr(gfa_environment, "GFAController", BadController)
    monkeypatch.setattr(gfa_environment, "GFAAstrometry", FakeAstrometry)
    monkeypatch.setattr(gfa_environment, "GFAGuider", FakeGuider)

    cfgp = tmp_path / "cams.json"
    _write_env_cams_json(cfgp)
    astp = tmp_path / "ast.json"
    astp.write_text("{}", encoding="utf-8")

    with pytest.raises(AttributeError):
        gfa_environment.GFAEnvironment(
            gfa_config_path=str(cfgp),
            ast_config_path=str(astp),
            role="plate",
        )


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

    # plate: camera_ids 순회 close_camera(cam_id)
    assert env.controller.close_camera_calls == [1, 2, 3, 4, 5, 6]


# -------------------------
# GFAEnvironment: finder
# -------------------------
def test_environment_finder_initializes_and_opens_cam7(tmp_path, monkeypatch):
    fake_logger = FakeLogger()
    monkeypatch.setattr(gfa_environment, "logger", fake_logger)

    monkeypatch.setattr(gfa_environment, "GFAController", FakeController)
    # finder에서는 astrometry/guider None 이므로 patch 필수는 아니지만 안전하게
    monkeypatch.setattr(gfa_environment, "GFAAstrometry", FakeAstrometry)
    monkeypatch.setattr(gfa_environment, "GFAGuider", FakeGuider)

    cfgp = tmp_path / "cams.json"
    _write_env_cams_json(cfgp)

    env = gfa_environment.GFAEnvironment(
        gfa_config_path=str(cfgp),
        ast_config_path=None,
        role="finder",
    )

    assert env.role == "finder"
    assert env.camera_ids == [7]
    assert env.controller.open_camera_calls == [7]
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

    # create_environment는 내부에서 get_config_path를 호출하므로 그 결과를 우리가 만든 파일로 돌려주기
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

    # 의존 클래스 fake
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