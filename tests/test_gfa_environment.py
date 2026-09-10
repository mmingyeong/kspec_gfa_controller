# tests/test_gfa_environment.py
import json
import sys
import types
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


UNDER_TEST_FULLNAME = "kspec_gfa_controller.gfa_environment__under_test"


class FakeLogger:
    def __init__(self, *args, **kwargs):
        self.logs = []

    def info(self, message):
        self.logs.append(("info", str(message)))

    def debug(self, message):
        self.logs.append(("debug", str(message)))

    def warning(self, message):
        self.logs.append(("warning", str(message)))

    def error(self, message):
        self.logs.append(("error", str(message)))


class FakeController:
    def __init__(self, config_path, logger):
        self.config_path = config_path
        self.logger = logger
        self.close_camera_calls = []

    def close_camera(self, camnum: int):
        self.close_camera_calls.append(camnum)


class FakeAstrometry:
    def __init__(self, config, logger, save_root=None):
        self.config = config
        self.logger = logger
        self.save_root = save_root


class FakeGuider:
    def __init__(self, config, logger, save_root=None):
        self.config = config
        self.logger = logger
        self.save_root = save_root


def _find_gfa_environment_py() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        repo_root / "src" / "kspec_gfa_controller" / "gfa_environment.py",
        repo_root / "kspec_gfa_controller" / "gfa_environment.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise RuntimeError(
        "gfa_environment.py not found. tried:\n"
        + "\n".join(str(candidate) for candidate in candidates)
    )


@pytest.fixture
def env_module(monkeypatch):
    source_path = _find_gfa_environment_py()
    repo_root = Path(__file__).resolve().parents[1]
    package_name = "kspec_gfa_controller"

    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [
            str(repo_root / "src" / package_name),
            str(repo_root / package_name),
        ]
        monkeypatch.setitem(sys.modules, package_name, package)

    dependencies = {
        "gfa_controller": ("GFAController", FakeController),
        "gfa_logger": ("GFALogger", FakeLogger),
        "gfa_astrometry": ("GFAAstrometry", FakeAstrometry),
        "gfa_guider": ("GFAGuider", FakeGuider),
    }
    for module_name, (attribute, value) in dependencies.items():
        fake_module = types.ModuleType(f"{package_name}.{module_name}")
        setattr(fake_module, attribute, value)
        monkeypatch.setitem(
            sys.modules,
            f"{package_name}.{module_name}",
            fake_module,
        )

    spec = spec_from_file_location(UNDER_TEST_FULLNAME, str(source_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for: {source_path}")

    module = module_from_spec(spec)
    monkeypatch.setitem(sys.modules, UNDER_TEST_FULLNAME, module)
    spec.loader.exec_module(module)
    return module


def _write_env_cams_json(path: Path):
    config = {
        "GfaController": {
            "Elements": {
                "Cameras": {
                    "Elements": {
                        "Cam6": {"Number": 6},
                        "Cam2": {"Number": 2},
                        "Cam1": {"Number": 1},
                        "Cam3": {"Number": 3},
                        "Cam4": {"Number": 4},
                        "Cam5": {"Number": 5},
                        "Cam7": {"Number": 7},
                        "CamX": {},
                        "Cam8": {"Number": 8},
                    }
                }
            }
        }
    }
    path.write_text(json.dumps(config), encoding="utf-8")


def test_get_config_path_success(tmp_path, monkeypatch, env_module):
    fake_module_file = tmp_path / "gfa_environment.py"
    fake_module_file.write_text("# dummy", encoding="utf-8")
    monkeypatch.setattr(env_module, "__file__", str(fake_module_file))
    target = tmp_path / "cams.json"
    target.write_text("{}", encoding="utf-8")

    assert env_module.get_config_path("cams.json") == str(target)


def test_get_config_path_missing_logs_and_raises(tmp_path, monkeypatch, env_module):
    fake_module_file = tmp_path / "gfa_environment.py"
    fake_module_file.write_text("# dummy", encoding="utf-8")
    monkeypatch.setattr(env_module, "__file__", str(fake_module_file))

    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        env_module.get_config_path("missing.json")

    assert any(
        level == "error" and "configuration file not found" in message.lower()
        for level, message in env_module.logger.logs
    )


@pytest.mark.parametrize(
    ("role", "expected"),
    [
        ("plate", [1, 2, 3, 4, 5, 6]),
        ("finder", [7]),
        ("unsupported", []),
    ],
)
def test_get_camera_ids_filters_and_sorts(tmp_path, env_module, role, expected):
    config_path = tmp_path / "cams.json"
    _write_env_cams_json(config_path)

    assert env_module.get_camera_ids(str(config_path), role=role) == expected


def test_environment_plate_initializes_all_components(tmp_path, env_module):
    config_path = tmp_path / "cams.json"
    astrometry_path = tmp_path / "astrometry.json"
    save_root = tmp_path / "data"
    _write_env_cams_json(config_path)
    astrometry_path.write_text("{}", encoding="utf-8")

    environment = env_module.GFAEnvironment(
        gfa_config_path=str(config_path),
        ast_config_path=str(astrometry_path),
        role="plate",
        save_root=str(save_root),
    )

    assert environment.save_root == save_root.resolve()
    assert environment.camera_ids == [1, 2, 3, 4, 5, 6]
    assert isinstance(environment.controller, FakeController)
    assert isinstance(environment.astrometry, FakeAstrometry)
    assert isinstance(environment.guider, FakeGuider)
    assert environment.astrometry.save_root == save_root.resolve()
    assert environment.guider.save_root == save_root.resolve()


def test_environment_finder_initializes_controller_only(tmp_path, env_module):
    config_path = tmp_path / "cams.json"
    _write_env_cams_json(config_path)

    environment = env_module.GFAEnvironment(
        gfa_config_path=str(config_path),
        ast_config_path=None,
        role="finder",
        save_root=str(tmp_path / "finder_data"),
    )

    assert environment.camera_ids == [7]
    assert isinstance(environment.controller, FakeController)
    assert environment.astrometry is None
    assert environment.guider is None


def test_environment_uses_default_save_root(tmp_path, monkeypatch, env_module):
    config_path = tmp_path / "cams.json"
    default_root = tmp_path / "default_data"
    _write_env_cams_json(config_path)
    monkeypatch.setattr(env_module, "DEFAULT_SAVE_ROOT", default_root)

    environment = env_module.GFAEnvironment(
        gfa_config_path=str(config_path),
        ast_config_path=None,
        role="finder",
    )

    assert environment.save_root == default_root.resolve()
    assert default_root.is_dir()


@pytest.mark.parametrize(
    ("role", "expected_closed"),
    [
        ("plate", [1, 2, 3, 4, 5, 6]),
        ("finder", [7]),
    ],
)
def test_environment_shutdown_closes_expected_cameras(
    tmp_path, env_module, role, expected_closed
):
    config_path = tmp_path / "cams.json"
    _write_env_cams_json(config_path)
    environment = env_module.GFAEnvironment(
        gfa_config_path=str(config_path),
        ast_config_path=str(tmp_path / "astrometry.json") if role == "plate" else None,
        role=role,
        save_root=str(tmp_path / f"{role}_data"),
    )

    environment.shutdown()

    assert environment.controller.close_camera_calls == expected_closed


@pytest.mark.parametrize(
    ("role", "expected_config_calls", "expected_astrometry_path"),
    [
        (
            "plate",
            ["etc/cams.json", "etc/astrometry_params.json"],
            "/resolved/astrometry_params.json",
        ),
        ("finder", ["etc/cams.json"], None),
    ],
)
def test_create_environment_resolves_paths_and_forwards_arguments(
    tmp_path,
    monkeypatch,
    env_module,
    role,
    expected_config_calls,
    expected_astrometry_path,
):
    config_calls = []
    constructor_calls = []
    sentinel = object()

    def fake_get_config_path(relative_path):
        config_calls.append(relative_path)
        return f"/resolved/{Path(relative_path).name}"

    def fake_environment(gfa_path, ast_path, *, role, save_root):
        constructor_calls.append((gfa_path, ast_path, role, save_root))
        return sentinel

    monkeypatch.setattr(env_module, "get_config_path", fake_get_config_path)
    monkeypatch.setattr(env_module, "GFAEnvironment", fake_environment)
    save_root = str(tmp_path / "requested_root")

    result = env_module.create_environment(role=role, save_root=save_root)

    assert result is sentinel
    assert config_calls == expected_config_calls
    assert constructor_calls == [
        (
            "/resolved/cams.json",
            expected_astrometry_path,
            role,
            save_root,
        )
    ]
