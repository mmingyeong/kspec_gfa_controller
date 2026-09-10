# tests/test_gfa_actions.py
import os
import sys
import types
import importlib
from pathlib import Path

import pytest
import numpy as np


# -------------------------
# Minimal fakes
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

    def exception(self, m):
        self.logs.append(("exception", str(m)))


class FakeImage:
    def __init__(self):
        self.save_calls = []

    def save_fits(self, **kwargs):
        self.save_calls.append(kwargs)


class FakeController:
    def __init__(self, grabone_result=None):
        self._grabone_result = grabone_result
        self.grabone_calls = []
        self.grab_calls = []
        self.ping_calls = []
        self.status_called = 0
        self.cam_params_calls = []

        self.open_all_called = 0
        self.close_all_called = 0
        self.img_class = FakeImage()

    async def open_all_cameras(self):
        self.open_all_called += 1
        return None

    async def close_all_cameras(self):
        self.close_all_called += 1
        return None

    async def grabone(self, **kwargs):
        self.grabone_calls.append(kwargs)
        cam = kwargs["CamNum"]
        if callable(self._grabone_result):
            return self._grabone_result(cam)
        if isinstance(self._grabone_result, dict):
            return dict(self._grabone_result)
        timeout_cameras = set(self._grabone_result or [])
        return {
            "cam_num": cam,
            "timeout": cam in timeout_cameras,
            "serial": f"SERIAL{cam}",
            "image": [[cam]],
        }

    async def grab(self, CamNum, ExpTime, Binning, **kwargs):
        self.grab_calls.append((CamNum, ExpTime, Binning, kwargs))
        return []

    def status(self):
        self.status_called += 1
        return {"Cam1": True, "Cam2": False}

    def ping(self, cam_id):
        self.ping_calls.append(cam_id)

    def cam_params(self, cam_id):
        self.cam_params_calls.append(cam_id)
        return {"mock": cam_id}


class FakeAstrometry:
    """
    최신 gfa_actions.py 기준:
    - set_subprocess_env(clean_env) 호출됨
    - ensure_astrometry_ready() 있으면 그걸 사용
    - guiding() 마지막에 clear_raw_files() 호출
    """

    def __init__(self, ensure_outputs=None):
        self.subprocess_env_set = None
        self.ensure_called = 0
        self.preproc_called = 0
        self.clear_raw_called = 0
        self._ensure_outputs = (
            ensure_outputs
            if ensure_outputs is not None
            else [
                "/tmp/astro_1.fits",
                "/tmp/astro_2.fits",
            ]
        )
        self.final_astrometry_dir = "/tmp/astrodir"
        self.inpar = {
            "paths": {
                "save_root": None,
                "directories": {
                    "grab_images": "grab",
                    "raw_images": "raw",
                    "guiding_save": "guiding_save",
                    "pointing_save": "pointing_save",
                    "unclean_images": "unclean",
                },
            },
            "pointing_filter": {
                "min_valid_images": 1,
                "min_std_bg": 1.0,
                "min_peaks": 1,
                "min_brightest_flux": 1.0,
                "dao": {"fwhm": 3.0, "sigma_threshold": 5.0},
            },
        }

    def set_subprocess_env(self, env: dict):
        self.subprocess_env_set = env

    def ensure_astrometry_ready(self):
        self.ensure_called += 1
        return list(self._ensure_outputs)

    # fallback 경로용(혹시 ensure_astrometry_ready가 없을 때)
    def preproc(self):
        self.preproc_called += 1

    def clear_raw_files(self):
        self.clear_raw_called += 1


class FakeGuider:
    def __init__(self, fdx=1.0, fdy=2.0, fwhm=3.0):
        self._ret = (fdx, fdy, fwhm)
        self.exe_called = 0

    def exe_cal(self):
        self.exe_called += 1
        return self._ret


class FakeEnv:
    def __init__(
        self, camera_ids=(1, 2, 3), controller=None, astrometry=None, guider=None
    ):
        self.logger = FakeLogger()
        self.camera_ids = list(camera_ids)
        self.controller = controller if controller is not None else FakeController()
        self.astrometry = astrometry if astrometry is not None else FakeAstrometry()
        self.guider = guider if guider is not None else FakeGuider()
        self.shutdown_called = 0
        self.save_root = None

    def shutdown(self):
        self.shutdown_called += 1


# -------------------------
# Import helper (핵심)
# -------------------------
@pytest.fixture
def ga_module(monkeypatch):
    """
    gfa_actions import 시 SciPy로 내려가는 체인을 끊기 위해,
    gfa_environment / gfa_logger 를 sys.modules에 fake로 주입 후 import.
    """
    pkg = "kspec_gfa_controller"

    # fake gfa_logger
    m_logger = types.ModuleType(f"{pkg}.gfa_logger")

    class _FakeGFALogger:
        def __init__(self, *_a, **_k):
            pass

        def info(self, *_a, **_k):
            pass

        def debug(self, *_a, **_k):
            pass

        def warning(self, *_a, **_k):
            pass

        def error(self, *_a, **_k):
            pass

    m_logger.GFALogger = _FakeGFALogger

    # fake gfa_environment (SciPy 안 타게)
    m_env = types.ModuleType(f"{pkg}.gfa_environment")

    def _fake_create_environment(*, role, save_root=None):
        return FakeEnv()

    class _FakeGFAEnvironment:
        pass

    m_env.create_environment = _fake_create_environment
    m_env.GFAEnvironment = _FakeGFAEnvironment

    monkeypatch.setitem(sys.modules, f"{pkg}.gfa_logger", m_logger)
    monkeypatch.setitem(sys.modules, f"{pkg}.gfa_environment", m_env)

    # 이제 안전하게 import
    mod = importlib.import_module(f"{pkg}.gfa_actions")
    return mod


@pytest.fixture
def actions(ga_module, tmp_path, monkeypatch):
    env = FakeEnv()
    env.save_root = tmp_path
    action = ga_module.GFAActions(env=env)
    monkeypatch.setattr(action, "_debug_path_block", lambda *a, **k: None)
    return action


def prepare_successful_pipeline(actions, monkeypatch, passed_files=None):
    passed_files = passed_files or ["/tmp/raw/a.fits"]

    async def fake_grab(**kwargs):
        return {"status": "success", "message": "ok", "grab_files": passed_files}

    monkeypatch.setattr(actions, "grab", fake_grab)
    monkeypatch.setattr(
        actions,
        "_filter_pointing_raw_images",
        lambda **kwargs: {
            "passed_files": list(passed_files),
            "failed_files": [],
            "n_passed": len(passed_files),
            "n_failed": 0,
        },
    )


def patch_fits_crvals(monkeypatch, crval1=1.0, crval2=2.0):
    class FakeHDU:
        header = {"CRVAL1": crval1, "CRVAL2": crval2}

    class FakeHDUL(list):
        def __init__(self):
            super().__init__([FakeHDU()])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        "kspec_gfa_controller.gfa_actions.fits.open", lambda path: FakeHDUL()
    )


# -------------------------
# __init__: env None branch
# -------------------------
def test_init_env_none_uses_create_environment(monkeypatch, ga_module):
    calls = []

    def fake_create_environment(*, role, save_root=None):
        calls.append((role, save_root))
        return FakeEnv()

    monkeypatch.setattr(ga_module, "create_environment", fake_create_environment)
    act = ga_module.GFAActions(env=None)

    assert isinstance(act.env, FakeEnv)
    assert calls == [("plate", None)]


# -------------------------
# Basic unit: response shape
# -------------------------
def test_generate_response(actions):
    r = actions._generate_response("success", "ok", a=1, b="x")
    assert r["status"] == "success"
    assert r["message"] == "ok"
    assert r["a"] == 1
    assert r["b"] == "x"


# -------------------------
# grab(): CamNum=int (single)
# -------------------------
@pytest.mark.asyncio
async def test_grab_single_camera_success_message(actions, monkeypatch):
    actions.env.controller._grabone_result = []  # timeout 없음
    monkeypatch.setattr(
        "kspec_gfa_controller.gfa_actions.os.makedirs", lambda *a, **k: None
    )

    r = await actions.grab(
        CamNum=2,
        ExpTime=1.5,
        Binning=4,
        packet_size=1500,
        cam_ipd=10,
        cam_ftd_base=123,
        ra="1",
        dec="2",
    )
    assert r["status"] == "success"
    assert "cameras [2]" in r["message"].lower()

    assert actions.env.controller.open_all_called == 1
    assert actions.env.controller.close_all_called == 1

    assert len(actions.env.controller.grabone_calls) == 1
    kwargs = actions.env.controller.grabone_calls[0]
    assert kwargs["CamNum"] == 2
    assert kwargs["ExpTime"] == 1.5
    assert kwargs["Binning"] == 4
    assert kwargs["packet_size"] == 1500
    assert kwargs["ipd"] == 10
    assert kwargs["ftd_base"] == 123
    assert kwargs["ra"] == "1"
    assert kwargs["dec"] == "2"
    assert "output_dir" in kwargs


@pytest.mark.asyncio
async def test_grab_single_camera_timeout_in_message(actions, monkeypatch):
    actions.env.controller._grabone_result = [2]
    monkeypatch.setattr(
        "kspec_gfa_controller.gfa_actions.os.makedirs", lambda *a, **k: None
    )

    r = await actions.grab(CamNum=2)
    assert r["status"] == "success"
    assert "timeout" in r["message"].lower()

    assert actions.env.controller.open_all_called == 1
    assert actions.env.controller.close_all_called == 1


# -------------------------
# grab(): CamNum=0 (all)
# -------------------------
@pytest.mark.asyncio
async def test_grab_all_cameras_aggregates_timeouts(actions, monkeypatch):
    monkeypatch.setattr(
        "kspec_gfa_controller.gfa_actions.os.makedirs", lambda *a, **k: None
    )

    async def fake_grabone(**kwargs):
        cam = kwargs["CamNum"]
        return {
            "cam_num": cam,
            "timeout": cam in (1, 3),
            "serial": f"SERIAL{cam}",
            "image": [[cam]],
        }

    actions.env.controller.grabone = fake_grabone

    r = await actions.grab(CamNum=0)
    assert r["status"] == "success"
    assert "all cameras" in r["message"].lower()
    assert "timeout" in r["message"].lower()
    assert "1" in r["message"]
    assert "3" in r["message"]

    assert actions.env.controller.open_all_called == 1
    assert actions.env.controller.close_all_called == 1


# -------------------------
# grab(): CamNum=list
# -------------------------
@pytest.mark.asyncio
async def test_grab_camera_list(actions, monkeypatch):
    monkeypatch.setattr(
        "kspec_gfa_controller.gfa_actions.os.makedirs", lambda *a, **k: None
    )

    async def fake_grabone(**kwargs):
        cam = kwargs["CamNum"]
        return {
            "cam_num": cam,
            "timeout": cam == 5,
            "serial": f"SERIAL{cam}",
            "image": [[cam]],
        }

    actions.env.camera_ids = [1, 2, 3, 4, 5]
    actions.env.controller.grabone = fake_grabone

    r = await actions.grab(CamNum=[4, 5])
    assert r["status"] == "success"
    assert "cameras" in r["message"].lower()
    assert "timeout" in r["message"].lower()

    assert actions.env.controller.open_all_called == 1
    assert actions.env.controller.close_all_called == 1


@pytest.mark.asyncio
async def test_grab_invalid_camnum_returns_error(actions, monkeypatch):
    monkeypatch.setattr(
        "kspec_gfa_controller.gfa_actions.os.makedirs", lambda *a, **k: None
    )

    r = await actions.grab(CamNum="bad")  # type: ignore
    assert r["status"] == "error"
    assert "grab failed" in r["message"].lower()

    assert actions.env.controller.open_all_called == 1
    assert actions.env.controller.close_all_called == 1


# -------------------------
# guiding(): success path (save=False)
# -------------------------
@pytest.mark.asyncio
async def test_guiding_success_no_save(actions, monkeypatch):
    # 디렉토리/리스트 I/O 막기
    monkeypatch.setattr(
        "kspec_gfa_controller.gfa_actions.os.makedirs", lambda *a, **k: None
    )
    monkeypatch.setattr("kspec_gfa_controller.gfa_actions.os.listdir", lambda p: [])

    prepare_successful_pipeline(actions, monkeypatch)
    r = await actions.guiding(
        ExpTime=2.0, SaveGrabRaw=False, ra="1", dec="2"
    )
    assert r["status"] == "success"
    assert "Offsets:" in r["message"]
    assert "fdx" in r and "fdy" in r and "fwhm" in r

    # clean env가 astrometry로 세팅됨
    assert actions.env.astrometry.subprocess_env_set is not None

    # ensure_astrometry_ready 사용
    assert actions.env.astrometry.ensure_called == 1

    # guider 실행
    assert actions.env.guider.exe_called == 1

    # raw clear 호출 (신규 API)
    assert actions.env.astrometry.clear_raw_called == 2

    # 응답에 astrometry_files basename 리스트 포함
    assert "astrometry_files" in r
    assert r["astrometry_files"] == ["astro_1.fits", "astro_2.fits"]


@pytest.mark.asyncio
async def test_guiding_success_with_save_and_copy(actions, monkeypatch):
    monkeypatch.setattr(
        "kspec_gfa_controller.gfa_actions.os.makedirs", lambda *a, **k: None
    )

    prepare_successful_pipeline(actions, monkeypatch, ["/tmp/raw/a.fits"])

    copy_calls = []

    def fake_copy2(src, dst):
        copy_calls.append((src, dst))

    monkeypatch.setattr("kspec_gfa_controller.gfa_actions.shutil.copy2", fake_copy2)

    r = await actions.guiding(
        ExpTime=1.5, SaveGrabRaw=True, ra="3", dec="4"
    )
    assert r["status"] == "success"

    # a.fits만 복사됨
    assert len(copy_calls) == 1
    src, dst = copy_calls[0]
    src_norm = os.path.normpath(src)
    dst_norm = os.path.normpath(dst)

    assert src_norm.endswith(os.path.normpath(os.path.join("raw", "a.fits")))
    assert os.path.normpath("guiding_save") in dst_norm
    assert dst_norm.endswith(os.path.normpath("a.fits"))


@pytest.mark.asyncio
async def test_guiding_fwhm_nonfloat_becomes_zero(actions, monkeypatch):
    actions.env.guider = FakeGuider(fdx=1.0, fdy=2.0, fwhm="bad")  # type: ignore
    monkeypatch.setattr(
        "kspec_gfa_controller.gfa_actions.os.makedirs", lambda *a, **k: None
    )
    monkeypatch.setattr("kspec_gfa_controller.gfa_actions.os.listdir", lambda p: [])

    prepare_successful_pipeline(actions, monkeypatch)
    r = await actions.guiding(SaveGrabRaw=False)
    assert r["status"] == "success"
    assert r["fwhm"] == 0.0


@pytest.mark.asyncio
async def test_guiding_exception_returns_error(actions, monkeypatch):
    def boom():
        raise RuntimeError("ensure failed")

    actions.env.astrometry.ensure_astrometry_ready = boom  # type: ignore
    monkeypatch.setattr(
        "kspec_gfa_controller.gfa_actions.os.makedirs", lambda *a, **k: None
    )
    monkeypatch.setattr("kspec_gfa_controller.gfa_actions.os.listdir", lambda p: [])

    prepare_successful_pipeline(actions, monkeypatch)
    r = await actions.guiding(SaveGrabRaw=False)
    assert r["status"] == "error"
    assert "guiding failed" in r["message"].lower()


# -------------------------
# pointing(): success + no images + exception
# -------------------------
@pytest.mark.asyncio
async def test_pointing_success(actions, monkeypatch):
    monkeypatch.setattr(
        "kspec_gfa_controller.gfa_actions.os.makedirs", lambda *a, **k: None
    )

    prepare_successful_pipeline(actions, monkeypatch, ["/tmp/raw/a.fits"])
    actions.env.astrometry._ensure_outputs = [
        "/tmp/astro_a.fits", "/tmp/astro_b.fits"
    ]
    patch_fits_crvals(monkeypatch)

    r = await actions.pointing(
        ra="1",
        dec="2",
        CamNum=0,
        clear_dir=True,
        SaveGrabRaw=False,
    )
    assert r["status"] == "success"
    assert r["images"] == ["astro_a.fits", "astro_b.fits"]
    assert r["crval1"] == [1.0, 1.0]
    assert r["crval2"] == [2.0, 2.0]


@pytest.mark.asyncio
async def test_pointing_no_images_returns_error(actions, monkeypatch):
    monkeypatch.setattr(
        "kspec_gfa_controller.gfa_actions.os.makedirs", lambda *a, **k: None
    )
    prepare_successful_pipeline(actions, monkeypatch)
    actions.env.astrometry._ensure_outputs = []
    r = await actions.pointing(ra="1", dec="2", SaveGrabRaw=False)
    assert r["status"] == "error"
    assert r["images"] == []
    assert r["crval1"] == []
    assert r["crval2"] == []


@pytest.mark.asyncio
async def test_pointing_exception_returns_error(actions, monkeypatch):
    monkeypatch.setattr(
        "kspec_gfa_controller.gfa_actions.os.makedirs", lambda *a, **k: None
    )
    prepare_successful_pipeline(actions, monkeypatch)

    def boom():
        raise RuntimeError("solve failed")

    actions.env.astrometry.ensure_astrometry_ready = boom

    r = await actions.pointing(ra="1", dec="2", SaveGrabRaw=False)
    assert r["status"] == "error"
    assert "pointing failed" in r["message"].lower()


# -------------------------
# status/ping/cam_params/shutdown (+error branches)
# -------------------------
def test_status_success(actions):
    r = actions.status()
    assert r["status"] == "success"
    assert isinstance(r["message"], dict)
    assert "Cam1" in r["message"]


def test_status_error(actions):
    def boom():
        raise RuntimeError("status failed")

    actions.env.controller.status = boom  # type: ignore
    r = actions.status()
    assert r["status"] == "error"


def test_ping_all_and_single(actions):
    r = actions.ping(CamNum=0)
    assert r["status"] == "success"
    assert actions.env.controller.ping_calls == actions.env.camera_ids

    actions.env.controller.ping_calls.clear()
    r = actions.ping(CamNum=2)
    assert r["status"] == "success"
    assert actions.env.controller.ping_calls == [2]


def test_ping_error(actions):
    def boom(cam_id):
        raise RuntimeError("ping failed")

    actions.env.controller.ping = boom  # type: ignore
    r = actions.ping(CamNum=2)
    assert r["status"] == "error"
    assert "ping failed" in r["message"].lower()


def test_cam_params_all_and_single(actions):
    r = actions.cam_params(CamNum=0)
    assert r["status"] == "success"
    assert "Cam1" in r["message"]

    r = actions.cam_params(CamNum=2)
    assert r["status"] == "success"
    assert "Cam2" in r["message"]


def test_cam_params_error(actions):
    def boom(cam_id):
        raise RuntimeError("params failed")

    actions.env.controller.cam_params = boom  # type: ignore
    r = actions.cam_params(CamNum=0)
    assert r["status"] == "error"
    assert "params failed" in r["message"].lower()


def test_shutdown_calls_env_shutdown_and_logs(actions):
    actions.shutdown()
    assert actions.env.shutdown_called == 1
    assert any(
        lvl == "info" and "shutdown complete" in msg.lower()
        for (lvl, msg) in actions.env.logger.logs
    )


# ---- 추가 테스트들: coverage holes 채우기 ----


def test_make_clean_subprocess_env_strips_pythonhome_pythonpath_and_prepends_pybin(
    ga_module, monkeypatch, tmp_path
):
    # env에 문제되는 변수 넣기
    monkeypatch.setenv("PYTHONHOME", "/bad/home")
    monkeypatch.setenv("PYTHONPATH", "/bad/path")
    monkeypatch.setenv("PATH", "/usr/bin")

    # os.sys.executable 기반으로 pybin이 앞에 붙는지 확인
    fake_py = tmp_path / "bin" / "python"
    fake_py.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(ga_module.os.sys, "executable", str(fake_py), raising=True)

    env = ga_module._make_clean_subprocess_env()
    assert "PYTHONHOME" not in env
    assert "PYTHONPATH" not in env
    assert env["PATH"].split(os.pathsep)[0] == str(fake_py.parent)


def test_apply_clean_env_to_astrometry_when_setter_missing_does_not_crash(ga_module):
    # astrometry에 set_subprocess_env가 없어도 조용히 통과해야 함
    class AstNoSetter:
        pass

    env = FakeEnv(astrometry=AstNoSetter())
    act = ga_module.GFAActions(env=env)
    act._apply_clean_env_to_astrometry()  # should not raise


def test_ensure_astrometry_outputs_ready_raises_when_no_astrometry(ga_module):
    class EnvNoAst:
        def __init__(self):
            self.logger = FakeLogger()

    act = ga_module.GFAActions(env=EnvNoAst())  # type: ignore
    with pytest.raises(RuntimeError):
        act._ensure_astrometry_outputs_ready()


def test_ensure_astrometry_outputs_ready_fallback_uses_existing_astro_files(
    ga_module, monkeypatch
):
    # ensure_astrometry_ready 없고, astro dir에 astro_*.fits가 있으면 바로 반환하는 fallback 커버
    class AstFallback:
        final_astrometry_dir = "/tmp/astrodir"
        dir_path = "/tmp/rawdir"

        def preproc(self):
            raise AssertionError("preproc should not be called when astro exists")

    env = FakeEnv(astrometry=AstFallback())
    act = ga_module.GFAActions(env=env)

    def fake_glob(pattern):
        if pattern.endswith(os.path.join("astrodir", "astro_*.fits")):
            return ["/tmp/astrodir/astro_a.fits", "/tmp/astrodir/astro_b.fits"]
        return []

    monkeypatch.setattr("kspec_gfa_controller.gfa_actions.glob.glob", fake_glob)
    outs = act._ensure_astrometry_outputs_ready()
    assert len(outs) == 2
    assert outs[0].endswith(".fits")


def test_ensure_astrometry_outputs_ready_fallback_runs_preproc_then_finds_files(
    ga_module, monkeypatch
):
    # ensure_astrometry_ready 없음 + 처음엔 astro가 없어서 preproc 수행 후 다시 glob로 찾는 fallback 커버
    state = {"after": False}

    class AstFallback:
        final_astrometry_dir = "/tmp/astroimg"

        def preproc(self):
            state["after"] = True
            return True

    env = FakeEnv(astrometry=AstFallback())
    act = ga_module.GFAActions(env=env)

    def fake_glob(pattern):
        # 디폴트 astro_dir = base_dir/img/astroimg -> ".../astroimg/astro_*.fits" 패턴
        if pattern.endswith(os.path.join("astroimg", "astro_*.fits")):
            return [] if not state["after"] else ["/tmp/astroimg/astro_x.fits"]
        return []

    monkeypatch.setattr("kspec_gfa_controller.gfa_actions.glob.glob", fake_glob)
    outs = act._ensure_astrometry_outputs_ready()
    assert outs == ["/tmp/astroimg/astro_x.fits"]


def test_ensure_astrometry_outputs_ready_fallback_raises_when_no_preproc(
    ga_module, monkeypatch
):
    # ensure_astrometry_ready도 없고 preproc도 없으면 RuntimeError
    class AstNoEnsureNoPreproc:
        final_astrometry_dir = "/tmp/astrodir"
        dir_path = "/tmp/rawdir"

    env = FakeEnv(astrometry=AstNoEnsureNoPreproc())
    act = ga_module.GFAActions(env=env)

    monkeypatch.setattr(
        "kspec_gfa_controller.gfa_actions.glob.glob", lambda *a, **k: []
    )
    with pytest.raises(RuntimeError):
        act._ensure_astrometry_outputs_ready()


@pytest.mark.asyncio
async def test_grab_custom_path_is_used(actions, monkeypatch, tmp_path):
    # grab(path=...) 분기 커버(기본 img/grab/YYYY-MM-DD 대신 사용)
    custom_path = tmp_path / "custom" / "save" / "here"
    r = await actions.grab(CamNum=1, path=str(custom_path))
    assert r["status"] == "success"
    assert Path(r["save_path"]) == custom_path.resolve()


@pytest.mark.asyncio
async def test_grab_close_all_cameras_failure_is_caught_and_warned(
    actions, monkeypatch
):
    # grab() finally에서 close_all_cameras 실패 warning branch(202-203 라인대) 커버
    monkeypatch.setattr(
        "kspec_gfa_controller.gfa_actions.os.makedirs", lambda *a, **k: None
    )

    async def boom_close():
        raise RuntimeError("close failed")

    actions.env.controller.close_all_cameras = boom_close  # type: ignore

    r = await actions.grab(CamNum=1)
    assert r["status"] == "success"  # close 실패해도 grab 자체 결과는 success로 감
    assert any(
        lvl == "warning" and "close_all_cameras failed" in msg
        for lvl, msg in actions.env.logger.logs
    )


@pytest.mark.asyncio
async def test_guiding_close_all_cameras_failure_is_caught_and_warned(
    actions, monkeypatch
):
    # 현재 guiding API는 grab 결과 오류를 그대로 guiding 오류로 변환한다.
    async def failed_grab(**kwargs):
        return {"status": "error", "message": "camera unavailable"}

    monkeypatch.setattr(actions, "grab", failed_grab)
    r = await actions.guiding(SaveGrabRaw=False)
    assert r["status"] == "error"
    assert "camera unavailable" in r["message"]


@pytest.mark.asyncio
async def test_pointing_save_true_copies_files(actions, monkeypatch, tmp_path):
    # pointing()의 save=True + shutil.copy2 경로 커버(322-323, 326-331 라인대)
    monkeypatch.setattr(
        "kspec_gfa_controller.gfa_actions.os.makedirs", lambda *a, **k: None
    )

    passed = ["/tmp/raw/a.fits", "/tmp/raw/b.fits"]
    prepare_successful_pipeline(actions, monkeypatch, passed)
    actions.env.astrometry._ensure_outputs = ["/tmp/astro_a.fits"]
    patch_fits_crvals(monkeypatch)

    copy_calls = []
    monkeypatch.setattr(
        "kspec_gfa_controller.gfa_actions.shutil.copy2",
        lambda s, d: copy_calls.append((s, d)),
    )

    r = await actions.pointing(
        ra="1",
        dec="2",
        clear_dir=False,
        SaveGrabRaw=True,
    )
    assert r["status"] == "success"
    assert len(copy_calls) == 2


# -----------------------------------------------------------------------------
# Release-coverage tests: path resolution and diagnostics
# -----------------------------------------------------------------------------
def test_get_save_root_uses_astrometry_config(ga_module, tmp_path):
    env = FakeEnv()
    env.save_root = None
    env.astrometry.inpar["paths"]["save_root"] = str(tmp_path / "configured")
    env.astrometry.inpar["paths"]["directories"] = {"raw_images": "camera_raw"}
    action = ga_module.GFAActions(env=env)

    root, dirs = action._get_save_root_and_dirs()

    assert root == (tmp_path / "configured").resolve()
    assert root.is_dir()
    assert dirs == {"raw_images": "camera_raw"}


def test_get_save_root_uses_default_without_astrometry(ga_module, tmp_path, monkeypatch):
    class BareEnv:
        logger = FakeLogger()
        save_root = None

    monkeypatch.setattr(ga_module.Path, "home", lambda: tmp_path)
    action = ga_module.GFAActions(env=BareEnv())

    root, dirs = action._get_save_root_and_dirs()

    assert root == (tmp_path / "work/DATA/GFADATA/img").resolve()
    assert dirs == {}


def test_debug_path_block_success(ga_module, tmp_path):
    actions = ga_module.GFAActions(env=FakeEnv())
    target = tmp_path / "debug-ok"

    actions._debug_path_block("unit", {"target": target})

    assert (target / "debug_write_test.txt").read_text() == "debug"
    assert any("write test ok" in msg for level, msg in actions.env.logger.logs)


def test_debug_path_block_handles_mkdir_failure(ga_module, tmp_path, monkeypatch):
    actions = ga_module.GFAActions(env=FakeEnv())

    def fail_mkdir(self, *args, **kwargs):
        raise OSError("mkdir denied")

    monkeypatch.setattr(Path, "mkdir", fail_mkdir)
    actions._debug_path_block("unit", {"target": tmp_path / "blocked"})

    assert any("mkdir failed" in msg for level, msg in actions.env.logger.logs)


def test_debug_path_block_handles_write_failure(ga_module, tmp_path, monkeypatch):
    actions = ga_module.GFAActions(env=FakeEnv())

    def fail_open(*args, **kwargs):
        raise OSError("write denied")

    monkeypatch.setattr("builtins.open", fail_open)
    actions._debug_path_block("unit", {"target": tmp_path / "write-blocked"})

    assert any("write test failed" in msg for level, msg in actions.env.logger.logs)


# -----------------------------------------------------------------------------
# Release-coverage tests: astrometry fallback failures
# -----------------------------------------------------------------------------
def test_ensure_astrometry_outputs_requires_final_directory(ga_module):
    class AstrometryWithoutDirectory:
        pass

    action = ga_module.GFAActions(env=FakeEnv(astrometry=AstrometryWithoutDirectory()))

    with pytest.raises(RuntimeError, match="final_astrometry_dir"):
        action._ensure_astrometry_outputs_ready()


def test_ensure_astrometry_outputs_rejects_failed_preproc(ga_module, monkeypatch):
    class FailedAstrometry:
        final_astrometry_dir = "/tmp/astro"

        def preproc(self):
            return False

    action = ga_module.GFAActions(env=FakeEnv(astrometry=FailedAstrometry()))
    monkeypatch.setattr(ga_module.glob, "glob", lambda pattern: [])

    with pytest.raises(RuntimeError, match="preproc failed"):
        action._ensure_astrometry_outputs_ready()


def test_ensure_astrometry_outputs_rejects_missing_post_preproc_files(
    ga_module, monkeypatch
):
    class SuccessfulAstrometryWithoutOutputs:
        final_astrometry_dir = "/tmp/astro"

        def preproc(self):
            return True

    action = ga_module.GFAActions(
        env=FakeEnv(astrometry=SuccessfulAstrometryWithoutOutputs())
    )
    monkeypatch.setattr(ga_module.glob, "glob", lambda pattern: [])

    with pytest.raises(RuntimeError, match="expected outputs not found"):
        action._ensure_astrometry_outputs_ready()


# -----------------------------------------------------------------------------
# Release-coverage tests: grab validation and multiple exposures
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_grab_rejects_exposure_time_above_limit(actions):
    with pytest.raises(ValueError, match="<= 10"):
        await actions.grab(ExpTime=10.1)


@pytest.mark.asyncio
async def test_grab_rejects_nonpositive_exposure_count(actions):
    with pytest.raises(ValueError, match=">= 1"):
        await actions.grab(ExpNum=0)


@pytest.mark.asyncio
async def test_grab_multiple_exposures_writes_combined_fits(actions):
    result = await actions.grab(CamNum=2, ExpTime=2.5, ExpNum=2)

    assert result["status"] == "success"
    assert len(actions.env.controller.grabone_calls) == 2
    assert len(actions.env.controller.img_class.save_calls) == 1
    save_call = actions.env.controller.img_class.save_calls[0]
    assert save_call["filename"].endswith("_combined.fits")
    assert save_call["exptime"] == 5.0
    assert len(save_call["image_array"]) == 2


@pytest.mark.asyncio
async def test_grab_skips_empty_image_lists(ga_module, tmp_path, monkeypatch):
    class EmptyImageMap:
        def __init__(self):
            self.camera_ids = []

        def __getitem__(self, camera_id):
            if camera_id not in self.camera_ids:
                self.camera_ids.append(camera_id)
            # Deliberately return a transient list so the defensive empty-list
            # guard in grab() can be exercised.
            return []

        def items(self):
            return [(camera_id, []) for camera_id in self.camera_ids]

    monkeypatch.setattr(ga_module, "defaultdict", lambda _factory: EmptyImageMap())
    action = ga_module.GFAActions(env=FakeEnv())
    monkeypatch.setattr(action, "_debug_path_block", lambda *a, **k: None)

    response = await action.grab(CamNum=2, path=str(tmp_path))

    assert response["status"] == "success"
    assert response["grab_files"] == []
    assert action.env.controller.img_class.save_calls == []


# -----------------------------------------------------------------------------
# Release-coverage tests: guiding retries and warning response
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_guiding_rejects_nonpositive_retry_count(actions):
    with pytest.raises(ValueError, match=">= 1"):
        await actions.guiding(MaxGrabRetry=0)


@pytest.mark.asyncio
async def test_guiding_retries_filter_then_succeeds(actions, monkeypatch):
    async def successful_grab(**kwargs):
        return {"status": "success", "message": "ok"}

    results = iter(
        [
            {"passed_files": [], "failed_files": ["bad.fits"], "n_passed": 0, "n_failed": 1},
            {"passed_files": ["good.fits"], "failed_files": [], "n_passed": 1, "n_failed": 0},
        ]
    )
    monkeypatch.setattr(actions, "grab", successful_grab)
    monkeypatch.setattr(actions, "_filter_pointing_raw_images", lambda **kwargs: next(results))

    response = await actions.guiding(SaveGrabRaw=False, MaxGrabRetry=2)

    assert response["status"] == "success"
    assert any("Retrying grab" in msg for level, msg in actions.env.logger.logs)


@pytest.mark.asyncio
async def test_guiding_returns_error_after_filter_retries_exhausted(actions, monkeypatch):
    async def successful_grab(**kwargs):
        return {"status": "success", "message": "ok"}

    failed_filter = {
        "passed_files": [],
        "failed_files": ["bad.fits"],
        "n_passed": 0,
        "n_failed": 1,
    }
    monkeypatch.setattr(actions, "grab", successful_grab)
    monkeypatch.setattr(
        actions, "_filter_pointing_raw_images", lambda **kwargs: failed_filter
    )

    response = await actions.guiding(SaveGrabRaw=False, MaxGrabRetry=2)

    assert response["status"] == "error"
    assert "not enough valid images" in response["message"]
    assert response["filter_result"] == failed_filter


@pytest.mark.asyncio
@pytest.mark.parametrize("offsets", [(None, 2.0, 3.0), (1.0, float("nan"), 3.0)])
async def test_guiding_warns_for_unreliable_offsets(actions, monkeypatch, offsets):
    prepare_successful_pipeline(actions, monkeypatch)
    actions.env.guider = FakeGuider(*offsets)

    response = await actions.guiding(SaveGrabRaw=False)

    assert response["status"] == "warning"
    assert "no reliable guide stars" in response["message"]


@pytest.mark.asyncio
async def test_guiding_treats_is_nan_check_exception_as_unreliable(
    actions, ga_module, monkeypatch
):
    import builtins

    sentinel = object()
    prepare_successful_pipeline(actions, monkeypatch)
    actions.env.guider = FakeGuider(sentinel, 2.0, 3.0)

    def raising_isinstance(value, expected_type):
        if value is sentinel and expected_type is float:
            raise RuntimeError("defensive isinstance failure")
        return builtins.isinstance(value, expected_type)

    monkeypatch.setattr(ga_module, "isinstance", raising_isinstance, raising=False)

    response = await actions.guiding(SaveGrabRaw=False)

    assert response["status"] == "warning"
    assert response["fdx"] is sentinel


# -----------------------------------------------------------------------------
# Release-coverage tests: pointing-image quality evaluation
# -----------------------------------------------------------------------------
class FakeSources:
    def __init__(self, flux=None, include_flux=True, length=None):
        self._flux = [] if flux is None else list(flux)
        self.colnames = ["flux"] if include_flux else ["xcentroid"]
        self._length = len(self._flux) if length is None else length

    def __len__(self):
        return self._length

    def __getitem__(self, name):
        if name != "flux" or "flux" not in self.colnames:
            raise KeyError(name)
        return self._flux


def patch_quality_dependencies(
    ga_module,
    monkeypatch,
    image,
    stats=(0.0, 0.0, 2.0),
    sources=None,
):
    monkeypatch.setattr(ga_module.fits, "getdata", lambda path: image)
    monkeypatch.setattr(ga_module, "sigma_clipped_stats", lambda data, sigma: stats)

    class Finder:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __call__(self, data):
            return sources

    monkeypatch.setattr(ga_module, "DAOStarFinder", Finder)


def test_evaluate_pointing_image_quality_passes_valid_image(
    actions, ga_module, monkeypatch
):
    patch_quality_dependencies(
        ga_module,
        monkeypatch,
        np.arange(100, dtype=float).reshape(10, 10),
        sources=FakeSources([5.0, np.nan, 10.0]),
    )

    result = actions._evaluate_pointing_image_quality(Path("good.fits"))

    assert result == {
        "passed": True,
        "n_peaks": 3,
        "brightest_flux": 10.0,
        "std_bg": 2.0,
        "reasons": [],
    }


def test_evaluate_pointing_image_quality_rejects_non_2d_image(
    actions, ga_module, monkeypatch
):
    monkeypatch.setattr(ga_module.fits, "getdata", lambda path: np.zeros((2, 2, 2)))

    result = actions._evaluate_pointing_image_quality(Path("cube.fits"))

    assert result["passed"] is False
    assert result["reasons"] == ["invalid_dimension=3"]


def test_evaluate_pointing_image_quality_rejects_all_nonfinite(
    actions, ga_module, monkeypatch
):
    monkeypatch.setattr(ga_module.fits, "getdata", lambda path: np.full((10, 10), np.nan))

    result = actions._evaluate_pointing_image_quality(Path("nan.fits"))

    assert result["passed"] is False
    assert result["reasons"] == ["no_finite_pixels"]


def test_evaluate_pointing_image_quality_reports_all_threshold_failures(
    actions, ga_module, monkeypatch
):
    image = np.ones((10, 10), dtype=float)
    image[0, :2] = np.nan
    patch_quality_dependencies(
        ga_module,
        monkeypatch,
        image,
        stats=(1.0, 1.0, 0.5),
        sources=FakeSources([], length=0),
    )

    result = actions._evaluate_pointing_image_quality(Path("weak.fits"))

    assert result["passed"] is False
    assert any("low_finite_fraction" in reason for reason in result["reasons"])
    assert any("low_std_bg" in reason for reason in result["reasons"])
    assert any("few_peaks" in reason for reason in result["reasons"])
    assert any("low_brightest_flux" in reason for reason in result["reasons"])


def test_evaluate_pointing_image_quality_handles_nonfinite_std(
    actions, ga_module, monkeypatch
):
    patch_quality_dependencies(
        ga_module,
        monkeypatch,
        np.ones((10, 10)),
        stats=(0.0, 0.0, np.nan),
    )

    result = actions._evaluate_pointing_image_quality(Path("bad-std.fits"))

    assert result["passed"] is False
    assert any("low_std_bg" in reason for reason in result["reasons"])


def test_evaluate_pointing_image_quality_handles_sources_without_flux(
    actions, ga_module, monkeypatch
):
    patch_quality_dependencies(
        ga_module,
        monkeypatch,
        np.ones((10, 10)),
        sources=FakeSources(include_flux=False, length=2),
    )

    result = actions._evaluate_pointing_image_quality(Path("no-flux.fits"))

    assert result["n_peaks"] == 2
    assert result["brightest_flux"] == 0.0
    assert result["passed"] is False


def test_evaluate_pointing_image_quality_handles_empty_finite_flux(
    actions, ga_module, monkeypatch
):
    patch_quality_dependencies(
        ga_module,
        monkeypatch,
        np.ones((10, 10)),
        sources=FakeSources([np.nan]),
    )

    result = actions._evaluate_pointing_image_quality(Path("nan-flux.fits"))

    assert result["n_peaks"] == 1
    assert result["brightest_flux"] == 0.0
    assert result["passed"] is False


def test_evaluate_pointing_image_quality_handles_detection_exception(
    actions, ga_module, monkeypatch
):
    monkeypatch.setattr(
        ga_module.fits, "getdata", lambda path: (_ for _ in ()).throw(OSError("bad FITS"))
    )

    result = actions._evaluate_pointing_image_quality(Path("broken.fits"))

    assert result["passed"] is False
    assert result["reasons"] == ["filter_error=OSError: bad FITS"]
    assert any(level == "exception" for level, msg in actions.env.logger.logs)


# -----------------------------------------------------------------------------
# Release-coverage tests: moving and filtering raw images
# -----------------------------------------------------------------------------
def test_move_with_unique_name_without_collision(actions, tmp_path):
    src = tmp_path / "raw" / "image.fits"
    src.parent.mkdir()
    src.write_text("data")
    destination = tmp_path / "unclean"

    moved = actions._move_with_unique_name(src, destination)

    assert moved == destination / "image.fits"
    assert moved.read_text() == "data"
    assert not src.exists()


def test_move_with_unique_name_adds_suffix_on_collision(actions, tmp_path):
    src = tmp_path / "raw" / "image.fits"
    src.parent.mkdir()
    src.write_text("new")
    destination = tmp_path / "unclean"
    destination.mkdir()
    (destination / "image.fits").write_text("old")

    moved = actions._move_with_unique_name(src, destination)

    assert moved == destination / "image_001.fits"
    assert moved.read_text() == "new"


def test_move_with_unique_name_raises_when_all_names_are_taken(
    actions, tmp_path, monkeypatch
):
    src = tmp_path / "image.fits"
    src.write_text("data")
    monkeypatch.setattr(Path, "exists", lambda self: True)

    with pytest.raises(RuntimeError, match="unique filename"):
        actions._move_with_unique_name(src, tmp_path / "unclean")


def test_filter_pointing_raw_images_separates_pass_and_fail(
    actions, tmp_path, monkeypatch
):
    raw = tmp_path / "raw-filter"
    unclean = tmp_path / "unclean-filter"
    raw.mkdir()
    good = raw / "good.fits"
    bad = raw / "bad.fit"
    ignored = raw / "notes.txt"
    good.write_text("good")
    bad.write_text("bad")
    ignored.write_text("ignore")

    def evaluate(path):
        if path.name == "good.fits":
            return {
                "passed": True,
                "std_bg": 2.0,
                "n_peaks": 3,
                "brightest_flux": 10.0,
                "reasons": [],
            }
        return {
            "passed": False,
            "std_bg": 0.0,
            "n_peaks": 0,
            "brightest_flux": 0.0,
            "reasons": ["bad image"],
        }

    monkeypatch.setattr(actions, "_evaluate_pointing_image_quality", evaluate)

    result = actions._filter_pointing_raw_images(raw, unclean, label="unit")

    assert result["n_passed"] == 1
    assert result["n_failed"] == 1
    assert result["passed_files"] == [str(good)]
    assert result["failed_files"] == [str(unclean / "bad.fit")]
    assert good.exists()
    assert not bad.exists()
    assert ignored.exists()


# -----------------------------------------------------------------------------
# Release-coverage tests: pointing retry and FITS-header failure branches
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_pointing_retries_filter_then_succeeds(actions, monkeypatch):
    results = iter(
        [
            {"passed_files": [], "failed_files": ["bad.fits"], "n_passed": 0, "n_failed": 1},
            {"passed_files": ["good.fits"], "failed_files": [], "n_passed": 1, "n_failed": 0},
        ]
    )
    monkeypatch.setattr(actions, "_filter_pointing_raw_images", lambda **kwargs: next(results))
    actions.env.astrometry._ensure_outputs = ["/tmp/astro_good.fits"]
    patch_fits_crvals(monkeypatch)

    response = await actions.pointing(
        ra="1", dec="2", SaveGrabRaw=False, MaxGrabRetry=2
    )

    assert response["status"] == "success"
    assert any("Retrying grab" in msg for level, msg in actions.env.logger.logs)


@pytest.mark.asyncio
async def test_pointing_returns_error_after_filter_retries_exhausted(
    actions, monkeypatch
):
    failed_filter = {
        "passed_files": [],
        "failed_files": ["bad.fits"],
        "n_passed": 0,
        "n_failed": 1,
    }
    monkeypatch.setattr(
        actions, "_filter_pointing_raw_images", lambda **kwargs: failed_filter
    )

    response = await actions.pointing(
        ra="1", dec="2", SaveGrabRaw=False, MaxGrabRetry=2
    )

    assert response["status"] == "error"
    assert "not enough valid images" in response["message"]
    assert response["filter_result"] == failed_filter


@pytest.mark.asyncio
async def test_pointing_returns_nan_when_fits_header_read_fails(actions, monkeypatch):
    prepare_successful_pipeline(actions, monkeypatch)
    actions.env.astrometry._ensure_outputs = ["/tmp/broken.fits"]
    monkeypatch.setattr(
        "kspec_gfa_controller.gfa_actions.fits.open",
        lambda path: (_ for _ in ()).throw(OSError("cannot read header")),
    )

    response = await actions.pointing(ra="1", dec="2", SaveGrabRaw=False)

    assert response["status"] == "success"
    assert np.isnan(response["crval1"][0])
    assert np.isnan(response["crval2"][0])
    assert any("Failed to read CRVAL" in msg for level, msg in actions.env.logger.logs)
