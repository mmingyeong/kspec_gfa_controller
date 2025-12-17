# tests/test_gfa_actions.py
import asyncio
import os
from pathlib import Path

import pytest


# -------------------------
# Minimal fakes
# -------------------------
class FakeLogger:
    def __init__(self):
        self.logs = []

    def info(self, m): self.logs.append(("info", str(m)))
    def debug(self, m): self.logs.append(("debug", str(m)))
    def warning(self, m): self.logs.append(("warning", str(m)))
    def error(self, m): self.logs.append(("error", str(m)))


class FakeController:
    def __init__(self, grabone_result=None):
        self._grabone_result = grabone_result if grabone_result is not None else []
        self.grabone_calls = []
        self.grab_calls = []
        self.ping_calls = []
        self.status_called = 0
        self.cam_params_calls = []

    async def grabone(self, **kwargs):
        self.grabone_calls.append(kwargs)
        return list(self._grabone_result)

    def grab(self, CamNum, ExpTime, Binning, **kwargs):
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
    def __init__(self):
        self.preproc_called = 0
        self.clear_called = 0

    def preproc(self):
        self.preproc_called += 1

    def clear_raw_and_processed_files(self):
        self.clear_called += 1


class FakeGuider:
    def __init__(self, fdx=1.0, fdy=2.0, fwhm=3.0):
        self._ret = (fdx, fdy, fwhm)
        self.exe_called = 0

    def exe_cal(self):
        self.exe_called += 1
        return self._ret


class FakeEnv:
    def __init__(self, camera_ids=(1, 2, 3), controller=None, astrometry=None, guider=None):
        self.logger = FakeLogger()
        self.camera_ids = list(camera_ids)
        self.controller = controller if controller is not None else FakeController()
        self.astrometry = astrometry if astrometry is not None else FakeAstrometry()
        self.guider = guider if guider is not None else FakeGuider()
        self.shutdown_called = 0

    def shutdown(self):
        self.shutdown_called += 1


@pytest.fixture
def actions(monkeypatch):
    # gfa_actions import (top-level)
    from gfa_actions import GFAActions
    return GFAActions(env=FakeEnv())


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
async def test_grab_single_camera_success_message(actions, tmp_path, monkeypatch):
    # 고정 경로로 만들기 위해 __file__ 기반 base_dir을 쓰는 코드라서,
    # mkdir 호출이 없어도 동작하게 controller.grabone만 확인하면 됨.
    actions.env.controller._grabone_result = []  # timeout 없음

    r = await actions.grab(CamNum=2, ExpTime=1.5, Binning=4, packet_size=1500, cam_ipd=10, cam_ftd_base=123, ra="1", dec="2")
    assert r["status"] == "success"
    assert "camera 2" in r["message"].lower()

    # grabone 호출 파라미터 확인
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
async def test_grab_single_camera_timeout_in_message(actions):
    actions.env.controller._grabone_result = [2]  # timeout cam 2

    r = await actions.grab(CamNum=2)
    assert r["status"] == "success"
    assert "timeout" in r["message"].lower()


# -------------------------
# grab(): CamNum=0 (all)
# -------------------------
@pytest.mark.asyncio
async def test_grab_all_cameras_aggregates_timeouts(actions, monkeypatch):
    # camera_ids= [1,2,3], 각 grabone이 timeout 반환하도록 순서대로 다르게 주고 싶으면
    # grabone을 monkeypatch로 교체
    async def fake_grabone(**kwargs):
        cam = kwargs["CamNum"]
        return [cam] if cam in (1, 3) else []

    actions.env.controller.grabone = fake_grabone

    r = await actions.grab(CamNum=0)
    assert r["status"] == "success"
    assert "all cameras" in r["message"].lower()
    assert "timeout" in r["message"].lower()
    assert "1" in r["message"]
    assert "3" in r["message"]


# -------------------------
# grab(): CamNum=list
# -------------------------
@pytest.mark.asyncio
async def test_grab_camera_list(actions, monkeypatch):
    async def fake_grabone(**kwargs):
        return [kwargs["CamNum"]] if kwargs["CamNum"] == 5 else []

    actions.env.camera_ids = [1, 2, 3, 4, 5]
    actions.env.controller.grabone = fake_grabone

    r = await actions.grab(CamNum=[4, 5])
    assert r["status"] == "success"
    assert "cameras" in r["message"].lower()
    assert "timeout" in r["message"].lower()


@pytest.mark.asyncio
async def test_grab_invalid_camnum_returns_error(actions):
    r = await actions.grab(CamNum="bad")  # type: ignore
    assert r["status"] == "error"
    assert "grab failed" in r["message"].lower()


# -------------------------
# guiding(): success path (save=False)
# -------------------------
@pytest.mark.asyncio
async def test_guiding_success_no_save(actions, monkeypatch, tmp_path):
    # os.makedirs / os.listdir / shutil.copy2 등 파일 I/O를 최소화
    monkeypatch.setattr("gfa_actions.os.makedirs", lambda *a, **k: None)
    monkeypatch.setattr("gfa_actions.os.listdir", lambda p: [])
    # controller.grab은 동기 호출
    actions.env.controller.grab_calls.clear()

    r = await actions.guiding(ExpTime=2.0, save=False, ra="1", dec="2")
    assert r["status"] == "success"
    assert "Offsets:" in r["message"]
    assert "fdx" in r and "fdy" in r and "fwhm" in r

    # grab 호출 확인 (CamNum=0)
    assert len(actions.env.controller.grab_calls) == 1
    camnum, exptime, binning, kwargs = actions.env.controller.grab_calls[0]
    assert camnum == 0
    assert exptime == 2.0
    assert binning == 4
    assert kwargs["ra"] == "1"
    assert kwargs["dec"] == "2"

    assert actions.env.astrometry.preproc_called == 1
    assert actions.env.guider.exe_called == 1
    assert actions.env.astrometry.clear_called == 1


@pytest.mark.asyncio
async def test_guiding_fwhm_nonfloat_becomes_zero(actions, monkeypatch):
    # guider가 문자열 fwhm 반환 -> 0.0으로 변환되는지
    actions.env.guider = FakeGuider(fdx=1.0, fdy=2.0, fwhm="bad")  # type: ignore

    monkeypatch.setattr("gfa_actions.os.makedirs", lambda *a, **k: None)
    monkeypatch.setattr("gfa_actions.os.listdir", lambda p: [])

    r = await actions.guiding()
    assert r["status"] == "success"
    assert r["fwhm"] == 0.0


@pytest.mark.asyncio
async def test_guiding_exception_returns_error(actions, monkeypatch):
    def boom():
        raise RuntimeError("preproc failed")

    actions.env.astrometry.preproc = boom  # type: ignore

    monkeypatch.setattr("gfa_actions.os.makedirs", lambda *a, **k: None)
    monkeypatch.setattr("gfa_actions.os.listdir", lambda p: [])

    r = await actions.guiding()
    assert r["status"] == "error"
    assert "guiding failed" in r["message"].lower()


# -------------------------
# pointing(): success + no images
# -------------------------
@pytest.mark.asyncio
async def test_pointing_success(actions, monkeypatch, tmp_path):
    # base_dir/img/pointing_raw/date 디렉토리 작업이 있으니 os.*를 mock
    monkeypatch.setattr("gfa_actions.os.makedirs", lambda *a, **k: None)

    # clear_dir True에서 os.listdir 호출 → 기존 파일 없다고 가정
    monkeypatch.setattr("gfa_actions.os.listdir", lambda p: ["a.fits", "b.fit", "c.txt"])
    monkeypatch.setattr("gfa_actions.os.path.isfile", lambda p: True)
    monkeypatch.setattr("gfa_actions.os.remove", lambda p: None)

    # controller.grab 호출 확인만
    actions.env.controller.grab_calls.clear()

    # get_crvals_from_images mock
    monkeypatch.setattr("gfa_actions.get_crvals_from_images", lambda images, max_workers: ([1.0]*len(images), [2.0]*len(images)))

    r = await actions.pointing(ra="1", dec="2", CamNum=0, save_by_date=False, clear_dir=True, max_workers=3)
    assert r["status"] == "success"
    assert len(r["images"]) == 2  # a.fits, b.fit만
    assert r["crval1"] == [1.0, 1.0]
    assert r["crval2"] == [2.0, 2.0]

    # grab이 호출되었는지
    assert len(actions.env.controller.grab_calls) == 1
    camnum, exptime, binning, kwargs = actions.env.controller.grab_calls[0]
    assert camnum == 0
    assert kwargs["ra"] == "1"
    assert kwargs["dec"] == "2"


@pytest.mark.asyncio
async def test_pointing_no_images_returns_error(actions, monkeypatch):
    monkeypatch.setattr("gfa_actions.os.makedirs", lambda *a, **k: None)
    monkeypatch.setattr("gfa_actions.os.listdir", lambda p: ["note.txt"])  # fits 없음
    monkeypatch.setattr("gfa_actions.os.path.isfile", lambda p: True)
    monkeypatch.setattr("gfa_actions.os.remove", lambda p: None)

    r = await actions.pointing(ra="1", dec="2", save_by_date=False, clear_dir=True)
    assert r["status"] == "error"
    assert r["images"] == []
    assert r["crval1"] == []
    assert r["crval2"] == []


# -------------------------
# status/ping/cam_params/shutdown
# -------------------------
def test_status_success(actions):
    r = actions.status()
    assert r["status"] == "success"
    assert isinstance(r["message"], dict)
    assert "Cam1" in r["message"]


def test_status_error(actions, monkeypatch):
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


def test_cam_params_all_and_single(actions):
    r = actions.cam_params(CamNum=0)
    assert r["status"] == "success"
    # 메시지에 Cam{n}: {dict}가 들어가므로 간단히 포함 여부만 체크
    assert "Cam1" in r["message"]

    r = actions.cam_params(CamNum=2)
    assert r["status"] == "success"
    assert "Cam2" in r["message"]


def test_shutdown_calls_env_shutdown(actions):
    actions.shutdown()
    assert actions.env.shutdown_called == 1
