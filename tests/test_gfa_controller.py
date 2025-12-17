# tests/test_gfa_controller.py
import importlib
import json
import sys
import types
from pathlib import Path

import pytest


# -------------------------
# Helpers: write configs
# -------------------------
def _write_cams_json(path: Path):
    cfg = {
        "GfaController": {
            "Elements": {
                "Cameras": {
                    "Elements": {
                        "Cam1": {
                            "IpAddress": "1.1.1.1",
                            "PacketSize": 1500,
                            "InterPacketDelay": 10,
                        },
                        "Cam2": {
                            "IpAddress": "2.2.2.2",
                            "PacketSize": "not_int",
                            "InterPacketDelay": None,
                        },
                    }
                }
            }
        }
    }
    path.write_text(json.dumps(cfg), encoding="utf-8")


def _write_cams_yaml(path: Path):
    path.write_text(
        """
GfaController:
  Elements:
    Cameras:
      Elements:
        Cam1:
          IpAddress: "1.1.1.1"
          PacketSize: 1500
          InterPacketDelay: 10
""".strip(),
        encoding="utf-8",
    )


# -------------------------
# Fakes: logger
# -------------------------
class FakeLogger:
    def __init__(self):
        self.msg = []

    def info(self, m):
        self.msg.append(("info", str(m)))

    def warning(self, m):
        self.msg.append(("warning", str(m)))

    def error(self, m):
        self.msg.append(("error", str(m)))

    def debug(self, m):
        self.msg.append(("debug", str(m)))


# -------------------------
# Fakes: pypylon/genicam + camera nodes
# -------------------------
class FakeTimeoutException(Exception):
    pass


class FakeNode:
    def __init__(self, value=None, raise_on_get=False):
        self._value = value
        self._raise_on_get = raise_on_get
        self.set_calls = []

    def SetValue(self, v):
        self.set_calls.append(v)
        self._value = v

    def GetValue(self):
        if self._raise_on_get:
            raise RuntimeError("AccessException")
        return self._value


class FakeGrabResult:
    def __init__(self, arr):
        self._arr = arr

    def GetArray(self):
        return self._arr


class FakeInstantCamera:
    def __init__(self, _device, *, open_state=False, raise_timeout=False):
        self._open = open_state
        self._raise_timeout = raise_timeout

        # nodes used by configure_and_grab
        self.GevSCPSPacketSize = FakeNode()
        self.GevSCPD = FakeNode()
        self.GevSCFTD = FakeNode()
        self.ExposureTime = FakeNode()
        self.PixelFormat = FakeNode()
        self.BinningHorizontal = FakeNode()
        self.BinningVertical = FakeNode()

        # node used by grabone
        self.DeviceSerialNumber = FakeNode("SERIAL123")

        # nodes used by cam_params (GetValue)
        self.DeviceModelName = FakeNode("MODEL")
        self.DeviceSerialNumber = FakeNode("SERIAL123")
        self.DeviceUserID = FakeNode("USERID")
        self.Width = FakeNode(100)
        self.Height = FakeNode(200)
        self.PixelFormat = FakeNode("Mono12")
        self.ExposureTime = FakeNode(10000)
        self.BinningHorizontalMode = FakeNode("Sum")
        self.BinningHorizontal = FakeNode(1)
        self.BinningVerticalMode = FakeNode("Sum")
        self.BinningVertical = FakeNode(1, raise_on_get=True)  # except branch cover

    def Open(self):
        self._open = True

    def Close(self):
        self._open = False

    def IsOpen(self):
        return self._open

    def GrabOne(self, timeout_ms):
        if self._raise_timeout:
            raise FakeTimeoutException("timeout")
        return FakeGrabResult(arr=[[1, 2], [3, 4]])


class FakeDeviceInfo:
    def __init__(self):
        self.ip = None

    def SetIpAddress(self, ip):
        self.ip = ip


class FakeTlFactory:
    @staticmethod
    def GetInstance():
        return FakeTlFactory()

    def CreateDevice(self, dev_info):
        return object()


# -------------------------
# Fixture: import module with fake pypylon
# -------------------------
@pytest.fixture
def gc_module(monkeypatch):
    # pypylon, pypylon.pylon, pypylon.genicam 을 가짜로 선주입
    pypylon_mod = types.ModuleType("pypylon")
    pylon_mod = types.ModuleType("pypylon.pylon")
    genicam_mod = types.ModuleType("pypylon.genicam")

    pylon_mod.TlFactory = FakeTlFactory
    pylon_mod.DeviceInfo = FakeDeviceInfo
    pylon_mod.InstantCamera = lambda dev: FakeInstantCamera(dev, open_state=False)

    genicam_mod.TimeoutException = FakeTimeoutException

    monkeypatch.setitem(sys.modules, "pypylon", pypylon_mod)
    monkeypatch.setitem(sys.modules, "pypylon.pylon", pylon_mod)
    monkeypatch.setitem(sys.modules, "pypylon.genicam", genicam_mod)

    # gfa_img.GFAImage도 fake로
    gfa_img_mod = types.ModuleType("gfa_img")

    class FakeGFAImage:
        def __init__(self, logger):
            self.logger = logger
            self.save_calls = []

        def save_fits(self, **kwargs):
            self.save_calls.append(kwargs)

    gfa_img_mod.GFAImage = FakeGFAImage
    monkeypatch.setitem(sys.modules, "gfa_img", gfa_img_mod)

    import gfa_controller

    importlib.reload(gfa_controller)
    return gfa_controller


@pytest.fixture
def controller(tmp_path, gc_module):
    cfgp = tmp_path / "cams.json"
    _write_cams_json(cfgp)
    return gc_module.GFAController(config=str(cfgp), logger=FakeLogger())


# -------------------------
# from_config()
# -------------------------
def test_from_config_json_loads(tmp_path, gc_module):
    cfgp = tmp_path / "cams.json"
    _write_cams_json(cfgp)
    data = gc_module.from_config(str(cfgp))
    assert "GfaController" in data


def test_from_config_yaml_loads(tmp_path, gc_module):
    cfgp = tmp_path / "cams.yaml"
    _write_cams_yaml(cfgp)
    data = gc_module.from_config(str(cfgp))
    assert (
        data["GfaController"]["Elements"]["Cameras"]["Elements"]["Cam1"]["IpAddress"]
        == "1.1.1.1"
    )


def test_from_config_unsupported_raises(tmp_path, gc_module):
    p = tmp_path / "cams.txt"
    p.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        gc_module.from_config(str(p))


# -------------------------
# default config/logger helpers
# -------------------------
def test_default_logger_no_duplicate_handlers(gc_module):
    lg1 = gc_module._get_default_logger()
    n1 = len(lg1.handlers)
    lg2 = gc_module._get_default_logger()
    n2 = len(lg2.handlers)
    assert lg1 is lg2
    assert n2 == n1


def test_default_config_path_missing_raises(monkeypatch, gc_module):
    real_isfile = gc_module.os.path.isfile

    def fake_isfile(p):
        if str(p).endswith(
            gc_module.os.path.normpath(gc_module.os.path.join("etc", "cams.json"))
        ):
            return False
        return real_isfile(p)

    monkeypatch.setattr(gc_module.os.path, "isfile", fake_isfile)
    with pytest.raises(FileNotFoundError):
        gc_module._get_default_config_path()


# -------------------------
# __init__ error branches
# -------------------------
def test_init_from_config_error_re_raises(tmp_path, gc_module):
    bad = tmp_path / "bad.json"
    bad.write_text("{ not-json", encoding="utf-8")

    with pytest.raises(Exception):
        gc_module.GFAController(config=str(bad), logger=FakeLogger())


def test_init_missing_key_raises(tmp_path, gc_module):
    cfg = {"wrong": 1}
    p = tmp_path / "cams.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")

    with pytest.raises(KeyError):
        gc_module.GFAController(config=str(p), logger=FakeLogger())


# -------------------------
# get_camera_param()
# -------------------------
def test_get_camera_param_int_and_nonint(controller):
    assert controller.get_camera_param(1, "PacketSize") == 1500
    assert controller.get_camera_param(2, "PacketSize") is None
    assert controller.get_camera_param(2, "InterPacketDelay") is None
    assert controller.get_camera_param(1, "DoesNotExist") is None


# -------------------------
# open/close/status/ping
# -------------------------
def test_open_all_cameras_and_status_and_close_all(controller):
    controller.open_all_cameras()
    assert (
        "Cam1" in controller.open_cameras
        and controller.open_cameras["Cam1"].IsOpen() is True
    )
    assert (
        "Cam2" in controller.open_cameras
        and controller.open_cameras["Cam2"].IsOpen() is True
    )

    st = controller.status()
    assert st["Cam1"] is True
    assert st["Cam2"] is True

    controller.close_all_cameras()
    assert controller.open_cameras == {}

    st2 = controller.status()
    assert st2["Cam1"] is False
    assert st2["Cam2"] is False


def test_ping_calls_os_system_with_ip(controller, monkeypatch):
    # ✅ os는 인스턴스 속성이 아니라 모듈 전역이므로 gfa_controller.os.system을 patch해야 함
    called = {}

    def fake_system(cmd):
        called["cmd"] = cmd
        return 0

    monkeypatch.setattr("gfa_controller.os.system", fake_system)

    controller.ping(CamNum=1)
    assert "1.1.1.1" in called["cmd"]


def test_ping_missing_camera_raises(controller):
    with pytest.raises(KeyError):
        controller.ping(CamNum=99)


# -------------------------
# cam_params()
# -------------------------
def test_cam_params_invalid_camnum_raises(controller):
    with pytest.raises(KeyError):
        controller.cam_params(99)


def test_cam_params_opens_temporarily_and_returns_params(controller):
    params = controller.cam_params(1)
    assert isinstance(params, dict)
    assert "DeviceModelName" in params
    assert params["BinningVertical"] is None  # GetValue 예외 -> None
    assert "Cam1" in controller.open_cameras
    assert controller.open_cameras["Cam1"].IsOpen() is True


# -------------------------
# configure_and_grab()
# -------------------------
@pytest.mark.asyncio
async def test_configure_and_grab_success_saves_fits(controller):
    cam = FakeInstantCamera(object(), open_state=True, raise_timeout=False)

    img = await controller.configure_and_grab(
        cam=cam,
        ExpTime=1.2,
        Binning=2,
        packet_size=1500,
        ipd=10,
        ftd_base=39000,
        cam_index=0,
        output_dir="OUT",
        serial_hint="SERIALX",
        ftd=None,
        ra="1",
        dec="2",
    )
    assert img is not None

    assert len(controller.img_class.save_calls) == 1
    call = controller.img_class.save_calls[0]
    assert call["exptime"] == 1.2
    assert call["output_directory"] == "OUT"
    assert call["ra"] == "1"
    assert call["dec"] == "2"
    assert call["filename"].endswith(".fits")


@pytest.mark.asyncio
async def test_configure_and_grab_timeout_returns_none(controller):
    cam = FakeInstantCamera(object(), open_state=True, raise_timeout=True)
    img = await controller.configure_and_grab(
        cam=cam,
        ExpTime=1.0,
        Binning=1,
        packet_size=1500,
        ipd=10,
        ftd_base=39000,
        cam_index=0,
        output_dir="OUT",
        serial_hint="SERIALX",
    )
    assert img is None


@pytest.mark.asyncio
async def test_configure_and_grab_ftd_override_used(controller):
    cam = FakeInstantCamera(object(), open_state=True, raise_timeout=False)

    await controller.configure_and_grab(
        cam=cam,
        ExpTime=1.0,
        Binning=1,
        packet_size=1000,
        ipd=5,
        ftd_base=39000,
        cam_index=2,
        output_dir="OUT",
        serial_hint="SERIALX",
        ftd=777,
    )
    assert cam.GevSCFTD.set_calls[-1] == 777


# -------------------------
# grabone() / grab()
# -------------------------
@pytest.mark.asyncio
async def test_grabone_camera_not_opened_returns_timeout_list(controller):
    controller.open_cameras.clear()
    assert await controller.grabone(CamNum=1, ExpTime=1.0, Binning=1) == [1]


@pytest.mark.asyncio
async def test_grabone_uses_config_params_and_success_returns_empty(controller):
    cam = FakeInstantCamera(object(), open_state=True, raise_timeout=False)
    controller.open_cameras["Cam1"] = cam

    out = await controller.grabone(
        CamNum=1, ExpTime=1.0, Binning=1, packet_size=None, ipd=None
    )
    assert out == []


@pytest.mark.asyncio
async def test_grabone_configure_returns_none_marks_timeout(controller, monkeypatch):
    cam = FakeInstantCamera(object(), open_state=True, raise_timeout=False)
    controller.open_cameras["Cam1"] = cam

    async def fake_configure_and_grab(*a, **k):
        return None

    monkeypatch.setattr(controller, "configure_and_grab", fake_configure_and_grab)

    out = await controller.grabone(CamNum=1, ExpTime=1.0, Binning=1)
    assert out == [1]


@pytest.mark.asyncio
async def test_grabone_configure_raises_exception_marks_timeout(
    controller, monkeypatch
):
    cam = FakeInstantCamera(object(), open_state=True, raise_timeout=False)
    controller.open_cameras["Cam1"] = cam

    async def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(controller, "configure_and_grab", boom)

    out = await controller.grabone(CamNum=1, ExpTime=1.0, Binning=1)
    assert out == [1]


@pytest.mark.asyncio
async def test_grab_builds_cam_list_and_aggregates_timeouts(tmp_path, gc_module):
    cfgp = tmp_path / "cams.json"
    _write_cams_json(cfgp)
    c = gc_module.GFAController(config=str(cfgp), logger=FakeLogger())
    assert c.NUM_CAMERAS == 2

    async def fake_grabone(CamNum, **kwargs):
        return [CamNum] if CamNum == 1 else []

    c.grabone = fake_grabone  # type: ignore

    assert await c.grab(CamNum=0, ExpTime=1.0, Binning=1) == [1]
    assert await c.grab(CamNum=[2, 1], ExpTime=1.0, Binning=1) == [1]
    assert await c.grab(CamNum=2, ExpTime=1.0, Binning=1) == []
