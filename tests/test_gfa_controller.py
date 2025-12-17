# tests/test_gfa_controller.py
import json
from pathlib import Path

import pytest

from gfa_controller import GFAController, from_config


def _write_cams_json(path: Path):
    cfg = {
        "GfaController": {
            "Elements": {
                "Cameras": {
                    "Elements": {
                        "Cam1": {"IpAddress": "1.1.1.1", "PacketSize": 1500, "InterPacketDelay": 10},
                        "Cam2": {"IpAddress": "2.2.2.2", "PacketSize": "not_int", "InterPacketDelay": None},
                    }
                }
            }
        }
    }
    path.write_text(json.dumps(cfg), encoding="utf-8")


class FakeLogger:
    def __init__(self):
        self.msg = []

    def info(self, m): self.msg.append(("info", str(m)))
    def warning(self, m): self.msg.append(("warning", str(m)))
    def error(self, m): self.msg.append(("error", str(m)))


class FakeTlFactory:
    @staticmethod
    def GetInstance():
        return FakeTlFactory()

    def CreateDevice(self, dev_info):
        return object()


def test_from_config_json_loads(tmp_path):
    cfgp = tmp_path / "cams.json"
    _write_cams_json(cfgp)
    data = from_config(str(cfgp))
    assert "GfaController" in data


def test_get_camera_param_int_and_nonint(tmp_path, monkeypatch):
    cfgp = tmp_path / "cams.json"
    _write_cams_json(cfgp)

    # 하드웨어 접근 방지: TlFactory만 fake로
    monkeypatch.setattr("gfa_controller.py.TlFactory", FakeTlFactory)

    c = GFAController(config=str(cfgp), logger=FakeLogger())

    assert c.get_camera_param(1, "PacketSize") == 1500
    assert c.get_camera_param(2, "PacketSize") is None
    assert c.get_camera_param(2, "InterPacketDelay") is None
    assert c.get_camera_param(1, "DoesNotExist") is None


@pytest.mark.asyncio
async def test_grab_builds_cam_list_and_aggregates_timeouts(tmp_path, monkeypatch):
    cfgp = tmp_path / "cams.json"
    _write_cams_json(cfgp)

    monkeypatch.setattr("gfa_controller.py.TlFactory", FakeTlFactory)
    c = GFAController(config=str(cfgp), logger=FakeLogger())
    assert c.NUM_CAMERAS == 2

    async def fake_grabone(CamNum, **kwargs):
        return [CamNum] if CamNum == 1 else []

    monkeypatch.setattr(c, "grabone", fake_grabone)

    # CamNum=0 => all cameras [1,2]
    assert await c.grab(CamNum=0, ExpTime=1.0, Binning=1) == [1]
    # list 입력
    assert await c.grab(CamNum=[2, 1], ExpTime=1.0, Binning=1) == [1]
    # 단일 입력
    assert await c.grab(CamNum=2, ExpTime=1.0, Binning=1) == []


@pytest.mark.asyncio
async def test_grabone_returns_timeout_if_camera_not_opened(tmp_path, monkeypatch):
    cfgp = tmp_path / "cams.json"
    _write_cams_json(cfgp)

    monkeypatch.setattr("gfa_controller.py.TlFactory", FakeTlFactory)
    c = GFAController(config=str(cfgp), logger=FakeLogger())

    # open_cameras에 Cam1이 없으면 바로 [1]
    assert await c.grabone(CamNum=1, ExpTime=1.0, Binning=1) == [1]

class FakeDeviceInfo:
    def __init__(self):
        self.ip = None

    def SetIpAddress(self, ip):
        self.ip = ip


class FakeInstantCamera:
    def __init__(self, _device):
        self._open = False
        # cam_params에서 쓸 수 있게 최소 노드 제공(지금 테스트에선 필수 아님)
        self.DeviceSerialNumber = type("N", (), {"GetValue": lambda self: "SERIAL"})()

    def Open(self):
        self._open = True

    def Close(self):
        self._open = False

    def IsOpen(self):
        return self._open


def test_open_camera_opens_and_registers(tmp_path, monkeypatch):
    cfgp = tmp_path / "cams.json"
    _write_cams_json(cfgp)

    # 하드웨어 접근 막기
    monkeypatch.setattr("gfa_controller.py.TlFactory", FakeTlFactory)
    monkeypatch.setattr("gfa_controller.py.DeviceInfo", FakeDeviceInfo)
    monkeypatch.setattr("gfa_controller.py.InstantCamera", FakeInstantCamera)

    c = GFAController(config=str(cfgp), logger=FakeLogger())

    c.open_camera(1)
    assert "Cam1" in c.open_cameras
    assert c.open_cameras["Cam1"].IsOpen() is True


def test_open_camera_already_open_no_duplicate(tmp_path, monkeypatch):
    cfgp = tmp_path / "cams.json"
    _write_cams_json(cfgp)

    monkeypatch.setattr("gfa_controller.py.TlFactory", FakeTlFactory)
    monkeypatch.setattr("gfa_controller.py.DeviceInfo", FakeDeviceInfo)
    monkeypatch.setattr("gfa_controller.py.InstantCamera", FakeInstantCamera)

    c = GFAController(config=str(cfgp), logger=FakeLogger())

    c.open_camera(1)
    cam_obj = c.open_cameras["Cam1"]
    c.open_camera(1)

    # 같은 객체 유지(중복 생성/덮어쓰기 방지 확인)
    assert c.open_cameras["Cam1"] is cam_obj
    assert c.open_cameras["Cam1"].IsOpen() is True


def test_open_camera_missing_key_raises(tmp_path, monkeypatch):
    cfgp = tmp_path / "cams.json"
    _write_cams_json(cfgp)

    monkeypatch.setattr("gfa_controller.py.TlFactory", FakeTlFactory)
    monkeypatch.setattr("gfa_controller.py.DeviceInfo", FakeDeviceInfo)
    monkeypatch.setattr("gfa_controller.py.InstantCamera", FakeInstantCamera)

    c = GFAController(config=str(cfgp), logger=FakeLogger())

    with pytest.raises(KeyError):
        c.open_camera(99)


def test_open_selected_cameras_opens_multiple_and_skips_open(tmp_path, monkeypatch):
    cfgp = tmp_path / "cams.json"
    _write_cams_json(cfgp)

    monkeypatch.setattr("gfa_controller.py.TlFactory", FakeTlFactory)
    monkeypatch.setattr("gfa_controller.py.DeviceInfo", FakeDeviceInfo)
    monkeypatch.setattr("gfa_controller.py.InstantCamera", FakeInstantCamera)

    c = GFAController(config=str(cfgp), logger=FakeLogger())

    # 1번 먼저 열어두고, selected로 [1,2] 호출 시 1은 스킵되고 2만 새로 열려야 함
    c.open_camera(1)
    cam1_obj = c.open_cameras["Cam1"]

    c.open_selected_cameras([1, 2])

    assert c.open_cameras["Cam1"] is cam1_obj
    assert c.open_cameras["Cam2"].IsOpen() is True


def test_status_reports_open_state(tmp_path, monkeypatch):
    cfgp = tmp_path / "cams.json"
    _write_cams_json(cfgp)

    monkeypatch.setattr("gfa_controller.py.TlFactory", FakeTlFactory)
    monkeypatch.setattr("gfa_controller.py.DeviceInfo", FakeDeviceInfo)
    monkeypatch.setattr("gfa_controller.py.InstantCamera", FakeInstantCamera)

    c = GFAController(config=str(cfgp), logger=FakeLogger())

    # Cam1만 오픈
    c.open_camera(1)

    st = c.status()
    assert st["Cam1"] is True
    assert st["Cam2"] is False


def test_close_all_cameras_closes_and_clears(tmp_path, monkeypatch):
    cfgp = tmp_path / "cams.json"
    _write_cams_json(cfgp)

    monkeypatch.setattr("gfa_controller.py.TlFactory", FakeTlFactory)
    monkeypatch.setattr("gfa_controller.py.DeviceInfo", FakeDeviceInfo)
    monkeypatch.setattr("gfa_controller.py.InstantCamera", FakeInstantCamera)

    c = GFAController(config=str(cfgp), logger=FakeLogger())

    c.open_selected_cameras([1, 2])
    assert c.open_cameras["Cam1"].IsOpen() is True
    assert c.open_cameras["Cam2"].IsOpen() is True

    c.close_all_cameras()
    assert c.open_cameras == {}


def test_ping_calls_os_system_with_ip(tmp_path, monkeypatch):
    cfgp = tmp_path / "cams.json"
    _write_cams_json(cfgp)

    monkeypatch.setattr("gfa_controller.py.TlFactory", FakeTlFactory)
    monkeypatch.setattr("gfa_controller.py.DeviceInfo", FakeDeviceInfo)
    monkeypatch.setattr("gfa_controller.py.InstantCamera", FakeInstantCamera)

    c = GFAController(config=str(cfgp), logger=FakeLogger())

    called = {}
    def fake_system(cmd):
        called["cmd"] = cmd
        return 0

    monkeypatch.setattr("gfa_controller.os.system", fake_system)

    c.ping(CamNum=1)
    # 구현상 "ping -c 3 -w 1 {ip}"
    assert "1.1.1.1" in called["cmd"]
