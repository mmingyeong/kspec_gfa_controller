# tests/test_finder_actions.py
import pytest


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
        self.status_called = 0
        self.ping_calls = []
        self.cam_params_calls = []

    async def grabone(self, **kwargs):
        self.grabone_calls.append(kwargs)
        return list(self._grabone_result)

    def grab(self, CamNum, ExpTime, Binning, **kwargs):
        self.grab_calls.append((CamNum, ExpTime, Binning, kwargs))
        return []

    def status(self):
        self.status_called += 1
        return {"Cam7": True}

    def ping(self, cam_id):
        self.ping_calls.append(cam_id)

    def cam_params(self, cam_id):
        self.cam_params_calls.append(cam_id)
        return {"mock": cam_id}


class FakeEnv:
    def __init__(self):
        self.logger = FakeLogger()
        self.controller = FakeController()
        self.shutdown_called = 0

    def shutdown(self):
        self.shutdown_called += 1


@pytest.fixture
def finder_actions():
    from finder_actions import FinderGFAActions
    return FinderGFAActions(env=FakeEnv())


def test_generate_response(finder_actions):
    r = finder_actions._generate_response("success", "ok", a=1)
    assert r["status"] == "success"
    assert r["message"] == "ok"
    assert r["a"] == 1


@pytest.mark.asyncio
async def test_grab_success_no_timeout(finder_actions):
    finder_actions.env.controller._grabone_result = []

    r = await finder_actions.grab(ExpTime=1.2, Binning=1, packet_size=1500, cam_ipd=10, cam_ftd_base=1, ra="1", dec="2")
    assert r["status"] == "success"
    assert "Cam7" in r["message"]

    assert len(finder_actions.env.controller.grabone_calls) == 1
    kwargs = finder_actions.env.controller.grabone_calls[0]
    assert kwargs["CamNum"] == 7
    assert kwargs["ExpTime"] == 1.2
    assert kwargs["Binning"] == 1
    assert kwargs["packet_size"] == 1500
    assert kwargs["ipd"] == 10
    assert kwargs["ftd_base"] == 1
    assert kwargs["ra"] == "1"
    assert kwargs["dec"] == "2"
    assert "output_dir" in kwargs


@pytest.mark.asyncio
async def test_grab_success_with_timeout_message(finder_actions):
    finder_actions.env.controller._grabone_result = [7]
    r = await finder_actions.grab()
    assert r["status"] == "success"
    assert "timeout" in r["message"].lower()


@pytest.mark.asyncio
async def test_grab_exception_returns_error(finder_actions, monkeypatch):
    async def boom(**kwargs):
        raise RuntimeError("grab failed")

    finder_actions.env.controller.grabone = boom  # type: ignore

    r = await finder_actions.grab()
    assert r["status"] == "error"
    assert "grab failed" in r["message"].lower()


@pytest.mark.asyncio
async def test_guiding_success_no_save(monkeypatch, finder_actions):
    # 파일 I/O 최소화
    monkeypatch.setattr("finder_actions.os.makedirs", lambda *a, **k: None)
    monkeypatch.setattr("finder_actions.os.listdir", lambda p: [])

    r = await finder_actions.guiding(ExpTime=2.0, save=False, ra="1", dec="2")
    assert r["status"] == "success"
    assert r["fdx"] == 0
    assert r["fdy"] == 0
    assert r["fwhm"] == 0

    # controller.grab 호출 확인 (코드가 0, ExpTime, 4로 고정 호출함)
    assert len(finder_actions.env.controller.grab_calls) == 1
    camnum, exptime, binning, kwargs = finder_actions.env.controller.grab_calls[0]
    assert camnum == 0
    assert exptime == 2.0
    assert binning == 4
    assert kwargs["ra"] == "1"
    assert kwargs["dec"] == "2"


@pytest.mark.asyncio
async def test_guiding_exception_returns_error(monkeypatch, finder_actions):
    def boom(*a, **k):
        raise RuntimeError("mkdir failed")

    monkeypatch.setattr("finder_actions.os.makedirs", boom)

    r = await finder_actions.guiding()
    assert r["status"] == "error"
    assert "guiding failed" in r["message"].lower()


def test_status_success(finder_actions):
    r = finder_actions.status()
    assert r["status"] == "success"
    assert isinstance(r["message"], dict)
    assert "Cam7" in r["message"]


def test_status_error(finder_actions):
    def boom():
        raise RuntimeError("status failed")

    finder_actions.env.controller.status = boom  # type: ignore
    r = finder_actions.status()
    assert r["status"] == "error"


def test_ping_success(finder_actions):
    r = finder_actions.ping()
    assert r["status"] == "success"
    assert finder_actions.env.controller.ping_calls == [7]


def test_ping_error(finder_actions):
    def boom(cam_id):
        raise RuntimeError("ping failed")

    finder_actions.env.controller.ping = boom  # type: ignore
    r = finder_actions.ping()
    assert r["status"] == "error"


def test_cam_params_success(finder_actions):
    r = finder_actions.cam_params()
    assert r["status"] == "success"
    assert "Cam7" in r["message"]
    assert finder_actions.env.controller.cam_params_calls == [7]


def test_cam_params_error(finder_actions):
    def boom(cam_id):
        raise RuntimeError("params failed")

    finder_actions.env.controller.cam_params = boom  # type: ignore
    r = finder_actions.cam_params()
    assert r["status"] == "error"


def test_shutdown_calls_env_shutdown(finder_actions):
    finder_actions.shutdown()
    assert finder_actions.env.shutdown_called == 1
