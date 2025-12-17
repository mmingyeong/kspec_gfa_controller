# tests/test_finder_actions.py
import os
import pytest


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


def test_init_uses_create_environment_when_env_none(monkeypatch):
    import finder_actions as fa

    calls = []

    def fake_create_environment(*, role):
        calls.append(role)
        return FakeEnv()

    monkeypatch.setattr(fa, "create_environment", fake_create_environment)

    act = fa.FinderGFAActions(env=None)
    assert isinstance(act.env, FakeEnv)
    assert act.cam_id == 7
    assert calls == ["finder"]


def test_generate_response(finder_actions):
    r = finder_actions._generate_response("success", "ok", a=1)
    assert r["status"] == "success"
    assert r["message"] == "ok"
    assert r["a"] == 1


@pytest.mark.asyncio
async def test_grab_success_no_timeout(finder_actions):
    finder_actions.env.controller._grabone_result = []

    r = await finder_actions.grab(
        ExpTime=1.2,
        Binning=1,
        packet_size=1500,
        cam_ipd=10,
        cam_ftd_base=1,
        ra="1",
        dec="2",
    )
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
async def test_grab_exception_returns_error(finder_actions):
    async def boom(**kwargs):
        raise RuntimeError("grab failed")

    finder_actions.env.controller.grabone = boom  # type: ignore

    r = await finder_actions.grab()
    assert r["status"] == "error"
    assert "grab failed" in r["message"].lower()


@pytest.mark.asyncio
async def test_guiding_success_no_save(monkeypatch, finder_actions):
    monkeypatch.setattr("finder_actions.os.makedirs", lambda *a, **k: None)
    monkeypatch.setattr("finder_actions.os.listdir", lambda p: [])

    r = await finder_actions.guiding(ExpTime=2.0, save=False, ra="1", dec="2")
    assert r["status"] == "success"
    assert r["fdx"] == 0
    assert r["fdy"] == 0
    assert r["fwhm"] == 0

    assert len(finder_actions.env.controller.grab_calls) == 1
    camnum, exptime, binning, kwargs = finder_actions.env.controller.grab_calls[0]
    assert camnum == 0
    assert exptime == 2.0
    assert binning == 4
    assert kwargs["ra"] == "1"
    assert kwargs["dec"] == "2"


@pytest.mark.asyncio
async def test_guiding_success_with_save_and_copy(monkeypatch, finder_actions):
    # save=True 분기 + isfile True/False 분기까지 커버 (Windows 경로도 안전하게)
    makedirs_calls = []

    def fake_makedirs(path, exist_ok=False):
        makedirs_calls.append((path, exist_ok))

    monkeypatch.setattr("finder_actions.os.makedirs", fake_makedirs)
    monkeypatch.setattr("finder_actions.os.listdir", lambda p: ["a.fits", "not_a_file"])

    def fake_isfile(p):
        return p.endswith("a.fits")

    monkeypatch.setattr("finder_actions.os.path.isfile", fake_isfile)

    copy_calls = []

    def fake_copy2(src, dst):
        copy_calls.append((src, dst))

    monkeypatch.setattr("finder_actions.shutil.copy2", fake_copy2)

    r = await finder_actions.guiding(ExpTime=1.5, save=True, ra="3", dec="4")
    assert r["status"] == "success"
    assert r["fdx"] == 0 and r["fdy"] == 0 and r["fwhm"] == 0

    assert len(finder_actions.env.controller.grab_calls) == 1
    camnum, exptime, binning, kwargs = finder_actions.env.controller.grab_calls[0]
    assert camnum == 0
    assert exptime == 1.5
    assert binning == 4
    assert kwargs["ra"] == "3"
    assert kwargs["dec"] == "4"

    # copy2는 a.fits만 복사됨
    assert len(copy_calls) == 1
    src, dst = copy_calls[0]

    # OS 상관없이 비교되도록 normalize
    src_norm = os.path.normpath(src)
    dst_norm = os.path.normpath(dst)

    assert src_norm.endswith(os.path.normpath(os.path.join("img", "raw", "a.fits")))
    assert os.path.normpath(os.path.join("img", "grab_finder")) in dst_norm
    assert dst_norm.endswith(os.path.normpath(os.path.join("a.fits")))

    # makedirs는 raw + grab 경로 2번 호출되는 게 정상
    assert len(makedirs_calls) >= 2


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


def test_shutdown_calls_env_shutdown_and_logs(finder_actions):
    finder_actions.shutdown()
    assert finder_actions.env.shutdown_called == 1
    assert any(
        lvl == "info" and "shutdown complete" in msg.lower()
        for (lvl, msg) in finder_actions.env.logger.logs
    )
