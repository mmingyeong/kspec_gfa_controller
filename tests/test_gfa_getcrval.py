# tests/test_gfa_getcrval.py
import json
import logging
import math
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

import kspec_gfa_controller.gfa_getcrval as mod


def _plain_test_logger() -> logging.Logger:
    """
    테스트 전용 로거.
    gfa_getcrval 내부의 GFALogger/adapter 로직과 충돌을 피하려고
    테스트에서는 _get_logger를 통째로 패치해서 이 로거만 쓰게 한다.
    """
    lg = logging.getLogger("test_gfa_getcrval_plain")
    lg.handlers.clear()
    lg.addHandler(logging.NullHandler())
    lg.propagate = False
    lg.setLevel(logging.DEBUG)
    return lg


@pytest.fixture
def lg(monkeypatch):
    """
    모든 테스트에서 gfa_getcrval._get_logger를 패치해서
    JobAdapter/LogRecordFactory 관련 부작용을 차단.
    """
    logger = _plain_test_logger()
    monkeypatch.setattr(mod, "_get_logger", lambda _logger=None: logger, raising=True)
    return logger


def _write_fits(path: Path, ra=None, dec=None, crval1=None, crval2=None):
    hdr = fits.Header()
    if ra is not None:
        hdr["RA"] = ra
    if dec is not None:
        hdr["DEC"] = dec
    if crval1 is not None:
        hdr["CRVAL1"] = crval1
    if crval2 is not None:
        hdr["CRVAL2"] = crval2

    data = np.zeros((2, 3), dtype=np.float32)
    fits.PrimaryHDU(data=data, header=hdr).writeto(path, overwrite=True)


def _write_config(path: Path):
    cfg = {
        "astrometry": {
            "scale_range": [0.1, 2.0],
            "radius": 1.0,
        },
        "settings": {"cpu": {"limit": 30}},
    }
    path.write_text(json.dumps(cfg), encoding="utf-8")


# -------------------------
# _get_logger()
# -------------------------
def test_get_logger_returns_plain_logger_via_patch(lg):
    out = mod._get_logger(None)
    assert out is lg


# -------------------------
# _read_ra_dec()
# -------------------------
def test_read_ra_dec_float_and_string(tmp_path, lg):
    f1 = tmp_path / "a.fits"
    _write_fits(f1, ra=12.34, dec=-56.78)

    ra, dec = mod._read_ra_dec(f1, lg)
    assert ra == "12.34"
    assert dec == "-56.78"

    f2 = tmp_path / "b.fits"
    _write_fits(f2, ra="123.0", dec="45.6")

    ra, dec = mod._read_ra_dec(f2, lg)
    assert ra == "123.0"
    assert dec == "45.6"


def test_read_ra_dec_missing_raises(tmp_path, lg):
    f = tmp_path / "no_radec.fits"
    _write_fits(f)
    with pytest.raises(ValueError):
        mod._read_ra_dec(f, lg)


# -------------------------
# _load_config()
# -------------------------
def test_load_config_from_path(tmp_path, lg):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp)

    cfg = mod._load_config(cfgp, lg)
    assert cfg["astrometry"]["scale_range"] == [0.1, 2.0]
    assert cfg["settings"]["cpu"]["limit"] == 30


def test_load_config_none_without_default_raises(monkeypatch, lg):
    """
    현재 구현은 config=None이면 _get_default_config_path()를 호출한다.
    _get_default_config_path 자체를 None으로 만들면 호출 시 TypeError가 나는 것이 정상.
    """
    monkeypatch.setattr(mod, "_get_default_config_path", None, raising=True)
    with pytest.raises(TypeError):
        mod._load_config(None, lg)


def test_load_config_none_uses_default_path(monkeypatch, tmp_path, lg):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp)

    monkeypatch.setattr(mod, "_get_default_config_path", lambda: str(cfgp), raising=True)
    cfg = mod._load_config(None, lg)
    assert cfg["settings"]["cpu"]["limit"] == 30


# -------------------------
# get_crval_from_image()
# -------------------------
def _patch_solve_field_ok(monkeypatch):
    """
    gfa_getcrval은 solve-field 경로를 Path.exists + os.access로 검사한다.
    테스트에서는 이 검사를 우회한다.
    """
    monkeypatch.setattr(mod.Path, "exists", lambda self: True, raising=False)
    monkeypatch.setattr(mod.os, "access", lambda p, m: True, raising=True)


def test_get_crval_from_image_input_missing_raises(tmp_path, monkeypatch, lg):
    _patch_solve_field_ok(monkeypatch)

    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp)

    with pytest.raises(FileNotFoundError):
        mod.get_crval_from_image(tmp_path / "nope.fits", config=cfgp, logger=lg)


def test_get_crval_from_image_subprocess_error_becomes_runtimeerror_and_keeps_persistent_dir(
    tmp_path, monkeypatch, lg
):
    """
    현재 구현은 work_dir=None이면 DEFAULT_RES_ROOT 아래에 per-job persistent work_dir을 만들고
    실패해도 기본적으로 삭제하지 않는다(tmp_created=True).
    따라서 이 테스트는 rmtree 호출을 기대하지 않고, RuntimeError만 확인한다.
    """
    _patch_solve_field_ok(monkeypatch)

    img = tmp_path / "img.fits"
    _write_fits(img, ra=10.0, dec=20.0)
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp)

    class R:
        returncode = 1
        stdout = ""
        stderr = "ERR!"

    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: R(), raising=True)

    # DEFAULT_RES_ROOT를 tmp_path 아래로 바꿔서 테스트가 시스템 경로를 건드리지 않게
    res_root = tmp_path / "res_root"
    monkeypatch.setattr(mod, "DEFAULT_RES_ROOT", res_root, raising=True)

    with pytest.raises(RuntimeError):
        mod.get_crval_from_image(img, config=cfgp, logger=lg, work_dir=None, keep_work_dir=False)

    # 실패했어도 persistent root는 생성되어 남아있을 수 있음(설계)
    assert res_root.exists()


def test_get_crval_from_image_stem_fallback_glob(tmp_path, monkeypatch, lg):
    _patch_solve_field_ok(monkeypatch)

    img = tmp_path / "img.fits"
    _write_fits(img, ra=10.0, dec=20.0)
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp)

    work = tmp_path / "work"
    work.mkdir()

    class R:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: R(), raising=True)

    solved = work / "img.new"
    _write_fits(solved, crval1=1.1, crval2=2.2)

    def fake_glob(pattern):
        s = str(pattern)
        if s.endswith("/img.new") or s.endswith("\\img.new"):
            return []
        return [str(solved)]

    monkeypatch.setattr(mod.glob, "glob", fake_glob, raising=True)

    c1, c2 = mod.get_crval_from_image(
        img, config=cfgp, logger=lg, work_dir=work, keep_work_dir=True
    )
    assert c1 == 1.1
    assert c2 == 2.2


def test_get_crval_from_image_new_file_missing_lists_dir(tmp_path, monkeypatch, lg):
    _patch_solve_field_ok(monkeypatch)

    img = tmp_path / "img.fits"
    _write_fits(img, ra=10.0, dec=20.0)
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp)

    work = tmp_path / "work"
    work.mkdir()
    (work / "dummy.txt").write_text("x", encoding="utf-8")

    class R:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: R(), raising=True)
    monkeypatch.setattr(mod.glob, "glob", lambda pattern: [], raising=True)

    with pytest.raises(FileNotFoundError) as e:
        mod.get_crval_from_image(img, config=cfgp, logger=lg, work_dir=work, keep_work_dir=True)

    assert "Files=" in str(e.value)
    assert "dummy.txt" in str(e.value)


def test_get_crval_from_image_happy_path_and_cleanup_when_keep_false(tmp_path, monkeypatch, lg):
    """
    caller가 work_dir를 명시적으로 준 경우(tmp_created=False),
    keep_work_dir=False면 cleanup(rmtree) 호출이 일어난다.
    """
    _patch_solve_field_ok(monkeypatch)

    img = tmp_path / "img.fits"
    _write_fits(img, ra=10.0, dec=20.0)
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp)

    work = tmp_path / "work"
    work.mkdir()

    class R:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: R(), raising=True)

    solved = work / "img.new"
    _write_fits(solved, crval1=123.456, crval2=-78.9)
    monkeypatch.setattr(mod.glob, "glob", lambda pattern: [str(solved)], raising=True)

    called = {"rm": 0}
    monkeypatch.setattr(
        mod.shutil,
        "rmtree",
        lambda p, ignore_errors=True: called.__setitem__("rm", called["rm"] + 1),
        raising=True,
    )

    c1, c2 = mod.get_crval_from_image(
        img, config=cfgp, logger=lg, work_dir=work, keep_work_dir=False
    )
    assert c1 == 123.456
    assert c2 == -78.9
    assert called["rm"] == 1


# -------------------------
# get_crvals_from_images()
# -------------------------
def test_get_crvals_from_images_preserves_order_and_nan(tmp_path, monkeypatch, lg):
    paths = [tmp_path / f"i{i}.fits" for i in range(4)]
    for p in paths:
        p.write_text("dummy", encoding="utf-8")

    def _fake_get_crval_from_image(
        p, config=None, logger=None, work_dir=None, keep_work_dir=False, solve_field=None, subprocess_env=None
    ):
        name = Path(p).name
        if name in ("i1.fits", "i3.fits"):
            raise RuntimeError("boom")
        idx = int(name[1])  # i0.fits -> 0
        return (100.0 + idx, 200.0 + idx)

    monkeypatch.setattr(mod, "get_crval_from_image", _fake_get_crval_from_image, raising=True)

    cr1, cr2 = mod.get_crvals_from_images(paths, config=None, logger=lg, max_workers=2)

    assert cr1[0] == 100.0 and cr2[0] == 200.0
    assert math.isnan(cr1[1]) and math.isnan(cr2[1])
    assert cr1[2] == 102.0 and cr2[2] == 202.0
    assert math.isnan(cr1[3]) and math.isnan(cr2[3])
