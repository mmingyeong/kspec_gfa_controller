# tests/test_gfa_getcrval.py
import json
import logging
import math
import subprocess
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

import gfa_getcrval as mod


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
def test_get_logger_fallback_minimal_when_no_default(monkeypatch):
    # default logger 함수가 없을 때 fallback logger 분기 태우기
    monkeypatch.setattr(mod, "_get_default_logger", None, raising=True)

    lg = mod._get_logger(None)
    assert isinstance(lg, logging.Logger)
    assert lg.name == "gfa_getcrval"


def test_get_logger_returns_passed_logger():
    lg0 = logging.getLogger("custom_logger")
    lg = mod._get_logger(lg0)
    assert lg is lg0


# -------------------------
# _read_ra_dec()
# -------------------------
def test_read_ra_dec_float_and_string(tmp_path):
    f1 = tmp_path / "a.fits"
    _write_fits(f1, ra=12.34, dec=-56.78)
    ra, dec = mod._read_ra_dec(f1)
    assert ra == 12.34
    assert dec == -56.78

    f2 = tmp_path / "b.fits"
    _write_fits(f2, ra="123.0", dec="45.6")
    ra, dec = mod._read_ra_dec(f2)
    assert ra == 123.0
    assert dec == 45.6


def test_read_ra_dec_missing_raises(tmp_path):
    f = tmp_path / "no_radec.fits"
    _write_fits(f)  # RA/DEC 없음
    with pytest.raises(ValueError):
        mod._read_ra_dec(f)


def test_read_ra_dec_not_convertible_raises(tmp_path):
    f = tmp_path / "bad_radec.fits"
    _write_fits(f, ra="abc", dec="def")
    with pytest.raises(ValueError):
        mod._read_ra_dec(f)


# -------------------------
# _load_config()
# -------------------------
def test_load_config_from_path(tmp_path):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp)
    cfg = mod._load_config(cfgp)
    assert cfg["astrometry"]["scale_range"] == [0.1, 2.0]
    assert cfg["settings"]["cpu"]["limit"] == 30


def test_load_config_none_without_default_raises(monkeypatch):
    monkeypatch.setattr(mod, "_get_default_config_path", None, raising=True)
    with pytest.raises(ValueError):
        mod._load_config(None)


def test_load_config_none_uses_default_path(monkeypatch, tmp_path):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp)

    monkeypatch.setattr(
        mod, "_get_default_config_path", lambda: str(cfgp), raising=True
    )
    cfg = mod._load_config(None)
    assert cfg["settings"]["cpu"]["limit"] == 30


# -------------------------
# get_crval_from_image()
# -------------------------
def test_get_crval_from_image_input_missing_raises(tmp_path):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp)

    with pytest.raises(FileNotFoundError):
        mod.get_crval_from_image(tmp_path / "nope.fits", config=cfgp)


def test_get_crval_from_image_missing_solve_field_raises(tmp_path, monkeypatch):
    img = tmp_path / "img.fits"
    _write_fits(img, ra=1.0, dec=2.0)
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp)

    monkeypatch.setattr(mod.shutil, "which", lambda name: None)
    with pytest.raises(FileNotFoundError):
        mod.get_crval_from_image(img, config=cfgp)


def test_get_crval_from_image_subprocess_error_becomes_runtimeerror(
    tmp_path, monkeypatch
):
    img = tmp_path / "img.fits"
    _write_fits(img, ra=10.0, dec=20.0)
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp)

    monkeypatch.setattr(mod.shutil, "which", lambda name: r"C:\fake\solve-field.exe")

    def boom_run(cmd, shell, capture_output, text, check):
        raise subprocess.CalledProcessError(returncode=1, cmd=cmd, stderr="ERR!")

    monkeypatch.setattr(mod.subprocess, "run", boom_run)

    # work_dir=None 경로도 같이 태우기 위해 tempfile.mkdtemp를 고정 경로로
    work = tmp_path / "tmpwork"
    work.mkdir()
    monkeypatch.setattr(mod.tempfile, "mkdtemp", lambda prefix: str(work))

    # cleanup 호출 확인
    called = {"rm": 0}
    monkeypatch.setattr(
        mod.shutil,
        "rmtree",
        lambda p, ignore_errors=True: called.__setitem__("rm", called["rm"] + 1),
    )

    with pytest.raises(RuntimeError):
        mod.get_crval_from_image(img, config=cfgp, work_dir=None, keep_work_dir=False)

    assert called["rm"] == 1


def test_get_crval_from_image_stem_fallback_glob(tmp_path, monkeypatch):
    # new_pat은 실패, stem 기반 glob은 성공하도록
    img = tmp_path / "img.fits"
    _write_fits(img, ra=10.0, dec=20.0)
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp)

    work = tmp_path / "work"
    work.mkdir()

    monkeypatch.setattr(mod.shutil, "which", lambda name: r"C:\fake\solve-field.exe")
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: None)

    solved = work / "img.new"
    _write_fits(solved, crval1=1.1, crval2=2.2)

    def fake_glob(pattern):
        # 첫 번째 pattern(new_pat)은 빈 리스트, 두 번째(stem.new)는 성공
        if str(pattern).endswith("img.new") and "img.fits" in str(pattern):
            return []
        return [str(solved)]

    monkeypatch.setattr(mod.glob, "glob", fake_glob)

    c1, c2 = mod.get_crval_from_image(
        img, config=cfgp, work_dir=work, keep_work_dir=True
    )
    assert c1 == 1.1
    assert c2 == 2.2


def test_get_crval_from_image_new_file_missing_lists_dir(tmp_path, monkeypatch):
    img = tmp_path / "img.fits"
    _write_fits(img, ra=10.0, dec=20.0)
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp)

    work = tmp_path / "work"
    work.mkdir()
    # work 안에 아무 파일이나 하나 넣고 listing 문자열이 만들어지는 분기 태움
    (work / "dummy.txt").write_text("x", encoding="utf-8")

    monkeypatch.setattr(mod.shutil, "which", lambda name: r"C:\fake\solve-field.exe")
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(mod.glob, "glob", lambda pattern: [])  # 둘 다 실패

    with pytest.raises(FileNotFoundError) as e:
        mod.get_crval_from_image(img, config=cfgp, work_dir=work, keep_work_dir=True)

    assert "Files:" in str(e.value)
    assert "dummy.txt" in str(e.value)


def test_get_crval_from_image_happy_path_and_cleanup_when_keep_false(
    tmp_path, monkeypatch
):
    # work_dir를 명시해도 keep_work_dir=False면 cleanup(rmtree) 수행하는 코드 경로 커버
    img = tmp_path / "img.fits"
    _write_fits(img, ra=10.0, dec=20.0)
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp)

    work = tmp_path / "work"
    work.mkdir()

    monkeypatch.setattr(mod.shutil, "which", lambda name: r"C:\fake\solve-field.exe")
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: None)

    solved = work / "img.new"
    _write_fits(solved, crval1=123.456, crval2=-78.9)
    monkeypatch.setattr(mod.glob, "glob", lambda pattern: [str(solved)])

    called = {"rm": 0}
    monkeypatch.setattr(
        mod.shutil,
        "rmtree",
        lambda p, ignore_errors=True: called.__setitem__("rm", called["rm"] + 1),
    )

    c1, c2 = mod.get_crval_from_image(
        img, config=cfgp, work_dir=work, keep_work_dir=False
    )
    assert c1 == 123.456
    assert c2 == -78.9
    assert called["rm"] == 1


# -------------------------
# get_crvals_from_images()
# -------------------------
def test_get_crvals_from_images_preserves_order_and_nan(tmp_path, monkeypatch):
    paths = [tmp_path / f"i{i}.fits" for i in range(4)]
    for p in paths:
        p.write_text("dummy", encoding="utf-8")  # 실제로 안 열 거라 더미로 충분

    def _fake_get_crval_from_image(
        p, config=None, logger=None, work_dir=None, keep_work_dir=False
    ):
        name = Path(p).name
        if name in ("i1.fits", "i3.fits"):
            raise RuntimeError("boom")
        idx = int(name[1])  # i0.fits -> 0
        return (100.0 + idx, 200.0 + idx)

    monkeypatch.setattr(mod, "get_crval_from_image", _fake_get_crval_from_image)

    cr1, cr2 = mod.get_crvals_from_images(paths, max_workers=2)

    assert cr1[0] == 100.0 and cr2[0] == 200.0
    assert math.isnan(cr1[1]) and math.isnan(cr2[1])
    assert cr1[2] == 102.0 and cr2[2] == 202.0
    assert math.isnan(cr1[3]) and math.isnan(cr2[3])