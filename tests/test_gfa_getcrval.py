# tests/test_gfa_getcrval.py
import json
import math
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
        "settings": {
            "cpu": {"limit": 30}
        },
    }
    path.write_text(json.dumps(cfg), encoding="utf-8")


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


def test_load_config_from_path(tmp_path):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp)
    cfg = mod._load_config(cfgp)
    assert cfg["astrometry"]["scale_range"] == [0.1, 2.0]
    assert cfg["settings"]["cpu"]["limit"] == 30


def test_load_config_none_without_default_raises(monkeypatch):
    # 모듈 상단의 optional default 경로 함수가 없을 때는 ValueError가 정상
    monkeypatch.setattr(mod, "_get_default_config_path", None, raising=True)
    with pytest.raises(ValueError):
        mod._load_config(None)


def test_get_crval_from_image_happy_path(tmp_path, monkeypatch):
    # 입력 FITS (RA/DEC 필요)
    img = tmp_path / "img.fits"
    _write_fits(img, ra=10.0, dec=20.0)

    # config
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp)

    # work dir (명시하면 여기 아래에서 .new를 찾게 됨)
    work = tmp_path / "work"
    work.mkdir()

    # solve-field 존재한다고 가정
    monkeypatch.setattr(mod.shutil, "which", lambda name: r"C:\fake\solve-field.exe")

    # subprocess.run은 실제 실행 대신 성공했다고 가정
    calls = {}
    def _fake_run(cmd, shell, capture_output, text, check):
        calls["cmd"] = cmd
        return None
    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    # solve-field 결과 .new 파일을 만든 것처럼 준비
    solved = work / "img.new"
    _write_fits(solved, crval1=123.456, crval2=-78.9)

    # glob이 .new를 찾게 만들기
    monkeypatch.setattr(mod.glob, "glob", lambda pattern: [str(solved)])

    c1, c2 = mod.get_crval_from_image(
        image_path=img,
        config=cfgp,
        work_dir=work,
        keep_work_dir=True,   # 테스트에서 work_dir 삭제 방지
    )

    assert c1 == 123.456
    assert c2 == -78.9
    # 커맨드에 RA/DEC가 들어갔는지 최소 확인
    assert "--ra 10.0" in calls["cmd"]
    assert "--dec 20.0" in calls["cmd"]


def test_get_crval_from_image_missing_solve_field_raises(tmp_path, monkeypatch):
    img = tmp_path / "img.fits"
    _write_fits(img, ra=1.0, dec=2.0)
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp)

    monkeypatch.setattr(mod.shutil, "which", lambda name: None)
    with pytest.raises(FileNotFoundError):
        mod.get_crval_from_image(img, config=cfgp)


def test_get_crvals_from_images_preserves_order_and_nan(tmp_path, monkeypatch):
    paths = [tmp_path / f"i{i}.fits" for i in range(4)]
    for p in paths:
        p.write_text("dummy", encoding="utf-8")  # 실제로 안 열 거라 더미로 충분

    # 0,2는 성공 / 1,3은 실패하도록 mock
    def _fake_get_crval_from_image(p, config=None, logger=None, work_dir=None, keep_work_dir=False):
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
