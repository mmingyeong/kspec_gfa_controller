# tests/test_gfa_img.py
import logging
import os
import warnings
from datetime import datetime
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

# (환경에 warning->error 설정이 있으면 수집 단계에서 죽을 수 있어 방어)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def _find_gfa_img_py() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        repo_root / "src" / "kspec_gfa_controller" / "gfa_img.py",   # src layout
        repo_root / "kspec_gfa_controller" / "gfa_img.py",          # non-src layout
    ]
    for p in candidates:
        if p.exists():
            return p
    raise RuntimeError("gfa_img.py not found. tried:\n" + "\n".join(map(str, candidates)))


def _load_gfa_image_class():
    """
    kspec_gfa_controller 패키지 import를 피하고 gfa_img.py만 직접 로딩한다.
    -> __init__.py / gfa_actions / gfa_guider / scipy.optimize import 체인 회피
    """
    path = _find_gfa_img_py()
    spec = spec_from_file_location("_test_gfa_img_module", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for: {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GFAImage


GFAImage = _load_gfa_image_class()


@pytest.fixture
def logger():
    lg = logging.getLogger("test_gfa_img")
    lg.setLevel(logging.DEBUG)
    return lg


def test_save_fits_writes_file_and_header(tmp_path, logger, caplog):
    caplog.set_level(logging.DEBUG)

    img = GFAImage(logger=logger)
    arr = (np.arange(12, dtype=np.float32).reshape(3, 4) / 100.0)

    img.save_fits(
        image_array=arr,
        filename="test_image",
        exptime=1.23,
        telescope="KMTNET",
        instrument="KSPEC-GFA",
        observer="Mingyeong",
        object_name="M42",
        date_obs="2025-12-17",
        time_obs="12:34:56",
        ra=None,
        dec=None,
        output_directory=str(tmp_path),
    )

    out = tmp_path / "test_image.fits"
    assert out.exists()

    with fits.open(out) as hdul:
        data = hdul[0].data
        hdr = hdul[0].header

    assert data.shape == (3, 4)
    assert np.isfinite(data).all()

    assert hdr["NAXIS"] == 2
    assert hdr["NAXIS1"] == 4
    assert hdr["NAXIS2"] == 3
    assert hdr["TELESCOP"] == "KMTNET"
    assert hdr["INSTRUME"] == "KSPEC-GFA"
    assert hdr["OBSERVER"] == "Mingyeong"
    assert hdr["OBJECT"] == "M42"
    assert hdr["DATE-OBS"] == "2025-12-17"
    assert hdr["TIME-OBS"] == "12:34:56"
    assert hdr["RA"] == "UNKNOWN"
    assert hdr["DEC"] == "UNKNOWN"
    assert hdr["EXPTIME"] == 1.23

    assert any("successfully saved" in r.message.lower() for r in caplog.records)


def test_save_fits_replaces_colon_in_filename(tmp_path, logger):
    img = GFAImage(logger=logger)
    arr = np.zeros((2, 2), dtype=np.float32)

    img.save_fits(
        image_array=arr,
        filename="2025-12-17T12:34:56",
        exptime=0.5,
        date_obs="2025-12-17",
        time_obs="12:34:56",
        output_directory=str(tmp_path),
    )

    assert (tmp_path / "2025-12-17T12-34-56.fits").exists()


def test_save_fits_creates_output_directory(tmp_path, logger):
    img = GFAImage(logger=logger)
    arr = np.ones((2, 3), dtype=np.float32)

    outdir = tmp_path / "new_dir"
    assert not outdir.exists()

    img.save_fits(
        image_array=arr,
        filename="abc",
        exptime=1.0,
        date_obs="2025-12-17",
        time_obs="00:00:00",
        output_directory=str(outdir),
    )

    assert outdir.exists()
    assert (outdir / "abc.fits").exists()


def test_save_fits_logs_warning_when_date_or_time_missing(tmp_path, logger, caplog, monkeypatch):
    caplog.set_level(logging.WARNING)

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2025, 12, 17, 12, 34, 56)

    # gfa_img.py 내부가 "from datetime import datetime" 이므로 모듈 심볼 patch
    # (직접 로딩 모듈 이름이 "_test_gfa_img_module"이라서 이 방식은 안 통함)
    # => 따라서 메서드가 참조하는 datetime 심볼을 인스턴스가 가진 모듈에서 직접 바꿔야 함
    module_path = _find_gfa_img_py()
    spec = spec_from_file_location("_test_gfa_img_module_patch", str(module_path))
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "datetime", FixedDatetime, raising=True)

    img = module.GFAImage(logger=logger)
    arr = np.zeros((1, 1), dtype=np.float32)

    img.save_fits(
        image_array=arr,
        filename="warn_case",
        exptime=1.0,
        date_obs=None,
        time_obs=None,
        output_directory=str(tmp_path),
    )

    msgs = [r.message.lower() for r in caplog.records]
    assert any("no date_obs provided" in m for m in msgs)
    assert any("no time_obs provided" in m for m in msgs)

    out = tmp_path / "warn_case.fits"
    with fits.open(out) as hdul:
        hdr = hdul[0].header
    assert hdr["DATE-OBS"] == "2025-12-17"
    assert hdr["TIME-OBS"] == "12:34:56"


def test_save_fits_uses_cwd_when_output_directory_none(tmp_path, logger, monkeypatch):
    img = GFAImage(logger=logger)
    arr = np.zeros((2, 2), dtype=np.float32)

    monkeypatch.setattr(os, "getcwd", lambda: str(tmp_path))

    img.save_fits(
        image_array=arr,
        filename="cwd_case",
        exptime=1.0,
        date_obs="2025-12-17",
        time_obs="00:00:00",
        output_directory=None,
    )

    assert (tmp_path / "cwd_case.fits").exists()


def test_save_fits_does_not_call_makedirs_if_dir_exists(tmp_path, logger, monkeypatch):
    img = GFAImage(logger=logger)
    arr = np.zeros((2, 2), dtype=np.float32)

    called = {"n": 0}

    def _boom(*args, **kwargs):
        called["n"] += 1
        raise AssertionError("os.makedirs should not be called when dir exists")

    monkeypatch.setattr(os, "makedirs", _boom)

    img.save_fits(
        image_array=arr,
        filename="exists_dir",
        exptime=1.0,
        date_obs="2025-12-17",
        time_obs="00:00:00",
        output_directory=str(tmp_path),
    )

    assert called["n"] == 0
    assert (tmp_path / "exists_dir.fits").exists()


def test_save_fits_raises_when_cannot_create_directory(tmp_path, logger, monkeypatch, caplog):
    caplog.set_level(logging.ERROR)

    img = GFAImage(logger=logger)
    arr = np.zeros((2, 2), dtype=np.float32)
    bad_dir = tmp_path / "cannot_make"
    assert not bad_dir.exists()

    def _raise(*args, **kwargs):
        raise OSError("no permission")

    monkeypatch.setattr(os, "makedirs", _raise)

    with pytest.raises(OSError):
        img.save_fits(
            image_array=arr,
            filename="x",
            exptime=1.0,
            output_directory=str(bad_dir),
        )

    assert any("error creating directory" in r.message.lower() for r in caplog.records)


def test_save_fits_when_filename_already_has_extension(tmp_path, logger):
    img = GFAImage(logger=logger)
    arr = np.zeros((1, 2), dtype=np.float32)

    img.save_fits(
        image_array=arr,
        filename="already.fits",
        exptime=1.0,
        date_obs="2025-12-17",
        time_obs="00:00:00",
        output_directory=str(tmp_path),
    )

    assert (tmp_path / "already.fits").exists()
