# tests/test_gfa_img.py
import logging
import os
from datetime import datetime

import numpy as np
import pytest
from astropy.io import fits

from gfa_img import GFAImage


@pytest.fixture
def logger():
    lg = logging.getLogger("test_gfa_img")
    lg.setLevel(logging.DEBUG)
    return lg


def test_save_fits_writes_file_and_header(tmp_path, logger, caplog):
    caplog.set_level(logging.DEBUG)

    img = GFAImage(logger=logger)
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)

    img.save_fits(
        image_array=arr,
        filename="test_image",  # .fits 자동 추가
        exptime=1.23,
        telescope="KMTNET",
        instrument="KSPEC-GFA",
        observer="Mingyeong",
        object_name="M42",
        date_obs="2025-12-17",
        time_obs="12:34:56",
        ra=None,  # UNKNOWN 기본값
        dec=None,  # UNKNOWN 기본값
        output_directory=str(tmp_path),
    )

    out = tmp_path / "test_image.fits"
    assert out.exists()

    with fits.open(out) as hdul:
        data = hdul[0].data
        hdr = hdul[0].header

    assert data.shape == (3, 4)
    assert np.allclose(data, arr)

    # 핵심 헤더 필드 검증
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

    # 저장 성공 로그 확인
    assert any("successfully saved" in r.message.lower() for r in caplog.records)


def test_save_fits_replaces_colon_in_filename(tmp_path, logger):
    img = GFAImage(logger=logger)
    arr = np.zeros((2, 2), dtype=np.float32)

    img.save_fits(
        image_array=arr,
        filename="2025-12-17T12:34:56",  # ':' -> '-'
        exptime=0.5,
        date_obs="2025-12-17",
        time_obs="12:34:56",
        output_directory=str(tmp_path),
    )

    out = tmp_path / "2025-12-17T12-34-56.fits"
    assert out.exists()


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


def test_save_fits_logs_warning_when_date_or_time_missing(
    tmp_path, logger, caplog, monkeypatch
):
    """
    date_obs/time_obs=None이면 warning 2개 찍히는 분기 타기.
    datetime.now()는 고정해서 flaky 방지.
    """
    caplog.set_level(logging.WARNING)

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2025, 12, 17, 12, 34, 56)

    # gfa_img 모듈 내부에서 "from datetime import datetime"로 import했으므로
    # gfa_img.datetime을 바꿔야 함
    monkeypatch.setattr("gfa_img.datetime", FixedDatetime)

    img = GFAImage(logger=logger)
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

    # 고정된 now() 값이 실제로 헤더에 들어갔는지까지 확인
    out = tmp_path / "warn_case.fits"
    with fits.open(out) as hdul:
        hdr = hdul[0].header
    assert hdr["DATE-OBS"] == "2025-12-17"
    assert hdr["TIME-OBS"] == "12:34:56"


def test_save_fits_uses_cwd_when_output_directory_none(tmp_path, logger, monkeypatch):
    """
    output_directory=None -> os.getcwd() 쓰는 분기 타기.
    """
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
    """
    output_directory가 이미 존재하면 os.makedirs 분기 안 타야 함.
    """
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
        output_directory=str(tmp_path),  # 이미 존재
    )

    assert called["n"] == 0
    assert (tmp_path / "exists_dir.fits").exists()


def test_save_fits_raises_when_cannot_create_directory(
    tmp_path, logger, monkeypatch, caplog
):
    """
    output_directory가 없고 os.makedirs가 실패하는 에러 분기 타기.
    """
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

    # 에러 로그도 남는지 확인
    assert any("error creating directory" in r.message.lower() for r in caplog.records)


def test_save_fits_when_filename_already_has_extension(tmp_path, logger):
    """
    filename이 이미 .fits면 중복으로 .fits 안 붙는지.
    """
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
