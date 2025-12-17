# tests/test_gfa_img.py
import logging
import os

import numpy as np
import pytest
from astropy.io import fits

from gfa_img import GFAImage


@pytest.fixture
def logger():
    # 테스트용 로거 (caplog로 메시지 잡기 쉬움)
    lg = logging.getLogger("test_gfa_img")
    lg.setLevel(logging.DEBUG)
    return lg


def test_save_fits_writes_file_and_header(tmp_path, logger, caplog):
    caplog.set_level(logging.DEBUG)

    img = GFAImage(logger=logger)
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)

    # date/time을 고정값으로 넣어 flaky 방지
    img.save_fits(
        image_array=arr,
        filename="test_image",  # .fits 자동 추가되는지 확인
        exptime=1.23,
        telescope="KMTNET",
        instrument="KSPEC-GFA",
        observer="Mingyeong",
        object_name="M42",
        date_obs="2025-12-17",
        time_obs="12:34:56",
        ra=None,   # UNKNOWN 기본값 확인
        dec=None,  # UNKNOWN 기본값 확인
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

    # 저장 로그가 남는지(너무 빡세게 문자열 매칭할 필요는 없지만 간단히)
    assert any("successfully saved" in r.message.lower() for r in caplog.records)


def test_save_fits_replaces_colon_in_filename(tmp_path, logger):
    img = GFAImage(logger=logger)
    arr = np.zeros((2, 2), dtype=np.float32)

    img.save_fits(
        image_array=arr,
        filename="2025-12-17T12:34:56",  # ':' → '-' 치환 확인
        exptime=0.5,
        date_obs="2025-12-17",
        time_obs="12:34:56",
        output_directory=str(tmp_path),
    )

    # ":"가 "-"로 바뀌고 .fits가 붙어야 함
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


def test_save_fits_logs_warning_when_date_or_time_missing(tmp_path, logger, caplog):
    caplog.set_level(logging.WARNING)

    img = GFAImage(logger=logger)
    arr = np.zeros((1, 1), dtype=np.float32)

    img.save_fits(
        image_array=arr,
        filename="warn_case",
        exptime=1.0,
        # date_obs/time_obs 일부러 None
        output_directory=str(tmp_path),
    )

    # 경고 2개가 찍히는지 확인(문구는 변경될 수 있으니 핵심만 체크)
    messages = [r.message for r in caplog.records]
    assert any("no date_obs provided" in m.lower() for m in messages)
    assert any("no time_obs provided" in m.lower() for m in messages)


def test_save_fits_raises_when_cannot_create_directory(tmp_path, logger, monkeypatch):
    img = GFAImage(logger=logger)
    arr = np.zeros((2, 2), dtype=np.float32)

    bad_dir = tmp_path / "cannot_make"

    # os.path.exists가 False일 때 os.makedirs가 실패하도록 강제
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
