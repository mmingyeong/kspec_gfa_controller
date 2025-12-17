# tests/test_gfa_guider.py
import json
import math
from pathlib import Path

import numpy as np
import pytest

# gfa_guider 모듈 자체가 scipy/photutils를 import하므로,
# 해당 의존성이 없으면 테스트를 skip하도록 처리
pytest.importorskip("scipy")
pytest.importorskip("photutils")
pytest.importorskip("astropy")

from gfa_guider import GFAGuider


def _write_config(path: Path, tmp_path: Path):
    # gfa_guider는 base_dir + dirs[...]로 경로를 만들기 때문에
    # 여기선 상대/절대 상관없이 "존재 가능한 형태"로만 맞추면 됨
    cfg = {
        "paths": {
            "directories": {
                "processed_images": str(tmp_path / "processed"),
                "final_astrometry_images": str(tmp_path / "final"),
                "cutout_directory": str(tmp_path / "cutout"),
            }
        },
        "detection": {
            "box_size": 20,
            "criteria": {"critical_outlier": 0.5},
            "peak_detection": {"max": 30000, "min": 10},
        },
        "catalog_matching": {
            "tolerance": {"angular_distance": 1.0, "mag_flux_min": 0.1},
            "fields": {"ra_column": "RA", "dec_column": "DEC", "mag_flux": "FLUX"},
        },
        "settings": {"image_processing": {"pixel_scale": 0.4}},
    }
    path.write_text(json.dumps(cfg), encoding="utf-8")


def test_background_returns_subtracted_and_stddev(tmp_path):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    g = GFAGuider(config=str(cfgp))

    # 600x1024로 만들어서 x_split=511 양쪽 모두 존재하게
    img = np.zeros((600, 1024), dtype=np.float32)
    img[:, :511] = 100.0
    img[:, 511:] = 200.0
    # 약간의 noise
    img[0, 0] = 101.0
    img[0, 511] = 199.0

    bg_sub, stddev = g.background(img)

    # 좌우 평균 제거 후, 평균이 대략 0 근처여야 함
    assert abs(float(np.mean(bg_sub[:, :511]))) < 1.0
    assert abs(float(np.mean(bg_sub[:, 511:]))) < 1.0
    assert stddev >= 0.0


def test_select_stars_filters_by_angle_and_flux(tmp_path):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    g = GFAGuider(config=str(cfgp))

    # 기준점: (0,0) radians
    ra1_rad, dec1_rad = 0.0, 0.0

    # 후보 3개
    ra_p = np.array([0.0, 10.0, 0.1])   # deg
    dec_p = np.array([0.0, 0.0, 0.1])   # deg
    flux = np.array([1.0, 1.0, np.nan]) # nan -> 0 처리됨

    ra2_rad = np.radians(ra_p)
    dec2_rad = np.radians(dec_p)

    # ang_dist=1deg, mag_flux_min=0.1
    ra_sel, dec_sel, flux_sel = g.select_stars(
        ra1_rad, dec1_rad, ra2_rad, dec2_rad, ra_p, dec_p, flux
    )

    # 0,0은 통과 / 10deg는 각도 초과로 탈락 / nan flux는 0이라 탈락
    assert np.allclose(ra_sel, [0.0])
    assert np.allclose(dec_sel, [0.0])
    assert np.allclose(flux_sel, [1.0])


def test_radec_to_xy_stars_rounding_rule(tmp_path):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    g = GFAGuider(config=str(cfgp))

    class FakeWCS:
        def world_to_pixel_values(self, ra, dec):
            # ra/dec 배열을 받아서 x,y로 반환(테스트용)
            # x= [0.2, 0.6], y=[1.49, 1.51] 같은 케이스로 rounding 확인
            return np.array([0.2, 0.6]), np.array([1.49, 1.51])

    ra = np.array([0.0, 0.0])
    dec = np.array([0.0, 0.0])

    dra, ddec, dra_f, ddec_f = g.radec_to_xy_stars(ra, dec, FakeWCS())

    # np.round(0.2)=0 -> +1 => 1, np.round(0.6)=1 -> +1 => 2
    assert np.all(dra == np.array([1, 2]))
    # np.round(1.49)=1 -> +1 => 2, np.round(1.51)=2 -> +1 => 3
    assert np.all(ddec == np.array([2, 3]))
    # float 버전은 +1.0만
    assert np.allclose(dra_f, np.array([1.2, 1.6]))
    assert np.allclose(ddec_f, np.array([2.49, 2.51]))


def test_peak_select_filters(tmp_path):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    g = GFAGuider(config=str(cfgp))

    dx = [1, 2, 3, 4]
    dy = [10, 20, 30, 40]
    peakc = [5, 50, 50000, 15]  # peakmin=10, peakmax=30000

    dxn, dyn, pindn = g.peak_select(dx, dy, peakc)

    # peak 50(인덱스1)와 15(인덱스3)만 통과
    assert np.all(pindn == np.array([1, 3]))
    assert np.all(dxn == np.array([2, 4]))
    assert np.all(dyn == np.array([20, 40]))


def test_cal_final_offset_warning_when_no_stars(tmp_path):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    g = GFAGuider(config=str(cfgp))

    fdx, fdy = g.cal_final_offset(np.array([]), np.array([]), np.array([]))
    assert fdx == "Warning"
    assert fdy == "Warning"


def test_cal_final_offset_returns_zero_when_below_threshold(tmp_path):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    g = GFAGuider(config=str(cfgp))

    # 작은 오프셋들 -> hypot(mean) <= crit_out(0.5) 되게
    dxp = np.array([0.05, 0.1, 0.0])
    dyp = np.array([0.05, 0.0, 0.1])
    pindp = np.array([0, 1, 2])

    fdx, fdy = g.cal_final_offset(dxp, dyp, pindp)
    assert fdx == 0.0
    assert fdy == 0.0


def test_cal_seeing_nan_when_no_cutouts(tmp_path):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    g = GFAGuider(config=str(cfgp))

    fwhm = g.cal_seeing([])
    assert math.isnan(fwhm)


def test_cal_seeing_uses_curve_fit_mock(tmp_path, monkeypatch):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    g = GFAGuider(config=str(cfgp))

    # cutout_path 존재하게
    cutout_dir = Path(g.cutout_path)
    cutout_dir.mkdir(parents=True, exist_ok=True)

    # curve_fit을 mock해서 sigma=2.0을 반환하게 만들면
    # fwhm = 2*sqrt(2*ln2)*sigma * pixel_scale
    def fake_curve_fit(func, xy, z, p0):
        params = np.array([100.0, 5.0, 5.0, 2.0, 0.0])  # amp,x0,y0,sigma,offset
        cov = np.eye(5)
        return params, cov

    monkeypatch.setattr("gfa_guider.curve_fit", fake_curve_fit)

    cutout = np.ones((11, 11), dtype=np.float32)
    fwhm = g.cal_seeing([cutout, cutout])

    expected = 2.0 * math.sqrt(2.0 * math.log(2.0)) * 2.0 * g.pixel_scale
    assert abs(fwhm - expected) < 1e-6


def test_cal_centroid_offset_success_and_failure(tmp_path, monkeypatch):
    """
    - photutils.find_peaks를 mock해서 첫 별은 성공, 두 번째는 예외 발생시키고
      cal_centroid_offset가 (dx,dy,peakc)에 대해
      성공: 값 계산 / 실패: (0,0,-1)로 처리하는지 확인
    """
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    g = GFAGuider(config=str(cfgp))

    # cutout 저장 디렉토리 보장
    Path(g.cutout_path).mkdir(parents=True, exist_ok=True)

    # boxsize를 작게 해서 cutout 인덱싱이 안전하게
    g.boxsize = 8

    # 충분히 큰 이미지
    image_data = np.zeros((100, 100), dtype=np.float32)
    image_data[50, 50] = 1000.0  # 밝은 점 하나 (실제로는 peaks mock이라 크게 중요 X)

    # 두 별 (두 번째는 일부러 실패시키기)
    dra = np.array([50, 60])      # 1-based 가정
    ddec = np.array([50, 60])
    dra_f = np.array([50.0, 60.0])
    ddec_f = np.array([50.0, 60.0])
    fluxn = np.array([1000.0, 2000.0])  # max_flux는 2000이지만, 실패로 저장 로직이 안 탈 수도 있음

    stddev = 1.0

    # WCS mock: pixel_to_world_values를 선형으로 만들어서 dx/dy 계산이 가능하도록
    class FakeWCS:
        def pixel_to_world_values(self, x, y):
            # ra = 0.001 * x (deg), dec = 0.001 * y (deg)
            return (0.001 * x, 0.001 * y)

    wcs = FakeWCS()

    # photutils.detection.find_peaks mock
    call = {"n": 0}

    def fake_find_peaks(cutout, threshold, box_size, npeaks):
        call["n"] += 1
        if call["n"] == 2:
            raise RuntimeError("peak finding failed")
        # 첫 호출: peak 하나 반환(테이블/array 흉내)
        return {"x_peak": [g.boxsize // 2], "y_peak": [g.boxsize // 2], "peak_value": [123.0]}

    monkeypatch.setattr("gfa_guider.pd.find_peaks", fake_find_peaks)

    cutoutn_stack = []
    dx, dy, peakc, cutoutn_stack = g.cal_centroid_offset(
        dra=dra,
        ddec=ddec,
        dra_f=dra_f,
        ddec_f=ddec_f,
        stddev=stddev,
        wcs=wcs,
        fluxn=fluxn,
        file_counter=1,
        cutoutn_stack=cutoutn_stack,
        image_data=image_data,
    )

    assert len(dx) == 2 and len(dy) == 2 and len(peakc) == 2

    # 첫 번째는 성공 → peak_value가 들어가야 함
    assert peakc[0] == 123.0

    # 두 번째는 실패 → (0,0,-1)
    assert dx[1] == 0
    assert dy[1] == 0
    assert peakc[1] == -1


def test_exe_cal_minimal_pipeline(tmp_path, monkeypatch):
    """
    exe_cal 전체 흐름을 "통합처럼 보이지만" 내부 계산은 전부 mock 처리해서
    - 파일 페어를 순회하는지
    - 최종 (fdx,fdy,fwhm) 리턴이 원하는 형태로 나오는지
    를 빠르게 확인.
    """
    # config에서 디렉터리를 절대경로로 넣어두면 os.path.join(base_dir, abs) => abs가 우선
    cfg = {
        "paths": {
            "directories": {
                "processed_images": str(tmp_path / "processed"),
                "final_astrometry_images": str(tmp_path / "final"),
                "cutout_directory": str(tmp_path / "cutout"),
            }
        },
        "detection": {
            "box_size": 20,
            "criteria": {"critical_outlier": 0.5},
            "peak_detection": {"max": 30000, "min": 10},
        },
        "catalog_matching": {
            "tolerance": {"angular_distance": 1.0, "mag_flux_min": 0.1},
            "fields": {"ra_column": "RA", "dec_column": "DEC", "mag_flux": "FLUX"},
        },
        "settings": {"image_processing": {"pixel_scale": 0.4}},
    }
    cfgp = tmp_path / "cfg.json"
    cfgp.write_text(json.dumps(cfg), encoding="utf-8")

    g = GFAGuider(config=str(cfgp))

    astro_dir = Path(g.final_astrometry_dir)
    proc_dir = Path(g.processed_dir)
    astro_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    # 더미 파일 2쌍 생성
    a1 = astro_dir / "a1.fits"
    a2 = astro_dir / "a2.fits"
    p1 = proc_dir / "p1.fits"
    p2 = proc_dir / "p2.fits"
    for f in (a1, a2, p1, p2):
        f.write_text("dummy", encoding="utf-8")

    # glob.glob이 위 파일들을 돌려주게 mock (정렬 유지)
    def fake_glob(pattern):
        if str(astro_dir) in pattern:
            return [str(a1), str(a2)]
        if str(proc_dir) in pattern:
            return [str(p1), str(p2)]
        return []

    monkeypatch.setattr("gfa_guider.glob.glob", fake_glob)

    # exe_cal 내부에서 호출되는 메서드들을 전부 mock
    class FakeWCS:
        pass

    def fake_load_image_and_wcs(path):
        header = {"CRVAL1": 1.0, "CRVAL2": 2.0}
        return np.zeros((10, 10), dtype=np.float32), header, FakeWCS()

    def fake_load_only_image(path):
        return np.zeros((10, 10), dtype=np.float32)

    def fake_background(img):
        return img, 1.0

    def fake_load_star_catalog(crval1, crval2):
        # 반환 튜플 형태만 맞추면 됨
        ra1_rad, dec1_rad = 0.0, 0.0
        ra2_rad = np.array([0.0])
        dec2_rad = np.array([0.0])
        ra_p = np.array([0.0])
        dec_p = np.array([0.0])
        flux = np.array([100.0])
        return ra1_rad, dec1_rad, ra2_rad, dec2_rad, ra_p, dec_p, flux

    def fake_select_stars(*args, **kwargs):
        return np.array([0.0]), np.array([0.0]), np.array([100.0])

    def fake_radec_to_xy_stars(*args, **kwargs):
        return np.array([5]), np.array([5]), np.array([5.0]), np.array([5.0])

    def fake_cal_centroid_offset(*args, **kwargs):
        return [0.2], [0.3], [100.0], []

    def fake_peak_select(dx, dy, peakc):
        return np.array(dx), np.array(dy), np.array([0])

    def fake_cal_final_offset(dxp, dyp, pindp):
        return 1.1, -2.2

    def fake_cal_seeing(cutoutn_stack):
        return 3.3

    monkeypatch.setattr(g, "load_image_and_wcs", fake_load_image_and_wcs)
    monkeypatch.setattr(g, "load_only_image", fake_load_only_image)
    monkeypatch.setattr(g, "background", fake_background)
    monkeypatch.setattr(g, "load_star_catalog", fake_load_star_catalog)
    monkeypatch.setattr(g, "select_stars", fake_select_stars)
    monkeypatch.setattr(g, "radec_to_xy_stars", fake_radec_to_xy_stars)
    monkeypatch.setattr(g, "cal_centroid_offset", fake_cal_centroid_offset)
    monkeypatch.setattr(g, "peak_select", fake_peak_select)
    monkeypatch.setattr(g, "cal_final_offset", fake_cal_final_offset)
    monkeypatch.setattr(g, "cal_seeing", fake_cal_seeing)

    fdx, fdy, fwhm = g.exe_cal()

    assert fdx == 1.1
    assert fdy == -2.2
    assert fwhm == 3.3
