# tests/test_gfa_guider.py
import json
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("photutils")
pytest.importorskip("astropy")

import gfa_guider as mod
from gfa_guider import GFAGuider


def _write_config(path: Path, tmp_path: Path, *, bad_json: bool = False):
    if bad_json:
        path.write_text("{bad json", encoding="utf-8")
        return

    cfg = {
        "paths": {
            "directories": {
                # 절대경로로 넣으면 os.path.join(base_dir, abs) => abs 우선이라 테스트 안전
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


# -------------------------
# helper: _get_default_config_path / _get_default_logger
# -------------------------
def test_get_default_config_path_missing_raises(monkeypatch, tmp_path):
    # default_path가 없다고 강제
    monkeypatch.setattr(mod.os.path, "isfile", lambda p: False)
    with pytest.raises(FileNotFoundError):
        mod._get_default_config_path()


def test_get_default_config_path_success(monkeypatch, tmp_path):
    # default_path 존재한다고 강제
    monkeypatch.setattr(mod.os.path, "isfile", lambda p: True)
    p = mod._get_default_config_path()
    assert p.endswith(
        "etc" + str(Path("/")).strip("/") + "astrometry_params.json"
    ) or p.endswith("etc/astrometry_params.json")


def test_default_logger_no_duplicate_handlers():
    lg1 = mod._get_default_logger()
    n1 = len(lg1.handlers)
    lg2 = mod._get_default_logger()
    n2 = len(lg2.handlers)
    assert lg1 is lg2
    assert n2 == n1


# -------------------------
# __init__ error branches
# -------------------------
def test_init_bad_json_raises_runtimeerror(tmp_path):
    cfgp = tmp_path / "bad.json"
    _write_config(cfgp, tmp_path, bad_json=True)
    with pytest.raises(RuntimeError):
        GFAGuider(config=str(cfgp))


def test_init_missing_config_raises_runtimeerror(tmp_path):
    with pytest.raises(RuntimeError):
        GFAGuider(config=str(tmp_path / "nope.json"))


# -------------------------
# load_image_and_wcs / load_only_image
# -------------------------
def test_load_image_and_wcs_file_not_found(tmp_path):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    g = GFAGuider(config=str(cfgp))

    with pytest.raises(FileNotFoundError):
        g.load_image_and_wcs(str(tmp_path / "missing.fits"))


def test_load_image_and_wcs_other_exception(monkeypatch, tmp_path):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    g = GFAGuider(config=str(cfgp))

    def boom(*a, **k):
        raise RuntimeError("fits broken")

    monkeypatch.setattr(mod.fits, "getdata", boom)

    with pytest.raises(RuntimeError):
        g.load_image_and_wcs("any.fits")


def test_load_only_image_just_calls_fits_getdata(monkeypatch, tmp_path):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    g = GFAGuider(config=str(cfgp))

    called = {"n": 0}

    def fake_getdata(path, ext=0):
        called["n"] += 1
        return np.zeros((2, 2), dtype=np.float32)

    monkeypatch.setattr(mod.fits, "getdata", fake_getdata)
    out = g.load_only_image("x.fits")
    assert out.shape == (2, 2)
    assert called["n"] == 1


# -------------------------
# background
# -------------------------
def test_background_returns_subtracted_and_stddev(tmp_path):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    g = GFAGuider(config=str(cfgp))

    img = np.zeros((600, 1024), dtype=np.float32)
    img[:, :511] = 100.0
    img[:, 511:] = 200.0
    img[0, 0] = 101.0
    img[0, 511] = 199.0

    bg_sub, stddev = g.background(img)

    assert abs(float(np.mean(bg_sub[:, :511]))) < 1.0
    assert abs(float(np.mean(bg_sub[:, 511:]))) < 1.0
    assert stddev >= 0.0


# -------------------------
# load_star_catalog + select_stars
# -------------------------
def test_load_star_catalog_missing_file_raises(tmp_path, monkeypatch):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    g = GFAGuider(config=str(cfgp))

    # star catalog path 존재 안한다고 강제
    monkeypatch.setattr(mod.os.path, "exists", lambda p: False)

    with pytest.raises(FileNotFoundError):
        g.load_star_catalog(1.0, 2.0)


def test_select_stars_filters_by_angle_and_flux(tmp_path):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    g = GFAGuider(config=str(cfgp))

    ra1_rad, dec1_rad = 0.0, 0.0

    ra_p = np.array([0.0, 10.0, 0.1])
    dec_p = np.array([0.0, 0.0, 0.1])
    flux = np.array([1.0, 1.0, np.nan])

    ra2_rad = np.radians(ra_p)
    dec2_rad = np.radians(dec_p)

    ra_sel, dec_sel, flux_sel = g.select_stars(
        ra1_rad, dec1_rad, ra2_rad, dec2_rad, ra_p, dec_p, flux
    )

    assert np.allclose(ra_sel, [0.0])
    assert np.allclose(dec_sel, [0.0])
    assert np.allclose(flux_sel, [1.0])


# -------------------------
# radec_to_xy_stars
# -------------------------
def test_radec_to_xy_stars_rounding_rule(tmp_path):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    g = GFAGuider(config=str(cfgp))

    class FakeWCS:
        def world_to_pixel_values(self, ra, dec):
            return np.array([0.2, 0.6]), np.array([1.49, 1.51])

    ra = np.array([0.0, 0.0])
    dec = np.array([0.0, 0.0])

    dra, ddec, dra_f, ddec_f = g.radec_to_xy_stars(ra, dec, FakeWCS())

    assert np.all(dra == np.array([1, 2]))
    assert np.all(ddec == np.array([2, 3]))
    assert np.allclose(dra_f, np.array([1.2, 1.6]))
    assert np.allclose(ddec_f, np.array([2.49, 2.51]))


# -------------------------
# cal_centroid_offset (success + failure)
# -------------------------
def test_cal_centroid_offset_success_and_failure(tmp_path, monkeypatch):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    g = GFAGuider(config=str(cfgp))

    Path(g.cutout_path).mkdir(parents=True, exist_ok=True)
    g.boxsize = 8

    image_data = np.zeros((100, 100), dtype=np.float32)
    image_data[50, 50] = 1000.0

    dra = np.array([50, 60])
    ddec = np.array([50, 60])
    dra_f = np.array([50.0, 60.0])
    ddec_f = np.array([50.0, 60.0])
    fluxn = np.array([1000.0, 2000.0])

    stddev = 1.0

    class FakeWCS:
        def pixel_to_world_values(self, x, y):
            return (0.001 * x, 0.001 * y)

    wcs = FakeWCS()

    call = {"n": 0}

    def fake_find_peaks(cutout, threshold, box_size, npeaks):
        call["n"] += 1
        if call["n"] == 2:
            raise RuntimeError("peak finding failed")
        return {
            "x_peak": [g.boxsize // 2],
            "y_peak": [g.boxsize // 2],
            "peak_value": [123.0],
        }

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
    assert peakc[0] == 123.0
    assert dx[1] == 0
    assert dy[1] == 0
    assert peakc[1] == -1


# -------------------------
# peak_select
# -------------------------
def test_peak_select_filters(tmp_path):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    g = GFAGuider(config=str(cfgp))

    dx = [1, 2, 3, 4]
    dy = [10, 20, 30, 40]
    peakc = [5, 50, 50000, 15]

    dxn, dyn, pindn = g.peak_select(dx, dy, peakc)

    assert np.all(pindn == np.array([1, 3]))
    assert np.all(dxn == np.array([2, 4]))
    assert np.all(dyn == np.array([20, 40]))


# -------------------------
# cal_final_offset branches
# -------------------------
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

    dxp = np.array([0.05, 0.1, 0.0])
    dyp = np.array([0.05, 0.0, 0.1])
    pindp = np.array([0, 1, 2])

    fdx, fdy = g.cal_final_offset(dxp, dyp, pindp)
    assert fdx == 0.0
    assert fdy == 0.0


def test_cal_final_offset_above_threshold_and_trim_minmax(tmp_path, monkeypatch):
    """
    len(cdx)>4 분기 + max/min 제거 분기 + hypot>crit_out 분기 태우기.
    sigma_clip은 mask 전부 False로 만들어서 cdx가 그대로 남게.
    """
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    g = GFAGuider(config=str(cfgp))

    class FakeClipped:
        def __init__(self, n):
            self.mask = np.array([False] * n)

    monkeypatch.setattr(
        "gfa_guider.sigma_clip",
        lambda distances, sigma, maxiters: FakeClipped(len(distances)),
    )

    # 6개면 >4라 min/max 제거 수행
    # min=2.0, max=10.0 제거 후 [2,2,2,2] => mean=2.0 (crit_out=0.5 보다 큼)
    dxp = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 10.0])
    dyp = np.zeros_like(dxp)
    pindp = np.arange(len(dxp))

    fdx, fdy = g.cal_final_offset(dxp, dyp, pindp)

    assert isinstance(fdx, float) and isinstance(fdy, float)
    assert fdx > g.crit_out
    assert fdy == 0.0


# -------------------------
# cal_seeing branches
# -------------------------
def test_cal_seeing_nan_when_no_cutouts(tmp_path):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    g = GFAGuider(config=str(cfgp))

    fwhm = g.cal_seeing([])
    assert math.isnan(fwhm)


def test_cal_seeing_save_fails_still_returns_value(tmp_path, monkeypatch):
    """
    fits.writeto 예외 분기 태우기 + curve_fit 정상 분기
    """
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    g = GFAGuider(config=str(cfgp))

    Path(g.cutout_path).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        "gfa_guider.fits.writeto",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk full")),
    )

    def fake_curve_fit(func, xy, z, p0):
        params = np.array([100.0, 5.0, 5.0, 2.0, 0.0])
        cov = np.eye(5)
        return params, cov

    monkeypatch.setattr("gfa_guider.curve_fit", fake_curve_fit)

    cutout = np.ones((11, 11), dtype=np.float32)
    fwhm = g.cal_seeing([cutout, cutout])

    expected = 2.0 * math.sqrt(2.0 * math.log(2.0)) * 2.0 * g.pixel_scale
    assert abs(fwhm - expected) < 1e-6


def test_cal_seeing_curve_fit_failure_returns_nan(tmp_path, monkeypatch):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    g = GFAGuider(config=str(cfgp))

    Path(g.cutout_path).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        "gfa_guider.curve_fit",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("fit fail")),
    )

    cutout = np.ones((11, 11), dtype=np.float32)
    fwhm = g.cal_seeing([cutout])
    assert math.isnan(fwhm)


# -------------------------
# exe_cal early error branches
# -------------------------
def test_exe_cal_missing_dirs_returns_nan(tmp_path, monkeypatch):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    g = GFAGuider(config=str(cfgp))

    # dir path 없다고 강제로
    g.final_astrometry_dir = ""
    g.processed_dir = ""

    fdx, fdy, fwhm = g.exe_cal()
    assert math.isnan(fdx) and math.isnan(fdy) and math.isnan(fwhm)


def test_exe_cal_no_astro_files_returns_nan(tmp_path, monkeypatch):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    g = GFAGuider(config=str(cfgp))

    # astro는 없음, proc는 있음처럼
    def fake_glob(pattern):
        if "final" in pattern:
            return []
        return ["x.fits"]

    monkeypatch.setattr("gfa_guider.glob.glob", fake_glob)

    fdx, fdy, fwhm = g.exe_cal()
    assert math.isnan(fdx) and math.isnan(fdy) and math.isnan(fwhm)


def test_exe_cal_no_proc_files_returns_nan(tmp_path, monkeypatch):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    g = GFAGuider(config=str(cfgp))

    def fake_glob(pattern):
        if "final" in pattern:
            return ["a.fits"]
        return []

    monkeypatch.setattr("gfa_guider.glob.glob", fake_glob)

    fdx, fdy, fwhm = g.exe_cal()
    assert math.isnan(fdx) and math.isnan(fdy) and math.isnan(fwhm)


def test_exe_cal_raises_runtimeerror_on_pair_processing_exception(
    tmp_path, monkeypatch
):
    """
    for-loop 내부 예외 -> RuntimeError로 재raise 되는 분기
    """
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    g = GFAGuider(config=str(cfgp))

    astro_dir = Path(g.final_astrometry_dir)
    proc_dir = Path(g.processed_dir)
    astro_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    a1 = astro_dir / "a1.fits"
    p1 = proc_dir / "p1.fits"
    a1.write_text("x", encoding="utf-8")
    p1.write_text("y", encoding="utf-8")

    def fake_glob(pattern):
        if str(astro_dir) in pattern:
            return [str(a1)]
        if str(proc_dir) in pattern:
            return [str(p1)]
        return []

    monkeypatch.setattr("gfa_guider.glob.glob", fake_glob)

    # load_image_and_wcs에서 예외 터뜨리기
    monkeypatch.setattr(
        g, "load_image_and_wcs", lambda p: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    with pytest.raises(RuntimeError):
        g.exe_cal()


def test_exe_cal_minimal_pipeline_success(tmp_path, monkeypatch):
    """
    정상 플로우: 파일 2쌍 + 내부 메서드 모킹으로 빠르게 성공 케이스 커버
    """
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

    a1 = astro_dir / "a1.fits"
    a2 = astro_dir / "a2.fits"
    p1 = proc_dir / "p1.fits"
    p2 = proc_dir / "p2.fits"
    for f in (a1, a2, p1, p2):
        f.write_text("dummy", encoding="utf-8")

    def fake_glob(pattern):
        if str(astro_dir) in pattern:
            return [str(a1), str(a2)]
        if str(proc_dir) in pattern:
            return [str(p1), str(p2)]
        return []

    monkeypatch.setattr("gfa_guider.glob.glob", fake_glob)

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
