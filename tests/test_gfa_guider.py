# tests/test_gfa_guider.py
import json
import math
import os
import sys
import types
import warnings
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest

warnings.filterwarnings("ignore", category=DeprecationWarning)


def _find_gfa_guider_py() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        repo_root / "src" / "kspec_gfa_controller" / "gfa_guider.py",  # src layout
        repo_root / "kspec_gfa_controller" / "gfa_guider.py",          # non-src layout
    ]
    for p in candidates:
        if p.exists():
            return p
    raise RuntimeError("gfa_guider.py not found. tried:\n" + "\n".join(map(str, candidates)))


def _install_fake_scipy_and_photutils():
    """
    현재 env에서 SciPy ABI(CXXABI_1.3.15) 때문에 scipy import가 깨지므로,
    guider 모듈 import 단계에서 죽지 않게 scipy + photutils를 함께 stub 한다.

    - gfa_guider.py는 `import photutils.detection as pd` 를 사용
      -> photutils가 scipy를 끌고 들어오므로 photutils도 같이 stub 필요
    - gfa_guider.py가 `from scipy.optimize import curve_fit`를 쓴다면 그것도 stub 필요

    이 stub는 "테스트에서 monkeypatch로 find_peaks / curve_fit을 교체"하기 위한 최소한만 제공한다.
    """

    # ---- fake scipy (package) ----
    if "scipy" not in sys.modules:
        fake_scipy = types.ModuleType("scipy")
        fake_scipy.__path__ = []  # 패키지처럼 보이게 (submodule import 허용)

        fake_opt = types.ModuleType("scipy.optimize")
        def _curve_fit_unavailable(*args, **kwargs):
            raise RuntimeError("scipy.optimize.curve_fit is unavailable in test environment")
        fake_opt.curve_fit = _curve_fit_unavailable

        # 가끔 다른 경로에서 scipy.ndimage / scipy.spatial 등을 참조할 수 있어 최소 stub 제공
        fake_nd = types.ModuleType("scipy.ndimage")
        def _maximum_filter_unavailable(*args, **kwargs):
            raise RuntimeError("scipy.ndimage.maximum_filter is unavailable in test environment")
        fake_nd.maximum_filter = _maximum_filter_unavailable

        fake_spatial = types.ModuleType("scipy.spatial")
        class _KDTreeUnavailable:  # photutils/utils가 KDTree를 import할 수 있음
            def __init__(self, *a, **k):
                raise RuntimeError("scipy.spatial.KDTree is unavailable in test environment")
        fake_spatial.KDTree = _KDTreeUnavailable

        sys.modules["scipy"] = fake_scipy
        sys.modules["scipy.optimize"] = fake_opt
        sys.modules["scipy.ndimage"] = fake_nd
        sys.modules["scipy.spatial"] = fake_spatial

        # scipy.optimize / scipy.ndimage / scipy.spatial attribute도 걸어두면 더 안전
        fake_scipy.optimize = fake_opt
        fake_scipy.ndimage = fake_nd
        fake_scipy.spatial = fake_spatial

    # ---- fake photutils (package) ----
    # photutils가 진짜로 import되면 scipy를 끌고 들어오므로, 아예 photutils를 stub으로 교체
    if "photutils" not in sys.modules:
        fake_photutils = types.ModuleType("photutils")
        fake_photutils.__path__ = []

        fake_detection = types.ModuleType("photutils.detection")

        # guider 코드에서 쓰는 pd.find_peaks를 제공
        # 테스트에서 monkeypatch로 교체하므로 기본 구현은 그냥 예외/더미여도 됨
        def _find_peaks_unavailable(*args, **kwargs):
            raise RuntimeError("photutils.detection.find_peaks is unavailable in test environment")

        fake_detection.find_peaks = _find_peaks_unavailable

        sys.modules["photutils"] = fake_photutils
        sys.modules["photutils.detection"] = fake_detection
        fake_photutils.detection = fake_detection


def _load_guider_module():
    """
    패키지 import(kspec_gfa_controller/__init__.py)를 타지 않고 gfa_guider.py만 직접 로드.
    SciPy/photutils ABI 문제는 stub 주입으로 회피.
    """
    _install_fake_scipy_and_photutils()

    path = _find_gfa_guider_py()
    spec = spec_from_file_location("_test_gfa_guider_module", str(path))
    if spec is None or spec.loader is None:
        pytest.fail(f"Could not create import spec for {path}")

    module = module_from_spec(spec)

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        pytest.fail(f"Failed to import guider module from {path}: {e}")

    return module


mod = _load_guider_module()
GFAGuider = mod.GFAGuider


@pytest.fixture(scope="function")
def guider_config(tmp_path: Path) -> Path:
    """
    gfa_guider의 요구 키를 만족하는 최소 config 생성.
    절대경로를 넣어 os.path.join(base_dir, abs) => abs 우선 되도록 함.
    """
    cfg = {
        "paths": {
            "directories": {
                "raw_images": str(tmp_path / "raw"),
                "final_astrometry_images": str(tmp_path / "final"),
                "cutout_directory": str(tmp_path / "cutout"),
                "star_catalog": str(tmp_path / "catalog"),
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
    return cfgp


def _write_bad_json(path: Path):
    path.write_text("{bad json", encoding="utf-8")


def _mk_guider(config_path: Path):
    """
    GFAGuider 시그니처가 logger를 요구하는 버전/아닌 버전 둘 다 대응.
    """
    kwargs = {}
    if "logger" in GFAGuider.__init__.__code__.co_varnames:
        kwargs["logger"] = mod._get_default_logger()
    return GFAGuider(config=str(config_path), **kwargs)


# -------------------------
# helper: _get_default_config_path / _get_default_logger
# -------------------------
def test_get_default_config_path_missing_raises(monkeypatch):
    monkeypatch.setattr(mod.os.path, "isfile", lambda p: False)
    with pytest.raises(FileNotFoundError):
        mod._get_default_config_path()


def test_get_default_config_path_success(monkeypatch):
    monkeypatch.setattr(mod.os.path, "isfile", lambda p: True)
    p = mod._get_default_config_path()
    norm = os.path.normpath(p)
    assert norm.endswith(os.path.normpath(os.path.join("etc", "astrometry_params.json")))


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
    _write_bad_json(cfgp)
    with pytest.raises(RuntimeError):
        _mk_guider(cfgp)


def test_init_missing_config_raises_runtimeerror(tmp_path):
    with pytest.raises(RuntimeError):
        _mk_guider(tmp_path / "nope.json")


# -------------------------
# load_image_and_wcs / load_only_image
# -------------------------
def test_load_image_and_wcs_file_not_found(guider_config, tmp_path):
    g = _mk_guider(guider_config)
    with pytest.raises(FileNotFoundError):
        g.load_image_and_wcs(str(tmp_path / "missing.fits"))


def test_load_image_and_wcs_other_exception(monkeypatch, guider_config):
    g = _mk_guider(guider_config)

    def boom(*a, **k):
        raise RuntimeError("fits broken")

    monkeypatch.setattr(mod.fits, "getdata", boom)
    with pytest.raises(RuntimeError):
        g.load_image_and_wcs("any.fits")


def test_load_only_image_just_calls_fits_getdata(monkeypatch, guider_config):
    g = _mk_guider(guider_config)

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
def test_background_returns_subtracted_and_stddev(guider_config):
    g = _mk_guider(guider_config)

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
def test_load_star_catalog_missing_file_raises(monkeypatch, guider_config):
    g = _mk_guider(guider_config)
    monkeypatch.setattr(mod.os.path, "exists", lambda p: False)
    with pytest.raises(FileNotFoundError):
        g.load_star_catalog(1.0, 2.0)


def test_select_stars_filters_by_angle_and_flux(guider_config):
    g = _mk_guider(guider_config)

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
def test_radec_to_xy_stars_rounding_rule(guider_config):
    g = _mk_guider(guider_config)

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
def test_cal_centroid_offset_success_and_failure(tmp_path, monkeypatch, guider_config):
    g = _mk_guider(guider_config)

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

    # gfa_guider.py에서 import photutils.detection as pd 로 잡힌 pd를 patch
    monkeypatch.setattr(mod.pd, "find_peaks", fake_find_peaks, raising=True)

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
def test_peak_select_filters(guider_config):
    g = _mk_guider(guider_config)

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
def test_cal_final_offset_warning_when_no_stars(guider_config):
    g = _mk_guider(guider_config)

    fdx, fdy = g.cal_final_offset(np.array([]), np.array([]), np.array([]))
    assert fdx == "Warning"
    assert fdy == "Warning"


def test_cal_final_offset_returns_zero_when_below_threshold(guider_config):
    g = _mk_guider(guider_config)

    dxp = np.array([0.05, 0.1, 0.0])
    dyp = np.array([0.05, 0.0, 0.1])
    pindp = np.array([0, 1, 2])

    fdx, fdy = g.cal_final_offset(dxp, dyp, pindp)
    assert fdx == 0.0
    assert fdy == 0.0


def test_cal_final_offset_above_threshold_and_trim_minmax(monkeypatch, guider_config):
    g = _mk_guider(guider_config)

    class FakeClipped:
        def __init__(self, n):
            self.mask = np.array([False] * n)

    monkeypatch.setattr(mod, "sigma_clip", lambda distances, sigma, maxiters: FakeClipped(len(distances)))

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
def test_cal_seeing_nan_when_no_cutouts(guider_config):
    g = _mk_guider(guider_config)

    fwhm = g.cal_seeing([])
    assert math.isnan(fwhm)


def test_cal_seeing_save_fails_still_returns_value(tmp_path, monkeypatch, guider_config):
    g = _mk_guider(guider_config)

    Path(g.cutout_path).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(mod.fits, "writeto", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk full")))

    def fake_curve_fit(func, xy, z, p0):
        params = np.array([100.0, 5.0, 5.0, 2.0, 0.0])
        cov = np.eye(5)
        return params, cov

    monkeypatch.setattr(mod, "curve_fit", fake_curve_fit, raising=True)

    cutout = np.ones((11, 11), dtype=np.float32)
    fwhm = g.cal_seeing([cutout, cutout])

    expected = 2.0 * math.sqrt(2.0 * math.log(2.0)) * 2.0 * g.pixel_scale
    assert abs(fwhm - expected) < 1e-6


def test_cal_seeing_curve_fit_failure_returns_nan(tmp_path, monkeypatch, guider_config):
    g = _mk_guider(guider_config)

    Path(g.cutout_path).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(mod, "curve_fit", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("fit fail")), raising=True)

    cutout = np.ones((11, 11), dtype=np.float32)
    fwhm = g.cal_seeing([cutout])
    assert math.isnan(fwhm)
