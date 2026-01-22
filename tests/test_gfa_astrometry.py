# tests/test_gfa_astrometry.py
import json
import os
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

import kspec_gfa_controller.gfa_astrometry as gfa_astrometry
from kspec_gfa_controller.gfa_astrometry import GFAAstrometry, _get_default_logger, _get_default_config_path


# -------------------------
# Helpers
# -------------------------
def _patch_solve_field_ok(monkeypatch, fake_path="/tmp/fake/solve-field"):
    """
    최신 소스는 shutil.which가 아니라 _get_solve_field_path(Path.exists, os.access)를 씀.
    그래서 DEFAULT_SOLVE_FIELD / env / Path.exists / os.access를 패치해서 통과시키는 게 핵심.
    """
    # DEFAULT_SOLVE_FIELD가 있으면 그걸 fake로 바꿈
    if hasattr(gfa_astrometry, "DEFAULT_SOLVE_FIELD"):
        monkeypatch.setattr(gfa_astrometry, "DEFAULT_SOLVE_FIELD", fake_path)

    # Path.exists: fake solve-field만 True
    real_exists = gfa_astrometry.Path.exists

    def fake_exists(self):
        if str(self) == str(fake_path):
            return True
        return real_exists(self)

    monkeypatch.setattr(gfa_astrometry.Path, "exists", fake_exists, raising=True)

    # 실행권한 체크 통과
    monkeypatch.setattr(gfa_astrometry.os, "access", lambda p, mode: True, raising=True)

    # 혹시 env 경로를 타면 그쪽도 맞춰줌
    monkeypatch.setenv("ASTROMETRY_SOLVE_FIELD", fake_path)


def _patch_solve_field_missing(monkeypatch, fake_path="/tmp/fake/solve-field"):
    if hasattr(gfa_astrometry, "DEFAULT_SOLVE_FIELD"):
        monkeypatch.setattr(gfa_astrometry, "DEFAULT_SOLVE_FIELD", fake_path)

    # Path.exists가 항상 False면 solve-field not found
    monkeypatch.setattr(gfa_astrometry.Path, "exists", lambda self: False, raising=True)
    monkeypatch.setenv("ASTROMETRY_SOLVE_FIELD", fake_path)


def _write_config(path: Path, tmp_path: Path):
    """
    최신 소스는 base_dir + config.paths.directories.* 를 join 하지만,
    네 config 값이 절대경로면 join 결과가 절대경로로 유지됨.
    """
    cfg = {
        "paths": {
            "directories": {
                "raw_images": str(tmp_path / "raw"),
                "temp_files": str(tmp_path / "temp"),
                "final_astrometry_images": str(tmp_path / "final"),
                "star_catalog": str(tmp_path / "stars.fits"),  # 파일로도 가능(소스가 지원)
            }
        },
        "settings": {"cpu": {"limit": 2}},
        "astrometry": {"scale_range": [0.1, 2.0], "radius": 1.0},
    }
    path.write_text(json.dumps(cfg), encoding="utf-8")


def _write_raw_fits(path: Path, data: np.ndarray, ra=10.0, dec=20.0):
    hdr = fits.Header()
    hdr["RA"] = ra
    hdr["DEC"] = dec
    fits.PrimaryHDU(data=data.astype(np.float32), header=hdr).writeto(path, overwrite=True)


# -------------------------
# default helpers
# -------------------------
def test_default_logger_no_duplicate_handlers():
    lg1 = _get_default_logger()
    n1 = len(lg1.handlers)
    lg2 = _get_default_logger()
    n2 = len(lg2.handlers)
    assert lg1 is lg2
    assert n2 == n1


def test_default_config_path_missing_raises(monkeypatch):
    real_isfile = gfa_astrometry.os.path.isfile

    def fake_isfile(p):
        # 모듈 내부 default 경로만 False
        if str(p).endswith(os.path.normpath(os.path.join("etc", "astrometry_params.json"))):
            return False
        return real_isfile(p)

    monkeypatch.setattr("kspec_gfa_controller.gfa_astrometry.os.path.isfile", fake_isfile)

    with pytest.raises(FileNotFoundError):
        _get_default_config_path()


# -------------------------
# __init__ and path wiring
# -------------------------
def test_init_creates_expected_paths(tmp_path, monkeypatch):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)

    _patch_solve_field_ok(monkeypatch)

    ast = GFAAstrometry(config=str(cfgp), logger=_get_default_logger())

    assert Path(ast.dir_path).exists()
    assert Path(ast.temp_dir).exists()
    assert Path(ast.final_astrometry_dir).exists()
    assert Path(ast.star_catalog_dir).exists()
    # star_catalog이 파일 경로이면 combined_star_path가 그 파일이어야 함
    assert ast.combined_star_path.endswith("stars.fits")


# -------------------------
# solve-field path resolution
# -------------------------
def test_solve_field_missing_raises(tmp_path, monkeypatch):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)

    _patch_solve_field_missing(monkeypatch)

    # __init__ 안에서 _resolve_solve_field_path 호출하므로 여기서 바로 예외 기대
    with pytest.raises(FileNotFoundError):
        GFAAstrometry(config=str(cfgp), logger=_get_default_logger())


# -------------------------
# astrometry_raw()
# -------------------------
def test_astrometry_raw_input_missing_raises(tmp_path, monkeypatch):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    _patch_solve_field_ok(monkeypatch)

    ast = GFAAstrometry(config=str(cfgp), logger=_get_default_logger())

    with pytest.raises(FileNotFoundError):
        ast.astrometry_raw(str(tmp_path / "nope.fits"))


def test_astrometry_raw_new_not_created_raises_runtimeerror(tmp_path, monkeypatch):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    _patch_solve_field_ok(monkeypatch)

    ast = GFAAstrometry(config=str(cfgp), logger=_get_default_logger())

    raw_dir = Path(ast.dir_path)
    raw_dir.mkdir(parents=True, exist_ok=True)

    raw = raw_dir / "img.fits"
    _write_raw_fits(raw, np.zeros((4, 4), dtype=np.float32), ra=11.0, dec=-22.0)

    # subprocess.run은 returncode만 주고, .new는 만들어주지 않음 -> RuntimeError
    class R:
        returncode = 1
        stdout = ""
        stderr = "boom"

    monkeypatch.setattr("kspec_gfa_controller.gfa_astrometry.subprocess.run", lambda *a, **k: R())

    with pytest.raises(RuntimeError):
        ast.astrometry_raw(str(raw))


def test_astrometry_raw_success_reads_crval_and_moves_and_writes_header(tmp_path, monkeypatch):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    _patch_solve_field_ok(monkeypatch)

    ast = GFAAstrometry(config=str(cfgp), logger=_get_default_logger())

    raw_dir = Path(ast.dir_path)
    raw_dir.mkdir(parents=True, exist_ok=True)

    raw = raw_dir / "img.fits"
    _write_raw_fits(raw, np.zeros((4, 4), dtype=np.float32), ra=111.0, dec=-22.0)

    # subprocess.run 호출 후에 .new / .corr 만들어주기
    def fake_run(cmd, capture_output, text, env):
        # cmd에 -D work_dir, -o outbase가 있음
        work_dir = cmd[cmd.index("-D") + 1]
        outbase = cmd[cmd.index("-o") + 1]
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        # .new 생성 (CRVAL1/2 포함)
        hdr = fits.Header()
        hdr["CRVAL1"] = 123.456
        hdr["CRVAL2"] = -78.9
        fits.PrimaryHDU(data=np.zeros((2, 2), dtype=np.float32), header=hdr).writeto(
            work_dir / f"{outbase}.new", overwrite=True
        )

        # .corr도 생성 (HDU[1] table 있는 형태)
        col = fits.Column(name="X", format="E", array=np.array([1.0], dtype=np.float32))
        hdu1 = fits.BinTableHDU.from_columns([col])
        fits.HDUList([fits.PrimaryHDU(), hdu1]).writeto(work_dir / f"{outbase}.corr", overwrite=True)

        class R:
            returncode = 0
            stdout = ""
            stderr = ""

        return R()

    monkeypatch.setattr("kspec_gfa_controller.gfa_astrometry.subprocess.run", fake_run)

    cr1, cr2, astro_path, corr_path = ast.astrometry_raw(str(raw))

    assert cr1 == 123.456
    assert cr2 == -78.9
    assert Path(astro_path).exists()
    assert Path(corr_path).exists()

    # astro header에 RA/DEC 기록되는지 확인
    hdr = fits.getheader(astro_path)
    assert str(hdr["RA"]) == "111.0"
    assert str(hdr["DEC"]) == "-22.0"


def test_astrometry_raw_header_read_failure_raises(tmp_path, monkeypatch):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    _patch_solve_field_ok(monkeypatch)

    ast = GFAAstrometry(config=str(cfgp), logger=_get_default_logger())

    raw_dir = Path(ast.dir_path)
    raw_dir.mkdir(parents=True, exist_ok=True)

    raw = raw_dir / "img.fits"
    _write_raw_fits(raw, np.zeros((4, 4), dtype=np.float32), ra=1.0, dec=2.0)

    def fake_run(cmd, capture_output, text, env):
        work_dir = Path(cmd[cmd.index("-D") + 1])
        outbase = cmd[cmd.index("-o") + 1]
        work_dir.mkdir(parents=True, exist_ok=True)
        # .new는 만들되 CRVAL 헤더를 일부러 안 넣음 -> 읽기 실패
        fits.PrimaryHDU(data=np.zeros((2, 2), dtype=np.float32)).writeto(
            work_dir / f"{outbase}.new", overwrite=True
        )

        class R:
            returncode = 0
            stdout = ""
            stderr = ""

        return R()

    monkeypatch.setattr("kspec_gfa_controller.gfa_astrometry.subprocess.run", fake_run)

    with pytest.raises(RuntimeError):
        ast.astrometry_raw(str(raw))


# -------------------------
# build_combined_star_from_corr()
# -------------------------
def test_build_combined_star_from_corr_no_files_raises(tmp_path, monkeypatch):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    _patch_solve_field_ok(monkeypatch)

    ast = GFAAstrometry(config=str(cfgp), logger=_get_default_logger())

    monkeypatch.setattr("kspec_gfa_controller.gfa_astrometry.glob.glob", lambda *a, **k: [])

    with pytest.raises(FileNotFoundError):
        ast.build_combined_star_from_corr(corr_files=None)


def test_build_combined_star_from_corr_reads_and_writes(tmp_path, monkeypatch):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    _patch_solve_field_ok(monkeypatch)

    ast = GFAAstrometry(config=str(cfgp), logger=_get_default_logger())

    # corr 2개 생성
    c1 = Path(ast.temp_dir) / "a.corr"
    c2 = Path(ast.temp_dir) / "b.corr"
    Path(ast.temp_dir).mkdir(parents=True, exist_ok=True)

    def _write_corr(p: Path, vals):
        col = fits.Column(name="X", format="E", array=np.array(vals, dtype=np.float32))
        hdu1 = fits.BinTableHDU.from_columns([col])
        fits.HDUList([fits.PrimaryHDU(), hdu1]).writeto(p, overwrite=True)

    _write_corr(c1, [1.0, 2.0])
    _write_corr(c2, [3.0])

    out = ast.build_combined_star_from_corr(corr_files=[str(c1), str(c2)])
    assert Path(out).exists()

    # vstack 결과 row 수 확인 (2 + 1)
    with fits.open(out) as hdul:
        assert len(hdul) >= 2
        assert len(hdul[1].data) == 3


def test_build_combined_star_from_corr_all_bad_raises_runtimeerror(tmp_path, monkeypatch):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    _patch_solve_field_ok(monkeypatch)

    ast = GFAAstrometry(config=str(cfgp), logger=_get_default_logger())

    bad = Path(ast.temp_dir) / "x.corr"
    Path(ast.temp_dir).mkdir(parents=True, exist_ok=True)
    bad.write_text("notfits", encoding="utf-8")

    with pytest.raises(RuntimeError):
        ast.build_combined_star_from_corr(corr_files=[str(bad)])


# -------------------------
# rm_tempfiles()
# -------------------------
def test_rm_tempfiles_exception_is_caught(tmp_path, monkeypatch):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    _patch_solve_field_ok(monkeypatch)

    ast = GFAAstrometry(config=str(cfgp), logger=_get_default_logger())

    def boom_rmtree(p):
        raise RuntimeError("rmtree failed")

    monkeypatch.setattr("kspec_gfa_controller.gfa_astrometry.shutil.rmtree", boom_rmtree)
    ast.rm_tempfiles()  # should not raise


# -------------------------
# delete_all_files_in_dir()
# -------------------------
def test_delete_all_files_in_dir_counts_only_files(tmp_path, monkeypatch):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    _patch_solve_field_ok(monkeypatch)

    ast = GFAAstrometry(config=str(cfgp), logger=_get_default_logger())

    d = tmp_path / "delme"
    d.mkdir()
    (d / "a.txt").write_text("x", encoding="utf-8")
    (d / "b.txt").write_text("y", encoding="utf-8")
    (d / "subdir").mkdir()

    n = ast.delete_all_files_in_dir(str(d))
    assert n == 2
    assert not (d / "a.txt").exists()
    assert not (d / "b.txt").exists()
    assert (d / "subdir").exists()


def test_delete_all_files_in_dir_dir_missing_returns_zero(tmp_path, monkeypatch):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    _patch_solve_field_ok(monkeypatch)

    ast = GFAAstrometry(config=str(cfgp), logger=_get_default_logger())
    assert ast.delete_all_files_in_dir(str(tmp_path / "no_such_dir")) == 0


def test_delete_all_files_in_dir_remove_exception_is_caught(tmp_path, monkeypatch):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    _patch_solve_field_ok(monkeypatch)

    ast = GFAAstrometry(config=str(cfgp), logger=_get_default_logger())

    d = tmp_path / "delerr"
    d.mkdir()
    (d / "a.txt").write_text("x", encoding="utf-8")

    def boom_remove(p):
        raise RuntimeError("remove failed")

    monkeypatch.setattr("kspec_gfa_controller.gfa_astrometry.os.remove", boom_remove)

    n = ast.delete_all_files_in_dir(str(d))
    assert n == 0


# -------------------------
# clear_raw_files()
# -------------------------
def test_clear_raw_files_calls_delete(tmp_path, monkeypatch):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    _patch_solve_field_ok(monkeypatch)

    ast = GFAAstrometry(config=str(cfgp), logger=_get_default_logger())

    called = {"n": 0}

    def fake_delete(p):
        assert p == ast.dir_path
        called["n"] += 1
        return 2

    monkeypatch.setattr(ast, "delete_all_files_in_dir", fake_delete)
    ast.clear_raw_files()
    assert called["n"] == 1


# -------------------------
# preproc() / ensure_astrometry_ready()
# -------------------------
def test_preproc_no_files_warns_and_returns(tmp_path, monkeypatch):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    _patch_solve_field_ok(monkeypatch)

    ast = GFAAstrometry(config=str(cfgp), logger=_get_default_logger())

    # input_files=None 이면 dir_path에서 찾는데, 비어있으니 warning branch
    res, corr_ok = ast.preproc(input_files=None)
    assert res == []
    assert corr_ok == []


def test_preproc_runs_astrometry_raw_and_collects_corr_ok(tmp_path, monkeypatch):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    _patch_solve_field_ok(monkeypatch)

    ast = GFAAstrometry(config=str(cfgp), logger=_get_default_logger())
    raw_dir = Path(ast.dir_path)
    raw_dir.mkdir(parents=True, exist_ok=True)

    p1 = raw_dir / "i1.fits"
    p2 = raw_dir / "i2.fits"
    _write_raw_fits(p1, np.zeros((4, 4), dtype=np.float32))
    _write_raw_fits(p2, np.zeros((4, 4), dtype=np.float32))

    def fake_astrometry_raw(path):
        stem = Path(path).stem
        astro_out = Path(ast.final_astrometry_dir) / f"astro_{stem}.fits"
        corr_out = Path(ast.temp_dir) / stem / f"{stem}.corr"
        corr_out.parent.mkdir(parents=True, exist_ok=True)
        # astro 파일 / corr 파일만 "존재"하게 만들어 주면 preproc이 corr_ok에 포함시킴
        fits.writeto(astro_out, np.zeros((2, 2), dtype=np.float32), overwrite=True)

        col = fits.Column(name="X", format="E", array=np.array([1.0], dtype=np.float32))
        hdu1 = fits.BinTableHDU.from_columns([col])
        fits.HDUList([fits.PrimaryHDU(), hdu1]).writeto(corr_out, overwrite=True)

        return (1.0, 2.0, str(astro_out), str(corr_out))

    monkeypatch.setattr(ast, "astrometry_raw", fake_astrometry_raw)

    res, corr_ok = ast.preproc(input_files=[p1, p2], force=True)
    assert len(res) == 2
    assert len(corr_ok) == 2


def test_ensure_astrometry_ready_reuses_existing_outputs(tmp_path, monkeypatch):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    _patch_solve_field_ok(monkeypatch)

    ast = GFAAstrometry(config=str(cfgp), logger=_get_default_logger())

    # 기존 astro_*.fits 만들어두기 + header RA/DEC 넣기 (세션 체크가 있으면 통과시키기 위함)
    out = Path(ast.final_astrometry_dir) / "astro_x.fits"
    hdr = fits.Header()
    hdr["RA"] = "10.0"
    hdr["DEC"] = "20.0"
    fits.PrimaryHDU(data=np.zeros((2, 2), dtype=np.float32), header=hdr).writeto(out, overwrite=True)

    # input raw도 만들어서 current RA/DEC 제공
    raw = Path(ast.dir_path) / "raw1.fits"
    _write_raw_fits(raw, np.zeros((2, 2), dtype=np.float32), ra=10.0, dec=20.0)

    # preproc가 호출되면 안 됨(재사용 path). 호출되면 실패 처리
    monkeypatch.setattr(ast, "preproc", lambda *a, **k: (_ for _ in ()).throw(AssertionError("preproc should not run")))

    outs = ast.ensure_astrometry_ready(input_files=[raw], force=False)
    assert str(out) in outs
