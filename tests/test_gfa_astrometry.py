# tests/test_gfa_astrometry.py
import json
import logging
import os
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

import kspec_gfa_controller.gfa_astrometry as gfa_astrometry
from kspec_gfa_controller.gfa_astrometry import (
    GFAAstrometry,
    _get_default_logger,
    _get_default_config_path,
    _get_solve_field_path,
)


# -------------------------
# Helpers
# -------------------------
# -------------------------
# Helpers
# -------------------------
def _patch_solve_field_ok(monkeypatch, fake_path="/tmp/fake/solve-field"):
    """
    Path.exists를 건드리지 않고, solve-field path resolver만 패치해서
    테스트가 pytest/pathlib 내부에 영향을 주지 않게 한다.
    """
    # DEFAULT_SOLVE_FIELD 값도 맞춰두면 로그/디버깅에 좋음(선택)
    if hasattr(gfa_astrometry, "DEFAULT_SOLVE_FIELD"):
        monkeypatch.setattr(
            gfa_astrometry, "DEFAULT_SOLVE_FIELD", fake_path, raising=False
        )

    # 핵심: 검증 로직을 가진 함수 자체를 통째로 우회
    monkeypatch.setattr(
        gfa_astrometry,
        "_get_solve_field_path",
        lambda env=None: fake_path,
        raising=True,
    )

    # 혹시 코드가 env를 참조하는 경우를 위해 같이 세팅(선택)
    monkeypatch.setenv("ASTROMETRY_SOLVE_FIELD", fake_path)


def _patch_solve_field_missing(monkeypatch, fake_path="/tmp/fake/solve-field"):
    """
    solve-field가 없을 때의 예외 분기를 확실히 타게 한다.
    """
    if hasattr(gfa_astrometry, "DEFAULT_SOLVE_FIELD"):
        monkeypatch.setattr(
            gfa_astrometry, "DEFAULT_SOLVE_FIELD", fake_path, raising=False
        )

    def _raise(env=None):
        raise FileNotFoundError(f"solve-field not found: {fake_path}")

    monkeypatch.setattr(gfa_astrometry, "_get_solve_field_path", _raise, raising=True)
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
                "star_catalog": str(
                    tmp_path / "stars.fits"
                ),  # 파일로도 가능(소스가 지원)
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
    fits.PrimaryHDU(data=data.astype(np.float32), header=hdr).writeto(
        path, overwrite=True
    )


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
        if str(p).endswith(
            os.path.normpath(os.path.join("etc", "astrometry_params.json"))
        ):
            return False
        return real_isfile(p)

    monkeypatch.setattr(
        "kspec_gfa_controller.gfa_astrometry.os.path.isfile", fake_isfile
    )

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

    monkeypatch.setattr(
        "kspec_gfa_controller.gfa_astrometry.subprocess.run", lambda *a, **k: R()
    )

    with pytest.raises(RuntimeError):
        ast.astrometry_raw(str(raw))


def test_astrometry_raw_success_reads_crval_and_moves_and_writes_header(
    tmp_path, monkeypatch
):
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
        fits.HDUList([fits.PrimaryHDU(), hdu1]).writeto(
            work_dir / f"{outbase}.corr", overwrite=True
        )

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

    monkeypatch.setattr(
        "kspec_gfa_controller.gfa_astrometry.glob.glob", lambda *a, **k: []
    )

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


def test_build_combined_star_from_corr_all_bad_raises_runtimeerror(
    tmp_path, monkeypatch
):
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

    monkeypatch.setattr(
        "kspec_gfa_controller.gfa_astrometry.shutil.rmtree", boom_rmtree
    )
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
    fits.PrimaryHDU(data=np.zeros((2, 2), dtype=np.float32), header=hdr).writeto(
        out, overwrite=True
    )

    # input raw도 만들어서 current RA/DEC 제공
    raw = Path(ast.dir_path) / "raw1.fits"
    _write_raw_fits(raw, np.zeros((2, 2), dtype=np.float32), ra=10.0, dec=20.0)

    # preproc가 호출되면 안 됨(재사용 path). 호출되면 실패 처리
    monkeypatch.setattr(
        ast,
        "preproc",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("preproc should not run")),
    )

    outs = ast.ensure_astrometry_ready(input_files=[raw], force=False)
    assert str(out) in outs


# -------------------------
# solve-field path resolution (_get_solve_field_path)
# -------------------------
def test_get_solve_field_path_uses_env_param_priority(monkeypatch, tmp_path):
    sf = tmp_path / "solve-field"
    sf.write_text("x", encoding="utf-8")

    # Path.exists True + executable True
    monkeypatch.setattr(
        gfa_astrometry.Path, "exists", lambda self: self == sf, raising=True
    )
    monkeypatch.setattr(gfa_astrometry.os, "access", lambda p, mode: True, raising=True)

    env = {"ASTROMETRY_SOLVE_FIELD": str(sf)}
    got = _get_solve_field_path(env=env)
    assert got == str(sf)


def test_get_solve_field_path_missing_raises_filenotfound(monkeypatch, tmp_path):
    sf = tmp_path / "solve-field"
    # exists False
    monkeypatch.setattr(gfa_astrometry.Path, "exists", lambda self: False, raising=True)
    monkeypatch.setattr(gfa_astrometry.os, "access", lambda p, mode: True, raising=True)

    with pytest.raises(FileNotFoundError):
        _get_solve_field_path(env={"ASTROMETRY_SOLVE_FIELD": str(sf)})


def test_get_solve_field_path_not_executable_raises_permission(monkeypatch, tmp_path):
    sf = tmp_path / "solve-field"
    sf.write_text("x", encoding="utf-8")

    monkeypatch.setattr(
        gfa_astrometry.Path, "exists", lambda self: self == sf, raising=True
    )
    monkeypatch.setattr(
        gfa_astrometry.os, "access", lambda p, mode: False, raising=True
    )

    with pytest.raises(PermissionError):
        _get_solve_field_path(env={"ASTROMETRY_SOLVE_FIELD": str(sf)})


def test_resolve_solve_field_path_caches_and_only_recomputes_on_change(
    tmp_path, monkeypatch
):
    cfgp = tmp_path / "cfg.json"
    cfgp.write_text(
        """
        {
          "paths": {"directories": {"raw_images": "raw", "temp_files": "temp", "final_astrometry_images": "final", "star_catalog": "stars.fits"}},
          "settings": {"cpu": {"limit": 2}},
          "astrometry": {"scale_range": [0.1, 2.0], "radius": 1.0}
        }
        """.strip(),
        encoding="utf-8",
    )

    sf1 = tmp_path / "sf1"
    sf2 = tmp_path / "sf2"
    sf1.write_text("x", encoding="utf-8")
    sf2.write_text("x", encoding="utf-8")

    # ✅ 진짜 resolver가 통과하도록 exists/access만 패치
    monkeypatch.setattr(
        gfa_astrometry.Path, "exists", lambda self: self in (sf1, sf2), raising=True
    )
    monkeypatch.setattr(gfa_astrometry.os, "access", lambda p, mode: True, raising=True)

    calls = {"n": 0}
    real_get = gfa_astrometry._get_solve_field_path

    def spy(env=None):
        calls["n"] += 1
        return real_get(env=env)

    monkeypatch.setattr(gfa_astrometry, "_get_solve_field_path", spy, raising=True)

    # ✅ init 전에 env를 sf1로 세팅해둬야 DEFAULT_SOLVE_FIELD를 안 탄다
    monkeypatch.setenv("ASTROMETRY_SOLVE_FIELD", str(sf1))
    ast = GFAAstrometry(config=str(cfgp), logger=_get_default_logger())

    # init 과정에서 1회는 호출됨
    assert calls["n"] >= 1

    # 같은 key면 재호출 안 됨
    n0 = calls["n"]
    p1 = ast._resolve_solve_field_path(log=True)
    p1b = ast._resolve_solve_field_path(log=True)
    assert p1 == p1b
    assert calls["n"] == n0  # ✅ 여기서 cache 확인

    # env key 바꾸면 재호출됨
    ast.set_subprocess_env({"ASTROMETRY_SOLVE_FIELD": str(sf2)})
    p2 = ast._resolve_solve_field_path(log=True)
    assert p2 == str(sf2)
    assert calls["n"] >= n0 + 1


# -------------------------
# RA/DEC parsing + angular separation
# -------------------------
def test_parse_radec_to_deg_numeric_and_sexagesimal(tmp_path, monkeypatch):
    cfgp = tmp_path / "cfg.json"
    cfgp.write_text(
        """
        {
          "paths": {"directories": {"raw_images": "raw", "temp_files": "temp", "final_astrometry_images": "final", "star_catalog": "stars.fits"}},
          "settings": {"cpu": {"limit": 2}},
          "astrometry": {"scale_range": [0.1, 2.0], "radius": 1.0}
        }
        """.strip(),
        encoding="utf-8",
    )

    # solve-field path patch
    sf = tmp_path / "solve-field"
    sf.write_text("x", encoding="utf-8")
    monkeypatch.setattr(
        gfa_astrometry.Path, "exists", lambda self: self == sf, raising=True
    )
    monkeypatch.setattr(gfa_astrometry.os, "access", lambda p, mode: True, raising=True)
    monkeypatch.setenv("ASTROMETRY_SOLVE_FIELD", str(sf))

    ast = GFAAstrometry(config=str(cfgp), logger=gfa_astrometry._get_default_logger())

    # numeric
    ra, dec = ast._parse_radec_to_deg("10.0", "-20.0")
    assert ra == 10.0
    assert dec == -20.0

    # sexagesimal (hourangle/deg) - Windows CI에서 astropy ply 인코딩 문제 회피
    if os.name == "nt":
        pytest.skip(
            "Astropy angle parser can fail on Windows runners due to encoding (cp1252) issues."
        )
    ra2, dec2 = ast._parse_radec_to_deg("01:00:00", "+00:00:00")
    assert abs(ra2 - 15.0) < 1e-6
    assert abs(dec2 - 0.0) < 1e-6


# -------------------------
# ensure_astrometry_ready: reuse / mismatch / star_catalog_force
# -------------------------
def test_ensure_astrometry_ready_reuse_skips_preproc_and_can_force_star_catalog(
    tmp_path, monkeypatch
):
    cfgp = tmp_path / "cfg.json"
    cfgp.write_text(
        """
        {
          "paths": {"directories": {"raw_images": "raw", "temp_files": "temp", "final_astrometry_images": "final", "star_catalog": "starcat"}},
          "settings": {"cpu": {"limit": 2}},
          "astrometry": {"scale_range": [0.1, 2.0], "radius": 1.0}
        }
        """.strip(),
        encoding="utf-8",
    )

    sf = tmp_path / "solve-field"
    sf.write_text("x", encoding="utf-8")
    monkeypatch.setattr(
        gfa_astrometry.Path, "exists", lambda self: self == sf, raising=True
    )
    monkeypatch.setattr(gfa_astrometry.os, "access", lambda p, mode: True, raising=True)
    monkeypatch.setenv("ASTROMETRY_SOLVE_FIELD", str(sf))

    ast = GFAAstrometry(config=str(cfgp), logger=gfa_astrometry._get_default_logger())

    # 기존 astro output 생성(헤더 RA/DEC 포함)
    out = Path(ast.final_astrometry_dir) / "astro_exist.fits"
    hdr = fits.Header()
    hdr["RA"] = "10.0"
    hdr["DEC"] = "20.0"
    fits.PrimaryHDU(np.zeros((2, 2), np.float32), header=hdr).writeto(
        out, overwrite=True
    )

    # raw도 하나 만들어서 세션 체크 통과하도록
    raw = Path(ast.dir_path) / "raw1.fits"
    raw.parent.mkdir(parents=True, exist_ok=True)
    hdr2 = fits.Header()
    hdr2["RA"] = 10.0
    hdr2["DEC"] = 20.0
    fits.PrimaryHDU(np.zeros((2, 2), np.float32), header=hdr2).writeto(
        raw, overwrite=True
    )

    # preproc가 돌면 실패해야 함(재사용 경로)
    monkeypatch.setattr(
        ast,
        "preproc",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("preproc should not run")),
    )

    built = {"n": 0}
    monkeypatch.setattr(
        ast,
        "build_combined_star_from_corr",
        lambda *a, **k: built.__setitem__("n", built["n"] + 1),
    )

    outs = ast.ensure_astrometry_ready(
        input_files=[raw], force=False, build_star_catalog=True, star_catalog_force=True
    )
    assert str(out) in outs
    assert built["n"] == 1  # star_catalog_force=True 분기 커버


def test_ensure_astrometry_ready_session_mismatch_deletes_old_and_runs_preproc(
    tmp_path, monkeypatch
):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    _patch_solve_field_ok(monkeypatch)

    ast = GFAAstrometry(config=str(cfgp), logger=_get_default_logger())

    # 기존 astro output (내용은 아무거나 OK)
    old1 = Path(ast.final_astrometry_dir) / "astro_old1.fits"
    fits.writeto(old1, np.zeros((2, 2), np.float32), overwrite=True)

    # current raw: RA/DEC=(10,20)
    raw = Path(ast.dir_path) / "raw1.fits"
    raw.parent.mkdir(parents=True, exist_ok=True)
    _write_raw_fits(raw, np.zeros((2, 2), np.float32), ra=10.0, dec=20.0)

    # ✅ astro RA/DEC 강제(세션 mismatch 확정)
    monkeypatch.setattr(
        ast,
        "_get_reference_radec_from_astro_outputs",
        lambda outputs: ("0.0", "0.0"),
        raising=True,
    )

    preproc_called = {"n": 0}

    def fake_preproc(*a, **k):
        preproc_called["n"] += 1
        newout = Path(ast.final_astrometry_dir) / "astro_new.fits"
        fits.writeto(newout, np.zeros((2, 2), np.float32), overwrite=True)
        return ([], [])

    monkeypatch.setattr(ast, "preproc", fake_preproc, raising=True)

    outs = ast.ensure_astrometry_ready(
        input_files=[raw], force=False, build_star_catalog=False
    )

    assert preproc_called["n"] == 1
    assert not old1.exists()  # mismatch면 내부에서 지우고 내려감
    assert any(Path(p).name.startswith("astro_") for p in outs)


# -------------------------
# preproc(): 분기 커버 (run_missing_only / force / missing_only)
# -------------------------
def test_preproc_existing_outputs_and_run_missing_only_false_skips(
    tmp_path, monkeypatch
):
    cfgp = tmp_path / "cfg.json"
    cfgp.write_text(
        """
        {
          "paths": {"directories": {"raw_images": "raw", "temp_files": "temp", "final_astrometry_images": "final", "star_catalog": "stars.fits"}},
          "settings": {"cpu": {"limit": 2}},
          "astrometry": {"scale_range": [0.1, 2.0], "radius": 1.0}
        }
        """.strip(),
        encoding="utf-8",
    )

    sf = tmp_path / "solve-field"
    sf.write_text("x", encoding="utf-8")
    monkeypatch.setattr(
        gfa_astrometry.Path, "exists", lambda self: self == sf, raising=True
    )
    monkeypatch.setattr(gfa_astrometry.os, "access", lambda p, mode: True, raising=True)
    monkeypatch.setenv("ASTROMETRY_SOLVE_FIELD", str(sf))

    ast = GFAAstrometry(config=str(cfgp), logger=gfa_astrometry._get_default_logger())

    # input raw 하나
    raw = Path(ast.dir_path) / "r1.fits"
    raw.parent.mkdir(parents=True, exist_ok=True)
    fits.writeto(raw, np.zeros((2, 2), np.float32), overwrite=True)

    # existing output 만들어두기 -> run_missing_only=False면 스킵
    out = Path(ast.final_astrometry_dir) / "astro_r1.fits"
    fits.writeto(out, np.zeros((2, 2), np.float32), overwrite=True)

    monkeypatch.setattr(
        ast,
        "astrometry_raw",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not run")),
    )

    res, corr_ok = ast.preproc(input_files=[raw], force=False, run_missing_only=False)
    assert res == []
    assert corr_ok == []


def test_preproc_runs_only_missing_outputs(tmp_path, monkeypatch):
    cfgp = tmp_path / "cfg.json"
    cfgp.write_text(
        """
        {
          "paths": {"directories": {"raw_images": "raw", "temp_files": "temp", "final_astrometry_images": "final", "star_catalog": "stars.fits"}},
          "settings": {"cpu": {"limit": 2}},
          "astrometry": {"scale_range": [0.1, 2.0], "radius": 1.0}
        }
        """.strip(),
        encoding="utf-8",
    )

    _patch_solve_field_ok(monkeypatch)  # 여기선 resolver만 우회해도 OK
    ast = GFAAstrometry(config=str(cfgp), logger=_get_default_logger())

    raw_dir = Path(ast.dir_path)
    raw_dir.mkdir(parents=True, exist_ok=True)

    r1 = raw_dir / "a.fits"
    r2 = raw_dir / "b.fits"
    fits.writeto(r1, np.zeros((2, 2), np.float32), overwrite=True)
    fits.writeto(r2, np.zeros((2, 2), np.float32), overwrite=True)

    # existing astro는 a만 있다고 "preproc가 인식"하게 만들기
    out_a = Path(ast.final_astrometry_dir) / "astro_a.fits"
    fits.writeto(out_a, np.zeros((2, 2), np.float32), overwrite=True)

    real_glob = gfa_astrometry.glob.glob

    def fake_glob(pattern, *a, **k):
        # preproc 내부에서 existing_outputs용 glob만 가로채기
        if pattern == os.path.join(ast.final_astrometry_dir, "astro_*.fits"):
            return [str(out_a)]
        return real_glob(pattern, *a, **k)

    monkeypatch.setattr(gfa_astrometry.glob, "glob", fake_glob, raising=True)

    called = {"paths": []}

    def fake_astrometry_raw(p):
        called["paths"].append(Path(p).name)
        astro_out = Path(ast.final_astrometry_dir) / f"astro_{Path(p).stem}.fits"
        corr_out = Path(ast.temp_dir) / Path(p).stem / f"{Path(p).stem}.corr"
        corr_out.parent.mkdir(parents=True, exist_ok=True)
        fits.writeto(astro_out, np.zeros((2, 2), np.float32), overwrite=True)
        col = fits.Column(name="X", format="E", array=np.array([1.0], dtype=np.float32))
        fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns([col])]).writeto(
            corr_out, overwrite=True
        )
        return (1.0, 2.0, str(astro_out), str(corr_out))

    monkeypatch.setattr(ast, "astrometry_raw", fake_astrometry_raw, raising=True)

    res, corr_ok = ast.preproc(input_files=[r1, r2], force=False, run_missing_only=True)

    assert called["paths"] == ["b.fits"]
    assert len(res) == 1
    assert len(corr_ok) == 1


# -------------------------
# Remaining helper and initialization branches
# -------------------------
def _new_astrometry(tmp_path, monkeypatch):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    _patch_solve_field_ok(monkeypatch)
    return GFAAstrometry(config=str(cfgp), logger=_get_default_logger())


def _write_corr(path: Path, values=(1.0,)):
    path.parent.mkdir(parents=True, exist_ok=True)
    column = fits.Column(
        name="X",
        format="E",
        array=np.asarray(values, dtype=np.float32),
    )
    table_hdu = fits.BinTableHDU.from_columns([column])
    fits.HDUList([fits.PrimaryHDU(), table_hdu]).writeto(path, overwrite=True)


def test_get_solve_field_path_uses_process_environment(monkeypatch, tmp_path):
    solve_field = tmp_path / "solve-field"
    solve_field.write_text("executable", encoding="utf-8")
    monkeypatch.setenv("ASTROMETRY_SOLVE_FIELD", str(solve_field))
    monkeypatch.setattr(
        gfa_astrometry.Path,
        "exists",
        lambda self: self == solve_field,
        raising=True,
    )
    monkeypatch.setattr(gfa_astrometry.os, "access", lambda *args: True)

    assert _get_solve_field_path(env={}) == str(solve_field)


def test_get_default_config_path_success(monkeypatch):
    monkeypatch.setattr(gfa_astrometry.os.path, "isfile", lambda path: True)

    result = _get_default_config_path()

    assert result.endswith(os.path.join("etc", "astrometry_params.json"))


def test_init_uses_default_config_and_logger(tmp_path, monkeypatch):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    expected_logger = logging.getLogger("test_default_astrometry_logger")

    monkeypatch.setattr(gfa_astrometry, "_get_default_config_path", lambda: str(cfgp))
    monkeypatch.setattr(gfa_astrometry, "_get_default_logger", lambda: expected_logger)
    _patch_solve_field_ok(monkeypatch)

    ast = GFAAstrometry()

    assert ast.logger is expected_logger
    assert Path(ast.dir_path).is_dir()


def test_header_and_reference_radec_edge_cases(tmp_path, monkeypatch):
    ast = _new_astrometry(tmp_path, monkeypatch)

    missing_dec = Path(ast.dir_path) / "missing_dec.fits"
    header = fits.Header()
    header["RA"] = 10.0
    fits.writeto(missing_dec, np.zeros((2, 2)), header=header, overwrite=True)

    with pytest.raises(KeyError, match="RA/DEC header missing"):
        ast._read_radec_from_header(str(missing_dec))
    missing_dec.unlink()

    valid = Path(ast.dir_path) / "valid.fits"
    _write_raw_fits(valid, np.zeros((2, 2)), ra=12.0, dec=-3.0)
    assert ast._get_reference_radec_from_inputs() == ("12.0", "-3.0")
    assert ast._get_reference_radec_from_inputs(input_files=[]) is None
    assert ast._get_reference_radec_from_astro_outputs([]) is None

    no_radec = Path(ast.final_astrometry_dir) / "astro_no_radec.fits"
    fits.writeto(no_radec, np.zeros((2, 2)), overwrite=True)
    assert ast._get_reference_radec_from_astro_outputs([str(no_radec)]) is None
    assert ast._get_reference_radec_from_astro_outputs(
        [str(tmp_path / "missing_astro.fits")]
    ) is None


def test_parse_radec_hourangle_path_without_platform_parser(tmp_path, monkeypatch):
    ast = _new_astrometry(tmp_path, monkeypatch)

    class Angle:
        def __init__(self, degrees):
            self.deg = degrees

    class Coordinate:
        ra = Angle(15.0)
        dec = Angle(-2.0)

    monkeypatch.setattr(gfa_astrometry, "SkyCoord", lambda *args, **kwargs: Coordinate())

    assert ast._parse_radec_to_deg("01:00:00", "-02:00:00") == (15.0, -2.0)


def test_parse_radec_falls_back_from_hourangle_to_degrees(tmp_path, monkeypatch):
    ast = _new_astrometry(tmp_path, monkeypatch)
    calls = []

    class Angle:
        def __init__(self, degrees):
            self.deg = degrees

    class Coordinate:
        ra = Angle(180.0)
        dec = Angle(5.0)

    def fake_skycoord(*args, **kwargs):
        calls.append(kwargs["unit"])
        if len(calls) == 1:
            raise ValueError("not hourangle input")
        return Coordinate()

    monkeypatch.setattr(gfa_astrometry, "SkyCoord", fake_skycoord)

    assert ast._parse_radec_to_deg("180 deg", "5 deg") == (180.0, 5.0)
    assert len(calls) == 2


def test_delete_astro_outputs_ignores_remove_error(tmp_path, monkeypatch):
    ast = _new_astrometry(tmp_path, monkeypatch)
    target = str(Path(ast.final_astrometry_dir) / "astro_locked.fits")
    monkeypatch.setattr(gfa_astrometry.glob, "glob", lambda pattern: [target])
    monkeypatch.setattr(
        gfa_astrometry.os,
        "remove",
        lambda path: (_ for _ in ()).throw(PermissionError("locked")),
    )

    assert ast._delete_astro_outputs() == 0


# -------------------------
# Remaining combined-star catalog branches
# -------------------------
def test_build_combined_star_handles_missing_cleanup_and_low_row_count(
    tmp_path, monkeypatch
):
    ast = _new_astrometry(tmp_path, monkeypatch)
    valid = Path(ast.temp_dir) / "valid.corr"
    missing = Path(ast.temp_dir) / "missing.corr"
    _write_corr(valid, values=(1.0,))

    output = ast.build_combined_star_from_corr(
        corr_files=[str(valid), str(missing)],
        min_rows=5,
        cleanup=True,
    )

    assert Path(output).exists()
    assert not valid.exists()


def test_build_combined_star_rejects_corr_without_table(tmp_path, monkeypatch):
    ast = _new_astrometry(tmp_path, monkeypatch)
    corr = Path(ast.temp_dir) / "primary_only.corr"
    fits.writeto(corr, np.zeros((2, 2)), overwrite=True)

    with pytest.raises(RuntimeError, match="All .corr files failed"):
        ast.build_combined_star_from_corr(corr_files=[str(corr)])


# -------------------------
# Remaining astrometry_raw branches
# -------------------------
def test_astrometry_raw_handles_unlistable_failed_workdir(tmp_path, monkeypatch):
    ast = _new_astrometry(tmp_path, monkeypatch)
    raw = Path(ast.dir_path) / "failed.fits"
    _write_raw_fits(raw, np.zeros((2, 2)), ra=10.0, dec=20.0)

    class Result:
        returncode = 1
        stdout = ""
        stderr = "solve failed"

    monkeypatch.setattr(gfa_astrometry.subprocess, "run", lambda *a, **k: Result())
    monkeypatch.setattr(
        gfa_astrometry.os,
        "listdir",
        lambda path: (_ for _ in ()).throw(OSError("cannot list")),
    )

    with pytest.raises(RuntimeError, match="new file not created"):
        ast.astrometry_raw(str(raw))


def test_astrometry_raw_nonzero_result_continues_and_logs_optional_failures(
    tmp_path, monkeypatch
):
    ast = _new_astrometry(tmp_path, monkeypatch)
    raw = Path(ast.dir_path) / "partial.fits"
    _write_raw_fits(raw, np.zeros((2, 2)), ra=30.0, dec=-10.0)

    existing_output = Path(ast.final_astrometry_dir) / "astro_partial.fits"
    fits.writeto(existing_output, np.ones((1, 1)), overwrite=True)

    def fake_run(cmd, capture_output, text, env):
        work_dir = Path(cmd[cmd.index("-D") + 1])
        outbase = cmd[cmd.index("-o") + 1]
        header = fits.Header()
        header["CRVAL1"] = 31.5
        header["CRVAL2"] = -9.5
        fits.writeto(
            work_dir / f"{outbase}.new",
            np.zeros((2, 2)),
            header=header,
            overwrite=True,
        )

        class Result:
            returncode = 2
            stdout = "partial stdout"
            stderr = "partial stderr"

        return Result()

    real_open = fits.open

    def selective_open(path, mode="readonly", *args, **kwargs):
        if mode == "update":
            raise OSError("header is read-only")
        return real_open(path, mode=mode, *args, **kwargs)

    monkeypatch.setattr(gfa_astrometry.subprocess, "run", fake_run)
    monkeypatch.setattr(
        gfa_astrometry.os,
        "listdir",
        lambda path: (_ for _ in ()).throw(OSError("cannot list")),
    )
    monkeypatch.setattr(gfa_astrometry.fits, "open", selective_open)

    crval1, crval2, output, corr = ast.astrometry_raw(str(raw))

    assert (crval1, crval2) == (31.5, -9.5)
    assert Path(output).exists()
    assert not Path(corr).exists()


def test_rm_tempfiles_recreates_empty_directory(tmp_path, monkeypatch):
    ast = _new_astrometry(tmp_path, monkeypatch)
    temp_dir = Path(ast.temp_dir)
    (temp_dir / "temporary.txt").write_text("temporary", encoding="utf-8")

    ast.rm_tempfiles()

    assert temp_dir.is_dir()
    assert list(temp_dir.iterdir()) == []


# -------------------------
# Remaining ensure_astrometry_ready branches
# -------------------------
def _write_astro_output(path: Path, ra=None, dec=None):
    header = fits.Header()
    if ra is not None:
        header["RA"] = ra
    if dec is not None:
        header["DEC"] = dec
    fits.writeto(path, np.zeros((2, 2)), header=header, overwrite=True)


def test_ensure_reuse_session_ok_catches_forced_catalog_failure(
    tmp_path, monkeypatch
):
    ast = _new_astrometry(tmp_path, monkeypatch)
    astro = Path(ast.final_astrometry_dir) / "astro_existing.fits"
    raw = Path(ast.dir_path) / "raw.fits"
    _write_astro_output(astro, ra=10.0, dec=20.0)
    _write_raw_fits(raw, np.zeros((2, 2)), ra=10.0, dec=20.0)

    monkeypatch.setattr(
        ast,
        "build_combined_star_from_corr",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no corr")),
    )
    monkeypatch.setattr(
        ast,
        "preproc",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must reuse")),
    )

    outputs = ast.ensure_astrometry_ready(
        input_files=[raw],
        star_catalog_force=True,
    )

    assert outputs == [str(astro)]


def test_ensure_reuse_without_radec_catches_catalog_failure(tmp_path, monkeypatch):
    ast = _new_astrometry(tmp_path, monkeypatch)
    astro = Path(ast.final_astrometry_dir) / "astro_without_radec.fits"
    _write_astro_output(astro)

    monkeypatch.setattr(
        ast,
        "build_combined_star_from_corr",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no corr")),
    )
    monkeypatch.setattr(
        ast,
        "preproc",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must reuse")),
    )

    outputs = ast.ensure_astrometry_ready(
        input_files=[],
        star_catalog_force=True,
    )

    assert outputs == [str(astro)]


def test_ensure_raises_when_preproc_produces_no_outputs(tmp_path, monkeypatch):
    ast = _new_astrometry(tmp_path, monkeypatch)
    monkeypatch.setattr(ast, "preproc", lambda *a, **k: ([], []))

    with pytest.raises(RuntimeError, match="expected outputs not found"):
        ast.ensure_astrometry_ready(force=True)


def test_ensure_builds_catalog_from_current_corr_files(tmp_path, monkeypatch):
    ast = _new_astrometry(tmp_path, monkeypatch)
    corr = Path(ast.temp_dir) / "current.corr"
    corr.write_text("placeholder", encoding="utf-8")
    output = Path(ast.final_astrometry_dir) / "astro_new.fits"
    built_with = []

    def fake_preproc(*args, **kwargs):
        _write_astro_output(output)
        return [], [str(corr)]

    monkeypatch.setattr(ast, "preproc", fake_preproc)
    monkeypatch.setattr(
        ast,
        "build_combined_star_from_corr",
        lambda corr_files=None: built_with.append(corr_files),
    )

    outputs = ast.ensure_astrometry_ready(force=True, build_star_catalog=True)

    assert outputs == [str(output)]
    assert built_with == [[str(corr)]]


def test_ensure_empty_corr_fallback_catalog_failure_is_nonfatal(
    tmp_path, monkeypatch
):
    ast = _new_astrometry(tmp_path, monkeypatch)
    output = Path(ast.final_astrometry_dir) / "astro_new.fits"

    def fake_preproc(*args, **kwargs):
        _write_astro_output(output)
        return [], []

    monkeypatch.setattr(ast, "preproc", fake_preproc)
    monkeypatch.setattr(
        ast,
        "build_combined_star_from_corr",
        lambda corr_files=None: (_ for _ in ()).throw(RuntimeError("no catalog")),
    )

    outputs = ast.ensure_astrometry_ready(
        force=True,
        build_star_catalog=True,
        fallback_glob_on_empty_corr_ok=True,
    )

    assert outputs == [str(output)]


# -------------------------
# Remaining preproc branches
# -------------------------
def test_preproc_empty_astro_directory_runs_full_processing(tmp_path, monkeypatch):
    ast = _new_astrometry(tmp_path, monkeypatch)
    raw = Path(ast.dir_path) / "full.fits"
    fits.writeto(raw, np.zeros((2, 2)), overwrite=True)

    monkeypatch.setattr(
        ast,
        "astrometry_raw",
        lambda path: (1.0, 2.0, "astro_full.fits", "missing.corr"),
    )

    results, corr_ok = ast.preproc(
        input_files=[raw],
        force=False,
        run_missing_only=True,
    )

    assert len(results) == 1
    assert corr_ok == []


def test_preproc_all_expected_outputs_exist_returns_empty(tmp_path, monkeypatch):
    ast = _new_astrometry(tmp_path, monkeypatch)
    raw = Path(ast.dir_path) / "done.fits"
    output = Path(ast.final_astrometry_dir) / "astro_done.fits"
    fits.writeto(raw, np.zeros((2, 2)), overwrite=True)
    fits.writeto(output, np.zeros((2, 2)), overwrite=True)
    monkeypatch.setattr(
        ast,
        "astrometry_raw",
        lambda path: (_ for _ in ()).throw(AssertionError("nothing should run")),
    )

    assert ast.preproc(input_files=[raw], run_missing_only=True) == ([], [])


def test_preproc_collects_worker_failure_and_returns(tmp_path, monkeypatch):
    ast = _new_astrometry(tmp_path, monkeypatch)
    raw = Path(ast.dir_path) / "broken.fits"
    fits.writeto(raw, np.zeros((2, 2)), overwrite=True)
    monkeypatch.setattr(
        ast,
        "astrometry_raw",
        lambda path: (_ for _ in ()).throw(RuntimeError("worker failed")),
    )

    results, corr_ok = ast.preproc(input_files=[raw], force=True)

    assert results == []
    assert corr_ok == []
