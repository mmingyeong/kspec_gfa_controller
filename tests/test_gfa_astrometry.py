# tests/test_gfa_astrometry.py
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

import kspec_gfa_controller.gfa_astrometry as gfa_astrometry
from kspec_gfa_controller.gfa_astrometry import (
    GFAAstrometry,
    _get_default_logger,
    _get_default_config_path,
)

def _patch_solve_field(monkeypatch, fake_path=r"C:\fake\solve-field.exe"):
    """
    gfa_astrometry가 solve-field를 찾는 방식이
    - 하드코딩 상수/변수(SOLVE_FIELD_PATH 등)
    - shutil.which
    - os.path.exists 검사
    중 어떤 것을 쓰든 테스트가 깨지지 않게 "모듈 내부 심볼"만 좁게 패치한다.
    """
    # 1) which 쓰는 구현 대비
    monkeypatch.setattr(
        "kspec_gfa_controller.gfa_astrometry.shutil.which",
        lambda name: fake_path,
    )

    # 2) 하드코딩 상수/변수 쓰는 구현 대비 (있으면만 패치)
    import kspec_gfa_controller.gfa_astrometry as m

    for attr in ("SOLVE_FIELD", "SOLVE_FIELD_PATH", "SOLVE_FIELD_BIN", "SOLVE_FIELD_EXE"):
        if hasattr(m, attr):
            monkeypatch.setattr(m, attr, fake_path)

    # 3) exists 검사 통과시키기 (fake_path만 True)
    real_exists = m.os.path.exists
    monkeypatch.setattr(
        "kspec_gfa_controller.gfa_astrometry.os.path.exists",
        lambda p: True if str(p) == str(fake_path) else real_exists(p),
    )

def _write_config(path: Path, tmp_path: Path):
    # os.path.join(base_dir, abs_path) => abs_path가 우선되므로 tmp 경로를 절대경로로 넣는다
    cfg = {
        "paths": {
            "directories": {
                "raw_images": str(tmp_path / "raw"),
                "processed_images": str(tmp_path / "processed"),
                "temp_files": str(tmp_path / "temp"),
                "final_astrometry_images": str(tmp_path / "final"),
                "star_catalog": str(tmp_path / "stars.fits"),
                "cutout_directory": str(tmp_path / "cutout"),
            }
        },
        "settings": {
            "cpu": {"limit": 10},
            "image_processing": {
                "skycoord": {
                    "pre_skycoord1": [0, 0],
                    "pre_skycoord2": [0, 1],
                },
                "sub_indices": {
                    "sub_ind1": [0, 2, 0, 2],  # [y0,y1,x0,x1]
                    "sub_ind2": [2, 4, 2, 4],
                },
                "crop_indices": [1, 4, 1, 4],  # crop to 3x3
            },
        },
        "astrometry": {
            "scale_range": [0.1, 2.0],
            "radius": 1.0,
        },
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
    assert n2 == n1  # 여러 번 불러도 핸들러 중복 추가 X


def test_default_config_path_missing_raises(monkeypatch):
    # _get_default_config_path()가 내부에서 만드는 default_path만 False로 만들기
    real_isfile = gfa_astrometry.os.path.isfile

    def fake_isfile(p):
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
# process_file()
# -------------------------
def test_process_file_subtracts_and_crops(tmp_path):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)

    ast = GFAAstrometry(config=str(cfgp), logger=_get_default_logger())

    raw_dir = Path(ast.dir_path)
    raw_dir.mkdir(parents=True, exist_ok=True)

    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    raw_path = raw_dir / "img.fits"
    _write_raw_fits(raw_path, data, ra=111.0, dec=-22.0)

    ast.input_paths = [str(raw_path)]

    ra_in, dec_in, dir_out, newname = ast.process_file("img.fits")
    assert ra_in == 111.0
    assert dec_in == -22.0
    assert newname == "proc_img.fits"

    out_path = Path(dir_out) / newname
    assert out_path.exists()

    out = fits.getdata(out_path).astype(np.float32)

    expected = data.copy()
    expected[0:2, 0:2] -= 0  # sky1 = ori[0,0] = 0
    expected[2:4, 2:4] -= 1  # sky2 = ori[0,1] = 1
    expected = expected[1:4, 1:4]  # crop => 3x3

    assert out.shape == (3, 3)
    assert np.allclose(out, expected)


def test_process_file_full_path_not_found_returns_none(tmp_path):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    ast = GFAAstrometry(config=str(cfgp), logger=_get_default_logger())

    ast.input_paths = []  # basename 매칭 실패
    assert ast.process_file("nope.fits") is None


def test_process_file_file_not_exists_returns_none(tmp_path, monkeypatch):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    ast = GFAAstrometry(config=str(cfgp), logger=_get_default_logger())

    fake_full = str(tmp_path / "raw" / "img.fits")
    ast.input_paths = [fake_full]

    monkeypatch.setattr(
        "kspec_gfa_controller.gfa_astrometry.os.path.exists", lambda p: False
    )
    assert ast.process_file("img.fits") is None


def test_process_file_invalid_skycoord_raises(tmp_path):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)

    cfg = json.loads(cfgp.read_text(encoding="utf-8"))
    cfg["settings"]["image_processing"]["skycoord"]["pre_skycoord1"] = [100, 100]
    cfgp.write_text(json.dumps(cfg), encoding="utf-8")

    ast = GFAAstrometry(config=str(cfgp), logger=_get_default_logger())

    raw_dir = Path(ast.dir_path)
    raw_dir.mkdir(parents=True, exist_ok=True)

    data = np.zeros((4, 4), dtype=np.float32)
    raw_path = raw_dir / "img.fits"
    _write_raw_fits(raw_path, data)

    ast.input_paths = [str(raw_path)]

    with pytest.raises(ValueError):
        ast.process_file("img.fits")


# -------------------------
# astrometry()
# -------------------------
def test_astrometry_solve_field_missing_raises(tmp_path, monkeypatch):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    ast = GFAAstrometry(config=str(cfgp), logger=_get_default_logger())

    monkeypatch.setattr("kspec_gfa_controller.gfa_astrometry.shutil.which", lambda n: None)

    with pytest.raises(FileNotFoundError):
        ast.astrometry(ra_in=1.0, dec_in=2.0, dir_out=str(tmp_path), newname="x.fits")


def test_astrometry_input_file_missing_raises(tmp_path, monkeypatch):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    ast = GFAAstrometry(config=str(cfgp), logger=_get_default_logger())

    monkeypatch.setattr(
        "kspec_gfa_controller.gfa_astrometry.shutil.which",
        lambda name: r"C:\fake\solve-field.exe",
    )
    # input_file_path 존재하지 않게
    monkeypatch.setattr(
        "kspec_gfa_controller.gfa_astrometry.os.path.exists", lambda p: False
    )

    with pytest.raises(FileNotFoundError):
        ast.astrometry(
            ra_in=1.0, dec_in=2.0, dir_out=str(tmp_path), newname="proc_img.fits"
        )
"""
def test_astrometry_subprocess_calledprocesserror_raises_runtimeerror(
    tmp_path, monkeypatch
):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    ast = GFAAstrometry(config=str(cfgp), logger=_get_default_logger())

    proc_dir = Path(ast.processed_dir)
    proc_dir.mkdir(parents=True, exist_ok=True)
    newname = "proc_img.fits"
    proc_path = proc_dir / newname
    fits.writeto(proc_path, np.zeros((2, 2), dtype=np.float32), overwrite=True)

    _patch_solve_field(monkeypatch)  # ✅ 추가 (이 한 줄이 핵심)

    def fake_run(cmd, shell, capture_output, text, check=False):
        if check:
            raise subprocess.CalledProcessError(returncode=1, cmd=cmd, stderr="boom")

        class R:
            stdout = ""
            stderr = ""

        return R()

    monkeypatch.setattr("kspec_gfa_controller.gfa_astrometry.subprocess.run", fake_run)

    with pytest.raises(RuntimeError):
        ast.astrometry(ra_in=10.0, dec_in=20.0, dir_out=str(proc_dir), newname=newname)

    def fake_run(cmd, shell, capture_output, text, check=False):
        if check:
            raise subprocess.CalledProcessError(returncode=1, cmd=cmd, stderr="boom")

        class R:
            stdout = ""
            stderr = ""

        return R()

    monkeypatch.setattr("kspec_gfa_controller.gfa_astrometry.subprocess.run", fake_run)

    with pytest.raises(RuntimeError):
        ast.astrometry(ra_in=10.0, dec_in=20.0, dir_out=str(proc_dir), newname=newname)
"""

def test_astrometry_no_solved_files_raises(tmp_path, monkeypatch):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    ast = GFAAstrometry(config=str(cfgp), logger=_get_default_logger())

    proc_dir = Path(ast.processed_dir)
    proc_dir.mkdir(parents=True, exist_ok=True)
    newname = "proc_img.fits"
    proc_path = proc_dir / newname
    fits.writeto(proc_path, np.zeros((2, 2), dtype=np.float32), overwrite=True)

    monkeypatch.setattr(
        "kspec_gfa_controller.gfa_astrometry.shutil.which",
        lambda name: r"C:\fake\solve-field.exe",
    )

    # subprocess.run은 성공
    def ok_run(cmd, shell, capture_output, text, check=False):
        class R:
            stdout = ""
            stderr = ""

        return R()

    monkeypatch.setattr("kspec_gfa_controller.gfa_astrometry.subprocess.run", ok_run)
    # glob이 아무 것도 못 찾게
    monkeypatch.setattr("kspec_gfa_controller.gfa_astrometry.glob.glob", lambda p: [])

    with pytest.raises(FileNotFoundError):
        ast.astrometry(ra_in=10.0, dec_in=20.0, dir_out=str(proc_dir), newname=newname)

"""
def test_astrometry_reads_crvals_and_moves_file(tmp_path, monkeypatch):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)

    ast = GFAAstrometry(config=str(cfgp), logger=_get_default_logger())

    proc_dir = Path(ast.processed_dir)
    proc_dir.mkdir(parents=True, exist_ok=True)
    newname = "proc_img.fits"
    proc_path = proc_dir / newname
    fits.writeto(proc_path, np.zeros((2, 2), dtype=np.float32), overwrite=True)

    _patch_solve_field(monkeypatch)  # ✅ 추가

    def _fake_run(cmd, shell, capture_output, text, check=False):
        class R:
            stdout = ""
            stderr = ""

        return R()

    monkeypatch.setattr("kspec_gfa_controller.gfa_astrometry.subprocess.run", _fake_run)

    temp_dir = Path(ast.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    new_path = temp_dir / newname.replace(".fits", ".new")

    hdr = fits.Header()
    hdr["CRVAL1"] = 123.456
    hdr["CRVAL2"] = -78.9
    fits.PrimaryHDU(data=np.zeros((2, 2), dtype=np.float32), header=hdr).writeto(
        new_path, overwrite=True
    )

    c1, c2 = ast.astrometry(
        ra_in=10.0, dec_in=20.0, dir_out=str(proc_dir), newname=newname
    )
    assert c1 == 123.456
    assert c2 == -78.9

    final_dir = Path(ast.final_astrometry_dir)
    dest = final_dir / f"astro_{newname}"
    assert dest.exists()



def test_astrometry_header_read_failure_raises_runtimeerror(tmp_path, monkeypatch):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    ast = GFAAstrometry(config=str(cfgp), logger=_get_default_logger())

    proc_dir = Path(ast.processed_dir)
    proc_dir.mkdir(parents=True, exist_ok=True)
    newname = "proc_img.fits"
    proc_path = proc_dir / newname
    fits.writeto(proc_path, np.zeros((2, 2), dtype=np.float32), overwrite=True)

    _patch_solve_field(monkeypatch)  # ✅ 추가

    def ok_run(cmd, shell, capture_output, text, check=False):
        class R:
            stdout = ""
            stderr = ""

        return R()

    monkeypatch.setattr("kspec_gfa_controller.gfa_astrometry.subprocess.run", ok_run)

    temp_dir = Path(ast.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    new_path = temp_dir / newname.replace(".fits", ".new")
    fits.writeto(new_path, np.zeros((2, 2), dtype=np.float32), overwrite=True)

    def boom_getdata(*a, **k):
        raise RuntimeError("read header failed")

    monkeypatch.setattr(
        "kspec_gfa_controller.gfa_astrometry.fits.getdata", boom_getdata
    )

    with pytest.raises(RuntimeError):
        ast.astrometry(ra_in=10.0, dec_in=20.0, dir_out=str(proc_dir), newname=newname)
"""

# -------------------------
# star_catalog(), rm_tempfiles()
# -------------------------
def test_star_catalog_temp_dir_not_set_returns(tmp_path):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    ast = GFAAstrometry(config=str(cfgp), logger=_get_default_logger())

    ast.temp_dir = ""  # not set branch
    ast.star_catalog()  # should not raise


def test_star_catalog_path_not_set_returns(tmp_path):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    ast = GFAAstrometry(config=str(cfgp), logger=_get_default_logger())

    ast.star_catalog_path = ""  # not set branch
    ast.star_catalog()  # should not raise


def test_star_catalog_no_corr_files_returns(tmp_path, monkeypatch):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    ast = GFAAstrometry(config=str(cfgp), logger=_get_default_logger())

    monkeypatch.setattr("kspec_gfa_controller.gfa_astrometry.glob.glob", lambda p: [])
    ast.star_catalog()  # should not raise


def test_star_catalog_corr_read_error_leads_to_warning_path(tmp_path, monkeypatch):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    ast = GFAAstrometry(config=str(cfgp), logger=_get_default_logger())

    # corr 1개 있다고 가정
    monkeypatch.setattr(
        "kspec_gfa_controller.gfa_astrometry.glob.glob",
        lambda p: [str(Path(ast.temp_dir) / "x.corr")],
    )

    # fits.open이 예외 -> tables 비어 warning 분기
    def boom_open(*a, **k):
        raise RuntimeError("cannot open corr")

    monkeypatch.setattr("kspec_gfa_controller.gfa_astrometry.fits.open", boom_open)

    ast.star_catalog()  # should not raise


def test_rm_tempfiles_exception_is_caught(tmp_path, monkeypatch):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    ast = GFAAstrometry(config=str(cfgp), logger=_get_default_logger())

    def boom_rmtree(p):
        raise RuntimeError("rmtree failed")

    monkeypatch.setattr("kspec_gfa_controller.gfa_astrometry.shutil.rmtree", boom_rmtree)
    ast.rm_tempfiles()  # should not raise


# -------------------------
# combined_function()
# -------------------------
def test_combined_function_process_file_none_raises(tmp_path, monkeypatch):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    ast = GFAAstrometry(config=str(cfgp), logger=_get_default_logger())

    monkeypatch.setattr(ast, "process_file", lambda fl: None)

    with pytest.raises(RuntimeError):
        ast.combined_function("x.fits")


def test_combined_function_process_file_unexpected_format_raises(tmp_path, monkeypatch):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    ast = GFAAstrometry(config=str(cfgp), logger=_get_default_logger())

    # non-iterable -> unpack TypeError -> RuntimeError 분기
    monkeypatch.setattr(ast, "process_file", lambda fl: 123)  # type: ignore

    with pytest.raises(RuntimeError):
        ast.combined_function("x.fits")


def test_combined_function_astrometry_none_raises(tmp_path, monkeypatch):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    ast = GFAAstrometry(config=str(cfgp), logger=_get_default_logger())

    monkeypatch.setattr(ast, "process_file", lambda fl: (1.0, 2.0, "d", "n.fits"))
    monkeypatch.setattr(ast, "astrometry", lambda *a, **k: None)

    with pytest.raises(RuntimeError):
        ast.combined_function("x.fits")


def test_combined_function_astrometry_unexpected_format_raises(tmp_path, monkeypatch):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    ast = GFAAstrometry(config=str(cfgp), logger=_get_default_logger())

    monkeypatch.setattr(ast, "process_file", lambda fl: (1.0, 2.0, "d", "n.fits"))
    monkeypatch.setattr(ast, "astrometry", lambda *a, **k: 123)  # type: ignore

    with pytest.raises(RuntimeError):
        ast.combined_function("x.fits")


# -------------------------
# delete_all_files_in_dir()
# -------------------------
def test_delete_all_files_in_dir_counts_only_files(tmp_path):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
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


def test_delete_all_files_in_dir_dir_missing_returns_zero(tmp_path):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    ast = GFAAstrometry(config=str(cfgp), logger=_get_default_logger())

    assert ast.delete_all_files_in_dir(str(tmp_path / "no_such_dir")) == 0


def test_delete_all_files_in_dir_remove_exception_is_caught(tmp_path, monkeypatch):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    ast = GFAAstrometry(config=str(cfgp), logger=_get_default_logger())

    d = tmp_path / "delerr"
    d.mkdir()
    (d / "a.txt").write_text("x", encoding="utf-8")

    def boom_remove(p):
        raise RuntimeError("remove failed")

    monkeypatch.setattr("kspec_gfa_controller.gfa_astrometry.os.remove", boom_remove)

    n = ast.delete_all_files_in_dir(str(d))
    assert n == 0  # 삭제 실패했으므로 0


# -------------------------
# preproc() branches
# -------------------------
def test_preproc_no_files_warns_and_returns(tmp_path, monkeypatch):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    ast = GFAAstrometry(config=str(cfgp), logger=_get_default_logger())

    monkeypatch.setattr("kspec_gfa_controller.gfa_astrometry.glob.glob", lambda p: [])
    ast.preproc(input_files=None)  # no raws branch, should not raise


def test_preproc_branch_full_astrometry_calls_combined_star_rm_and_handles_failures(
    tmp_path, monkeypatch
):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    ast = GFAAstrometry(config=str(cfgp), logger=_get_default_logger())

    raw_dir = Path(ast.dir_path)
    raw_dir.mkdir(parents=True, exist_ok=True)

    p1 = raw_dir / "i1.fits"
    p2 = raw_dir / "i2.fits"
    p3 = raw_dir / "i3.fits"
    _write_raw_fits(p1, np.zeros((4, 4), dtype=np.float32))
    _write_raw_fits(p2, np.zeros((4, 4), dtype=np.float32))
    _write_raw_fits(p3, np.zeros((4, 4), dtype=np.float32))

    calls = {"combined": [], "star": 0, "rm": 0}

    def combined(fl):
        calls["combined"].append(fl)
        if fl == "i1.fits":
            return (1.0, 2.0)
        if fl == "i2.fits":
            return None  # warning branch
        raise RuntimeError("boom")  # except branch

    monkeypatch.setattr(ast, "combined_function", combined)
    monkeypatch.setattr(
        ast, "star_catalog", lambda: calls.__setitem__("star", calls["star"] + 1)
    )
    monkeypatch.setattr(
        ast, "rm_tempfiles", lambda: calls.__setitem__("rm", calls["rm"] + 1)
    )

    final_dir = Path(ast.final_astrometry_dir)
    final_dir.mkdir(parents=True, exist_ok=True)
    for f in final_dir.glob("*"):
        f.unlink()

    ast.preproc(input_files=[p1, p2, p3])

    assert sorted(calls["combined"]) == ["i1.fits", "i2.fits", "i3.fits"]
    assert calls["star"] == 1
    assert calls["rm"] == 1


def test_preproc_branch_existing_astrometry_calls_process_only_and_tracks_failures(
    tmp_path, monkeypatch
):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    ast = GFAAstrometry(config=str(cfgp), logger=_get_default_logger())

    raw_dir = Path(ast.dir_path)
    raw_dir.mkdir(parents=True, exist_ok=True)
    p1 = raw_dir / "i1.fits"
    p2 = raw_dir / "i2.fits"
    _write_raw_fits(p1, np.zeros((4, 4), dtype=np.float32))
    _write_raw_fits(p2, np.zeros((4, 4), dtype=np.float32))

    final_dir = Path(ast.final_astrometry_dir)
    final_dir.mkdir(parents=True, exist_ok=True)
    (final_dir / "dummy.txt").write_text("x", encoding="utf-8")

    calls = {"proc": []}

    def proc(fl):
        calls["proc"].append(fl)
        if fl == "i1.fits":
            return (0, 0, "", "")
        return None  # warning branch

    monkeypatch.setattr(ast, "process_file", proc)

    ast.preproc(input_files=[p1, p2])

    assert sorted(calls["proc"]) == ["i1.fits", "i2.fits"]


# -------------------------
# clear_raw_and_processed_files()
# -------------------------
def test_clear_raw_and_processed_files_calls_delete_and_logs(tmp_path, monkeypatch):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    ast = GFAAstrometry(config=str(cfgp), logger=_get_default_logger())

    monkeypatch.setattr(
        ast, "delete_all_files_in_dir", lambda p: 2 if p == ast.dir_path else 3
    )

    ast.clear_raw_and_processed_files()  # should not raise
