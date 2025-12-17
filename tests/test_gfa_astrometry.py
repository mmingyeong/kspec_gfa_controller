
# tests/test_gfa_astrometry.py
import json
import os
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from gfa_astrometry import GFAAstrometry, _get_default_logger


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
    fits.PrimaryHDU(data=data.astype(np.float32), header=hdr).writeto(path, overwrite=True)


def test_default_logger_no_duplicate_handlers():
    lg1 = _get_default_logger()
    n1 = len(lg1.handlers)
    lg2 = _get_default_logger()
    n2 = len(lg2.handlers)
    assert lg1 is lg2
    assert n2 == n1  # 여러 번 불러도 핸들러 중복 추가 X


def test_process_file_subtracts_and_crops(tmp_path):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)

    ast = GFAAstrometry(config=str(cfgp), logger=_get_default_logger())

    raw_dir = Path(ast.dir_path)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 4x4 데이터
    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    raw_path = raw_dir / "img.fits"
    _write_raw_fits(raw_path, data, ra=111.0, dec=-22.0)

    # process_file은 self.input_paths에서 full path를 찾음
    ast.input_paths = [str(raw_path)]

    ra_in, dec_in, dir_out, newname = ast.process_file("img.fits")
    assert ra_in == 111.0
    assert dec_in == -22.0
    assert newname == "proc_img.fits"

    out_path = Path(dir_out) / newname
    assert out_path.exists()

    out = fits.getdata(out_path).astype(np.float32)

    # 기대 결과 계산:
    # sky1 = ori[0,0] = 0
    # sky2 = ori[0,1] = 1
    expected = data.copy()
    expected[0:2, 0:2] -= 0
    expected[2:4, 2:4] -= 1
    expected = expected[1:4, 1:4]  # crop => 3x3

    assert out.shape == (3, 3)
    assert np.allclose(out, expected)


def test_process_file_invalid_skycoord_raises(tmp_path):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)

    # config 깨기: skycoord가 범위를 벗어나게
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


def test_astrometry_reads_crvals_and_moves_file(tmp_path, monkeypatch):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)

    ast = GFAAstrometry(config=str(cfgp), logger=_get_default_logger())

    # processed 파일 준비
    proc_dir = Path(ast.processed_dir)
    proc_dir.mkdir(parents=True, exist_ok=True)
    newname = "proc_img.fits"
    proc_path = proc_dir / newname
    fits.writeto(proc_path, np.zeros((2, 2), dtype=np.float32), overwrite=True)

    # solve-field 존재한다고 가정
    monkeypatch.setattr("gfa_astrometry.shutil.which", lambda name: r"C:\fake\solve-field.exe")

    # subprocess.run은 실제 실행 대신 "성공" 처리
    def _fake_run(cmd, shell, capture_output, text, check=False):
        class R:
            stdout = ""
            stderr = ""
        return R()
    monkeypatch.setattr("gfa_astrometry.subprocess.run", _fake_run)

    # solve-field 결과물(.new)을 temp_dir에 미리 만들어 둔다 (astrometry()가 rename/move 할 것)
    temp_dir = Path(ast.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    new_path = temp_dir / newname.replace(".fits", ".new")

    hdr = fits.Header()
    hdr["CRVAL1"] = 123.456
    hdr["CRVAL2"] = -78.9
    fits.PrimaryHDU(data=np.zeros((2, 2), dtype=np.float32), header=hdr).writeto(new_path, overwrite=True)

    c1, c2 = ast.astrometry(ra_in=10.0, dec_in=20.0, dir_out=str(proc_dir), newname=newname)
    assert c1 == 123.456
    assert c2 == -78.9

    # 최종 이동 경로 확인
    final_dir = Path(ast.final_astrometry_dir)
    dest = final_dir / f"astro_{newname}"
    assert dest.exists()


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


def test_preproc_branch_full_astrometry_calls_combined_star_rm(tmp_path, monkeypatch):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    ast = GFAAstrometry(config=str(cfgp), logger=_get_default_logger())

    # 입력 FITS 두 개
    raw_dir = Path(ast.dir_path)
    raw_dir.mkdir(parents=True, exist_ok=True)
    p1 = raw_dir / "i1.fits"
    p2 = raw_dir / "i2.fits"
    _write_raw_fits(p1, np.zeros((4, 4), dtype=np.float32))
    _write_raw_fits(p2, np.zeros((4, 4), dtype=np.float32))

    calls = {"combined": [], "star": 0, "rm": 0}

    monkeypatch.setattr(ast, "combined_function", lambda fl: (1.0, 2.0) if not calls["combined"].append(fl) else (1.0, 2.0))
    monkeypatch.setattr(ast, "star_catalog", lambda: calls.__setitem__("star", calls["star"] + 1))
    monkeypatch.setattr(ast, "rm_tempfiles", lambda: calls.__setitem__("rm", calls["rm"] + 1))

    # final dir을 비워둬서 "full astrometry" 분기로 들어가게
    final_dir = Path(ast.final_astrometry_dir)
    final_dir.mkdir(parents=True, exist_ok=True)
    for f in final_dir.glob("*"):
        f.unlink()

    ast.preproc(input_files=[p1, p2])

    assert sorted(calls["combined"]) == ["i1.fits", "i2.fits"]
    assert calls["star"] == 1
    assert calls["rm"] == 1


def test_preproc_branch_existing_astrometry_calls_process_only(tmp_path, monkeypatch):
    cfgp = tmp_path / "cfg.json"
    _write_config(cfgp, tmp_path)
    ast = GFAAstrometry(config=str(cfgp), logger=_get_default_logger())

    raw_dir = Path(ast.dir_path)
    raw_dir.mkdir(parents=True, exist_ok=True)
    p1 = raw_dir / "i1.fits"
    p2 = raw_dir / "i2.fits"
    _write_raw_fits(p1, np.zeros((4, 4), dtype=np.float32))
    _write_raw_fits(p2, np.zeros((4, 4), dtype=np.float32))

    # final dir에 뭔가 하나라도 있으면 "이미 astrometry 있음" 분기
    final_dir = Path(ast.final_astrometry_dir)
    final_dir.mkdir(parents=True, exist_ok=True)
    (final_dir / "dummy.txt").write_text("x", encoding="utf-8")

    calls = {"proc": []}
    monkeypatch.setattr(ast, "process_file", lambda fl: calls["proc"].append(fl) or (0, 0, "", ""))

    ast.preproc(input_files=[p1, p2])

    assert sorted(calls["proc"]) == ["i1.fits", "i2.fits"]
