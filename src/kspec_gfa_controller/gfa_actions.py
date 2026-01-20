#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import asyncio
import shutil
from datetime import datetime
from typing import Union, List, Dict, Any, Optional
from pathlib import Path
import re

from .gfa_logger import GFALogger
from .gfa_environment import create_environment, GFAEnvironment
from .gfa_getcrval import get_crvals_from_images, get_crval_from_image

logger = GFALogger(__file__)


def _make_clean_subprocess_env() -> dict:
    """
    외부 프로세스(solve-field 등)가 현재 프로세스의 env를 상속받아 깨지는 문제 방지용.
    - PYTHONHOME/PYTHONPATH 제거
    - 현재 실행 중인 파이썬 bin 경로를 PATH 최우선으로 배치 (conda/venv 대응)
    """
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)

    pybin = os.path.dirname(os.path.realpath(os.sys.executable))
    env["PATH"] = pybin + os.pathsep + env.get("PATH", "")
    return env

from pathlib import Path
import re
import numpy as np
from astropy.io import fits

# dark 폴더(고정). 여기 아래에 exp1s/exp2s... 등 dark들이 있다고 가정
DARK_DIR = Path("/home/GAFOL/work/kspec_gfa_controller/src/kspec_gfa_controller/img/dark")

# 파일명 예: D20260119_T141119_40103667_exp3s.fits
# camera_id=40103667, exp_token=exp3s
FNAME_RE = re.compile(r".*_(\d+)_((?:exp|EXP)[0-9p]+s)\.fits?$", re.IGNORECASE)

def _extract_key_from_filename(path: Path):
    m = FNAME_RE.match(path.name)
    if not m:
        return None
    cam_id = m.group(1)
    exp_token = m.group(2).lower()  # exp3s 형태로 통일
    return cam_id, exp_token

def _extract_exp_from_header(header):
    # 헤더 키는 장비/파이프라인마다 다를 수 있어 후보를 여러 개 둠
    for k in ("EXPTIME", "EXPOSURE", "EXP_TIME", "ITIME"):
        if k in header:
            try:
                return float(header[k])
            except Exception:
                pass
    return None

def _exp_to_token(exptime: float):
    # 3.0 -> exp3s, 0.5 -> exp0p5s 처럼 변환
    if exptime is None:
        return None
    if abs(exptime - round(exptime)) < 1e-6:
        return f"exp{int(round(exptime))}s"
    # 소수점은 p로 치환(0.5 => 0p5)
    s = str(exptime).replace(".", "p")
    return f"exp{s}s"

def _index_dark_files(dark_dir: Path):
    exts = (".fits", ".fit", ".fts")
    index = {}  # (cam_id, exp_token) -> [Path, Path, ...]
    for p in dark_dir.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        key = _extract_key_from_filename(p)
        if key is None:
            continue
        index.setdefault(key, []).append(p)
    return index

def _load_master_dark(dark_paths, cache_path: Path):
    """
    dark_paths가 여러 개면 median master를 만들어 cache_path에 저장.
    cache_path가 이미 있으면 그대로 로드.
    """
    if cache_path.exists():
        with fits.open(cache_path, memmap=False) as hd:
            return hd[0].data.astype(np.float32)

    # dark 여러 장 로드 -> (N, H, W) 스택 후 median
    stack = []
    for dp in dark_paths:
        with fits.open(dp, memmap=False) as hd:
            d = hd[0].data
        if d is None:
            continue
        stack.append(d.astype(np.float32))

    if not stack:
        raise ValueError("유효한 dark 데이터가 없습니다.")

    master = np.median(np.stack(stack, axis=0), axis=0).astype(np.float32)

    # 캐시에 저장(헤더는 첫 번째 dark의 헤더를 참고)
    with fits.open(dark_paths[0], memmap=False) as hd0:
        hdr0 = hd0[0].header
    hdu = fits.PrimaryHDU(data=master, header=hdr0)
    hdu.header["N_DARK"] = (len(stack), "Number of dark frames combined (median)")
    hdu.writeto(cache_path, overwrite=True)

    return master

def raw_dark(image_path, output_dir="pointing_raw_dark", suffix="_darksub", clip_to_zero=True):
    """
    입력 FITS(단일/폴더) 각각에 대해,
    (camera_id + expNs) 가 동일한 dark를 찾아 빼고 저장한 뒤 결과 경로를 반환.

    Returns
    -------
    list[Path] : 저장된 결과 FITS 경로 리스트
    """

    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not DARK_DIR.exists():
        raise FileNotFoundError(f"Dark 폴더를 찾을 수 없습니다: {DARK_DIR}")

    # dark 인덱스 생성: (cam_id, exp_token) -> dark 파일들
    dark_index = _index_dark_files(DARK_DIR)

    # 입력 목록 구성
    exts = (".fits", ".fit", ".fts")
    if image_path.is_file():
        input_files = [image_path]
    else:
        input_files = sorted([p for p in image_path.iterdir() if p.is_file() and p.suffix.lower() in exts])

    # master dark 캐시 폴더
    cache_dir = output_dir / "_master_dark_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    output_paths = []

    for f in input_files:
        with fits.open(f, memmap=False) as hdul:
            data = hdul[0].data
            header = hdul[0].header

        if data is None:
            continue

        # 1) 파일명에서 (cam_id, exp_token) 추출 시도
        key = _extract_key_from_filename(f)

        # 2) 파일명에서 못 뽑으면, 헤더 EXPTIME으로 exp_token 만들고 cam_id는 파일명에서 숫자 찾아보기
        if key is None:
            exptime = _extract_exp_from_header(header)
            exp_token = _exp_to_token(exptime)

            # cam_id: 파일명에서 _<digits>_ 형태 숫자 하나를 대충 찾는 fallback
            cam_m = re.search(r"_(\d{6,})_", f.name)
            cam_id = cam_m.group(1) if cam_m else None

            if cam_id and exp_token:
                key = (cam_id, exp_token)

        if key is None:
            raise ValueError(f"입력 파일에서 camera/exp 정보를 추출하지 못했습니다: {f.name}")

        cam_id, exp_token = key

        # 대응되는 dark 후보 찾기
        dark_candidates = dark_index.get((cam_id, exp_token), [])
        if not dark_candidates:
            raise FileNotFoundError(
                f"대응 dark를 찾을 수 없습니다: cam={cam_id}, exp={exp_token}\n"
                f"dark 폴더({DARK_DIR})에 같은 패턴의 파일이 있어야 합니다."
            )

        # 여러 장이면 median master dark 생성(캐시)
        cache_path = cache_dir / f"master_dark_{cam_id}_{exp_token}.fits"
        master_dark = _load_master_dark(dark_candidates, cache_path)

        # shape 검사
        if data.shape != master_dark.shape:
            raise ValueError(
                f"shape 불일치: {f.name} {data.shape} vs dark {master_dark.shape} "
                f"(cam={cam_id}, exp={exp_token})"
            )

        # subtraction
        corrected = data.astype(np.float32) - master_dark
        if clip_to_zero:
            corrected = np.clip(corrected, 0, None)

        # 저장
        out_name = f"{f.stem}{suffix}{f.suffix}"
        out_path = output_dir / out_name

        hdu = fits.PrimaryHDU(data=corrected, header=header)
        hdu.header["DARKSUB"] = (True, "Dark frame subtracted")
        hdu.header["DARKCAM"] = (cam_id, "Camera id used for dark subtraction")
        hdu.header["DARKEXP"] = (exp_token, "Exposure token used for dark subtraction")
        hdu.header["DARKMAST"] = (str(cache_path.name), "Master dark filename (cached)")
        hdu.header.add_history(f"Dark subtracted: cam={cam_id}, exp={exp_token}")
        hdu.writeto(out_path, overwrite=True)

        output_paths.append(out_path)

    return output_paths


class GFAActions:
    """
    GFA actions: grab, guiding, pointing, camera status utilities.
    """

    def __init__(self, env: Optional[GFAEnvironment] = None):
        if env is None:
            env = create_environment(role="plate")
        self.env = env

    def _generate_response(self, status: str, message: str, **kwargs) -> dict:
        response = {"status": status, "message": message}
        response.update(kwargs)
        return response

    async def grab(
        self,
        CamNum: Union[int, List[int]] = 0,
        ExpTime: float = 1.0,
        Binning: int = 4,
        *,
        packet_size: int = None,
        cam_ipd: int = None,
        cam_ftd_base: int = 0,
        ra: str = None,
        dec: str = None,
        path: str = None
    ) -> Dict[str, Any]:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        date_str = datetime.now().strftime("%Y-%m-%d")
        if path:
            grab_save_path = path
        else:
            grab_save_path = os.path.join(base_dir, "img", "grab", date_str)
        os.makedirs(grab_save_path, exist_ok=True)

        timeout_cameras: List[int] = []

        self.env.logger.info("Open all plate cameras...")
        await self.env.controller.open_all_cameras()

        try:
            if isinstance(CamNum, int) and CamNum != 0:
                res = await self.env.controller.grabone(
                    CamNum=CamNum,
                    ExpTime=ExpTime,
                    Binning=Binning,
                    output_dir=grab_save_path,
                    packet_size=packet_size,
                    ipd=cam_ipd,
                    ftd_base=cam_ftd_base,
                    ra=ra,
                    dec=dec,
                )
                timeout_cameras.extend(res)

                msg = f"Image grabbed from camera {CamNum}."
                if timeout_cameras:
                    msg += f" Timeout: {timeout_cameras[0]}"
                return self._generate_response("success", msg)

            if isinstance(CamNum, int) and CamNum == 0:
                tasks = [
                    self.env.controller.grabone(
                        CamNum=cam_id,
                        ExpTime=ExpTime,
                        Binning=Binning,
                        output_dir=grab_save_path,
                        packet_size=packet_size,
                        ipd=cam_ipd,
                        ftd_base=cam_ftd_base,
                        ra=ra,
                        dec=dec,
                    )
                    for cam_id in self.env.camera_ids
                ]
                results = await asyncio.gather(*tasks)
                for r in results:
                    timeout_cameras.extend(r)

                msg = "Images grabbed from all cameras."
                if timeout_cameras:
                    msg += f" Timeout: {timeout_cameras}"
                return self._generate_response("success", msg)

            if isinstance(CamNum, list):
                tasks = [
                    self.env.controller.grabone(
                        CamNum=cam_id,
                        ExpTime=ExpTime,
                        Binning=Binning,
                        output_dir=grab_save_path,
                        packet_size=packet_size,
                        ipd=cam_ipd,
                        ftd_base=cam_ftd_base,
                        ra=ra,
                        dec=dec,
                    )
                    for cam_id in CamNum
                ]
                results = await asyncio.gather(*tasks)
                for r in results:
                    timeout_cameras.extend(r)

                msg = f"Images grabbed from cameras {CamNum}."
                if timeout_cameras:
                    msg += f" Timeout: {timeout_cameras}"
                return self._generate_response("success", msg)

            raise ValueError(f"Invalid CamNum: {CamNum}")

        except Exception as e:
            self.env.logger.error(f"Grab failed: {e}")
            return self._generate_response("error", f"Grab failed: {e}")

        finally:
            self.env.logger.info("Close all plate cameras...")
            try:
                await self.env.controller.close_all_cameras()
            except Exception as e:
                self.env.logger.warning(f"close_all_cameras failed: {e}")

    async def guiding(
        self,
        ExpTime: float = 1.0,
        save: bool = False,
        ra: str = None,
        dec: str = None,
    ) -> Dict[str, Any]:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        date_str = datetime.now().strftime("%Y-%m-%d")

        raw_save_path = os.path.join(base_dir, "img", "raw")
        grab_save_path = os.path.join(base_dir, "img", "grab", date_str)
        os.makedirs(raw_save_path, exist_ok=True)

        try:
            self.env.logger.info("Starting guiding sequence...")

            await self.env.controller.open_all_cameras()
            try:
            # output_dir는 실제 raw_save_path로 통일하는 걸 권장
                await self.env.controller.grab(
                    0, ExpTime, 4,
                    output_dir="./img/raw",
                    ra=ra, dec=dec
                )
            finally:
                try:
                    await self.env.controller.close_all_cameras()
                except Exception as e:
                    self.env.logger.warning(f"close_all_cameras failed: {e}")

            if save:
                os.makedirs(grab_save_path, exist_ok=True)
                for fname in os.listdir(raw_save_path):
                    src = os.path.join(raw_save_path, fname)
                    dst = os.path.join(grab_save_path, fname)
                    if os.path.isfile(src):
                        shutil.copy2(src, dst)

            # --- astrometry: clean env 적용 (해결책 핵심) ---
            clean_env = _make_clean_subprocess_env()
            if hasattr(self.env.astrometry, "set_subprocess_env"):
                self.env.astrometry.set_subprocess_env(clean_env)

            self.env.logger.info("Running astrometry preprocessing...")
            self.env.astrometry.preproc()

            self.env.logger.info("Executing guider offset calculation...")
            fdx, fdy, fwhm = self.env.guider.exe_cal()
            
            #self.env.astrometry.clear_raw_and_processed_files()

            try:
                fwhm_val = float(fwhm)
            except Exception:
                fwhm_val = 0.0

            msg = f"Offsets: fdx={fdx}, fdy={fdy}, FWHM={fwhm_val} arcsec"
            return self._generate_response(
                "success", msg, fdx=fdx, fdy=fdy, fwhm=fwhm_val
            )

        except Exception as e:
            self.env.logger.error(f"Guiding failed: {str(e)}")
            return self._generate_response("error", f"Guiding failed: {str(e)}")


    async def guiding_from_saved_grab(
        self,
        *,
        grab_save_path: str = None,
        save: bool = False,
        ra: str = None,
        dec: str = None,
        num_images: int = 6,
    ) -> Dict[str, Any]:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        date_str = datetime.now().strftime("%Y-%m-%d")

        raw_save_path = os.path.join(base_dir, "img", "raw")
        default_grab_save_path = os.path.join(base_dir, "img", "grab", date_str)
        grab_save_path = grab_save_path or default_grab_save_path
        os.makedirs(raw_save_path, exist_ok=True)

        # 예: D20260116_T124224_40103651_exp3s.fits
        # key = "20260116_124224"
        ts_re = re.compile(r"D(\d{8})_T(\d{6})_", re.IGNORECASE)

        try:
            self.env.logger.info("Starting guiding_from_saved_grab (same T-set) ...")
            self.env.logger.info(f"Using grab_save_path: {grab_save_path}")

            if not os.path.isdir(grab_save_path):
                msg = f"grab_save_path does not exist or is not a directory: {grab_save_path}"
                self.env.logger.error(msg)
                return self._generate_response("error", msg)

            # 1) 세트(T)가 같은 것끼리 그룹화
            groups = {}  # key -> [fullpath...]
            unparsable = []

            for fn in os.listdir(grab_save_path):
                fp = os.path.join(grab_save_path, fn)
                if not os.path.isfile(fp):
                    continue
                if not fn.lower().endswith((".fits", ".fit", ".fts")):
                    continue

                m = ts_re.search(fn)
                if not m:
                    unparsable.append(fn)
                    continue

                key = f"{m.group(1)}_{m.group(2)}"  # YYYYMMDD_HHMMSS
                groups.setdefault(key, []).append(fp)

            if not groups:
                msg = "No FITS images with parsable DYYYYMMDD_Txxxxxx pattern found."
                self.env.logger.error(msg)
                return self._generate_response("error", msg, unparsable=unparsable[:20])

            # 2) 가장 최근 key 선택 (문자열 정렬이 시간 정렬과 동일)
            latest_key = sorted(groups.keys())[-1]
            latest_files = groups[latest_key]

            if len(latest_files) < num_images:
                msg = f"Latest set {latest_key} has only {len(latest_files)} files (<{num_images})."
                self.env.logger.error(msg)
                return self._generate_response(
                    "error",
                    msg,
                    latest_key=latest_key,
                    latest_count=len(latest_files),
                    latest_files=[os.path.basename(p) for p in sorted(latest_files)],
                )

            # 3) 최신 세트 안에서 6개 선택
            # 세트 내 파일은 이름으로 정렬하면 (4010...) 순서가 안정적이라 재현성 좋음
            latest_files_sorted = sorted(latest_files)
            selected = latest_files_sorted[:num_images]

            self.env.logger.info(
                f"Selected set key={latest_key}, files=" +
                ", ".join(os.path.basename(p) for p in selected)
            )

            # 4) raw 폴더 정리(파일/링크만) 후 복사
            for name in os.listdir(raw_save_path):
                p = os.path.join(raw_save_path, name)
                if os.path.isfile(p) or os.path.islink(p):
                    try:
                        os.remove(p)
                    except Exception as e:
                        self.env.logger.warning(f"Failed to remove raw file {p}: {e}")

            copied_names = []
            for src in selected:
                dst = os.path.join(raw_save_path, os.path.basename(src))
                shutil.copy2(src, dst)
                copied_names.append(os.path.basename(src))

            if save:
                # 이미 grab_save_path가 존재/저장된 상태라 특별 동작 없음
                pass

            # --- guiding()와 동일: clean env 적용 ---
            clean_env = _make_clean_subprocess_env()
            if hasattr(self.env.astrometry, "set_subprocess_env"):
                self.env.astrometry.set_subprocess_env(clean_env)

            # 5) preproc -> exe_cal
            self.env.logger.info("Running astrometry preprocessing...")
            self.env.astrometry.preproc()

            self.env.logger.info("Executing guider offset calculation...")
            fdx, fdy, fwhm = self.env.guider.exe_cal()

            try:
                fwhm_val = float(fwhm)
            except Exception:
                fwhm_val = 0.0

            msg = f"Offsets: fdx={fdx}, fdy={fdy}, FWHM={fwhm_val} arcsec"
            return self._generate_response(
                "success",
                msg,
                fdx=fdx,
                fdy=fdy,
                fwhm=fwhm_val,
                grabbed_from=grab_save_path,
                selected_set_key=latest_key,
                selected_images=copied_names,
            )

        except Exception as e:
            self.env.logger.error(f"guiding_from_saved_grab failed: {str(e)}")
            return self._generate_response("error", f"guiding_from_saved_grab failed: {str(e)}")

        finally:
            # 요청: gfa_astrometry class 함수 실행
            try:
                if hasattr(self.env.astrometry, "Clear_raw_and_processed_files"):
                    self.env.astrometry.Clear_raw_and_processed_files()
                else:
                    self.env.logger.warning("env.astrometry has no Clear_raw_and_processed_files()")
            except Exception as e:
                self.env.logger.warning(f"Clear_raw_and_processed_files failed: {e}")


    async def pointing(
        self,
        ra: str,
        dec: str,
        ExpTime: float = 1.0,
        Binning: int = 4,
        CamNum: int = 0,
        max_workers: int = 4,
        save_by_date: bool = True,
        clear_dir: bool = True,
    ) -> Dict[str, Any]:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        pointing_raw_path="/home/GAFOL/work/kspec_gfa_controller/src/kspec_gfa_controller/img/pointing_raw/2026-01-19"

        #pointing_raw_path = (
        #    os.path.join(base_dir, "img", "pointing_raw", date_str)
        #    if save_by_date else
        #    os.path.join(base_dir, "img", "pointing_raw")
        #)
        os.makedirs(pointing_raw_path, exist_ok=True)

        # ✅ dark-sub 결과 저장 폴더 (원하는 경로로 바꿔도 됨)
        pointing_dark_path = (
            os.path.join(base_dir, "img", "pointing_raw_dark", date_str)
            if save_by_date else
            os.path.join(base_dir, "img", "pointing_raw_dark")
        )
        os.makedirs(pointing_dark_path, exist_ok=True)

        try:
            self.env.logger.info("Starting pointing sequence...")
            self.env.logger.info(f"Target RA/DEC: {ra}, {dec}")

            # --- (선택) raw 폴더 정리 ---
            # if clear_dir:
            #     for fn in os.listdir(pointing_raw_path):
            #         fp = os.path.join(pointing_raw_path, fn)
            #         if os.path.isfile(fp):
            #             os.remove(fp)

            # --- (원래 주석 처리된 grab 로직을 쓰려면 여기 활성화) ---
            # await self.env.controller.open_all_cameras()
            # try:
            #     await self.env.controller.grab(
            #         CamNum, ExpTime, Binning,
            #         output_dir=pointing_raw_path,
            #         ra=ra, dec=dec
            #     )
            # finally:
            #     try:
            #         await self.env.controller.close_all_cameras()
            #     except Exception as e:
            #         self.env.logger.warning(f"close_all_cameras failed: {e}")

            # ✅ 1) dark subtraction 실행
            # raw_dark()는 list[Path]를 반환함!
            dark_sub_paths = raw_dark(
                pointing_raw_path,
                output_dir=pointing_dark_path,
                suffix="_darksub",
                clip_to_zero=True,
            )

            # ✅ 2) FITS만 필터링해서 Path 리스트 구성
            image_list = [
                Path(p) for p in dark_sub_paths
                if str(p).lower().endswith((".fits", ".fit", ".fts"))
            ]

            if not image_list:
                msg = f"No FITS images found after dark subtraction. input={pointing_raw_path}, output={pointing_dark_path}"
                self.env.logger.error(msg)
                return self._generate_response("error", msg, images=[], crval1=[], crval2=[])

            # --- clean env 적용 (solve-field 등 외부 실행 깨짐 방지) ---
            clean_env = _make_clean_subprocess_env()
            if hasattr(self.env.astrometry, "set_subprocess_env"):
                self.env.astrometry.set_subprocess_env(clean_env)

            self.env.logger.info(
                f"Solving astrometry for CRVALs using {len(image_list)} dark-subtracted images (max_workers={max_workers})..."
            )

            # ✅ 3) CRVAL 계산: dark-subtracted 이미지들로 수행
            crval1_list, crval2_list = get_crvals_from_images(
                image_list,
                max_workers=max_workers,
            )

            # ✅ 4) 응답은 파일명만 반환 (기존 유지)
            image_names = [p.name for p in image_list]

            msg = f"Pointing completed. Computed CRVALs for {len(image_list)} images."
            return self._generate_response(
                "success",
                msg,
                images=image_names,
                crval1=crval1_list,
                crval2=crval2_list,
                dark_output_dir=pointing_dark_path,   # 디버깅/사용 편의를 위해 추가
            )

        except Exception as e:
            self.env.logger.error(f"Pointing failed: {str(e)}")
            return self._generate_response("error", f"Pointing failed: {str(e)}")


    def status(self) -> Dict[str, Any]:
        try:
            status_info = self.env.controller.status()
            return self._generate_response("success", status_info)
        except Exception as e:
            return self._generate_response("error", f"Status query failed: {e}")

    def ping(self, CamNum: int = 0) -> Dict[str, Any]:
        try:
            if CamNum == 0:
                for cam_id in self.env.camera_ids:
                    self.env.controller.ping(cam_id)
                return self._generate_response("success", "Pinged all cameras.")
            else:
                self.env.controller.ping(CamNum)
                return self._generate_response("success", f"Pinged Cam{CamNum}.")
        except Exception as e:
            return self._generate_response("error", f"Ping failed: {e}")

    def cam_params(self, CamNum: int = 0) -> Dict[str, Any]:
        try:
            if CamNum == 0:
                messages = []
                for cam_id in self.env.camera_ids:
                    param = self.env.controller.cam_params(cam_id)
                    messages.append(f"Cam{cam_id}: {param}")
                return self._generate_response("success", "\n".join(messages))
            else:
                param = self.env.controller.cam_params(CamNum)
                return self._generate_response("success", f"Cam{CamNum}: {param}")
        except Exception as e:
            return self._generate_response("error", f"Parameter fetch failed: {e}")

    def shutdown(self) -> None:
        self.env.shutdown()
        self.env.logger.info("GFAActions shutdown complete.")
