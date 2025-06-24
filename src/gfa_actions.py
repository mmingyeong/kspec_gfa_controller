#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2024-05-16
# @Filename: gfa_actions.py

import os
import asyncio
from typing import Union, List, Dict, Any, Optional
from datetime import datetime

from gfa_logger import GFALogger
from gfa_controller import GFAController
from gfa_astrometry import GFAAstrometry
from gfa_guider import GFAGuider

###############################################################################
# Global Config Paths
###############################################################################
gfa_relative_config_path = "etc/cams.json"
ast_relative_config_path = "etc/astrometry_params.json"

###############################################################################
# Logger
###############################################################################
logger = GFALogger(__file__)

###############################################################################
# Helper Functions
###############################################################################
def get_config_path(relative_config_path: str) -> str:
    """
    Calculate and return the absolute path of the configuration file.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, relative_config_path)
    if not os.path.isfile(config_path):
        logger.error(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")
    logger.info(f"Configuration file found: {config_path}")
    return config_path

###############################################################################
# Environment Class for Dependency Injection
###############################################################################
class GFAEnvironment:
    """
    Holds references to controller, astrometry, and guider,
    as well as paths, logger, and camera count.
    """
    def __init__(self,
                 gfa_config_path: str,
                 ast_config_path: str,
                 logger,
                 camera_count: int = 7):
        self.gfa_config_path = gfa_config_path
        self.ast_config_path = ast_config_path
        self.logger = logger
        self.camera_count = camera_count # Cam0 포함한 전체 카메라 수

        # Initialize dependencies
        self.controller = GFAController(self.gfa_config_path, self.logger)
        # --- OPEN ALL CAMERAS ONCE AT STARTUP ---
        self.controller.open_all_cameras()

        self.astrometry = GFAAstrometry(self.ast_config_path, self.logger)
        self.guider = GFAGuider(self.ast_config_path, self.logger)

    def shutdown(self):
        """Cleanly close all cameras before exit."""
        self.logger.info("Shutting down environment: closing all cameras.")
        self.controller.close_all_cameras()

def create_environment() -> GFAEnvironment:
    """
    Creates and returns a GFAEnvironment object with the default config paths
    and logger, and opens all cameras immediately.
    """
    gfa_config_path = get_config_path(gfa_relative_config_path)
    ast_config_path = get_config_path(ast_relative_config_path)
    env = GFAEnvironment(gfa_config_path, ast_config_path, logger, camera_count=6)
    return env

###############################################################################
# GFA Actions Class
###############################################################################
class GFAActions:
    """
    A class to handle GFA actions such as grabbing images, guiding,
    and controlling the cameras.
    """

    def __init__(self, env: Optional[GFAEnvironment] = None):
        if env is None:
            env = create_environment()
        self.env = env

    def _generate_response(self, status: str, message: str, **kwargs) -> dict:
        response = {"status": status, "message": message}
        response.update(kwargs)
        return response

    async def grab(
        self,
        CamNum: Union[int, List[int]] = -2,
        ExpTime: float = 1.0,
        Binning: int = 4,
        *,
        packet_size: int = 8192,
        cam_ipd: int = 367318,
        cam_ftd_base: int = 0
    ) -> Dict[str, Any]:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        date_str = datetime.now().strftime("%Y-%m-%d")
        grab_save_path = os.path.join(base_dir, "img", "grab", date_str)
        self.env.logger.info(f"Image save path: {grab_save_path}")

        try:
            camera_ids = self._parse_camnum(CamNum)
            self.env.logger.info(f"Grabbing images from cameras: {camera_ids}")

            # Call controller.grab() which already uses asyncio.gather()
            timeout_cameras = await self.env.controller.grab(
                CamNum=camera_ids,
                ExpTime=ExpTime,
                Binning=Binning,
                packet_size=packet_size,
                ipd=cam_ipd,
                ftd_base=cam_ftd_base,
                output_dir=grab_save_path,
            )

            msg = f"Images grabbed from cameras {camera_ids}."
            if timeout_cameras:
                msg += f" Timeout: {timeout_cameras}"
            return self._generate_response("success", msg)

        except Exception as e:
            self.env.logger.error(f"Error in grab(): {e}")
            return self._generate_response(
                "error",
                f"Error in grab(): {e} (CamNum={CamNum}, ExpTime={ExpTime}, Binning={Binning}, PacketSize={packet_size}, IPD={cam_ipd}, FTD_Base={cam_ftd_base})"
            )


    async def guiding(self, ExpTime: float = 1.0, save: bool = False) -> Dict[str, Any]:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        date_str = datetime.now().strftime("%Y-%m-%d")

        raw_save_path = os.path.join(base_dir, "img", "raw")
        grab_save_path = os.path.join(base_dir, "img", "grab", date_str)

        try:
            self.env.logger.info("Guiding starts...")

            # Always perform grab
            self.env.logger.info("Grabbing image from controller...")
            os.makedirs(raw_save_path, exist_ok=True)
            #await self.env.controller.grab(-2, ExpTime, 4, output_dir=raw_save_path)
            self.env.logger.info(f"Image saved to: {raw_save_path}")

            # If save is True, also copy to grab path
            if save:
                os.makedirs(grab_save_path, exist_ok=True)
                for fname in os.listdir(raw_save_path):
                    src = os.path.join(raw_save_path, fname)
                    dst = os.path.join(grab_save_path, fname)
                    if os.path.isfile(src):
                        shutil.copy2(src, dst)
                self.env.logger.info(f"Image(s) additionally saved to: {grab_save_path}")

            self.env.logger.info("Astrometry preprocessing...")
            self.env.astrometry.preproc()

            self.env.logger.info("Calculating guider offsets...")
            fdx, fdy, fwhm = self.env.guider.exe_cal()

            self.env.logger.info("Clearing raw and processed files after guiding...")
            self.env.astrometry.clear_raw_and_processed_files()

            msg = f"Offsets: fdx={fdx}, fdy={fdy}, FWHM={fwhm:.5f} arcsec"
            return self._generate_response("success", msg, fdx=fdx, fdy=fdy, fwhm=fwhm)

        except Exception as e:
            self.env.logger.error(f"Error during guiding: {e}")
            return self._generate_response("error", f"Error during guiding: {e}")

    def status(self, CamNum: Union[int, List[int]] = -2) -> Dict[str, Any]:
        try:
            camera_ids = self._parse_camnum(CamNum)
            self.env.logger.info(f"Checking status for cameras: {camera_ids}")

            msg_lines = []
            for cam in camera_ids:
                status = self.env.controller.status(cam)
                msg_lines.append(f"Cam{cam}: {status}")
            return self._generate_response("success", "\n".join(msg_lines))

        except Exception as e:
            self.env.logger.error(f"Error checking status: {e}")
            return self._generate_response("error", f"Error checking status: {e}")

    def ping(self, CamNum: Union[int, List[int]] = -2) -> Dict[str, Any]:
        try:
            camera_ids = self._parse_camnum(CamNum)
            self.env.logger.info(f"Pinging cameras: {camera_ids}")

            for cam in camera_ids:
                self.env.logger.info(f"Pinging Cam{cam}...")
                self.env.controller.ping(cam)

            return self._generate_response("success", f"Pinged cameras: {camera_ids}")

        except Exception as e:
            self.env.logger.error(f"Error pinging camera(s): {e}")
            return self._generate_response("error", f"Error pinging camera(s): {e}")

    def cam_params(self, CamNum: Union[int, List[int]] = -2) -> Dict[str, Any]:
        try:
            camera_ids = self._parse_camnum(CamNum)
            self.env.logger.info(f"Retrieving parameters for cameras: {camera_ids}")

            msg_lines = []
            for cam in camera_ids:
                self.env.logger.info(f"Retrieving parameters for Cam{cam}")
                param = self.env.controller.cam_params(cam)
                msg_lines.append(f"Cam{cam}: {param}")

            return self._generate_response("success", "\n".join(msg_lines))

        except Exception as e:
            self.env.logger.error(f"Error retrieving parameters: {e}")
            return self._generate_response("error", f"Error retrieving parameters: {e}")


    def shutdown(self) -> None:
        """
        Call this to cleanly close cameras before exiting.
        """
        self.env.shutdown()
        self.env.logger.info("GFAActions shutdown complete.")

    def _parse_camnum(self, CamNum) -> List[int]:
        valid_ids = [0] + list(range(1, self.env.camera_count))  # Cam0~6 only

        if isinstance(CamNum, int):
            if CamNum == -2:
                return valid_ids
            elif CamNum == -1:
                return valid_ids[1:]  # Only Cam1~6
            elif CamNum in valid_ids:
                return [CamNum]
            else:
                raise ValueError(f"Invalid CamNum: {CamNum}")
        elif isinstance(CamNum, list):
            for cam in CamNum:
                if cam not in valid_ids:
                    raise ValueError(f"Invalid camera ID in list: {cam}")
            return CamNum
        else:
            raise TypeError("CamNum must be int or list")
