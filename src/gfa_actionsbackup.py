#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2024-05-16
# @Filename: gfa_actions.py

import os
import asyncio
from typing import Union, List, Dict, Any, Optional

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
                 camera_count: int = 6):
        self.gfa_config_path = gfa_config_path
        self.ast_config_path = ast_config_path
        self.logger = logger
        self.camera_count = camera_count

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
        CamNum: Union[int, List[int]] = 0,
        ExpTime: float = 1.0,
        Binning: int = 4
    ) -> Dict[str, Any]:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        grab_save_path = os.path.join(base_dir, "img", "grab")
        self.env.logger.info(f"Image save path: {grab_save_path}")

        timeout_cameras = []
        try:
            if isinstance(CamNum, int):
                if CamNum == 0:
                    self.env.logger.info(
                        f"Grabbing image from ALL cameras (ExpTime={ExpTime}, Binning={Binning})."
                    )
                    timeout_cameras = await self.env.controller.grab(
                        CamNum, ExpTime, Binning, output_dir=grab_save_path
                    )
                    msg = f"Images grabbed from all cameras (ExpTime={ExpTime}, Binning={Binning})."
                    if timeout_cameras:
                        msg += f" Timeout: {timeout_cameras}"
                    return self._generate_response("success", msg)
                else:
                    self.env.logger.info(
                        f"Grabbing image from camera {CamNum} (ExpTime={ExpTime}, Binning={Binning})."
                    )
                    result = await self.env.controller.grabone(
                        CamNum, ExpTime, Binning, output_dir=grab_save_path
                    )
                    timeout_cameras.extend(result)
                    msg = f"Image grabbed from camera {CamNum}."
                    if timeout_cameras:
                        msg += f" Timeout: {timeout_cameras[0]}"
                    return self._generate_response("success", msg)

            elif isinstance(CamNum, list):
                self.env.logger.info(
                    f"Grabbing images from cameras {CamNum} (ExpTime={ExpTime}, Binning={Binning})."
                )
                timeout_cameras = await self.env.controller.grab(
                    CamNum, ExpTime, Binning, output_dir=grab_save_path
                )
                msg = f"Images grabbed from cameras {CamNum}."
                if timeout_cameras:
                    msg += f" Timeout: {timeout_cameras}"
                return self._generate_response("success", msg)

            else:
                raise ValueError(f"Wrong input for CamNum: {CamNum}")

        except Exception as e:
            self.env.logger.error(f"Error occurred: {e}")
            return self._generate_response(
                "error",
                f"Error occurred: {e} (CamNum={CamNum}, ExpTime={ExpTime}, Binning={Binning})"
            )

    async def guiding(self, ExpTime: float = 1.0) -> Dict[str, Any]:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        raw_save_path = os.path.join(base_dir, "img", "raw")

        try:
            self.env.logger.info("Guiding starts...")
            # Grab if needed:
            # await self.env.controller.grab(0, ExpTime, 4, output_dir=raw_save_path)

            self.env.logger.info("Astrometry preprocessing...")
            self.env.astrometry.preproc()

            self.env.logger.info("Calculating guider offsets...")
            fdx, fdy, fwhm = self.env.guider.exe_cal()

            msg = f"Offsets: fdx={fdx}, fdy={fdy}, FWHM={fwhm:.5f} arcsec"
            return self._generate_response("success", msg, fdx=fdx, fdy=fdy, fwhm=fwhm)

        except Exception as e:
            self.env.logger.error(f"Error during guiding: {e}")
            return self._generate_response("error", f"Error during guiding: {e}")

    def status(self) -> Dict[str, Any]:
        try:
            self.env.logger.info("Checking status of all cameras.")
            status_info = self.env.controller.status()
            return self._generate_response("success", status_info)
        except Exception as e:
            self.env.logger.error(f"Error checking status: {e}")
            return self._generate_response("error", f"Error checking status: {e}")

    def ping(self, CamNum: int = 0) -> Dict[str, Any]:
        try:
            if CamNum == 0:
                self.env.logger.info("Pinging all cameras.")
                for i in range(1, self.env.camera_count + 1):
                    self.env.controller.ping(i)
                return self._generate_response("success", "Pinged all cameras.")
            else:
                self.env.logger.info(f"Pinging camera {CamNum}.")
                self.env.controller.ping(CamNum)
                return self._generate_response("success", f"Pinged camera {CamNum}.")
        except Exception as e:
            self.env.logger.error(f"Error pinging camera(s): {e}")
            return self._generate_response("error", f"Error pinging camera(s): {e}")

    def cam_params(self, CamNum: int = 0) -> Dict[str, Any]:
        try:
            if CamNum == 0:
                self.env.logger.info("Retrieving parameters for all cameras.")
                messages = []
                for i in range(1, self.env.camera_count + 1):
                    param = self.env.controller.cam_params(i)
                    messages.append(f"Cam{i}: {param}")
                return self._generate_response("success", "\n".join(messages))
            else:
                self.env.logger.info(f"Retrieving parameters for camera {CamNum}.")
                param = self.env.controller.cam_params(CamNum)
                return self._generate_response("success", f"Cam{CamNum}: {param}")
        except Exception as e:
            self.env.logger.error(f"Error retrieving parameters: {e}")
            return self._generate_response("error", f"Error retrieving parameters: {e}")

    def shutdown(self):
        """
        Call this to cleanly close cameras before exiting.
        """
        self.env.shutdown()
        self.env.logger.info("GFAActions shutdown complete.")
