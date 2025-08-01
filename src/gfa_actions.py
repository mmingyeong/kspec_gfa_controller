#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2024-05-16
# @Filename: gfa_actions.py

import os
import asyncio
import shutil
from datetime import datetime
from typing import Union, List, Dict, Any, Optional

from gfa_logger import GFALogger
from gfa_environment import create_environment, GFAEnvironment

logger = GFALogger(__file__)

###############################################################################
# GFA Actions Class
###############################################################################


class GFAActions:
    """
    A class to handle GFA actions such as grabbing images, guiding,
    and controlling the plate camera array (Cam1–6).
    """

    def __init__(self, env: Optional[GFAEnvironment] = None):
        """
        Initialize GFAActions with a GFA environment.

        Parameters
        ----------
        env : GFAEnvironment, optional
            The environment object for controlling GFA system.
        """
        if env is None:
            env = create_environment(role="plate")
        self.env = env

    def _generate_response(self, status: str, message: str, **kwargs) -> dict:
        """
        Generate a structured response dictionary.

        Parameters
        ----------
        status : str
            Status string, e.g., "success" or "error".
        message : str
            Descriptive message.
        kwargs : dict
            Additional key-value pairs to include.

        Returns
        -------
        dict
            Response dictionary.
        """
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
    ) -> Dict[str, Any]:
        """
        Grab images from one or more plate cameras.

        Parameters
        ----------
        CamNum : int or List[int], optional
            Target camera(s). If 0, grabs from all cameras.
        ExpTime : float, optional
            Exposure time in seconds.
        Binning : int, optional
            Binning factor.
        packet_size : int, optional
            GigE packet size. If None, use cams.json per camera.
        cam_ipd : int, optional
            Inter-packet delay. If None, use cams.json per camera.
        cam_ftd_base : int, optional
            Frame transmission delay base.

        Returns
        -------
        dict
            Result summary and timeout information.
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        date_str = datetime.now().strftime("%Y-%m-%d")
        grab_save_path = os.path.join(base_dir, "img", "grab", date_str)
        self.env.logger.info(f"Image save path: {grab_save_path}")

        timeout_cameras: List[int] = []

        try:
            if isinstance(CamNum, int) and CamNum != 0:
                self.env.logger.info(
                    f"Grabbing from camera {CamNum} (ExpTime={ExpTime}, Binning={Binning})"
                )
                result = await self.env.controller.grabone(
                    CamNum=CamNum,
                    ExpTime=ExpTime,
                    Binning=Binning,
                    output_dir=grab_save_path,
                    packet_size=packet_size,
                    ipd=cam_ipd,
                    ftd_base=cam_ftd_base,
                )
                timeout_cameras.extend(result)
                msg = f"Image grabbed from camera {CamNum}."
                if timeout_cameras:
                    msg += f" Timeout: {timeout_cameras[0]}"
                return self._generate_response("success", msg)

            if isinstance(CamNum, int) and CamNum == 0:
                self.env.logger.info("Grabbing from all plate cameras...")

                tasks = []
                for cam_id in self.env.camera_ids:
                    self.env.logger.info(
                        f"Grabbing from Cam{cam_id} (ExpTime={ExpTime}, Binning={Binning})"
                    )
                    task = self.env.controller.grabone(
                        CamNum=cam_id,
                        ExpTime=ExpTime,
                        Binning=Binning,
                        output_dir=grab_save_path,
                        packet_size=packet_size,
                        ipd=cam_ipd,
                        ftd_base=cam_ftd_base,
                    )
                    tasks.append(task)

                # Run all camera grabs concurrently
                results = await asyncio.gather(*tasks)
                timeout_cameras = []
                for res in results:
                    timeout_cameras.extend(res)

                msg = "Images grabbed from all cameras."
                if timeout_cameras:
                    msg += f" Timeout: {timeout_cameras}"
                return self._generate_response("success", msg)

            if isinstance(CamNum, list):
                self.env.logger.info(
                    f"Grabbing from cameras {CamNum} (ExpTime={ExpTime}, Binning={Binning})"
                )
                tasks = []
                for cam_id in CamNum:
                    self.env.logger.info(
                        f"Grabbing from Cam{cam_id} (ExpTime={ExpTime}, Binning={Binning})"
                    )
                    task = self.env.controller.grabone(
                        CamNum=cam_id,
                        ExpTime=ExpTime,
                        Binning=Binning,
                        output_dir=grab_save_path,
                        packet_size=packet_size,
                        ipd=cam_ipd,
                        ftd_base=cam_ftd_base,
                    )
                    tasks.append(task)

                # Run all camera grabs concurrently
                results = await asyncio.gather(*tasks)
                timeout_cameras = []
                for res in results:
                    timeout_cameras.extend(res)

                msg = f"Images grabbed from cameras {CamNum}."
                if timeout_cameras:
                    msg += f" Timeout: {timeout_cameras}"
                return self._generate_response("success", msg)

            raise ValueError(f"Invalid CamNum: {CamNum}")

        except Exception as e:
            self.env.logger.error(f"Grab failed: {e}")
            return self._generate_response(
                "error", f"Grab failed: {e} (CamNum={CamNum}, ExpTime={ExpTime})"
            )


    async def guiding(self, ExpTime: float = 1.0, save: bool = False) -> Dict[str, Any]:
        """
        Execute guiding procedure using all plate cameras.

        Parameters
        ----------
        ExpTime : float, optional
            Exposure time in seconds.
        save : bool, optional
            Whether to also save images to grab directory.

        Returns
        -------
        dict
            Result with measured guider offsets and FWHM.
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        date_str = datetime.now().strftime("%Y-%m-%d")

        raw_save_path = os.path.join(base_dir, "img", "raw")
        grab_save_path = os.path.join(base_dir, "img", "grab", date_str)

        try:
            self.env.logger.info("Starting guiding sequence...")

            os.makedirs(raw_save_path, exist_ok=True)
            self.env.logger.info("Grabbing raw image...")
            self.env.controller.grab(0, ExpTime, 4, output_dir=raw_save_path)

            if save:
                os.makedirs(grab_save_path, exist_ok=True)
                for fname in os.listdir(raw_save_path):
                    src = os.path.join(raw_save_path, fname)
                    dst = os.path.join(grab_save_path, fname)
                    if os.path.isfile(src):
                        shutil.copy2(src, dst)
                self.env.logger.info(f"Images also copied to: {grab_save_path}")

            self.env.logger.info("Running astrometry preprocessing...")
            self.env.astrometry.preproc()

            self.env.logger.info("Executing guider offset calculation...")
            fdx, fdy, fwhm = self.env.guider.exe_cal()

            self.env.logger.info("Clearing temp astrometry data...")
            self.env.astrometry.clear_raw_and_processed_files()
            
            try:
                fwhm_val = float(fwhm)
            except ValueError:
                fwhm_val = 0.0  # 또는 예외 처리

            msg = f"Offsets: fdx={fdx}, fdy={fdy}, FWHM={fwhm_val} arcsec"
            return self._generate_response("success", msg, fdx=fdx, fdy=fdy, fwhm=fwhm_val)

        except Exception as e:
            self.env.logger.error(f"Guiding failed: {str(e)}")
            return self._generate_response("error", f"Guiding failed: {str(e)}")


    def status(self) -> Dict[str, Any]:
        """
        Retrieve current status of all plate cameras.

        Returns
        -------
        dict
            Status report from controller.
        """
        try:
            self.env.logger.info("Querying camera statuses...")
            status_info = self.env.controller.status()
            return self._generate_response("success", status_info)
        except Exception as e:
            self.env.logger.error(f"Status query failed: {e}")
            return self._generate_response("error", f"Status query failed: {e}")

    def ping(self, CamNum: int = 0) -> Dict[str, Any]:
        """
        Ping specific or all plate cameras.

        Parameters
        ----------
        CamNum : int, optional
            If 0, ping all cameras. Else ping specific CamNum.

        Returns
        -------
        dict
            Ping result message.
        """
        try:
            if CamNum == 0:
                self.env.logger.info("Pinging all cameras...")
                for cam_id in self.env.camera_ids:
                    self.env.controller.ping(cam_id)
                return self._generate_response("success", "Pinged all cameras.")
            else:
                self.env.logger.info(f"Pinging Cam{CamNum}...")
                self.env.controller.ping(CamNum)
                return self._generate_response("success", f"Pinged Cam{CamNum}.")
        except Exception as e:
            self.env.logger.error(f"Ping failed: {e}")
            return self._generate_response("error", f"Ping failed: {e}")

    def cam_params(self, CamNum: int = 0) -> Dict[str, Any]:
        """
        Retrieve camera parameters.

        Parameters
        ----------
        CamNum : int, optional
            If 0, retrieve all cameras. Else only for CamNum.

        Returns
        -------
        dict
            Parameter information.
        """
        try:
            if CamNum == 0:
                self.env.logger.info("Fetching parameters for all cameras...")
                messages = []
                for cam_id in self.env.camera_ids:
                    param = self.env.controller.cam_params(cam_id)
                    messages.append(f"Cam{cam_id}: {param}")
                return self._generate_response("success", "\n".join(messages))
            else:
                self.env.logger.info(f"Fetching parameters for Cam{CamNum}...")
                param = self.env.controller.cam_params(CamNum)
                return self._generate_response("success", f"Cam{CamNum}: {param}")
        except Exception as e:
            self.env.logger.error(f"Parameter fetch failed: {e}")
            return self._generate_response("error", f"Parameter fetch failed: {e}")

    def shutdown(self) -> None:
        """
        Shutdown and release resources.
        """
        self.env.shutdown()
        self.env.logger.info("GFAActions shutdown complete.")
