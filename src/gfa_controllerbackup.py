#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2023-01-03
# @Filename: gfa_controller.py

import asyncio
import json
import os
import time
import logging
import sys
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import pypylon.pylon as py
import yaml
from pypylon import genicam

from gfa_img import GFAImage

__all__ = ["GFAController"]


###############################################################################
# Default Config and Logger
###############################################################################
def _get_default_config_path() -> str:
    """
    Returns the default path for the GFA camera config file.
    Adjust this path as needed for your environment.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_path = os.path.join(script_dir, "etc", "cams.json")
    if not os.path.isfile(default_path):
        raise FileNotFoundError(
            f"Default config file not found at: {default_path}. "
            "Please adjust `_get_default_config_path()`."
        )
    return default_path


def _get_default_logger() -> logging.Logger:
    """
    Returns a simple default logger if none is provided.
    """
    logger = logging.getLogger("gfa_controller_default")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger


def from_config(config_path: str) -> dict:
    """
    Loads GFA camera configuration from a YAML or JSON file.

    Parameters
    ----------
    config_path : str
        Path to the configuration file. The file can be in .yml, .yaml, or .json format.

    Returns
    -------
    dict
        A dictionary containing the GFA camera configuration data.

    Raises
    ------
    ValueError
        If the file format is unsupported (i.e., not .yml, .yaml, or .json).
    """
    file_extension = os.path.splitext(config_path)[1].lower()
    with open(config_path, 'r') as f:
        if file_extension in [".yml", ".yaml"]:
            data = yaml.load(f, Loader=yaml.FullLoader)
        elif file_extension == ".json":
            data = json.load(f)
        else:
            raise ValueError(
                "Unsupported file format. Please use a .yml, .yaml, or .json file."
            )
    return data


###############################################################################
# Main Controller Class
###############################################################################
class GFAController:
    """Talk to KSPEC GFA Cameras over TCP/IP with optimized open/grab/close."""

    def __init__(self, config: str = None, logger: logging.Logger = None):
        """
        Initializes the GFAController with configuration and logger.

        Parameters
        ----------
        config : str, optional
            Path to the configuration file. If None, a default path is used.
        logger : logging.Logger, optional
            Logger instance for logging. If None, a default logger is created.

        Raises
        ------
        FileNotFoundError
            If no valid default configuration file is found.
        KeyError
            If expected keys are not present in the configuration dictionary.
        """
        # 1. Default config path
        if config is None:
            config = _get_default_config_path()

        # 2. Default logger
        if logger is None:
            logger = _get_default_logger()
        self.logger = logger

        # 3. Load configuration
        try:
            self.config = from_config(config)
            self.logger.info("Initializing GFAController with provided config.")
        except Exception as e:
            self.logger.error(f"Error loading configuration from {config}: {e}")
            raise

        # 4. Extract camera info
        try:
            self.cameras_info = self.config["GfaController"]["Elements"]["Cameras"]["Elements"]
        except KeyError as e:
            self.logger.error(f"Configuration key error: {e}")
            raise

        # 5. Pylon environment
        self.NUM_CAMERAS = len(self.cameras_info)
        os.environ["PYLON_CAMEMU"] = f"{self.NUM_CAMERAS}"
        self.tlf = py.TlFactory.GetInstance()

        # 6. Internal attrs
        self.grab_timeout = 5000
        self.img_class = GFAImage(logger)
        self.open_cameras = {}  # will hold opened camera objects

        self.logger.info("GFAController initialization complete.")

    def open_all_cameras(self):
        """Open all cameras once at startup."""
        self.logger.info("Opening all cameras...")
        for cam_key, cam_info in self.cameras_info.items():
            ip = cam_info["IpAddress"]
            dev_info = py.DeviceInfo()
            dev_info.SetIpAddress(ip)
            cam = py.InstantCamera(self.tlf.CreateDevice(dev_info))
            cam.Open()
            self.open_cameras[cam_key] = cam
            self.logger.info(f"{cam_key} opened (IP {ip}).")
        self.logger.info("All cameras opened successfully.")

    def close_all_cameras(self):
        """Close all opened cameras."""
        self.logger.info("Closing all cameras...")
        for cam_key, cam in self.open_cameras.items():
            if cam.IsOpen():
                cam.Close()
                self.logger.info(f"{cam_key} closed.")
        self.open_cameras.clear()
        self.logger.info("All cameras closed.")

    def ping(self, CamNum: int = 0):
        """Ping the specified camera to verify connectivity."""
        self.logger.info(f"Pinging camera {CamNum}...")
        key = f"Cam{CamNum}"
        if key not in self.cameras_info:
            self.logger.error(f"Camera {key} not found in config.")
            raise KeyError(f"{key} missing")
        ip = self.cameras_info[key]["IpAddress"]
        result = os.system(f"ping -c 3 -w 1 {ip}")
        self.logger.info(f"Ping result for {key} ({ip}): {result}")

    def status(self):
        """
        Check whether each camera is open/standby.
        Returns a dict {cam_key: bool}.
        """
        self.logger.info("Checking camera status...")
        status = {}
        for cam_key in self.cameras_info.keys():
            cam = self.open_cameras.get(cam_key)
            is_open = cam.IsOpen() if cam else False
            status[cam_key] = is_open
            if is_open:
                self.logger.info(f"{cam_key} is online and standby.")
            else:
                self.logger.warning(f"{cam_key} is not open.")
        return status

    def cam_params(self, CamNum: int):
        """Retrieve and log parameters for one camera."""
        key = f"Cam{CamNum}"
        if key not in self.open_cameras:
            self.logger.error(f"{key} not opened.")
            return
        cam = self.open_cameras[key]
        self.logger.info(f"Device parameters for {key}:")
        attrs = [
            "DeviceModelName", "DeviceSerialNumber", "DeviceUserID",
            "Width", "Height", "PixelFormat",
            "ExposureTime", "BinningHorizontal", "BinningVertical"
        ]
        for attr in attrs:
            try:
                val = getattr(cam, attr).GetValue()
                self.logger.info(f"  {attr}: {val}")
            except Exception as e:
                self.logger.error(f"  Failed to get {attr}: {e}")

    async def configure_and_grab(self, cam, ExpTime, Binning,
                                 output_dir=None, serial_hint=None):
        """
        Common routine: apply transmission settings, exposure, grab, save FITS.
        Returns image array or None on timeout/error.
        """
        now1 = time.time()
        formatted = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime(now1))
        loop = asyncio.get_running_loop()
        img = None
        timeout = False

        try:
            # Transmission optimization
            await loop.run_in_executor(None, cam.GevSCPSPacketSize.SetValue, 1440)
            await loop.run_in_executor(None, cam.GevSCPD.SetValue, 2000)

            # Frame transmission delay stagger
            delay_base = 1500
            if serial_hint:
                try:
                    idx = int(serial_hint[-1]) - 1
                except:
                    idx = 0
            else:
                idx = 0
            frame_delay = delay_base * idx
            await loop.run_in_executor(None, cam.GevSCFTD.SetValue, frame_delay)

            # Exposure and binning
            microsec = ExpTime * 1_000_000
            await loop.run_in_executor(None, cam.ExposureTime.SetValue, microsec)
            await loop.run_in_executor(None, cam.PixelFormat.SetValue, "Mono12")
            await loop.run_in_executor(None, cam.BinningHorizontal.SetValue, Binning)
            await loop.run_in_executor(None, cam.BinningVertical.SetValue, Binning)

            # Grab image
            result = await loop.run_in_executor(None, cam.GrabOne, self.grab_timeout)
            img = result.GetArray()

            # Save FITS
            filename = f"{formatted}_cam_{serial_hint or 'unknown'}.fits"
            self.img_class.save_fits(
                image_array=img,
                filename=filename,
                exptime=ExpTime,
                output_directory=output_dir,
            )
            self.logger.info(f"Image grabbed & saved: {filename}")

        except genicam.TimeoutException:
            self.logger.error("Timeout while grabbing image.")
            timeout = True
        except Exception as e:
            self.logger.error(f"Error grabbing image: {e}")
            timeout = True

        now2 = time.time()
        self.logger.debug(f"Grab time: {now2 - now1:.2f}s")
        return None if timeout else img

    async def grabone(self, CamNum: int, ExpTime: float,
                      Binning: int, output_dir: str = None):
        """
        Grab one image from a single camera using configure_and_grab().
        Returns [CamNum] on timeout, [] on success.
        """
        key = f"Cam{CamNum}"
        cam = self.open_cameras.get(key)
        if cam is None:
            self.logger.error(f"{key} not opened.")
            return [CamNum]
        # get serial for hint
        serial = cam.DeviceSerialNumber.GetValue()
        img = await self.configure_and_grab(cam, ExpTime, Binning,
                                            output_dir, serial_hint=serial)
        return [] if img is not None else [CamNum]

    async def process_camera(self, CamNum, ExpTime, Binning, output_dir=None):
        """
        Helper for parallel grab: same as grabone but returns numpy array or None.
        """
        key = f"Cam{CamNum}"
        cam = self.open_cameras.get(key)
        if cam is None:
            self.logger.error(f"{key} not opened.")
            return None
        serial = await asyncio.get_running_loop().run_in_executor(
            None, cam.DeviceSerialNumber.GetValue
        )
        return await self.configure_and_grab(cam, ExpTime, Binning,
                                             output_dir, serial_hint=serial)

    async def grab(self, CamNum, ExpTime: float,
                   Binning: int, output_dir: str = None):
        """
        Grab images from all or multiple cameras in parallel.
        Returns list of cameras that timed out.
        """
        # build list of numbers
        if CamNum == 0:
            nums = [int(k.replace("Cam", "")) for k in self.cameras_info.keys()]
        elif isinstance(CamNum, list):
            nums = CamNum
        else:
            self.logger.error("Invalid CamNum; must be 0 or list.")
            return []

        self.logger.info(f"Parallel grab from cameras {nums}, ExpTime={ExpTime}")
        now = time.time()
        tasks = [
            self.process_camera(n, ExpTime, Binning, output_dir)
            for n in nums
        ]
        results = await asyncio.gather(*tasks)
        timed_out = [n for n, img in zip(nums, results) if img is None]
        self.logger.debug(f"Parallel grab time: {time.time() - now:.2f}s")
        return timed_out
