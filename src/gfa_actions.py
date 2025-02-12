#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2024-05-16
# @Filename: gfa_actions.py

import os
import json
import shutil
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

from gfa_logger import gfa_logger
from gfa_controller import gfa_controller
from gfa_astrometry import gfa_astrometry
from gfa_guider import gfa_guider

logger = gfa_logger(__file__)

gfa_relative_config_path = "etc/cams.json"
ast_relative_config_path = "etc/astrometry_params.json"

def get_config_path(relative_config_path):
    """
    Calculate and return the absolute path of the configuration file.

    Parameters
    ----------
    relative_config_path : str
        The relative path of the configuration file.

    Returns
    -------
    str
        Absolute path of the configuration file.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist at the calculated path.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, relative_config_path)

    if not os.path.isfile(config_path):
        logger.error(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")

    logger.info(f"Configuration file found: {config_path}")
    return config_path

def initialize():
    global gfa_config_path, controller, ast_config_path, astrometry, guider
    
    gfa_config_path = get_config_path(gfa_relative_config_path)
    controller = gfa_controller(gfa_config_path, logger)

    ast_config_path = get_config_path(ast_relative_config_path)
    astrometry = gfa_astrometry(ast_config_path, logger)
    guider = gfa_guider(ast_config_path, logger)

initialize()

class gfa_actions:
    """
    A class to handle GFA actions such as grabbing images, guiding, and controlling the cameras.
    """

    def __init__(self):
        pass

    @staticmethod
    def _generate_response(status: str, message: str) -> dict:
        """
        Generate a response dictionary.

        Parameters
        ----------
        status : str
            Status of the operation ('success' or 'error').
        message : str
            Message describing the operation result with additional data included.

        Returns
        -------
        dict
            A dictionary representing the response.
        """
        return {"status": status, "message": message}

    @staticmethod
    async def grab(CamNum=0, ExpTime=1, Bininng=4):
        """
        Grab an image from the specified camera(s).
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))  # 현재 파일 위치
        grab_save_path = os.path.join(base_dir, "img", "grab")

        print("이미지 저장 경로:", grab_save_path)

        timeout_cameras = []  # Timeout 발생한 카메라 추적

        try:
            if isinstance(CamNum, int):
                if CamNum == 0:
                    logger.info(f"Grabbing image from all cameras with ExpTime={ExpTime}, Binning={Bininng}")
                    timeout_cameras = await controller.grab(CamNum, ExpTime, Bininng, output_dir=grab_save_path)

                    message = f"Images grabbed from all cameras. (ExpTime: {ExpTime}, Binning: {Bininng})"
                    if timeout_cameras:
                        message += f" | Timeout occurred for cameras: {timeout_cameras}"

                    return gfa_actions._generate_response("success", message)
                else:
                    logger.info(f"Grabbing image from camera {CamNum} with ExpTime={ExpTime}, Binning={Bininng}")
                    
                    # 리스트로 반환되므로 extend() 사용
                    timeout_cameras.extend(await controller.grabone(CamNum, ExpTime, Bininng, output_dir=grab_save_path))

                    message = f"Image grabbed from camera {CamNum}. (ExpTime: {ExpTime}, Binning: {Bininng})"
                    if timeout_cameras:
                        message += f" | Timeout occurred for camera {timeout_cameras[0]}"

                    return gfa_actions._generate_response("success", message)

            elif isinstance(CamNum, list):
                logger.info(f"Grabbing image from cameras {CamNum} with ExpTime={ExpTime}, Binning={Bininng}")
                timeout_cameras = await controller.grab(CamNum, ExpTime, Bininng, output_dir=grab_save_path)

                message = f"Images grabbed from cameras {CamNum}. (ExpTime: {ExpTime}, Binning: {Bininng})"
                if timeout_cameras:
                    message += f" | Timeout occurred for cameras: {timeout_cameras}"

                return gfa_actions._generate_response("success", message)

            else:
                logger.error(f"Wrong Input {CamNum}")
                raise ValueError(f"Wrong Input {CamNum}")

        except Exception as e:
            logger.error(f"Error occurred: {str(e)}")
            return gfa_actions._generate_response("error", f"Error occurred: {str(e)} (CamNum: {CamNum}, ExpTime: {ExpTime}, Binning: {Bininng})")


    @staticmethod
    async def guiding():
        """
        The main guiding loop that grabs images, processes them with astrometry, and calculates offsets.
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))  # 현재 파일 위치
        grab_save_path = os.path.join(base_dir, "img", "raw")

        try:
            logger.info("Guiding starts...")
            logger.info("Step #1: Grab an image")
            #CamNum=0, ExpTime=1, Bininng=4
            #controller.grab(CamNum=0, ExpTime=1, Bininng=4, output_dir=grab_save_path)

            logger.info("Step #2: Astrometry...")
            astrometry.preproc()

            #logger.info("Step #3: Calculating the offset...")
            fdx, fdy, fwhm = guider.exe_cal()

            logger.info(f"Offsets calculated: fdx={fdx}, fdy={fdy}, FWHM={fwhm:.2f} arcsec")
            return gfa_actions._generate_response(
                "success",
                f"Guiding completed successfully. Offsets calculated: fdx={fdx}, fdy={fdy}, FWHM={fwhm:.5f} arcsec"
            )
        except Exception as e:
            logger.error(f"Error occurred during guiding: {str(e)}")
            return gfa_actions._generate_response("error", f"Error occurred during guiding: {str(e)}")

    @staticmethod
    def status():
        """
        Check and log the status of all cameras.
        """
        try:
            logger.info("Checking status of all cameras.")
            status_info = controller.status()
            status_message = "\n".join([f"Camera {i+1}: {info}" for i, info in enumerate(status_info)])
            return gfa_actions._generate_response(
                "success",
                f"Camera status retrieved successfully:\n{status_message}"
            )
        except Exception as e:
            logger.error(f"Error occurred while checking status: {str(e)}")
            return gfa_actions._generate_response("error", f"Error occurred while checking status: {str(e)}")

    @staticmethod
    def ping(CamNum=0):
        """
        Ping the specified camera(s) to check connectivity.
        """
        try:
            if CamNum == 0:
                logger.info("Pinging all cameras.")
                for n in range(6):
                    index = n + 1
                    controller.ping(index)
                return gfa_actions._generate_response("success", "Pinging all cameras completed.")
            else:
                logger.info(f"Pinging camera {CamNum}.")
                controller.ping(CamNum)
                return gfa_actions._generate_response("success", f"Camera {CamNum} pinged successfully.")
        except Exception as e:
            logger.error(f"Error occurred while pinging cameras: {str(e)}")
            return gfa_actions._generate_response("error", f"Error occurred while pinging camera {CamNum}: {str(e)}")

    @staticmethod
    def cam_params(CamNum=0):
        """
        Retrieve and log parameters from the specified camera(s).
        """
        try:
            if CamNum == 0:
                logger.info("Retrieving parameters for all cameras.")
                params = []
                for n in range(6):
                    index = n + 1
                    param = controller.cam_params(index)
                    params.append(param)
                param_message = "\n".join([f"Camera {i+1}: {param}" for i, param in enumerate(params)])
                return gfa_actions._generate_response(
                    "success",
                    f"Parameters retrieved for all cameras:\n{param_message}"
                )
            else:
                logger.info(f"Retrieving parameters for camera {CamNum}.")
                param = controller.cam_params(CamNum)
                return gfa_actions._generate_response(
                    "success",
                    f"Parameters retrieved for camera {CamNum}: {param}"
                )
        except Exception as e:
            logger.error(f"Error occurred while retrieving camera parameters: {str(e)}")
            return gfa_actions._generate_response("error", f"Error occurred while retrieving parameters for camera {CamNum}: {str(e)}")
