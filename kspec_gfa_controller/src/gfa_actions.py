#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2024-05-16
# @Filename: gfa_actions.py

import os
import asyncio

from kspec_gfa_controller.src.gfa_controller import gfa_controller
from kspec_gfa_controller.src.gfa_logger import gfa_logger

def get_config_path():
    """
    Calculate and return the absolute path of the configuration file.

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
    # relative_config_path = "etc/cams.yml"
    relative_config_path = "etc/cams.json"
    config_path = os.path.join(script_dir, relative_config_path)

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    return config_path

config_path = get_config_path()
logger = gfa_logger(__file__)
controller = gfa_controller(config_path, logger)

# 전역 변수로 grab_loop의 실행 여부를 제어
_grab_loop_running = False

def status():
    """
    Check and log the status of all cameras.
    """
    controller.status()


def ping(CamNum=0):
    """
    Ping the specified camera(s) to check connectivity.

    Parameters
    ----------
    CamNum : int, optional
        The camera number to ping. If 0, pings all cameras (default is 0).
    """
    if CamNum == 0:
        for n in range(6):
            index = n + 1
            controller.ping(index)
    else:
        controller.ping(CamNum)


def cam_params(CamNum=0):
    """
    Retrieve and log parameters from the specified camera(s).

    Parameters
    ----------
    CamNum : int, optional
        The camera number to retrieve parameters from.
        If 0, retrieves from all cameras (default is 0).
    """
    if CamNum == 0:
        for n in range(6):
            index = n + 1
            controller.cam_params(index)
    else:
        controller.cam_params(CamNum)


async def grab(CamNum=0, ExpTime=1, Bininng=4, save=True):
    """
    Grab an image from the specified camera(s).

    Parameters
    ----------
    CamNum : int or list of int
        The camera number(s) to grab images from.
        If 0, grabs from all cameras.
    ExpTime : float, optional
        Exposure time in seconds (default is 1).
    Bininng : int, optional
        Binning size (default is 4).

    Raises
    ------
    ValueError
        If CamNum is neither an integer nor a list of integers.
    """
    if isinstance(CamNum, int):
        if CamNum == 0:
            await controller.grab(CamNum, ExpTime, Bininng, save)
        else:
            await controller.grabone(CamNum, ExpTime, Bininng, save)
    elif isinstance(CamNum, list):
        await controller.grab(CamNum, ExpTime, Bininng, save)
    else:
        print(f"Wrong Input {CamNum}")

async def grab_loop(CamNum=0, ExpTime=1, Bininng=4, save=False, interval=10):
    """
    Continuously calls the `grab` method at specified intervals until `stop_grab_loop` is called.

    Parameters
    ----------
    CamNum : int or list of int
        The number identifier of the camera(s) from which to grab images.
        If 0, grabs from all available cameras. If a list, grabs from the specified cameras.
    ExpTime : float
        The exposure time in seconds for the image capture.
    Bininng : int
        The binning size for both horizontal and vertical directions.
    save : bool, optional
        Whether to save the grabbed images to storage, by default False.
    interval : int, optional
        The interval in seconds between consecutive grabs, by default 5.

    Notes
    -----
    The loop will continue running until `stop_grab_loop` is called.
    """
    global _grab_loop_running
    logger.info("Starting grab loop with interval: {interval} seconds")
    logger.debug(f"Executing grab with CamNum={str(CamNum)}, ExpTime={ExpTime}, Bininng={Bininng}, save={save}")
    _grab_loop_running = True
    while _grab_loop_running:
        await controller.grab(CamNum, ExpTime, Bininng, save)
        logger.info(f"Waiting for {interval} seconds before the next grab")
        await asyncio.sleep(interval)

def stop_grab_loop():
    """
    Stops the `grab_loop` from running.

    This function sets the internal flag that controls the execution of the `grab_loop` to False,
    causing the loop to exit after the current iteration.

    Notes
    -----
    If the `grab_loop` is not running, this function has no effect.
    """
    global _grab_loop_running
    if _grab_loop_running:
        logger.info("Stopping grab loop.")
        _grab_loop_running = False
    else:
        logger.warning("Attempted to stop grab loop, but it was not running.")

def is_grab_loop_running():
    """
    Returns whether the `grab_loop` is currently running.

    Returns
    -------
    bool
        True if `grab_loop` is running, False otherwise.

    Notes
    -----
    This function is useful for checking the status of the `grab_loop` in cases where
    you need to ensure the loop is either running or has been stopped.
    """
    logger.info(f"Grab loop running status: {_grab_loop_running}")
    return _grab_loop_running

def offset(Ra, Dec):
    """
    Placeholder for offset function.

    Parameters
    ----------
    Ra : float
        Right ascension coordinate.
    Dec : float
        Declination coordinate.
    """
    # To be implemented
    pass
