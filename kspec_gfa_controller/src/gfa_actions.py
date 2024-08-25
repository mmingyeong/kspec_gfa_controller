#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2024-05-16
# @Filename: gfa_actions.py

import os
import asyncio

from controller.src.gfa_controller import gfa_controller
from controller.src.gfa_logger import gfa_logger

__all__ = ["offset", "status", "ping", "cam_params", "grab"]

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


async def grab(CamNum=0, ExpTime=1, Bininng=4):
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
            await controller.grab(CamNum, ExpTime, Bininng)
        else:
            await controller.grabone(CamNum, ExpTime, Bininng)
    elif isinstance(CamNum, list):
        await controller.grab(CamNum, ExpTime, Bininng)
    else:
        print(f"Wrong Input {CamNum}")


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
