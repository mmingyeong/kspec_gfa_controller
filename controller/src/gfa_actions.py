#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2024-05-16
# @Filename: gfa_actions.py

import os
import sys

from controller.src.gfa_controller import gfa_controller
from controller.src.gfa_logger import gfa_logger

__all__ = ["expose", "offset", "status", "ping", "cam_params", "grabone", "graball"]

# 설정 파일의 절대 경로를 계산합니다
def get_config_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    relative_config_path = "etc/cams.yml"
    config_path = os.path.join(script_dir, relative_config_path)
    
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    return config_path

config_path = get_config_path()
logger = gfa_logger(__file__)
controller = gfa_controller(config_path, logger)

def grabone(CamNum=1, ExpTime=0.01):
    controller.grabone(CamNum, ExpTime)
    
async def graball(ExpTime=0.01):
    await controller.graball(ExpTime)

def expose(CamNum=0, ExpTime=0.01):
    # To be implemented
    pass

def offset(Ra, Dec):
    # To be implemented
    pass

def status():
    controller.status()

def ping(CamNum=0):
    controller.ping(CamNum)

def cam_params(CamNum=0):
    controller.cam_params(CamNum)
