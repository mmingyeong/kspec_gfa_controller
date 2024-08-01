#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2024-08-01
# @Filename: main.py

import os
import sys
import asyncio

# Add the parent directory of the script to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gfa_actions import ping, status, grabone, cam_params, graball

async def main():
    CamNum = 1

    # Check the status of the camera
    print("Checking status of camera...")
    status()
    
    # Example of how you might use other functions
    # Uncomment and adjust as needed
    print("Pinging camera...")
    ping(0)
    
    # print(f"Grabbing one frame from camera {CamNum}...")
    # grabone(CamNum, 0.1)
    
    # print(f"Checking the parameters from camera {CamNum}...")
    # cam_params(CamNum)
    
    # await graball()

if __name__ == "__main__":
    asyncio.run(main())
