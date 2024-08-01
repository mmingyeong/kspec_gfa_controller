#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2023-01-03
# @Filename: gfa_controller.py

import os
import time
import yaml
import asyncio

import matplotlib.pyplot as plt
import pypylon.pylon as py
from pypylon import genicam
from concurrent.futures import ThreadPoolExecutor

from controller.src.gfa_exceptions import GFAinitError, GFACamNumError, GFAConfigError, GFAError, GFAPingError

__all__ = ["gfa_controller"]

class gfa_controller:
    """Talk to an KSPEC GFA Camera over TCP/IP.

    Parameters
    ----------
    name
        A name identifying this controller.
    config
        The configuration defined on the .yaml file under /etc/cameras.yml
    """

    def __init__(self, config, logger):
        self.logger = logger
        
        try:
            self.config = from_config(config)
        except:
            self.logger.error("No config")
            raise GFAConfigError("Wrong config")
        
        self.logger.info("Initializing gfa_controller")
        self.cameras_info = self.config["GfaController"]["Elements"]["Cameras"]["Elements"]
        self.camera_list = list(self.cameras_info.keys())

        self.NUM_CAMERAS = 5
        os.environ["PYLON_CAMEMU"] = f"{self.NUM_CAMERAS}"

        try:
            self.tlf = py.TlFactory.GetInstance()
            self.devices = self.tlf.EnumerateDevices()
            di = py.DeviceInfo()
            self.devs = self.tlf.EnumerateDevices([di,])
            self.cam_array = py.InstantCameraArray(self.NUM_CAMERAS)
        except:
            self.logger.error("GetInstance() fail")
            raise GFAinitError("GetInstance() fail")
        
        self.logger.info(f"Found {len(self.devices)} devices:")


    def ping(self, CamNum=0):
        self.logger.info(f"Pinging camera(s), CamNum={CamNum}")
        if not CamNum == 0:
            Cam_IpAddress = self.cameras_info[f"Cam{CamNum}"]["IpAddress"]
            self.logger.debug(f"Camera {CamNum} IP address: {Cam_IpAddress}")
            ping_test = os.system("ping -c 3 -w 1 " + Cam_IpAddress)
            self.logger.info(f"Pinging camera {CamNum} at {Cam_IpAddress}, result: {ping_test}")
        else:
            for idx, cam in enumerate(self.cam_array):
                cam.Attach(self.tlf.CreateDevice(self.devs[idx]))
                camera_ip = cam.DeviceInfo.GetIpAddress()
                self.logger.debug(f"Camera {idx+1} IP address: {camera_ip}")
                ping_test = os.system("ping -c 3 -w 1 " + camera_ip)
                self.logger.info(f"Pinging camera {idx+1} at {camera_ip}, result: {ping_test}")

        return ping_test

    def status(self):
        """Return connection status of the camera"""
        self.logger.info("Checking status of all cameras")
        for idx, cam in enumerate(self.cam_array):
            cam.Attach(self.tlf.CreateDevice(self.devs[idx]))
            camera_serial = cam.DeviceInfo.GetSerialNumber()
            self.logger.info(f"camera {camera_serial}: standby")
            print(f"camera {camera_serial}: standby")

    def cam_params(self, CamNum):
        self.logger.info(f"Checking the parameters from camera {CamNum}")
        now1 = time.time()
        lt = time.localtime(now1)

        # Cam Info check
        Cam_IpAddress = self.cameras_info[f"Cam{CamNum}"]["IpAddress"]
        Cam_SerialNumber = self.cameras_info[f"Cam{CamNum}"]["SerialNumber"]
        self.logger.debug(f"Camera {CamNum} IP address: {Cam_IpAddress}, Serial number: {Cam_SerialNumber}")

        tlf = py.TlFactory.GetInstance()
        tlf.EnumerateDevices()
        cam_info = py.DeviceInfo()
        cam_info.SetIpAddress(Cam_IpAddress)
        camera = py.InstantCamera(self.tlf.CreateDevice(cam_info))
        
        # cam open
        self.logger.debug(f"Opening camera {CamNum}")
        camera.Open()
        
        self.logger.info("Camera Device Information")
        self.logger.info("=========================")
        info_attributes = [
            ("DeviceModelName", "DeviceModelName"),
            ("DeviceSerialNumber", "DeviceSerialNumber"),
            ("DeviceUserID", "DeviceUserID"),
            ("Width", "Width"),
            ("Height", "Height"),
            ("PixelFormat", "PixelFormat"),
            ("ExposureTime (마이크로초)", "ExposureTime"),
        ]

        for label, attribute in info_attributes:
            try:
                value = getattr(camera, attribute).GetValue()
                self.logger.info(f"{label} : {value}")
            except:
                self.logger.error(f"AccessException occurred while accessing {label}")
                
        # cam close
        camera.Close()

        now2 = time.time()
        self.logger.info(f"Process time for camera {CamNum}: {now2-now1}")

    def grabone(self, CamNum, ExpTime):
        """Grab the image

        Parameters
        ----------
        CamNum
            Camera number
        ExpTime
            Exposure time (sec)
        """
        self.logger.info(f"Grabbing image from camera {CamNum}, ExpTime={ExpTime}")
        now1 = time.time()
        lt = time.localtime(now1)
        formatted = time.strftime("%Y-%m-%d_%H:%M:%S", lt)

        # Cam Info check
        Cam_IpAddress = self.cameras_info[f"Cam{CamNum}"]["IpAddress"]
        Cam_SerialNumber = self.cameras_info[f"Cam{CamNum}"]["SerialNumber"]
        self.logger.debug(f"Camera {CamNum} IP address: {Cam_IpAddress}, Serial number: {Cam_SerialNumber}")

        tlf = py.TlFactory.GetInstance()
        tlf.EnumerateDevices()
        cam_info = py.DeviceInfo()
        cam_info.SetIpAddress(Cam_IpAddress)
        camera = py.InstantCamera(self.tlf.CreateDevice(cam_info))
        
        # cam open
        self.logger.debug(f"Opening camera {CamNum}")
        camera.Open()

        # Expose
        ExpTime_microsec = ExpTime * 1000000
        self.logger.debug(f"Setting exposure time for camera {CamNum} to {ExpTime_microsec} microseconds")
        camera.ExposureTime.SetValue(ExpTime_microsec)
        res = camera.GrabOne(1000)
        img = res.GetArray()

        # save the image
        plt.imshow(img)
        plt.title(f"cam:{Cam_SerialNumber}, Exptime: {ExpTime}sec")
        img_path = f"/home/kspec/mingyeong/kspec_gfa_controller/controller/src/save/{formatted}_cam{CamNum}_img.png"
        plt.savefig(img_path)
        self.logger.info(f"Image from camera {CamNum} saved to {img_path}")

        # cam close
        camera.Close()

        now2 = time.time()
        self.logger.info(f"Exposure time for camera {CamNum}: {ExpTime} sec")
        self.logger.info(f"Process time for camera {CamNum}: {now2-now1}")

    async def graball(self, ExpTime):
        """Grab the image

        Parameters
        ----------
        ExpTime
            Exposure time (sec)
        """
        self.logger.info(f"Grabbing images from all cameras, ExpTime={ExpTime}")
        now1 = time.time()

        self.logger.debug("Opening all cameras in the array")
        
        # 비동기 작업을 위한 실행자 설정
        with ThreadPoolExecutor() as executor:
            loop = asyncio.get_running_loop()
            tasks = [self.process_camera(d, ExpTime) for d in self.devices]
            await asyncio.gather(*tasks)
        
        now2 = time.time()
        self.logger.info(f"Final process time for graball: {now2 - now1:.2f} seconds")

    async def process_camera(self, device, ExpTime):
        now1 = time.time()
        lt = time.localtime(now1)
        formatted = time.strftime("%Y-%m-%d %H:%M:%S", lt)
        loop = asyncio.get_running_loop()
        
        # 카메라 열기
        camera = py.InstantCamera(self.tlf.CreateDevice(device))
        open_start_time = time.time()
        camera.Open()
        open_end_time = time.time()
        open_duration = open_end_time - open_start_time
        self.logger.info(f"Camera {device.GetFriendlyName()} opened in {open_duration:.2f} seconds")
        
        try:
            serial_number = await loop.run_in_executor(None, camera.DeviceSerialNumber.GetValue)
            self.logger.info(f"Opened camera: {serial_number}")
            
            ExpTime_microsec = ExpTime * 1000000
            await loop.run_in_executor(None, camera.ExposureTime.SetValue, ExpTime_microsec)
            
            # 이미지 잡기
            grab_start_time = time.time()
            try:
                result = await loop.run_in_executor(None, camera.GrabOne, 5000)
                img = result.GetArray()
                grab_end_time = time.time()
                grab_duration = grab_end_time - grab_start_time
                self.logger.info(f"Image grabbed from camera: {serial_number} in {grab_duration:.2f} seconds")
                
                # 이미지 저장
                save_start_time = time.time()
                plt.imshow(img)
                plt.title(f"{serial_number}")
                plt.savefig(f"/home/kspec/mingyeong/kspec_gfa_controller/controller/src/save/{formatted}_{serial_number}_img.png")
                save_end_time = time.time()
                save_duration = save_end_time - save_start_time
                self.logger.info(f"Image saved for camera: {serial_number} in {save_duration:.2f} seconds")
                
            except genicam.TimeoutException:
                self.logger.error(f"TimeoutException occurred while grabbing an image from camera {serial_number}")
            except Exception as e:
                self.logger.error(f"An unexpected error occurred while grabbing an image from camera {serial_number}: {str(e)}")
        
        except genicam.AccessException as e:
            self.logger.error(f"AccessException occurred with camera {device.GetFriendlyName()}: {str(e)}")
        except Exception as e:
            self.logger.error(f"An unexpected error occurred with camera {device.GetFriendlyName()}: {str(e)}")
        
        await loop.run_in_executor(None, camera.Close)
        self.logger.info(f"Closed camera: {device.GetFriendlyName()}")


def from_config(config):
    """Creates an dictionary of the GFA camera information from a configuration file."""
    with open(config) as f:
        film = yaml.load(f, Loader=yaml.FullLoader)
    return film

