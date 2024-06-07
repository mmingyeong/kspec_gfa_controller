#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2023-01-03
# @Filename: gfa_controller.py

import os
import time
import yaml

import matplotlib.pyplot as plt
import pypylon.pylon as py

__all__ = ["gfa_controller"]
config_path = "./etc/cams.yml"


class gfa_controller:
    """Talk to an KSPEC GFA Camera over TCP/IP.

    Parameters
    ----------
    name
        A name identifying this controller.
    config
        The configuration defined on the .yaml file under /etc/cameras.yml
    """

    def __init__(self):
        self.config = from_config(config_path)
        self.cameras_info = self.config["GfaController"]["Elements"]["Cameras"]["Elements"]
        self.camera_list = list(self.cameras_info.keys())

        self.NUM_CAMERAS = 5
        os.environ["PYLON_CAMEMU"] = f"{self.NUM_CAMERAS}"

        self.tlf = py.TlFactory.GetInstance()
        #self.tl = self.tlf.CreateTl("BaslerGigE")
        di = py.DeviceInfo()
        self.devs = self.tlf.EnumerateDevices([di,])
        self.cam_array = py.InstantCameraArray(self.NUM_CAMERAS)

    def ping(self, CamNum=0):

        if not CamNum == 0:
            Cam_IpAddress = self.cameras_info[f"Cam{CamNum}"]["IpAddress"]
            ping_test = os.system("ping -c 3 -w 1 " + Cam_IpAddress)

        else:
            for idx, cam in enumerate(self.cam_array):
                cam.Attach(self.tlf.CreateDevice(self.devs[idx]))
                camera_ip = cam.DeviceInfo.GetIpAddress()
                ping_test = os.system("ping -c 3 -w 1 " + camera_ip)

            #for cam in self.camera_list:
            #    Cam_IpAddress = self.cameras_info[cam]["IpAddress"]
            #    ping_test = os.system("ping -c 3 -w 1 " + Cam_IpAddress)

        return ping_test

    def status(self):
        """Return connection status of the camera"""

        for idx, cam in enumerate(self.cam_array):
            cam.Attach(self.tlf.CreateDevice(self.devs[idx]))
            camera_serial = cam.DeviceInfo.GetSerialNumber()
            print(f"camera {camera_serial}: standby")


    def graball(self, ExpTime=0.1):
        """Grab the image

        Parameters
        ----------
        ExpTime
            Exposure time (sec)
        """

        now1 = time.time()
        lt = time.localtime(now1)
        formatted = time.strftime("%Y-%m-%d %H:%M:%S", lt)
        
        self.cam_array.Open()
        self.cam_array.StartGrabbing()
        for idx, cam in enumerate(self.cam_array):
            now1 = time.time()
            cam.Attach(self.tlf.CreateDevice(self.devs[idx]))
            num=self.devs[idx].GetSerialNumber()
            print(f"{num} open")
            cam.Open()

            # Expose
            ExpTime_microsec = ExpTime * 1000000
            cam.ExposureTime.SetValue(ExpTime_microsec)
            print(f"{num} grab")
            res = cam.GrabOne(100000)
            img = res.GetArray()
            plt.imshow(img)
            plt.savefig(f"./img_saved/{formatted}_cam{num}_img.png")
            print(f"{num} done")

            # cam close
            cam.Close()
            now2 = time.time()
            print(f"process time: {now2-now1}")

        now2 = time.time()
        print(f"Final process time: {now2-now1}")

    def grabone(self, CamNum, ExpTime=0.1):
        """Grab the image

        Parameters
        ----------
        CamNum
            Camera number
        ExpTime
            Exposure time (sec)
        """
        
        now1 = time.time()
        lt = time.localtime(now1)
        formatted = time.strftime("%Y-%m-%d %H:%M:%S", lt)

        # Cam Info check
        # Cam_name = self.cameras_info[f"Cam{CamNum}"]["Name"]
        Cam_IpAddress = self.cameras_info[f"Cam{CamNum}"]["IpAddress"]
        Cam_SerialNumber = self.cameras_info[f"Cam{CamNum}"]["SerialNumber"]

        # Get the transport layer factory.
        # https://github.com/basler/pypylon/issues/572
        tlf = py.TlFactory.GetInstance()
        tlf.EnumerateDevices()
        cam_info = py.DeviceInfo()
        cam_info.SetIpAddress(Cam_IpAddress)
        cam_ready = py.InstantCamera(tlf.CreateDevice(cam_info))
        
        # cam open
        cam_ready.Open()
        cam_ready.AcquisitionMode.SetValue('Continuous')

        # Expose
        ExpTime_microsec = ExpTime * 1000000
        cam_ready.ExposureTime.SetValue(ExpTime_microsec)
        res = cam_ready.GrabOne(5000)
        img = res.GetArray()

        # save the image
        plt.imshow(img)
        plt.title(f"cam:{Cam_SerialNumber}, Exptime: {ExpTime}sec")
        plt.savefig(f"./img_saved/{formatted}_cam{CamNum}_img.png")

        # cam close
        cam_ready.Close()

        now2 = time.time()
        print(f"Exposure time: {ExpTime}")
        print(f"process time: {now2-now1}")

    def grabsome(self, CamNums:list, ExpTime=0.1, frames_to_grab=1):
        """Grab the image

        Parameters
        ----------
        ExpTime
            Exposure time (sec)
        """

        now1 = time.time()
        lt = time.localtime(now1)
        formatted = time.strftime("%Y-%m-%d %H:%M:%S", lt)

        NUM_CAMERAS = len(CamNums)

        cam_ip_list = []
        for num in CamNums:
            cam_ip_list.append(self.cameras_info[f"Cam{num}"]["IpAddress"])
            #cam_ip_list[f"cam{num}"] = self.cameras_info[f"Cam{num}"]["IpAddress"]

        cam_array = py.InstantCameraArray(NUM_CAMERAS)
        for ip in cam_ip_list:
            cam_array.Attach(self.tlf.CreateDevice(ip))

        cam_array.Open()

        frame_counts = [0]*self.NUM_CAMERAS
        cam_array.StartGrabbing()
        while True:
            with cam_array.RetrieveResult(1000) as res:
                if res.GrabSucceeded():
                    img_nr = res.ImageNumber
                    cam_id = res.GetCameraContext()
                    frame_counts[cam_id] = img_nr
                    print(f"cam #{cam_id}  image #{img_nr}")
                    
                    # do something with the image ....
                    img = res.GetArray()
                    plt.imshow(img)
                    plt.savefig(f"./img_saved/{formatted}_{cam_id}_img.png")
                    
                    # check if all cameras have reached 100 images
                    if min(frame_counts) >= frames_to_grab:
                        print( f"all cameras have acquired {frames_to_grab} frames")
                        break
            
        cam_array.StopGrabbing()
        cam_array.Close()

        now2 = time.time()
        print(f"process time: {now2-now1}")


def from_config(config):
    """Creates an dictionary of the GFA camera information
    from a configuration file.
    """

    with open(config) as f:
        film = yaml.load(f, Loader=yaml.FullLoader)

    return film
