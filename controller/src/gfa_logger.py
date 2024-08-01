#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2023-12-07
# @Filename: gfa_log.ipynb

import logging
import os

__all__ = ["gfa_logger"]


# Logger 정의
class gfa_logger:
    """Custom logging system.

    Parameters
    ----------
    file : str
        The name of the logger.
    """

    def __init__(self, file):
        self.logger = logging.getLogger("gfa_logger")
        self.file_name = os.path.basename(file)
        self.logger.setLevel(logging.INFO) # should be changed ERROR later

        # StreamHandler
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.INFO)

        stream_handler.setLevel(logging.INFO)
        self.logger.addHandler(stream_handler)

        # FileHandler
        log_name = self.file_name.rstrip(".py")
        file_handler = logging.FileHandler(f"./src/log/{log_name}.log")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)

        file_handler.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)

    def info(self, value):
        self.logger.info("%s (at %s)" % (str(value), self.file_name))

    def debug(self, value):
        self.logger.debug("%s (at %s)" % (str(value), self.file_name))

    def warning(self, value):
        self.logger.warning("%s (at %s)" % (str(value), self.file_name))

    def error(self, value):
        self.logger.error("%s (at %s)" % (str(value), self.file_name))
