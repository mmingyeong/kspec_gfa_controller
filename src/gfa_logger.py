#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2023-12-07
# @Filename: gfa_logger.py

import logging
import os
from datetime import datetime

__all__ = ["gfa_logger"]

class gfa_logger:
    """
    Custom logging system for the GFA project.

    Parameters
    ----------
    stream_level : int, optional
        The logging level for the console (default is logging.INFO).
    """

    _initialized_loggers = set()  # Track initialized loggers

    def __init__(self, file, stream_level=logging.INFO, log_dir="/home/kspec/mingyeong/kspec_gfa_controller/src/log"):
        """
        Initializes the logger with both a stream handler (console output) and a file handler (log file storage).

        Parameters
        ----------
        file : str
            The name of the file that will be used to create the logger.
        stream_level : int, optional
            The logging level for the console (default is logging.INFO).
        log_dir : str, optional
            Directory where log files are stored (default: /home/kspec/mingyeong/kspec_gfa_controller/src/log).
        """
        self.file_name = os.path.basename(file)
        self.logger = logging.getLogger(self.file_name)

        if self.file_name in gfa_logger._initialized_loggers:
            # If logger is already initialized, return without adding handlers again
            return

        self.logger.setLevel(logging.INFO)  # Set logger to capture all levels

        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)

        # Generate log file path with date-based naming
        log_filename = f"gfa_{datetime.now().strftime('%Y-%m-%d')}.log"
        log_file_path = os.path.join(log_dir, log_filename)

        # Console output formatting
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

        # StreamHandler for console output
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(stream_level)
        self.logger.addHandler(stream_handler)

        # FileHandler for log file storage
        file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)

        # Mark this logger as initialized
        gfa_logger._initialized_loggers.add(self.file_name)

    def info(self, message):
        """Log an INFO level message."""
        self.logger.info(f"{message} (at {self.file_name})")

    def debug(self, message):
        """Log a DEBUG level message."""
        self.logger.debug(f"{message} (at {self.file_name})")

    def warning(self, message):
        """Log a WARNING level message."""
        self.logger.warning(f"{message} (at {self.file_name})")

    def error(self, message):
        """Log an ERROR level message."""
        self.logger.error(f"{message} (at {self.file_name})")
