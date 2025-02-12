#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Yongmin Yoon, Mingyeong Yang (yyoon@kasi.re.kr, mmingyeong@kasi.re.kr)
# @Date: 2024-05-16
# @Filename: gfa_astrometry.py

import os
import sys
import time
import json
import glob
import numpy as np
import shutil
import subprocess

from astropy.io import fits
from astropy.table import Table, vstack
from astropy.utils.data import get_pkg_data_filename

from concurrent.futures import ThreadPoolExecutor, as_completed

class gfa_astrometry:
    """
    A class to perform astrometry operations on GFA images.

    Attributes
    ----------
    logger : logging.Logger
        Logger for logging messages.
    inpar : dict
        Dictionary containing parameters loaded from a JSON file.
    raws : list of str
        List of raw image filenames.
    """

    def __init__(self, config, logger):
        """
        Initializes the gfa_astrometry class.

        Parameters
        ----------
        config : str
            Path to the configuration JSON file.
        logger : logging.Logger
            Logger instance for logging.
        """
        self.logger = logger
        self.logger.info("Initializing gfa_astrometry class.")

        with open(config, 'r') as file:
            self.inpar = json.load(file)

        base_dir = os.path.abspath(os.path.dirname(__file__))

        # Extract directories from config
        self.dir_path = os.path.join(base_dir, self.inpar['paths']['directories']['raw_images'])
        self.processed_dir = os.path.join(base_dir, self.inpar['paths']['directories']['processed_images'])
        self.temp_dir = os.path.join(base_dir, self.inpar['paths']['directories']['temp_files'])
        self.final_astrometry_dir = os.path.join(base_dir, self.inpar['paths']['directories']['final_astrometry_images'])
        self.star_catalog_path = os.path.join(base_dir, self.inpar['paths']['directories']['star_catalog'])
        self.cutout_path = os.path.join(base_dir, self.inpar['paths']['directories'].get('cutout_directory', 'cutout'))

        for directory in [self.dir_path, self.processed_dir, self.temp_dir, self.final_astrometry_dir, self.cutout_path]:
            os.makedirs(directory, exist_ok=True)

        # Load raw image files
        filepre = glob.glob(os.path.join(self.dir_path, '*.fits'))
        self.raws =  [os.path.basename(file) for file in filepre]
        self.raws = sorted(self.raws)
        self.logger.info(f"Loaded {len(self.raws)} FITS files.")

    def process_file(self, flname):
        """
        Processes a FITS file: subtracts sky values, crops the image, and saves the processed file.
        """
        self.logger.info(f"Processing file: {flname}")
        data_in_path = os.path.join(self.dir_path, flname)

        if not os.path.exists(data_in_path):
            self.logger.error(f"File not found: {data_in_path}")
            return None

        with fits.open(data_in_path, mode='update') as hdu_list:
            ori = hdu_list[0].data
            header = hdu_list[0].header
            ra_in, dec_in = header.get('RA'), header.get('DEC')

            # Extract sky values safely
            try:
                sky1 = ori[self.inpar['settings']['image_processing']['skycoord']['pre_skycoord1'][0], self.inpar['settings']['image_processing']['skycoord']['pre_skycoord1'][1]]
                sky2 = ori[self.inpar['settings']['image_processing']['skycoord']['pre_skycoord2'][0], self.inpar['settings']['image_processing']['skycoord']['pre_skycoord2'][1]]
            except IndexError:
                raise ValueError("Invalid sky coordinate indices.")

            # Subtract sky values in specified regions
            sub1 = tuple(self.inpar['settings']['image_processing']['sub_indices']['sub_ind1'])
            sub2 = tuple(self.inpar['settings']['image_processing']['sub_indices']['sub_ind2'])

            ori[sub1[0]:sub1[1], sub1[2]:sub1[3]] -= sky1
            ori[sub2[0]:sub2[1], sub2[2]:sub2[3]] -= sky2

            # Crop the image
            crop = tuple(self.inpar['settings']['image_processing']['crop_indices'])
            orif = ori[crop[0]:crop[1], crop[2]:crop[3]]

            # Define output directory and filename
            dir_out = self.processed_dir
            os.makedirs(dir_out, exist_ok=True)  # Ensure output directory exists
            newname = f'proc_{flname}'
            data_file_path = os.path.join(dir_out, newname)

            # Write to a new FITS file
            fits.writeto(data_file_path, orif, hdu_list[0].header, overwrite=True)

            # Close the original FITS file
            hdu_list.close()

            return  ra_in, dec_in, dir_out, newname

    def astrometry(self, ra_in, dec_in, dir_out, newname):
        """
        Performs astrometry on the processed FITS file.

        Parameters
        ----------
        ra_in : float
            Right Ascension from the processed FITS file header.
        dec_in : float
            Declination from the processed FITS file header.
        dir_out : str
            Directory where the processed file is stored.
        newname : str
            Name of the processed FITS file.

        Returns
        -------
        tuple
            A tuple containing:
            - CRVAL1 (float): RA value from the astrometry result.
            - CRVAL2 (float): DEC value from the astrometry result.
        """
        self.logger.info(f"Starting astrometry process for {newname}.")

        # solve-field 경로 확인
        solve_field_path = shutil.which("solve-field")
        if not solve_field_path:
            raise FileNotFoundError("solve-field not found! Ensure it is installed and available in the system PATH.")

        self.logger.info(f"Using solve-field from: {solve_field_path}")

        scale_low, scale_high = self.inpar['astrometry']['scale_range']
        radius = self.inpar['astrometry']['radius']
        cpu_limit = self.inpar['settings']['cpu']['limit']

        input_file_path = os.path.join(dir_out, newname)
        if not os.path.exists(input_file_path):
            self.logger.error(f"Input file for solve-field not found: {input_file_path}")
            raise FileNotFoundError(f"Input file for solve-field not found: {input_file_path}")

        # solve-field 명령어 실행
        input_command = (
            f"{solve_field_path} --cpulimit {cpu_limit} --dir {self.temp_dir} --scale-units degwidth "
            f"--scale-low {scale_low} --scale-high {scale_high} "
            f"--no-verify --no-plots --crpix-center -O --ra {ra_in} --dec {dec_in} --radius {radius} {input_file_path}"
        )

        self.logger.info(f"Running command: {input_command}")

        try:
            result = subprocess.run(input_command, shell=True, capture_output=True, text=True, check=True)
            #self.logger.info(f"solve-field stdout: {result.stdout}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"solve-field execution failed for {newname}")
            self.logger.error(f"solve-field stderr: {e.stderr}")
            raise RuntimeError(f"solve-field execution failed for {newname}")

        # solve-field 실행 후 temp 디렉토리 파일 확인
        self.logger.info(f"Checking generated files in {self.temp_dir} after solve-field execution.")
        list_files_command = f"ls -lh {self.temp_dir}"
        list_files = subprocess.run(list_files_command, shell=True, capture_output=True, text=True)
        self.logger.info(f"Files in temp directory:\n{list_files.stdout}")

        # `.new` 파일을 찾도록 변경
        solved_file_pattern = os.path.join(self.temp_dir, newname.replace('.fits', '.new'))
        solved_files = glob.glob(solved_file_pattern)

        if not solved_files:
            self.logger.error(f"Astrometry output file not found in {self.temp_dir}. Expected pattern: {solved_file_pattern}")
            self.logger.error(f"Files in directory: {os.listdir(self.temp_dir)}")
            raise FileNotFoundError(f"Astrometry output file not found: {solved_file_pattern}")

        new_fits_file = solved_files[0]
        converted_fits_file = new_fits_file.replace('.new', '.fits')

        # 파일 확장자 변경
        os.rename(new_fits_file, converted_fits_file)
        self.logger.info(f"Renamed {new_fits_file} to {converted_fits_file}")

        # Ensure the final astrometry directory exists
        os.makedirs(self.final_astrometry_dir, exist_ok=True)

        # Construct output file path
        dest_file_name = f"astro_{newname}"
        dest_path = os.path.join(self.final_astrometry_dir, dest_file_name)

        # Move the solved file to the final astrometry directory
        os.rename(converted_fits_file, dest_path)
        self.logger.info(f"Astrometry results moved to {dest_path}.")

        # Read the FITS header and extract CRVAL1, CRVAL2
        try:
            image_data_p, header = fits.getdata(dest_path, ext=0, header=True)
            crval1 = header['CRVAL1']
            crval2 = header['CRVAL2']
        except Exception as e:
            self.logger.error(f"Failed to read CRVAL1, CRVAL2 from {dest_path}: {e}")
            raise RuntimeError(f"Failed to extract astrometry data from {dest_path}")

        self.logger.info(f"Astrometry completed with CRVAL1: {crval1}, CRVAL2: {crval2}.")

        return crval1, crval2



    def star_catalog(self):
        """
        Combines multiple star catalog files into a single catalog.
        """
        self.logger.info("Starting star catalog generation.")
        
        # Check if temp_dir and star_catalog_path are set
        if not self.temp_dir:
            self.logger.error("Temp directory is not set.")
            return

        if not self.star_catalog_path:
            self.logger.error("Star catalog path is not set.")
            return
        
        # Get all files matching the '*.corr' pattern in the output directory
        filepre = glob.glob(os.path.join(self.temp_dir, '*.corr'))
        self.logger.info(f"Found {len(filepre)} .corr files in {self.temp_dir}.")
        
        if not filepre:
            self.logger.error("No .corr files found in the temp directory.")
            return
        
        # Extract the base names of the files    
        star_files_p = [os.path.basename(file) for file in filepre]
        star_files_p.sort()
        
        # Recreate the full paths for the sorted files
        star_files = [os.path.join(self.temp_dir, file) for file in star_files_p]

        # Read each FITS file and extract the first table (assuming HDU 1 is a table)
        tables = []
        
        for file in star_files:
            self.logger.info(f"Processing file: {file}")
            try:
                with fits.open(file) as hdul:
                    table_data = Table(hdul[1].data)
                    tables.append(table_data)
            except Exception as e:
                self.logger.error(f"Error reading {file}: {e}")
        
        if tables:
            combined_table = vstack(tables)
            combined_table.write(self.star_catalog_path, overwrite=True)
            self.logger.info(f"Star catalog generated and saved to {self.star_catalog_path}.")
        else:
            self.logger.warning("No valid tables found for stacking. Skipping star catalog generation.")


    def rm_tempfiles(self):
        """
        Removes temporary files in `temp_dir`.
        """
        self.logger.info("Removing temporary files.")
        try:
            shutil.rmtree(self.temp_dir)
            os.makedirs(self.temp_dir, exist_ok=True)
            self.logger.info(f"Temporary files removed from {self.temp_dir}.")
        except Exception as e:
            self.logger.error(f"Error removing temporary files: {e}")

    def combined_function(self, flname):
        """
        Combines the process_file and astrometry methods.

        Parameters
        ----------
        flname : str
            The name of the FITS file to process.

        Returns
        -------
        tuple
            A tuple containing:
            - CRVAL1 (float): RA value from the astrometry result.
            - CRVAL2 (float): DEC value from the astrometry result.

        Raises
        ------
        RuntimeError
            If `process_file` or `astrometry` fails, an error is raised and the program stops.
        """
        self.logger.info(f"Starting combined function for file: {flname}.")

        # Process the FITS file
        result = self.process_file(flname)
        if result is None:
            raise RuntimeError(f"Processing failed for {flname}. Stopping execution.")

        # Ensure correct unpacking
        try:
            ra_in, dec_in, dir_out, newname = result
        except TypeError as e:
            raise RuntimeError(f"Unexpected result format from process_file({flname}): {result}. Error: {e}")

        # Perform astrometry
        astrometry_result = self.astrometry(ra_in, dec_in, dir_out, newname)
        if astrometry_result is None:
            raise RuntimeError(f"Astrometry failed for {flname}. Stopping execution.")

        # Ensure correct unpacking
        try:
            crval1, crval2 = astrometry_result
        except TypeError as e:
            raise RuntimeError(f"Unexpected result format from astrometry({newname}): {astrometry_result}. Error: {e}")

        self.logger.info(f"Combined function completed for {flname}. CRVAL1: {crval1}, CRVAL2: {crval2}.")
        return crval1, crval2

    
    def preproc(self):
        """
        Preprocesses raw FITS files.

        - If no astrometric files exist, runs `combined_function` in parallel.
        - Otherwise, processes raw files separately.
        - Uses `ThreadPoolExecutor` for parallel execution.
        """
        start_time = time.time()

        if not self.raws:
            self.logger.warning("No FITS files found. Exiting preprocessing.")
            return

        self.logger.info(f"Starting preprocessing for {len(self.raws)} files.")

        # 확인: astrometry 디렉토리에 기존 데이터가 있는지 검사
        if not os.path.exists(self.final_astrometry_dir) or not os.listdir(self.final_astrometry_dir):
            self.logger.info(f"No astrometry files found in {self.final_astrometry_dir}, starting astrometric processing.")

            crval1_results, crval2_results = [], []
            failed_files = []

            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {executor.submit(self.combined_function, flname): flname for flname in self.raws}

                for future in as_completed(futures):
                    flname = futures[future]
                    try:
                        result = future.result()
                        if result is not None:
                            crval1, crval2 = result
                            crval1_results.append(crval1)
                            crval2_results.append(crval2)
                            self.logger.debug(f"Processed {flname} successfully. CRVAL1: {crval1}, CRVAL2: {crval2}")
                        else:
                            self.logger.warning(f"Skipping {flname} due to errors.")
                            failed_files.append(flname)
                    except Exception as e:
                        self.logger.error(f"Error processing {flname}: {e}")
                        failed_files.append(flname)

            self.logger.info(f"CRVAL1 results: {crval1_results}")
            self.logger.info(f"CRVAL2 results: {crval2_results}")

            if failed_files:
                self.logger.warning(f"Failed to process {len(failed_files)} files: {failed_files}")

            self.star_catalog()
            self.rm_tempfiles()

        else:
            self.logger.info(f"Astrometry data exists. Processing raw files separately.")

            failed_files = []
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {executor.submit(self.process_file, flname): flname for flname in self.raws}

                for future in as_completed(futures):
                    flname = futures[future]
                    try:
                        result = future.result()
                        if result is not None:
                            self.logger.debug(f"Processed {flname} successfully.")
                            self.logger.info(result)
                        else:
                            self.logger.warning(f"Skipping {flname} due to errors.")
                            failed_files.append(flname)
                    except Exception as e:
                        self.logger.error(f"Error processing {flname}: {e}")
                        failed_files.append(flname)

            if failed_files:
                self.logger.warning(f"Failed to process {len(failed_files)} files: {failed_files}")
        end_time = time.time()
        running_time = end_time - start_time
        self.logger.info(f"All files processed in {running_time} seconds.")   
        self.logger.info("Preprocessing completed.")
