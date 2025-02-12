#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Yongmin Yoon, Mingyeong Yang (yyoon@kasi.re.kr, mmingyeong@kasi.re.kr)
# @Date: 2024-05-16
# @Filename: gfa_guider.py

import os
import sys
import json
import glob
import math
import warnings

import numpy as np
from scipy.optimize import curve_fit

from astropy.io import fits, ascii
from astropy.wcs import WCS
from astropy.stats import sigma_clip
from astropy.utils.exceptions import AstropyWarning

import photutils.detection as pd


class gfa_guider:
    """
    A class to handle guide star operations using GFA (Guide Focus Alignment) data.

    Attributes
    ----------
    logger : logging.Logger
        Logger for logging messages.
    inpar : dict
        Dictionary containing parameters loaded from a JSON file.
    boxsize : int
        Size of the box used for star centroiding.
    crit_out : float
        Critical threshold for offset computation.
    peakmax : float
        Maximum peak value for star selection.
    peakmin : float
        Minimum peak value for star selection.
    ang_dist : float
        Maximum angular distance for star selection.
    """

    def __init__(self, config, logger):
        """
        Initialize the gfa_guider class with a configuration file.

        Parameters
        ----------
        config_file : str
            Path to the JSON file containing the input parameters.
        logger : logging.Logger
            Logger instance for logging.
        """
        self.logger = logger
        self.logger.info("Initializing gfa_guider class.")
        
        # Load configuration JSON file with error handling
        try:
            with open(config, 'r') as file:
                self.inpar = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.error(f"Error loading JSON file: {e}")
            raise RuntimeError("Failed to load configuration file.")

        # 현재 실행 중인 파일의 위치 기준으로 디렉토리 설정
        base_dir = os.path.abspath(os.path.dirname(__file__))

        self.processed_dir = os.path.join(base_dir, self.inpar['paths']['directories']['processed_images'])
        self.final_astrometry_dir = os.path.join(base_dir, self.inpar['paths']['directories']['final_astrometry_images'])
        self.cutout_path = os.path.join(base_dir, self.inpar['paths']['directories']['cutout_directory'])

        # Assign parameters from the JSON config
        self.boxsize = self.inpar['detection']['box_size']
        self.crit_out = self.inpar['detection']['criteria']['critical_outlier']
        self.peakmax = self.inpar['detection']['peak_detection']['max']
        self.peakmin = self.inpar['detection']['peak_detection']['min']
        self.ang_dist = self.inpar['catalog_matching']['tolerance']['angular_distance']
        self.pixel_scale = self.inpar['settings']['image_processing']['pixel_scale']


    def load_image_and_wcs(self, image_file):
        """
        Load image data and WCS from a FITS file.

        Parameters
        ----------
        image_file : str
            Path to the FITS file.

        Returns
        -------
        image_data_p : ndarray
            The image data array.
        header : Header
            The FITS header.
        wcs : WCS
            The WCS object.
        """
        self.logger.info(f"Loading image and WCS from: {image_file}")

        warnings.filterwarnings('ignore', category=AstropyWarning)

        try:
            image_data_p, header = fits.getdata(image_file, ext=0, header=True)
            wcs = WCS(header)
            return image_data_p, header, wcs
        except FileNotFoundError:
            self.logger.error(f"File not found: {image_file}")
            raise
        except Exception as e:
            self.logger.error(f"Error reading FITS file: {e}")
            raise


    def load_only_image(self, image_file):
        """
        Load only the image data from a FITS file.

        Parameters
        ----------
        image_file : str
            Path to the FITS file containing the image.

        Returns
        -------
        image_data_p : ndarray
            The image data array.
        """
        self.logger.info(f"Loading image data from file: {image_file}")
        warnings.filterwarnings('ignore', category=AstropyWarning)
        image_data_p = fits.getdata(image_file, ext=0)
        return image_data_p

    def background(self, image_data_p):
        """
        Perform sigma clipping to derive background and its standard deviation.

        Parameters
        ----------
        image_data_p : ndarray
            The image data array.

        Returns
        -------

        image_data : ndarray
            The background-subtracted image data.
        stddev : float
            The standard deviation of the background.
        """
        self.logger.info("Performing sigma clipping to derive background and standard deviation.")
        image_data = np.zeros_like(image_data_p, dtype=float)
        x_split = 511

        region1 = image_data_p[:, :x_split]
        sigclip1 = sigma_clip(region1, sigma=3, maxiters=False, masked=False)
        avg1 = np.mean(sigclip1)
        image_data[:, :x_split] = region1 - avg1

        region2 = image_data_p[:, x_split:]
        sigclip2 = sigma_clip(region2, sigma=3, maxiters=False, masked=False)
        avg2 = np.mean(sigclip2)
        image_data[:, x_split:] = region2 - avg2

        sigclip = sigma_clip(image_data, sigma=3, maxiters=False, masked=False)
        stddev = np.std(sigclip)

        self.logger.info(f"Background subtraction completed with standard deviation: {stddev:.4f}")
        return image_data, stddev

    def load_star_catalog(self, crval1, crval2):
        """
        Load the guide star catalog.

        Parameters
        ----------
        crval1 : float
            The reference RA value from the WCS header.
        crval2 : float
            The reference DEC value from the WCS header.

        Returns
        -------
        ra1_rad, dec1_rad, ra2_rad, dec2_rad, ra_p, dec_p, flux
        """
        base_dir = os.path.abspath(os.path.dirname(__file__))
        star_catalog_path = os.path.join(base_dir, 'img', 'combined_star.fits')

        self.logger.info(f"Using star catalog file: {star_catalog_path}")

        if not os.path.exists(star_catalog_path):
            self.logger.error(f"Star catalog file not found: {star_catalog_path}")
            raise FileNotFoundError(f"Star catalog file not found: {star_catalog_path}")

        with fits.open(star_catalog_path) as hdul:
            data = hdul[1].data
            ra_p = data[self.inpar['catalog_matching']['fields']['ra_column']]
            dec_p = data[self.inpar['catalog_matching']['fields']['dec_column']]
            flux = data[self.inpar['catalog_matching']['fields']['mag_flux']]

        ra1_rad = np.radians(crval1)
        dec1_rad = np.radians(crval2)
        ra2_rad = np.radians(ra_p)
        dec2_rad = np.radians(dec_p)

        self.logger.info(f"Guide star catalog loaded successfully from {star_catalog_path} with {len(ra_p)} stars.")
        return ra1_rad, dec1_rad, ra2_rad, dec2_rad, ra_p, dec_p, flux


    def select_stars(self, ra1_rad, dec1_rad, ra2_rad, dec2_rad, ra_p, dec_p, flux):
        """
        Select guide stars based on angular distance and flux.

        Parameters
        ----------
        ra1_rad, dec1_rad : ndarray
            Reference RA/DEC in radians.
        ra2_rad, dec2_rad : ndarray
            Star catalog RA/DEC in radians.
        ra_p, dec_p : ndarray
            Star catalog RA/DEC in degrees.
        flux : ndarray
            Flux values from the catalog.

        Returns
        -------
        ra : ndarray
            Selected RA values.
        dec : ndarray
            Selected DEC values.
        """
        self.logger.info("Selecting stars based on angular distance and flux.")

        # Compute angular distance
        delta_sigma = np.arccos(
            np.sin(dec1_rad) * np.sin(dec2_rad) +
            np.cos(dec1_rad) * np.cos(dec2_rad) * np.cos(ra1_rad - ra2_rad)
        )
        angular_distance_degrees = np.degrees(delta_sigma)

        # Remove NaN values from flux before filtering
        valid_flux = np.nan_to_num(flux, nan=0.0)

        # Apply selection mask
        mask = (angular_distance_degrees < self.ang_dist) & (valid_flux > self.inpar['catalog_matching']['tolerance']['mag_flux_min'])
        
        ra_selected = ra_p[mask]
        dec_selected = dec_p[mask]
        flux_selected = valid_flux[mask]

        self.logger.info(f"Selected {len(ra_selected)} stars.")
        return ra_selected, dec_selected, flux_selected


    def radec_to_xy_stars(self, ra, dec, wcs):
        """
        Convert RA/DEC of guide stars into X/Y positions in the image.

        Parameters
        ----------
        ra : ndarray
            The RA values of the selected stars.
        dec : ndarray
            The DEC values of the selected stars.
        wcs : WCS
            The WCS (World Coordinate System) object.

        Returns
        -------
        dra : ndarray
            The X positions of the stars (rounded to nearest pixel).
        ddec : ndarray
            The Y positions of the stars (rounded to nearest pixel).
        dra_f : ndarray
            The precise X positions of the stars.
        ddec_f : ndarray
            The precise Y positions of the stars.
        """
        self.logger.info("Converting RA/DEC to X/Y positions in the image.")
        dra = np.zeros(len(ra), dtype=float)
        ddec = np.zeros(len(ra), dtype=float)
        dra_f = np.zeros(len(ra), dtype=float)
        ddec_f = np.zeros(len(ra), dtype=float)
        for ii in range(len(ra)):
            dra_p, ddec_p = wcs.world_to_pixel_values(ra[ii], dec[ii])
            dra[ii] = np.round(dra_p) + 1
            ddec[ii] = np.round(ddec_p) + 1
            dra_f[ii] = dra_p + 1
            ddec_f[ii] = ddec_p + 1
        self.logger.info("Conversion to X/Y positions completed.")
        return dra, ddec, dra_f, ddec_f

    def cal_centroid_offset(self, dra, ddec, dra_f, ddec_f, stddev, wcs, fluxn, file_counter, cutoutn_stack, image_data):
        """
        Calculate the centroid offset for guide stars.

        Parameters
        ----------
        dra : ndarray
            The X positions of the stars (rounded to nearest pixel).
        ddec : ndarray
            The Y positions of the stars (rounded to nearest pixel).
        dra_f : ndarray
            The precise X positions of the stars.
        ddec_f : ndarray
            The precise Y positions of the stars.
        stddev : float
            The standard deviation of the background.
        wcs : WCS
            The WCS (World Coordinate System) object.
        image_data : ndarray
            The image data array.

        Returns
        -------
        dx : list
            List of X offsets.
        dy : list
            List of Y offsets.
        peakc : list
            List of peak values.
        """

        self.logger.info("Calculating centroid offsets for guide stars.")
        dx = []
        dy = []
        peakc = []
        max_flux = max([flux for flux in fluxn if flux < 30000], default=None)
        boxsize = self.boxsize

        # Get directory path for saving FITS files
        dir_cutout = self.cutout_path
        for jj in range(len(dra)):
            try: 
                #finding peak in the cutout image of each guide star
                cutout=image_data[int(ddec[jj]-1-boxsize/2) : int(ddec[jj]-1+boxsize/2+1), int(dra[jj]-1-boxsize/2) :int(dra[jj]-1+boxsize/2+1)]

                thres=np.zeros((cutout.shape[0], cutout.shape[1]), dtype=float)+5*stddev
                peak=pd.find_peaks(cutout,thres, box_size=boxsize/4, npeaks=1)       

                x_peak=peak['x_peak'][0]
                y_peak=peak['y_peak'][0]
                peakv=peak['peak_value'][0]

                peakc.append(peakv)

                nra=int(dra[jj]-(0.5*boxsize-x_peak))
                ndec=int(ddec[jj]-(0.5*boxsize-y_peak))

                #calculating the center of light of each guide star in a smaller cutout image
                cutout2=image_data[int(ndec-1-boxsize/4) : int(ndec-1+boxsize/4+1), int(nra-1-boxsize/4) :int(nra-1+boxsize/4+1)]

                # Save cutout2 as FITS file if fluxn[jj] is equal to max(fluxn)
                if fluxn[jj] == max_flux:
                    cutoutnp=image_data[int(ndec-1-boxsize/2) : int(ndec-1+boxsize/2+1), int(nra-1-boxsize/2) :int(nra-1+boxsize/2+1)]
                    cutoutn=cutoutnp/np.max(cutoutnp)*1000
                    fits_file = f"{dir_cutout}/cutout_fluxmax_{file_counter}.fits"
                    fits.writeto(fits_file, cutoutn, overwrite=True)

                    # Append cutoutn to the stack list for averaging
                    cutoutn_stack.append(cutoutn)

                xcs=0
                ycs=0

                for kk in range(cutout2.shape[0]):
                    for ll in range(cutout2.shape[1]):
                        xcs += cutout2[kk, ll] * ll
                        ycs += cutout2[kk, ll] * kk

                xc=xcs/np.sum(cutout2) 
                yc=ycs/np.sum(cutout2) 

                fra=(nra-(boxsize/4-xc))
                fdec=(ndec-(boxsize/4-yc))   
            
                #deriving ra dec offsets corresponding to 1 pixel offset along the 'x-axis'
                x1 = fra  
                y1 = fdec 
                x2 = fra+1  
                y2 = fdec
                ra1, dec1 = wcs.pixel_to_world_values(x1, y1)
                ra2, dec2 = wcs.pixel_to_world_values(x2, y2)
                x1d=(ra2-ra1)*3600
                x2d=(dec2-dec1)*3600
                #print(x1d)
                #print(x2d)
                #print( x1d*math.cos(cdec*math.pi/180.) )

                #deriving ra dec offsets corresponding to 1 pixel offset along the 'y-axis'
                x1 = fra
                y1 = fdec 
                x2 = fra   
                y2 = fdec+1 
                ra1, dec1 = wcs.pixel_to_world_values(x1, y1)
                ra2, dec2 = wcs.pixel_to_world_values(x2, y2)
                y1d=(ra2-ra1)*3600
                y2d=(dec2-dec1)*3600
                #print(y1d)
                #print(y2d)

                #converting the x y offset of guide star locations into ra dec offsets
                dx.append((fra -dra_f[jj]) * x1d + (fdec - ddec_f[jj]) * x2d)
                dy.append((fra -dra_f[jj]) * y1d + (fdec - ddec_f[jj]) * y2d)
    
            except Exception as e:
                dx.append(0)
                dy.append(0)
                peakc.append(-1)
                self.logger.warning(f"Error finding peaks: {e}")

        self.logger.info("Centroid offset calculation completed.")
        return dx, dy, peakc, cutoutn_stack

    def peak_select(self, dx, dy, peakc):
        """
        Select guide stars based on peak values.

        Parameters
        ----------
        dx : list
            List of X offsets.
        dy : list
            List of Y offsets.
        peakc : list
            List of peak values.

        Returns
        -------
        dxn : ndarray
            Selected X offsets.
        dyn : ndarray
            Selected Y offsets.
        pindn : ndarray
            Indices of selected stars.
        """
        self.logger.info("Selecting guide stars based on peak values.")
        peakn = np.array(peakc)
        pind = np.where((peakn > self.peakmin) & (peakn < self.peakmax))
        pindn = pind[0]
        dxn = np.array([dx[i] for i in pindn])
        dyn = np.array([dy[i] for i in pindn])
        self.logger.info(f"Selected {len(dxn)} guide stars based on peaks.")
        return dxn, dyn, pindn

    def cal_final_offset(self, dxp, dyp, pindp):
        """
        Calculate the final offset using selected guide stars.

        Parameters
        ----------
        dxp : ndarray
            Selected X offsets.
        dyp : ndarray
            Selected Y offsets.
        pindp : ndarray
            Indices of selected stars.

        Returns
        -------
        fdx : float or str
            Final X offset or 'Warning' if the number of guide stars is insufficient.
        fdy : float or str
            Final Y offset or 'Warning' if the number of guide stars is insufficient.
        """
        self.logger.info("Calculating the final offset using selected guide stars.")
        if len(pindp) > 0.5:
            distances = np.sqrt(dxp ** 2 + dyp ** 2)
            clipped_data = sigma_clip(distances, sigma=3, maxiters=False)
            cdx = dxp[~clipped_data.mask]
            cdy = dyp[~clipped_data.mask]

            if len(cdx) > 4:
                max_dist_index = np.argmax(np.sqrt(cdx**2 + cdy**2))
                min_dist_index = np.argmin(np.sqrt(cdx**2 + cdy**2))
                cdx = np.delete(cdx, [min_dist_index, max_dist_index])
                cdy = np.delete(cdy, [min_dist_index, max_dist_index])

            fdx = np.mean(cdx)
            fdy = np.mean(cdy)

            if np.sqrt(fdx ** 2 + fdy ** 2) > self.crit_out:
                self.logger.info(f"Final offset: fdx={fdx}, fdy={fdy}")
                return fdx, fdy
            else:
                self.logger.warning("Final offset is within critical threshold; returning 0, 0.")
                return 0, 0
        else:
            self.logger.warning("Insufficient guide stars for offset calculation; returning 'Warning'.")
            return 'Warning', 'Warning'


    def isotropic_gaussian_2d(self, xy, amp, x0, y0, sigma, offset):
        """
        2D Isotropic Gaussian function.

        Parameters
        ----------
        xy : tuple of ndarray
            Meshgrid of (x, y) coordinates.
        amp : float
            Amplitude (peak value) of the Gaussian.
        x0, y0 : float
            Center coordinates of the Gaussian.
        sigma : float
            Standard deviation (spread) of the Gaussian.
        offset : float
            Constant background offset.

        Returns
        -------
        ndarray
            Flattened Gaussian function values.
        """
        x, y = xy
        g = offset + amp * np.exp(-(((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2)))
        return g.ravel()


    def cal_seeing(self, cutoutn_stack):
        """
        Calculate the Full-Width at Half-Maximum (FWHM) of a star in an image cutout.

        This function stacks multiple cutouts, fits an isotropic Gaussian function, and 
        extracts the FWHM, which indicates the seeing condition.

        Parameters
        ----------
        cutoutn_stack : list of ndarray
            List of image cutouts.

        Returns
        -------
        float
            Calculated FWHM value in arcseconds.

        Logs:
        -----
        - INFO: Saves the stacked image as a FITS file.
        - ERROR: If Gaussian fitting fails.
        - WARNING: If FWHM is unexpectedly large or small.
        """
        # Compute median stack of cutouts
        averaged_cutoutn = np.median(cutoutn_stack, axis=0)
        fits_file = os.path.join(self.cutout_path, "averaged_cutoutn.fits")
        
        try:
            fits.writeto(fits_file, averaged_cutoutn, overwrite=True)
            self.logger.info(f"Saved averaged cutout to {fits_file}")
        except Exception as e:
            self.logger.error(f"Failed to save averaged cutout FITS file: {e}")

        # Generate coordinate grid
        x = np.arange(averaged_cutoutn.shape[1])
        y = np.arange(averaged_cutoutn.shape[0])
        x, y = np.meshgrid(x, y)
        xdata = np.vstack((x.ravel(), y.ravel()))

        # Initial guess: amp, x0, y0, sigma, offset
        initial_guess = (np.max(averaged_cutoutn), averaged_cutoutn.shape[1] // 2,
                        averaged_cutoutn.shape[0] // 2, 1, 0)

        # Initial guess for parameters: amp, x0, y0, sigma, offset
        initial_guess = (np.max(averaged_cutoutn), averaged_cutoutn.shape[1] // 2, averaged_cutoutn.shape[0] // 2, 1, 0)

        # Fit the model to the data
        params, _ = curve_fit(self.isotropic_gaussian_2d, xdata, averaged_cutoutn.ravel(), p0=initial_guess)

        # Extract the fitted parameters
        amp, x0, y0, sigma, offset = params

        # Calculate FWHM
        fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma

        return fwhm*self.pixel_scale


    def exe_cal(self):
        """
        Execute the guide star calibration process.

        This function processes astrometric and guide star data, determines offsets,
        and calculates the final correction values, including the Full-Width at Half-Maximum (FWHM).

        Returns
        -------
        tuple:
            - fdx (float): Final X offset in arcseconds.
            - fdy (float): Final Y offset in arcseconds.
            - fwhm (float): Calculated Full-Width at Half-Maximum (FWHM) value in arcseconds.

        Logs:
        -----
        - INFO: Processing start, file counts, and completion.
        - ERROR: Issues during file processing, missing files.
        - WARNING: If no valid FITS files are found.
        """
        self.logger.info("Starting guide star calibration process.")

        # Load astrometric files
        astro_dir = self.final_astrometry_dir
        proc_dir = self.processed_dir

        if not astro_dir or not proc_dir:
            self.logger.error("Astrometric or processed directory path is missing in configuration.")
            return np.nan, np.nan, np.nan

        astro_files = sorted(glob.glob(os.path.join(astro_dir, '*.fits')))
        proc_files = sorted(glob.glob(os.path.join(proc_dir, '*.fits')))

        if not astro_files:
            self.logger.error(f"No astrometry FITS files found in {astro_dir}. Exiting calibration.")
            return np.nan, np.nan, np.nan

        if not proc_files:
            self.logger.error(f"No processed FITS files found in {proc_dir}. Exiting calibration.")
            return np.nan, np.nan, np.nan

        self.logger.info(f"Found {len(astro_files)} astrometry files and {len(proc_files)} processed files.")

        dxpp, dypp, pindpp = [], [], []
        cutoutn_stack = []
        file_counter = 1  # Counter for FITS file names

        # Process each file pair
        for astro_file, proc_file in zip(astro_files, proc_files):
            try:
                self.logger.info(f"Processing FITS files: {astro_file}, {proc_file}")

                image_file_a = astro_file        
                image_data_x, header, wcs = self.load_image_and_wcs(image_file_a)        
                crval1, crval2 = header['CRVAL1'], header['CRVAL2']       

                image_file = proc_file
                image_data_p = self.load_only_image(image_file)
                
                image_data, stddev=self.background(image_data_p)
                ra1_rad, dec1_rad, ra2_rad, dec2_rad, ra_p, dec_p, flux = self.load_star_catalog(crval1, crval2)
                ra, dec, fluxn = self.select_stars(ra1_rad, dec1_rad, ra2_rad, dec2_rad, ra_p, dec_p, flux)
                dra, ddec, dra_f, ddec_f = self.radec_to_xy_stars(ra, dec, wcs)
            
                dx, dy, peakc, cutoutn_stack = self.cal_centroid_offset(dra, ddec, dra_f, ddec_f, stddev, wcs, fluxn, file_counter, cutoutn_stack, image_data)   
                # Increment the counter after saving
                file_counter += 1

                dxn, dyn, pindn = self.peak_select(dx, dy, peakc)        
                dxpp.append(dxn)
                dypp.append(dyn)
                pindpp.append(pindn)

            except Exception as e:
                self.logger.error(f"Error processing {astro_file} and {proc_file}: {e}")
                raise RuntimeError(f"Critical error in guide star processing for {astro_file}: {e}")

        if not dxpp or not dypp or not pindpp:
            self.logger.error("No valid guide star data collected. Calibration failed.")
            return np.nan, np.nan, np.nan

        # Compute final offsets
        dxp, dyp, pindp = np.concatenate(dxpp), np.concatenate(dypp), np.concatenate(pindpp)
        fdx, fdy = self.cal_final_offset(dxp, dyp, pindp)

        # Compute FWHM
        fwhm = self.cal_seeing(cutoutn_stack)
        self.logger.info(f"FWHM: {fwhm:.2f} arcsec")

        self.logger.info(f"Guide star calibration completed. Final offsets: fdx={fdx:.5f}, fdy={fdy:.5f}, FWHM={fwhm:.5f} arcsec")
        return fdx, fdy, fwhm
