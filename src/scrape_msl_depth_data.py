import os
import math
import io

import numpy as np
import wget

import numpy as np
from PIL import Image
from planetaryimage import PDS3Image
  
import argparse


def msl_extract_sol(filename):
    """
    Extracts the sol from the filename for msl data

    Parameters:
    -----------
    filename: the name of the msl file that we are trying to find the sol of

    Returns:
    --------
    sol: the sol that corresponds to the provided filename
    """
    sclk = filename[4:13]
    sol = math.floor((float(sclk)-397446468.)/88775.42064)
    return sol
    
def msl_create_url(filename, prodid = "RNG", extension='.IMG'):
    """
    Generates a url that can be used to download msl files from the PDS. Default values will
    return url for RNG depth data.

    Parameters:
    -----------
    filename: the name of the MSL EDR file contained in AI4Mars dataset

    prodid: the prodid of the desired url. this determines which type of data will be  
    downloaded. the default value is "RNG". more information about available prodids at
    https://pds-imaging.jpl.nasa.gov/data/msl/MSLNAV_1XXX/DOCUMENT/MSL_CAMERA_SIS.PDF


    extension: the extension to be used for the desired url. the default value is ".IMG"


    Returns:
    --------
    combined_url : the generated url that can be used to download files from the PDS
    """

    sol = msl_extract_sol(filename)
    base_url = "https://pds-imaging.jpl.nasa.gov/data/msl/MSLNAV_1XXX" 
    if sol<1870: #Changed location of data at sol 1870
        data_url = "/DATA_V1/"
    else:
        data_url = "/DATA/"
    sol_url = "SOL" + f'{sol:05d}'+"/"
    file_url = update_filename_from_edr(filename, prodid = prodid, extension=extension)
    combined_url = base_url + data_url + sol_url + file_url
    
    return combined_url

def update_filename_from_edr(edr_filename, prodid, extension):
    """
    Generates an updated filename from an edr filename for msl files from the PDS.

    Parameters:
    -----------
    edr_filename: the name of the MSL EDR file contained in AI4Mars dataset

    prodid: the prodid of the desired url. this determines which type of data will be  
    downloaded. more information about available prodids is available at
    https://pds-imaging.jpl.nasa.gov/data/msl/MSLNAV_1XXX/DOCUMENT/MSL_CAMERA_SIS.PDF


    extension: the extension to be used for the desired url


    Returns:
    --------
    updated_filename : the updated filename that incorporates the desired prodid and
    extension
    """
    updated_filename = edr_filename[0:13] + prodid + edr_filename[16:-4] + extension

    return updated_filename

def download_file_from_url(url, output_path, filename):
    """
    Downloads the desired file from PDS and saves it in the desired location and
    format

    Parameters:
    -----------
    url: the url that will be used to download data from PDS

    output_path: the location that the downloaded file should be saved at

    filename: the name and format the downloaded file should be saved as


    Returns:
    --------
    1 or 0 : returns 1 if a file is downloaded or 0 if the file is already 
    present in output path
    """
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(output_path+filename):
        file = wget.download(url, out = output_path)
        img = PDS3Image.open(file)
        img_arr = img.data
        img_arr = img_arr.reshape(img_arr.shape[1:])
        img_arr = img_arr.astype(np.float32)
        pil_img = Image.fromarray(img_arr, mode='F')
        pil_img.save(output_path+filename)
        os.remove(output_path+filename[:-4]+'IMG')

        return 1

    return 0