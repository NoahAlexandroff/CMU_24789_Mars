import os
import math
import io

import numpy as np
import wget

import numpy as np
from PIL import Image
from planetaryimage import PDS3Image
  
import argparse


def mer_extract_sol(filename):
    """
    Extracts the sol from the filename for mer data

    Parameters:
    -----------
    filename: the name of the mer file that we are trying to find the sol of

    Returns:
    --------
    sol: the sol that corresponds to the provided filename
    """
    sclk = filename[4:13]
    if filename[0]=='1':
        sol = math.floor((float(sclk)-128141132.)/88775.42064)
    else:
        sol = math.floor((float(sclk)-126322960.)/88775.42064) 
    
    return sol

def mer_create_url(filename, prodid = "rnl", extension='.img'):
    """
    Generates a url that can be used to download mer files from the PDS. Default values will
    return url for rnl depth data.

    Parameters:
    -----------
    filename: the name of the MER eff file contained in AI4Mars dataset

    prodid: the prodid of the desired url. this determines which type of data will be  
    downloaded. the default value is "rnl". more information about available prodids at either
    https://pds-imaging.jpl.nasa.gov/data/mer/spirit/mer2no_0xxx/document/CAMSIS_V4-4_7-31-14.PDF or
    https://pds-imaging.jpl.nasa.gov/data/mer/opportunity/mer1no_0xxx/document/CAMSIS_V4-4_7-31-14.PDF
    depending upon the desired rover

    extension: the extension to be used for the desired url. the default value is ".img"


    Returns:
    --------
    combined_url : the generated url that can be used to download files from the PDS
    """

    sol = msl_extract_sol(filename)
    base_url = "https://pds-imaging.jpl.nasa.gov/data/mer/"
    if filename[0]=='1':
        data_url = "opportunity/mer1no_0xxx/data/"
    else: 
        data_url = "spirit/mer2no_0xxx/data/"
    sol_url = "sol" + f'{sol:05d}'+"/rdr/"
    file_url = update_filename_from_edr(filename, prodid = prodid, extension=extension)
    combined_url = base_url + data_url + sol_url + file_url
    
    return combined_url


def update_filename_from_eff(eff_filename, prodid, extension):
    """
    Generates an updated filename from an eff filename for mer files from the PDS.

    Parameters:
    -----------
    eff_filename: the name of the MSL eff file contained in AI4Mars dataset

    prodid: the prodid of the desired url. this determines which type of data will be  
    downloaded. more information about available prodids is available at
    https://pds-imaging.jpl.nasa.gov/data/mer/spirit/mer2no_0xxx/document/CAMSIS_V4-4_7-31-14.PDF or
    https://pds-imaging.jpl.nasa.gov/data/mer/opportunity/mer1no_0xxx/document/CAMSIS_V4-4_7-31-14.PDF
    depending upon the desired rover

    extension: the extension to be used for the desired url

    Returns:
    --------
    updated_filename : the updated filename that incorporates the desired prodid and
    extension
    """
    updated_filename = edr_filename[0:11] + prodid + edr_filename[14:-4] + extension

    return updated_filename