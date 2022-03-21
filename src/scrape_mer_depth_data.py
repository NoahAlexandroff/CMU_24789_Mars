import os
import math
import io
import logging
import argparse

import wget
import numpy as np
from PIL import Image
from planetaryimage import PDS3Image
  
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
    sclk = filename[2:11]
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

    sol = mer_extract_sol(filename)
    base_url = "https://pds-imaging.jpl.nasa.gov/data/mer/"
    if filename[0]=='1':
        data_url = "opportunity/mer1no_0xxx/data/"
    else: 
        data_url = "spirit/mer2no_0xxx/data/"
    sol_url = "sol" + f'{sol:04d}'+"/rdr/"
    file_url = update_filename_from_eff(filename, prodid = prodid, extension=extension)
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
    updated_filename = eff_filename[0:11] + prodid + eff_filename[14:-4] + extension

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
        try:
            file = wget.download(url, out = output_path, bar=None)
            img = PDS3Image.open(file)
            img_arr = img.data
            img_arr = img_arr.reshape(img_arr.shape[1:])
            img_arr = img_arr.astype(np.float32)
            pil_img = Image.fromarray(img_arr, mode='F')
            pil_img.save(output_path+filename)
            os.remove(output_path+filename[:-4]+'IMG')

            return 1
        except:
            logging.info(" Missing URL: " + url)

    return 0

def main(data_path, output_path, starting_index, ending_index):
    """
    Main function that downloads the desired file types that map to the existing
    eff files in the AI4Mars dataset. The default values are for range data. 

    Parameters:
    -----------
    data_path: the path to the data directory in the AI4Mars dataset

    output_path: the path of the directory the newly downloaded data should be stored

    Returns:
    --------
    num_files_downloaded: the number of files that were downloaded
    """
    num_files_downloaded = 0
    raw_filenames = os.listdir(data_path+'/mer/images/eff')[starting_index:ending_index]
    for filename in raw_filenames:
        depth_filename = update_filename_from_eff(filename, prodid = "rnl", extension = ".tiff")
        depth_url = mer_create_url(filename) 
        num_files_downloaded += download_file_from_url(depth_url, output_path, depth_filename)
    
    return num_files_downloaded

if __name__=='__main__':
    logging.basicConfig(filename="mer_scrape.log", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', help='The path to the data folder in the AI4Mars dataset',
                        default="./data/")
    parser.add_argument('--output-path', help='The path to the directory the msl depth data should be downloaded to.',
                        default=None)
    parser.add_argument('--starting-index', help='The starting index of files to allow for distributed download.',
                        default=0)
    parser.add_argument('--ending-index', help='The ending index of files to allow for distributed download.',
                        default=-1)
    args = parser.parse_args()
    if args.output_path == None:
        args.output_path = args.data_path + 'mer/images/rnl/'
    output = main(args.data_path, args.output_path, int(args.starting_index), int(args.ending_index))
    print('\n')
    print(f"Downloaded {output} files.")