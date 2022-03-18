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
    sol = math.floor((float(sclk)-128141132.)/88775.42064)
    return sol

