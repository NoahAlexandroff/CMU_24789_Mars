# CMU_24789_Mars
Repository of our final project for 24-789 "Deep Learning for Engineers" at Carnegie Mellon University.


## Python Environment Setup
You can configure your Python environment using ```conda```:

```
    conda env create -f requirements.yml
```

## Downloading Depth Data
The AI4Mars dataset does not (yet) contain depth data. This directory contains scripts that can be used
to scrape the correct depth data for the files already contained within the dataset. The default settings
assume that version 3 of the AI4Mars dataset is already downloaded and placed in the "data" folder in 
this directory. If this is not the case, these defaults can be overwritten. This, and other useful inputs
that the user may find helpful, can be found using:

```
    python scrape_msl_depth_data.py --help
```

You can scrape the depth files that correspond to the MSL (Curiosity) labeled data by being in the root 
of this directory and using:

```
    python scrape_msl_depth_data.py
```

Due to the large amount of data that will be scraped (~73 GB), it is useful to run this script in 
parallel to download multiple files at once. This can be done by being in the root of this directory and 
using:

```
    ./shell_scripts/scrape_msl.sh
```

This shell script will run the above python script multiple times in the background to download the 
desired files faster.

The same process can be used to download depth data for the MER (Spirit and Opportunity) labeled data,
but wih the scripts that have "mer" substituted for "msl".

Alternatively, you may download the depth data from [MSL](https://drive.google.com/drive/folders/138KDuqOHcYZUWs0cA14fSSpyohjsM5_S?usp=sharing) and [MER](https://drive.google.com/drive/folders/1ybI8Wid3mHunNyBFf6BKdyI4Plj9pu8H?usp=sharing).