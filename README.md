# RAMCES

RAnking Markers for CEll Segmentation

![RAMCES Figure](ramces_fig.png)

## Installation

[//]: # (instructions, typical install time)

To use RAMCES, first type in the following commands into your command line.

```
git clone git@github.com:mdayao/ramces.git
cd ramces
```

### Dependencies

RAMCES requires `python (>=3.7)`. We *highly* recommend creating a separate **conda** environment to use RAMCES.

You can use the `environment.yml` or `requirements.txt` file to create your environment by doing either of the following:

```
conda env create -f environment.yml
conda activate ramces

OR

# without conda
python -m venv ramces # Python 3 required
source ramces/bin/activate
pip install -r requirements.txt # pip must be installed
```

Alternatively, the required packages are listed below. These can be installed via `conda` or `pip`. 

```
numpy (>=1.18.1)
pandas (>=1.0.0)
pytorch  (>=1.0.0)
torchvision (>=0.2.2)
opencv (>=3.4.2)
pywavelets (>=1.1.1)
tifffile (>=2021.2.26)
```

## Using RAMCES to output marker rankings and weighted images

This section describes how you can use RAMCES to rank markers and create weighted images on your own data. To do this, use the `rank_markers.py` script. There are a number of arguments that you must specify; you can see these options by running

```
python rank_markers.py -h
```

### Input file format

Before you can use RAMCES, you will need to have the following files prepared:

1. The trained model file (`--model-path`)
    
    By default, this will be the already-trained CNN model `models/trained_model.h5`. Unless you have your own trained model that you want to use, you do not need to specify anything here.

2. The image data, CODEX or otherwise (`--data-dir`)

    Put all of the data in a single directory. The data *must* be formatted in a specific way in order to use this script. Currently, it is formatted for images that are acquired in cycles (e.g. CODEX). In a future update of this repository, the allowed data input will be more flexible to accommodate a range of data formats.
    
    - There must be a single 2D TIFF file for each marker/protein for every tile in the dataset. This means that if there are 16 image tiles and 20 distinct proteins profiled, there would be 320 TIFF files in the data directory.
    - Each TIFF filename should contain the following patterns to specify the cycle and channel of each file (and hence specify the marker):
        - `tXXX` for cycle number, starting at 1
        - `cXXX` for channel number, starting at 1
        - The tile number should also be specified in some way, but there is no required pattern for this.
        - For example, the filename for a marker at cycle 3, channel 2 at tile 5 could be: `mydataset_0005_t003_c002.tif`

    Note: This setup assumes that the images are already at the optimal/most focused z-plane. 

3. A CSV file with two columns listing the marker channel name and whether the channel should be scored by RAMCES (`--channels`)

    An example file with a total of 9 channels and 5 markers/proteins to input to RAMCES would look like:

    ```
    DAPI, False
    CD20, True
    CD4, True
    DAPI1, False
    CD45, True
    CD8, True
    DAPI2, False
    Podoplanin, True
    Blank, False
    ```

### Output

There are two items that RAMCES outputs when using the `rank_markers.py` script:

1. Marker ranking and scores (`--rank-path`)
    
    The script will output a csv file that gives the scores (between 0 and 1) for each marker. The higher the score, the more confident RAMCES is that the marker is suitable for cell segmentation. The filename/path must be specified in the `--rank-path` argument.

2. Weighted images (`--create-images`, `--output-weighted`, optional)

    **NOTE: This functionality is not available yet. A future update of this repository will include this.**

    If the `--create-images` flag is set, then the script will output images that combine the top ranked membrane markers, weighted by the scores given by RAMCES. The number of top ranked markers to use is given by the `--num-weighted` argument. The files are saved to the directory specified by the `--output-weighted` arugment. 

### Demo on CODEX data

[//]: # (instructions, expected output, expected run time for demo)

#### Data

A small demo dataset can be found in the `demo/` directory. There are 19 distinct markers of interest in this dataset. To output marker rankings and scores from RAMCES, run the following:

```
python rank_markers.py --data-dir ./demo/data --channels ./demo/channels.csv --num-cycles 7 --num-channels-per-cycle 4 --rank-path ./demo/ranking.csv
```

This will output the `ranking.csv` file in the `demo/` directory, which gives the scores given to each marker in the dataset.

## Training your own models

A future update of this repository will include instructions on how to train a CNN model with your own data. This may be desirable if your data is dissimilar to the CODEX data RAMCES was originally trained on.

[//]: # (### Required input files)

