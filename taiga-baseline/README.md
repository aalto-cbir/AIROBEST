# AIROBEST
Repository for [AIROBEST project](https://sensillence.github.io/AIROBEST/)

## Data set
The TAIGA dataset is available at https://doi.org/10.23729/fe7ce882-8125-44e7-b0cf-ae652d7ed0d5

Download all the files in a directory /any/path/TAIGA .

## Getting started
Make sure to install all the requirements, run: `pip install -r requirements.txt`

### Folder structure
Taiga-baseline contains three directories:
- `preprocess` contains the code used to preprocess the above TAIGA dataset into pytorch Tensor that can be used for training. 
- `training`: contains the code to train the baseline model.
- `analysis`:

#### Data preprocessing
- Access `preprocess` directory
- Specify paths to hyperspectral data and forest data in `spreprocess.sh` script, for more options, run: `python preprocess.py -h`
- Run `sbatch spreprocess.sh` to submit preprocessing job on Puhti
- This process will generate the following files and save them under `./data` directory:
    * `hyper_image.pt`: hyperspectral image saved as pytorch Tensor. We use this instead of spectral format to increase GPU utilization.
    * `hyperspectral_src_l2norm_along_channel.pt`: tensor contains the corresponding norm of each pixel from hyperspectral image, norms are computed along color channel.
    * `hyperspectral_tgt_normalized.pt`: tensor of target labels, labels for regression tasks has been normalized to [0, 1] scale.
    * `metadata.pt`: dictionary contains information about the input data, such as class values for each classification task


#### Training
To be updated


#### Analysis
To be updated
