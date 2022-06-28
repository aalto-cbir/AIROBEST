# AIROBEST
Repository for [AIROBEST project](https://sensillence.github.io/AIROBEST/)

## Data set
The TAIGA dataset is available at https://doi.org/10.23729/fe7ce882-8125-44e7-b0cf-ae652d7ed0d5.

Download all the files in your local directory /any/path/TAIGA.

## Getting started
- Create virtual environment: `python3 -mvenv venv`.
- Activate virtual environment: `source venv/bin/activate`.
- Make sure to install all the requirements, run: `pip install -r requirements.txt`.

### Folder structure
Taiga-baseline contains three directories:
- `preprocess`: the code used to preprocess the above TAIGA dataset into pytorch Tensor that can be used for training. 
- `training`: the code to train the baseline model and produce predictions.
- `analysis`: the code to evaluate prediction results.

#### Data preprocessing
- Enter `preprocess` directory.

- Specify directory of TAIGA hyperspectral and forest data in `spreprocess.sh` script.
- For more options, run: `./preprocess.py -h`.

- Run `spreprocess.sh`.
- This process will generate the following files and save them under `./data/TAIGA` directory:
    * `hyperspectral_src.pt`: hyperspectral image saved as pytorch Tensor. We use this instead of spectral format to increase GPU utilization.
    * `hyperspectral_tgt_normalized.pt`: tensor of target labels, labels for regression tasks has been normalized to [0, 1] scale.
    * `image_norm_l2norm_along_channel.pt`: tensor contains the corresponding norm of each pixel from hyperspectral image, norms are computed along color channel.
    * `metadata.pt`: dictionary contains information about the input data, such as class values for each classification task.
    * `rgb_image.png`: hyperspectral image in RGB format.

#### Data splitting
- Enter `./training/custom_split_data` directory.
- Run `./create_stand_ids.py` to create stand_ids map `stands_ids.npy` under `./data/TAIGA` directory.
- Run `./scustom_split_data.sh` to generate `mask.npy` and create a custom data split.
- Run `./taiga_split_data.py` to generate a custom data split based on stands in `forest_stand_data.csv`.

#### Training
- Go back to `./training` directory.
- If you want to see real-time visualization, you must start `visdom` server first (see `visdom.md`).
- `sbatch strain.sh` to train on Puhti server.
- `sbatch sinference.sh` runs the inference on test data.

Training configurations:
- `hyper_data_path`: path to hyperspectral image saved in preprocessing stage.
- `src_norm_multiplier`: path to file contains pixel norm (corresponding to `hyperspectral_src_l2norm_along_channel.pt` file in preprocessing stage).
- `tgt_path`: path to target label.

For other options, run: `python train.py -h` for detailed explanation.

#### Analysis
- Enter `./analysis` directory.
- Run `./calculate_categorical_metrics.py` to calculate the performance metrics of the predicted categorical values, such as micro and macro accuracies.
- Run `./calculate_continuous_metrics.py` to calculate the performance metrics of the predicted continuous values, such as RMSE, rRMSE, rBias, R2, etc.
- Run `./predict_full_image.py` and `./create_full_dataset` to predict every pixel of the hyperspectral image.
- Run `./calculate_evaluation_metrics_full_image.py` to calculate the evaluation metrics for the whole hyperspectral image.
