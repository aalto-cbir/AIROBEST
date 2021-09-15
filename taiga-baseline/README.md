# AIROBEST
Repository for [AIROBEST project](https://sensillence.github.io/AIROBEST/)

## Data set
The TAIGA dataset is available at https://doi.org/10.23729/fe7ce882-8125-44e7-b0cf-ae652d7ed0d5

Download all the files in your local directory /any/path/TAIGA .

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

#### Data split
- Access `./training/custom_split_data` directory
- Run `python ./create_stand_ids.py` to create stand_ids map `stands_ids.npy` under `./data` directory
- Run `sbatch ./scustom_split_data.sh` to generate `mask.npy` and create a custom data split.
- Run `python ./eelis_split_data.py` to generate a custom data split based on Eelis's stands in `eelis_forest_stand_data.csv`

#### Training
- Go back to `training` directory
- If you want to see real-time visualization, you must start `visdom` server first.

**How to keep visom server running while waiting for submitted batch job to start?**
- Log in to Puhti using local forwarding: `ssh -L 8097:127.0.0.1:8097 username@puhti.csc.fi`
- Open tmux session by running: `tmux`
- Start visdom server: `sbatch server.sh`
- Now one can detach the tmux session and end the ssh session without interrupting visdom server
- To attach again to tmux session, ssh to remote server and run `tmux attach`

Training configurations:
- `hyper_data_path`: path to hyperspectral image saved in preprocessing stage.
- `src_norm_multiplier`: path to file contains pixel norm (corresponding to `hyperspectral_src_l2norm_along_channel.pt` file in preprocessing stage)
- `tgt_path`: path to target label

For other options, run: `python train.py -h` for detailed explanation.

Visualization can be seen by opening [http://localhost:8097/](http://localhost:8097/) from web browser. You first need to login on Puhti with local forwarding: `ssh -L 8097:127.0.0.1:8097 username@puhti.csc.fi`
- Run `sbatch strain.sh` to train on Puhti server.





#### Analysis
To be updated
