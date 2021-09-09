# AIROBEST
Repository for AIROBEST project's PyTorch based DCNN code

### Getting started
Make sure to install all the requirements, run: `module load pytorch/1.3.1 ; pip install -r requirements.txt --user`

The following instruction assumes your current directory is `$PATH_TO_AIROBEST_CODE/source/cnn`
#### Data preprocessing
- Specify paths to hyperspectral data and forest data in `spreprocess.sh` script, for more options, run: `python preprocess.py -h`
- Run `sbatch spreprocess.sh` to submit preprocessing job on Puhti
- This process will generate the following files and save them under `./data` directory:
    * `hyper_image.pt`: hyperspectral image saved as pytorch Tensor. We use this instead of spectral format to increase GPU utilization.
    * `hyperspectral_src_l2norm_along_channel.pt`: tensor contains the corresponding norm of each pixel from hyperspectral image, norms are computed along color channel.
    * `hyperspectral_tgt_normalized.pt`: tensor of target labels, labels for regression tasks has been normalized to [0, 1] scale.
    * `metadata.pt`: dictionary contains information about the input data, such as class values for each classification task
    
#### Training
If you want to see real-time visualization, you must start `visdom` server first.

**How to keep visom server running while waiting for submitted batch job to start?**
- Log in to Puhti using local forwarding: `ssh -L 8097:127.0.0.1:8097 username@puhti.csc.fi`
- Open tmux session by running: `tmux`
- cd to project folder: `cd $PATH_TO_AIROBEST_CODE/source/cnn`
- Start visdom server: `bash server.sh`
- Now one can detach the tmux session and end the ssh session without interrupting visdom server
- To attach again to tmux session, ssh to remote server and run `tmux attach`

Training configurations:
- `hyper_data_path`: path to hyperspectral image saved in preprocessing stage.
- `src_norm_multiplier`: path to file contains pixel norm (corresponding to `hyperspectral_src_l2norm_along_channel.pt` file in preprocessing stage)
- `tgt_path`: path to target label

For other options, run: `python train.py -h` for detailed explanation.

Visualization can be seen by opening [http://localhost:8097/](http://localhost:8097/) from web browser. You first need to login on Puhti with local forwarding: `ssh -L 8097:127.0.0.1:8097 username@puhti.csc.fi`
