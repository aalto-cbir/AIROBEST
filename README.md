# AIROBEST
Repository for AIROBEST project

#### How to keep visom server running while waiting for submitted batch job to start?
- Log in to taito using local forwarding: `ssh -L 8097:127.0.0.1:8097 username@taito-gpu.csc.fi`
- Open tmux session by running: `tmux`
- cd to project folder: `cd $path_to_project_folder/source/cnn`
- Start visdom server: `bash server.sh`
- Now one can detach the tmux session and end the ssh session without interrupting visom server
- To attach again to tmux session, ssh to remote server and run `tmux attach`
