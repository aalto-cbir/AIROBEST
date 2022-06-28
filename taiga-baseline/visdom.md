**How to keep visdom server running while waiting for submitted batch job to start?**
- Log in to Puhti using local forwarding: `ssh -L 8097:127.0.0.1:8097 username@puhti.csc.fi`
- Open tmux session by running: `tmux`
- Start visdom server: `sbatch server.sh`
- Now one can detach the tmux session and end the ssh session without interrupting visdom server
- To attach again to tmux session, ssh to remote server and run `tmux attach`

Visualization can be seen by opening [http://localhost:8097/](http://localhost:8097/) from web browser. You first need to login on Puhti with local forwarding: `ssh -L 8097:127.0.0.1:8097 username@puhti.csc.fi`
