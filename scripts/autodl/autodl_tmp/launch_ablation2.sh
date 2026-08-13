#!/bin/bash
cd /root/CytoBridge/code
exec /root/venvs/cytobridge-scgpt-py310/bin/python -u train.py   loss.lam_contrast=0.0   run_name=no_contrast   ckpt.dirpath=ckpts/no_contrast   wandb.use=false   2>&1 | tee /root/autodl-tmp/ablation_no_contrast.log
