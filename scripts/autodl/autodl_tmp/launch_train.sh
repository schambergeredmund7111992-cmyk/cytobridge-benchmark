#!/bin/bash
cd /root/CytoBridge/code
exec /root/venvs/cytobridge-scgpt-py310/bin/python -u train.py   run_name=v1_full   ckpt.dirpath=ckpts/v1   wandb.use=false   2>&1 | tee /root/autodl-tmp/train_v1_full.log
