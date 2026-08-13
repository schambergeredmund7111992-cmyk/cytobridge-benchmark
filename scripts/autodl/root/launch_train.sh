#!/bin/bash
cd /root/CytoBridge/code
CUDA_VISIBLE_DEVICES=0 nohup /root/venvs/cytobridge-scgpt-py310/bin/python -u train.py   --config-name=v1   run_name=v1_sanity   ckpt.dirpath=ckpts/v1_sanity   trainer.max_epochs=1   wandb.use=false   > /root/autodl-tmp/train_v1_sanity_log.txt 2>&1 &
echo "PID=$!"
