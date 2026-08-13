#!/bin/bash
cd /root/CytoBridge/code
exec /root/venvs/cytobridge-scgpt-py310/bin/python -u train.py   run_name=v1_full5ep   trainer.max_epochs=5   ckpt.dirpath=ckpts/v1_full   wandb.use=false   2>&1 | tee /root/autodl-tmp/train_v1_full5ep.log
