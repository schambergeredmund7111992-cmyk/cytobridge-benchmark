#!/bin/bash
cd /root/CytoBridge/code
exec /root/venvs/cytobridge-scgpt-py310/bin/python -u train.py   loss.lam_pathway=0.0 loss.lam_kl=0.0   run_name=no_pathway_loss   ckpt.dirpath=ckpts/no_pathway_loss   wandb.use=false   2>&1 | tee /root/autodl-tmp/ablation_no_pathway_loss.log
