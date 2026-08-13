#!/bin/bash
cd /root/CytoBridge/code
exec /root/venvs/cytobridge-scgpt-py310/bin/python -u train.py   model.use_pathway_gate=false   run_name=no_pathway_gating   ckpt.dirpath=ckpts/no_pathway_gating   wandb.use=false   2>&1 | tee /root/autodl-tmp/ablation_no_pathway_gating.log
