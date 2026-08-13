#!/bin/bash
cd /root/CytoBridge/code
exec /root/venvs/cytobridge-scgpt-py310/bin/python -u train.py   data.randomize_drug_emb=true   run_name=no_molformer   ckpt.dirpath=ckpts/no_molformer   wandb.use=false   2>&1 | tee /root/autodl-tmp/ablation_no_molformer.log
