#!/bin/bash
cd /root/CytoBridge
export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
export PYTHONPATH=/root/CytoBridge/code
nohup /root/venvs/cytobridge-scgpt-py310/bin/python /root/run_molformer_cache.py   /root/CytoBridge/data/processed/sciplex/drugs_canonical.csv   /root/CytoBridge/data/cache/sciplex_molformer_emb.npz   > /root/autodl-tmp/molformer_embed_log.txt 2>&1 &
echo "PID=$!"
