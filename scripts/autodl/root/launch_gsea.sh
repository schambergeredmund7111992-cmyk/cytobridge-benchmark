#!/bin/bash
cd /root/CytoBridge/code
nohup /root/venvs/cytobridge-scgpt-py310/bin/python -u data/pathway_gsea.py   --config configs/data/sciplex.yaml   > /root/autodl-tmp/pathway_gsea_log.txt 2>&1 &
echo "PID=$!"
