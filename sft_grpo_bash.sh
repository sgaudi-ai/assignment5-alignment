#!/bin/bash

set -e  # Exit if any command fails

cd /home/ec2-user/assignment5-alignment

source /home/ec2-user/assignment5-alignment/.venv/bin/activate

echo "Starting SFT: $(date)"
python train_sft.py 

echo "Starting GRPO: $(date)"
python train_grpo_after_sft.py

echo "Finished: $(date)"