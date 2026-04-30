#!/bin/bash
set -e
echo "Step 1: Training"
python train.py --epochs 50

echo "Step 2: Freezing"
python freeze.py

echo "Step 3: Quantisation"
bash quantize.sh

echo "Step 4: Compilation"
bash compile.sh
