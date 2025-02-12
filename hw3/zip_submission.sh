#!/bin/bash

rm -rf submission
mkdir submission
cp -r src setup.py requirements.txt submission/
cd submission
zip -qr ../submission.zip . -x src/calculator/__pycache__/**\* src/pytest_utils/__pycache__/**\* src/retriever/__pycache__/**\* src/retriever/logs/**\* src/retriever/wandb/**\* src/retriever/data/**\* src/cmu_llms_hw3.egg-info/**\* src/retriever/driver/__pycache__/**\* src/retriever/modeling/__pycache__/**\*
