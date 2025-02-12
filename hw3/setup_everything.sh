bash setup-conda.sh
source ~/.bashrc
conda create -n cmu-llms-hw3 python=3.11
conda activate cmu-llms-hw3
pip install --no-input -r requirements.txt
conda install --yes -c pytorch -c nvidia faiss-gpu=1.8.0
pip install --no-input ninja
pip install --no-input flash-attn --no-build-isolation
pip install --no-input wandb
pip install -e .
