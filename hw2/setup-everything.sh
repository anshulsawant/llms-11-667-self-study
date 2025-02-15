bash setup-conda.sh
source ~/.bashrc
conda create -n cmu-llms-hw2 python=3.11
conda activate cmu-llms-hw2
pip install -r requirements.txt
pip install -e .
curl https://huggingface.co/datasets/yimingzhang/llms-hw2/resolve/main/tokens.npz -o data/tokens.npz -L
export PYTHONPATH=$(pwd)/src/$PYTHONPATH
wandb login
