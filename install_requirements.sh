#!/bin/bash

curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt-get install -y --allow-unauthenticated git-lfs

git lfs install
git clone https://huggingface.co/cointegrated/rut5-base-multitask

pip3 install transformers 
pip3 install sentencepiece
pip3 install torch
pip3 install datasets
pip3 install pynvml
pip3 install evaluate
pip3 install bert_score
pip3 install peft
pip3 install accelerate