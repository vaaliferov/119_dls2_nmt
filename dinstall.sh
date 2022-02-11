#!/usr/bin/env bash

mkdir data
source config.sh

apt update && apt install -y python3-pip
pip3 install --upgrade pip cython wheel gdown
pip3 install --no-cache-dir -r req_bot.txt

gdown --id $MODEL_FILE_ID -O data/model.pt
gdown --id $SRC_TOK_FILE_ID -O data/ru_bpe.yttm
gdown --id $TRG_TOK_FILE_ID -O data/en_bpe.yttm