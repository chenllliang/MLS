#!/bin/bash
DATA_DIR=$1
MODEL_DIR=$2
out_dir=$3
export CUDA_VISIBLE_DEVICES=7
fairseq-generate $DATA_DIR --path $MODEL_DIR/checkpoint_best.pt --beam 4 --remove-bpe --batch-size 400   > $out_dir

bash comp_bleu.sh $out_dir