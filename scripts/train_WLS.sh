export CUDA_VISIBLE_DEVICES=0
fairseq-train ../databin/wmt16-ro-en-joined \
--save-dir ../checkpoints/ro-en-wls-442 \
--arch transformer_wmt_en_de \
--optimizer adam --adam-betas '(0.9, 0.98)' \
--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 2000 \
--lr 0.0007 \
--eval-bleu \
--eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
--eval-bleu-detok moses \
--eval-bleu-remove-bpe \
--best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
--max-tokens  1024  \
--update-freq 8 \
--no-progress-bar --log-format json --log-interval 500 \
--max-epoch 25 \
--keep-last-epochs 2 \
--fp16 \
--seed 666 \
--joint-vocab-dir ../databin/wmt16-ro-en-joined/dict.en.txt \
--tgt-vocab-dir ../databin/wmt16-ro-en-isolated/dict.en.txt \
--src-vocab-dir ../databin/wmt16-ro-en-isolated/dict.ro.txt \
--criterion weighted_label_smoothed_cross_entropy \
--lp-beta 0.4 --lp-gamma 0.4 --lp-eps 0.2 \
--label-smoothing 0.1 --dropout 0.3 --weight-decay 0.0001 \

