export CUDA_VISIBLE_DEVICES=0
fairseq-train ../databin/iwslt14-de-en-joined-new \
--save-dir ../checkpoints/de-en-wls-442 \
--arch transformer_wmt_en_de \
--optimizer adam --adam-betas '(0.9, 0.98)' \
--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 1000 \
--lr 0.0007 \
--eval-bleu \
--eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
--eval-bleu-detok moses \
--eval-bleu-remove-bpe \
--best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
--max-tokens  2048  \
--update-freq 4 \
--no-progress-bar --log-format json --log-interval 500 \
--max-epoch 20 \
--keep-last-epochs 3 \
--fp16 \
--seed 666 \
--joint-vocab-dir ../databin/iwslt14-de-en-joined-new/dict.de.txt \
--tgt-vocab-dir ../databin/iwslt14-de-en-isolated-new/dict.en.txt \
--src-vocab-dir ../databin/iwslt14-de-en-isolated-new/dict.de.txt \
--criterion weighted_label_smoothed_cross_entropy \
--lp-beta 0.4 --lp-gamma 0.4 --lp-eps 0.2 \
--label-smoothing 0.1 --dropout 0.3 --weight-decay 0.0001 \

