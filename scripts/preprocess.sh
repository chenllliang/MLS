lan=$2
lan2=$3
dir=$1



echo "start tokenization"

perl ../Tools/mosesdecoder/scripts/tokenizer/tokenizer.perl \
-threads 20 -l $lan < $dir"dev."$lan > $dir"dev.tok."$lan 

perl ../Tools/mosesdecoder/scripts/tokenizer/tokenizer.perl \
-threads 20 -l $lan < $dir"train."$lan > $dir"train.tok."$lan

perl ../Tools/mosesdecoder/scripts/tokenizer/tokenizer.perl \
-threads 20 -l $lan < $dir"test."$lan > $dir"test.tok."$lan

perl ../Tools/mosesdecoder/scripts/tokenizer/tokenizer.perl \
-threads 20 -l $lan2 < $dir"dev."$lan2 > $dir"dev.tok."$lan2

perl ../Tools/mosesdecoder/scripts/tokenizer/tokenizer.perl \
-threads 20 -l $lan2 < $dir"train."$lan2 > $dir"train.tok."$lan2

perl ../Tools/mosesdecoder/scripts/tokenizer/tokenizer.perl \
-threads 20 -l $lan2 < $dir"test."$lan2 > $dir"test.tok."$lan2

echo "bpe language "$lan
python ../Tools/subword-nmt/learn_joint_bpe_and_vocab.py \
-s 16000 -i $dir"train.tok."$lan -o $dir"bpe."$lan --write-vocabulary $dir"voc."$lan

echo "bpe language "$lan2
python ../Tools/subword-nmt/learn_joint_bpe_and_vocab.py \
-s 16000 -i $dir"train.tok."$lan2 -o $dir"bpe."$lan2 --write-vocabulary $dir"voc."$lan2

echo "applying bpe"

python ../Tools/subword-nmt/apply_bpe.py \
-c $dir"bpe."$lan < $dir"dev.tok."$lan > $dir"dev.tok.bpe."$lan

python ../Tools/subword-nmt/apply_bpe.py \
-c $dir"bpe."$lan < $dir"train.tok."$lan > $dir"train.tok.bpe."$lan

python ../Tools/subword-nmt/apply_bpe.py \
-c $dir"bpe."$lan < $dir"test.tok."$lan > $dir"test.tok.bpe."$lan

python ../Tools/subword-nmt/apply_bpe.py \
-c $dir"bpe."$lan2 < $dir"dev.tok."$lan2 > $dir"dev.tok.bpe."$lan2

python ../Tools/subword-nmt/apply_bpe.py \
-c $dir"bpe."$lan2 < $dir"train.tok."$lan2 > $dir"train.tok.bpe."$lan2

python ../Tools/subword-nmt/apply_bpe.py \
-c $dir"bpe."$lan2 < $dir"test.tok."$lan2 > $dir"test.tok.bpe."$lan2

# echo "finish preprocessing"





fairseq-preprocess --source-lang $lan2 --target-lang $lan \
    --trainpref $dir"train.tok.bpe" --validpref $dir"dev.tok.bpe" --testpref $dir"test.tok.bpe" \
    --destdir ../databin/iwslt14-de-en-isolated-new \
    --thresholdsrc 10 --thresholdtgt 10 \
    --workers 20 \


fairseq-preprocess --source-lang $lan2 --target-lang $lan \
    --trainpref $dir"train.tok.bpe" --validpref $dir"dev.tok.bpe" --testpref $dir"test.tok.bpe" \
    --destdir ../databin/iwslt14-de-en-joined-new \
    --thresholdsrc 10 --thresholdtgt 10 \
    --workers 20 \
    --joined-dictionary \