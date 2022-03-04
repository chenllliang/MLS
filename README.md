# Focus on the Target’s Vocabulary: Masked Label Smoothing for Machine Translation
Hi, this is the source code of our paper "Focus on the Target’s Vocabulary: Masked Label Smoothing for Machine Translation" accepted by ACL 2022. You can find the paper in the root directory (uploading to arxiv soon).

For latest update, please refer to this [repo](https://github.com/chenllliang/MLS).

## Introduction


Label smoothing and vocabulary sharing are two widely used techniques in neural machine translation models. However, we argue that simply applying both techniques can be conflicting and even leads to sub-optimal performance. When allocating smoothed probability, original label smoothing treats the **source-side words** that would never appear in the target language equally to the real **target-side words**, which could bias the translation model. To address this issue, we propose **Masked Label Smoothing (MLS)**, a new mechanism that masks the soft label probability of source-side words to zero. Simple yet effective, MLS manages to better integrate label smoothing with vocabulary sharing. 


<br>

<div align=center>
<img width="600" src="./venn.png"/>
  
 Venn graph showing the token distribution between lanugages.
</div>




<div align=center>
<img width="600" src="./bars.png"/>
  
  Illustration of Masked Label Smoothing (bottom right) and Weighted Label Smoothing (upper right & bottom left)

</div>




## Preparations

```bash
git clone git@github.com:chenllliang/MLS.git
cd MLS

conda create -n MLS python=3.7
conda activate MLS

cd fairseq # We place the MLS criterions inside fairseq's criterion sub-folder, you can find them there.
pip install --editable ./
pip install sacremoses

pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
# Make sure you have the right version of pytorch and CUDA, we use torch 1.9.0+cu111
```

We adopt [mosesdecoder](https://github.com/moses-smt/mosesdecoder) for tokenization , [subword-nmt](https://github.com/rsennrich/subword-nmt) for BPE and [fairseq](https://github.com/pytorch/fairseq) for experiment pipelines.


## Preprocess

We have prepared a pre-processed binary data of WMT16 RO-EN on Google and BaiduDisk (coming soon, download it and unzip in ../databin/ folder, you can jump to next section then) .


If you plan to try your own dataset. You may refer to this [script](https://github.com/chenllliang/MLS/blob/main/scripts/preprocess.sh) for preprocessing and parameter setting.

```bash
cd script
bash preprocess.sh ../data/dataset-src-tgt/ src tgt
```

if it works succeefully, two folders containing binary files will be saved in the databin folder.



## Train with MLS and original LS 


```bash
cd scripts
bash train_LS.sh # end up in 50 epoches with valid_best_bleu = 22.38

bash train_MLS.sh # end up in 50 epoches with valid_best_bleu = 22.72
```

The best valid checkpoint will be saved in checkpoints folder for testing.


## Get result on Test Set 

```bash
cd scripts

bash generate.sh ../databin/wmt16-ro-en-joined ../checkpoints/ro-en-ori-0.1 ../Output/ro-en-ori-ls.out # should get BLEU4 = 22.54


bash generate.sh ../databin/wmt16-ro-en-joined ../checkpoints/ro-en-MLS-0.1 ../Output/ro-en-MLS-ls.out # should get BLEU4 = 22.89
```
We have uploaded the generated texts in the Output folder, which you can also refer to.

## Using Weighted Label Smoothing

You can change the lp_beta,lp_gamma,lp_eps in `train_WLS.sh` to control the weights distribution.

```bash
cd scripts

bash train_WLS.sh  # you should change the path to the source,target and joined vocabulary individually

```

The test procedure follows previous section.

## Citation
If you feel our work helpful, please kindly cite

```bib
@inproceedings{chen2022focus,
   title={Focus on the Target’s Vocabulary: Masked Label Smoothing for Machine Translation},
   author={Chen, Liang and Xu, Runxin and Chang, Baobao},
   booktitle={The 60th Annual Meeting of the Association for Computational Linguistics},
   year={2022}
}
```
