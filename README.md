# Focus on the Target’s Vocabulary: Masked Label Smoothing for Machine Translation
News 🚩
- Release preprocessed data and model output. 2022.03.05
- Code released at Github. 2022.03.04
- Accepted by ACL 2022 Main Conference. 2022.02.24

Hi, this is the source code of our paper "Focus on the Target’s Vocabulary: Masked Label Smoothing for Machine Translation" accepted by ACL 2022. You can find the paper in the root directory (uploading to arxiv soon).

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


# Make sure you have the right version of pytorch and CUDA, we use torch 1.9.0+cu111
```

We adopt [mosesdecoder](https://github.com/moses-smt/mosesdecoder) for tokenization , [subword-nmt](https://github.com/rsennrich/subword-nmt) for BPE and [fairseq](https://github.com/pytorch/fairseq) for experiment pipelines. **You need to clone the first two repos into `./Tools` before next step.**




## Preprocess

We have prepared a pre-processed binary data of IWSLT14 DE-EN in the ../databin folder (unzip it and put the two unzipped folders under ../databin/, you can jump to next section then) .


If you plan to try your own dataset. You may refer to this [script](https://github.com/chenllliang/MLS/blob/main/scripts/preprocess.sh) for preprocessing and parameter setting.

```bash
cd script
bash preprocess.sh ../data/dataset-src-tgt/ src tgt
```

if it works succeefully, two folders containing binary files will be saved in the databin folder.



## Train with MLS and original LS 


```bash
cd scripts
bash train_LS.sh # end up in 20 epoches with valid_best_bleu = 36.91

bash train_MLS.sh # end up in 20 epoches with valid_best_bleu = 37.16
```

The best valid checkpoint will be saved in checkpoints folder for testing.


## Get result on Test Set 

```bash
cd scripts

bash generate.sh ../databin/iwslt14-de-en-joined-new ../checkpoints/de-en-LS-0.1 ../Output/de-en-ls-0.1.out # get BLEU4 = 35.20


bash generate.sh ../databin/iwslt14-de-en-joined-new ../checkpoints/de-en-MLS-0.1 ../Output/de-en-mls-0.1.out # get BLEU4 = 35.76
```
We have uploaded the generated texts in the Output folder, which you can also refer to.

## Some results on single GPU

| BLEU  |  IWSLT14 DE-EN  | WMT16 RO-EN  |
|  ----  | ----  |----  |
| LS  | dev: 36.91 test: 35.20 |dev: 22.38 test: 22.54 |
| MLS(Ours)  | dev: **37.16** test: **35.76** |dev: **22.72** test: **22.89**  |



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
