# Focus on the Target’s Vocabulary: Masked Label Smoothing for Machine Translation
**News** 🚩
- Release result of MLS in new O2M multilingual translation tasks. 2022.03.11
- Release preprocessed data and model output. 2022.03.05
- Code released at Github. 2022.03.04
- Accepted by ACL 2022 Main Conference. 2022.02.24

**Work in Progress** 🚩


Hi, this is the official source code of our paper "Focus on the Target’s Vocabulary: Masked Label Smoothing for Machine Translation" accepted by ACL 2022. You can find the paper in https://arxiv.org/abs/2203.02889.



## Introduction


Label smoothing and vocabulary sharing are two widely used techniques in neural machine translation models. However, we argue that simply applying both techniques can be conflicting and even leads to sub-optimal performance. When allocating smoothed probability, original label smoothing treats the **source-side words** that would never appear in the target language equally to the real **target-side words**, which could bias the translation model. To address this issue, we propose **Masked Label Smoothing (MLS)**, a new mechanism that masks the soft label probability of source-side words to zero. Simple yet effective, MLS manages to better integrate label smoothing with vocabulary sharing. 


<br>

<div align=center>
<img width="600" src="./venn.png"/>
  
 Venn graph showing the token distribution between lanugages.
</div>




<div align=center>
<img width="600" src="./bars.png"/>
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

# Make sure you have the right version of pytorch and CUDA, we use torch 1.10+cu113

```

We adopt [mosesdecoder](https://github.com/moses-smt/mosesdecoder) for tokenization , [subword-nmt](https://github.com/rsennrich/subword-nmt) for BPE and [fairseq](https://github.com/pytorch/fairseq) for experiment pipelines. **You need to clone the first two repos into `./Tools` before next step.**




## Preprocess

**We have prepared** a pre-processed binary data of IWSLT14 DE-EN in the `./databin` folder (unzip it and put the two unzipped folders under ./databin/, you can jump to next section then) .


If you plan to try your own dataset. You may refer to this [script](https://github.com/chenllliang/MLS/blob/main/scripts/preprocess.sh) for preprocessing and parameter setting.

Before running code, you should have your original translation data's structure looks like belows, each line contains one sentence.

```bash
./data/dataset-src-tgt/ # here src,tgt are the language id, you need to change them to your own languages, like en,zh,ro,de ...
-- train.src
-- train.tgt
-- dev.src
-- dev.tgt
-- test.src
-- test.tgt
```
Then,

```bash
cd script
bash preprocess.sh ../data/dataset-src-tgt/ src tgt
```

if it works succeefully, two folders containing binary files and the src/tgt/joint dictionaries will be saved in the databin folder.



## Train with MLS and original LS 


```bash
cd scripts
bash train_LS.sh # end up in 20 epoches with valid_best_bleu = 36.91

bash train_MLS.sh # end up in 20 epoches with valid_best_bleu = 37.16

# train_MLS requires the path of dictionary of src/tgt/joint languages, which are together generated by the preprocess script.
```

The best valid checkpoint will be saved in checkpoints folder for testing.


## Get result on Test Set 

```bash
cd scripts

bash generate.sh ../databin/iwslt14-de-en-joined-new ../checkpoints/de-en-LS-0.1 ../Output/de-en-ls-0.1.out # get BLEU4 = 35.20


bash generate.sh ../databin/iwslt14-de-en-joined-new ../checkpoints/de-en-MLS-0.1 ../Output/de-en-mls-0.1.out # get BLEU4 = 35.76
```
We have uploaded the generated texts in the Output folder, which you can also refer to.

## Some Results 

### Bilingual

| BLEU  |  IWSLT14 DE-EN  | WMT16 RO-EN  |
|  ----  | ----  |----  |
| LS  | dev: 36.91 test: 35.20 |dev: 22.38 test: 22.54 |
| MLS(Ours)  | dev: **37.16** test: **35.76** |dev: **22.72** test: **22.89**  |

### Multilingual (O2M)

| BLEU  |  EN-FR(high)  | EN-DE(mid)  | EN-GU(low) | EN-HI(low) |
|  ----  | ----  | ----  | ----  | ----  |
| No LS | **28.39** | **31.44**| 7.48 | 12.74 |
| LS | 27.93 | 30.9 | 8.01 | 13.22 |
| MLS(Ours)  | 28.16 | 30.77 | **8.23** | **13.70** |



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
@inproceedings{chen-etal-2022-focus,
    title = "Focus on the Target{'}s Vocabulary: Masked Label Smoothing for Machine Translation",
    author = "Chen, Liang  and
      Xu, Runxin  and
      Chang, Baobao",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-short.74",
    pages = "665--671",
    abstract = "Label smoothing and vocabulary sharing are two widely used techniques in neural machine translation models. However, we argue that simply applying both techniques can be conflicting and even leads to sub-optimal performance. When allocating smoothed probability, original label smoothing treats the source-side words that would never appear in the target language equally to the real target-side words, which could bias the translation model. To address this issue, we propose Masked Label Smoothing (MLS), a new mechanism that masks the soft label probability of source-side words to zero. Simple yet effective, MLS manages to better integrate label smoothing with vocabulary sharing. Our extensive experiments show that MLS consistently yields improvement over original label smoothing on different datasets, including bilingual and multilingual translation from both translation quality and model{'}s calibration. Our code is released at https://github.com/PKUnlp-icler/MLS",
}
```
