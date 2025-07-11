### Code for the paper [On the Generalization of Representation Uncertainty in Earth Observation](https://arxiv.org/abs/2503.07082) (ICCV 2025)


## 📖 Table of Contents
- [About](#about)
- [Installation](#installation)
- [Usage](#usage)
- [Pretrained Uncertainty Checkpoints](#pretrained-uncertainty-checkpoints)
- [Citation](#citation)

## About

This is the repository for the paper [On the Generalization of Representation Uncertainty in Earth Observation](https://arxiv.org/abs/2503.07082). It builds on and extends [Pretrained Visual Uncertainties](https://arxiv.org/abs/2402.16569), [GitHub repo](https://github.com/mkirchhof/url). 

The documentation of this repo will be continuously updated.

## Installation

 `conda create env -f environment.yml`

## Usage
`main.py` is the main driver of the project, aggregating the options selected in configs, depending on the task: i.e
      - Train uncertainties
      - Inference
      - Save features (caching)

To initiate an experiment execute `python main.py`.

The training/inference dataset can be selected by modifying the `configs/configs.json` `"dataset"` field.

Training hyperparameters can be modified in `configs/train/train_configs.json` and inference options in `configs/inference/inference_configs.json`.

Overal the configuration structure is defined as:
```plaintext
configs
├── configs.json
├── data
│   ├── data_configs.json
│   └── webdataset_configs.json
├── inference
│   └── inference_configs.json
├── stats
│   └── stats.json
└── train
    └── train_configs.json
```

## Pretrained uncertainty checkpoints

| Dataset     | Bands |ViT-Tiny | ViT-Small | ViT-Base | ViT-Large |
|------------|------|---------------------|----------------------|---------------------|----------------------|
| BigEarthNet | RGB | [Download](https://www.dropbox.com/scl/fi/c6j6wa8po22eutyv2w0cd/vit_tiny.pth?rlkey=94u4xcnv1xme2ns93xe7fqor3&st=20uednd6&dl=0)       | [Download](https://www.dropbox.com/scl/fi/c1wswangof9ruk1ixnj74/vit_small.pth?rlkey=uql5vufnv1bmhk54l2n2lh9i7&st=10h4f3ru&dl=0)       | [Download](https://www.dropbox.com/scl/fi/1636ox2ns09c3om2hjfft/vit_base.pth?rlkey=luhgpsgcvzdb5jaspoa40kcjg&st=6dqocezu&dl=0)       | [Download](https://www.dropbox.com/scl/fi/dd2dfknz0ak023ylo99ge/vit_large.pth?rlkey=6pjh3xy1lktlp184y7p8k9zc6&st=zq1cs1po&dl=0)       |
| BigEarthNet5 | RGB |[Download](https://www.dropbox.com/scl/fi/5ac5qxh0d9h36nsxezr4k/vit_tiny.pth?rlkey=owpr8lkwcrjle8be7kqqymiv9&st=r6r433w5&dl=0)       | [Download](https://www.dropbox.com/scl/fi/lbnn7foxsf6zlk3ul11y9/vit_small.pth?rlkey=zev1mf17dxxdbuy8g27vyjnch&st=orxeg4js&dl=0)       | [Download](https://www.dropbox.com/scl/fi/twb9oha66i5y7jkf7e192/vit_base.pth?rlkey=01obs6o1oleb48e57y03jzxgh&st=2ucuo7ao&dl=0)       | [Download](https://www.dropbox.com/scl/fi/3mh39ym276y0t13y4o826/vit_large.pth?rlkey=w9eez98we38qra7tmtg3402vv&st=g4tw4zy3&dl=0)       |
| BigEarthNet-SAR | SAR | [Download](https://www.dropbox.com/scl/fi/7uc6w9mw7visn6zv3tles/vit_tiny.pth?rlkey=c45euew2uru67kitcue34e6q5&st=ojwl5kka&dl=0)       | [Download](https://www.dropbox.com/scl/fi/92efjiof2czh0c2zx0pps/vit_small.pth?rlkey=5pd0xn2uug6m1j7acs982ouk1&st=yeqs0d2a&dl=0)       | [Download](https://www.dropbox.com/scl/fi/hgnsb0q8yle2ajpt05jay/vit_base.pth?rlkey=gs9sx3qlaxnp29cxymc4c9a7d&st=kjlqr60q&dl=0)       | [Download](https://www.dropbox.com/scl/fi/9v2v5c0rll0io0sbolwk2/vit_large.pth?rlkey=qx2t9wnogs0kvxvxxxze29dbb&st=q6ezfud0&dl=0)       |
| BigEarthNet-MS | Multispectral |[Download](https://www.dropbox.com/scl/fi/kzasvy17okm5at9nahbw3/vit_tiny.pth?rlkey=mwhdcyzbmlqs0k6yi70u4yjov&st=ga64zvau&dl=0)       | [Download](https://www.dropbox.com/scl/fi/mvb2is3j6n1nrk7wt0xtz/vit_small.pth?rlkey=p1vuybfjertz9tqj39na7day5&st=lah70e45&dl=0)       | [Download](https://www.dropbox.com/scl/fi/pctt0og5ybazbu2fca3qi/vit_base.pth?rlkey=uhi7iui62xowftcrx8huqgesx&st=ka77wbc5&dl=0)       | [Download](https://www.dropbox.com/scl/fi/cxwbknfpeyirbe9k70yjf/vit_large.pth?rlkey=8zv1sfv10hnixsmftpe4jd5zr&st=1u2fooxk&dl=0)       |
| Flair      | RGB |[Download](https://www.dropbox.com/scl/fi/pfzt2etyguns0u2z0bsyc/vit_tiny.pth?rlkey=hyp3g6w91vsm6i8uhhdc06thi&st=1nbhge22&dl=0)       | [Download](https://www.dropbox.com/scl/fi/mggofntujre6h1p1xewoz/vit_small.pth?rlkey=a5uqcxz2yktuvkz0soyu128kg&st=cp6k6vfo&dl=0)       | [Download](https://www.dropbox.com/scl/fi/mt9ht2jqbqmyahp417sbg/vit_base.pth?rlkey=ag8ml7siefnbikrucws4xzfe7&st=fnu61bv9&dl=0)       | [Download](https://www.dropbox.com/scl/fi/59r7wuh6s5bqth1oykno1/vit_large.pth?rlkey=cp3vi6elc2v4oahcc0ukkrobb&st=casw4cpz&dl=0)       |

ImageNet pretrained models can be found in this [repo](https://github.com/mkirchhof/url).

### Download example
```
wget -O bigearthnet_vit_tiny.pth "https://www.dropbox.com/scl/fi/c6j6wa8po22eutyv2w0cd/vit_tiny.pth?rlkey=94u4xcnv1xme2ns93xe7fqor3&st=20uednd6&dl=0"
```


## Citation

If you use our work, please cite:
```
@misc{kondylatos2025generalizationrepresentationuncertaintyearth,
      title={On the Generalization of Representation Uncertainty in Earth Observation}, 
      author={Spyros Kondylatos and Nikolaos Ioannis Bountos and Dimitrios Michail and Xiao Xiang Zhu and Gustau Camps-Valls and Ioannis Papoutsis},
      year={2025},
      eprint={2503.07082},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.07082}, 
}
```
