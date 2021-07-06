### A Simple Codebase for Image-based Person Re-identification

#### Requirements: Python 3.6, Pytorch 1.6.0, yacs

#### Supported losses
##### Classification Losses
- [x] CrossEntropy Loss
- [x] CrossEntropy Loss with Label Smooth
- [x] CosFace Loss
- [x] ArcFace Loss
- [x] Circle Loss
##### Pairwise Losses
- [x] Triplet Loss
- [x] Contrastive Loss
- [x] Pairwise CosFace Loss
- [x] Pairwise Circle Loss

#### Supported models
- [x] ResNet-50
- [ ] ResNet-50-IBN
- [ ] IANet


#### Get Started
- Replace `_C.DATA.ROOT` and `_C.OUTPUT` in `configs/default.py` with your own `data path` and `output path`, respectively.
- Run `train.sh`

#### Some Results

##### Market-1501

| classification loss | pairwise loss |backbone |top-1 | mAP|
|:---|:---|:---:|:---:|:---:| 
| CrossEntropy    | Triplet       | ResNet-50 | 94.5 | 86.6 |
| CrossEntropy    | Contrastive | ResNet-50 | 94.3 | 86.4 |
| CrossEntropy    | Cosface      | ResNet-50 | 94.3 | 86.2 |
| CELabelSmooth | Triplet       | ResNet-50 | 95.0 | 87.4 |
| CELabelSmooth | Contrastive | ResNet-50 | 94.5 | 87.1 |
| CELabelSmooth | Cosface     | ResNet-50 | 94.1 | 86.4 |
| Cosface           | Triplet       | ResNet-50 | 95.1 | 86.7 |
| Cosface           | Cosface     | ResNet-50 | 94.5 | 87.1 |
| Arcface           | Triplet       | ResNet-50 | 94.2 | 86.3 |
| Circle              | Circle        | ResNet-50 | 94.7 | 87.3 |

##### MSMT

| classification loss | pairwise loss |backbone |top-1 | mAP|
|:---|:---|:---:|:---:|:---:| 
| CrossEntropy    | Triplet       | ResNet-50 | 78.9 | 57.0 |
| CrossEntropy    | Contrastive | ResNet-50 | 79.3 | 56.7 |
| CrossEntropy    | Cosface      | ResNet-50 | 78.2 | 55.2 |
| CELabelSmooth | Triplet       | ResNet-50 | 79.9 | 58.0 |
| CELabelSmooth | Contrastive | ResNet-50 | 80.3 | 58.7 |
| CELabelSmooth | Cosface     | ResNet-50 | 79.2 | 56.6 |
| Cosface           | Triplet       | ResNet-50 | 78.1 | 54.1 |
| Cosface           | Cosface     | ResNet-50 | 78.8 | 55.9 |
| Arcface           | Triplet       | ResNet-50 | 78.2 | 54.2 |
| Circle              | Circle        | ResNet-50 | 79.7 | 57.0 |

#### Citation

If you use our code in your research or wish to refer to the baseline results, please use the following BibTeX entry.

    @InProceedings{CVPR2019IANet
    author = {Hou, Ruibing and Ma, Bingpeng and Chang, Hong and Gu, Xinqian and Shan, Shiguang and Chen, Xilin},
    title = {Interaction-And-Aggregation Network for Person Re-Identification},
    booktitle = {CVPR},
    year = {2019}
    }
