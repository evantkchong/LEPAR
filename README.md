# LEPAR
Label Embeddings for Pedestrian Attribute Recognition as a Multi-Label Classification Task

## Introduction
For the Person Attribute Recognition problem, one method is [DeepMAR](http://dangweili.github.io/misc/pdfs/acpr15-att.pdf) which predicts multiple attributes for a given pedestrian image. 

DeepMAR is essentially a classification model (the feature extractor used isn't specified in the paper but in their code they use a resnet) trained with a Binary Cross-Entropy loss with labels that are multi-hot encoded. Despite this simple approach, DeepMAR produces results that remain competitive against other more complicated approaches.

|         | mean Acc | Accuracy | Precision | Recall | F1    |
|---------|----------|----------|-----------|--------|-------|
| DeepMAR | 82.89    | 75.07    | 83.68     | 83.14  | 83.41 |
| [VAA](https://arxiv.org/pdf/1807.03903.pdf) | 84.59    | 78.56    | 86.79     | 86.12  | 86.46 |
| [GRL](https://www.ijcai.org/Proceedings/2018/0441.pdf) | 86.70    | 84.34    | 88.82     | 86.51  | 86.51 |
| [RPAR](https://arxiv.org/abs/2005.11909) | 85.11    | 79.14    | 86.99     | 86.33  | 86.39 |

However, one caveat of DeepMAR is that if we want to add additional classes to the classification system, we need to fine-tune the model by adding new labels to the dataset and re-training it.

In this work, we use an [adapted Triplet Loss for multi-label classification](https://github.com/abarthakur/multilabel-deep-metric) and apply it to the Pedestrian Attribute Recognition task. As in the standard Triplet Loss, the network embeds features of the same class while maximizing the distance between embeddings of different classes, but this version works for multi-label classification.

## Multi-Label Triplet Loss
The Triplet Loss implemnentation is borrowed from https://github.com/omoindrot/tensorflow-triplet-loss/blob/master/model/triplet_loss.py while applying ideas from https://github.com/abarthakur/multilabel-deep-metric/blob/master/src/utils.py.

The main change in the Multi-Label Triplet Loss is as follows:
When using multi-hot encoded labels, labels can be entirely correct, somewhat correct, or totally wrong.
In order to determine whether a triplet is positive or negative, we consider the number of matches in between multi-hot encoded labels which can be given by the dot product between two labels (we transpose one of them so that we get a scalar output).

In the original function, A triplet `(i, j, k)` is valid if:
* `i`, `j`, `k` are distinct
* `labels[i] == labels[j]` and `labels[i] != labels[k]`

To accomodate our new definition of positive and negative triplets, we add on the following constraints
* `distance(a, p) + margin > distance(a, n)` where `distance(x, y)` indicates the distance between embeddings for `x` and `y`
* `num_matches(a, p) < num_matches(a, n)` where `num_matches(x, y)` indicates the number of matching classes between the labels for `x` and `y`

## Running this repo
You are advised to use this repository in conjuction with a virtualenv with the project
requirements installed.

Prior to training the model, you may download the PETA dataset using the `_download_peta_dataset.sh` script.

After that, run `python3 train_lepar.py` to train the LEPAR model.

## Requirements
Developed with Python 3.9.5 on [Ubuntu(Windows 10 WSL2)](https://ubuntu.com/blog/ubuntu-on-wsl-2-is-generally-available) with the following package versions:
```
tensorflow==2.5.0
tensorflow-addons==0.13.0
tqdm==4.61.0
black==21.5b2
```

## References
1. Dangwei Li, Xiaotang Chen, Kaiqi Huang. _Multi-attribute Learning for Pedestrian Attribute Recognition in Surveillance Scenarios_ ACPR'2015 [PDF@arXiv](http://dangweili.github.io/misc/pdfs/acpr15-att.pdf) [repo@Github](https://github.com/dangweili/pedestrian-attribute-recognition-pytorch)
2. Aneesh Barthakur. _multilabel-deep-metric_ Github'2020 [repo@Github](https://github.com/abarthakur/multilabel-deep-metric)
3. Yubin Deng, Ping Luo, Chen Change Loy, Xiaoou Tang. _Pedestrian Attribute Recognition At Far Distance_ MM'14 http://mmlab.ie.cuhk.edu.hk/projects/PETA.html [PDF@MMlab](http://mmlab.ie.cuhk.edu.hk/projects/PETA_files/Pedestrian%20Attribute%20Recognition%20At%20Far%20Distance.pdf)
4. Piotr Szymański, Tomasz Kajdanowicz, Nitesh Chawla. _LNEMLC: Label Network Embeddings for Multi-Label Classification_ ArXiv'2018 [PDF@arXiv](https://arxiv.org/pdf/1812.02956.pdf) 
5. Chih-Kuan Yeh, Wei-Chieh Wu, Wei-Jen Ko, Yu-Chiang Frank Wang. _Learning Deep Latent Spaces for Multi-Label Classification_ AAAI'17 https://dl.acm.org/doi/10.5555/3298483.3298647 [PDF@arXiv](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/viewFile/14166/14487) [repo@Github](https://github.com/chihkuanyeh/C2AE)
6. Kuan-Hao Huang, Hsuan-Tien Lin. _Cost-sensitive label embedding for multi-label classification_. Mach Learn 106, 1725–1746 (2017). https://doi.org/10.1007/s10994-017-5659-z [PDF@Springer](https://link.springer.com/content/pdf/10.1007/s10994-017-5659-z.pdf) [PDF@arXiv](https://arxiv.org/pdf/1603.09048.pdf) [repo@Github](https://github.com/ej0cl6/csmlc)

