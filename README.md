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

In this work, we try to design a loss function inspired by Triplet Loss such that the network generates feature vectors that are similar for the same label. When making a prediction, we extract the features of an image and compare them against "exemplar" feature vectors for the available labels. If the similarity between the features and the "exemplar" feature vectors are within a certain threshold, we consider it a positive prediction.

## References
1. Dangwei Li, Xiaotang Chen, Kaiqi Huang. _Multi-attribute Learning for Pedestrian Attribute Recognition in Surveillance Scenarios_ ACPR'2015 [PDF@arXiv](http://dangweili.github.io/misc/pdfs/acpr15-att.pdf) [repo@Github](https://github.com/dangweili/pedestrian-attribute-recognition-pytorch)
2. Yubin Deng, Ping Luo, Chen Change Loy, Xiaoou Tang. _Pedestrian Attribute Recognition At Far Distance_ MM'14 http://mmlab.ie.cuhk.edu.hk/projects/PETA.html [PDF@MMlab](http://mmlab.ie.cuhk.edu.hk/projects/PETA_files/Pedestrian%20Attribute%20Recognition%20At%20Far%20Distance.pdf)
3. Piotr Szymański, Tomasz Kajdanowicz, Nitesh Chawla. _LNEMLC: Label Network Embeddings for Multi-Label Classification_ ArXiv'2018 [PDF@arXiv](https://arxiv.org/pdf/1812.02956.pdf) 
4. Chih-Kuan Yeh, Wei-Chieh Wu, Wei-Jen Ko, Yu-Chiang Frank Wang. _Learning Deep Latent Spaces for Multi-Label Classification_ AAAI'17 https://dl.acm.org/doi/10.5555/3298483.3298647 [PDF@arXiv](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/viewFile/14166/14487) [repo@Github](https://github.com/chihkuanyeh/C2AE)
5. Kuan-Hao Huang, Hsuan-Tien Lin. _Cost-sensitive label embedding for multi-label classification_. Mach Learn 106, 1725–1746 (2017). https://doi.org/10.1007/s10994-017-5659-z [PDF@Springer](https://link.springer.com/content/pdf/10.1007/s10994-017-5659-z.pdf) [PDF@arXiv](https://arxiv.org/pdf/1603.09048.pdf) [repo@Github](https://github.com/ej0cl6/csmlc)

