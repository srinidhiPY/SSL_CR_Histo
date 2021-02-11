# SSL_CR_Histo
#### by [Chetan Srinidhi](https://srinidhipy.github.io), [Seung Wook Kim](https://seung-kim.github.io/seungkim/), [Fu-Der Chen](https://www.photon.utoronto.ca/people) and [Anne Martel](https://medbio.utoronto.ca/faculty/martel)

* Official repository for [Self-Supervised driven Consistency Training for Annotation Efficient Histopathology Image Analysis](https://arxiv.org/pdf/2102.03897.pdf).

## Abstract
Training a neural network with a large labeled dataset is still a dominant paradigm in computational histopathology. However, obtaining such exhaustive manual annotations is often expensive, laborious, and prone to inter and intra-observer variability. While recent self-supervised and semi-supervised methods can alleviate this need by learning unsupervised feature representations, they still struggle to generalize well to downstream tasks when the number of labeled instances is small. In this work, we overcome this challenge by leveraging both *task-agnostic* and *task-specific* unlabeled data based on two novel strategies: 

* A self-supervised pretext task that harnesses the underlying multi-resolution contextual cues in histology whole-slide images to learn a powerful supervisory signal for unsupervised representation learning.

* A new teacher-student semi-supervised consistency paradigm that learns to effectively transfer the pretrained representations to downstream tasks based on prediction consistency with the task-specific unlabeled data.

We carry out extensive validation experiments on three histopathology benchmark datasets across two classification and one regression based tasks, i.e., *tumor metastasis detection, tissue type classification, and tumor cellularity quantification*. Under limited label data, the proposed method yields tangible improvements, which is close or even outperforming other state-of-the-art self-supervised and supervised baselines. Furthermore, we empirically show that the idea of bootstrapping the self-supervised pretrained features is an effective way to improve the task-specific semi-supervised learning on standard benchmarks.

## Self-Supervised pretext task

<img src="Fig2_RSP.png" width="600px"/>

## Consistency training

<img src="Fig1_Main.png" width="600px"/>

## Table of contents
* [Table of contents](#table-of-contents)
* [Requirements](#requirements)
* [Usage](#usage)

## Requirements 
Core implementation:
* python 3.7+
* pytorch 1.7+
* Openslide-python 1.1+
* Albumentations 1.8+
* Scikit-image 0.15+
* Scikit-learn 0.22+
* Matplotlib 3.2+
* scipy (any version)

Additional packages can be installed via environment.yml / req.txt file.

## Usage

The implementation has three main components:
* the task-agnostic self-supervised pretext task (`Resolution sequence prediction (RSP)`) 
* the task-specific supervised fine-tuning (`SSL`)
* the task-specific consistency training (`SSL_CR`)

### 1. self-supervised pretext task: Resolution sequence prediction (RSP) in WSIs

From the main file (**pretrain_BreastPathQ.py / pretrain_Camelyon16.py**) you can pretrain the network (ResNet18) for predicting the resolution sequence ordering in WSIs on BreastPathQ & Camelyon16 dataset, respectively. This can be easily adopted to any other dataset of choice. The choice of resolution levels for RSP task can also be set in [dataset.py#L277](dataset.py#L277), while pretraining on any other datasets. The argument --train_image_pth is the only required argument and should be set to the directory containing your imgs. There are many more arguments that can be set and these are all explained in the corresponding files. 

```python
python pretrain_BreastPathQ.py    // pretraining on BreastPathQ   
python pretrain_Camelyon16.py    // pretraining on Camelyon16
```
We also provided the pretrained models for BreastPathQ and Camelyon16, which can found in "Pretrained_models" folder.

### 2. Task specific supervised fine-tuning on downstream task

### Citation

If you use significant portions of our code or ideas from our paper in your research, please cite our work:
```
@article{srinidhi2021self,
  title={Self-Supervised driven Consistency Training for Annotation Efficient Histopathology Image Analysis},
  author={Srinidhi, Chetan and Kim, Seung Wook and Chen, Fu-Der and Martel, Anne},
  journal={arXiv preprint arXiv:2102.03897},
  year={2021}
}
```

### Questions or Comments

Please direct any questions or comments to me; I am happy to help in any way I can. You can email me directly at chetan.srinidhi@utoronto.ca.


