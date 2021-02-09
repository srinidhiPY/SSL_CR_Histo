# SSL_CR_Histo
#### by [Chetan Srinidhi](https://srinidhipy.github.io), [Seung Wook Kim](https://seung-kim.github.io/seungkim/), [Fu-Der Chen](https://www.photon.utoronto.ca/people) and [Anne Martel](https://medbio.utoronto.ca/faculty/martel)

* Official repository for [Self-Supervised driven Consistency Training for Annotation Efficient Histopathology Image Analysis](https://arxiv.org/pdf/2102.03897.pdf).

## Abstract
Training a neural network with a large labeled dataset is still a dominant paradigm in computational histopathol- ogy. However, obtaining such exhaustive manual annotations is often expensive, laborious, and prone to inter and intra-observer variability. While recent self-supervised and semi-supervised methods can alleviate this need by learn- ing unsupervised feature representations, they still struggle to generalize well to downstream tasks when the number of labeled instances is small. In this work, we overcome this challenge by leveraging both task-agnostic and task- specific unlabeled data based on two novel strategies: 

* i) a self-supervised pretext task that harnesses the underlying multi-resolution contextual cues in histology whole-slide images to learn a powerful supervisory signal for unsuper- vised representation learning; 

* ii) a new teacher-student semi-supervised consistency paradigm that learns to effectively transfer the pretrained representations to downstream tasks based on prediction consistency with the task-specific un- labeled data.

We carry out extensive validation experiments on three histopathology benchmark datasets across two classifi- cation and one regression based tasks, i.e., tumor metastasis detection, tissue type classification, and tumor cellu- larity quantification. Under limited-label data, the proposed method yields tangible improvements, which is close or even outperforming other state-of-the-art self-supervised and supervised baselines. Furthermore, we empirically show that the idea of bootstrapping the self-supervised pretrained features is an effective way to improve the task- specific semi-supervised learning on standard benchmarks.

## Self-Supervised pretext task

<img src="Fig2_RSP.png" width="600px"/>

## Consistency training

<img src="Fig1_Main.png" width="600px"/>
