# RANSynCoders



### Industry-Practical Anomaly Detection and Localization for Asynchronous Multivariate Time Series

RANSynCoders (or RANCoders) is an unsupervised deep learning architecture for real-time detection and localizaiton within large multivariate time series. The method utilizes synchrony-analysis on latent representations for adjusting asynchronous variates fed into an encoder, bootstrap aggregation of decoders, and quantile loss optimization for anomaly inference, localization, as well as optionally creating interpretable charts of anomalies for industy users.



## Getting Started

#### Clone the repo

```
git clone https://github.com/.... && cd RANSynCoders
```

#### Get data

Pooled Server Metrics (PSM) dataset is in the folder `data`. 

You can get the public datasets from:

* SMD: <https://github.com/NetManAIOps/OmniAnomaly>
* SWaT: <https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/#swat>

#### Install dependencies (with python 3.7) 

(virtualenv is recommended)

```shell
pip install -r requirements.txt
```

If installing on CUDA-enabled GPU device, add the following packages to the requirements.txt file:

* cudatoolkit == 10.1.243
* cudnn == 7.6.5
* tensorflow-gpu == 2.1.0

#### Running an Example

An example using the attached PSM dataset, including instructions for inference and evaluation, is provided in the notebook `example.ipynb`, including instructions for inference and 

## How to cite

If you are utilizing this work in any way, please cite the following papers as appropriate:

### Impact of Synchronization on Representation Learning (ICASSP 2021):

    Audibert, J., Michiardi, P., Guyard, F., Marti, S., Zuluaga, M. A. (2020).
    USAD : UnSupervised Anomaly Detection on multivariate time series.
    Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, August 23-27, 2020

### RANSynCoders (KDD 2021):

    Audibert, J., Michiardi, P., Guyard, F., Marti, S., Zuluaga, M. A. (2020).
    USAD : UnSupervised Anomaly Detection on multivariate time series.
    Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, August 23-27, 2020


If you want to change the default configuration, you can edit `ExpConfig` in `main.py` or overwrite the config in `main.py` using command line args. For example:

```
python main.py --dataset='MSL' --max_epoch=20
```



## Data

### Dataset Information

| Dataset name| Number of entities | Number of dimensions | Training set size |Testing set size |Anomaly ratio(%)|
|------|----|----|--------|--------|-------|
| SMAP | 55 | 25 | 135183 | 427617 | 13.13 |
|MSL | 27 | 55 | 58317 | 73729 | 10.72|
|SMD | 28 |38 | 708405 | 708420 | 4.16 |



### SMAP and MSL

SMAP (Soil Moisture Active Passive satellite) and MSL (Mars Science Laboratory rover) are two public datasets from NASA.

For more details, see: <https://github.com/khundman/telemanom>



### SMD

SMD (Server Machine Dataset) is a new 5-week-long dataset. We collected it from a large Internet company. This dataset contains 3 groups of entities. Each of them is named by `machine-<group_index>-<index>`.

SMD is made up by data from 28 different machines, and the 28 subsets should be trained and tested separately. For each of these subsets, we divide it into two parts of equal length for training and testing. We provide labels for whether a point is an anomaly and the dimensions contribute to every anomaly.

Thus SMD is made up by the following parts:

* train: The former half part of the dataset.
* test: The latter half part of the dataset.
* test_label: The label of the test set. It denotes whether a point is an anomaly. 
* interpretation_label: The lists of dimensions contribute to each anomaly.

concatenate



## Processing

With the default configuration, `main.py` follows these steps:

* Train the model with training set, and validate at a fixed frequency. Early stop method is applied by default.
* Test the model on both training set and testing set, and save anomaly score in `train_score.pkl` and `test_score.pkl`.
* Find the best F1 score on the testing set, and print the results.
* Init POT model on `train_score` to find the threshold of anomaly score, and using this threshold to predict on the testing set.


## Training loss

The figure below are the training loss of our model on MSL and SMAP, which indicates that our model can converge well on these two datasets.

![image](https://github.com/smallcowbaby/OmniAnomaly/blob/master/images/MSL_loss.png)
![image](https://github.com/smallcowbaby/OmniAnomaly/blob/master/images/SMAP_loss.png)