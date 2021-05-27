# RANSynCoders



### Industry-Practical Anomaly Detection and Localization for Asynchronous Multivariate Time Series

RANSynCoders (or RANCoders) is an unsupervised deep learning architecture for real-time anomaly detection and localizaiton within large multivariate time series. The method utilizes synchrony-analysis on latent representations for adjusting asynchronous variates fed into an encoder, bootstrap aggregation of decoders, and quantile loss optimization for anomaly inference, localization, as well as optionally creating interpretable charts of anomalies for industy users.



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

An example using the attached PSM dataset, including instructions for inference and evaluation, is provided in the notebook `example.ipynb`.

## How to cite

If you are utilizing this work in any way, please cite the following papers as appropriate:

### Impact of Synchronization on Representation Learning (ICASSP 2021 - Published):

    Abdulaal, A., Lancewicki, T. (2021).
    Real-Time Synchronization in Neural Networks for Multivariate Time Series Anomaly Detection.
    2021 IEEE International Conference on Acoustics, Speech, and Signal Processing, June 6-11, 2021
click [here](https://ieeexplore.ieee.org/document/9413847) to access the above publication.

### RANSynCoders (KDD 2021 - Accepted):

    Abdulaal, A., Liu, Z., Lancewicki, T. (2021).
    Practical Approach to Asynchronous Multivariate Time Series Anomaly Detection and Localization.
    Proceedings of the 27th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, August 14-18, 2021

## Copyright and licensing

Copyright 2021 eBay Inc

The content of this repository (other than sample data) are released under the BSD-3 license. More information are provided in the LICENSE file

The sample data files are released under the Creative Commons Attribution 4.0 Internation License (<https://creativecommons.org/licenses/by/4.0/>). More information are provided in the data/LICENSE file
