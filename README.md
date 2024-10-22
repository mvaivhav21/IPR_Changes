# Bayesian Crowd Counting (ICCV 2019 oral)

- [Arxiv](https://arxiv.org/abs/1908.03684) 
- [CVF](http://openaccess.thecvf.com/content_ICCV_2019/papers/Ma_Bayesian_Loss_for_Crowd_Count_Estimation_With_Point_Supervision_ICCV_2019_paper.pdf)

## Code

### Install Dependencies

Make sure you have the following installed:
- **torch** >= 1.0 
- **torchvision**
- **opencv**
- **numpy**
- **scipy**

You can install these via `pip` or `conda` as needed.

This code was tested with Python 3.6.

### Train and Test Instructions

#### 1. Download the UCF-QNRF Dataset

Download the dataset from this [link](https://www.crcv.ucf.edu/data/ucf-qnrf/).

#### 2. Pre-process the Data

Resize the images and split the data for training and validation using the following command:
```bash
python preprocess_dataset.py --origin_dir <directory of original data> --data_dir <directory of processed data>

#### 3. Testing Model(validate on single GTX Titan X)

```bash
python train.py --data_dir <directory of processed data> --save_dir <directory of log and model>





