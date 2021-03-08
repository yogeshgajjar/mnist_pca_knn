# MNIST_PCA_KNN

Image classification using PCA and K-NN on MNIST Dataset

## Overview

**Dataset : MNIST :**
The MNIST dataset contain gray-scale handwritten digits (28 × 28 each) with 10 classes
(i.e. 0, 1, ..., 9). There are in total 60000 images for training and 10000 images for testing.
It is a widely-used dataset for bench-marking image classification models.

## Objective

- Split the images and labels into training and testing sets. The first N images are used as testing images which are queries for K-NN classifier. The rest of (1000 − N ) images are used for training. (N is specified as an input argument.)
- Perform dimensionality reduction on both training and testing sets to reduce the dimension of data from 784 to D. (D is specified as an input argument.)
- Implement a K-Nearest Neighbors classifier to predict the class labels of testing images. Use weighted majority polling for classification. (K is specified as an input argument.)
**Not allowed to use any library provided by K-NN classifier**

## Pre-Processing

- Extracting the images and label using pre-installed libraries in Python. The dataset is in idx format dataset file which can be downloaded from website cited in the dataset topic.
- Extracting 1000 out of 60000 images and labels from the dataset file.
- Columnnize images by reshaping all the images as 784 D vectors.
- Train-test split : Split the images and labels into training and testing sets. The first N images are used as testing images which are queries for K-NN classifier. The rest of (1000 − N ) images are used for training. (N is specified as an input argument.)

## Concept

- **PCA :**
Instead of classifying images in the pixel domain, we usually first project them into a feature space since raw input data is often too large, noisy and redundant for analysis. Here is where dimensionality reduction techniques come into play. Dimensionality reduction is the process of reducing the number of dimensions of each data point while preserving as much essential information as possible. PCA is one of the main techniques of dimensionality reduction. It performs a linear mapping of the data to a lower-dimensional space in such a way that the variance of the data in the lower-dimensional representation is maximized.

- **KNN :**
K-nearest neighbors algorithm (K-NN) is a non-parametric method used for classification. A query object is classified by a majority vote of the K closest training examples (i.e. its neighbors) in the feature space.

## Program Description

### Script

The .py file includes sys.argv[] and takes four argument inputs including K, N, D and PATH_TO_DATA_DIR (where the image and label files are). The last argument will the folder directory where you will have the dataset files.
```
python3 john.py 5 12 20 /home/john/
```

### Output

The .py file outputs a text file named **finaloutput.txt** after execution. Each line of the output contains a predicted label and a ground truth label, separated by a single space.

## Dataset

The dataset can be downloaded from the link : http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz and http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz

- For more description about extracting images and labels from the downloaded file refer http://yann.lecun.com/exdb/mnist/ 
