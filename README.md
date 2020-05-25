# Project 3

Classifying images from a modified MNIST [dataset](https://www.kaggle.com/c/comp-551-w2019-project-3-modified-mnist/data). Each image contains multiple digits.
Goal is to find largest digit and predict its label between 0 and 9.

See `requirements.txt` for required packages.

## Models

Our model is based on [VGG16](https://arxiv.org/abs/1409.1556), we simply removed the last two blocks of convolutions and one of the hidden layers, and then divided the dimensionality by 2 for the convolutions and 4 for the hidden layers. Our implementation can be found under `models/cnn_vgg.py`.

We also included a simple ensemble-type model that implements bagging under `models/cnn_ensemble.py`. This can be used along with our previous model to increase accuracy.
