## GAN in Pytorch for Normal and Time-Series Data

This repository contains implimentation of three GAN architecture in PyTorch.

1. Normal GAN
2. WGAN
3. LSGAN

### Dataset Used:

1. Sine dataset with 5 dimension
2. Google Stock Dataset

Data are kept in the `data` directory.

### Required package

Install the required packages by using this command:

`pip install -r requirements.txt`

### To run the code:

To use the different version of GAN and different datasets, change the `logs` dictionary in the `train_gan.py`.

Such as:

For using the google dataset, set the `n_features` to `6` also, for the different version of GAN, change the `loss_mode`. Options are `'wgan', 'normal', 'lsgan'`.

To run:

`python train_gan.py`