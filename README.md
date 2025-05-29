# LightSTD: Simplifying Diffusion Models for Efficient Probabilistic Spatio-Temporal Graph Forecasting

This repository contains the Pytorch implementation code for the paper "LightSTD: Simplifying Diffusion Models for Efficient Probabilistic Spatio-Temporal Graph Forecasting"

## Architectures

![overall](https://github.com/user-attachments/assets/66fee33d-58e0-4cf0-82a4-7e6105551862)

(a) shows the overall flow of LightSTD. Condition Network makes condition $c_\phi$ from history, and Denoising Network removes noise iteratively to generate future predictions from Gaussian noise. (b) and (c) show the architecture of Condition Network and Denoising Network, respectively.

## Dependencies
- CUDA 11.7
- python 3.11.11
- pytorch 2.4.0
- torch-geometric 2.6.1
- torch_geometric_temporal 0.54.0
- ema-pytorch 0.7.7
- numpy 1.26.4
- hydra-core 1.3.2
- tqdm 4.67.1

##  Datasets
We used two benchmark datasets; METR-LA and PEMS-BAY. You can refer to torch-geometric-temporal documentation for datasets [here](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/dataset.html#module-torch_geometric_temporal.dataset.chickenpox).

## Usage
You can run Spatio-Temporal Forecasting with LightSTD using the following commands.

```
python train.py
```

You can use the following commands if you want to run with GPUs.

```
python train.py device=cuda
```

## Hyperparameters
You can change hyperparameters through the additional command "{name}={value}".

For example:

```
python train.py time_step=12 num_samples=8
```

Please check [config.yaml](https://github.com/dxlabskku/LightSTD/blob/main/config.yaml) for the hyperparameters.
