# LightSTD

This repository contains the Pytorch implementation code for the paper "LightSTD"

## Architectures

![overall](https://github.com/user-attachments/assets/66fee33d-58e0-4cf0-82a4-7e6105551862)

(a) shows the overall architecture of LightSTD. Condition Network makes representations of history 

## Dependencies
- CUDA 12.1
- python 3.11.9
- pytorch 2.1.1
- torch-geometric 2.5.3
- torchmetrics 1.3.2
- numpy 1.26.3
- hydra-core 1.3.2

##  Datasets
We used seven benchmark datasets; MNIST Superpixels, CoMA, FAUST, AIFB, MUTAG, BGS, and AM. You can refer to torch-geometric documentation for datasets [here](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html).

## Results
Testing accuracy of Graph Classification are summarized below.

<table>
  <tr>
    <td><b>Method</b></td>
    <td><b>MNIST Superpixels</b></td>
    <td><b>CoMA</b></td>
    <td><b>FAUST</b></td>
  </tr>
  <tr>
    <td>GCN</td>
    <td align="right">81.83</td>
    <td align="right">27.02</td>
    <td align="right">69.00</td>
  </tr>
  <tr>
    <td>GraphSAGE</td>
    <td align="right">94.44</td>
    <td align="right">74.17</td>
    <td align="right">70.00</td>
  </tr>
  <tr>
    <td>$k$-GNN</td>
    <td align="right">91.58</td>
    <td align="right">73.54</td>
    <td align="right">71.00</td>
  </tr>
  <tr>
    <td>GIN</td>
    <td align="right">90.15</td>
    <td align="right">87.40</td>
    <td align="right">73.00</td>
  </tr>
  <tr>
    <td>GAT</td>
    <td align="right">97.08</td>
    <td align="right">84.41</td>
    <td align="right">73.00</td>
  </tr>
  <tr>
    <td>GATv2</td>
    <td align="right">98.08</td>
    <td align="right">88.00</td>
    <td align="right"><i>78.00</i></td>
  </tr>
  <tr>
    <td>EdgeConv</td>
    <td align="right"><b>99.00</b></td>
    <td align="right"><b>99.46</b></td>
    <td align="right">75.00</td>
  </tr>
  <tr>
    <td>SelectionGCN</td>
    <td align="right">95.84</td>
    <td align="right">73.14</td>
    <td align="right">57.00</td>
  </tr>
  <tr>
    <td>vRGCN</td>
    <td align="right"><i>98.29</i></td>
    <td align="right"><i>90.88</i></td>
    <td align="right"><b>82.00</b></td>
  </tr>
</table>

Testing accuracy of Node Classification are summarized below.

<table>
  <tr>
    <td><b>Method</b></td>
    <td><b>AIFB</b></td>
    <td><b>MUTAG</b></td>
    <td><b>BGS</b></td>
    <td><b>AM</b></td>
  </tr>
  <tr>
    <td>GCN</td>
    <td align="right">90.00</td>
    <td align="right">60.88</td>
    <td align="right">60.69</td>
    <td align="right">34.85</td>
  </tr>
  <tr>
    <td>GraphSAGE</td>
    <td align="right">82.78</td>
    <td align="right">63.82</td>
    <td align="right">57.24</td>
    <td align="right">40.71</td>
  </tr>
  <tr>
    <td>$k$-GNN</td>
    <td align="right">91.11</td>
    <td align="right">73.53</td>
    <td align="right">62.76</td>
    <td align="right">70.30</td>
  </tr>
  <tr>
    <td>GIN</td>
    <td align="right">87.78</td>
    <td align="right"><i>75.59</i></td>
    <td align="right">65.52</td>
    <td align="right">57.88</td>
  </tr>
  <tr>
    <td>GAT</td>
    <td align="right"><i>94.44</i></td>
    <td align="right">70.29</td>
    <td align="right">62.76</td>
    <td align="right">64.34</td>
  </tr>
  <tr>
    <td>GATv2</td>
    <td align="right"><i>94.44</i></td>
    <td align="right">66.47</td>
    <td align="right">66.21</td>
    <td align="right">61.92</td>
  </tr>
  <tr>
    <td>EdgeConv</td>
    <td align="right">70.56</td>
    <td align="right">62.65</td>
    <td align="right">57.93</td>
    <td align="right">63.03</td>
  </tr>
  <tr>
    <td>RGCN</td>
    <td align="right"><b>97.22</b></td>
    <td align="right">71.76</td>
    <td align="right"><b>78.62</b></td>
    <td align="right"><b>89.60</b></td>
  </tr>
  <tr>
    <td>vRGCN</td>
    <td align="right">93.33</td>
    <td align="right"><b>80.00</b></td>
    <td align="right"><i>75.17</i></td>
    <td align="right"><i>72.12</i></td>
  </tr>
</table>

## Usage
You can run graph classification or node classification using the following commands.

```
python train_graph_classification.py
python train_node_classification.py
```

You can use the following commands if you want to run with GPUs.

```
python train_node_classification.py device=cuda
python train_link_prediction.py device=cuda
```

## Hyperparameters
You can change hyperparameters through the additional command "{name}={value}".

For example:

```
python train_graph_classification.py learner_type=GNNLearner
```

Please check [config.yaml](https://github.com/dxlabskku/vRGCN/tree/main/config.yaml) for the hyperparamters.
