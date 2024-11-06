# Image Captioning

<div align="center">

[![python](https://img.shields.io/badge/-Python_3.11.10-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![pytorch](https://img.shields.io/badge/Torch_2.5.1-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.4.0-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1)](https://pycqa.github.io/isort/) </br>
[![demo](https://img.shields.io/badge/Demo-HuggingFace-F7DF1E)](https://huggingface.co/spaces/tanthinhdt/IMCAP)
[![report](https://img.shields.io/badge/Report-Wandb-F7DF1E)](https://huggingface.co/spaces/tanthinhdt/IMCAP)

</div>

## Table of Contents

- [Description](#description)
- [Results](#results)
- [Installation](#installation)
  - [Pip](#pip)
  - [Conda](#conda)
- [Training](#training)
- [Inference](#inference)

## Description

In this project, I develop, train, and evaluate models for image captioning, inspired by BLIP's approach. The goal is to create a system that can generate descriptive and accurate captions for images. Additionally, I build a demo web app [here](https://huggingface.co/spaces/tanthinhdt/IMCAP) to showcase these models in action, providing an interactive platform for users to experience the capabilities of AI-driven image captioning firsthand.

## Results

The Flickr30k dataset is divided into training and testing sets with a 70/30 split.

| Model | Test WER | Test BLEU@4 | Train WER | Train BLEU@4 | Config | Checkpoint | Report | Paper |
| ----- | --------- | ------------ | -------- | ----------- | ------ | ---------- | ------ | ----- |
| BLIP Base | 59.15 | 14.11 | 55.61 | 16.11 | [Config](configs/experiment/flickr30k_blip-base_v2-0.yaml) | [HuggingFace](tanthinhdt/blip-base_with-pretrained_flickr30k) | [Wandb](https://wandb.ai/tanthinhdt24/IMCAP/reports/BLIP-Base--VmlldzoxMDAwNzI5Mw) | [Arxiv](https://arxiv.org/abs/2201.12086) |

## Demo

You can this [notebook](notebooks/inference.ipynb) ([Colab](https://colab.research.google.com/drive/1nwZMduGusla-5esQPTZJjuEkJlu_j4j1?usp=sharing)) or this [demo](https://huggingface.co/spaces/tanthinhdt/IMCAP) on HuggingFace for inference. You can also use the Streamlit demo offline by running this command from the root directory.

```
streamlit src/app.py
```

## Installation

### Pip

```bash
# clone project
git clone https://github.com/tanthinhdt/imcap
cd imcap

# [OPTIONAL] create conda environment
conda create -n imcap python=3.11.10
conda activate imcap

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

### Conda

```bash
# clone project
git clone https://github.com/tanthinhdt/imcap
cd imcap

# create conda environment and install dependencies
conda env create -f environment.yaml -n imcap

# activate conda environment
conda activate imcap
```

## Training

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```