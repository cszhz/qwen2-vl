# qwen-vl

## Getting started
Qwen-VL model on AWS Inferentia2

The model is based on Neuron Version 2.22.0 

## Neuron Version
1. First need to install Neuron 2.22.0 
https://awsdocs-neuron.readthedocs-hosted.com/en/v2.22.0/frameworks/torch/inference-torch-neuronx.html#inference-torch-neuronx

2. Install NxDI required package
```
cd neuronx-distributed-inference
pip install -e .
```

## Features
Add multiple images support

## Test 
The script locates in `neuronx-distributed-inference/examples/qwen2_vl`

Compile model
```
python compile.py
```

Run model
```
python run.py
```
