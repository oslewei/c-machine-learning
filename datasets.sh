#!/bin/bash

# CHeck if datasets is there
mkdir datasets

if [ "$1" != "mnist" ] && [ "$1" != "fashion" ]; then
    echo "Usage: $0 [mnist|fashion]"
    exit 1
fi

if [ "$1" = "mnist" ]; then
    if [ -d "./datasets/mnist" ]; then
        echo "MNIST already downloaded, skipping."
    else
        curl -L -o ./datasets/mnist-dataset.zip \
          https://www.kaggle.com/api/v1/datasets/download/hojjatk/mnist-dataset
        unzip ./datasets/mnist-dataset.zip -d ./datasets/mnist
        rm ./datasets/mnist-dataset.zip || true
    fi
fi

if [ "$1" = "fashion" ]; then
    if [ -d "./datasets/fashionmnist" ]; then
        echo "Fashion MNIST already downloaded, skipping."
    else
        curl -L -o ./datasets/fashionmnist.zip \
          https://www.kaggle.com/api/v1/datasets/download/zalando-research/fashionmnist
        unzip ./datasets/fashionmnist.zip -d ./datasets/fashionmnist
        rm ./datasets/fashionmnist.zip
    fi
fi
