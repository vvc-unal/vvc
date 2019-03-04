#!/bin/bash

err_report() {
    echo "Error on line $1"
}

trap 'err_report $LINENO' ERR

# declare enviroment variables
IMAGE_TAG="vvc/opencv3-keras-tensorflow-gpu-py3"

# Build image, with local Dockerfile
docker build -t $IMAGE_TAG .
