#!/bin/bash

err_report() {
    echo "Error on line $1"
}

trap 'err_report $LINENO' ERR

# declare variables
IMAGE_TAG="vvc/opencv3-keras-tensorflow-gpu-py3"
CONTAINER_NAME="mlserver"

if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
	# Delete previous container
	docker rm $CONTAINER_NAME
  echo "$CONTAINER_NAME container deleted"
fi

# Create container
nvidia-docker run -it --name $CONTAINER_NAME \
  -v "${HOME}"/workspace/Maestria/Repositories:/vvc/Repositories \
  -v "${HOME}"/workspace/Maestria/Model:/vvc/Model \
  -v /media/Store/juan/VVC_Videos:/vvc/Videos \
  -p 8888:8888 $IMAGE_TAG

# Start the container
# nvidia-docker start -it $CONTAINER_NAME

# nvidia-docker exec -it $CONTAINER_NAME bash 
