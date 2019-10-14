# Dockerfile with tensorflow gpu support on python3, opencv3.3, and keras
FROM tensorflow/tensorflow:1.4.1-gpu-py3

LABEL maintainer="jsnavarroa@unal.edu.co"


# The code below is based on the repos:
#   https://github.com/janza/docker-python3-opencv
#   https://github.com/fbcotter/docker-tensorflow-opencv


# Core linux dependencies.
RUN apt-get update && apt-get install -y \
        apt-utils \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        yasm \
        pkg-config \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libjasper-dev \
        libavformat-dev \
        libhdf5-dev \
        libpq-dev \
        ffmpeg && \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/*


# Python dependencies
RUN pip3 --no-cache-dir install --upgrade pip
RUN pip3 --no-cache-dir install \
    numpy \
    hdf5storage \
    h5py \
    scipy \
    py3nvml


# Install VVC dependencies
RUN pip3 --no-cache-dir install \
    keras~=2.1 \
    scikit-video~=1.1


# OpenCV
ENV OPENCV_VERSION="3.3.1"

WORKDIR /
RUN wget -nv https://github.com/opencv/opencv/archive/$OPENCV_VERSION.zip \
	&& unzip -q $OPENCV_VERSION.zip \
	&& mkdir /opencv-$OPENCV_VERSION/cmake_binary \
	&& cd /opencv-$OPENCV_VERSION/cmake_binary \
	&& cmake -DBUILD_TIFF=ON \
		  -DBUILD_opencv_java=OFF \
		  -DWITH_CUDA=OFF \
		  -DENABLE_AVX=ON \
		  -DWITH_OPENGL=ON \
		  -DWITH_OPENCL=ON \
		  -DWITH_IPP=ON \
		  -DWITH_TBB=ON \
		  -DWITH_EIGEN=ON \
		  -DWITH_V4L=ON \
		  -DBUILD_TESTS=OFF \
		  -DBUILD_PERF_TESTS=OFF \
		  -DCMAKE_BUILD_TYPE=RELEASE \
		  -DCMAKE_INSTALL_PREFIX=$(python3 -c "import sys; print(sys.prefix)") \
		  -DPYTHON_EXECUTABLE=$(which python3) \
		  -DPYTHON_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
		  -DPYTHON_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") .. \
	&& make -j$(nproc) \
	&& make install \
	&& rm /$OPENCV_VERSION.zip \
	&& rm -r /opencv-$OPENCV_VERSION
