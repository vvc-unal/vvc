#!/bin/bash

# declare enviroment variable
ENVIROMENT="py27-gpu"

source activate $ENVIROMENT

# Update conda packages
conda update conda -y
conda update --all -y

# Update pip packages
pip install --upgrade numpy sk-video

pip install --ignore-installed --upgrade tensorflow-gpu

pip install --upgrade keras
