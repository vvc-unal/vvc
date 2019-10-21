# List conda enviroments

conda env list

# Install enviroment from file

apt install libcuda1

conda env create -f vvc_environment.yml

# Verify tensorflow gpu 

python -c "import tensorflow as tf;tf.test.gpu_device_name()"

# Remove environment

conda env remove --name vvc
