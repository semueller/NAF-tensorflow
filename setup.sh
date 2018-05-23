#!/bin/bash

source activate naf
echo "creating temporary installation directory"
mkdir -p tmp
cd tmp

echo "installing tqdm"
pip install tqdm

echo "installing datetutils"
pip install python-dateutil

echo "installing tensorflow"
pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0-cp27-none-linux_x86_64.whl

echo "cloning mujoco_py"
git clone https://github.com/openai/mujoco-py.git
cd mujoco-py
git checkout 3d91a9a6b9c13a631dbe68bd826d7c93a896b1cc

echo "installing mujoco_py"
pip install .
cd ../

echo "cloning openaigym"
git clone https://github.com/openai/gym
cd gym
git checkout 3d29fb541b79cb4a0f2f873d7303c83c92e25b8c

echo "installing gym"
pip install .
cd ../

echo "downloading mujoco"
wget https://www.roboti.us/download/mjpro131_linux.zip
echo "unpacking mujoco"
mkdir -p ~/.mujoco
unzip -d ~/.mujoco mjpro131_linux.zip

if [ ! -f ~/.mujoco/mjkey.txt ]; then
	echo "key for mojoco not found, see https://www.roboti.us/license.html on how to get one"
fi

echo "setup.sh finished. you can delete tmp directory after verifying the installation"

