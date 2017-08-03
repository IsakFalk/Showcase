sudo apt update -y && sudo apt upgrade -y
sudo apt install build-essential linux-image-extra-`uname -r` -y

chmod +x NVIDIA-Linux-x86_64-375.39.run
sudo ./NVIDIA-Linux-x86_64-375.39.run

chmod +x cuda_8.0.61_375.26_linux.run
./cuda_8.0.61_375.26_linux.run --extract=`pwd`/extracts
sudo ./extracts/cuda-linux64-rel-8.0.61-21551265.run

echo -e "export CUDA_HOME=/usr/local/cuda\nexport PATH=\$PATH:\$CUDA_HOME/bin\nexport LD_LIBRARY_PATH=\$LD_LINKER_PATH:\$CUDA_HOME/lib64" >> ~/.bashrc
source ~/.bashrc

tar xf cudnn-8.0-linux-x64-v5.1.tgz
cd cuda
sudo cp lib64/* /usr/local/cuda/lib64/
sudo cp include/cudnn.h /usr/local/cuda/include/
