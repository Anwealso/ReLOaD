# Full setup script from complete BASE ubuntu installation
# NOTE: Not meant to run and install everythin in one go, simply meant to be a recorded list of all operations requred to install all necessary packages

# Setup dir tree
cd ~/Documents
mkdir dev
cd dev

# Install core applications
sudo apt update
sudo snap install --classic code
sudo apt install git
sudo snap install spotify

# Install nvidia drivers and cuda (iGibson Tutorial Version)
# DEFINITELY do it this was as I have not found any other way to reliably work
# https://stanfordvl.github.io/iGibson/installation.html#installing-the-environment
# Add the nvidia ubuntu repositories
sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    gnupg2 curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

# The following cuda libraries are required to compile igibson
sudo apt-get update && sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    xserver-xorg-video-nvidia-470 \
    cuda-cudart-11-1=11.1.74-1 \
    cuda-compat-11-1 \
    cuda-command-line-tools-11-1=11.1.1-1 \
    cuda-libraries-dev-11-1=11.1.1-1 \

# For building and running igibson
sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    cmake \
    git \
    g++ \
    libegl-dev

# Install anaconda
wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh
bash Anaconda3-2023.03-1-Linux-x86_64.sh
rm -f Anaconda3-2023.03-1-Linux-x86_64.sh
source ~/.bashrc

# Clone my main repo
git clone https://github.com/Anwealso/ReLOaD.git

# Clone the yolo repo
cd ReLOaD
git clone https://github.com/anushkadhiman/YOLOv3-TensorFlow-2.x.git

# Setup conda envs
# Pygame simplesim env
conda env create -f setup/environment-pygame.yml

# Test igibson installation
bash check_igibson_requirements.sh

# iGibson env
conda update -y conda
conda create -y -n igibson python=3.8
conda activate igibson
# Install igibson with pip
pip install igibson  # This step takes about 4 minutes
# run the demo
python -m igibson.examples.environments.env_nonint_example