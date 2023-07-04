# Make sure everything is updated
# conda update --all;

# Create env
# conda create -y -n igibson python=3.8;
# conda activate igibson;

# Clone and install igibson
# cd lib;
# git clone https://github.com/StanfordVL/iGibson --recursive;
# cd iGibson;
# pip install -e .;

# Install other minor packages
pip install absl-py;
pip install gin-config;

# Install Tensorflow for macOS ARM
conda install -c apple tensorflow-deps;
pip install tensorflow-macos tensorflow-metal;
pip install numpy==1.23; # to fix the numpy tf incompatibility issue

# # Install tf-agents
# # pip install tf-agents;
# # Install the 'gibson' branch of their fork of tf-agents
# cd agents;
# git checkout igibson;
# # git clone https://github.com/StanfordVL/agents.git;
# pip install -e .;