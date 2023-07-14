# Check os version
echo $'\n# ------------------------------------ OS ------------------------------------ #'
cat /etc/os-release
# Check graphics card model and driver version
echo $'\n# ------------------------------- Nvidi Driver ------------------------------- #'
nvidia-smi
# Check cuda version
echo $'\n# ----------------------------------- CUDA ----------------------------------- #'
nvcc --version
# Check cmake version
echo $'\n# ----------------------------------- CMAKE ---------------------------------- #'
cmake --version
# GNU c++ compiler
echo $'\n# ------------------------------- g++ Compiler ------------------------------- #'
g++ --version
# Check OpenGl version
echo $'\n# ---------------------------------- OpenGL ---------------------------------- #'
glxinfo | grep "OpenGL version"
