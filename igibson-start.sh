docker run --gpus all -ti --rm igibson/igibson-vnc:v2.0.6 /bin/bash
# run a GUI example after the container command line prompt shows:
python -m igibson.examples.environments.env_nonint_example


# pip install igibson  # This step takes about 4 minutes
# # run the demo
# python -m igibson.examples.environments.env_nonint_example


# Make up a docker image based on ubuntu with a conda env with igibson installed