# ReLOaD: Reinforcement Learning for Object Detection

Alex Nicholson, The University of Queensland, 2023

Bachleor of Engineering Honors Thesis

---

## Overview

### Aim
The aim of the thesis project was to create an system that augments the abilities of an exitsing object detection algorithm with a reinforcement learning trained agent that can manoeuvre the camera view in order to improve the accuracy of object detection results.

### Envirionment

In order to train this behaviour, the agent is trained in a custom simulator to control the movement of a non-holonomic mobile robot to explore its environment with the goal of accurately identifying a set of known-position target objects randomly scattered around the environment. The robots goal is given a certain time budget to explore the environmen and its goal is to collect as much information as possible by taking as many high-quality observations of the targets as possible. Essentially the agent tries to plan the most 'informative' path through the environment.

The entire simulator environment was built in python using Pygame for rendering and the interface built purely using the Gymnasium API. The formal problem definition MDP as implemented in the Gymnasoum environment is as follows:
- State Space:
```python
self.observation_space = spaces.utils.flatten_space(
    spaces.Dict(
        {
            "targets": spaces.Box(
                low=-max_dist,
                high=self.game.max_budget * self.game.max_targets,
                shape=(3, max_targets),
                dtype=np.float32
            ),  # target info (relative_x_dist, relative_y_dist, entropy)
            "environment": spaces.Box(
                low=0,
                high=self.game.max_budget,
                shape=(1, 1),
                dtype=np.float32
            )  # environment remaining budget
        }
    )
)
```

- Action Space: 
```python
self.action_space = spaces.Box( # robot twist motion (v, w)
    low=-1,
    high=1,
    shape=(2,),
    dtype=np.float32
)
```
- Reward Function: 

The reward given at each timestep is a scaled version of the average information gain across the targets in the environment.

$$reward = R_{s} = \frac{1}{M}\sum_{m=0}^{M-1}{{IG}_m}$$

where: $M$ = number of targets

$${IG}_m = \frac{1}{T}   \sum_{t=0}^{T-1}{\left(  1 - \sum_{n=0}^{N-1}{-p_n \log_{N}p_n}   \right)}$$

where: $N$ = number of classes, $T$ = number of timesteps so far, and $p_n$ is the probability of target $m$ being of class $n$ (as returned by the object detection algorithm at timestep $t$)

### Method

The model selected 

### Results
- Environment image...
- Confidence histograms...
- Training plots...


## Installation / Setup

### Installing Core Dependancies

1. Clone this repository:
```bash
git clone https://github.com/Anwealso/ReLOaD.git
cd ReLOaD/
```

2. Create a new conda environment and install dependancies:
```bash
conda env create -f environment.yml
conda activate reload
```


### Installing iGibson

- Follow installation guide at: https://stanfordvl.github.io/iGibson/installation.html
- Test the installation by running `python -m igibson.examples.environments.env_nonint_example`

### Installing YOLO
```bash
conda activate reload
pip install -U ultralytics
```


## Usage

- To train the model, launch jupyter-notebook and run the training notebook: `train_agent_simplesim.ipynb`.
- To demo the trained model, run:
```bash
python run_eval.py
```

## Extra Resources

- Gymnasium Documentation: https://gymnasium.farama.org/
- Stable Baselines Documentation: https://stable-baselines.readthedocs.io/en/master/
- Soft Actor Critic Model Information: https://spinningup.openai.com/en/latest/algorithms/sac.html
- Package Structure: https://docs.python-guide.org/writing/structure/


---

Made with ❤️ by Alex Nicholson