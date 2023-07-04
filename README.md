# ReLOaD: Reinforcement Learning for Object Detection

Alex Nicholson, The University of Queensland, 2023

---

## TODO

- [DONE] Build an environment class and function that can replicate the internal state and progression of the igibson env
- [DONE] Plot this in matplotlib or pygame to visualise how it works
- [DONE] Wrap it into a gym environment to allow for quicker porting of the agent code to igibson later

- [DONE] FOV indicator lines for the robot
- [WIP] Hook up the TF-Agents rl agent up to this and let it train for a bit
- Save screenshots of the completed environment every couple of episodes
- Add some walls as obstructions

Factors to include in detection function
- Distance
- Angle off frontal
- Size of object
- Obstructions

# ...

---

## Installation / Setup

### Installing iGibson

- Follow installation guide at: https://stanfordvl.github.io/iGibson/installation.html
- Test the installation by running `python -m igibson.examples.environments.env_nonint_example`


## Usage

- ...

## Extra Resources

- https://docs.python-guide.org/writing/structure/


---

Made with ❤️ by Alex Nicholson