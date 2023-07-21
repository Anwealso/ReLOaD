# ReLOaD: Reinforcement Learning for Object Detection

Alex Nicholson, The University of Queensland, 2023

---

## TODO

### Want to implement these methods:

my_trainer = Trainer()

my_trainer.set_envs(py_env, train_tf_env, eval_tf_env)
    - sets the env for training
my_trainer.restore_agent_checkpoint()
    - restores agent from checkpoint
my_trainer.set_agent(agent)
    - sets the agent for training
returns = my_trainer.train()
    - trains the agent, saving checkpoints and exporting finished policy
    - also saves a picture of the training_progress graph to file
    - Returns: the avg returns over training




### Extra:
train model script - trains a model 
    - optional --checkpoint argument that allows us to resume the training from a given checkpoint
    - optional --save_path arg that tells it where to save checkpoints and SavedPolicies (if no value given just save in a default saved_models/ location)

run model script
    - Loads the saved policy from file and validates its performance with a few episodes.

retrain_script
make_thing





Train Script
- Make train script into a trainer wrapper class

Extra
- [DONE] Train a basic TF-Agents DQN agent (or any agent really) on the simplesim env
- [DONE] Save screenshots of raining progress
- Add checkpoint and finished model saving with: https://www.tensorflow.org/agents/tutorials/10_checkpointer_policysaver_tutorial
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

### Install My Lib
`git clone https://github.com/Anwealso/ReLOaD.git`


### Install YOLO Lib
`git clone https://github.com/anushkadhiman/YOLOv3-TensorFlow-2.x.git`




## Usage

- ...

## Extra Resources

- https://docs.python-guide.org/writing/structure/


---

Made with ❤️ by Alex Nicholson