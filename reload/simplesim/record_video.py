# simplesim/record_video.py
#
# Used to record videos of trained policies
#
# Alex Nichoson

# Library Imports
from env import SimpleSim
import math
import numpy as np
import time
import random
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import DQN, PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# Load Custom Environment
from env_gym import SimpleSimGym

# Set up fake display; otherwise rendering will fail
import os
os.system("Xvfb :1 -screen 0 1024x768x24 &")
os.environ['DISPLAY'] = ':1'
import base64
from pathlib import Path
from stable_baselines3.common.vec_env import VecVideoRecorder


def record_video(eval_env, model, video_length=500, prefix="", video_folder="videos/"):
    """
    Records a video of a trained model

    :param eval_env: (vec env)
    :param model: (RL model)
    :param video_length: (int)
    :param prefix: (str)
    :param video_folder: (str)
    """
    # Start the video at step=0 and record 500 steps
    eval_env = VecVideoRecorder(
        eval_env,
        video_folder=video_folder,
        record_video_trigger=lambda step: step == 0,
        video_length=video_length,
        name_prefix=prefix,
    )

    obs = eval_env.reset()
    
    for _ in range(video_length):
        action, _ = model.predict(obs)
        obs, _, _, _ = eval_env.step(action)

    # Close the video recorder
    eval_env.close()



# ------------------------------ Hyperparameters ----------------------------- #
# Env
MAX_BUDGET = 400
MAX_TARGETS = 5
NUM_CLASSES = 10
PLAYER_FOV = 30
ACTION_FORMAT = "continuous"


# ------------------------------ Load Model(s) ------------------------------- #
# Load the model (SAC)
sac_1 = SAC.load("saved_models/video_sac_0.4M.zip")
sac_2 = SAC.load("saved_models/video_sac_0.8M.zip")
sac_3 = SAC.load("saved_models/video_sac_1.6M.zip")
sac_4 = SAC.load("saved_models/video_sac_2.8M.zip")
sac_5 = SAC.load("saved_models/video_sac_4M.zip")
best_sac = SAC.load("saved_models/video_best_sac.zip")
best_ppo = PPO.load("saved_models/video_best_ppo.zip")
models = {"best_sac": best_sac,
          "best_ppo": best_ppo,
          "sac_0.4M": sac_1,
          "sac_0.8M": sac_2,
          "sac_1.6M": sac_3,
          "sac_2.8M": sac_4,
          "sac_4M": sac_5,
          }


# best_dqn = DQN.load("saved_models/video_best_dqn.zip")
# models = {"best_dqn": best_dqn,
#           }


# ------------------------------ Record Video(s) ----------------------------- #
# Create videos dir
videos_dir = "./videos/"
os.makedirs(videos_dir, exist_ok=True)

# Instantiate the eval env
eval_env = make_vec_env(SimpleSimGym, 
    n_envs=1, 
    # monitor_dir=config["logdir"], 
    env_kwargs=dict(
        max_budget=MAX_BUDGET, 
        max_targets=MAX_TARGETS, 
        num_classes=NUM_CLASSES, 
        player_fov=PLAYER_FOV, 
        render_mode="rgb_array", 
        action_format=ACTION_FORMAT
    )
    )

for key in models.keys():
    print(f"\nRecording video for policy '{key}'...")
    # Reset the eval env
    eval_env.reset()
    # Record a video of the trained policy
    record_video(eval_env, models[key], video_length=500*4, prefix=f"{key.replace('_', '-')}-simplesim")
