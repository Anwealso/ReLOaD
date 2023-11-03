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


# ------------------------------ Hyperparameters ----------------------------- #
# Env
MAX_BUDGET = 50
MAX_TARGETS = 5
NUM_CLASSES = 10
PLAYER_FOV = 30
ACTION_FORMAT = "continuous"

# Eval
num_episodes = 10
RENDER_PLOTS = True


# -------------------------------- Load Model -------------------------------- #
# Load the model (SAC)
# model_sac = SAC.load("saved_models/last_sac_4M.zip")

# Best model teraied at v30
model_sac = SAC.load("saved_models/MlpPolicy_SAC_step4000000.zip")

# Load the model (PPO)
# model_ppo = PPO.load("saved_models/last_ppo_4M.zip")


# --------------------------- Run Interactive Eval --------------------------- #
model = model_sac

# Wrap the env for the model
env = make_vec_env(
    SimpleSimGym,
    n_envs=1,
    # monitor_dir=config["logdir"],
    env_kwargs=dict(
        max_budget=MAX_BUDGET,
        max_targets=MAX_TARGETS,
        num_classes=NUM_CLASSES,
        player_fov=PLAYER_FOV,
        action_format=ACTION_FORMAT,
        render_plots=RENDER_PLOTS,
        render_mode="human",
        # seed=808,
    ),
)


obs = env.reset()
for i in range(num_episodes):
    print(f"EP {i}")
    terminated = False
    truncated = False
    ep_reward = 0
    found = False
    j = 0

    while True:
        # For agent
        action, _ = model.predict(obs)
        obs, reward, dones, info = env.step(action)

        j += 1
        ep_reward += reward

        if terminated or truncated:
            print(f"Total Ep Reward: {ep_reward}")
            print(f"step_avg_reward: {(ep_reward/MAX_BUDGET):.2f}\n")
            obs, info = env.reset()
            break

        # time.sleep(0.01)

