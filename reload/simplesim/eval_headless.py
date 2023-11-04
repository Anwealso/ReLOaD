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
MAX_BUDGET = 400
MAX_TARGETS = 5
NUM_CLASSES = 10
PLAYER_FOV = 30
ACTION_FORMAT = "continuous"

# Eval
num_episodes = 500
RENDER_PLOTS = True


# -------------------------------- Load Model -------------------------------- #
# Load the model (SAC)
# model_sac = SAC.load("saved_models/last_sac_4M.zip")
best_sac = SAC.load("saved_models/best_sac.zip")
model_sac = SAC.load("saved_models/MlpPolicy_SAC_step4000000.zip")

# Load the model (PPO)
# model_ppo = PPO.load("saved_models/last_ppo_4M.zip")

# For a random policy, simply do:
# env.action_space.sample()


# --------------------------- Evaluate Performance --------------------------- #
# Wrap the env for the model
eval_env = make_vec_env(
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
        render_mode=None,
    ),
)

# Check performance of best vs last model
models = {"model_sac": model_sac,
          "best_sac": best_sac}

print("Evaluating Models (Headless)...")
for key in models.keys():
    # Reset the eval env
    eval_env.reset()
    # Test average reward over multiple episodes
    mean_reward, std_reward = evaluate_policy(models[key], eval_env, n_eval_episodes=100)
    print(f"\n===== ep_avg_reward:{mean_reward:.2f} +/- {std_reward:.2f} =====")
    print(f"MODEL TYPE: {key}")
    print(f"estim_max_reward:{((MAX_BUDGET/2) * 10):.2f}")
    print(f"step_avg_reward:{(mean_reward/(MAX_BUDGET/2)):.2f}\n")
