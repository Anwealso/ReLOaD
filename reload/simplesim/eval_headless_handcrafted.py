# simplesim/eval_headless_naive.py
# 
# A version of eval_headless but for benchmarking handcrafted policies that 
# need to have the actions drawn one by one (like the handcrafted naïve policy 
# or a random policy sampling the action space)
#
# Alex Nichoson

# Library Imports
from env import SimpleSim, NaivePolicy
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
num_episodes = 10
RENDER_PLOTS = True


# -------------------------------- Environment ------------------------------- #
# Instantiate two environments: one for training and one for evaluation.
env = SimpleSimGym(
    max_budget=MAX_BUDGET,
    max_targets=MAX_TARGETS,
    num_classes=NUM_CLASSES,
    player_fov=PLAYER_FOV,
    action_format=ACTION_FORMAT,
    render_mode=None,
    # seed=808,
)
obs = env.reset()

# --------------------------- LOAD MODEL IF DESIRED -------------------------- #

# --------------------------------- RUN EVAL --------------------------------- #
num_episodes = 1000 # number of episodes to eval over
obs = env.reset()

ep_rewards = []

for mode in ["naive"]:
    for i in range(num_episodes):
        terminated = False
        truncated = False
        ep_reward = 0
        found = False
        j = 0

        ep_reward = 0

        naive_policy = NaivePolicy(env.game)
        
        while not (terminated or truncated):
            if mode == "naive":
                # For naive policy
                action = naive_policy.get_action(env.game.robot)
            # if mode == "random":
            #     # For a random policy, simply do:
            #     action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)

            j += 1
            ep_reward += reward

            if terminated or truncated:
                obs, info = env.reset()

        # print(f"Episode {i}, reward={ep_reward}")
        ep_rewards.append(ep_reward)

    std_dev = math.sqrt(np.var(ep_rewards))
    avg_ep_reward = np.average(ep_rewards)

    print(f"\nPOLICY: {mode}")
    print(f"Average Ep Reward: {avg_ep_reward:.2f}, Std Deviation: {std_dev:.2f}")
