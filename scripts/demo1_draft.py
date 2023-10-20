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
num_episodes = 10
RENDER_PLOTS = False

# -------------------------------- Load Model -------------------------------- #
# Load the model (ubuntu)
model_1 = SAC.load("saved_models/best_model.zip")
model_2 = SAC.load("saved_models/ubuntu/last_sac_4M.zip")
model_3 = SAC.load("saved_models/ubuntu/MlpPolicy_SAC_step4000000")

# model_3 = SAC.load("saved_models/macos/saved_models/MlpPolicy_SAC")
# model_4 = SAC.load("saved_models/macos/saved_models/MlpPolicy_SAC_2")
# model_5 = SAC.load("saved_models/macos/saved_models/MlpPolicy_SAC_3")

# model_3 = SAC.load("saved_models/best_model")
# model_4 = SAC.load("saved_models/MlpPolicy_SAC")
# model_5 = SAC.load("saved_models/MlpPolicy_SAC_2")


# model_3 = PPO.load("saved_models/ubuntu/best_ppo.zip")
# model_4 = PPO.load("saved_models/ubuntu/last_ppo_4M.zip")


# # Load the model (macos)
# model_5 = SAC.load("saved_models/macos/saved_models/last_sac_3.6M.zip")
model_4 = SAC.load("saved_models/macos/saved_models/last_sac_4M.zip")

# model_7 = PPO.load("saved_models/macos/saved_models/last_ppo_3.2M.zip")
# model_8 = PPO.load("saved_models/macos/saved_models/last_ppo_3.4M.zip")

# model_9 = SAC.load("saved_models/macos/saved_models/best_sac.zip")
# model_10 = PPO.load("saved_models/macos/saved_models/best_ppo.zip")

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
# models = {"last_sac": model_sac, "best_sac": best_sac, "last_ppo": model_ppo, "best_ppo": best_ppo}
models = {"model_1": model_1,
          "model_2": model_2, 
          "model_3": model_3, 
          "model_4": model_4, 
        #   "model_5": model_5, 
        #   "model_6": model_6, 
        #   "model_7": model_7, 
        #   "model_8": model_8, 
        #   "model_9": model_9, 
        #   "model_10": model_10, 
          }
# models = {"best_sac": model}

print("Evaluating Models (Headless)...")
for key in models.keys():
    # Reset the eval env
    eval_env.reset()
    # Test average reward over multiple episodes
    mean_reward, std_reward = evaluate_policy(models[key], eval_env, n_eval_episodes=100)
    print(f"MODEL TYPE: {key}")
    print(f"ep_avg_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"estim_max_reward:{((MAX_BUDGET * 10) / 2):.2f}")
    print(f"step_avg_reward:{(mean_reward/MAX_BUDGET):.2f}\n")



# # --------------------------- Run Interactive Eval --------------------------- #
# model = model_2

# # Wrap the env for the model
# env = make_vec_env(
#     SimpleSimGym,
#     n_envs=1,
#     # monitor_dir=config["logdir"],
#     env_kwargs=dict(
#         max_budget=MAX_BUDGET,
#         max_targets=MAX_TARGETS,
#         num_classes=NUM_CLASSES,
#         player_fov=PLAYER_FOV,
#         action_format=ACTION_FORMAT,
#         render_plots=RENDER_PLOTS,
#         render_mode="human",
#     ),
# )


# obs = env.reset()
# for i in range(num_episodes):
#     terminated = False
#     truncated = False
#     ep_reward = 0
#     found = False
#     j = 0

#     while not (terminated or truncated):
#         # For agent
#         action, _ = model.predict(obs)
#         obs, reward, dones, info = env.step(action)

#         j += 1
#         ep_reward += reward

#         if terminated or truncated:
#             obs, info = env.reset()

#         # time.sleep(0.01)

#     print(f"Total Ep Reward: {ep_reward}")
#     print(f"step_avg_reward:{(ep_reward/MAX_BUDGET):.2f}\n")
