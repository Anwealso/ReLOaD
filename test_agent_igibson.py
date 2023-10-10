# ReLOaD
#
# test_agent_igibson.py
#
# Tests the full RL - YOLO - iGibson stack
#
# Alex Nichoson
# 10/10/2023


# ---------------------------------------------------------------------------- #
#                                    IMPORTS                                   #
# ---------------------------------------------------------------------------- #

import torch
import time
import numpy as np
import logging
import os
from sys import platform
import yaml
from matplotlib import pyplot as plt
import random

from stable_baselines3 import SAC, PPO, DQN
import reload

import igibson
from igibson.envs.igibson_env import iGibsonEnv
from igibson.render.profiler import Profiler
from igibson.utils.assets_utils import download_assets, download_demo_data
from igibson.controllers import ControlType

# ---------------------------------------------------------------------------- #
#                                   FUNCTIONS                                  #
# ---------------------------------------------------------------------------- #

def init_igibson(config_filename):
    # If they have not been downloaded before, download assets and Rs Gibson (non-interactive) models
    download_assets()
    download_demo_data()
    
    # config_filename = os.path.join(igibson.configs_path, "turtlebot_static_nav.yaml")
    config_data = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    # Reduce texture scale for Mac.
    if platform == "darwin":
        config_data["texture_scale"] = 0.5

    # Shadows and PBR do not make much sense for a Gibson static mesh
    config_data["enable_shadow"] = False
    config_data["enable_pbr"] = False

    return config_data


def create_observation_vector(detections, env):
    """
    Take the YOLO detections output and internal iGibson position data and 
    shape them into the correct standardised observation vector format

    TODO: Implement proper obs vector format once standardised
    """
    print("\n\n===== create_observation_vector =====\n")
    
    # detections = detections.pandas().xyxy

    targets = []
    # Get internal position data from iGibson
    targets = []
    for obj in env.scene.get_objects():
        # print(obj)
        # print(obj.get_position())
        if str(type(obj)) != "Turtlebot": # only add target positions
            robot = obj.get_position()

        else:    
            targets.append(obj.get_position())

    # Now we have loaded the data into: detections, robot, targets
    print(f"detections: {detections}\n")
    print(f"robot: {robot}\n")
    print(f"targets: {targets}\n")

    # observation = np.zeros(shape=(2 + len(targets)*3 + 1, 1))
    observation = np.zeros(shape=(2*3,))
    
    # Add all target info to the observation vector (x, y, fully_explored)
    # for i in range(0, len(targets)):
    for i in range(0, 2):
        # observation[2*i, 0] = targets[i, 0] 
        # observation[2*i + 1, 0] = targets[i, 1] 
        # observation[2*i + 2, 0] = float(detections[i]) 
        # Dummy values:
        observation[2*i,] = random.randint(0, 100) 
        observation[2*i + 1,] = random.randint(0, 100) 
        observation[2*i + 2,] = 0

    print(f"observation: {observation}")

    return observation



# ---------------------------------------------------------------------------- #
#                                     MAIN                                     #
# ---------------------------------------------------------------------------- #

def main(selection="user", headless=False, short_exec=False):
    """
    Creates an iGibson environment from a config file with a turtlebot in Rs (not interactive).
    It steps the environment 100 times with random actions sampled from the action space,
    using the Gym interface, resetting it 10 times.
    """
    print("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)

    # ----------------------------- SETUP IGIBSON SIM ---------------------------- #
    # config_filename = os.path.join(igibson.configs_path, "turtlebot_static_nav.yaml")
    config_filename = "/home/alex/Documents/METR4911/ReLOaD/turtlebot_nav_custom.yaml"
    config_data = init_igibson(config_filename)
    env = iGibsonEnv(config_file=config_data, mode="gui_interactive" if not headless else "headless")
    # # View Env Specs
    # reload.utils.show_env_summary(env)


    # --------------------------- LOAD TRAINED RL MODEL -------------------------- #
    SAVEDIR = "/home/alex/Documents/METR4911/ReLOaD/reload/simplesim/saved_models"
    rl_model = SAC.load(f"{SAVEDIR}/best_model.zip")


    # -------------------------------- LOAD YOLO --------------------------------- #
    vision_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    

    # -------------------------------- RUN TESTING ------------------------------- #
    NUM_EPISODES = 10 if not short_exec else 1
    MAX_TIMESTEPS = 100

    # img = np.zeros( (512,512,3), dtype=np.uint8)
    # img[0, 0, 0] = 255
    # print(np.shape(img))

    # plt.figure()
    # plt.imshow(img, interpolation='nearest')
    # plt.show() 
    # # quit()

    for j in range(NUM_EPISODES):
        print("Resetting environment")
        obs, reward, done, info = env.reset()
        ep_reward = 0
        
        for i in range(MAX_TIMESTEPS):
            with Profiler("Environment action step"):            
                # -------------------------------- GET ACTION -------------------------------- #
                if i == 0:
                    # Pick a random action cos for some reason the obs returned by env.reset() is bugged
                    action = env.action_space.sample()

                else:
                    action = env.action_space.sample()

                    # Get image from igibson
                    img = obs["rgb"]

                    # Visualise image
                    print(np.shape(img))
                    print(img)
                    # plt.figure()
                    # plt.imshow(img)
                    # plt.show() 
                    plt.imsave(f"current_step.jpg", img)

                    # Run YOLO inference in image
                    detections = vision_model(img)
                    # Visualise YOLO output (optional)
                    # print(f"detections: {info['detections']}")
                    # detections.save(f"step_{i}")  # or .show()

                    # Combine the YOLO results and position data into the standardises observation vector format
                    obs = create_observation_vector(detections, env)

                    # Run the RL model on the observation vector to get the action
                    action, _states = rl_model.predict(obs)

                    v, omega = action
                    
                    # # F/B
                    # self.keypress_mapping["i"] = {"idx": info["start_idx"] + 0, "val": 0.2}
                    # self.keypress_mapping["k"] = {"idx": info["start_idx"] + 0, "val": -0.2}
                    # # L/R
                    # self.keypress_mapping["l"] = {"idx": info["start_idx"] + 1, "val": 0.1}
                    # self.keypress_mapping["j"] = {"idx": info["start_idx"] + 1, "val": -0.1}

                    action_transformed = [float(v), float(-omega)]
                    action = action_transformed
                    print(f"Action: {action}")

                # ----------------------------------- STEP ----------------------------------- #
                # Step the iGibson environment with the selected action
                obs, reward, done, info = env.step(action)

                # Print out the results of this loop (optional)
                print(f"Reward: {reward}\n")

                ep_reward += reward

                # print(f"Observation: {obs}")
                # print(type(obs))
                # print(len(obs))
                # print(obs.keys())
                print("\n\n\n")

                if done:
                    print("Episode finished after {} timesteps".format(i + 1))
                    break
        
        print(f"Total Ep Reward: {ep_reward}")
    
    env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()