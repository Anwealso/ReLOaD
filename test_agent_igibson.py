import torch
import time
from stable_baselines3 import SAC, PPO, DQN
from igibson.envs.igibson_env import iGibsonEnv
import reload

# ----------------------------- START IGIBSON SIM ---------------------------- #
env = iGibsonEnv()

# View Env Specs
reload.utils.show_env_summary(env)


# --------------------------- LOAD TRAINED RL MODEL -------------------------- #
SAVEDIR = "/home/alex/Documents/METR4911/ReLOaD/reload/simplesim/saved_models"
rl_model = SAC.load(f"{SAVEDIR}/best_model.zip")


# -------------------------------- START YOLO -------------------------------- #
vision_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


# -------------------------------- RUN TESTING ------------------------------- #

num_episodes = 10
obs = env.reset()

for i in range(num_episodes):
    done = False
    ep_reward = 0

    while not done:
        # Get image from igibson
        img = ...
        
        # Run YOLO inference in image
        results = vision_model(img)
        # # Visualise results
        # results.print()
        # results.save()  # or .show()

        # Get internal position data from iGibson
        ...

        # Combine the YOLO results and position data into the standardises observation vector format
        obs = ... 

        # Run the RL model on the observation vector to get the action
        action, _states = rl_model.predict(obs)

        # Step the iGibson environment with the selected action
        obs, reward, done, info = env.step(action)
        # env.render()

        # Print out the results of this loop
        # print(f"Observation: {obs}, Action: {action}, Reward: {reward}\n")

        ep_reward += reward

        if done:
            obs = env.reset()

    print(f"Total Ep Reward: {ep_reward}")