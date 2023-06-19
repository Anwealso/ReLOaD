# ReLOaD Simple Simulator
# 
# run_basic.py
# 
# Runs a basic agent on the non-gym SimpleSim environment
# 
# Alex Nichoson
# 27/05/2023

from SimpleSim import SimpleSim
import pygame
import numpy as np
import time

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts


# ---------------------------------------------------------------------------- #
#                                  GLOBAL VARS                                 #
# ---------------------------------------------------------------------------- #

STARTING_BUDGET = 100
NUM_TARGETS = 8
PLAYER_FOV = 90

# ---------------------------------------------------------------------------- #
#                                     MAIN                                     #
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    """
    Run a basic handcrafted agent on the env (with optional interactive user 
    input)
    """
    game = SimpleSim(STARTING_BUDGET, NUM_TARGETS, PLAYER_FOV)
    state = {}
    action = (None, None)
    last_reward = []

    while game.run:
        # Get the games state
        state = game.get_state()

        # Get the agent's action
        last_conf_sum = np.sum(last_reward)
        current_conf_sum = np.sum(state["current_confidences"])
        if (current_conf_sum > last_conf_sum):
            action = ("F", "None")
        elif (current_conf_sum < last_conf_sum):
            action = ("B", "None")
        else:
            action = (None, None)

        # Perform the agents action
        game.perform_action(action)

        # Get optional additional user action
        game.perform_action_interactive()

        # Get the reward
        last_reward = game.get_reward()
        
        # Step the game engine
        time.sleep(0.1)
        game.step()

    pygame.quit()