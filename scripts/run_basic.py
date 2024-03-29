# ReLOaD Simple Simulator
# 
# run_basic.py
# 
# Runs a basic agent on the non-gym SimpleSim environment
# 
# Alex Nichoson
# 27/05/2023

from env import SimpleSim
import pygame
import numpy as np
import time

# ---------------------------------------------------------------------------- #
#                                  GLOBAL VARS                                 #
# ---------------------------------------------------------------------------- #

STARTING_BUDGET = 2000
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

        # Get the reward
        last_reward = game.get_reward()

        # Get the agent's action
        last_conf_sum = np.sum(last_reward)
        current_conf_sum = np.sum(state["current_confidences"])
        if (current_conf_sum > last_conf_sum):
            action = 1
        elif (current_conf_sum < last_conf_sum):
            action = 3
        else:
            action = None

        # Get optional additional user action
        game.perform_action_interactive()

        # Step the game engine
        time.sleep(0.02)
        game.step(action)

    pygame.quit()

    