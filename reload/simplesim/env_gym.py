# ReLOaD Simple Simulator
# 
# env.py
# 
# A gym wrapped version of the pygame SimpleSim environment
# 
# When run standalone, executes a simple random policy on the gym-wrapped 
# tf PyEnvironment env.
# 
# Alex Nichoson
# 27/05/2023

import pygame
import math
import random
import numpy as np
import time
import abc
from reload.simplesim.env_pygame import Game
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts


# ---------------------------------------------------------------------------- #
#                                    CLASSES                                   #
# ---------------------------------------------------------------------------- #

class GameEnv(py_environment.PyEnvironment):
    """
    A gym wrapper for our simple simulator environment
    """
    def __init__(self, max_timesteps, starting_budget, num_targets, player_fov):
        # Actions: 0, 1, 2, 3 for F, B, L, R
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')
        
        # State: 
        # Num previous samples
        timestep_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=max_timesteps, name='timestep')
        # Remaining budget
        budget_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=max_timesteps, name='budget')
        # Avg of previous confidences
        avg_conf_spec = array_spec.BoundedArraySpec(shape=(8, 1), dtype=np.float32, minimum=0, name='avg_conf')
        # Current object confidences
        curr_conf_spec = array_spec.BoundedArraySpec(shape=(8, 1), dtype=np.float32, minimum=0, name='curr_conf')
        self._observation_spec = (timestep_spec, budget_spec, avg_conf_spec, curr_conf_spec)
        # self._observation_spec = array_spec.BoundedArraySpec(
        #     shape=(1,), dtype=np.int32, minimum=0, name='observation')

        self.game = Game(starting_budget, num_targets, player_fov)

        # self._state = 0
        # self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        # self._state = 0
        # self._episode_ended = False
        self.game.reset()

        observation = (self.game.count, self.game.budget, self.game.avg_confidences, self.game.confidences)

        return ts.restart(observation)

    def _step(self, action):

        if self.game.gameover:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        # Step the game
        if action in [0, 1, 2, 3]:
            self.game.step(action)
        else:
            raise ValueError('`action` should be 0, 1, 2, or 3.')

        observation = (self.game.count, self.game.budget, self.game.avg_confidences, self.game.confidences)
        if self.game.gameover:
            # Reward only given at the end of the episode
            reward = self.game.get_reward()
            return ts.termination(observation, reward)
        else:
            # No continuous rewards recieved at each timestep
            return ts.transition(observation, reward=0.0, discount=1.0)


# ---------------------------------------------------------------------------- #
#                                     MAIN                                     #
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Hyperparameters
    MAX_TIMESTEPS = 100
    STARTING_BUDGET = 2000
    NUM_TARGETS = 8
    PLAYER_FOV = 90

    get_new_action = np.array(0, dtype=np.int32)

    env = GameEnv(MAX_TIMESTEPS, STARTING_BUDGET, NUM_TARGETS, PLAYER_FOV)
    time_step = env.reset()
    print(f"time_step:{time_step}")
    cumulative_reward = time_step.reward

    for _ in range(100):
        time_step = env.step(get_new_action)
        print(f"time_step:{time_step}, time_step_reward:{time_step.reward}")
        cumulative_reward += time_step.reward

    print(f"time_step:{time_step}")
    cumulative_reward += time_step.reward
    print('Final Reward = ', cumulative_reward)
