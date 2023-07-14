# ReLOaD Simple Simulator
# 
# simplesim/gym.py
# 
# Gym wrapped version of ReLOad SimpleSim environment
# 
# Alex Nichoson
# 27/05/2023

from reload.simplesim.env import SimpleSim
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
#                                    CLASSES                                   #
# ---------------------------------------------------------------------------- #

class SimpleSimGym(py_environment.PyEnvironment):
    """
    A gym wrapper for our simple simulator environment
    """
    def __init__(self, starting_budget, num_targets, player_fov):
        # Actions: 0, 1, 2, 3 for F, B, L, R
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')

        MAX_TIMESTEPS = 100
        
        # Observations (visible state): 
        obs_spec = (
            # Num previous samples
            array_spec.BoundedArraySpec(shape=(1, 1), dtype=np.float32, minimum=0, maximum=MAX_TIMESTEPS, name='count'),
            # Remaining budget
            array_spec.BoundedArraySpec(shape=(1, 1), dtype=np.float32, minimum=0, maximum=MAX_TIMESTEPS, name='budget'),
            # Avg of previous confidences
            array_spec.BoundedArraySpec(shape=(8, 1), dtype=np.float32, minimum=0, name='avg_conf'),
            # Current object confidences
            array_spec.BoundedArraySpec(shape=(8, 1), dtype=np.float32, minimum=0, name='curr_conf'),
        )

        self._observation_spec = obs_spec

        # Internal State:
        self.game = SimpleSim(starting_budget, num_targets, player_fov)

        # Timestep Fields: 
        self._discount_spec = array_spec.BoundedArraySpec(shape=(1, 1), dtype=np.float32, minimum=0, name='discount')
        self._reward_spec = array_spec.BoundedArraySpec(shape=(1, 1), dtype=np.float32, minimum=0, name='reward')

        # self._time_step_spec = ts.time_step_spec(self._observation_spec, self._reward_spec)

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec
    
    def discount_spec(self):
        return self._discount_spec
    
    def reward_spec(self):
        return self._reward_spec
    
    # def time_step_spec(self):
    #     return self._time_step_spec
    
    def get_observation(self):
        observation = (
            np.array([[self.game.count]], dtype=np.float32),
            np.array([[self.game.budget]], dtype=np.float32),
            # np.float32(self.game.count),
            # np.float32(self.game.budget),
            np.float32(self.game.avg_confidences),
            np.float32(self.game.confidences),
        )

        return observation

    def _reset(self):
        self.game.reset()
        
        return ts.restart(self.get_observation())

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

        if self.game.gameover:
            # Reward only given at the end of the episode
            reward = self.game.get_reward()
            return ts.termination(self.get_observation(), np.sum(reward))
        else:
            # Continuous rewards recieved at each timestep
            reward = self.game.get_reward()
            return ts.transition(self.get_observation(), reward=np.sum(reward), discount=1.0)
