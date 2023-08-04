# ReLOaD Simple Simulator
#
# simplesim/gym.py
#
# Gym wrapped version of ReLOad SimpleSim environment
#
# Alex Nichoson
# 27/05/2023

from reload.simplesim.env import SimpleSim
import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts


# ---------------------------------------------------------------------------- #
#                                    CLASSES                                   #
# ---------------------------------------------------------------------------- #


class SimpleSimGym(py_environment.PyEnvironment):
    """
    A gym wrapper for our simple simulator environment
    """

    def __init__(self, starting_budget, num_targets, player_fov, visualize=True):
        # batch_size = 1
        # MAX_TIMESTEPS = 100

        # Internal State:
        self.game = SimpleSim(
            starting_budget, num_targets, player_fov, visualize=visualize
        )

        # Actions: 0, 1, 2, 3 for F, B, L, R
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=2, name="action"
        )
        # self.action_cost = action_cost

        # Observations (visible state):
        observation_shape = np.shape(self.get_observation())[0]
        obs_spec = array_spec.BoundedArraySpec(
            shape=(observation_shape,),
            dtype=np.float32,
            minimum=0,
            name="observation",
        )
        self._observation_spec = obs_spec


    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def get_observation(self):
        observation = np.squeeze(
            np.concatenate(
                [
                    np.array([[self.game.budget]], dtype=np.float32),
                    np.float32(self.game.avg_confidences), # average confidences
                    np.float32(self.game.current_confidences), # confidences at current timestep
                    # np.array([[self.game.robot.x, self.game.robot.y, self.game.robot.angle]], dtype=np.float32), # robot coords
                    # np.array([[self.game.targets[0].x, self.game.targets[0].y]], dtype=np.float32), # target coords
                ],
                axis=0,
            ),
            axis=1,
        )

        # np.array([[self.game.count]], dtype=np.float32),
        # np.float32(self.game.confidences),
        
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
        if action in [0, 1, 2, 3, 4]:
            self.game.step(action)
        else:
            raise ValueError("`action` should be 0, 1, 2, 3, or 4.")

        if self.game.gameover:
            # Reward only given at the end of the episode
            reward = self.game.get_reward()
            # if action != 0:
                # reward -= self.action_cost
            return ts.termination(self.get_observation(), reward=reward)
        else:
            # Continuous rewards recieved at each timestep
            reward = self.game.get_reward()
            # if action != 0:
            #     reward -= self.action_cost
            return ts.transition(
                self.get_observation(), 
                reward=reward, 
                # discount=1.0
            )
