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
import tensorflow as tf
from tf_agents.environments import py_environment
import tf_agents.specs
from tf_agents.trajectories import time_step as ts
import math


class SimpleSimGym(py_environment.PyEnvironment):
    """
    A gym wrapper for our simple simulator environment
    """

    def __init__(self, starting_budget, num_targets, player_fov, visualize=True):
        # Internal State:
        self.game = SimpleSim(
            starting_budget, num_targets, player_fov, visualize=visualize
        )

        # Actions: 0, 1, 2, 3 for L, R, F, B
        self._action_spec = tf_agents.specs.BoundedTensorSpec(
            shape=(), dtype=np.int32, minimum=0, maximum=3, name="action"
        )

        # Observations (visible state):
        observation_shape = np.shape(self.get_observation())[0]
        self._observation_spec = tf_agents.specs.TensorSpec(
            shape=(observation_shape,),
            dtype=np.float32,
            # minimum=0,
            name="observation",
        )

        self._reward_spec = tf_agents.specs.TensorSpec(
            shape=(1,1),
            dtype=np.float32,
            # minimum=0,
            name="reward",
        )


    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def reward_spec(self):
        return self._reward_spec

    def get_observation(self):
        # target_rel_positions = []
        # for target in self.game.targets:
        #     rel_x = target.x - self.game.robot.x 
        #     norm_rel_x = rel_x / self.game.sw # normalise 
        #     target_rel_positions.append(norm_rel_x)
        #     rel_y = target.y - self.game.robot.y
        #     norm_rel_y = rel_y / self.game.sh # normalise
        #     target_rel_positions.append(norm_rel_y)

        observation = np.squeeze(
            np.concatenate(
                [
                    # np.array([[self.game.budget / self.game.starting_budget]], dtype=np.float32), # fraction of bugdet remaining
                    # np.float32(self.game.avg_confidences), # average confidences
                    # np.float32(self.game.current_confidences), # confidences at current timestep
                    # np.transpose(np.array([target_rel_positions], dtype=np.float32)), # target relative positions from robot
                    np.transpose(np.array([[self.game.robot.x, self.game.robot.y, self.game.robot.angle]], dtype=np.float32)), # target relative positions from robot
                ],
                axis=0,
            ),
            axis=1,
        )

        print(observation, end='\r')
        print("                                                  ", end='\r')
        
        return observation
    
    def get_reward(self, action):
        """
        The reward function for the RL agent
          - Agent gets positive reward for the current instantaneous confidence
            it achieved in the last time step
          - Agent receives negative reward for moving
        
        Agent will learn to maximise this instantaneous reward at each time 
        step.
        """
        norm_reward = ((self.game.robot.x + self.game.robot.y) / (self.game.sw + self.game.sh))
        reward = 100 * math.exp(5*(norm_reward-1)) # (y=100 e^{5(x-1)})

        # reward = np.sum(self.game.current_confidences)
        # if action != 0: # if action is not do-nothing
        #     reward -= 0.5

        reward = np.array(reward)
        reward = np.expand_dims(reward, axis=0)
        reward = np.expand_dims(reward, axis=1)
        reward = tf.convert_to_tensor(reward, dtype=np.float32)

        return reward

    def _reset(self):
        self.game.reset()
        return ts.restart(self.get_observation())

    def _step(self, action):
        if self.game.gameover:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        # Step the game
        self.game.step(action)

        self.game.perform_action_interactive()

        # Show info on scoreboard
        # self.game.set_scoreboard({"Reward": format(self.get_reward(action), ".2f")})
        # print(self.get_reward(action))
        self.game.set_scoreboard({"Reward": format(int(self.get_reward(action)[0][0]), ".2f"), "Observation": self.get_observation()})

        # Return reward
        if self.game.gameover:
            # End of episode case
            return ts.termination(
                self.get_observation(), 
                reward=self.get_reward(action)
            )
        else:
            # Mid-episode case
            return ts.transition(
                self.get_observation(), 
                reward=self.get_reward(action)
            )
