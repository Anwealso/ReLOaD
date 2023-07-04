"""
Whole Episodes
"""

import numpy as np
import tensorflow as tf
from tf_agents.environments import tf_py_environment
from reload.simplesim.env_gym import GameEnv


if __name__ == "__main__":
    # Hyperparameters
    MAX_TIMESTEPS = 100
    STARTING_BUDGET = 2000
    NUM_TARGETS = 8
    PLAYER_FOV = 90

    env = GameEnv(MAX_TIMESTEPS, STARTING_BUDGET, NUM_TARGETS, PLAYER_FOV)
    tf_env = tf_py_environment.TFPyEnvironment(env)

    time_step = tf_env.reset()
    rewards = []
    steps = []
    num_episodes = 5

    for _ in range(num_episodes):
        episode_reward = 0
        episode_steps = 0
        while not time_step.is_last():
            # Pick random action from 0,1,2,3
            action = tf.random.uniform([1], 0, 4, dtype=tf.int32)
            print(action)
            time_step = tf_env.step(action)
            episode_steps += 1
            episode_reward += time_step.reward.numpy()
        rewards.append(episode_reward)
        steps.append(episode_steps)
        time_step = tf_env.reset()

    num_steps = np.sum(steps)
    avg_length = np.mean(steps)
    avg_reward = np.mean(rewards)

    print('num_episodes:', num_episodes, 'num_steps:', num_steps)
    print('avg_length', avg_length, 'avg_reward:', avg_reward)