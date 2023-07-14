# ReLOaD Simple Simulator
# 
# env.py
# 
# Runs a RandomTFPolicy over multiple episodes of the simplesim env
# 
# Alex Nichoson
# 19/06/2023

import numpy as np
import tensorflow as tf
import tf_agents
from tf_agents.environments import tf_py_environment
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from reload.simplesim.env_gym import GameEnv


if __name__ == "__main__":
    # Hyperparameters
    MAX_TIMESTEPS = 100
    STARTING_BUDGET = 2000
    NUM_TARGETS = 8
    PLAYER_FOV = 90
    NUM_EPISODES = 5

    # Setup the environment
    env = GameEnv(MAX_TIMESTEPS, STARTING_BUDGET, NUM_TARGETS, PLAYER_FOV)
    tf_env = tf_py_environment.TFPyEnvironment(env)

    time_step = tf_env.reset()
    rewards = []
    steps = []

    # Setup the agent / policy
    policy = RandomTFPolicy(action_spec=tf_env.action_spec(), 
                                                          time_step_spec=tf_env.time_step_spec())

    # Run the episodes
    for _ in range(NUM_EPISODES):
        episode_reward = 0
        episode_steps = 0
        while not time_step.is_last():
            # Pick random action from 0,1,2,3
            action = policy.action(time_step)
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

    print('NUM_EPISODES:', NUM_EPISODES, 'num_steps:', num_steps)
    print('avg_length', avg_length, 'avg_reward:', avg_reward)