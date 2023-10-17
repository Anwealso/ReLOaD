# ReLOaD Simple Simulator
# 
# run_random.py
# 
# Runs a random agent on multiple episodes of the SimpleSim environment
# 
# Alex Nichoson
# 27/05/2023


import numpy as np
import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.policies import random_tf_policy
from gym import SimpleSimGym


# ---------------------------------------------------------------------------- #
#                                HYPERPARAMETERS                               #
# ---------------------------------------------------------------------------- #
STARTING_BUDGET = 1000
NUM_TARGETS = 8
PLAYER_FOV = 90
NUM_EPISODES = 5


# ---------------------------------------------------------------------------- #
#                                 MAIN FUNCTION                                #
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    env = SimpleSimGym(STARTING_BUDGET, NUM_TARGETS, PLAYER_FOV)
    tf_env = tf_py_environment.TFPyEnvironment(env)

    random_policy = random_tf_policy.RandomTFPolicy(
        tf_env.time_step_spec(), tf_env.action_spec()
    )


    print("ENV SPEC ORIGINAL:")
    print(tf_env.time_step_spec())
    time_step = tf_env.reset()
    print("ENV SPECS AFTER RESET:")
    print(time_step)
    print("")
    
    rewards = []
    steps = []

    for _ in range(NUM_EPISODES):
        episode_reward = 0
        episode_steps = 0
        while not time_step.is_last():
            # Pick random action from 0,1,2,3
            # action = tf.random.uniform([1], 0, 4, dtype=tf.int32)

            print(f"time_step: {time_step}\n\n")
            action = random_policy.action(time_step)
            print(f"action: {action}\n\n")


            time_step = tf_env.step(action)
            episode_steps += 1
            episode_reward += time_step.reward.numpy()

        print("End of Episode")
        print(f"    Num Steps: {steps}")
        print(f"    Reward: {episode_reward}")
        print("")
        rewards.append(episode_reward)
        steps.append(episode_steps)
        time_step = tf_env.reset()

    num_steps = np.sum(steps)
    avg_length = np.mean(steps)
    avg_reward = np.mean(rewards)

    print("=======================================================")
    print('NUM_EPISODES:', NUM_EPISODES, 'num_steps:', num_steps)
    print('avg_length', avg_length, 'avg_reward:', avg_reward)
    print("=======================================================")
