# ReLOaD
#
# train_agent_simplesim.py
#
# Trains a dqn policy on the simplesim env, saving checkpoints along the way
# and exporting the final trained policy to file.
#
# Alex Nichoson
# 19/07/2023


# ---------------------------------------------------------------------------- #
#                                    IMPORTS                                   #
# ---------------------------------------------------------------------------- #

from reload.simplesim.gym import SimpleSimGym
import reload.utils
import reload.eval
import reload.agents

import matplotlib.pyplot as plt
import numpy as np
import reverb
import tensorflow as tf
import os
import copy
from datetime import datetime

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import tf_py_environment
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents import PPOAgent
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import value_network


# ---------------------------------------------------------------------------- #
#                                   FUNCTIONS                                  #
# ---------------------------------------------------------------------------- #


def create_envs(py_env):
    """
    Gets the test and train tf py envs for the trainer instance
    """
    # self.py_env = py_env

    # Instantiate two environments: one for training and one for evaluation.
    train_py_env = copy.deepcopy(py_env)
    eval_py_env = copy.deepcopy(py_env)
    train_env = tf_py_environment.TFPyEnvironment(
        train_py_env,
        check_dims=True,
    )
    eval_env = tf_py_environment.TFPyEnvironment(
        eval_py_env,
        check_dims=True,
    )
    return (train_env, eval_env)


def setup_data_collection(
    env,
    agent,
    replay_buffer_max_length,
    collect_steps_per_iteration,
    batch_size,
    verbose=False,
):
    """
    Sets up the training data collection pipeline including replay buffer, data
    collection driver, and dataset access via tf dataset iterator.
    """

    # Setup replay buffer
    table_name = "uniform_table"
    replay_buffer_signature = tensor_spec.from_spec(agent.collect_data_spec)
    replay_buffer_signature = tensor_spec.add_outer_dim(replay_buffer_signature)

    table = reverb.Table(
        table_name,
        max_size=replay_buffer_max_length,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=replay_buffer_signature,
    )

    reverb_server = reverb.Server([table])

    replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
        agent.collect_data_spec,
        table_name=table_name,
        sequence_length=2,
        local_server=reverb_server,
    )

    rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
        replay_buffer.py_client, table_name, sequence_length=2
    )

    # Create a driver to collect experience.
    collect_driver = py_driver.PyDriver(
        env,
        py_tf_eager_policy.PyTFEagerPolicy(agent.collect_policy, use_tf_function=True),
        [rb_observer],
        # max_episodes=2,
        max_steps=collect_steps_per_iteration,
    )

    collect_driver.run(env.reset())

    """The replay buffer is now a collection of Trajectories."""

    # For the curious:
    # Uncomment to peel one of these off and inspect it.
    # iter(replay_buffer.as_dataset()).next()

    """The agent needs access to the replay buffer. This is provided by creating an iterable `tf.data.Dataset` pipeline which will feed data to the agent.

    Each row of the replay buffer only stores a single observation step. But since the DQN Agent needs both the current and next observation to compute the loss, the dataset pipeline will sample two adjacent rows for each item in the batch (`num_steps=2`).

    This dataset is also optimized by running parallel calls and prefetching data.
    """

    # Dataset generates trajectories with shape [Bx2x...]
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2
    ).prefetch(3)

    iterator = iter(dataset)

    if verbose == True:
        print(iterator)

    return rb_observer, iterator, collect_driver, replay_buffer


def evaluate_random_baseline(
    train_py_env,
    eval_env,
    num_eval_episodes,
    rb_observer,
    initial_collect_steps,
):
    """
    Evaluates the performance of a tf random agent in the environment over a 
    number of episodes to stablish a performance baseline before training.

    Some good hyperparameter starting points: 
    - initial_collect_steps = 100
    - num_eval_episodes = 10
    """


    print("\nRunning baseline random policy...")

    random_policy = random_tf_policy.RandomTFPolicy(
        tensor_spec.add_outer_dim(eval_env.time_step_spec(), 1),
        tensor_spec.add_outer_dim(eval_env.action_spec(), 1),
    )

    avg_return = reload.eval.compute_avg_return(
        eval_env, random_policy, num_eval_episodes
    )

    py_driver.PyDriver(
        train_py_env,
        py_tf_eager_policy.PyTFEagerPolicy(random_policy, use_tf_function=True),
        [rb_observer],
        max_steps=initial_collect_steps,
    ).run(train_py_env.reset())
    
    avg_return = 0
    print(f"Finished evaluating baseline policy. Avg Return: {avg_return}\n")

    return avg_return


def train_agent(
    agent,
    py_env,
    eval_env,
    save_dir,
    resume_checkpoint=False,
    num_iterations=10000,
    # initial_collect_steps=100,
    collect_steps_per_iteration=1,
    replay_buffer_max_length=100000,
    batch_size=64,
    # log_interval=200,
    num_eval_episodes=10,
    eval_interval=5000,
):
    """
    Trains the agent.

    Two things must happen during the training loop:
    -   collect data from the environment
    -   use that data to train the agent's neural network(s)

    This example also periodicially evaluates the policy and prints the current score.
    """

    # ------------------------------- Replay Buffer ------------------------------ #

    rb_observer, iterator, collect_driver, replay_buffer = setup_data_collection(
        py_env, agent, replay_buffer_max_length, collect_steps_per_iteration, batch_size
    )

    # -------------------------- Setup Checkpoint Saver -------------------------- #

    checkpoint_dir = os.path.join(save_dir, "checkpoint")
    # Checkpointer
    train_checkpointer = common.Checkpointer(
        ckpt_dir=checkpoint_dir,
        max_to_keep=5,
        agent=agent,
        policy=agent.policy,
        replay_buffer=replay_buffer,
        global_step=agent.train_step_counter,
    )

    # ------------------- Restore from Checkpoint (if desired) ------------------- #
    if resume_checkpoint:
        train_checkpointer.initialize_or_restore()

    # -------------------------- Run the actual training ------------------------- #

    # (Optional) Optimize by wrapping some of the code in a graph using TF function
    agent.train = common.function(agent.train)

    # Reset the train step.
    agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = reload.eval.compute_avg_return(eval_env, agent.policy, 1)
    returns = [avg_return]

    # Setup PolicySaver
    tf_policy_saver = policy_saver.PolicySaver(agent.policy)

    # Reset the environment.
    time_step = py_env.reset()

    log_interval = eval_interval // 10
    for _ in range(num_iterations):
        # Collect a few steps and save to the replay buffer.
        time_step, _ = collect_driver.run(time_step)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, _ = next(iterator)

        # print(f"agent.train(experience): {agent.train(experience)}")
        train_loss = agent.train(experience).loss

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print("step = {0:,}: \033[91mloss = {1:,}\033[00m".format(step, train_loss))
            # print(f"\n            obs = {time_step.observation}")

        if step % eval_interval == 0:
            avg_return = reload.eval.compute_avg_return(
                eval_env, agent.policy, num_eval_episodes
            )
            print("step = {0:,}: \033[92mAverage Return = {1}\033[00m, Average Return / Step = {2}".format(step, avg_return, avg_return/2000))
            returns.append(avg_return)

            # Save a checkpoint
            train_checkpointer.save(agent.train_step_counter)

    # Save the finished policy as a SavedPolicy
    print(f"Saving policy ...")
    policy_dir = os.path.join(save_dir, "policy")
    tf_policy_saver.save(policy_dir)
    print(f"\nTrained policy saved to: {policy_dir}\n")

    return returns


# ---------------------------------------------------------------------------- #
#                                     MAIN                                     #
# ---------------------------------------------------------------------------- #

if __name__ == "__main__":
    # ------------------------------ Hyperparameters ----------------------------- #
    # Trainer
    num_iterations = 40000  # @param {type:"integer"}
    eval_interval = 8000  # @param {type:"integer"}
    num_eval_episodes = 10

    # Env
    STARTING_BUDGET = 2000
    NUM_TARGETS = 3
    PLAYER_FOV = 60

    # Agent
    LEARNING_RATE = 1e-3  # @param {type:"number"}

    # Saving
    SAVE_PARENT_DIR = "saved_models"  # parent directory where models are saved

    # -------------------------------- Environment ------------------------------- #

    # Instantiate two environments: one for training and one for evaluation.
    py_env = SimpleSimGym(STARTING_BUDGET, NUM_TARGETS, PLAYER_FOV, visualize=False)
    train_env, eval_env = create_envs(py_env)

    # View Env Specs
    reload.utils.show_env_summary(py_env)

    # ----------------------------------- Agent ---------------------------------- #

    # Create fresh DQN agent
    agent = reload.agents.get_dqn_agent(
        py_env, train_env, verbose=True, learning_rate=LEARNING_RATE
    )

    # --------------------------------- Training --------------------------------- #
    # Saving
    env_name = py_env.__class__.__name__
    agent_name = agent.__class__.__name__
    date_str = datetime.today().strftime("%Y_%m_%dT%H:%M")
    model_name = f"{env_name}-{agent_name}-{num_iterations//1000}k-{date_str}"
    save_dir = f"{SAVE_PARENT_DIR}/{model_name}"  # dirs to save checkpoints

    returns = train_agent(
        agent,
        py_env,
        eval_env,
        save_dir,
        num_iterations=num_iterations,
        eval_interval=eval_interval,
        num_eval_episodes = num_eval_episodes,
        resume_checkpoint = False
    )
    # --------------------------- Visualise Performance -------------------------- #

    # Visualize the training progress
    reload.utils.show_training_graph(returns, num_iterations, model_name)
