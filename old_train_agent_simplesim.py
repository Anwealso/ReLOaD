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

def setup_data_collection(
    env, agent, replay_buffer_max_length, collect_steps_per_iteration, verbose=False
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
        max_steps=collect_steps_per_iteration,
    )

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


def evaluate_baseline(
    train_py_env,
    train_env,
    eval_env,
    num_eval_episodes,
    rb_observer,
    initial_collect_steps,
):
    print("\nRunning baseline random policy...")

    random_policy = random_tf_policy.RandomTFPolicy(
        tensor_spec.add_outer_dim(train_env.time_step_spec(), 1),
        tensor_spec.add_outer_dim(train_env.action_spec(), 1),
    )

    reload.eval.compute_avg_return(eval_env, random_policy, num_eval_episodes)

    py_driver.PyDriver(
        train_py_env,
        py_tf_eager_policy.PyTFEagerPolicy(random_policy, use_tf_function=True),
        [rb_observer],
        max_steps=initial_collect_steps,
    ).run(train_py_env.reset())

    print("Finished evaluating baseline policy.\n")


def train_agent(
    agent,
    eval_env,
    train_py_env,
    collect_driver,
    iterator,
    num_eval_episodes,
    num_iterations,
    log_interval,
    eval_interval,
    train_checkpointer,
    policy_dir
):
    """
    Trains the agent.

    Two things must happen during the training loop:
    -   collect data from the environment
    -   use that data to train the agent's neural network(s)

    This example also periodicially evaluates the policy and prints the current score.
    """

    # (Optional) Optimize by wrapping some of the code in a graph using TF function
    agent.train = common.function(agent.train)

    # Reset the train step.
    agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = reload.eval.compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    returns = [avg_return]

    # Setup PolicySaver
    tf_policy_saver = policy_saver.PolicySaver(agent.policy)

    # Reset the environment.
    time_step = train_py_env.reset()

    for _ in range(num_iterations):
        # Collect a few steps and save to the replay buffer.
        time_step, _ = collect_driver.run(time_step)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print("step = {0:,}: loss = {1}".format(step, train_loss))

        if step % eval_interval == 0:
            avg_return = reload.eval.compute_avg_return(eval_env, agent.policy, num_eval_episodes)
            print("step = {0:,}: Average Return = {1}".format(step, avg_return))
            returns.append(avg_return)

            # Save a checkpoint
            train_checkpointer.save(agent.train_step_counter)

    # Save the finished policy as a SavedPolicy
    print(f"Saving policy ...")
    tf_policy_saver.save(policy_dir)
    print(f"Trained policy saved to: {policy_dir}")

    return returns



# ---------------------------------------------------------------------------- #
#                                     MAIN                                     #
# ---------------------------------------------------------------------------- #

if __name__ == "__main__":
    # ------------------------------ Hyperparameters ----------------------------- #
    num_iterations = 10000  # @param {type:"integer"}

    initial_collect_steps = 100  # @param {type:"integer"}
    collect_steps_per_iteration = 1  # @param {type:"integer"}
    replay_buffer_max_length = 100000  # @param {type:"integer"}

    batch_size = 64  # @param {type:"integer"}
    learning_rate = 1e-3  # @param {type:"number"}
    log_interval = 200  # @param {type:"integer"}

    num_eval_episodes = 10  # @param {type:"integer"}
    eval_interval = 5000  # @param {type:"integer"}

    STARTING_BUDGET = 400
    NUM_TARGETS = 1
    PLAYER_FOV = 60

    save_dir = "saved_models" # dirs to save checkpoints
    checkpoint_dir = os.path.join(save_dir, 'checkpoint')
    policy_dir = os.path.join(save_dir, 'policy')

    # -------------------------------- Environment ------------------------------- #

    py_env = SimpleSimGym(STARTING_BUDGET, NUM_TARGETS, PLAYER_FOV, visualize=False)


    # Instantiate two environments: one for training and one for evaluation.
    train_py_env = SimpleSimGym(STARTING_BUDGET, NUM_TARGETS, PLAYER_FOV, visualize=False)
    eval_py_env = SimpleSimGym(STARTING_BUDGET, NUM_TARGETS, PLAYER_FOV, visualize=False)
    train_env = tf_py_environment.TFPyEnvironment(
        train_py_env,
        check_dims=True,
    )
    eval_env = tf_py_environment.TFPyEnvironment(
        eval_py_env,
        check_dims=True,
    )

    # View Env Specs
    reload.utils.show_env_summary(train_py_env)

    # ----------------------------------- Agent ---------------------------------- #

    # Create fresh agent
    agent = reload.agents.get_dqn_agent(train_py_env, train_env, verbose=True, learning_rate=learning_rate)
    # agent = reload.agents.get_ppo_agent(train_env, verbose=True)

    # Restore agent from checkpoint (if desired)
    # restore_checkpoint(agent)

    # ------------------------------- Replay Buffer ------------------------------ #

    rb_observer, iterator, collect_driver, replay_buffer = setup_data_collection(
        train_py_env, agent, replay_buffer_max_length, collect_steps_per_iteration
    )

    # Evaluate baseline (random policy)
    evaluate_baseline(
        train_py_env,
        train_env,
        eval_env,
        num_eval_episodes,
        rb_observer,
        initial_collect_steps,
    )

    # -------------------------- Setup Checkpoint Saver -------------------------- #
    # Checkpointer
    train_checkpointer = common.Checkpointer(
        ckpt_dir=checkpoint_dir,
        max_to_keep=3,
        agent=agent,
        policy=agent.policy,
        replay_buffer=replay_buffer,
        global_step=agent.train_step_counter
    )

    # ------------------- Restore from Checkpoint (if desired) ------------------- #
    # train_checkpointer.initialize_or_restore()


    # --------------------------------- Training --------------------------------- #
    returns = train_agent(
        agent,
        eval_env,
        train_py_env,
        collect_driver,
        iterator,
        num_eval_episodes,
        num_iterations,
        log_interval,
        eval_interval,
        train_checkpointer,
        # tf_policy_saver,
        policy_dir
    )

    # --------------------------- Visualise Performance -------------------------- #

    # Visualize the training progress
    reload.utils.show_training_graph(returns, num_iterations, eval_interval)
