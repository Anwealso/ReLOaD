# ReLOaD Simple Simulator
# 
# env.py
# 
# Runs a PPOAgent over multiple episodes of the simplesim env
# 
# Alex Nichoson
# 19/06/2023

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import reverb

from tf_agents.agents import PPOAgent
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import value_network

from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common


from reload.simplesim.env_gym import GameEnv


def compute_avg_return(environment, policy, num_episodes=10):
    # TODO: Replace with lib inbuilt way
    # See also the metrics module for standard implementations of different metrics.
    # https://github.com/tensorflow/agents/tree/master/tf_agents/metrics

    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            # print(f"time_step: {time_step}")
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

def plot_returns(iterations, returns):
    iterations = range(0, num_iterations + 1, eval_interval)
    plt.plot(iterations, returns)
    plt.ylabel('Average Return')
    plt.xlabel('Iterations')
    plt.ylim(top=250)


def get_dqn_agent(env):
    # --------------------------------- DQN AGENT -------------------------------- #
    fc_layer_params = (100, 50)
    action_tensor_spec = tensor_spec.from_spec(env.action_spec())
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

    # Define a helper function to create Dense layers configured with the right
    # activation and kernel initializer.
    def dense_layer(num_units):
        return tf.keras.layers.Dense(
            num_units,
            activation=tf.keras.activations.relu,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0, mode='fan_in', distribution='truncated_normal'))

    # QNetwork consists of a sequence of Dense layers followed by a dense layer
    # with `num_actions` units to generate one q_value per available action as
    # its output.
    dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
    q_values_layer = tf.keras.layers.Dense(
        num_actions,
        activation=None,
        kernel_initializer=tf.keras.initializers.RandomUniform(
            minval=-0.03, maxval=0.03),
        bias_initializer=tf.keras.initializers.Constant(-0.2))
    q_net = sequential.Sequential(dense_layers + [q_values_layer])

    """Now use `tf_agents.agents.dqn.dqn_agent` to instantiate a `DqnAgent`. In addition to the `time_step_spec`, `action_spec` and the QNetwork, the agent constructor also requires an optimizer (in this case, `AdamOptimizer`), a loss function, and an integer step counter."""

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    train_step_counter = tf.Variable(0)

    agent = dqn_agent.DqnAgent(
        env.time_step_spec(),
        env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter)

    agent.initialize()


def get_ppo_agent(env):
    # --------------------------------- PPO AGENT -------------------------------- #
    print(env.observation_spec())
    actor_net = actor_distribution_network.ActorDistributionNetwork(
            env.observation_spec(),
            env.action_spec(),
            preprocessing_combiner=tf.keras.layers.Concatenate(axis=1))
    
    value_net = value_network.ValueNetwork(
            env.observation_spec(),
            preprocessing_combiner=tf.keras.layers.Concatenate(axis=1))


    # Setup the agent / policy
    agent = PPOAgent(
            time_step_spec=env.time_step_spec(),
            action_spec=env.action_spec(),
            actor_net=actor_net,
            value_net=value_net)
    agent.initialize()

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)

    # Reset the train step.
    agent.train_step_counter.assign(0)

    return agent


def show_env_summary(env):
    print("Observation spec: {} \n".format(env.observation_spec()))
    print("Action spec: {} \n".format(env.action_spec()))
    print("Time step spec: {} \n".format(env.time_step_spec()))


if __name__ == "__main__":
    # ------------------------------ HYPERPARAMETERS ----------------------------- #

    MAX_TIMESTEPS = 100
    STARTING_BUDGET = 20
    NUM_TARGETS = 8
    PLAYER_FOV = 90
    NUM_EPISODES = 5

    # num_iterations = 20000 # @param {type:"integer"}
    num_iterations = 20 # @param {type:"integer"}

    initial_collect_steps = 20  # @param {type:"integer"}
    collect_steps_per_iteration =   1# @param {type:"integer"}
    replay_buffer_max_length = 100000  # @param {type:"integer"}

    batch_size = 64  # @param {type:"integer"}
    learning_rate = 1e-3  # @param {type:"number"}
    log_interval = 200  # @param {type:"integer"}

    num_eval_episodes = 10  # @param {type:"integer"}
    eval_interval = 1000  # @param {type:"integer"}


    # -------------------------------- ENVIRONMENT ------------------------------- #

    # Setup the environment
    env = GameEnv(MAX_TIMESTEPS, STARTING_BUDGET, NUM_TARGETS, PLAYER_FOV)

    env = tf_py_environment.TFPyEnvironment(env)
    
    # Reset the env
    time_step = env.reset()

    show_env_summary(env) # Display environment specs


    # ----------------------------------- AGENT ---------------------------------- #

    agent = get_ppo_agent(env)


    # ------------------------------- REPLAY BUFFER ------------------------------ #

    # Setup replay buffer for training data collection
    table_name = 'uniform_table'
    replay_buffer_signature = tensor_spec.from_spec(
        agent.collect_data_spec)
    replay_buffer_signature = tensor_spec.add_outer_dim(
        replay_buffer_signature)

    table = reverb.Table(
        table_name,
        max_size=replay_buffer_max_length,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=replay_buffer_signature)

    reverb_server = reverb.Server([table])

    replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
        agent.collect_data_spec,
        table_name=table_name,
        sequence_length=2,
        local_server=reverb_server)

    rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
        replay_buffer.py_client,
        table_name,
        sequence_length=2)

    # Create a driver to collect experience during the the training loop
    collect_driver = py_driver.PyDriver(
        env,
        py_tf_eager_policy.PyTFEagerPolicy(
        agent.collect_policy, use_tf_function=True),
        [rb_observer],
        max_steps=collect_steps_per_iteration)

    # ------------------------------ DATA COLLECTION ----------------------------- #

    # Wrap the replay buffer in a TFDataset so the agent can access it
    # Dataset generates trajectories with shape [Bx2x...]
    # This dataset is also optimized by running parallel calls and prefetching data.
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=2).prefetch(3)
    iterator = iter(dataset)


    # ---------------------- TODO: Figure out what this does --------------------- #

    # # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(env, agent.policy, num_eval_episodes)
    returns = [avg_return]

    # Reset the env
    time_step = env.reset()


    # ------------------------------- TRAINING LOOP ------------------------------ #

    for i in range(num_iterations):
        print("#0")
        # Collect a few steps and save to the replay buffer.
        time_step, _ = collect_driver.run(time_step)
        print("#1")

        # # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = dataset
        print("#2")
        train_loss = agent.train(experience).loss
        # train_loss = 8
        print("#3")

        step = agent.train_step_counter.numpy()
        # print("4")

        # if step % log_interval == 0:
        #     print(f'step = {step}: loss = {train_loss}')

        if step % eval_interval == 0:
            avg_return = compute_avg_return(env, agent.policy, num_eval_episodes)
            print(f'step = {i}: Average Return = {avg_return}')
            returns.append(avg_return)

    print("birdabo")

