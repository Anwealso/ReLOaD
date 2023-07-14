# ReLOaD Simple Simulator
#
# run_ppo.py
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


from reload.simplesim.gym import SimpleSimGym


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
    plt.ylabel("Average Return")
    plt.xlabel("Iterations")
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
                scale=2.0, mode="fan_in", distribution="truncated_normal"
            ),
        )

    # QNetwork consists of a sequence of Dense layers followed by a dense layer
    # with `num_actions` units to generate one q_value per available action as
    # its output.
    dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
    q_values_layer = tf.keras.layers.Dense(
        num_actions,
        activation=None,
        kernel_initializer=tf.keras.initializers.RandomUniform(
            minval=-0.03, maxval=0.03
        ),
        bias_initializer=tf.keras.initializers.Constant(-0.2),
    )
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
        train_step_counter=train_step_counter,
    )

    agent.initialize()


def get_ppo_agent(env):
    # --------------------------------- PPO AGENT -------------------------------- #
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        env.observation_spec(),
        env.action_spec(),
        preprocessing_combiner=tf.keras.layers.Concatenate(axis=1),
    )

    value_net = value_network.ValueNetwork(
        env.observation_spec(),
        preprocessing_combiner=tf.keras.layers.Concatenate(axis=1),
    )

    # Setup the agent / policy
    agent = PPOAgent(
        time_step_spec=env.time_step_spec(),
        action_spec=env.action_spec(),
        actor_net=actor_net,
        value_net=value_net,
    )
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


def collect_step(environment, policy):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer
    replay_buffer.add_batch(traj)


if __name__ == "__main__":
    # ------------------------------ HYPERPARAMETERS ----------------------------- #

    # MAX_TIMESTEPS = 100
    STARTING_BUDGET = 50
    NUM_TARGETS = 8
    PLAYER_FOV = 90
    NUM_EPISODES = 5

    # num_iterations = 20000 # @param {type:"integer"}
    num_iterations = 40  # @param {type:"integer"}

    initial_collect_steps = 10  # @param {type:"integer"}
    # collect_steps_per_iteration = 1  # @param {type:"integer"}
    collect_episodes_per_iteration = 1  # @param {type:"integer"}
    replay_buffer_max_length = 100000  # @param {type:"integer"}

    batch_size = 64  # @param {type:"integer"}
    learning_rate = 1e-3  # @param {type:"number"}
    log_interval = 200  # @param {type:"integer"}

    num_eval_episodes = 10  # @param {type:"integer"}
    eval_interval = 1000  # @param {type:"integer"}

    # -------------------------------- ENVIRONMENT ------------------------------- #

    # Setup the environment
    env = SimpleSimGym(STARTING_BUDGET, NUM_TARGETS, PLAYER_FOV)

    env = tf_py_environment.TFPyEnvironment(env)

    # Reset the env
    time_step = env.reset()
    # print("### time_step")
    # print(time_step)

    show_env_summary(env)  # Display environment specs

    # ----------------------------------- AGENT ---------------------------------- #

    agent = get_ppo_agent(env)

    # ------------------------------- REPLAY BUFFER ------------------------------ #

    # Setup replay buffer for training data collection
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

    # Create a driver to collect experience during the the training loop
    def collect_episode(environment, policy, num_episodes):
        driver = py_driver.PyDriver(
            environment,
            py_tf_eager_policy.PyTFEagerPolicy(policy, use_tf_function=True),
            [rb_observer],
            max_episodes=num_episodes,
        )
        initial_time_step = environment.reset()
        driver.run(initial_time_step)

    # ----------------------------- PREP FOR TRAINING ---------------------------- #

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)

    # Reset the train step
    agent.train_step_counter.assign(0)

    # # Evaluate the agent's policy once before training.
    # avg_return = compute_avg_return(env, agent.policy, num_eval_episodes)
    # returns = [avg_return]
    # print(f'Pre-train Evaluation: Average Return = {avg_return}')

    # ------------------------------- TRAINING LOOP ------------------------------ #

    returns = []
    for iteration_index in range(num_iterations):
        # Collect a few episodes using collect_policy and save to the replay buffer.
        collect_episode(env, agent.collect_policy, collect_episodes_per_iteration)
        print(f"iteration_index: {iteration_index}")

        # Use data from the buffer and update the agent's network.
        iterator = iter(replay_buffer.as_dataset(sample_batch_size=1))

        # TODO: Fix error being thrown when calling next on the iterator
        print()
        print()
        print(replay_buffer.as_dataset(sample_batch_size=1))
        print()
        print(next(iterator))
        # trajectories, _ = next(iterator)
        # train_loss = agent.train(experience=trajectories)

        # replay_buffer.clear()

        step = agent.train_step_counter.numpy()

        # if step % log_interval == 0:
        #     print('step = {0}: loss = {1}'.format(step, train_loss.loss))

        if step % eval_interval == 0:
            avg_return = compute_avg_return(env, agent.policy, num_eval_episodes)
            print("step = {0}: Average Return = {1}".format(step, avg_return))
            returns.append(avg_return)
