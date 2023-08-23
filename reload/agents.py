# ReLOaD Simple Simulator
# 
# agents.py
# 
# Holds builder functions for major RL agents
# 
# Alex Nichoson
# 04/07/2023

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



def get_dqn_agent(env, train_env, fc_layer_params=(100,50), verbose=False, learning_rate=None):
    """
    Creates a DQN agent for the give env and returns it.

    ## Agent

    The algorithm used to solve an RL problem is represented by an `Agent`.
    TF-Agents provides standard implementations of a variety of `Agents`,
    including:

    -   [DQN](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) (used in this tutorial)
    -   [REINFORCE](https://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf)
    -   [DDPG](https://arxiv.org/pdf/1509.02971.pdf)
    -   [TD3](https://arxiv.org/pdf/1802.09477.pdf)
    -   [PPO](https://arxiv.org/abs/1707.06347)
    -   [SAC](https://arxiv.org/abs/1801.01290)

    The DQN agent can be used in any environment which has a discrete action
    space.

    At the heart of a DQN Agent is a `QNetwork`, a neural network model that can
    learn to predict `QValues` (expected returns) for all actions, given an
    observation from the environment.

    We will use `tf_agents.networks.` to create a `QNetwork`. The network will
    consist of a sequence of `tf.keras.layers.Dense` layers, where the final
    layer will have 1 output for each possible action.

    ## Policies

    A policy defines the way an agent acts in an environment. Typically, the goal
    of reinforcement learning is to train the underlying model until the policy
    produces the desired outcome.

    Agents contain two policies:

    -   `agent.policy` — The main policy that is used for evaluation and
            deployment.
    -   `agent.collect_policy` — A second policy that is used for data
            collection.

    Policies can be created independantly of Agents. To get an action from a
    policy, call the `policy.action(time_step)` method. The `time_step` contains
    the observation from the environment. This method returns a `PolicyStep`,
    which is a named tuple with three components:

    -   `action` — the action to be taken (in this case, `0` or `1`)
    -   `state` — used for stateful (that is, RNN-based) policies
    -   `info` — auxiliary data, such as log probabilities of actions
    """

    # fc_layer_params = (100, 50)
    action_tensor_spec = tensor_spec.from_spec(env.action_spec())
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1
    print(f"num_actions: {num_actions}")

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

    """Now use `tf_agents.agents.dqn.dqn_agent` to instantiate a `DqnAgent`. 
    In addition to the `time_step_spec`, `action_spec` and the QNetwork, the 
    agent constructor also requires an optimizer (in this case, 
    `AdamOptimizer`), a loss function, and an integer step counter."""

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    train_step_counter = tf.Variable(0)

    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter,
    )

    agent.initialize()

    if verbose == True:
        # See a summary of the QNet model architecture
        print(f"Agent QNet Summary: {agent._q_network.summary()}")

    return agent


def get_ppo_agent(train_env, verbose=False):
    """
    Creates a PPO agent for the given env and returns it.
    """

    actor_net = actor_distribution_network.ActorDistributionNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
    )
    value_net = value_network.ValueNetwork(
        train_env.observation_spec(),
    )
    # Setup the agent / policy
    agent = PPOAgent(
        time_step_spec=train_env.time_step_spec(),
        action_spec=train_env.action_spec(),
        actor_net=actor_net,
        value_net=value_net,
    )

    agent.initialize()

    return agent
