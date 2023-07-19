# ---------------------------------------------------------------------------- #
#                                    IMPORTS                                   #
# ---------------------------------------------------------------------------- #

import base64

# import imageio
# import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image

# import pyvirtualdisplay
import reverb

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
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
from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents import PPOAgent
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import value_network

tf.version.VERSION


# ---------------------------------------------------------------------------- #
#                                   FUNCTIONS                                  #
# ---------------------------------------------------------------------------- #
def show_training_graph():
    """
    Use `matplotlib.pyplot` to chart how the policy improved during training.
    One iteration of `Cartpole-v0` consists of 200 time steps. The environment
    gives a reward of `+1` for each step the pole stays up, so the maximum return
    for one episode is 200. The charts shows the return increasing towards that
    maximum each time it is evaluated during training. (It may be a little
    unstable and not increase monotonically each time.)
    """

    iterations = range(0, num_iterations + 1, eval_interval)
    plt.plot(iterations, returns)
    plt.ylabel("Average Return")
    plt.xlabel("Iterations")
    plt.ylim(top=250)
    plt.imsave("training_graph.png")


def show_env_summary(env):
    """
    Prints Env Specs
    """

    print("Observation Spec:")
    print(env.time_step_spec().observation)

    print("Reward Spec:")
    print(env.time_step_spec().reward)

    print("Action Spec:")
    print(env.action_spec())

    time_step = env.reset()
    print("Time step:")
    print(time_step)

    action = np.array(1, dtype=np.int32)

    next_time_step = env.step(action)
    print("Next time step:")
    print(next_time_step)


def compute_avg_return(environment, policy, num_episodes=10):
    """
    Computes the average return of a policy, given the policy, environment, and
    a number of episodes.

    The most common metric used to evaluate a policy is the average return.
    The return is the sum of rewards obtained while running a policy in an
    environment for an episode. Several episodes are run, creating an average
    return.

    See also the metrics module for standard implementations of different metrics.
    https://github.com/tensorflow/agents/tree/master/tf_agents/metrics

    Running this computation on the `random_policy` shows a baseline performance
    in an environment.
    """

    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
            total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


def get_dqn_agent(env, train_env, verbose=False):
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

    fc_layer_params = (100, 50)
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


def get_ppo_agent(env, verbose=False):
    """
    Creates a PPO agent for the give env and returns it.
    """

    actor_net = actor_distribution_network.ActorDistributionNetwork(
        env.observation_spec(),
        env.action_spec(),
    )
    value_net = value_network.ValueNetwork(
        env.observation_spec(),
    )
    # Setup the agent / policy
    agent = PPOAgent(
        time_step_spec=env.time_step_spec(),
        action_spec=env.action_spec(),
        actor_net=actor_net,
        value_net=value_net,
    )

    agent.initialize()

    return agent


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

    return rb_observer, iterator, collect_driver


def evaluate_baseline(train_env, eval_env, num_eval_episodes, env, rb_observer, initial_collect_steps, train_py_env):
    print("\nRunning baseline random policy...")

    random_policy = random_tf_policy.RandomTFPolicy(
        tensor_spec.add_outer_dim(train_env.time_step_spec(), 1),
        tensor_spec.add_outer_dim(train_env.action_spec(), 1),
    )

    compute_avg_return(eval_env, random_policy, num_eval_episodes)

    py_driver.PyDriver(
        env,
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
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    returns = [avg_return]

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
            print("step = {0}: loss = {1}".format(step, train_loss))

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
            print("step = {0}: Average Return = {1}".format(step, avg_return))
            returns.append(avg_return)

    return returns


# ---------------------------------------------------------------------------- #
#                                     MAIN                                     #
# ---------------------------------------------------------------------------- #

if __name__ == "__main__":
    # ------------------------------ Hyperparameters ----------------------------- #

    num_iterations = 20000  # @param {type:"integer"}

    initial_collect_steps = 100  # @param {type:"integer"}
    collect_steps_per_iteration = 1  # @param {type:"integer"}
    replay_buffer_max_length = 100000  # @param {type:"integer"}

    batch_size = 64  # @param {type:"integer"}
    learning_rate = 1e-3  # @param {type:"number"}
    log_interval = 200  # @param {type:"integer"}

    num_eval_episodes = 10  # @param {type:"integer"}
    eval_interval = 1000  # @param {type:"integer"}

    STARTING_BUDGET = 80
    NUM_TARGETS = 2
    PLAYER_FOV = 60
    NUM_EPISODES = 5

    # -------------------------------- Environment ------------------------------- #

    env = SimpleSimGym(STARTING_BUDGET, NUM_TARGETS, PLAYER_FOV)

    # View Env Specs
    show_env_summary(env)

    # Instantiate two environments: one for training and one for evaluation.
    train_py_env = SimpleSimGym(STARTING_BUDGET, NUM_TARGETS, PLAYER_FOV)
    eval_py_env = SimpleSimGym(STARTING_BUDGET, NUM_TARGETS, PLAYER_FOV)
    train_env = tf_py_environment.TFPyEnvironment(
        train_py_env,
        check_dims=True,
    )
    eval_env = tf_py_environment.TFPyEnvironment(
        eval_py_env,
        check_dims=True,
    )

    # ----------------------------------- Agent ---------------------------------- #

    agent = get_dqn_agent(env, train_env, verbose=True)

    # ------------------------------- Replay Buffer ------------------------------ #

    rb_observer, iterator, collect_driver = setup_data_collection(
        env, agent, replay_buffer_max_length, collect_steps_per_iteration
    )

    # Evaluate baseline (random policy)
    evaluate_baseline(train_env, eval_env, num_eval_episodes, env, rb_observer, initial_collect_steps, train_py_env)

    # --------------------------------- TRAINING --------------------------------- #

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
    )

    # Visualize the training progress
    show_training_graph(returns, num_iterations, eval_interval)
