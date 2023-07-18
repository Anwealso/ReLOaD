from tf_agents.specs import tensor_spec
from tf_agents.networks import sequential
from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils import common
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
import reverb
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.drivers import py_driver
from tf_agents.environments import BatchedPyEnvironment
from answer_agent import cGame
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
import tensorflow as tf


class mTrainer:
    def __init__(self):

        self.returns = None
        self.train_env = tf_py_environment.TFPyEnvironment(cGame())
        self.eval_env = tf_py_environment.TFPyEnvironment(cGame())

        self.num_iterations = 20000  # @param {type:"integer"}
        self.initial_collect_steps = 100  # @param {type:"integer"}
        self.collect_steps_per_iteration = 100  # @param {type:"integer"}
        self.replay_buffer_max_length = 100000  # @param {type:"integer"}
        self.batch_size = 64  # @param {type:"integer"}
        self.learning_rate = 1e-3  # @param {type:"number"}
        self.log_interval = 200  # @param {type:"integer"}
        self.num_eval_episodes = 10  # @param {type:"integer"}
        self.eval_interval = 1000  # @param {type:"integer"}

    def createAgent(self):
        fc_layer_params = (100, 50)
        action_tensor_spec = tensor_spec.from_spec(self.train_env.action_spec())
        num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

        def dense_layer(num_units):
            return tf.keras.layers.Dense(
                num_units,
                activation=tf.keras.activations.relu,
                kernel_initializer=tf.keras.initializers.VarianceScaling(
                    scale=2.0, mode='fan_in', distribution='truncated_normal'))

        dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
        q_values_layer = tf.keras.layers.Dense(
            num_actions,
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-0.03, maxval=0.03),
            bias_initializer=tf.keras.initializers.Constant(-0.2))

        self.q_net = sequential.Sequential(dense_layers + [q_values_layer])

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        # rain_step_counter = tf.Variable(0)

        self.agent = dqn_agent.DqnAgent(
            time_step_spec=self.train_env.time_step_spec(),
            action_spec=self.train_env.action_spec(),
            q_network=self.q_net,
            optimizer=optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=tf.Variable(0))

        self.agent.initialize()

        self.eval_policy = self.agent.policy
        self.collect_policy = self.agent.collect_policy
        self.random_policy = random_tf_policy.RandomTFPolicy(
            self.train_env.time_step_spec(), self.train_env.action_spec())
        return True

    def compute_avg_return(self, environment, policy, num_episodes=10):
        # mT.compute_avg_return(mT.eval_env, mT.random_policy, 50)
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
        print('average return :', avg_return.numpy()[0])
        return avg_return.numpy()[0]

    def create_replaybuffer(self):

        table_name = 'uniform_table'
        replay_buffer_signature = tensor_spec.from_spec(
            self.agent.collect_data_spec)
        replay_buffer_signature = tensor_spec.add_outer_dim(
            replay_buffer_signature)

        table = reverb.Table(table_name,
                                max_size=self.replay_buffer_max_length,
                                sampler=reverb.selectors.Uniform(),
                                remover=reverb.selectors.Fifo(),
                                rate_limiter=reverb.rate_limiters.MinSize(1),
                                signature=replay_buffer_signature)

        reverb_server = reverb.Server([table])

        self.replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
            self.agent.collect_data_spec,
            table_name=table_name,
            sequence_length=2,
            local_server=reverb_server)

        self.rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
            self.replay_buffer.py_client,
            table_name,
            sequence_length=2)

        self.dataset = self.replay_buffer.as_dataset(num_parallel_calls=3,
                                                        sample_batch_size=self.batch_size,
                                                        num_steps=2).prefetch(3)
        self.iterator = iter(self.dataset)

    def testReplayBuffer(self):
        py_env = cGame()
        py_driver.PyDriver(
            py_env,
            py_tf_eager_policy.PyTFEagerPolicy(
                self.random_policy,
                use_tf_function=True),
            [self.rb_observer],
            max_steps=self.initial_collect_steps).run(self.train_env.reset())

    def trainAgent(self):

        self.returns = list()
        print(self.collect_policy)
        py_env = cGame()
        # Create a driver to collect experience.
        collect_driver = py_driver.PyDriver(
            py_env, # CHANGE 1
            py_tf_eager_policy.PyTFEagerPolicy(
                self.agent.collect_policy,
                # batch_time_steps=False, # CHANGE 2
                use_tf_function=True),
            [self.rb_observer],
            max_steps=self.collect_steps_per_iteration)

        # Reset the environment.
        # time_step = self.train_env.reset()
        time_step = py_env.reset()
        for _ in range(self.num_iterations):

            # Collect a few steps and save to the replay buffer.
            time_step, _ = collect_driver.run(time_step)

            # Sample a batch of data from the buffer and update the agent's network.
            experience, unused_info = next(self.iterator)
            train_loss = self.agent.train(experience).loss

            step = self.agent.train_step_counter.numpy()

            if step % self.log_interval == 0:
                print('step = {0}: loss = {1}'.format(step, train_loss))

            if step % self.eval_interval == 0:
                avg_return = self.compute_avg_return(self.eval_env,
                                                        self.agent.policy,
                                                        self.num_eval_episodes)
                print(
                    'step = {0}: Average Return = {1}'.format(step, avg_return))
                self.returns.append(avg_return)

    def run(self):
        self.createAgent()
        # self.compute_avg_return(self.train_env,self.eval_policy)
        self.create_replaybuffer()
        # self.testReplayBuffer()
        self.trainAgent()
        return True

if __name__ == '__main__':
    mT = mTrainer()
    mT.run()