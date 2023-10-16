# Library Imports
from env import SimpleSim
import math
import numpy as np
import time
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env


class SimpleSimGym(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        starting_budget=2000,
        num_targets=8,
        player_fov=60,
        num_classes=10,
        render_mode=None,
    ):
        """
        Description,
            Initializes the openai-gym environment with it's features.
        """

        self.step_cost = 1
        self.action_cost = 1  # was 1
        self.goal_reached_cutoff = 0.9

        # Init. Renders
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Internal State:
        self.game = SimpleSim(
            starting_budget,
            num_targets,
            player_fov,
            num_classes,
            render_mode=render_mode,
            render_fps=self.metadata["render_fps"],
        )

        # Discrete action space
        # self.action_space = spaces.Discrete(5) # Actions: 0, 1, 2, 3, 4 for do nothing, R, F, L, B
        # Continuous action space
        self.action_space = spaces.Box( # Actions: twist vector
            low=-1,
            high=1,
            shape=(2,),
            dtype=np.float32,
        )

        max_dist = math.sqrt(2 * (self.game.env_size**2))
        # Observations (visible state):
        self.observation_space_unflattened = spaces.Dict(
            {
                "targets": spaces.Box(
                    low=-max_dist,
                    high=self.game.starting_budget * self.game.num_targets,
                    shape=(3, num_targets),
                    dtype=np.float32,
                ),  # target position (rel_x, rel_y)
                "environment": spaces.Box(
                    low=0,
                    high=self.game.starting_budget,
                    shape=(1, 1),
                    dtype=np.float32,
                ),  # environment remaining budget
            }
        )
        self.observation_space = spaces.utils.flatten_space(
            self.observation_space_unflattened
        )

        self.entropies = np.ones(shape=(num_targets,))
        self.min_entropies = np.ones(shape=(num_targets,))  # the max ever entropies

        self.window = None
        self.clock = None

    def _get_obs(self):
        # ----------------------------------- AGENT ---------------------------------- #
        # agent_x_cart = self.game.robot.x
        # agent_y_cart = self.game.env_size - self.game.robot.x
        # agent_info = np.array(
        #     [agent_x_cart, agent_y_cart, self.game.robot.angle]
        # ).astype(np.float32)

        # ---------------------------------- TARGETS --------------------------------- #
        # Target relative positions (dx,dy)
        target_info = np.zeros(shape=(3, len(self.game.targets)), dtype=np.float32)
        for i, target in enumerate(self.game.targets):
            target_x_cart = target.x
            target_y_cart = self.game.env_size - target.y

            # Add target relative positions
            robot_x_cart = self.game.robot.x
            robot_y_cart = self.game.env_size - self.game.robot.y
            dx = target_x_cart - robot_x_cart
            dy = target_y_cart - robot_y_cart
            (dx, dy) = self.world_to_body_frame(dx, dy)  # convert to body frame
            target_info[0, i] = dx
            target_info[1, i] = dy

            # Add current object sum of confidence over all time
            target_info[2, i] = self.entropies[i]

        observation = spaces.utils.flatten(
            self.observation_space_unflattened,
            {
                # "agent": agent_info,
                "targets": target_info,
                "environment": self.game.budget,
            },
        )
        # print(observation, "\n")
        return observation

    def _get_reward(self, action):
        """
        The reward function for the RL agent.

        Current Reward Format:
        - Agent receives a -1 reward every time step in which it doesnt reach the goal
        - Agent receives a goal reward for each timestep it spends inside the goal area (within 90% closeness).
          The goal reward is calcuclated so that the best total reward possible over the whole episode is 0.

        Essentially the goal structure we have is:
        moving && !goal < !moving && !goal < !moving && goal < !moving && goal

        But to improve trining exploraiton we want:
        !moving && !goal <= moving && !goal < !moving && goal < !moving && goal
        """
        reward = 0

        # Apply reward based on observation entropy
        reward += self.get_entropy_reward(verbose=0)

        return reward

    def entropy(self, probability):
        """
        Calculates the entropy of a given probability distribution - can do one
        or multiple timesteps at a time.

        Args:
            probability: The (classes, observation) length distribution vector to calculate the entropy of
        Returns:
            The entropy(ies) of the probability distribution
        """

        # Get the suprrise associated with each class in the pdf (use log base num_classes)
        suprise = np.divide(
            -np.log(
                probability,
                where=(probability > 0),
                out=np.zeros_like(probability),
            ),
            np.log(
                self.game.num_classes,
            ),
        )

        # The entropy associated with the target (sum over the classes axis)
        entropy = np.sum(np.multiply(probability, suprise), axis=0)

        # Set the entropies for the timesteps where all class probabilities (target not in view) to 1
        # entropy[np.where(np.count_nonzero(probability, axis=0)==0)] = 1

        return entropy

    def cross_entropy(self, probability, true_class_id):
        """
        Calculates the cross entropy (a.k.a. log loss) of a given probability distribution - can do one
        or multiple timesteps at a time.

        Args:
            probability: The (classes, observation) length distribution vector to calculate the entropy of
        Returns:
            The cross-entropy(ies) of the probability distribution
        """

        # The cross entropy associated with the observation
        cross_entropy = -np.log(
            probability[true_class_id],
            where=(probability > 0),
            out=np.zeros_like(probability),
        )

        return cross_entropy

    def get_target_entropy(
        self, target_confidence_history, method="avg_entropy", verbose=0
    ):
        """
        Get the current entropy on a target, given an array of its observation
        confidence history.

        - avg_entropy: Calculates the average of the entropies observed at each
            time step
        - entropy_of_average: Calculates the entropy of the average probability
            distribution across all time steps

        Args:
            target_confidence_history [np.array]: A 2D array of the confidences
                observed on the target over all time (dims:
                (num_classes, num_timesteps))
            method [string]: The method of entropy computing to use - either
                "avg_entropy" or "entropy_of_average",
        Returns:
            The entropy of the target (dims: (num_timesteps,))
        """
        num_observations = np.count_nonzero(target_confidence_history[0, :])
        if num_observations == 0:
            return 1

        if method == "avg_entropy":
            # METHOD 1 - AVERAGE ENTROPY
            # Get the entropies of the data at each timestep
            entropy = self.entropy(target_confidence_history)
            # Get the average of the entropies across all timesteps where target was in view
            entropy = np.divide(np.sum(entropy), num_observations)

        elif method == "entropy_of_average":
            # METHOD 2 - ENTROPY OF THE AVERAGE PROBABILITY
            # Get the average probability distribution for this target over time
            avg_probability = np.average(target_confidence_history, axis=1)
            # Get the entropy of the average probability distribution
            entropy = self.entropy(avg_probability)

        return entropy

    def get_entropy_reward(self, method="differential", verbose=0):
        """
        An entropy based reward for the agent. Reward can be conputed in one of
        two ways:

        - Absolute: Reward is recieved for the amount of entropy reduction
            achieved over all of the targets so far (sum of 1 minus the current
            entropy)
        - Differential: Reward is recieved each time the agent reduces the
            entropy of a target below its previously achieved minimum

        Args:
            method [string]: Sets the reward computing metod - either
                "differential" or "absolute"

        Returns:
            [int]: The reward for the agent
        """
        entropy_reward = 0
        for i in range(0, len(self.game.targets)):
            new_entropy = self.get_target_entropy(
                self.game.confidences[i, :, :]
            )  # confidence distribution for target over all timesteps
            self.entropies[i] = new_entropy

            if method == "differential":
                # Get entropy diff
                entropy_change = new_entropy - self.min_entropies[i]
                # If better than the previous best
                if entropy_change < 0:
                    entropy_reward += -entropy_change
                    self.min_entropies[i] = new_entropy

        # Update variance in target entropies
        self.variance = float(np.var(self.entropies))

        if method == "differential":
            reward_multiplier = 1000

        elif method == "absolute":
            entropy_reward = self.game.num_targets - np.sum(self.entropies)
            reward_multiplier = 10

        # Add a multiplier to ensure it is worth it for the robot to  seek more
        # reward even though it entails more movement cost
        entropy_reward = entropy_reward * reward_multiplier

        # Normalise the reward against the number of targets
        entropy_reward = entropy_reward / self.game.num_targets

        return entropy_reward

    def world_to_body_frame(self, x, y):
        """
        Rotates (no translation) a relative distance vector from world frame into body frame
        """
        v = np.array([[x], [y]])

        d_theta = -(self.game.robot.angle - 90)
        rotation_matrix = np.array(
            [
                [math.cos(math.radians(d_theta)), -math.sin(math.radians(d_theta))],
                [math.sin(math.radians(d_theta)), math.cos(math.radians(d_theta))],
            ]
        )

        v_prime = np.matmul(rotation_matrix, v)
        x_prime, y_prime = v_prime[:, 0]

        return (x_prime, y_prime)

    def step(self, action):
        """
        Description,
            Steps the environment with the given action

        Args:
            action ([np.float32]): The discrete action to be carried out

        Returns:
            [np.array]: Next State
            [np.float32]: Reward
            [np.bool]: Terminal condition (is episode terminated or truncated).
        """
        if action is not None:  # First step without action, called from reset()
            # Step the game
            self.game.step(action)

            reward = 0
            terminated = False  # if we reached the goal
            truncated = False  # if the episode was cut off by timeout
            info = {}

            # Get game step results
            reward = self._get_reward(action)
            obs = self._get_obs()

            true_class_confidences = np.zeros_like(self.entropies)
            for i, target in enumerate(self.game.targets):
                true_class_confidences[i] = self.game.current_confidences[
                    i, target.class_id
                ]

            # Show info on scoreboard
            self.game.set_scoreboard(
                {
                    "True Class Confidences": np.round(true_class_confidences, 2),
                    "Entropies": np.round(self.entropies, 2),
                    "Variance": np.round(self.variance, 2),
                    "Reward": np.round(reward, 6),
                    # "Observation": np.round(obs, 2),
                }
            )

            if self.game.gameover:
                # Reward only given at the end of the episode
                truncated = True

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Resets the env

        Returns:
            Standard return format as specified by Gymnasium API
        """
        self.game.reset()

        info = {}  # no extra info at this stage
        return self._get_obs(), info

    def render(self):
        """
        Renders the env

        Returns:
            If render_mode="rgb_array", returns an rgb image of the rendered env in numpy array
        """
        return self.game.render()


if __name__ == "__main__":
    # ------------------------------ Hyperparameters ----------------------------- #
    # Env
    STARTING_BUDGET = 500
    NUM_TARGETS = 4
    NUM_CLASSES = 10
    PLAYER_FOV = 30

    # Whether to play it interactively or let the agent drive
    INTERACTIVE = True

    # -------------------------------- Environment ------------------------------- #

    # Instantiate two environments: one for training and one for evaluation.
    if INTERACTIVE:
        env = SimpleSimGym(
            starting_budget=STARTING_BUDGET,
            num_targets=NUM_TARGETS,
            num_classes=NUM_CLASSES,
            player_fov=PLAYER_FOV,
            render_mode="human",
        )

    # --------------------------- LOAD MODEL IF DESIRED -------------------------- #
    if not INTERACTIVE:
        config = {"policy": "MlpPolicy", "logdir": "logs/", "savedir": "saved_models/"}
        # Load the best model
        model = PPO.load(f"{config['savedir']}/best_model.zip")
        # Wrap the env for the model
        env = make_vec_env(
            SimpleSimGym,
            n_envs=1,
            monitor_dir=config["logdir"],
            env_kwargs=dict(
                starting_budget=STARTING_BUDGET,
                num_targets=NUM_TARGETS,
                player_fov=PLAYER_FOV,
                render_mode="human",
            ),
        )

    # --------------------------------- RUN EVAL --------------------------------- #
    num_episodes = 10
    obs = env.reset()

    for i in range(num_episodes):
        terminated = False
        truncated = False
        ep_reward = 0
        found = False
        j = 0

        while not (terminated or truncated):
            if INTERACTIVE:
                # For human
                action = env.game.get_action_interactive()
                obs, reward, terminated, truncated, info = env.step(action)
            else:
                # For agent
                action, _ = model.predict(obs)
                obs, reward, dones, info = env.step(action)

            j += 1
            ep_reward += reward

            if terminated or truncated:
                obs, info = env.reset()

            # time.sleep(0.01)

        print(f"Total Ep Reward: {ep_reward}")
        quit()
