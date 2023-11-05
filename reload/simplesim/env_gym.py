# Library Imports
from env import SimpleSim, NaivePolicy
import math
import numpy as np
import time
import random
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import DQN, PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env


class SimpleSimGym(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        max_budget=2000,
        max_targets=5,
        num_classes=10,
        player_fov=60,
        action_format="continuous",
        render_mode=None,
        render_plots=True,
        seed=None,
    ):
        """
        Description,
            Initializes the openai-gym environment with it's features.
        """

        # Init. Renders
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Internal State:
        self.game = SimpleSim(
            max_budget,
            max_targets,
            num_classes,
            player_fov,
            action_format,
            render_mode=render_mode,
            render_fps=self.metadata["render_fps"],
            render_plots=render_plots,
            seed=seed,
        )

        if action_format == "discrete":
            # Discrete action space
            self.action_space = spaces.Discrete(
                5
            )  # Actions: 0, 1, 2, 3, 4 for do nothing, R, F, L, B
        elif action_format == "continuous":
            # Continuous action space
            self.action_space = spaces.Box(  # Actions: twist vector
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
                    high=max_dist,
                    shape=(3, self.game.max_targets),
                    dtype=np.float32,
                ),  # target position (rel_x, rel_y, entropy)
                "environment": spaces.Box(
                    low=0,
                    high=self.game.max_budget,
                    shape=(2, 1),
                    dtype=np.float32,
                ),  # environment (remaining budget, num_targets)
            }
        )
        self.observation_space = spaces.utils.flatten_space(
            self.observation_space_unflattened
        )

        self.entropies = np.ones(shape=(max_targets,))
        
        # For differential reward (oldstyle monotonic) modes:
        self.min_entropies = np.ones(shape=(max_targets,))  # the max ever entropies
        # For monotonic reward modes:
        # self.best_reward = 0  # keep track of the best reward in the current episode so far

        self.window = None
        self.clock = None

    def _get_obs(self):
        # Target relative positions (dx,dy)
        target_info = np.zeros(shape=(3, self.game.max_targets), dtype=np.float32)
        for i in range(0, self.game.max_targets):
            # If this target slot is in use for this episode
            if i < self.game.num_targets:
                target = self.game.targets[i]

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

            else:
                # Fill the rest of the target information with zeros if there are no more targets
                target_info[0, i] = 0
                target_info[1, i] = 0
                target_info[2, i] = 0

        observation = spaces.utils.flatten(
            self.observation_space_unflattened,
            {
                "targets": target_info,
                "environment": [self.game.budget, self.game.num_targets],
            },
        )
        return observation

    def _get_reward(self, action):
        """
        The reward function for the RL agent.

        Current Reward Format:
        - Reward for targets that have not yet been fully explored
        """
        reward = 0

        # Apply reward based on observation entropy
        reward += self.get_entropy_reward(method="absolute",verbose=0)

        return reward

    def get_confidence_reward(self, verbose=0):
        """
        An confidence based reward where the agent gets reward in 
        proportion to the sum of true-class confidence across all targets at 
        each timestep.

        Essentially every time the agent observesa target, it gets a reward.
        This is working on the assumption that all information is good 
        information.

        Returns:
            [int]: The reward for the agent
        """

        reward = 0
        for target_num in range(0, len(self.game.targets)):
            # Get the new entropy (jusat for display)
            self.entropies[target_num] = self.get_target_entropy(self.game.confidences[target_num, :, :])

            # Get the sum of all past confidences on the true class
            class_id = self.game.targets[target_num].class_id
            true_confidence_sum = np.sum(self.game.confidences[target_num, class_id, :])

            # Add a diminishing returns to the confidence sum and scale the 
            # sum to between 0 and 1
            scaled_confidence_sum = 1 - math.exp(-true_confidence_sum / 5)

            reward += scaled_confidence_sum

        # Update variance in target entropies
        self.variance = float(np.var(self.entropies))

        # Normalise against the number of targets
        reward = reward / self.game.num_targets
        
        # Normalise against the episode length
        reward = reward / self.game.starting_budget
        
        # Normalise the max total episode reward to 2000
        reward_multiplier = 2000
        reward = reward * reward_multiplier
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

        # Let entropies for the timesteps where target not in view (all class
        # probabilities are zero) stay as zero (they will contribute 0 to
        # the average entropy calculation later)

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

    def get_entropy_reward(self, method="absolute", verbose=0):
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

        # Update variance in target entropies
        self.variance = float(np.var(self.entropies))

        if method == "absolute":
            entropy_reward = self.game.max_targets - np.sum(self.entropies)

        # Normalise the reward against the number of targets
        # (Reward is now average information gain per target)
        entropy_reward = entropy_reward / self.game.num_targets
        
        # Normalise the reward against the episode length
        entropy_reward = entropy_reward / self.game.starting_budget
        # Normalise total episode reward to 2k
        entropy_reward = entropy_reward * 2000

        return entropy_reward

    

    def get_monotonic_entropy_reward(self, method="monotonic", verbose=0):
        """
        An entropy based reward for the agent. Reward can be conputed in one of
        two ways:

        - Absolute: Reward is recieved for the amount of entropy reduction
            achieved over all of the targets so far (sum of 1 minus the current
            entropy)
        - Monotonic Hidden: Reward is equal to the maximum total IG state that
            has been achieved in the episode so far (behind the scenes, though,
            IG has a hidded state that can go up or down dependin on quality of
            observations)
        - Monotonic: Reward is equal to the maximum total IG state that
            has been achieved in the episode so far (but this time there is no
            hidden state of IG - this is the real state of IG and we make sure
            IG only goes up by simply deleting observatinos that cause IG to go
            down)

        Args:
            method [string]: Sets the reward computing metod - either
                "absolute",  "monotonic_hidden" or "monotonic"

        Returns:
            [int]: The reward for the agent
        """
        if method not in ["absolute", "monotonic_hidden", "monotonic"]:
            raise Exception

        entropy_reward = 0
        for i in range(0, len(self.game.targets)):
            # Get the new entropy (entropy at current timestep)
            new_entropy = self.get_target_entropy(self.game.confidences[i, :, :])

            if method == "monotonic" and new_entropy > self.entropies[i]:
                # If this new timestep's vector of observations made our
                # entropy lower, include it, otherwise set this observation to
                # zeros, and dont update the current entropy
                self.game.confidences[i, :, -1] = 0
            else:
                self.entropies[i] = new_entropy

        # Update variance in target entropies
        self.variance = float(np.var(self.entropies))

        reward_multiplier = 10

        entropy_reward = self.game.max_targets - np.sum(self.entropies)

        if method == "monotonic_hidden":
            self.best_reward = max(entropy_reward, self.best_reward)
            entropy_reward = self.best_reward

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
                    "Entropy Variance": np.round(self.variance, 2),
                    "Reward": np.round(reward, 6),
                    # "Observation": np.round(obs, 2),
                }
            )

            reward = self._get_reward(action)
            if self.game.budget == 0:
                # Reward only given at the end of the episode
                truncated = True

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Resets the env

        Returns:
            Standard return format as specified by Gymnasium API
        """
        self.game.reset() # reset the underlying game state

        self.entropies = np.ones(shape=(self.game.max_targets,))
        self.min_entropies = np.ones(shape=(self.game.max_targets,))  # the max ever entropies
        self.best_reward = 0 # reset best reward for episode

        info = {}  # no extra info at this stage
        return self._get_obs(), info

    def render(self):
        """
        Renders the env

        Returns:
            If render_mode="rgb_array", returns an rgb image of the rendered env in numpy array
        """
        return self.game.render()


def main():
    # ------------------------------ Hyperparameters ----------------------------- #
    # Env
    MAX_BUDGET = 500
    MAX_TARGETS = 5
    NUM_CLASSES = 10
    PLAYER_FOV = 30
    ACTION_FORMAT = "continuous"

    # Whether to play it interactively or let the agent drive
    INTERACTIVE = False

    # -------------------------------- Environment ------------------------------- #
    # Instantiate two environments: one for training and one for evaluation.
    if INTERACTIVE:
        env = SimpleSimGym(
            max_budget=MAX_BUDGET,
            max_targets=MAX_TARGETS,
            num_classes=NUM_CLASSES,
            player_fov=PLAYER_FOV,
            action_format=ACTION_FORMAT,
            render_mode="human",
        )

    # --------------------------- LOAD MODEL IF DESIRED -------------------------- #
    if not INTERACTIVE:
        config = {"policy": "MlpPolicy", "logdir": "logs/", "savedir": "saved_models/"}
        # Load the best model
        model = SAC.load(f"{config['savedir']}/best_sac.zip")
        # model = SAC.load(f"{config['savedir']}/MlpPolicy_SAC_step4000000.zip")
        # Wrap the env for the model
        env = make_vec_env(
            SimpleSimGym,
            n_envs=1,
            monitor_dir=config["logdir"],
            env_kwargs=dict(
                max_budget=MAX_BUDGET,
                max_targets=MAX_TARGETS,
                num_classes=NUM_CLASSES,
                player_fov=PLAYER_FOV,
                action_format=ACTION_FORMAT,
                render_mode="human",
                seed=574,
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
        # quit()


def run_naive_policy():
    # ------------------------------ Hyperparameters ----------------------------- #
    # Env
    MAX_BUDGET = 500
    MAX_TARGETS = 5
    NUM_CLASSES = 10
    PLAYER_FOV = 30
    ACTION_FORMAT = "continuous"

    # -------------------------------- Environment ------------------------------- #
    # Instantiate two environments: one for training and one for evaluation.
    env = SimpleSimGym(
        max_budget=MAX_BUDGET,
        max_targets=MAX_TARGETS,
        num_classes=NUM_CLASSES,
        player_fov=PLAYER_FOV,
        action_format=ACTION_FORMAT,
        render_mode="human",
        # seed=574,
    )

    # --------------------------- LOAD MODEL IF DESIRED -------------------------- #

    # --------------------------------- RUN EVAL --------------------------------- #
    num_episodes = 10
    obs = env.reset()

    for i in range(num_episodes):
        terminated = False
        truncated = False
        ep_reward = 0
        found = False
        j = 0

        naive_policy = NaivePolicy(env.game)
        
        while not (terminated or truncated):
            action = naive_policy.get_action(env.game.robot)
            obs, reward, terminated, truncated, info = env.step(action)

            j += 1
            ep_reward += reward

            if terminated or truncated:
                obs, info = env.reset()

            # time.sleep(0.01)

        print(f"Total Ep Reward: {ep_reward}")
        # quit()


if __name__ == "__main__":
    main()
    # run_naive_policy()