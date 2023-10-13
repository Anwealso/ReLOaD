# Library Imports
from env import SimpleSim
import math
import numpy as np
import time
import gymnasium as gym
from gymnasium import spaces


class SimpleSimGym(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self, starting_budget=2000, num_targets=8, player_fov=60, render_mode=None
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
            render_mode=render_mode,
            render_fps=self.metadata["render_fps"],
        )

        # # Actions: 0, 1, 2, 3, 4 for do nothing, R, F, L, B
        self.action_space = spaces.Discrete(5)

        max_dist = math.sqrt(2 * (self.game.env_size**2))
        # Observations (visible state):
        self.observation_space_unflattened = spaces.Dict(
            {
                "targets": spaces.Box(
                    low=-max_dist,
                    high=self.game.starting_budget * self.game.num_targets,
                    shape=(2, num_targets),
                    dtype=np.float32,
                ),  # target position (rel_x, rel_y)
                "environment": spaces.Box(
                    np.array([0]).astype(np.float32),
                    np.array([self.game.starting_budget]).astype(np.float32),
                ),  # environment remaining budget
            }
        )
        self.observation_space = spaces.utils.flatten_space(
            self.observation_space_unflattened
        )

        self.entropies = np.zeros(shape=(num_targets, 1))

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
            target_info[0][i] = dx
            target_info[1][i] = dy

            # Add current object sum of confidence over all time
            target_info[2][i] = float(np.sum(self.game.confidences[i, :]))

        observation = spaces.utils.flatten(
            self.observation_space_unflattened,
            {
                # "agent": agent_info,
                "targets": target_info,
                "environment": self.game.budget,
            },
        )

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

    def get_target_entropy(self, target_confidence_history, verbose=0):
        """
        Get the current entropy on a target, given an array of its observation
        confidence history

        Args:
            target_confidence_history: A 2D array of the confidences observed
            on the target over all time (dims: (num_classes,num_timesteps))
        Returns:
            The entropy of the target

        # TODO: Resolve which entropy to use - I like method 1 since it seems to adjust faster 
        """
        num_observations = np.count_nonzero(target_confidence_history[0,:])
        if num_observations == 0:
            return 1
        

        # # --------------------- METHOD 1 - LOG THEN SUM OVER TIME -------------------- #
        # # ------------- THIS IS THE: Average entropy across all timesteps ------------ #

        # # # Get the weighted probability that the object is of each class
        # # probability = np.divide(target_confidence_history, num_observations)
        # probability = target_confidence_history

        # # Get the suprrise associated with identifying the target as its true
        # # class or falsely as another class from an observation
        # suprise = np.log2(
        #     probability,
        #     where=(probability > 0),
        #     out=np.zeros_like(target_confidence_history),
        # )

        # # The entropy associated with the target
        # entropy = -np.multiply(probability, suprise)
        # entropy = np.sum(entropy)

        # # Now compute the average entropy across all timesteps

        # # The total entropy of the dataset of observation on the target
        # entropy = np.divide(entropy, num_observations)


        # --------------------- METHOD 2 - SUM OVER TIME THEN LOG -------------------- #
        # -------------- THIS IS THE: Entropy of the average probability ------------- #

        # # Get the weighted probability that the object is of each class
        # probability = np.divide(target_confidence_history, num_observations)
        probability_sum = np.sum(target_confidence_history, axis=1)
        probability = np.divide(probability_sum, num_observations) # normalise

        # Now compute the entropy of the average probability

        # Get the suprrise associated with identifying the target as its true
        # class or falsely as another class from an observation
        suprise = np.log2(
            probability,
            where=(probability > 0),
            out=np.zeros_like(probability),
        )

        # The entropy associated with the target
        entropy = -np.multiply(probability, suprise)
        entropy = np.sum(entropy)
        print(f"entropy2: {entropy}")





        # Set entropy to 1 for objects that do not have any observations
        if entropy == 0:
            entropy = 1

        if verbose > 0:
            print(f"num_observations: {num_observations}")
            print(f"target_confidence_history: {target_confidence_history}")
            print(f"entropy: {entropy}")
            print("\n")

        return entropy

    def get_entropy_reward(self, verbose=0):
        """
        An entropy / information gain based reward feedback for the agent.
        Essentially we want to implement an exploit vs explore (epsilon greedy
        type) tradeoff into the reward mechanism by giving the the agent:
        - High reward for looking at an object that it has surveyed relatively
          little.
        - But then decreasing that reward over time as the agent observes the
          object more and more.

        Each object in view will have a relevant reward calculated for it,
        based upon its information gain.

        The total reward will then be a sum of the reward of each object in
        view (objects out of view contribute zero reward).
        """
        entropy_reward = 0
        for i in range(0, len(self.game.targets)):
            class_id = self.game.targets[i].class_id
            old_entropy = self.get_target_entropy(
                self.game.confidences[i, :, :-1]
            )  # all timesteps up until and excluding last
            new_entropy = self.get_target_entropy(
                self.game.confidences[i, :, :]
            )  # all timesteps

            entropy_diff = float(old_entropy - new_entropy)
            entropy_reward += entropy_diff
            # entropy_reward += max(entropy_diff, 0)

            self.entropies[i, 0] = new_entropy

        # Normalise against vartying budgets and number of targets
        entropy_reward = entropy_reward * (
            (self.game.starting_budget * 2) / self.game.num_targets
        )

        # Add a multiplier to ensure it is worth it for the robot to  seek more
        # reward even though it entails more movement cost
        reward_multiplier = 1
        entropy_reward = entropy_reward * reward_multiplier

        # Update variance in target entropies
        self.variance = float(np.var(self.entropies, ddof=1))

        if verbose > 0:
            print(f"self.game.confidences: {self.game.confidences}")
            print(f"old_entropy: {old_entropy}")
            print(f"new_entropy: {new_entropy}")
            print(f"new_entropy: {new_entropy}")
            print(
                f"====================== entropy_reward: {entropy_reward} ======================"
            )
            print("\n")

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
        x_prime, y_prime = v_prime

        return (x_prime, y_prime)

    def step(self, action):
        """
        Description,
            Computes the physics of cart based on action applied.

        Args:
            action ([np.float32]): Apply +ve or -ve force.

        Returns:
            ([np.array size=(2,1) type=np.float32]): Next State
            ([np.float32]): Reward as norm distance from '0' state
            ([np.bool: ENV]). Terminal condition.
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

            # Show info on scoreboard
            self.game.set_scoreboard(
                {
                    "Curr Confidences": np.round(
                        self.game.current_confidences.flatten(), 2
                    ),
                    "Curr Entropies": np.round(self.entropies.flatten(), 2),
                    "Variance": np.round(self.variance, 2),
                    "Reward": np.round(reward, 2),
                    "Observation": np.round(obs, 2),
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
    NUM_TARGETS = 2
    PLAYER_FOV = 60

    # -------------------------------- Environment ------------------------------- #

    # Instantiate two environments: one for training and one for evaluation.
    env = SimpleSimGym(
        starting_budget=STARTING_BUDGET,
        num_targets=NUM_TARGETS,
        player_fov=PLAYER_FOV,
        render_mode="human",
    )

    # View Env Specs
    num_episodes = 10
    env.reset()

    for i in range(num_episodes):
        terminated = False
        truncated = False
        ep_reward = 0
        found = False
        j = 0

        while not (terminated or truncated):
            time.sleep(0.05)
            action = env.game.get_action_interactive()

            observation, reward, terminated, truncated, info = env.step(action)

            j += 1
            ep_reward += reward

            if terminated or truncated:
                observation, info = env.reset()

        print(f"Total Ep Reward: {ep_reward}")
        quit()
