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

        # Actions: 0, 1, 2, 3 for do nothing, R, F, L
        # self.action_space = spaces.Discrete(4)

        # # Actions: 0, 1, 2, 3, 4 for do nothing, R, F, L, B
        self.action_space = spaces.Discrete(5)

        max_dist = math.sqrt(2 * (self.game.window_size**2))
        # Observations (visible state):
        self.observation_space_unflattened = spaces.Dict(
            {
                # "agent": spaces.Box(
                #     np.array([0, 0, 0]).astype(np.float32),
                #     np.array(
                #         [self.game.window_size, self.game.window_size, 359]
                #     ).astype(np.float32),
                # ),  # agent (x, y, angle)
                "targets": spaces.Box(
                    low=-max_dist,
                    high=self.game.starting_budget*self.game.num_targets,
                    shape=(3, num_targets),
                    dtype=np.float32,
                ),  # target position (rel_x, rel_y, target_sum_conf)
                "environment": spaces.Box(
                    np.array([0]).astype(np.float32),
                    np.array([self.game.starting_budget]).astype(np.float32),
                ),  # environment remaining budget
            }
        )
        self.observation_space = spaces.utils.flatten_space(
            self.observation_space_unflattened
        )

        self.current_entropies = np.zeros((self.game.num_targets, 1), dtype=np.float32)
        self.target_rewards = [0, 0]

        self.window = None
        self.clock = None

    def _get_obs(self):
        # ----------------------------------- AGENT ---------------------------------- #
        # Agent x,y,angle
        agent_x_cart = self.game.robot.x
        agent_y_cart = self.game.window_size - self.game.robot.x
        agent_info = np.array(
            [agent_x_cart, agent_y_cart, self.game.robot.angle]
        ).astype(np.float32)
        # agent = np.array([self.game.robot.angle]).astype(np.float32)

        # ---------------------------------- TARGETS --------------------------------- #
        # Target relative positions (dx,dy)
        target_info = np.zeros(shape=(3, len(self.game.targets)), dtype=np.float32)
        for i, target in enumerate(self.game.targets):
            target_x_cart = target.x
            target_y_cart = self.game.window_size - target.y
            
            # # Add target absolute positions
            # target_info[0][i] = target_x_cart
            # target_info[1][i] = target_y_cart

            # Add target relative positions
            robot_x_cart = self.game.robot.x
            robot_y_cart = self.game.window_size - self.game.robot.y
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
        self.variance = np.var(np.sum(self.game.confidences, axis=1), ddof=1)
        reward = 0

        # Apply penalty per step to incentivise to get to goal fast
        reward -= self.step_cost

        # Apply penalty per action so that it doesn't move extraneously once it
        # gets to the goal
        if action != 0:
            reward -= self.action_cost

        # Apply reward based on observation entropy
        # reward += self.get_entropy_reward(verbose=0)
        # reward += self.get_goal_reward(verbose=0)
        reward += np.sum(self.game.current_confidences)

        return reward

    def _get_end_reward(self):
        """
        The end reward for the RL agent.
        """
        reward = 0

        # Apply reward based on sum confidence variance
        all_conf_sum = np.sum(self.game.confidences)
        variance = np.var(np.sum(self.game.confidences, axis=1), ddof=1)
        if reward > 0:
            reward = all_conf_sum / variance

        return reward

    def get_target_entropy(self, target_confidence_history, verbose=0):
        """
        Get the current entropy on a target, given an array of its observation
        confidence history

        Args:
            target_confidence_history: A 1D array of the confidences observed
            on the target over all time (dims: (1,num_timesteps))
        Returns:
            The entropy of the target
        """
        num_observations = np.count_nonzero(target_confidence_history)
        if num_observations == 0:
            return 1
        target_unconfidence_history = 1 - target_confidence_history
        
        # Get the weighted probability that the object is of the true class 
        # or the false class gained from each observation
        # probability_true_class = np.divide(target_confidence_history, num_observations)
        # probability_false_class = np.divide(target_unconfidence_history, num_observations)
        probability_true_class = target_confidence_history
        probability_false_class = target_unconfidence_history

        # Get the suprrise associated with identifying the target as its true 
        # class or falsely as another class from an observation
        suprise_true = np.log2(
            probability_true_class,
            where=(probability_true_class > 0),
            out=np.zeros_like(target_confidence_history),
        )
        suprise_false = np.log2(
            probability_false_class,
            where=(probability_false_class > 0),
            out=np.zeros_like(target_confidence_history),
        )

        # The entropy associated with identifying the target as its true class 
        # or falsely as another class from an observation
        entropy_true = -np.multiply(probability_true_class, suprise_true)
        entropy_false = -np.multiply(probability_false_class, suprise_false)
        # The total entropy of the dataset of observation on the target
        entropy = np.sum(
            entropy_true + entropy_false
        )
        entropy = np.divide(entropy, num_observations)
        # Set entropy to 1 for objects that do not have any observations
        if (entropy == 0):
            entropy = 1
        
        if verbose > 0:
            print("\n", num_observations)
            print(f"target_confidence_history: {target_confidence_history}")
            print(f"target_unconfidence_history: {target_unconfidence_history}")
            print(f"entropy_true: {entropy_true}")
            print(f"entropy_false: {entropy_false}")
            print(f"entropy: {entropy}")

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

        entropies = np.zeros_like(self.game.current_confidences)
        entropy_reward = 0
        for i in range(0, len(self.game.targets)):
            # if self.game.can_see(target):
            old_entropy = self.get_target_entropy(
                self.game.confidences[i, :-1]
            )  # all timesteps up until and excluding last
            new_entropy = self.get_target_entropy(
                self.game.confidences[i, :]
            )  # all timesteps
            entropies[i, 0] = new_entropy

            entropy_diff = float(old_entropy - new_entropy)
            entropy_reward += entropy_diff
            # entropy_reward += max(entropy_diff, 0)
        self.current_entropies = entropies

        # Normalise against vartying budgets and number of targets
        entropy_reward = entropy_reward * (
            (self.game.starting_budget * 2) / self.game.num_targets
        )

        # Add a multiplier to ensure it is worth it for the robot to  seek more
        # reward even though it entails more movement cost
        reward_multiplier = 1
        entropy_reward = entropy_reward * reward_multiplier

        # Apply a penalty factor for the amount of (sample) variance in target entropies
        self.variance = float(np.var(self.current_entropies,ddof=1))
        # A penalty factor that scales from [1, 0.5) for variance from [0, infty)
        similarity_factor = 1 / (10*self.variance + 1)
        entropy_reward = entropy_reward * similarity_factor


        if verbose > 0:
            print(f"self.game.confidences: {self.game.confidences}")

            print(f"old_entropy: {old_entropy}")
            print(f"new_entropy: {new_entropy}")
            print(
                f"====================== entropy_reward: {entropy_reward} ======================"
            )
            print("\n")

        return entropy_reward

    def get_goal_reward(self, verbose=0):
        """
        Gets the reward to give to the agent for each step that it remains in
        the goal area.

        Essentially the best path is if the agent expends min_cost_to_goal cost
        to get to the coal and then stays within the goal area for the rest of
        the episode.
        """
        # Get the (optimal possible) number of steps that can be spent at the goal
        # steps_at_goal = self.game.starting_budget - (self.min_cost_to_goal()/self.action_cost)
        steps_at_goal = self.game.starting_budget - self.min_steps_to_goal()
        # print("")
        # print(f"steps_at_goal: {steps_at_goal}")

        # Goal reward must compensate the agent for the cost to get to the goal
        # plus the step costs across the whole episode.
        total_goal_reward = self.min_cost_to_goal() + (
            self.game.starting_budget * self.step_cost
        )
        # print(f"total_goal_reward: {total_goal_reward}")

        # The amount of the reward to give the agent for each step
        step_goal_reward = total_goal_reward / steps_at_goal
        # print(f"step_goal_reward: {step_goal_reward}")
        # print("")

        reward = 0
        for i, target in enumerate(self.game.targets):
            if self.get_closeness(self.game.robot, target) > 0.95:
                reward += step_goal_reward
                self.target_rewards[i] += step_goal_reward

        # Apply a penalty factor for the amount of (sample) variance in target entropies
        self.variance = float(np.var(self.target_rewards,ddof=1))
        # A penalty factor that scales from [1, 0.5) for variance from [0, infty)
        similarity_factor = 1 / ((self.variance/10) + 1)
        reward = reward * similarity_factor

        # Add a multiplier to ensure it is worth it for the robot to  seek more
        # reward even though it entails more movement cost
        reward_multiplier = 100
        reward = reward * reward_multiplier

        return reward

    def get_closeness(self, robot, target):
        """
        Gets the closeness of a target to the robot
        """
        dS = math.sqrt((target.x - robot.x) ** 2 + (target.y - robot.y) ** 2)
        farness = abs(dS) / math.sqrt(
            self.game.window_size**2 + self.game.window_size**2
        )  # dS as a fraction of max (scaled from 0 to 1)
        closeness = 1 - farness  # closeness = opposite of farness (scales from 1 to 0)
        return closeness

    def min_steps_to_goal(self):
        """
        Gets the cost of the shortest path to the goal
        """
        min_farness = 1 - self.goal_reached_cutoff
        buffer_distance = min_farness * math.sqrt(
            self.game.window_size**2 + self.game.window_size**2
        )

        # Get cartesian coord distances
        target_x_cart = self.game.targets[0].x
        target_y_cart = self.game.window_size - self.game.targets[0].y
        robot_starting_x_cart = self.game.robot.starting_x
        robot_starting_y_cart = self.game.window_size - self.game.robot.starting_y

        dy = target_y_cart - robot_starting_y_cart
        dx = target_x_cart - robot_starting_x_cart
        dS = math.sqrt((dx) ** 2 + (dy) ** 2)
        min_dist = dS - buffer_distance
        num_move_actions = math.floor(min_dist / self.game.robot.move_rate)

        target_angle = math.degrees(math.atan2(dy, dx))
        target_angle = (
            360 + target_angle if target_angle < 0 else target_angle
        )  # correct for ambiguous case
        dtheta = abs(self.game.robot.starting_angle - target_angle)
        dtheta = (
            dtheta - 180 if dtheta > 180 else dtheta
        )  # correct since we can go clockwise or anti-clockwise
        # print(f"dtheta1: {dtheta}\r")
        # dtheta = 180-dtheta if dtheta>90 else dtheta # correct since we can also reverse into the target
        # print(f"dtheta2: {dtheta}")
        num_turn_actions = math.floor(
            dtheta / self.game.robot.turn_rate
        )  # each turn action is 5deg

        if (
            num_move_actions < 1
        ):  # we start inside the object, any movement will cause us to reach goal
            total_num_actions = 1
        else:
            total_num_actions = num_turn_actions + num_move_actions
        return total_num_actions

    def min_cost_to_goal(self):
        """
        Gets the cost of the shortest path to the goal
        """
        min_cost = self.min_steps_to_goal() * self.action_cost
        return min_cost

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

            # Return reward
            # Mid-episode case
            reward = self._get_reward(action)
            obs = self._get_obs()

            # Show info on scoreboard
            self.game.set_scoreboard(
                {
                    "Confidence": format(np.sum(self.game.current_confidences), ".2f"),
                    "Variance": format(self.variance, ".2f"),
                    "Reward": format(reward, ".2f"),
                    "Curr Entropies": format(self.current_entropies),
                    "Observation": np.round(obs, 2),
                }
            )

            if self.game.gameover:
                # Reward only given at the end of the episode
                truncated = True
                reward = self._get_reward(action)
            else:
                # No continuous rewards recieved at each timestep
                reward = self._get_reward(action) + self._get_end_reward()

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Description,
            Resets the ENV.

        Returns:
            ([np.array size=(2,1) type=np.float32]): Random State
        """
        self.game.reset()

        info = {}  # no extra info at this stage
        return self._get_obs(), info

    def render(self):
        """
        Description,
            Renders the ENV.
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
    # utils.show_env_summary(py_env)
    num_episodes = 10
    env.reset()

    # curriculum = np.linspace(0, 1, num=(int(num_episodes))) # increase from 5 to farthest corner distance
    # print(curriculum)

    # env.game.curriculum = curriculum[0]
    for i in range(num_episodes):
        # Set the level of the curriculum
        # env.game.curriculum = curriculum[i]
        # env.reset()
        # print(f"curriculum: {curriculum[i]}")

        terminated = False
        truncated = False
        ep_reward = 0
        found = False
        j = 0

        # DEBUG
        # print(f"min_cost_to_goal: {env.min_cost_to_goal()}")

        while not (terminated or truncated):
            time.sleep(0.05)
            action = env.game.get_action_interactive()

            # action = env.action_space.sample()  # this is where you would insert your policy
            observation, reward, terminated, truncated, info = env.step(action)

            j += 1
            # print(f"Reward: {format(reward, '.2f')}, Observation: {observation}\n")

            # if reward > -env.step_cost and found == False:
            #     # print(f"Real Cost to Reach Goal: {j * env.action_cost}")
            #     # print(f"Real Actions to Reach Goal: {j}")

            #     # env.print_goal_reward()
            #     found = True

            ep_reward += reward

            if terminated or truncated:
                observation, info = env.reset()

        print(f"Total Ep Reward: {ep_reward}")
        quit()
