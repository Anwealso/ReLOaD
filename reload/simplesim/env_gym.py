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

        max_dist = math.sqrt(2*(self.game.window_size**2))
        # Observations (visible state):
        self.observation_space_unflattened = spaces.Dict(
            {
                # "agent": spaces.Box(
                #     np.array([0, 0, 0]).astype(np.float32),
                #     np.array([self.game.sw, self.game.sw, 359]).astype(np.float32),
                # ), # agent x,y,angle
                # "agent": spaces.Box(
                #     np.array([0]).astype(np.float32),
                #     np.array([359]).astype(np.float32),
                # ), # agent angle
                "targets": spaces.Box(
                    low=-max_dist,
                    high=max_dist,
                    shape=(3, num_targets),
                    dtype=np.float32,
                ),  # target relative positions (x,y)
                # "current_conf": spaces.Box(
                #     low=0, high=1, shape=(num_targets, 1), dtype=np.float32
                # ), # confidences on each object
            }
        )
        self.observation_space = spaces.utils.flatten_space(
            self.observation_space_unflattened
        )

        self.entropy = np.zeros((self.game.num_targets, 1), dtype=np.float32)

        self.window = None
        self.clock = None

    def _get_obs(self):
        # Get seen or not

        # Get the the set of all correct observations for each target
        correct_obs_confidences = np.zeros_like(self.game.confidences)
        correct_obs_confidences = np.add(
            correct_obs_confidences, 
            self.game.confidences,
            where=(self.game.confidences > 0.5), 
            out=np.zeros_like(self.game.confidences)
        )
        # The sum confidences of each target over all time
        target_sum_confidences = np.sum(
            correct_obs_confidences,
            axis=1,
        )
        # The sum confidences of each target over all time
        targets_seen_status = np.zeros_like(target_sum_confidences)
        targets_seen_status = np.add(
            targets_seen_status,
            1,
            where=(target_sum_confidences > 0), 
            out=np.zeros_like(target_sum_confidences)
        )
        # print(targets_seen_status, "\n")

        # Target relative positions (dx,dy)
        target_info = np.zeros(
            shape=(3, len(self.game.targets)), dtype=np.float32
        )
        for i, target in enumerate(self.game.targets):
            target_x_cart = target.x
            target_y_cart = self.game.window_size - target.y
            robot_x_cart = self.game.robot.x
            robot_y_cart = self.game.window_size - self.game.robot.y
            dx = target_x_cart - robot_x_cart
            dy = target_y_cart - robot_y_cart
            (dx, dy) = self.world_to_body_frame(dx, dy)  # convert to body frame
            # Add target relative positions
            target_info[0][i] = dx
            target_info[1][i] = dy
            # Add current object confidences
            target_info[2][i] = self.game.current_confidences[i]
            # Add target seen status
            target_info[2][i] = targets_seen_status[i]

        # Agent x,y,angle
        # agent = np.array([self.game.robot.x, self.game.robot.y, self.game.robot.angle]).astype(np.float32)
        # agent = np.array([self.game.robot.angle]).astype(np.float32)


        # observation =  spaces.utils.flatten(self.observation_space_unflattened, {"agent": agent, "targets": target_rel_positions})
        observation = spaces.utils.flatten(
            self.observation_space_unflattened, {"targets": target_info}
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

        # Apply penalty per step to incentivise to get to goal fast
        reward -= self.step_cost

        # Apply penalty per action so that it doesnt move extraneously once it
        # gets to the goal
        if action != 0:
            reward -= self.action_cost

        # Apply reward based on observation entropy
        reward += self.get_entropy_reward(verbose=0) * self.game.starting_budget

        return reward    

    def get_entropy_reward(self, verbose=0):
        """
        An entropy / information gain based reward feedback for the agent.
        Essentially we want to implement an exploit vs explore (epsilon greedy
        type) tradeoff into the reward mechanism by giving the the agent:
        - High reward for looking at an object that it has surveyed relativel
          little
        - But then decreasing that reward over time as the agent observes the
          object more and more.

        Each object in view will have a relevant reward calculated for it,
        based upon its information gain.

        The total reward will then be a sum of the reward of each object in
        view (objects out of view contribute zero reward).
        """
        entropy_reward = 0

        # Num observations:
        num_observations = np.count_nonzero(self.game.confidences, axis=1)

        occurrence_prob = self.game.confidences
        # Get the experimental probability of each event:

        reciprocal_positive = np.reciprocal(
            self.game.confidences, 
            where=(self.game.confidences > 0), 
            out=np.zeros_like(self.game.confidences)
        )
        suprises_positive = np.log2(
            reciprocal_positive,
            where=(self.game.confidences > 0),
            out=np.zeros_like(self.game.confidences)
        )

        occurrence_prob_negative = 1 - self.game.confidences
        reciprocal_negative = np.reciprocal(
            occurrence_prob_negative,
            where=(occurrence_prob_negative > 0),
            out=np.zeros_like(self.game.confidences)
        )
        suprises_negative = np.log2(
            reciprocal_negative,
            where=(reciprocal_negative > 0),
            out=np.zeros_like(self.game.confidences)
        )

        entropy_positive = np.multiply(occurrence_prob, suprises_positive)
        entropy_negative = np.multiply(occurrence_prob_negative, suprises_negative)

        entropy_unnormalised = np.sum(
            entropy_positive + entropy_negative,
            axis=1,
        )
        new_entropy = np.divide(
            entropy_unnormalised,
            num_observations,
            where=(num_observations>0),
            out=np.zeros_like(entropy_unnormalised)
        )
        # Set entropy to 1 for objects that do not have any observations
        new_entropy[new_entropy == 0] = 1
        
        entropy_reward = 0
        for i, target in enumerate(self.game.targets):
            if self.game.robot.can_see(target):
                if self.entropy[i] == 0:
                    self.entropy[i] = 1
                
                # The reduction in entropy achieved by this new observation
                entropy_diff = float(self.entropy[i] - new_entropy[i])
                
                # TODO: Test which of thes strategies results in the best training
                # entropy_reward = entropy_diff
                # The reward is either the entropy_diff if we made a reduction 
                # or 0 otherwise.
                entropy_reward += max(entropy_diff, 0)

        if verbose > 0:
            print(f"FIXED")
            print(f"self.game.confidences: {self.game.confidences}")

            print(f"num_observations: {num_observations}")
            print(f"occurrence_prob: {occurrence_prob}")
            print(f"occurrence_prob_negative: {occurrence_prob_negative}")
            
            print(f"reciprocal_positive: {reciprocal_positive}")
            print(f"reciprocal_negative: {reciprocal_negative}")

            print(f"suprises_positive: {suprises_positive}")
            print(f"suprises_negative: {suprises_negative}")

            print(f"entropy_positive: {entropy_positive}")
            print(f"entropy_negative: {entropy_negative}")

            print(f"old_entropy: {self.entropy}")
            print(f"new_entropy: {new_entropy}")
            print(f"====================== entropy_reward: {entropy_reward} ======================")
            print("\n")

        self.entropy = new_entropy
        return entropy_reward

    def get_goal_reward(self):
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

        return step_goal_reward

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


            # Get the the set of all correct observations for each target
            correct_obs_confidences = np.zeros_like(self.game.confidences)
            correct_obs_confidences = np.add(
                correct_obs_confidences, 
                self.game.confidences,
                where=(self.game.confidences > 0.5), 
                out=np.zeros_like(self.game.confidences)
            )
            # The sum confidences of each target over all time
            target_sum_confidences = np.sum(
                # correct_obs_confidences,
                correct_obs_confidences,
                axis=1,
            )   
            variance = float(np.var(target_sum_confidences))


            # Show info on scoreboard
            self.game.set_scoreboard(
                {
                    "Confidence": format(np.sum(self.game.current_confidences), ".2f"),
                    "Variance": format(variance, ".2f"),
                    "Reward": format(reward, ".2f"),
                    "Observation": np.round(obs, 2),
                }
            )

            if self.game.gameover:
                # End of episode case
                truncated = True
                # step_reward = -100

            # Dont truncate the episode any more
            # elif reward > 0:
            #     terminated = True
            
        # return spaces.utils.flatten(self.observation_space, self._get_obs()), reward, terminated, truncated, info
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
