# Library Imports
from env import SimpleSim
# from env import SimpleSim
import math
import numpy as np
import time
import gymnasium as gym
from gymnasium import spaces


class SimpleSimGym(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, starting_budget=2000, num_targets=8, player_fov=60, render_mode=None):
        """
        Description,
            Initializes the openai-gym environment with it's features.
        """

        # Init. Renders
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Internal State:
        self.game = SimpleSim(
            starting_budget, num_targets, player_fov, render_mode=render_mode, render_fps=self.metadata["render_fps"]
        )

        # Actions: 0, 1, 2, 3, 4 for D/N, L, R, F, B
        self.action_space = spaces.Discrete(5) # actions are: do nothing, R, F, L, B

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
                    low=-self.game.window_size, high=self.game.window_size, shape=(2, num_targets), dtype=np.float32
                ), # target relative positions (x,y)
                # "current_conf": spaces.Box(
                #     low=0, high=1, shape=(num_targets, 1), dtype=np.float32
                # ), # confidences on each object
            }
        )
        self.observation_space = spaces.utils.flatten_space(self.observation_space_unflattened)


        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None



    def _get_obs(self):
        # Target relative positions (dx,dy)
        target_rel_positions = np.zeros(shape=(2, len(self.game.targets)), dtype=np.float32)
        for i, target in enumerate(self.game.targets):
            dx = self.game.robot.x - target.x 
            dy = target.y - self.game.robot.y
            target_rel_positions[0][i] = dx
            target_rel_positions[1][i] = dy

        # Agent x,y,angle
        # agent = np.array([self.game.robot.x, self.game.robot.y, self.game.robot.angle]).astype(np.float32)
        agent = np.array([self.game.robot.angle]).astype(np.float32)

        # Current object confidences
        # conf = self.game.current_confidences
        
        observation =  spaces.utils.flatten(self.observation_space_unflattened, {"agent": agent, "targets": target_rel_positions})
        return observation

    def _get_reward(self, action):
        """
        The reward function for the RL agent
          - Agent gets positive reward for the current instantaneous confidence
            it achieved in the last time step
          - Agent receives negative reward for moving
        
        Agent will learn to maximise this instantaneous reward at each time
        step.

        Current reward is a function of how close the robot is to the target
        """
        dS = math.sqrt((self.game.targets[0].x - self.game.robot.x)**2 + (self.game.targets[0].y - self.game.robot.y)**2)
        farness = abs(dS) / math.sqrt(self.game.window_size**2 + self.game.window_size**2) # dS as a fraction of max (scaled from 0 to 1)
        norm_reward = 1 - farness # closeness = opposite of farness (scales from 1 to 0)
        
        # 100 if closeness is > 95%, zero else wise reward 
        reward = 0
        if norm_reward > 0.9:
            reward = 100

        # # Exponentially scaled closeness reward
        # reward = 100 * math.exp(5*(norm_reward-1)) # (y=100 e^{5(x-1)}) - norm_reward but scaled exponentially between 0 and 100

        # # reward = np.sum(self.game.current_confidences)
        # if action != 0: # if action is not do-nothing
        #     reward = reward / 2

        return reward


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

        if action is not None: # First step without action, called from reset()
            # Step the game
            self.game.step(action)
            self.game.perform_action_interactive()

            # Show info on scoreboard
            # self.game.set_scoreboard({"Reward": format(self._get_reward(action), ".2f")})
            self.game.set_scoreboard({"Reward": format(self._get_reward(action), ".2f"), "Observation": self._get_obs()})
            print(f"Reward: {format(self._get_reward(action), '.2f')}, Observation: {self._get_obs()}\n")

        reward = 0
        terminated = False # if we reached the goal
        truncated = False # if the episode was cut off by timeout
        info = {}

        # Return reward
        if action is not None:  # First step without action, called from reset()
            # Mid-episode case
            reward = self._get_reward(action)

            if self.game.gameover:
                # End of episode case
                truncated = True
                # step_reward = -100

            elif reward > 0:
                terminated = True
                
        # return spaces.utils.flatten(self.observation_space, self._get_obs()), reward, terminated, truncated, info
        return self._get_obs(), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Description,
            Resets the ENV.

        Returns:
            ([np.array size=(2,1) type=np.float32]): Random State
        """
        self.game.reset()

        # print(1)
        # print(self._get_obs())
        # print(2)
        # # print(spaces.utils.flatten(self.observation_space, self._get_obs()))
        # print(self._get_obs())
        # print(3)

        info = {} # no extra info at this stage
        # return spaces.utils.flatten(self.observation_space, self._get_obs()), info
        return self._get_obs(), info

    def render(self, mode="human"):
        """
        Description,
            Renders the ENV.
        """
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None:
            return None

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0
        self.carttrans.set_translation(cartx, carty)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")
    

if __name__ == "__main__":
    # ------------------------------ Hyperparameters ----------------------------- #
    # Env
    STARTING_BUDGET = 2000
    NUM_TARGETS = 1
    PLAYER_FOV = 60

    # -------------------------------- Environment ------------------------------- #

    # Instantiate two environments: one for training and one for evaluation.
    env = SimpleSimGym(starting_budget=STARTING_BUDGET, 
                       num_targets=NUM_TARGETS, 
                       player_fov=PLAYER_FOV, 
                       render_mode="human")

    # View Env Specs
    # utils.show_env_summary(py_env)

    num_episodes = 10

    env.reset()

    for i in range(num_episodes):
        terminated = False
        truncated = False

        while not (terminated or truncated):
            time.sleep(0.05)
            action = env.game.get_action_interactive()

            # action = env.action_space.sample()  # this is where you would insert your policy
            observation, reward, terminated, truncated, info = env.step(action)
            # print(f"Reward: {format(reward, '.2f')}, Observation: {observation}\n")

            if terminated or truncated:
                observation, info = env.reset()
