# Library Imports
from ReLOaD.reload.simplesim.env import SimpleSim
import math
import gym
from gym import spaces
import numpy as np


class SimpleSimGym(gym.Env):

    # metadata = {"render.modes": ["human", "rgb_array"], "video.FPS": 50}

    def __init__(self, starting_budget, num_targets, player_fov, visualize=True):
        """
        Description,
            Initializes the openai-gym environment with it's features.
        """

        # Internal State:
        self.game = SimpleSim(
            starting_budget, num_targets, player_fov, visualize=visualize
        )

        # # Actions: 0, 1, 2, 3 for L, R, F, B
        # self._action_spec = array_spec.BoundedArraySpec(
        #     shape=(), dtype=np.int32, minimum=0, maximum=3, name="action"
        # )
        self.action_space = spaces.Discrete(5) # actions are: do nothing, R, F, L, B

        # # Observations (visible state):
        # observation_shape = np.shape(self._get_obs())[0]
        # self._observation_spec = array_spec.BoundedArraySpec(
        #     shape=(observation_shape,),
        #     dtype=np.float32,
        #     minimum=0,
        #     name="observation",
        # )
        # Observations are dictionaries with the agent's and the targets' locations.
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(
                    np.array([0, 0, 0]).astype(np.float32),
                    np.array([self.game.sw, self.game.sw, 359]).astype(np.float32),
                ), # agent x,y,angle
                "targets": spaces.Box(
                    low=0, high=self.game.sw, shape=(2, num_targets), dtype=np.float32
                ), # target relative positions (x,y)
            }
        )

        # Init. Renders
        # self.viewer = None

    def _get_obs(self):
        # Target relative positions (dx,dy)
        target_rel_positions = np.zeros(shape=(2, len(self.game.targets)), dtype=np.float32)
        for i, target in enumerate(self.game.targets):
            dx = target.x - self.game.robot.x 
            dy = target.y - self.game.robot.y
            target_rel_positions[0][i] = dx
            target_rel_positions[1][i] = dy

        # Agent x,y,angle
        agent = np.array([self.game.robot.x, self.game.robot.y, self.game.robot.angle]).astype(np.float32)

        # Print the observation
        # print(observation, end='\r')
        # print("                                                  ", end='\r')
        
        return {"agent": agent, "targets": target_rel_positions}

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
        dS = (self.game.targets[0].x - self.game.robot.x) + (self.game.targets[0].y - self.game.robot.y)
        norm_reward = dS / (self.game.sw + self.game.sh)
        reward = 100 * math.exp(5*(norm_reward-1)) # (y=100 e^{5(x-1)})

        # reward = np.sum(self.game.current_confidences)
        # if action != 0: # if action is not do-nothing
        #     reward -= 0.5

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
            # self.game.perform_action_interactive()

            # Show info on scoreboard
            # self.game.set_scoreboard({"Reward": format(self._get_reward(action), ".2f")})
            self.game.set_scoreboard({"Reward": format(self._get_reward(action), ".2f"), "Observation": self._get_obs()})

        reward = 0
        terminated = False
        truncated = False
        info = None

        # Return reward
        if action is not None:  # First step without action, called from reset()
            # Mid-episode case
            reward = self._get_reward(action)

            if self.game.gameover:
                # End of episode case
                terminated = True
                # step_reward = -100

        return spaces.utils.flatten(self.observation_space, self._get_obs()), reward, terminated, truncated, info

    def reset(self):
        """
        Description,
            Resets the ENV.

        Returns:
            ([np.array size=(2,1) type=np.float32]): Random State
        """
        self.game.reset()
        print(self._get_obs())
        print(spaces.utils.flatten(self.observation_space, self._get_obs()))
        info = None
        return spaces.utils.flatten(self.observation_space, self._get_obs()), info

    # def render(self, mode="human"):
    #     """
    #     Description,
    #         Renders the ENV.
    #     """
    #     screen_width = 600
    #     screen_height = 400

    #     world_width = self.x_threshold * 2
    #     scale = screen_width / world_width
    #     carty = 100
    #     cartwidth = 50.0
    #     cartheight = 30.0

    #     if self.viewer is None:
    #         from gym.envs.classic_control import rendering

    #         self.viewer = rendering.Viewer(screen_width, screen_height)
    #         l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
    #         cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
    #         self.carttrans = rendering.Transform()
    #         cart.add_attr(self.carttrans)
    #         self.viewer.add_geom(cart)
    #         self.track = rendering.Line((0, carty), (screen_width, carty))
    #         self.track.set_color(0, 0, 0)
    #         self.viewer.add_geom(self.track)

    #     if self.state is None:
    #         return None

    #     x = self.state
    #     cartx = x[0] * scale + screen_width / 2.0
    #     self.carttrans.set_translation(cartx, carty)

    #     return self.viewer.render(return_rgb_array=mode == "rgb_array")

    # def close(self):
    #     """
    #     Description,
    #         Closes the rendering window if provoked.
    #     """
    #     if self.viewer:
    #         self.viewer.close()
    #         self.viewer = None