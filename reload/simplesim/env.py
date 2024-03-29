# ReLOaD Simple Simulator
#
# simplesim/env.py
#
# A simple simulator environment for feasibility testing the object detection RL
#
# When run standalone, executes a simple custom policy that moves the
# robot forwards if the current prediction score is higher than the average
# and backwards otherwise.
#
# Alex Nichoson
# 27/05/2023

import pygame
import math
import random
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from PIL import Image
import time

# ---------------------------------------------------------------------------- #
#                                  GLOBAL VARS                                 #
# ---------------------------------------------------------------------------- #
font_family = "Helvetica"

# ---------------------------------------------------------------------------- #
#                                    CLASSES                                   #
# ---------------------------------------------------------------------------- #

class Wall(object):
    """
    Environment obstacles that cannot be seen through or passed through
    """

    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.height = ymax - ymin
        self.width = xmax - xmin
        self.centre_x = xmin + self.width / 2
        self.centre_y = ymin + self.height / 2


class Target(object):
    """
    The targets that the RL agent must attempt to classify using the mobile robot
    """

    def __init__(self, rank, x, y, class_id):
        # The class of this target
        self.class_id = class_id
        # Size
        self.rank = rank
        self.size = 50 * rank
        # Position
        self.x = x
        self.y = y
        # # Orientation
        # self.angle = 90 # unit circle angle format
        # Randomise orientation
        self.angle = random.randint(0, 359)  # unit circle angle format
        self.xdir = math.cos(math.radians(self.angle))
        self.ydir = math.sin(math.radians(self.angle))


class Robot(object):
    """
    The mobile robot that the RL algorithm will control
    """

    def __init__(
        self, fov, size, starting_x, starting_y, starting_angle, env_size, num_classes
    ):
        self.env_size = env_size
        self.size = size
        self.num_classes = num_classes

        # Set position
        self.starting_x = starting_x
        self.starting_y = starting_y
        self.x = self.starting_x
        self.y = self.starting_y

        # Set starting orientation (facing upwards)
        self.starting_angle = starting_angle  # 90  # unit circle angles
        self.angle = self.starting_angle  # unit circle angles

        # Set move speeds
        self.turn_rate_discrete = 5
        self.move_rate_discrete = 6
        self.turn_rate_continuous = 20
        self.move_rate_continuous = 20

        self.trail = []  # a trail of all past x,y coords
        # Draw player fov indicator
        self.fov = fov

        if self.fov > 180:
            raise ValueError("FOV must be <= 180 (required by can_see())")

    def set_rot_velocity(self, omega):
        # Left is +ve
        move_angle = omega * self.turn_rate_continuous
        self.angle += move_angle
        if self.angle < 0:
            self.angle = self.angle + 360  # wrap angle back around

    def set_velocity(self, v):
        # Forward is +ve
        move_dist = v * self.move_rate_continuous
        self.x += math.cos(math.radians(self.angle)) * move_dist
        self.y -= math.sin(math.radians(self.angle)) * move_dist
        self.trail.append((self.x, self.y))

    def turn_left(self):
        self.angle += self.turn_rate_discrete
        if self.angle >= 360:
            self.angle = self.angle - 360  # wrap angle back around

    def turn_right(self):
        self.angle -= self.turn_rate_discrete
        if self.angle < 0:
            self.angle = self.angle + 360  # wrap angle back around

    def move_forward(self):
        self.x += math.cos(math.radians(self.angle)) * self.move_rate_discrete
        self.y -= math.sin(math.radians(self.angle)) * self.move_rate_discrete
        self.trail.append((self.x, self.y))

    def move_backward(self):
        self.x -= math.cos(math.radians(self.angle)) * self.move_rate_discrete
        self.y += math.sin(math.radians(self.angle)) * self.move_rate_discrete
        self.trail.append((self.x, self.y))


class SimpleSim(object):
    """
    A class to handle all of the data structures and logic of the game
    """

    def __init__(
        self,
        max_budget,
        max_targets,
        num_classes,
        player_fov,
        action_format,
        seed=None,
        render_mode=None,
        render_fps=None,
        render_plots=True,
    ):
        """
        Initialises a SimpleSim instance

        Args:
            max_budget (int): The maximum budget the agent will start each
                episode of the game with.
            max_targets (int): The maximum number of targets in the game for
                the agent to identify.
            player_fov (int): The FOV of the robot (in degrees)
        Returns:
            None
        """
        # Seed the RNGs
        if not seed:
            seed = random.randint(0, 1000)
        print(f"Environment seed: {seed}")
        np.random.seed(seed=seed)
        random.seed(seed)

        # Game state info
        self.env_size = 500  # number of coords size of the env
        self.player_size = 50
        self.target_size = 50

        self.max_budget = max_budget
        self.max_targets = max_targets
        self.num_classes = num_classes
        self.num_walls = 0
        self.player_fov = player_fov

        self.action_format = action_format

        # Rendering info
        self.render_mode = render_mode
        self.render_fps = render_fps
        self.window = None
        self.clock = None
        self.paused = False
        self.display_scale = 1.5  # display size in pixels
        self.display_env_size = self.env_size * self.display_scale
        self.display_player_size = self.player_size * self.display_scale
        self.display_target_size = self.target_size * self.display_scale
        self.scoreboard_items = {}
        self.render_plots = render_plots

        # Get class names
        text_file = open("classlist.txt", "r")
        self.class_names = [line.strip() for line in text_file.readlines()]
        text_file.close()
        self.class_names = self.class_names[
            0 : self.num_classes
        ]  # truncate to however many classes we have


        # Reset and re-randomise environment conditions
        self.count = 0
        self.gameover = False

        # Reset game state to initial
        self.reset()

    def spawn_walls(self, num_to_spawn):
        """
        Spawns wall obstacles
        """

        min_size = 50
        max_size = 200

        walls = []
        for _ in range(0, num_to_spawn):
            xmin, ymin = (
                random.randrange(0, self.env_size - min_size),
                random.randrange(0, self.env_size - min_size),
            )
            xmax, ymax = (
                random.randrange(xmin + min_size, min(self.env_size, xmin + max_size)),
                random.randrange(ymin + min_size, min(self.env_size, ymin + max_size)),
            )

            walls.append(Wall(xmin, ymin, xmax, ymax))

        return walls

    def spawn_robot(self, player_fov):
        """
        Spawns the robot / player
        """
        # Spawn robot at centre
        # x, y = (self.env_size//2, self.env_size//2)

        # Spawn robot at random position and orientation
        while True:
            x, y = (
                random.randrange(
                    0 + self.player_size // 2, self.env_size - self.player_size // 2
                ),
                random.randrange(
                    0 + self.player_size // 2, self.env_size - self.player_size // 2
                ),
            )

            valid = True
            for wall in self.walls:
                # If any part of robot is inside wall
                if (
                    (x + (self.player_size / 2) > wall.xmin)
                    and (y + (self.player_size / 2) > wall.ymin)
                    and (x - (self.player_size / 2) < wall.xmax)
                    and (y - (self.player_size / 2) < wall.ymax)
                ):
                    # Point is inside wall
                    valid = False

            if valid:
                break

        theta = random.randrange(0, 360)

        return Robot(
            player_fov,
            self.player_size,
            x,
            y,
            theta,
            self.env_size,
            self.num_classes,
        )

    def spawn_targets(self, num_to_spawn):
        """
        Spawn objects to identify

        Args:
            num_to_spawn (int): Number of targets to spawn in the game
        Returns:
            None
        """
        targets = []
        for _ in range(0, num_to_spawn):
            # rank = random.choice([1, 1, 1, 2, 2, 3])
            rank = 1
            size = self.target_size * rank
            target_class_id = random.randint(0, self.num_classes - 1)

            while True:
                x, y = (
                    random.randrange(0 + size // 2, self.env_size - size // 2),
                    random.randrange(0 + size // 2, self.env_size - size // 2),
                )

                valid = True
                for wall in self.walls:
                    if (
                        (x + (size / 2) > wall.xmin)
                        and (y + (size / 2) > wall.ymin)
                        and (x - (size / 2) < wall.xmax)
                        and (y - (size / 2) < wall.ymax)
                    ):
                        # Inside wall
                        valid = False
                if valid:
                    # If target spawn wasnt inside and walls
                    break

            targets.append(Target(rank, x, y, target_class_id))
        return targets

    def can_see(self, target: Target):
        """
        Checks if the robot can see this target

        Note: Everything in this function is done in cartesian coordinates not
        screen pixel coordinates (bottom left = (0,0), angles measuresd
        anti-clockwise from right).
        """

        # Convert pixel coordinated to cartesian coordinates
        robot_x_cart = self.robot.x
        robot_y_cart = self.env_size - self.robot.y
        target_x_cart = target.x
        target_y_cart = self.env_size - target.y

        # First check if the view is obscured by a wall
        if self.is_obscured(target):
            return False

        left_angle = (self.robot.angle + (self.robot.fov / 2)) % 360
        right_angle = (self.robot.angle - (self.robot.fov / 2)) % 360
        target_dy = target_y_cart - robot_y_cart
        target_dx = target_x_cart - robot_x_cart

        target_angle = np.degrees(
            np.arctan2(target_dy, target_dx)
        )  # angle of ray to target
        target_angle = target_angle + 360 if (target_angle < 0) else target_angle

        if (target_angle <= left_angle) and (target_angle >= right_angle):
            #  Normal case
            return True
        elif (left_angle < self.robot.fov) and (right_angle > 360 - self.robot.fov):
            #  Angle-zero-crossing case
            if (target_angle <= left_angle) and (target_angle + 360 >= right_angle):
                # Target angle after 0 crossover and inside
                return True
            elif (target_angle <= left_angle + 360) and (target_angle >= right_angle):
                # Target angle before 0 crossover and inside
                return True
            else:
                return False
        else:
            return False

    def is_obscured(self, target: Target):
        """
        Checks if target is obscured from robot by wall
        """

        # First check if the view is obscured by a wall
        blocked = False
        wall: Wall
        for wall in self.walls:
            point_a1 = (self.robot.x, self.robot.y)
            point_a2 = (target.x, target.y)
            # Get the wall's sight-blocking-lines (the diagonals either
            # top-L to bot-R or bot-L to top-R)
            point_b1 = (wall.xmin, wall.ymax)
            point_b2 = (wall.xmax, wall.ymin)
            point_c1 = (wall.xmin, wall.ymin)
            point_c2 = (wall.xmax, wall.ymax)

            # Check is sight line is blocked by blocking line
            blocked_1 = self.lines_intersect(point_a1, point_a2, point_b1, point_b2)
            blocked_2 = self.lines_intersect(point_a1, point_a2, point_c1, point_c2)
            blocked = blocked or blocked_1 or blocked_2

        return blocked

    def get_confidence(self, target: Target):
        """
        Computes the simulated YOLO confidence of an object in view
        """
        if self.can_see(target):
            # Convert pixel coordinated to cartesian coordinates
            robot_x_cart = self.robot.x
            robot_y_cart = self.env_size - self.robot.y
            target_x_cart = target.x
            target_y_cart = self.env_size - target.y
            target_xdir_cart = target.xdir
            target_ydir_cart = -1 * target.ydir

            # Get the distances and vectors from the target to the robot
            robot_dy = robot_y_cart - target_y_cart
            robot_dx = robot_x_cart - target_x_cart
            target_to_robot_vect = np.array([robot_dy, robot_dx])
            target_orientation_vect = np.array([target_ydir_cart, target_xdir_cart])

            # A factor representing the distance difficulty (should be between 0 and 1)
            max_dist = math.sqrt(self.env_size**2 + self.env_size**2)
            target_dist = math.sqrt(robot_dy**2 + robot_dx**2)

            # distance_factor = 1 - (target_dist / max_dist) # linear distance factor
            falloff_steepness = 3  # was 5 but too steep
            distance_factor = np.exp(
                -(falloff_steepness * (target_dist / max_dist))
            )  # exponential distance factor

            # A factor representing the orientation difficulty (should be between 0 and 1)
            angle_between = np.degrees(
                np.arccos(
                    target_to_robot_vect.dot(target_orientation_vect)
                    / (
                        np.linalg.norm(target_to_robot_vect)
                        * np.linalg.norm(target_orientation_vect)
                    )
                )
            )
            orientation_factor = 1 - (angle_between / 180)

            # Decide the weightings between how much the distance and orientation
            # factors affect the confidence
            distance_weighting = 4
            orientation_weighting = 1
            weighting_sum = distance_weighting + orientation_weighting
            distance_weighting = distance_weighting / weighting_sum
            orientation_weighting = orientation_weighting / weighting_sum

            # Compute the final confidence
            true_confidence = (distance_factor * distance_weighting) + (
                orientation_factor * orientation_weighting
            )

            # Add gaussian noise (simulated confusion) to the true class probability
            std_dev = 0.1  # was 0.1
            noisy_true_confidence = np.clip(
                true_confidence + np.random.normal(0, std_dev), 0, 1
            )

            # Then split the remaining false-class probability across the other classes
            false_confidences = np.random.rand(self.num_classes - 1)  # random numbers
            false_confidences = (false_confidences / np.sum(false_confidences)) * (
                1 - true_confidence
            )  # normalise sum to desired

            # Combine into full confidence vector
            confidence = np.concatenate(
                [
                    false_confidences[0 : target.class_id],
                    [noisy_true_confidence],
                    false_confidences[target.class_id : len(false_confidences)],
                ]
            )
            return confidence

        else:
            # Return 0 confidences if we know the target is not in view
            confidence = np.zeros(shape=(self.num_classes,))
            return confidence

    def lines_intersect(self, point_a1, point_a2, point_b1, point_b2):
        """
        Check if line segments a1-a2 and b1-b2 intersect
        """
        x00, y00 = point_a1
        x01, y01 = point_a2

        x10, y10 = point_b1
        x11, y11 = point_b2

        d = (x11 - x10) * (y01 - y00) - (x01 - x00) * (y11 - y10)
        if d == 0:
            # Lines are parallel therefore don't intersect
            return False

        s = (1 / d) * ((x00 - x10) * (y01 - y00) - (y00 - y10) * (x01 - x00))
        t = (1 / d) * -(-(x00 - x10) * (y11 - y10) + (y00 - y10) * (x11 - x10))

        if (s > 0 and s < 1) and (t > 0 and t < 1):
            # Lines intersect
            return True
        else:
            return False

    def detect_targets(self):
        """
        Looks for objects in the view of the player and detects them

        Args:
            None
        Returns:
            None
        """

        # Get simulated confidences
        for i, target in enumerate(self.targets):
            self.current_confidences[i] = self.get_confidence(target)

        # Add the current timestep confidences to the comprehensive all timesteps list
        self.confidences = np.append(
            self.confidences, np.expand_dims(self.current_confidences, 2), axis=2
        )

    def set_scoreboard(self, scoreboard_items):
        self.scoreboard_items = scoreboard_items
        if self.render_mode == "human":
            self._render_frame(self.render_mode)

    def handle_boundary_collisions(self):
        """
        Stops player from going off screen or going into walls
        """
        # Handle off-screen left/right
        if self.robot.x > self.env_size - (self.robot.size / 2):
            self.robot.x = self.env_size - (self.robot.size / 2)
        elif self.robot.x < 0 + (self.robot.size / 2):
            self.robot.x = 0 + (self.robot.size / 2)

        # Handle off-screen up/down
        if self.robot.y > self.env_size - (self.robot.size / 2):
            self.robot.y = self.env_size - (self.robot.size / 2)
        elif self.robot.y < 0 + (self.robot.size / 2):
            self.robot.y = 0 + (self.robot.size / 2)

        # Handle wall collisions
        wall: Wall
        for wall in self.walls:
            # If part of the robot is inside wall
            if (
                (self.robot.x + (self.robot.size / 2) > wall.xmin)
                and (self.robot.y + (self.robot.size / 2) > wall.ymin)
                and (self.robot.x - (self.robot.size / 2) < wall.xmax)
                and (self.robot.y - (self.robot.size / 2) < wall.ymax)
            ):
                dx = self.robot.x - wall.centre_x
                dy = self.robot.y - wall.centre_y

                # Bump the robot back out towards the closest wall
                if abs(dx) > abs(dy):
                    # Bump in x direction
                    if dx < 0:
                        # Bump x to -ve direction
                        self.robot.x = wall.xmin - (self.robot.size / 2)
                    else:
                        # Bump x to +ve direction
                        self.robot.x = wall.xmax + (self.robot.size / 2)
                else:
                    # Bumpy in y direction
                    if dy < 0:
                        # Bump y to -ve direction
                        self.robot.y = wall.ymin - (self.robot.size / 2)
                    else:
                        # Bump y to +ve direction
                        self.robot.y = wall.ymax + (self.robot.size / 2)

    def perform_action(self, action):
        """
        Given an action tuple, execute the action in the environment.
        Actions 0,1,2,3,4 - None,R,L,F,B
        """

        # Discrete actions
        if self.action_format == "discrete":
            # Handle agent controls and movement (note action 0 does nothing)
            if action == 1:
                self.robot.turn_right()
            elif action == 2:
                self.robot.move_forward()
            if action == 3:
                self.robot.turn_left()
            elif action == 4:
                self.robot.move_backward()

        # Continuous actions
        elif self.action_format == "continuous":
            # Handle agent controls and movement (note action 0 does nothing)
            self.robot.set_velocity(float(action[0]))
            self.robot.set_rot_velocity(float(action[1]))

        # Stop the robot going off the screen or inside walls
        self.handle_boundary_collisions()

    def get_action_interactive(self):
        """
        Get player commands from keyboard and return the relevant action vector.
        """

        # Handle player controls and movement
        if self.action_format == "discrete":
            # Discrete actions
            while True:
                event = pygame.event.wait()
                if (not self.paused) and (not self.gameover):
                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_RIGHT]:
                        return 1
                    elif keys[pygame.K_UP]:
                        return 2
                    elif keys[pygame.K_LEFT]:
                        return 3
                    elif keys[pygame.K_DOWN]:
                        return 4
                    elif keys[pygame.K_SPACE]:
                        return 0

        elif self.action_format == "continuous":
            # Continuous actions
            while True:
                event = pygame.event.wait()
                action = np.zeros(shape=(2, 1))
                if (not self.paused) and (not self.gameover):
                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_RIGHT]:
                        action[1] = -1
                        break
                    if keys[pygame.K_UP]:
                        action[0] = 1
                        break
                    if keys[pygame.K_LEFT]:
                        action[1] = 1
                        break
                    if keys[pygame.K_DOWN]:
                        action[0] = -1
                        break
                    if keys[pygame.K_SPACE]:
                        break
            return action

    def step(self, action):
        """
        Runs the game logic (controller)
        """
        if not self.gameover:
            # self.clock.tick(600)
            self.count += 1

            # Peform the given action and pdate the player positions
            self.perform_action(action)

            # Decrement the budget over time
            self.budget -= 1

            # Get the detection confidences on the environment
            self.detect_targets()

            # Check gameover
            if self.budget <= 0:
                self.gameover = True

        if self.render_mode == "human":
            # Handle menu keyboard events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_TAB:
                        if self.gameover:
                            self.reset()
                    if event.key == pygame.K_p:
                        self.paused = not self.paused
                    if event.key == pygame.K_v:
                        self.visualize = not self.visualize
                        # Set the screen to black
                        self.render_black_screen()

            while self.paused:
                # Do nothing, pause until key pressed again
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                        self.paused = not self.paused
                        break

            self._render_frame(self.render_mode)

    def save_obs_distribution(self):
        """
        Saves the distribution of observations across all of the targets
        """
        # print(np.shape(self.confidences), self.confidences)

        # over all timesteps, per class and target
        num_observations = np.count_nonzero(self.confidences, axis=2)
        # print(np.shape(num_observations), num_observations)
        
        # over all timesteps and classes, per target
        num_observations = num_observations[:, 0]
        # print(np.shape(num_observations), num_observations)
        
        out_file = open("obs_dist.csv", "a")  # append mode
        
        total_num_obs = np.sum(num_observations)
        for target_num_obs in num_observations:
            if total_num_obs == 0:
                continue

            outstr = f"{self.num_targets}, {self.starting_budget}, {target_num_obs}, {target_num_obs/self.starting_budget}, {total_num_obs}, {target_num_obs/total_num_obs}"
            out_file.write(outstr+"\n")

        # print(data)
        # print(outstr)

        out_file.close()

        # quit()

    def save_images(self):
        """
        Saves images of the current environment and histograms to file
        """
        current_time = time.time()

        # print("Saving images...")
        # Save env image
        frame = self._render_frame("rgb_array")
        env_image = Image.fromarray(frame)
        # print(f"out/{time.time()}_environment.png")
        env_image.save(f"out/{current_time}_environment.pdf")

        # Save histogram image
        plt.savefig(f"out/{current_time}_histograms.pdf")


    def reset(self):
        """
        Resets the game to its initial state

        Args:
            None
        Returns:
            None
        """
        # if self.gameover == True:
            # Before we reset, save an image of the final environment and histograms
            # self.save_images()
            # Before we reset, save the obs distribution
            # self.save_obs_distribution()

        # Reset and re-randomise environment conditions
        self.count = 0
        self.gameover = False
        # Randomise the budget between 0 and a max of max_budget
        self.starting_budget = 1 + round((self.max_budget - 1) * random.random())
        self.budget = self.starting_budget
        # Randomise the number of targets between 0 and a max of max_budget
        self.num_targets = 1 + round((self.max_targets - 1) * random.random())
        # self.num_targets = 5
        
        # Re-spawn entites
        self.walls = self.spawn_walls(self.num_walls)
        self.robot = self.spawn_robot(self.player_fov)
        self.targets = self.spawn_targets(self.num_targets)

        # Reset information gathering vectors
        self.current_confidences = np.zeros(
            (self.num_targets, self.num_classes), dtype=np.float32
        )
        self.confidences = np.zeros(
            (self.num_targets, self.num_classes, 1), dtype=np.float32
        )

        # Re-setup confidence histograms
        if self.render_mode == "human" and self.render_plots == True:
            try:
                self.plot.close()
            except Exception:
                pass
            self.plot = Plot(self.num_targets, self.num_classes, self.class_names)

        self.render()

    def render(self):
        if self.render_mode != None:
            frame = self._render_frame(self.render_mode)
            if self.render_mode == "rgb_array":
                return frame

    def _render_frame(self, render_mode):
        pygame.init()
        if render_mode == "human":
            if self.window is None:
                pygame.display.init()
                self.window = pygame.display.set_mode(
                    (self.display_env_size, self.display_env_size)
                )
                pygame.display.set_caption("ReLOaD Simulator")
            if self.clock is None:
                self.clock = pygame.time.Clock()

        # ------------------ Create the base canvas and load assets ------------------ #
        SPRITE_SIZE_MULTIPLEIER = 1.2

        canvas = pygame.Surface((self.display_env_size, self.display_env_size))

        if os.path.dirname(__file__) != "":
            sprites_dir = os.path.dirname(__file__) + "/sprites/"
        else:
            sprites_dir = "sprites/"

        # bg = pygame.transform.scale(
        #     pygame.image.load(sprites_dir + "roombg.jpg"),
        #     (self.display_env_size, self.display_env_size),
        # )
        player_robot = pygame.transform.scale(
            pygame.image.load(sprites_dir + "robot.png"),
            (
                self.display_player_size * SPRITE_SIZE_MULTIPLEIER,
                self.display_player_size * SPRITE_SIZE_MULTIPLEIER,
            ),
        )

        target_size = 50

        # Load class sprites
        class_sprites = []
        for class_id in range(0, self.num_classes):
            class_sprites.append(
                pygame.transform.scale(
                    pygame.image.load(
                        sprites_dir + f"{self.class_names[class_id]}.png"
                    ),
                    (
                        target_size * self.display_scale * SPRITE_SIZE_MULTIPLEIER,
                        target_size * self.display_scale * SPRITE_SIZE_MULTIPLEIER,
                    ),
                )
            )

        fov_color = (0, 0, 255, 70)
        fov_line_length = int(500 * self.display_scale)
        fov_line_thickness = int(5 * self.display_scale)
        font = pygame.font.SysFont(font_family, int(15 * self.display_scale))
        bold_font = pygame.font.SysFont(
            font_family, int(20 * self.display_scale), bold=True
        )

        # --------------------------- Draw all the entities -------------------------- #
        # # Draw the background image
        # canvas.blit(bg, (0, 0))
        # White background
        pygame.draw.rect(
            canvas,
            (255, 255, 255),
            pygame.Rect(0, 0, self.display_env_size, self.display_env_size),
        )

        # Draw the robot
        # Make a surface with a line on it
        fov_surf = pygame.Surface(
            (self.display_env_size, self.display_env_size), pygame.SRCALPHA
        )
        fov_surf.set_colorkey((0, 0, 0, 0))
        rotated_fov_surf = pygame.transform.rotate(fov_surf, self.robot.angle)
        pygame.draw.line(
            fov_surf,
            fov_color,
            (
                fov_surf.get_rect().w / 2,
                fov_surf.get_rect().h / 2,
            ),
            (
                (fov_surf.get_rect().w / 2)
                - (fov_line_length * math.sin(math.radians(self.robot.fov / 2))),
                (fov_surf.get_rect().h / 2)
                - (fov_line_length * math.cos(math.radians(self.robot.fov / 2))),
            ),
            fov_line_thickness,
        )
        pygame.draw.line(
            fov_surf,
            fov_color,
            (
                fov_surf.get_rect().w / 2,
                fov_surf.get_rect().h / 2,
            ),
            (
                (fov_surf.get_rect().w / 2)
                + (fov_line_length * math.sin(math.radians(self.robot.fov / 2))),
                (fov_surf.get_rect().h / 2)
                - (fov_line_length * math.cos(math.radians(self.robot.fov / 2))),
            ),
            fov_line_thickness,
        )

        # Update player sprite position
        # 90deg rotated version of the sprite surf image
        rotated_player_surf = pygame.transform.rotate(
            player_robot, self.robot.angle - 90
        )
        # The rectangle bounding box of the surf
        rotated_player_rect = rotated_player_surf.get_rect()
        # Set the centre position of the surf to the player position vars
        rotated_player_rect.center = (
            self.robot.x * self.display_scale,
            self.robot.y * self.display_scale,
        )
        # Update fov indicator lines position
        rotated_fov_surf = pygame.transform.rotate(fov_surf, self.robot.angle - 90)
        rotated_fov_rect = rotated_fov_surf.get_rect()
        rotated_fov_rect.center = (
            self.robot.x * self.display_scale,
            self.robot.y * self.display_scale,
        )

        # Redraw the player surfs
        canvas.blit(rotated_player_surf, rotated_player_rect)
        # Redraw the fov indicator surfs
        canvas.blit(rotated_fov_surf, rotated_fov_rect)
        # pygame.draw.circle(canvas, (0, 255, 0), (self.robot.x, self.robot.y), 5) # robot centre

        # Draw the targets
        for i, target in enumerate(self.targets):
            # Set the sprite and hitbox size
            # if target.rank == 1:
            #     image = target50
            # elif target.rank == 2:
            #     image = target100
            # else:
            #     image = target150

            # Setup Render Surfs
            rotated_target_surf = pygame.transform.rotate(
                class_sprites[target.class_id], target.angle - 90
            )
            rotated_target_rect = rotated_target_surf.get_rect()
            rotated_target_rect.center = (
                target.x * self.display_scale,
                target.y * self.display_scale,
            )
            canvas.blit(rotated_target_surf, rotated_target_rect)
            # Draw target centre
            circle_width = 8 * self.display_scale
            pygame.draw.circle(
                canvas,
                (255, 255, 255),
                (target.x * self.display_scale, target.y * self.display_scale),
                circle_width,
            )
            # Draw target number
            target_text = bold_font.render(str(i), 2, (0, 0, 0))
            canvas.blit(
                target_text,
                (
                    target.x * self.display_scale - target_text.get_width() // 2,
                    target.y * self.display_scale - target_text.get_width() // 2 - 3,
                ),
            )

        # Draw the walls
        wall: Wall
        for wall in self.walls:
            # Setup Render Surfs
            pygame.draw.rect(
                canvas,
                (0, 0, 0),
                pygame.Rect(
                    wall.xmin * self.display_scale,
                    wall.ymin * self.display_scale,
                    wall.width * self.display_scale,
                    wall.height * self.display_scale,
                ),
            )

        # Draw the robot's trail
        for point in self.robot.trail:
            x, y = point

            pygame.draw.circle(
                canvas,
                (255, 0, 0),
                (x * self.display_scale, y * self.display_scale),
                2 * self.display_scale,
            )

        # Draw the onscreen menu text
        # play_again_text = font.render("Press Tab to Play Again", 1, (0, 0, 0))
        # if self.gameover:
        #     canvas.blit(
        #         play_again_text,
        #         (
        #             self.env_size // 2 - play_again_text.get_width() // 2,
        #             self.env_size // 2 - play_again_text.get_height() // 2,
        #         ),
        #     )

        pause_text = font.render("Press P to Unpause", 1, (0, 0, 0))
        if self.paused:
            canvas.blit(
                pause_text,
                (
                    self.env_size // 2 - pause_text.get_width() // 2,
                    self.env_size // 2 - pause_text.get_height() // 2,
                ),
            )

        # Render the socreboard info
        self.render_scoreboard(canvas, font)

        if render_mode == "human":
            if self.render_plots == True:
                # Render the matplotlib histograms
                confidences_sum = np.sum(self.confidences, axis=2)
                num_observations = np.count_nonzero(self.confidences, axis=2)
                all_time_avg = np.divide(
                    confidences_sum,
                    num_observations,
                    where=(num_observations > 0),
                    out=np.full_like(confidences_sum, 1 / self.num_classes),
                )
                self.plot.update(all_time_avg)

            # --------------- Send the rendered view to the relevant viewer -------------- #
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.render_fps)

        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def render_scoreboard(self, canvas, font):
        metric_count = 0
        budget_text = font.render("Budget: " + str(self.budget), 1, (0, 0, 0))
        canvas.blit(
            budget_text,
            (
                # self.env_size * self.disply_scale - budget_text.get_width() - 25,
                25,
                (metric_count * 35) + budget_text.get_height(),
            ),
        )

        metric_count += 1
        for metric_name in self.scoreboard_items.keys():
            item = self.scoreboard_items[metric_name]

            metric_text = font.render(
                f"{metric_name}: " + str(item),
                1,
                (0, 0, 0),
            )
            canvas.blit(
                metric_text,
                (
                    # self.env_size * self.disply_scale - metric_text.get_width() - 25,
                    25,
                    (metric_count * 35) + metric_text.get_height(),
                ),
            )
            metric_count += 1


class Plot(object):
    """
    A plotting util class
    """

    def __init__(self, num_targets, num_classes, class_names):
        self.num_targets = num_targets
        self.num_classes = num_classes
        self.x = class_names
        self.cols = math.ceil(math.sqrt(num_targets))
        self.rows = math.ceil(num_targets / self.cols)
        # print(f"cols:{self.cols}, rows:{self.rows}")
        self.bars = []
        self.bars_text = []

        # Create figure and customise formatting
        plt.ion()
        plt.rc("font", family=font_family)
        # plt.style.use('ggplot')
        new_cmap = self.rand_cmap(
            100, type="bright", first_color_black=False, last_color_black=False
        )
        self.fig, self.ax = plt.subplots(self.rows, self.cols)
        self.fig.suptitle(
            f"Target Confidence Distributions", color="#333333"
        )  # , weight='bold')
        if self.rows == 1:
            self.ax = np.expand_dims(self.ax, axis=0)
            if self.cols == 1:
                self.ax = np.expand_dims(self.ax, axis=1)

        # Create Subplots
        target_index = 0
        for row in range(0, self.rows):
            for col in range(0, self.cols):
                # print(f"col:{col}, row:{row}")
                if target_index > self.num_targets - 1:
                    self.ax[row, col].remove()
                    break

                self.bars.append(
                    self.ax[row, col].bar(
                        x=self.x,
                        height=np.full_like(self.x, 1 / num_classes),
                        tick_label=class_names,
                    )
                )  # Returns a tuple of line objects, thus the comma
                self.ax[row, col].set_xticklabels(self.x, rotation=90, ha="right")

                # Axis formatting.
                self.ax[row, col].set_ylim(0, 1)
                self.ax[row, col].spines["top"].set_visible(False)
                self.ax[row, col].spines["right"].set_visible(False)
                self.ax[row, col].spines["left"].set_visible(False)
                self.ax[row, col].spines["bottom"].set_color("#DDDDDD")
                self.ax[row, col].tick_params(bottom=False, left=False)
                self.ax[row, col].set_axisbelow(True)
                self.ax[row, col].yaxis.grid(True, color="#EEEEEE")
                self.ax[row, col].xaxis.grid(False)

                # Add text annotations to the top of the bars.
                bar_text = []
                bar_index = 0
                for bar in self.bars[target_index]:
                    bar_text.append(
                        self.ax[row, col].text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.1,
                            round(bar.get_height(), 1),
                            horizontalalignment="center",
                            weight="bold",
                            size=6,
                        )
                    )

                    bar.set_color(new_cmap(bar_index))
                    bar_index += 1

                self.bars_text.append(bar_text)

                # Add labels and a title.
                self.ax[row, col].set_xlabel(
                    "Class Index", labelpad=15, color="#333333"
                )
                self.ax[row, col].set_ylabel(
                    "Average Confidence", labelpad=15, color="#333333"
                )
                self.ax[row, col].set_title(
                    f"Target {target_index}", pad=15, color="#333333"
                )

                target_index += 1

        self.fig.tight_layout()

    def update(self, data):
        for i in range(0, self.num_targets):
            for j in range(0, self.num_classes):
                # Update the bar height
                self.bars[i][j].set_height(data[i][j])
                # Update the top-of-bar value labels
                self.bars_text[i][j].set_position(
                    (
                        self.bars[i][j].get_x() + self.bars[i][j].get_width() / 2,
                        self.bars[i][j].get_height() + 0.1,
                    )
                )
                self.bars_text[i][j].set_text(round(self.bars[i][j].get_height(), 2))

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        # Closes the current plot fig
        plt.close(self.fig)

    def rand_cmap(
        self,
        nlabels,
        type="bright",
        first_color_black=True,
        last_color_black=False,
        verbose=0,
    ):
        """
        Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks.

        Args:
            nlabels [int]: Number of labels (size of colormap)
            type [string]: 'bright' for strong colors, 'soft' for pastel colors
            first_color_black [bool]: Option to use first color as black, True or False
            last_color_black [bool]: Option to use last color as black, True or False
            verbose [int]: Prints the number of labels and shows the colormap. True or False

        Returns:
            colormap for matplotlib

        Author: Felipe Delestro Matos (https://github.com/delestro/rand_cmap)
        """
        from matplotlib.colors import LinearSegmentedColormap
        import colorsys
        import numpy as np

        if type not in ("bright", "soft"):
            print('Please choose "bright" or "soft" for type')
            return

        if verbose > 0:
            print("Number of labels: " + str(nlabels))

        # Generate color map for bright colors, based on hsv
        if type == "bright":
            randHSVcolors = [
                (
                    np.random.uniform(low=0.0, high=1),
                    np.random.uniform(low=0.2, high=1),
                    np.random.uniform(low=0.9, high=1),
                )
                for i in range(nlabels)
            ]

            # Convert HSV list to RGB
            randRGBcolors = []
            for HSVcolor in randHSVcolors:
                randRGBcolors.append(
                    colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2])
                )

            if first_color_black:
                randRGBcolors[0] = [0, 0, 0]

            if last_color_black:
                randRGBcolors[-1] = [0, 0, 0]

            random_colormap = LinearSegmentedColormap.from_list(
                "new_map", randRGBcolors, N=nlabels
            )

        # Generate soft pastel colors, by limiting the RGB spectrum
        if type == "soft":
            low = 0.6
            high = 0.95
            randRGBcolors = [
                (
                    np.random.uniform(low=low, high=high),
                    np.random.uniform(low=low, high=high),
                    np.random.uniform(low=low, high=high),
                )
                for i in range(nlabels)
            ]

            if first_color_black:
                randRGBcolors[0] = [0, 0, 0]

            if last_color_black:
                randRGBcolors[-1] = [0, 0, 0]
            random_colormap = LinearSegmentedColormap.from_list(
                "new_map", randRGBcolors, N=nlabels
            )

        # Display colorbar
        if verbose > 0:
            from matplotlib import colors, colorbar
            from matplotlib import pyplot as plt

            fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

            bounds = np.linspace(0, nlabels, nlabels + 1)
            norm = colors.BoundaryNorm(bounds, nlabels)

            cb = colorbar.ColorbarBase(
                ax,
                cmap=random_colormap,
                norm=norm,
                spacing="proportional",
                ticks=None,
                boundaries=bounds,
                format="%1i",
                orientation="horizontal",
            )

        return random_colormap


class NaivePolicy(object):
    def __init__(
        self, game
    ):
        self.targets: list = list(game.targets)
        self.unvisited: list = list(game.targets)
        self.current_target: Target = None
        self.time_per_target: int = game.budget // game.num_targets
        self.time_on_target: int = 0 # the time spent observing the current target

    def get_action(self, robot):
        robot_x = robot.x
        robot_y = robot.y
        robot_theta = robot.angle
        
        turn_rate_continuous = 20
        move_rate_continuous = 20

        if len(self.unvisited) == 0:
            # If we run out of targets before time's up, just stay still
            return (0,0)

        # Check if we should swap targets
        if self.time_on_target > self.time_per_target or self.current_target == None:
            # Switch targets
            # Remove the current target from the unvisited list
            if self.current_target:
                self.unvisited.remove(self.current_target)

            # Get the next closest target in the unvisited list
            closest_dist = sys.maxsize
            for target in self.unvisited:
                dist = self.get_distance((target.x,target.y), (robot_x,robot_y))
                if dist < closest_dist:
                    closest_dist = dist
                    self.current_target = target
            self.time_on_target = 1


        # Now check what our movement action should be
        angle_to_target = self.get_angle_between((self.current_target.x,self.current_target.y), (robot_x,robot_y))
        
        # Get the diff the robot needs to turn to get to the target 
        if angle_to_target < 0:
            angle_to_target = 360 + angle_to_target
        angle_diff = angle_to_target - robot_theta
        if angle_diff > 180:
            angle_diff = -360 + angle_diff
        if angle_diff < -180:
            angle_diff = 360 + angle_diff

        if angle_diff != 0:
            # If we are not facing towards the target already, turn towards the target
            if angle_diff > 0:
                # Turn right
                action = (0, min(1, angle_diff/turn_rate_continuous))
            else:
                # Turn left
                action = (0, max(-1, angle_diff/turn_rate_continuous))

        else:
            # If we are facing towards the target
            distance = self.get_distance((self.current_target.x,self.current_target.y), (robot_x,robot_y))
            if distance > 20:
                # If we are not near the target, drive towards it at full speed
                action = (min(1, distance/move_rate_continuous), 0)
            else:
                # Stay here and observe (no action)
                action = (0, 0)

        self.time_on_target += 1

        return action
    
    def get_angle_between(self, robot_pos, target_pos):
        """
        Gets the heading angle from the robot to the target
        """
        (robot_x, robot_y) = robot_pos
        (target_x, target_y) = target_pos

        dx = - (target_x - robot_x)
        dy = target_y - robot_y

        degrees = np.angle(dx + (dy * 1j), deg=True)
        
        return degrees

    def get_distance(self, point_1, point_2):
        """
        Gets the cartesian distance between two points
        """
        return math.sqrt((point_1[0] - point_2[0])**2 + (point_1[1] - point_2[1])**2)




if __name__ == "__main__":
    # Hyperparameters
    MAX_BUDGET = 2000
    NUM_TARGETS = 8
    PLAYER_FOV = 60

    game = SimpleSim(
        max_budget=MAX_BUDGET,
        num_targets=NUM_TARGETS,
        player_fov=PLAYER_FOV,
        render_mode="human",
        render_fps=30,
    )
    game._render_frame()

    state = {}

    while game.run:
        # Step the game engine
        game.step(game.get_action_interactive())
        game._render_frame()

    pygame.quit()
