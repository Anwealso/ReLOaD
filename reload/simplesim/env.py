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

# ---------------------------------------------------------------------------- #
#                                  GLOBAL VARS                                 #
# ---------------------------------------------------------------------------- #


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

    def __init__(self, rank, x, y):
        # Size
        self.rank = rank
        self.size = 50 * rank
        # self.h = 50 * rank
        # Position
        self.x = x
        self.y = y
        # Orientation
        self.angle = 90  # unit circle angle format
        self.xdir = int(math.cos(math.radians(self.angle)))
        self.ydir = int(math.sin(math.radians(self.angle)))


class Robot(object):
    """
    The mobile robot that the RL algorithm will control
    """

    def __init__(self, fov, size, starting_x, starting_y, env_size):
        # self.image = player_robot
        self.env_size = env_size
        self.size = size

        # Set position
        self.starting_x = starting_x
        self.starting_y = starting_y
        self.x = self.starting_x
        self.y = self.starting_y

        # Set starting orientation (facing upwards)
        self.starting_angle = 90  # unit circle angles
        self.angle = self.starting_angle  # unit circle angles

        # Set move speeds
        self.turn_rate = 5
        self.move_rate = 6

        self.trail = []  # a trail of all past x,y coords
        # Draw player fov indicator
        self.fov = fov

        if self.fov > 180:
            raise ValueError("FOV must be <= 180 (required by can_see())")

    def turn_left(self):
        self.angle += self.turn_rate
        if self.angle >= 360:
            self.angle = self.angle - 360  # wrap angle back around

    def turn_right(self):
        self.angle -= self.turn_rate
        if self.angle < 0:
            self.angle = self.angle + 360  # wrap angle back around

    def move_forward(self):
        self.x += math.cos(math.radians(self.angle)) * self.move_rate
        self.y -= math.sin(math.radians(self.angle)) * self.move_rate
        self.trail.append((self.x, self.y))

    def move_backward(self):
        self.x -= math.cos(math.radians(self.angle)) * self.move_rate
        self.y += math.sin(math.radians(self.angle)) * self.move_rate
        self.trail.append((self.x, self.y))

    def get_confidence(self, target: Target):
        """
        Computes the simulated YOLO confidence of an object in view
        """
        # Convert pixel coordinated to cartesian coordinates
        robot_x_cart = self.x
        robot_y_cart = self.env_size - self.y
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
        distance_factor = 1 - (target_dist / max_dist)

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
        confidence = (distance_factor * distance_weighting) + (
            orientation_factor * orientation_weighting
        )

        return confidence


class SimpleSim(object):
    """
    A class to handle all of the data structures and logic of the game
    """

    def __init__(
        self,
        starting_budget,
        num_targets,
        player_fov,
        render_mode=None,
        render_fps=None,
    ):
        """
        Initialises a SimpleSim instance

        Args:
            starting_budget (int): The budget the agent will start each episode
                of the game with.
            num_targets (int): The number of targets in the game for the agent
                to identify.
            player_fov (int): The FOV of the robot (in degrees)
        Returns:
            None
        """

        # Rendering
        self.render_mode = render_mode
        self.render_fps = render_fps
        self.window = None
        self.clock = None
        self.window_size = 500
        self.player_size = 50
        self.target_size = 50

        # Game state
        self.starting_budget = starting_budget
        self.budget = self.starting_budget
        self.num_targets = num_targets
        self.gameover = False
        self.paused = False
        self.count = 0
        self.scoreboard_items = {}

        self.targets = []
        self.walls = []
        # 1D array of confidences on each object at current timestep
        self.current_confidences = np.zeros((self.num_targets, 1), dtype=np.float32)
        # 2D array of confidences on each object at each timestep
        self.confidences = np.zeros((self.num_targets, 1), dtype=np.float32)
        # 1D array of confidences on each object avg over all past timesteps
        self.avg_confidences = np.zeros((self.num_targets, 1), dtype=np.float32)
        # Note: Using avg might cause agent to simply find the best spot with
        # the most objects in view and camp there to farm for max score

        self.curriculum = 1  # no limit unless this member variable is set manually
        self.min_target_dist = 0  # was 80
        self.spawn_walls(1)
        self.spawn_robot(player_fov)
        self.spawn_targets(self.num_targets)

    def spawn_walls(self, num_to_spawn):
        """
        Spawns wall obstacles
        """

        min_size = 50
        max_size = 200

        for _ in range(0, num_to_spawn):
            xmin, ymin = (
                random.randrange(0, self.window_size - min_size),
                random.randrange(0, self.window_size - min_size),
            )
            xmax, ymax = (
                random.randrange(
                    xmin + min_size, min(self.window_size, xmin + max_size)
                ),
                random.randrange(
                    ymin + min_size, min(self.window_size, ymin + max_size)
                ),
            )

            self.walls.append(Wall(xmin, ymin, xmax, ymax))

    def spawn_robot(self, player_fov):
        """
        Spawns robot / player
        """

        x, y = (0, 0)
        while True:
            x, y = (
                random.randrange(
                    0 + self.player_size // 2, self.window_size - self.player_size // 2
                ),
                random.randrange(
                    0 + self.player_size // 2, self.window_size - self.player_size // 2
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
                    valid == False

            if valid:
                break

        self.robot = Robot(
            player_fov,
            self.player_size,
            x,
            y,
            self.window_size,
        )

    def spawn_targets(self, num_to_spawn):
        """
        Spawn objects to identify

        Args:
            num_to_spawn (int): Number of targets to spawn in the game
        Returns:
            None
        """
        for _ in range(0, num_to_spawn):
            # rank = random.choice([1, 1, 1, 2, 2, 3])
            rank = 1
            size = self.target_size * rank

            # If target is within allowable distance to robot, break
            max_band_gap = (
                math.sqrt(2 * (self.window_size) ** 2) - self.min_target_dist
            )  # the max width of the band between the min and max spawn limits
            current_max_target_dist = self.min_target_dist + (
                self.curriculum * max_band_gap
            )
            if current_max_target_dist == self.min_target_dist:
                current_max_target_dist += 5  # avoid infinite or very long loops

            while True:
                x, y = (
                    random.randrange(0 + size // 2, self.window_size - size // 2),
                    random.randrange(0 + size // 2, self.window_size - size // 2),
                )

                for wall in self.walls:
                    # If part of the target is inside wall
                    if (
                        (x + (size / 2) > wall.xmin)
                        and (y + (size / 2) > wall.ymin)
                        and (x - (size / 2) < wall.xmax)
                        and (y - (size / 2) < wall.ymax)
                    ):
                        break

                dS = math.sqrt((x - self.robot.x) ** 2 + (y - self.robot.y) ** 2)

                if (dS >= self.min_target_dist) and (dS <= current_max_target_dist):
                    break

            # print(f"current_max_target_dist: {current_max_target_dist}")
            self.targets.append(Target(rank, x, y))

    def can_see(self, target: Target):
        """
        Checks if the robot can see this target

        Note: Everything in this function is done in cartesian coordinates not
        screen pixel coordinates (bottom left = (0,0), angles measuresd
        anti-clockwise from right).
        """

        # Convert pixel coordinated to cartesian coordinates
        robot_x_cart = self.robot.x
        robot_y_cart = self.window_size - self.robot.y
        target_x_cart = target.x
        target_y_cart = self.window_size - target.y

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
            # Get the wall sight blocking line (either top-L bot-R or bot-L top-R)
            point_b1 = (wall.xmin, wall.ymax)
            point_b2 = (wall.xmax, wall.ymin)
            point_c1 = (wall.xmin, wall.ymin)
            point_c2 = (wall.xmax, wall.ymax)

            # Check is sight line is blocked by blocking line
            blocked_1 = self.lines_intersect(point_a1, point_a2, point_b1, point_b2)
            # print(f"blocked_fwdslash: {blocked_1}")
            blocked_2 = self.lines_intersect(point_a1, point_a2, point_c1, point_c2)
            # print(f"blocked_backslash: {blocked_2}")
            blocked = blocked or blocked_1 or blocked_2

        return blocked

    def lines_intersect(self, point_a1, point_a2, point_b1, point_b2):
        """
        Check if line segments a1-a2 and b1-b2 intersect
        """
        # print(f"LINE A:    {point_a1}    {point_a2}")
        x00, y00 = point_a1
        x01, y01 = point_a2

        # print(f"LINE B:    {point_b1}    {point_b2}")
        x10, y10 = point_b1
        x11, y11 = point_b2

        # d = x11*y01 - x01*y11
        d = (x11 - x10) * (y01 - y00) - (x01 - x00) * (y11 - y10)
        if d == 0:
            # Lines are parallel therefore don't intersect
            return False

        # s = (1/d) * ((x00 - x10)*y01 - (y00 - y10)*x01)
        # t = (1/d) * -(-(x00 - x10)*y11 + (y00 - y10)*x11)
        s = (1 / d) * ((x00 - x10) * (y01 - y00) - (y00 - y10) * (x01 - x00))
        t = (1 / d) * -(-(x00 - x10) * (y11 - y10) + (y00 - y10) * (x11 - x10))

        # print(s, t)
        if (s > 0 and s < 1) and (t > 0 and t < 1):
            # Lines intersect
            # print("s and t in range!")
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
        numSeen = 0

        # Get which of the objects are in the FOV
        for i, target in enumerate(self.targets):
            if self.can_see(target):
                numSeen += 1
                # Assign a non-zero confidence for those in view
                confidence = self.robot.get_confidence(target)

            else:
                # Assign confidence of 0 for those out of view
                confidence = 0
            self.current_confidences[i] = confidence

        # current_confidences = 1xN array of confidences at current timestep
        # avg_confidences = 1xN array of confidences averaged over all past timesteps
        # confidences = MxN array of confidences for all past timesteps

        # Add the current observation to the past average
        self.avg_confidences = (
            np.add(self.avg_confidences.dot(self.count), self.current_confidences)
        ) / (self.count + 1)

        # Add the current timestep confidences to the comprehensive all timesteps list
        self.confidences = np.append(self.confidences, self.current_confidences, axis=1)

    def set_scoreboard(self, scoreboard_items):
        self.scoreboard_items = scoreboard_items
        if self.render_mode == "human":
            self._render_frame()

    def handle_boundary_collisions(self):
        """
        Stops player from going off screen or going into walls
        """
        # Handle off-screen
        if self.robot.x > self.window_size - (self.robot.size / 2):
            self.robot.x = self.window_size - (self.robot.size / 2)
        elif self.robot.x < 0 + (self.robot.size / 2):
            self.robot.x = 0 + (self.robot.size / 2)
        elif self.robot.y > self.window_size - (self.robot.size / 2):
            self.robot.y = self.window_size - (self.robot.size / 2)
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
        # Check action format
        if action not in [0, 1, 2, 3, 4]:
            raise ValueError("`action` should be None, 0, 1, 2, 3, or 4.")

        # Handle agent controls and movement (note action 0 does nothing)
        if action == 1:
            self.robot.turn_right()
        elif action == 2:
            self.robot.move_forward()

        if action == 3:
            self.robot.turn_left()
        elif action == 4:
            self.robot.move_backward()

        # Stop the robot going off the screen or inside walls
        self.handle_boundary_collisions()

    def get_action_interactive(self):
        """
        Get player commands from keyboard and return the relevant action vector.
        """

        # Handle player controls and movement
        while True:
            event = pygame.event.wait()
            if (not self.paused) and (not self.gameover):
                keys = pygame.key.get_pressed()
                if keys[pygame.K_RIGHT]:
                    return 1
                if keys[pygame.K_UP]:
                    return 2
                if keys[pygame.K_LEFT]:
                    return 3
                if keys[pygame.K_DOWN]:
                    return 4
                elif keys[pygame.K_SPACE]:
                    return 0

    def get_state(self):
        """
        Gets the current state of the game

        Args:
            None
        Returns:
            (dict) a dictionary containing all the state variables
        """
        state_dict = {}
        state_dict["count"] = self.count
        state_dict["budget"] = self.budget
        state_dict["current_confidences"] = self.current_confidences
        state_dict["confidences"] = self.confidences
        state_dict["avg_confidences"] = self.avg_confidences

        return state_dict

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

            self._render_frame()

    def reset(self):
        """
        Resets the game to its initial state

        Args:
            None
        Returns:
            None
        """
        self.gameover = False
        self.budget = self.starting_budget
        
        self.walls.clear()
        self.spawn_walls(2)
        self.spawn_robot(self.robot.fov)
        self.targets.clear()
        self.spawn_targets(self.num_targets)

        self.current_confidences = np.zeros((self.num_targets, 1), dtype=np.float32)
        self.confidences = np.zeros((self.num_targets, 1), dtype=np.float32)
        self.avg_confidences = np.zeros((self.num_targets, 1), dtype=np.float32)
        self.count = 0

        if self.render_mode == "human":
            self._render_frame()

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        pygame.init()
        if self.window is None and self.render_mode == "human":
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("ReLOaD Simulator")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # ------------------ Create the base canvas and load assets ------------------ #
        canvas = pygame.Surface((self.window_size, self.window_size))

        # if os.path.dirname(__file__) != "":
        #     sprites_dir = os.path.dirname(__file__) + "/sprites/"
        # else:
        #     sprites_dir = "sprites/"
        sprites_dir = ""

        bg = pygame.transform.scale(
            pygame.image.load(sprites_dir + "roombg.jpg"),
            (self.window_size, self.window_size),
        )
        player_robot = pygame.transform.scale(
            pygame.image.load(sprites_dir + "robot.png"),
            (self.player_size, self.player_size),
        )
        target50 = pygame.transform.scale(
            pygame.image.load(sprites_dir + "apple.png"), (50, 50)
        )
        target100 = pygame.transform.scale(
            pygame.image.load(sprites_dir + "apple.png"), (100, 100)
        )
        target150 = pygame.transform.scale(
            pygame.image.load(sprites_dir + "apple.png"), (150, 150)
        )
        fov_color = (0, 0, 255, 70)
        fov_line_length = 500
        fov_line_thickness = 10

        # --------------------------- Draw all the entities -------------------------- #
        # Draw the background image
        canvas.blit(bg, (0, 0))

        # Draw the robot
        # Make a surface with a line on it
        fov_surf = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
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
        rotated_player_rect.center = (self.robot.x, self.robot.y)
        # Update fov indicator lines position
        rotated_fov_surf = pygame.transform.rotate(fov_surf, self.robot.angle - 90)
        rotated_fov_rect = rotated_fov_surf.get_rect()
        rotated_fov_rect.center = (self.robot.x, self.robot.y)

        # Redraw the player surfs
        canvas.blit(rotated_player_surf, rotated_player_rect)
        # Redraw the fov indicator surfs
        canvas.blit(rotated_fov_surf, rotated_fov_rect)
        # pygame.draw.circle(canvas, (0, 255, 0), (self.robot.x, self.robot.y), 5) # robot centre

        # Draw the targets
        for target in self.targets:
            # Set the sprite and hitbox size
            if target.rank == 1:
                image = target50
            elif target.rank == 2:
                image = target100
            else:
                image = target150

            # Setup Render Surfs
            rotated_target_surf = pygame.transform.rotate(image, target.angle - 90)
            rotated_target_rect = rotated_target_surf.get_rect()
            rotated_target_rect.center = (target.x, target.y)
            canvas.blit(rotated_target_surf, rotated_target_rect)
            # pygame.draw.circle(canvas, (0, 255, 0), (target.x, target.y), 5) # target centre

        # Draw the walls
        wall: Wall
        for wall in self.walls:
            # Setup Render Surfs
            pygame.draw.rect(
                canvas,
                (0, 0, 0),
                pygame.Rect(wall.xmin, wall.ymin, wall.width, wall.height),
            )
            # Draw corner points
            # pygame.draw.circle(canvas, (255, 0, 255), (wall.xmin, wall.ymax), 5)
            # pygame.draw.circle(canvas, (255, 0, 255), (wall.xmax, wall.ymin), 5)
            # pygame.draw.circle(canvas, (0, 0, 255), (wall.xmin, wall.ymin), 5)
            # pygame.draw.circle(canvas, (0, 0, 255), (wall.xmax, wall.ymax), 5)

        # Draw the robot's trail
        for point in self.robot.trail:
            pygame.draw.circle(canvas, (255, 0, 0), point, 2)

        # Draw the onscreen menu text
        # font = pygame.font.SysFont("arial", 30)
        font = pygame.font.Font(None, 25)

        budget_text = font.render("Budget: " + str(self.budget), 1, (0, 255, 0))
        canvas.blit(budget_text, (25, 25))

        play_again_text = font.render("Press Tab to Play Again", 1, (0, 255, 0))
        if self.gameover:
            canvas.blit(
                play_again_text,
                (
                    self.window_size // 2 - play_again_text.get_width() // 2,
                    self.window_size // 2 - play_again_text.get_height() // 2,
                ),
            )

        pause_text = font.render("Press P to Unpause", 1, (0, 255, 0))
        if self.paused:
            canvas.blit(
                pause_text,
                (
                    self.window_size // 2 - pause_text.get_width() // 2,
                    self.window_size // 2 - pause_text.get_height() // 2,
                ),
            )

        metric_count = 0
        for metric_name in self.scoreboard_items.keys():
            metric_text = font.render(
                f"{metric_name}: " + str(self.scoreboard_items[metric_name]),
                1,
                (0, 255, 0),
            )
            canvas.blit(
                metric_text,
                (
                    self.window_size - metric_text.get_width() - 25,
                    (metric_count * 35) + metric_text.get_height(),
                ),
            )
            metric_count += 1

        # --------------- Send the rendered view to the relevant viewer -------------- #

        if self.render_mode == "human":
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


# ---------------------------------------------------------------------------- #
#                                     MAIN                                     #
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Hyperparameters
    STARTING_BUDGET = 2000
    NUM_TARGETS = 8
    PLAYER_FOV = 60

    game = SimpleSim(
        starting_budget=STARTING_BUDGET,
        num_targets=NUM_TARGETS,
        player_fov=PLAYER_FOV,
        render_mode="human",
        render_fps=30,
    )
    game._render_frame()

    state = {}
    # last_reward = []

    while game.run:
        # Get the games state
        state = game.get_state()

        # # Get the reward
        # last_reward = game.get_reward()

        # Step the game engine
        game.step(game.get_action_interactive())
        game._render_frame()

    pygame.quit()
