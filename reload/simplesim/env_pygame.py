# ReLOaD Simple Simulator
# 
# simplesim.py
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
import time
import os

# ---------------------------------------------------------------------------- #
#                                  GLOBAL VARS                                 #
# ---------------------------------------------------------------------------- #
pygame.init()

sw = 1000
sh = 1000

player_width = 100
player_height = 100

sprites_dir = os.path.dirname(__file__) + '/sprites/'

bg = pygame.transform.scale(pygame.image.load(sprites_dir + 'roombg.jpg'), (sw, sh))
player_robot = pygame.transform.scale(pygame.image.load(sprites_dir + 'robot.png'), (player_width, player_height))
target50 = pygame.transform.scale(pygame.image.load(sprites_dir + 'apple.png'), (50, 50))
target100 = pygame.transform.scale(pygame.image.load(sprites_dir + 'apple.png'), (100, 100))
target150 = pygame.transform.scale(pygame.image.load(sprites_dir + 'apple.png'), (150, 150))
fov_color = (0, 0, 255, 70)
fov_line_length = 500
fov_line_thickness = 10

pygame.display.set_caption('ReLOaD Simulator')
win = pygame.display.set_mode((sw, sh))
clock = pygame.time.Clock()

# ---------------------------------------------------------------------------- #
#                                    CLASSES                                   #
# ---------------------------------------------------------------------------- #

class Target(object):
    """
    The targets that the RL agent must attempt to classify using the mobile robot
    """
    def __init__(self, rank):
        self.rank = rank

        # Set the sprite and hitboxcsize
        if self.rank == 1:
            self.image = target50
        elif self.rank == 2:
            self.image = target100
        else:
            self.image = target150
        self.w = 50 * rank
        self.h = 50 * rank

        # Set the initial position
        self.ranPoint = (random.randrange(0+self.w, sw-self.w), 
                         random.randrange(0+self.h, sh-self.h))
        self.x, self.y = self.ranPoint

        # Set the orientation
        self.angle = 90 # unit circle format
        self.rotated_player_surf = pygame.transform.rotate(self.image, self.angle-90)
        self.rotated_player_rect = self.rotated_player_surf.get_rect()
        self.rotated_player_rect.center = (self.x, self.y)

        self.xdir = int(math.cos(math.radians(self.angle)))
        self.ydir = int(math.sin(math.radians(self.angle)))
        # print(f"angle:{self.angle}, xdir:{self.xdir}, ydir:{self.ydir}")

    def draw(self, win):
        win.blit(self.rotated_player_surf, self.rotated_player_rect)

class Robot(object):
    """
    The mobile robot that the RL algorith will control
    """
    def __init__(self, fov):
        self.image = player_robot
        self.w = self.image.get_width()
        self.h = self.image.get_height()
        # Set position
        self.x = sw//2
        self.y = sh//2
        self.trail = [] # a trail of all past x,y coords
        # Set orientation
        self.angle = 90 # unit circle angles
        
        # Draw sprite at starting position
        
        # 90deg rotated version of the sprite surf image
        self.rotated_player_surf = pygame.transform.rotate(self.image, self.angle-90)
        # The rectangle bounding box of the surf
        self.rotated_player_rect = self.rotated_player_surf.get_rect()
        # Set the centre position of the surf to the player position vars
        self.rotated_player_rect.center = (self.x, self.y)

        # Draw player fov indicator
        self.fov = fov
        # Make a surface with a line on it
        self.fov_surf = pygame.Surface((sw,sw), pygame.SRCALPHA)
        self.fov_surf.set_colorkey((0,0,0,0))
        self.rotated_fov_surf = pygame.transform.rotate(self.fov_surf, self.angle)
        pygame.draw.line(self.fov_surf, fov_color, (self.rotated_fov_surf.get_rect().w/2, self.rotated_fov_surf.get_rect().h/2), ((self.rotated_fov_surf.get_rect().w/2)-(fov_line_length*math.sin(math.radians(self.fov/2))), (self.rotated_fov_surf.get_rect().h/2)-(fov_line_length*math.cos(math.radians(self.fov/2)))), fov_line_thickness)
        pygame.draw.line(self.fov_surf, fov_color, (self.rotated_fov_surf.get_rect().w/2, self.rotated_fov_surf.get_rect().h/2), ((self.rotated_fov_surf.get_rect().w/2)+(fov_line_length*math.sin(math.radians(self.fov/2))), (self.rotated_fov_surf.get_rect().h/2)-(fov_line_length*math.cos(math.radians(self.fov/2)))), fov_line_thickness)
        # Set the position and rotation of the surface
        self.rotated_fov_rect = self.rotated_fov_surf.get_rect()
        self.rotated_fov_rect.center = (self.x, self.y)
        win.blit(self.rotated_fov_surf, self.rotated_fov_rect)

        if self.fov > 180:
            raise ValueError("FOV must be <= 180 (required by can_see())")

    def reset(self):
        # Reset position
        self.x = sw//2
        self.y = sh//2
        self.trail.clear() # a trail of all past x,y coords
        # Reset orientation
        self.angle = 90 # unit circle angles
        # Draw sprite at starting position
        self.rotated_player_surf = pygame.transform.rotate(self.image, self.angle-90)
        self.rotated_player_rect = self.rotated_player_surf.get_rect()
        self.rotated_player_rect.center = (self.x, self.y)

    def draw(self, win):
        # Redraw the player surfs
        win.blit(self.rotated_player_surf, self.rotated_player_rect)
        # Redraw the fov indicator surfs
        win.blit(self.rotated_fov_surf, self.rotated_fov_rect)

    def turn_left(self):
        self.angle += 5
        self.handle_boundary_collisions()
        self.update_surfs()

    def turn_right(self):
        self.angle -= 5
        self.handle_boundary_collisions()
        self.update_surfs()

    def move_forward(self):
        self.x += math.cos(math.radians(self.angle)) * 6
        self.y -= math.sin(math.radians(self.angle)) * 6
        self.trail.append((self.x, self.y))
        self.handle_boundary_collisions()
        self.update_surfs()

    def move_backward(self):
        self.x -= math.cos(math.radians(self.angle)) * 6
        self.y += math.sin(math.radians(self.angle)) * 6
        self.trail.append((self.x, self.y))
        self.handle_boundary_collisions()
        self.update_surfs()

    def handle_boundary_collisions(self):
        """
        Stops player from going off screen
        """
        if self.x > sw - (self.w / 2):
            self.x = sw - (self.w / 2)
        elif self.x < 0 + (self.w / 2):
            self.x = 0 + (self.w / 2)
        elif self.y > sh - (self.h / 2):
            self.y = sh - (self.h / 2)
        elif self.y < 0 + (self.h / 2):
            self.y = 0 + (self.h / 2)

    def update_surfs(self):
        """
        Updates the surf / sprite positions and orientations for the player anf the fov indicator lines
        """
        # Update player sprite position
        self.rotated_player_surf = pygame.transform.rotate(self.image, self.angle-90)
        self.rotated_player_rect = self.rotated_player_surf.get_rect()
        self.rotated_player_rect.center = (self.x, self.y)
        # Update fov indicator lines position
        self.rotated_fov_surf = pygame.transform.rotate(self.fov_surf, self.angle-90)
        self.rotated_fov_rect = self.rotated_fov_surf.get_rect()
        self.rotated_fov_rect.center = (self.x, self.y)

    def can_see(self, target: Target):
        """
        Checks if the robot can see this target

        Note: Everything in this function is done in cartesian coordinates not 
        screen pixel coordinates (bottom left = (0,0), angles measuresd 
        anti-clockwise from right).
        """

        # Convert pixel coordinated to cartesian coordinates
        robot_x_cart = self.x
        robot_y_cart = sh - self.y
        target_x_cart = target.x
        target_y_cart = sh - target.y
        # print(f"robot_x_cart:{robot_x_cart}, robot_y_cart:{robot_y_cart} //// target_x_cart:{target_x_cart}, target_y_cart:{target_y_cart}")

        left_angle = (self.angle + (self.fov/2)) % 360
        right_angle = (self.angle - (self.fov/2)) % 360
        target_dy = target_y_cart - robot_y_cart
        target_dx = target_x_cart - robot_x_cart
        # print(f"target_dy:{target_dy}, target_dx:{target_dx}")

        target_angle = np.degrees(np.arctan2(target_dy, target_dx)) # angle of ray to target
        target_angle = target_angle+360 if (target_angle<0) else target_angle
        # print(f"target_angle:{target_angle}, left_angle:{left_angle}, right_angle:{right_angle}")

        if ((target_angle <= left_angle) and (target_angle >= right_angle)):
            #  Normal case
            return True
        elif ((left_angle < self.fov) and (right_angle > 360-self.fov)):
            #  Angle-zero-crossing case
            if ((target_angle <= left_angle) and (target_angle+360 >= right_angle)):
                # Target angle after 0 crossover and inside
                return True
            elif ((target_angle <= left_angle+360) and (target_angle >= right_angle)):
                # Target angle before 0 crossover and inside
                return True
            else:
                return False
        else:
            return False

    def get_confidence(self, target: Target):
        """
        Computes the simulated YOLO confidence of an object in view
        """
        # Convert pixel coordinated to cartesian coordinates
        robot_x_cart = self.x
        robot_y_cart = sh - self.y
        target_x_cart = target.x
        target_y_cart = sh - target.y
        target_xdir_cart = target.xdir
        target_ydir_cart = -1 * target.ydir

        # Get the distances and vectors from the target to the robot
        robot_dy = robot_y_cart - target_y_cart
        robot_dx = robot_x_cart - target_x_cart
        target_to_robot_vect = np.array([robot_dy, robot_dx])
        target_orientation_vect = np.array([target_ydir_cart, target_xdir_cart])

        # A factor representing the distance difficulty (should be between 0 and 1)
        max_dist = math.sqrt(sw**2 + sh**2)
        target_dist = math.sqrt(robot_dy**2 + robot_dx**2)
        distance_factor = 1 - (target_dist / max_dist)

        # A factor representing the orientation difficulty (should be between 0 and 1)
        angle_between = np.degrees(np.arccos(target_to_robot_vect.dot(target_orientation_vect) / (np.linalg.norm(target_to_robot_vect) * np.linalg.norm(target_orientation_vect))))
        orientation_factor = 1 - (angle_between / 180)

        # Decide the weightings between how much the distance and orientation 
        # factors affect the confidence
        distance_weighting = 4
        orientation_weighting = 1
        weighting_sum = distance_weighting + orientation_weighting
        distance_weighting = distance_weighting / weighting_sum
        orientation_weighting = orientation_weighting / weighting_sum

        # Compute the final confidence
        confidence = (distance_factor * distance_weighting) + (orientation_factor * orientation_weighting)

        return confidence

class SimpleSim(object):
    """"
    A class to handle all of the data structures and logic of the game
    """
    def __init__(self, starting_budget, num_targets, player_fov):
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
        self.starting_budget = starting_budget
        self.num_targets = num_targets

        self.gameover = False
        self.paused = False
        self.budget = self.starting_budget
        # self.score = 0
        # self.high_score = 0        
        self.count = 0
        self.run = True

        self.robot = Robot(player_fov)
        self.targets = []
        # 1D array of confidences on each object at current timestep
        self.current_confidences = np.zeros((self.num_targets, 1))
        # 2D array of confidences on each object at each timestep
        self.confidences = np.zeros((self.num_targets, 1))
        # 1D array of confidences on each object avg over all past timesteps
        self.avg_confidences = np.zeros((self.num_targets, 1)) 
        # Note: Using avg might cause agent to simply find the best spot with 
        # the most objects in view and camp there to farm for max score

        self.spawn_targets(self.num_targets)

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
        self.targets.clear()
        self.spawn_targets(self.num_targets)
        self.robot.reset()

        self.current_confidences = np.zeros((self.num_targets, 1))
        self.confidences = np.zeros((self.num_targets, 1))
        self.count = 0

        # if self.score > self.high_score:
        #     self.high_score = self.score

    def spawn_targets(self, num_to_spawn):
        """
        Spawn objects to identify

        Args:
            num_to_spawn (int): Number of targets to spawn in the game
        Returns:
            None
        """
        for _ in range(0, num_to_spawn):
            ran = random.choice([1,1,1,2,2,3])
            self.targets.append(Target(ran))

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
        for i in range(0, len(self.targets)):
            target = self.targets[i]

            if self.robot.can_see(target):
                numSeen+=1
                # Assign a non-zero confidence for those in view 
                confidence = self.robot.get_confidence(target)

            else:
                # Assign confidence of 0 for those out of view
                confidence = 0            
            self.current_confidences[i] = confidence

        self.avg_confidences = (np.add(self.avg_confidences.dot(self.count), self.current_confidences)) / (self.count + 1)
        np.append(self.confidences, self.current_confidences, axis=1)

    def redraw_game_window(self):
        """
        Main render function

        Args:
            None
        Returns:
            None
        """
        win.blit(bg, (0,0))

        # Draw the robot
        self.robot.draw(win)
        # Draw the targets
        for target in self.targets:
            target.draw(win)
        # Draw the robot's trail
        for point in self.robot.trail:
            pygame.draw.circle(win, (255, 0, 0), point, 2)

        # Draw the onscreen menu text
        font = pygame.font.SysFont('arial',30)
        budget_text = font.render('Budget: ' + str(self.budget), 1, (0, 255, 0))
        play_again_text = font.render('Press Tab to Play Again', 1, (0, 255, 0))
        pause_text = font.render('Press P to Unpause', 1, (0, 255, 0))
        score_text = font.render('Current Confidences: ' + str(format(np.sum(self.current_confidences), ".2f")), 1, (0, 255, 0))
        high_score_text = font.render('Avg Confidences: ' + str(format(np.sum(self.avg_confidences), ".2f")), 1, (0, 255, 0))    
        # high_score_text = font.render('Time Weighted Confidence: ' + str(int(np.sum(self.confidences))), 1, (255, 255, 255))   

        if self.paused:
            win.blit(pause_text, (sw//2-pause_text.get_width()//2, sh//2 - pause_text.get_height()//2))
        if self.gameover:
            win.blit(play_again_text, (sw//2-play_again_text.get_width()//2, sh//2 - play_again_text.get_height()//2))
        win.blit(score_text, (sw - score_text.get_width() - 25, 25))
        win.blit(budget_text, (25, 25))
        win.blit(high_score_text, (sw - high_score_text.get_width() -25, 35 + score_text.get_height()))
        pygame.display.update()

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
        # state_dict["avg_confidences"] = self.avg_confidences
        state_dict["budget"] = self.budget
        # state_dict["object_positions"] = self.get_object_positions() 
        # object_positions is a tuple containing posiiton and orientation
        # (x, y, angle) (angle in unit circle format)
        state_dict["current_confidences"] = self.current_confidences

        return state_dict

    def get_reward(self):
        """
        Get the reward for this step, based on the time-weighted confidences

        Args:
            None
        Returns:
            (np.ndarray) the avg_confidences vector
        """
        return np.sum(self.avg_confidences)

    def perform_action(self, action):
        """
        Given an action tuple, execute the action in the environment.
        Action is given as tuple (["F"/"B"/None], ["L"/"R"/None]).
        """

        # Handle agent controls and movement
        if action == 0:
            self.robot.turn_right()
        elif action == 1:
            self.robot.move_forward()
        
        if action == 2:
            self.robot.turn_left()
        elif action == 3:
            self.robot.move_backward()
        

    def perform_action_interactive(self):
        """
        Get player commands from keyboard and execute the action in the 
        environment.
        """

        # Handle player controls and movement
        if (not self.paused) and (not self.gameover):
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                self.robot.turn_left()
            if keys[pygame.K_RIGHT]:
                self.robot.turn_right()
            if keys[pygame.K_UP]:
                self.robot.move_forward()
            if keys[pygame.K_DOWN]:
                self.robot.move_backward()

    def step(self, action):
        """
        Runs the game logic (controller)
        """
        self.perform_action(action)

        if (not self.paused) and (not self.gameover):
            clock.tick(60)
            self.count += 1

            # Decrement the budget over time
            self.budget -= 1

            # Update the player potisions
            self.robot.handle_boundary_collisions()

            # Get the detection confidences on the environment
            # print("Detecting...")
            self.detect_targets()

            # Check gameover
            if self.budget <= 0:
                self.gameover = True

            # Perform the action
            self.perform_action(action)

        # Handle menu keyboard events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.run = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_TAB:
                    if self.gameover:
                        self.reset()

                if event.key == pygame.K_p:
                    self.paused = not self.paused

                # if event.key == pygame.K_m:
                    # pygame.display.set_mode(flags=pygame.HIDDEN)
                    # screen = pygame.display.set_mode((800, 600), flags=pygame.SHOWN)


        # Re-render the scene
        self.redraw_game_window()


# ---------------------------------------------------------------------------- #
#                                     MAIN                                     #
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Hyperparameters
    STARTING_BUDGET = 2000
    NUM_TARGETS = 8
    PLAYER_FOV = 90

    game = SimpleSim(STARTING_BUDGET, NUM_TARGETS, PLAYER_FOV)
    
    state = {}
    action = (None, None)
    last_reward = []

    while game.run:
        # Get the games state
        state = game.get_state()     

        # Get the reward
        last_reward = game.get_reward()
        
        # Get optional additional user action
        game.perform_action_interactive()
        
        # Step the game engine
        game.step(None)

    pygame.quit()

# ---------------------------------------------------------------------------- #
#                                     MAIN                                     #
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    run_game()
