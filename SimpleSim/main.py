# ReLOaD Simple Simulator
# 
# main.py
# 
# Alex Nichoson
# 27/05/2023

import pygame
import math
import random
import numpy as np


# ---------------------------------------------------------------------------- #
#                                  GLOBAL VARS                                 #
# ---------------------------------------------------------------------------- #
pygame.init()

sw = 800
sh = 800

bg = pygame.transform.scale(pygame.image.load('sprites/roombg.jpg'), (800, 800))
player_rocket = pygame.transform.scale(pygame.image.load('sprites/robot.png'), (100, 100))
asteroid50 = pygame.transform.scale(pygame.image.load('sprites/apple.png'), (50, 50))
asteroid100 = pygame.transform.scale(pygame.image.load('sprites/apple.png'), (100, 100))
asteroid150 = pygame.transform.scale(pygame.image.load('sprites/apple.png'), (150, 150))

pygame.display.set_caption('ReLOaD Simulator')
win = pygame.display.set_mode((sw, sh))
clock = pygame.time.Clock()

STARTING_BUDGET = 2000
NUM_TARGETS = 8
PLAYER_FOV = 90


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
            self.image = asteroid50
        elif self.rank == 2:
            self.image = asteroid100
        else:
            self.image = asteroid150
        self.w = 50 * rank
        self.h = 50 * rank

        # Set the initial position
        self.ranPoint = (random.randrange(0+self.w, sw-self.w), 
                         random.randrange(0+self.h, sh-self.h))
        self.x, self.y = self.ranPoint

        # Set the orientation
        self.angle = 90 # unit circle format
        self.rotated_surf = pygame.transform.rotate(self.image, self.angle-90)
        self.rotated_rect = self.rotated_surf.get_rect()
        self.rotated_rect.center = (self.x, self.y)

        self.xdir = int(math.cos(math.radians(self.angle)))
        self.ydir = int(math.sin(math.radians(self.angle)))
        # print(f"angle:{self.angle}, xdir:{self.xdir}, ydir:{self.ydir}")

    def draw(self, win):
        win.blit(self.rotated_surf, self.rotated_rect)

class Robot(object):
    """
    The mobile robot that the RL algorith will control
    """
    def __init__(self, fov):
        self.image = player_rocket
        self.w = self.image.get_width()
        self.h = self.image.get_height()
        # Set position
        self.x = sw//2
        self.y = sh//2
        self.trail = [] # a trail of all past x,y coords
        # Set orientation
        self.angle = 90 # unit circle angles
        # Draw sprite at starting position
        self.rotated_surf = pygame.transform.rotate(self.image, self.angle-90)
        self.rotated_rect = self.rotated_surf.get_rect()
        self.rotated_rect.center = (self.x, self.y)

        self.fov = fov

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
        self.rotated_surf = pygame.transform.rotate(self.image, self.angle-90)
        self.rotated_rect = self.rotated_surf.get_rect()
        self.rotated_rect.center = (self.x, self.y)

    def draw(self, win):
        win.blit(self.rotated_surf, self.rotated_rect)

    def turn_left(self):
        self.angle += 5
        self.rotated_surf = pygame.transform.rotate(self.image, self.angle-90)
        self.rotated_rect = self.rotated_surf.get_rect()
        self.rotated_rect.center = (self.x, self.y)

    def turn_right(self):
        self.angle -= 5
        self.rotated_surf = pygame.transform.rotate(self.image, self.angle-90)
        self.rotated_rect = self.rotated_surf.get_rect()
        self.rotated_rect.center = (self.x, self.y)


    def move_forward(self):
        self.x += math.cos(math.radians(self.angle)) * 6
        self.y -= math.sin(math.radians(self.angle)) * 6
        self.trail.append((self.x, self.y))
        self.update_location()
        self.rotated_surf = pygame.transform.rotate(self.image, self.angle-90)
        self.rotated_rect = self.rotated_surf.get_rect()
        self.rotated_rect.center = (self.x, self.y)

    def move_backward(self):
        self.x -= math.cos(math.radians(self.angle)) * 6
        self.y += math.sin(math.radians(self.angle)) * 6
        self.trail.append((self.x, self.y))
        self.update_location()
        self.rotated_surf = pygame.transform.rotate(self.image, self.angle-90)
        self.rotated_rect = self.rotated_surf.get_rect()
        self.rotated_rect.center = (self.x, self.y)

    def update_location(self):
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

        return confidence;

class Game(object):
    """"
    A class to handle all of the data structures and logic of the game
    """
    def __init__(self, starting_budget, num_targets, player_fov):
        self.starting_budget = starting_budget
        self.num_targets = num_targets

        self.gameover = False
        self.paused = False
        self.budget = self.starting_budget
        self.score = 0
        self.high_score = 0        
        self.count = 0
        self.run = True

        self.robot = Robot(player_fov)
        self.targets = []
        # 1D array of confidences on each object at current timestep
        self.current_confidences = []
        # 2D array of confidences on each object at each timestep
        self.confidences = []

        self.spawn_targets(self.num_targets)

    def reset(self):
        self.gameover = False
        self.budget = self.starting_budget
        self.targets.clear()
        self.spawn_targets(self.num_targets)
        self.robot.reset()

        self.current_confidences = []
        self.confidences = []

        if self.score > self.high_score:
            self.high_score = self.score
        self.score = 0

    def get_state(self):
        state_dict = {}

        state_dict["current_confidences"] = self.current_confidences
        state_dict["confidences"] = self.confidences
        self.gameover = False
        self.budget = self.starting_budget
        self.targets.clear()
        self.spawn_targets(self.num_targets)
        self.robot.reset()

        self.current_confidences = []
        self.confidences = []

        if self.score > self.high_score:
            self.high_score = self.score
        self.score = 0

        return state_dict

    def spawn_targets(self, num_to_spawn):
        """
        Spawn objects to identify
        """
        for _ in range(0, num_to_spawn):
            ran = random.choice([1,1,1,2,2,3])
            self.targets.append(Target(ran))

    def redraw_game_window(self):
        """
        Main render function
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
        score_text = font.render('Current Confidence: ' + str(round(np.sum(self.current_confidences), 2)), 1, (0, 255, 0))
        high_score_text = font.render('High Score: ' + str(self.high_score), 1, (0, 255, 0))    
        # high_score_text = font.render('Time Weighted Confidence: ' + str(int(np.sum(self.confidences))), 1, (255, 255, 255))   

        if self.paused:
            win.blit(pause_text, (sw//2-pause_text.get_width()//2, sh//2 - pause_text.get_height()//2))
        if self.gameover:
            win.blit(play_again_text, (sw//2-play_again_text.get_width()//2, sh//2 - play_again_text.get_height()//2))
        win.blit(score_text, (sw - score_text.get_width() - 25, 25))
        win.blit(budget_text, (25, 25))
        win.blit(high_score_text, (sw - high_score_text.get_width() -25, 35 + score_text.get_height()))
        pygame.display.update()

    def detect_targets(self):
        """
        Looks for objects in the view of the player and detects them
        """
        numSeen = 0
        self.current_confidences = []

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
            # print(f"Target {i}: {confidence}")
            
            self.current_confidences.append(confidence)
        
        # print(f"numSeen: {numSeen}\n")
        
        self.confidences.append(self.current_confidences)

    def get_reward(self):
        """
        Get the rewaard for this step, based on the time-weighted confidences
        """
        reward = np.mean(self.confidences)
        return reward

    def run_game(self):
        """
        Runs the game logic (controller)
        """
        while self.run:
            if (not self.paused) and (not self.gameover):
                clock.tick(60)
                self.count += 1

                # Decrement the budget over time
                self.budget -= 1

                # Update the player potisions
                self.robot.update_location()

                # Get the detection confidences on the environment
                # print("Detecting...")
                self.detect_targets()

                # Check gameover
                if self.budget <= 0:
                    self.gameover = True

                # Handle player controls and movement
                keys = pygame.key.get_pressed()
                if keys[pygame.K_LEFT]:
                    self.robot.turn_left()
                if keys[pygame.K_RIGHT]:
                    self.robot.turn_right()
                if keys[pygame.K_UP]:
                    self.robot.move_forward()
                if keys[pygame.K_DOWN]:
                    self.robot.move_backward()

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

            # Re-render the scene
            self.redraw_game_window()


# ---------------------------------------------------------------------------- #
#                                     MAIN                                     #
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    game = Game(STARTING_BUDGET, NUM_TARGETS, PLAYER_FOV)
    game.run_game()
    pygame.quit()