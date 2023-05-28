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

bg = pygame.transform.scale(pygame.image.load('sprites/roombg.png'), (800, 800))
playerRocket = pygame.transform.scale(pygame.image.load('sprites/robot.png'), (100, 100))
asteroid50 = pygame.transform.scale(pygame.image.load('sprites/apple.png'), (50, 50))
asteroid100 = pygame.transform.scale(pygame.image.load('sprites/apple.png'), (100, 100))
asteroid150 = pygame.transform.scale(pygame.image.load('sprites/apple.png'), (150, 150))

pygame.display.set_caption('ReLOaD Simulator')
win = pygame.display.set_mode((sw, sh))
clock = pygame.time.Clock()

STARTING_BUDGET = 2000
NUM_TARGETS = 1
PLAYER_FOV = 90


# ---------------------------------------------------------------------------- #
#                                    CLASSES                                   #
# ---------------------------------------------------------------------------- #

class Target(object):
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
        if self.x < sw//2:
            self.xdir = 1
        else:
            self.xdir = -1
        if self.y < sh//2:
            self.ydir = 1
        else:
            self.ydir = -1

    def draw(self, win):
        win.blit(self.image, (self.x - (self.w/2), self.y - (self.h/2)))


class Robot(object):
    def __init__(self, fov):
        self.img = playerRocket
        self.w = self.img.get_width()
        self.h = self.img.get_height()
        self.x = sw//2
        self.y = sh//2
        self.angle = 0
        self.rotatedSurf = pygame.transform.rotate(self.img, self.angle)
        self.rotatedRect = self.rotatedSurf.get_rect()
        self.rotatedRect.center = (self.x, self.y)
        self.cosine = math.cos(math.radians(self.angle + 90))
        self.sine = math.sin(math.radians(self.angle + 90))
        self.head = (self.x + self.cosine * self.w//2, self.y - self.sine * self.h//2)
        self.fov = fov

        if self.fov > 180:
            raise ValueError("FOV must be <= 180 (required by can_see())")

    def draw(self, win):
        win.blit(self.rotatedSurf, self.rotatedRect)

    def turnLeft(self):
        self.angle += 5
        self.rotatedSurf = pygame.transform.rotate(self.img, self.angle)
        self.rotatedRect = self.rotatedSurf.get_rect()
        self.rotatedRect.center = (self.x, self.y)
        self.cosine = math.cos(math.radians(self.angle + 90))
        self.sine = math.sin(math.radians(self.angle + 90))
        self.head = (self.x + self.cosine * self.w//2, self.y - self.sine * self.h//2)

    def turnRight(self):
        self.angle -= 5
        self.rotatedSurf = pygame.transform.rotate(self.img, self.angle)
        self.rotatedRect = self.rotatedSurf.get_rect()
        self.rotatedRect.center = (self.x, self.y)
        self.cosine = math.cos(math.radians(self.angle + 90))
        self.sine = math.sin(math.radians(self.angle + 90))
        self.head = (self.x + self.cosine * self.w//2, self.y - self.sine * self.h//2)

    def moveForward(self):
        self.x += self.cosine * 6
        self.y -= self.sine * 6
        self.updateLocation()
        self.rotatedSurf = pygame.transform.rotate(self.img, self.angle)
        self.rotatedRect = self.rotatedSurf.get_rect()
        self.rotatedRect.center = (self.x, self.y)
        self.cosine = math.cos(math.radians(self.angle + 90))
        self.sine = math.sin(math.radians(self.angle + 90))
        self.head = (self.x + self.cosine * self.w // 2, self.y - self.sine * self.h // 2)

    def updateLocation(self):
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

        left_angle = (self.angle + 90 + (self.fov/2)) % 360
        right_angle = (self.angle + 90 - (self.fov/2)) % 360
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
        # TODO: Fix this orientation factor which fr some reason seems ot be max at 45deg from robot to target
        angle_between = np.degrees(np.arccos(target_to_robot_vect.dot(target_orientation_vect) / (np.linalg.norm(target_to_robot_vect) * np.linalg.norm(target_orientation_vect))))
        orientation_factor = 1 - (angle_between / 180)

        # Decide the weightings between how much the distance and orientation 
        # factors affect the confidence
        distance_weighting = 0
        orientation_weighting = 1
        weighting_sum = distance_weighting + orientation_weighting
        distance_weighting = distance_weighting / weighting_sum
        orientation_weighting = orientation_weighting / weighting_sum

        # Compute the final confidence
        confidence = (orientation_factor * orientation_weighting)

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
        self.highScore = 0

        self.robot = Robot(player_fov)
        self.targets = []
        # 1D array of confidences on each object at current timestep
        self.currentConfidences = []
        # 2D array of confidences on each object at each timestep
        self.confidences = []
        self.count = 0
        self.run = True

        self.spawn_targets(self.num_targets)

    def spawn_targets(self, numToSpawn):
        """
        Spawn objects to identify
        """
        for _ in range(0, numToSpawn):
            ran = random.choice([1,1,1,2,2,3])
            self.targets.append(Target(ran))

    def redrawGameWindow(self):
        """
        Main render function
        """
        win.blit(bg, (0,0))
        font = pygame.font.SysFont('arial',30)
        budgetText = font.render('Budget: ' + str(self.budget), 1, (0, 255, 0))
        playAgainText = font.render('Press Tab to Play Again', 1, (0, 255, 0))
        pauseText = font.render('Press P to Unpause', 1, (0, 255, 0))
        # scoreText = font.render('Score: ' + str(self.score), 1, (255,255,255))
        scoreText = font.render('Current Confidence: ' + str(round(np.sum(self.currentConfidences), 2)), 1, (0, 255, 0))
        highScoreText = font.render('High Score: ' + str(self.highScore), 1, (0, 255, 0))    
        # highScoreText = font.render('Time Weighted Confidence: ' + str(int(np.sum(self.confidences))), 1, (255, 255, 255))   

        self.robot.draw(win)
        for target in self.targets:
            target.draw(win)
        
        if self.paused:
            win.blit(pauseText, (sw//2-pauseText.get_width()//2, sh//2 - pauseText.get_height()//2))
        if self.gameover:
            win.blit(playAgainText, (sw//2-playAgainText.get_width()//2, sh//2 - playAgainText.get_height()//2))
        win.blit(scoreText, (sw - scoreText.get_width() - 25, 25))
        win.blit(budgetText, (25, 25))
        win.blit(highScoreText, (sw - highScoreText.get_width() -25, 35 + scoreText.get_height()))
        pygame.display.update()

    def detect_targets(self):
        """
        Looks for objects in the view of the player and detects them
        """
        numSeen = 0
        self.currentConfidences = []

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
            print(f"Target {i}: {confidence}")
            
            self.currentConfidences.append(confidence)
        
        print(f"numSeen: {numSeen}\n")
        
        self.confidences.append(self.currentConfidences)

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
                self.robot.updateLocation()

                # Get the detection confidences on the environment
                # print("Detecting...")
                self.detect_targets()

                # Check gameover
                if self.budget <= 0:
                    self.gameover = True

                # Handle player controls and movement
                keys = pygame.key.get_pressed()
                if keys[pygame.K_LEFT]:
                    self.robot.turnLeft()
                if keys[pygame.K_RIGHT]:
                    self.robot.turnRight()
                if keys[pygame.K_UP]:
                    self.robot.moveForward()

            # Handle menu keyboard events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.run = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_TAB:
                        if self.gameover:
                            self.gameover = False
                            self.budget = self.starting_budget
                            self.targets.clear()
                            self.spawn_targets(self.num_targets)
                            if self.score > self.highScore:
                                self.highScore = self.score
                            self.score = 0

                    if event.key == pygame.K_p:
                        self.paused = not self.paused

            # Re-render the scene
            self.redrawGameWindow()


# ---------------------------------------------------------------------------- #
#                                     MAIN                                     #
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    game = Game(STARTING_BUDGET, NUM_TARGETS, PLAYER_FOV)
    game.run_game()
    pygame.quit()