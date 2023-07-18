    from __future__ import absolute_import
    from __future__ import division
    from __future__ import print_function

    import numpy as np
    import random
    from IPython.display import clear_output
    import time



    import abc
    import tensorflow as tf
    import numpy as np

    from tf_agents.environments import py_environment
    from tf_agents.environments import tf_environment
    from tf_agents.environments import tf_py_environment
    from tf_agents.environments import utils
    from tf_agents.specs import array_spec
    from tf_agents.environments import wrappers
    from tf_agents.environments import suite_gym
    from tf_agents.trajectories import time_step as ts


    class cGame(py_environment.PyEnvironment):
        def __init__(self):
            self.xdim = 21
            self.ydim = 21
            self.mmap = np.array([[0] * self.xdim] * self.ydim)
            self._turnNumber = 0
            self.playerPos = {"x": 1, "y": 1}
            self.totalScore = 0
            self.reward = 0.0
            self.input = 0
            self.addRewardEveryNTurns = 4
            self.addBombEveryNTurns = 3
            self._episode_ended = False

            ## player = 13
            ## bomb   = 14

            self._action_spec = array_spec.BoundedArraySpec(shape=(),
                                                            dtype=np.int32,
                                                            minimum=0, maximum=3,
                                                            name='action')
            self._observation_spec = array_spec.BoundedArraySpec(shape=(441,),
                                                                 minimum=np.array(
                                                                     [-1] * 441),
                                                                 maximum=np.array(
                                                                     [20] * 441),
                                                                 dtype=np.int32,
                                                                 name='observation')  # (self.xdim, self.ydim)  , self.mmap.shape,  minimum = -1, maximum = 10

        def action_spec(self):
            return self._action_spec

        def observation_spec(self):
            return self._observation_spec

        def addMapReward(self):
            dx = random.randint(1, self.xdim - 2)
            dy = random.randint(1, self.ydim - 2)
            if dx != self.playerPos["x"] and dy != self.playerPos["y"]:
                self.mmap[dy][dx] = random.randint(1, 9)
            return True

        def addBombToMap(self):
            dx = random.randint(1, self.xdim - 2)
            dy = random.randint(1, self.ydim - 2)
            if dx != self.playerPos["x"] and dy != self.playerPos["y"]:
                self.mmap[dy][dx] = 14
            return True

        def _reset(self):
            self.mmap = np.array([[0] * self.xdim] * self.ydim)
            for y in range(self.ydim):
                self.mmap[y][0] = -1
                self.mmap[y][self.ydim - 1] = -1
            for x in range(self.xdim):
                self.mmap[0][x] = -1
                self.mmap[self.ydim - 1][x] = -1

            self.playerPos["x"] = random.randint(1, self.xdim - 2)
            self.playerPos["y"] = random.randint(1, self.ydim - 2)
            self.mmap[self.playerPos["y"]][self.playerPos["x"]] = 13

            for z in range(10):
                ## place 10 targets
                self.addMapReward()
            for z in range(5):
                ## place 5 bombs
                ## bomb   = 14
                self.addBombToMap()
            self._turnNumber = 0
            self._episode_ended = False
            # return ts.restart (self.mmap)
            dap = ts.restart(np.array(self.mmap, dtype=np.int32).flatten())
            return (dap)

        def render(self, mapToRender):
            mapToRender.reshape(21, 21)
            for y in range(self.ydim):
                o = ""
                for x in range(self.xdim):
                    if mapToRender[y][x] == -1:
                        o = o + "#"
                    elif mapToRender[y][x] > 0 and mapToRender[y][x] < 10:
                        o = o + str(mapToRender[y][x])
                    elif mapToRender[y][x] == 13:
                        o = o + "@"
                    elif mapToRender[y][x] == 14:
                        o = o + "*"
                    else:
                        o = o + " "
                print(o)
            print('TOTAL SCORE:', self.totalScore, 'LAST TURN SCORE:', self.reward)
            return True

        def getInput(self):
            self.input = 0
            i = input()
            if i == 'w' or i == '0':
                print('going N')
                self.input = 1
            if i == 's' or i == '1':
                print('going S')
                self.input = 2
            if i == 'a' or i == '2':
                print('going W')
                self.input = 3
            if i == 'd' or i == '3':
                print('going E')
                self.input = 4
            if i == 'x':
                self.input = 5
            return self.input

        def processMove(self):

            self.mmap[self.playerPos["y"]][self.playerPos["x"]] = 0
            self.reward = 0
            if self.input == 0:
                self.playerPos["y"] -= 1
            if self.input == 1:
                self.playerPos["y"] += 1
            if self.input == 2:
                self.playerPos["x"] -= 1
            if self.input == 3:
                self.playerPos["x"] += 1

            cloc = self.mmap[self.playerPos["y"]][self.playerPos["x"]]

            if cloc == -1 or cloc == 14:
                self.totalScore = 0
                self.reward = -99

            if cloc > 0 and cloc < 10:
                self.totalScore += cloc
                self.reward = cloc
                self.mmap[self.playerPos["y"]][self.playerPos["x"]] = 0

            self.mmap[self.playerPos["y"]][self.playerPos["x"]] = 13

            self.render(self.mmap)

        def runTurn(self):
            clear_output(wait=True)
            if self._turnNumber % self.addRewardEveryNTurns == 0:
                self.addMapReward()
            if self._turnNumber % self.addBombEveryNTurns == 0:
                self.addBombToMap()

            self.getInput()
            self.processMove()
            self._turnNumber += 1
            if self.reward == -99:
                self._turnNumber += 1
                self._reset()
                self.totalScore = 0
                self.render(self.mmap)
            return (self.reward)

        def _step(self, action):

            if self._episode_ended == True:
                return self._reset()

            clear_output(wait=True)
            if self._turnNumber % self.addRewardEveryNTurns == 0:
                self.addMapReward()
            if self._turnNumber % self.addBombEveryNTurns == 0:
                self.addBombToMap()

            ## make sure action does produce exceed range
            # if action > 5 or action <1:
            #    action =0
            self.input = action  ## value 1 to 4
            self.processMove()
            self._turnNumber += 1

            if self.reward == -99:
                self._turnNumber += 1
                self._episode_ended = True
                # self._reset()
                self.totalScore = 0
                self.render(self.mmap)
                return ts.termination(np.array(self.mmap, dtype=np.int32).flatten(),
                                      reward=self.reward)
            else:
                return ts.transition(np.array(self.mmap, dtype=np.int32).flatten(),
                                     reward=self.reward)  # , discount = 1.0

        def run(self):
            self._reset()
            self.render(self.mmap)
            while (True):
                self.runTurn()
                if self.input == 5:
                    return ("EXIT on input x ")


    env = cGame()