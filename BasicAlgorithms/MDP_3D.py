from typing import List, Any
import numpy as np
import PolicyIteration as pi
import PI_3D as pi3

class MDP3D:

    def __init__(self, numRows, numCol, numZ, wall, goal):
        self.numRows = numRows
        self.numCol = numCol
        self.numZ = numZ
        # TODO we should have wall variable as a list to contain more that  one wall (demands many changes)
        self.wall = wall
        self.goal = goal
        self.numActions = []  # this variable tell us how many actions i have from a state

        # lets make four actions right-0, left-1, up-2, down-3, upRight-4, upLeft-5, downRight-6, downLeft-7, do-noth-8
        # 9-f, 10-b
        self.actions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                        26]

        self.states = []  # grid and states are one-by-one lists, to know which state
        self.grid = []

        self.transition = {}
        self.reward = {}
        self.gamma = 0.9  # in range (0,1)
        self.mult = numZ * numCol
        self.mult2 = numRows * numCol * numZ
        self.prob = 0.8

    def CreateGrid(self):

        cnt = 0
        for i in range(0, self.numRows):
            for j in range(0, self.numCol):
                for z in range(0, self.numZ):
                    self.grid.append(self.mult2 * i + self.mult * j + z)
                    self.states.append(cnt)
                    self.numActions.append(0)
                    cnt += 1
                    # self.reward.append(0)

    def TnR(self):

        arStates = pi3.AroundStates(self.states, self.actions, self.grid, self.numRows, self.numCol, self.numZ, self.wall, self.mult, self.mult2)

        for s in self.states:

            if self.grid[s] in self.wall:
                continue

            self.transition[s] = {}
            self.reward[s] = {}
            for a in self.actions:
                if pi3.validAction(s, a, self.grid, self.numRows, self.numCol, self.numZ, self.wall, self.mult, self.mult2):
                    if a == 0:  # right
                        self.transition[s][a] = {}
                        self.reward[s][a] = {}

                        for kappa in arStates[s]:
                            self.transition[s][a][kappa] = 0  # [s + 1] = 0
                            self.reward[s][a][kappa] = 0  # [s + 1] = 0
                        # nActions[s].append(a)

                    elif a == 1:  # left
                        self.transition[s][a] = {}
                        self.reward[s][a] = {}

                        for kappa in arStates[s]:
                            self.transition[s][a][kappa] = 0  # [s + 1] = 0
                            self.reward[s][a][kappa] = 0  # [s + 1] = 0
                        # nStates[s].append(s - 1)
                        # nActions[s].append(a)

                    elif a == 2:  # up
                        self.transition[s][a] = {}
                        self.reward[s][a] = {}

                        for kappa in arStates[s]:
                            self.transition[s][a][kappa] = 0  # [s + 1] = 0
                            self.reward[s][a][kappa] = 0  # [s + 1] = 0
                        # nStates[s].append(s - col)
                        # nActions[s].append(a)

                    elif a == 3:  # down
                        self.transition[s][a] = {}
                        self.reward[s][a] = {}

                        for kappa in arStates[s]:
                            self.transition[s][a][kappa] = 0  # [s + 1] = 0
                            self.reward[s][a][kappa] = 0  # [s + 1] = 0
                        # nStates[s].append(s + col)
                        # nActions[s].append(a)

                    elif a == 4:  # up-right
                        self.transition[s][a] = {}
                        self.reward[s][a] = {}

                        for kappa in arStates[s]:
                            self.transition[s][a][kappa] = 0  # [s + 1] = 0
                            self.reward[s][a][kappa] = 0  # [s + 1] = 0

                        # nStates[s].append(s - col + 1)
                        # nActions[s].append(a)

                    elif a == 5:  # up-left
                        self.transition[s][a] = {}
                        self.reward[s][a] = {}

                        for kappa in arStates[s]:
                            self.transition[s][a][kappa] = 0  # [s + 1] = 0
                            self.reward[s][a][kappa] = 0  # [s + 1] = 0

                        # nStates[s].append(s - col - 1)
                        # nActions[s].append(a)

                    elif a == 6:  # down-right
                        self.transition[s][a] = {}
                        self.reward[s][a] = {}

                        for kappa in arStates[s]:
                            self.transition[s][a][kappa] = 0  # [s + 1] = 0
                            self.reward[s][a][kappa] = 0  # [s + 1] = 0

                        # nStates[s].append(s + col + 1)
                        # nActions[s].append(a)

                    elif a == 7:  # down-left
                        self.transition[s][a] = {}
                        self.reward[s][a] = {}

                        for kappa in arStates[s]:
                            self.transition[s][a][kappa] = 0  # [s + 1] = 0
                            self.reward[s][a][kappa] = 0  # [s + 1] = 0

                        # nStates[s].append(s + col - 1)
                        # nActions[s].append(a)
                    elif a == 8:  # down nothing
                        self.transition[s][a] = {}
                        self.reward[s][a] = {}

                        for kappa in arStates[s]:
                            self.transition[s][a][kappa] = 0  # [s + 1] = 0
                            self.reward[s][a][kappa] = 0  # [s + 1] = 0
                    elif a == 9:  # f
                        self.transition[s][a] = {}
                        self.reward[s][a] = {}

                        for kappa in arStates[s]:
                            self.transition[s][a][kappa] = 0  # [s + 1] = 0
                            self.reward[s][a][kappa] = 0  # [s + 1] = 0
                    elif a == 10:  # b
                        self.transition[s][a] = {}
                        self.reward[s][a] = {}

                        for kappa in arStates[s]:
                            self.transition[s][a][kappa] = 0  # [s + 1] = 0
                            self.reward[s][a][kappa] = 0  # [s + 1] = 0
                    elif a == 11:  # f right
                        self.transition[s][a] = {}
                        self.reward[s][a] = {}

                        for kappa in arStates[s]:
                            self.transition[s][a][kappa] = 0  # [s + 1] = 0
                            self.reward[s][a][kappa] = 0  # [s + 1] = 0
                    elif a == 12:  # f left
                        self.transition[s][a] = {}
                        self.reward[s][a] = {}

                        for kappa in arStates[s]:
                            self.transition[s][a][kappa] = 0  # [s + 1] = 0
                            self.reward[s][a][kappa] = 0  # [s + 1] = 0
                    elif a == 13:  # f up
                        self.transition[s][a] = {}
                        self.reward[s][a] = {}

                        for kappa in arStates[s]:
                            self.transition[s][a][kappa] = 0  # [s + 1] = 0
                            self.reward[s][a][kappa] = 0  # [s + 1] = 0
                    elif a == 14:  # f down
                        self.transition[s][a] = {}
                        self.reward[s][a] = {}

                        for kappa in arStates[s]:
                            self.transition[s][a][kappa] = 0  # [s + 1] = 0
                            self.reward[s][a][kappa] = 0  # [s + 1] = 0
                    elif a == 15:  # b right
                        self.transition[s][a] = {}
                        self.reward[s][a] = {}

                        for kappa in arStates[s]:
                            self.transition[s][a][kappa] = 0  # [s + 1] = 0
                            self.reward[s][a][kappa] = 0  # [s + 1] = 0
                    elif a == 16:  # b left
                        self.transition[s][a] = {}
                        self.reward[s][a] = {}

                        for kappa in arStates[s]:
                            self.transition[s][a][kappa] = 0  # [s + 1] = 0
                            self.reward[s][a][kappa] = 0  # [s + 1] = 0
                    elif a == 17:  # b up
                        self.transition[s][a] = {}
                        self.reward[s][a] = {}

                        for kappa in arStates[s]:
                            self.transition[s][a][kappa] = 0  # [s + 1] = 0
                            self.reward[s][a][kappa] = 0  # [s + 1] = 0

                    elif a == 18:  # b down
                        self.transition[s][a] = {}
                        self.reward[s][a] = {}

                        for kappa in arStates[s]:
                            self.transition[s][a][kappa] = 0  # [s + 1] = 0
                            self.reward[s][a][kappa] = 0  # [s + 1] = 0
                    elif a == 19:  # f up-right
                        self.transition[s][a] = {}
                        self.reward[s][a] = {}

                        for kappa in arStates[s]:
                            self.transition[s][a][kappa] = 0  # [s + 1] = 0
                            self.reward[s][a][kappa] = 0  # [s + 1] = 0
                    elif a == 20:  # f up-left
                        self.transition[s][a] = {}
                        self.reward[s][a] = {}

                        for kappa in arStates[s]:
                            self.transition[s][a][kappa] = 0  # [s + 1] = 0
                            self.reward[s][a][kappa] = 0  # [s + 1] = 0
                    elif a == 21:  # f down-right
                        self.transition[s][a] = {}
                        self.reward[s][a] = {}

                        for kappa in arStates[s]:
                            self.transition[s][a][kappa] = 0  # [s + 1] = 0
                            self.reward[s][a][kappa] = 0  # [s + 1] = 0
                    elif a == 22:  # f down-left
                        self.transition[s][a] = {}
                        self.reward[s][a] = {}

                        for kappa in arStates[s]:
                            self.transition[s][a][kappa] = 0  # [s + 1] = 0
                            self.reward[s][a][kappa] = 0  # [s + 1] = 0
                    elif a == 23:  # b up-right
                        self.transition[s][a] = {}
                        self.reward[s][a] = {}

                        for kappa in arStates[s]:
                            self.transition[s][a][kappa] = 0  # [s + 1] = 0
                            self.reward[s][a][kappa] = 0  # [s + 1] = 0
                    elif a == 24:  # b up-left
                        self.transition[s][a] = {}
                        self.reward[s][a] = {}

                        for kappa in arStates[s]:
                            self.transition[s][a][kappa] = 0  # [s + 1] = 0
                            self.reward[s][a][kappa] = 0  # [s + 1] = 0
                    elif a == 25:  # b down-right
                        self.transition[s][a] = {}
                        self.reward[s][a] = {}

                        for kappa in arStates[s]:
                            self.transition[s][a][kappa] = 0  # [s + 1] = 0
                            self.reward[s][a][kappa] = 0  # [s + 1] = 0

                    else:  # b down-left
                        self.transition[s][a] = {}
                        self.reward[s][a] = {}

                        for kappa in arStates[s]:
                            self.transition[s][a][kappa] = 0  # [s + 1] = 0
                            self.reward[s][a][kappa] = 0  # [s + 1] = 0

                        # nStates[s].append(s)
                        # nActions[s].append(a)



    def InitRnT(self):
        # print(self.mult)
        # for every state
        for s in self.states:

            # for every possible action
            for a in self.actions:
                # i have to check if the action is valid from this state
                if pi3.validAction(s, a, self.grid, self.numRows, self.numCol, self.numZ, self.wall, self.mult, self.mult2):
                    self.numActions[s] += 1  # then count how many actions the state has

            # in this loop i assign equal probability in the transition matrix and the reward in final state only
            for a in self.actions:
                if pi3.validAction(s, a, self.grid, self.numRows, self.numCol, self.numZ, self.wall, self.mult, self.mult2):

                    if a == 0:  # right

                        for kappa in self.transition[s][a]:

                            if kappa == s + self.numZ:
                                self.transition[s][a][kappa] = self.prob

                            else:
                                self.transition[s][a][kappa] = (1 - self.prob) / (self.numActions[s] - 1)

                            if self.grid[kappa] == self.goal:
                                self.reward[s][a][kappa] = self.mult2

                    elif a == 1:  # left
                        for kappa in self.transition[s][a]:

                            if kappa == s - self.numZ:
                                self.transition[s][a][kappa] = self.prob

                            else:
                                self.transition[s][a][kappa] = (1 - self.prob) / (self.numActions[s] - 1)

                            if self.grid[kappa] == self.goal:
                                self.reward[s][a][kappa] = self.mult2

                    elif a == 2:  # up
                        for kappa in self.transition[s][a]:

                            if kappa == s - (self.numZ * self.numCol):
                                self.transition[s][a][kappa] = self.prob

                            else:
                                self.transition[s][a][kappa] = (1 - self.prob) / (self.numActions[s] - 1)

                            if self.grid[kappa] == self.goal:
                                self.reward[s][a][kappa] = self.mult2

                    elif a == 3:  # down
                        for kappa in self.transition[s][a]:

                            if kappa == s + (self.numZ * self.numCol):
                                self.transition[s][a][kappa] = self.prob

                            else:
                                self.transition[s][a][kappa] = (1 - self.prob) / (self.numActions[s] - 1)

                            if self.grid[kappa] == self.goal:
                                self.reward[s][a][kappa] = self.mult2

                    elif a == 4:  # up-right
                        for kappa in self.transition[s][a]:

                            if kappa == s - (self.numZ * self.numCol) + self.numZ:
                                self.transition[s][a][kappa] = self.prob

                            else:
                                self.transition[s][a][kappa] = (1 - self.prob) / (self.numActions[s] - 1)

                            if self.grid[kappa] == self.goal:
                                self.reward[s][a][kappa] = self.mult2

                    elif a == 5:  # up-left
                        for kappa in self.transition[s][a]:

                            if kappa == s - (self.numZ * self.numCol) - self.numZ:
                                self.transition[s][a][kappa] = self.prob

                            else:
                                self.transition[s][a][kappa] = (1 - self.prob) / (self.numActions[s] - 1)

                            if self.grid[kappa] == self.goal:
                                self.reward[s][a][kappa] = self.mult2

                    elif a == 6:  # down-right
                        for kappa in self.transition[s][a]:

                            if kappa == s + (self.numZ * self.numCol) + self.numZ:
                                self.transition[s][a][kappa] = self.prob

                            else:
                                self.transition[s][a][kappa] = (1 - self.prob) / (self.numActions[s] - 1)

                            if self.grid[kappa] == self.goal:
                                self.reward[s][a][kappa] = self.mult2

                    elif a == 7:  # down-left
                        for kappa in self.transition[s][a]:

                            if kappa == s + (self.numZ * self.numCol) - self.numZ:
                                self.transition[s][a][kappa] = self.prob

                            else:
                                self.transition[s][a][kappa] = (1 - self.prob) / (self.numActions[s] - 1)

                            if self.grid[kappa] == self.goal:
                                self.reward[s][a][kappa] = self.mult2

                    elif a == 8:  # down nothing
                        for kappa in self.transition[s][a]:

                            if kappa == s:
                                self.transition[s][a][kappa] = self.prob

                            else:
                                self.transition[s][a][kappa] = (1 - self.prob) / (self.numActions[s] - 1)

                            if self.grid[kappa] == self.goal:
                                self.reward[s][a][kappa] = self.mult2

                    elif a == 9:  # f
                        for kappa in self.transition[s][a]:

                            if kappa == s + 1:
                                self.transition[s][a][kappa] = self.prob

                            else:
                                self.transition[s][a][kappa] = (1 - self.prob) / (self.numActions[s] - 1)

                            if self.grid[kappa] == self.goal:
                                self.reward[s][a][kappa] = self.mult2

                    elif a == 10:  # b
                        for kappa in self.transition[s][a]:

                            if kappa == s - 1:
                                self.transition[s][a][kappa] = self.prob

                            else:
                                self.transition[s][a][kappa] = (1 - self.prob) / (self.numActions[s] - 1)

                            if self.grid[kappa] == self.goal:
                                self.reward[s][a][kappa] = self.mult2

                    elif a == 11:  # f right
                        for kappa in self.transition[s][a]:

                            if kappa == s + 1 + self.numZ:
                                self.transition[s][a][kappa] = self.prob

                            else:
                                self.transition[s][a][kappa] = (1 - self.prob) / (self.numActions[s] - 1)

                            if self.grid[kappa] == self.goal:
                                self.reward[s][a][kappa] = self.mult2

                    elif a == 12:  # f left
                        for kappa in self.transition[s][a]:

                            if kappa == s + 1 - self.numZ:
                                self.transition[s][a][kappa] = self.prob

                            else:
                                self.transition[s][a][kappa] = (1 - self.prob) / (self.numActions[s] - 1)

                            if self.grid[kappa] == self.goal:
                                self.reward[s][a][kappa] = self.mult2

                    elif a == 13:  # f up
                        for kappa in self.transition[s][a]:

                            if kappa == s + 1 - (self.numZ * self.numCol):
                                self.transition[s][a][kappa] = self.prob

                            else:
                                self.transition[s][a][kappa] = (1 - self.prob) / (self.numActions[s] - 1)

                            if self.grid[kappa] == self.goal:
                                self.reward[s][a][kappa] = self.mult2

                    elif a == 14:  # f down
                        for kappa in self.transition[s][a]:

                            if kappa == s + 1 + (self.numZ * self.numCol):
                                self.transition[s][a][kappa] = self.prob

                            else:
                                self.transition[s][a][kappa] = (1 - self.prob) / (self.numActions[s] - 1)

                            if self.grid[kappa] == self.goal:
                                self.reward[s][a][kappa] = self.mult2

                    elif a == 15:  # b right
                        for kappa in self.transition[s][a]:

                            if kappa == s - 1 + self.numZ:
                                self.transition[s][a][kappa] = self.prob

                            else:
                                self.transition[s][a][kappa] = (1 - self.prob) / (self.numActions[s] - 1)

                            if self.grid[kappa] == self.goal:
                                self.reward[s][a][kappa] = self.mult2

                    elif a == 16:  # b left
                        for kappa in self.transition[s][a]:

                            if kappa == s - 1 - self.numZ:
                                self.transition[s][a][kappa] = self.prob

                            else:
                                self.transition[s][a][kappa] = (1 - self.prob) / (self.numActions[s] - 1)

                            if self.grid[kappa] == self.goal:
                                self.reward[s][a][kappa] = self.mult2

                    elif a == 17:  # b up
                        for kappa in self.transition[s][a]:

                            if kappa == s - 1 - (self.numZ * self.numCol):
                                self.transition[s][a][kappa] = self.prob

                            else:
                                self.transition[s][a][kappa] = (1 - self.prob) / (self.numActions[s] - 1)

                            if self.grid[kappa] == self.goal:
                                self.reward[s][a][kappa] = self.mult2

                    elif a == 18:  # b down
                        for kappa in self.transition[s][a]:

                            if kappa == s - 1 + (self.numZ * self.numCol):
                                self.transition[s][a][kappa] = self.prob

                            else:
                                self.transition[s][a][kappa] = (1 - self.prob) / (self.numActions[s] - 1)

                            if self.grid[kappa] == self.goal:
                                self.reward[s][a][kappa] = self.mult2

                    elif a == 19:  # f up-right
                        for kappa in self.transition[s][a]:

                            if kappa == s + 1 - (self.numZ * self.numCol) + self.numZ:
                                self.transition[s][a][kappa] = self.prob

                            else:
                                self.transition[s][a][kappa] = (1 - self.prob) / (self.numActions[s] - 1)

                            if self.grid[kappa] == self.goal:
                                self.reward[s][a][kappa] = self.mult2

                    elif a == 20:  # f up-left
                        for kappa in self.transition[s][a]:

                            if kappa == s + 1 - (self.numZ * self.numCol) - self.numZ:
                                self.transition[s][a][kappa] = self.prob

                            else:
                                self.transition[s][a][kappa] = (1 - self.prob) / (self.numActions[s] - 1)

                            if self.grid[kappa] == self.goal:
                                self.reward[s][a][kappa] = self.mult2

                    elif a == 21:  # f down-right
                        for kappa in self.transition[s][a]:

                            if kappa == s + 1 + (self.numZ * self.numCol) + self.numZ:
                                self.transition[s][a][kappa] = self.prob

                            else:
                                self.transition[s][a][kappa] = (1 - self.prob) / (self.numActions[s] - 1)

                            if self.grid[kappa] == self.goal:
                                self.reward[s][a][kappa] = self.mult2

                    elif a == 22:  # f down-left
                        for kappa in self.transition[s][a]:

                            if kappa == s + 1 + (self.numZ * self.numCol) - self.numZ:
                                self.transition[s][a][kappa] = self.prob

                            else:
                                self.transition[s][a][kappa] = (1 - self.prob) / (self.numActions[s] - 1)

                            if self.grid[kappa] == self.goal:
                                self.reward[s][a][kappa] = self.mult2

                    elif a == 23:  # b up-right
                        for kappa in self.transition[s][a]:

                            if kappa == s - 1 - (self.numZ * self.numCol) + self.numZ:
                                self.transition[s][a][kappa] = self.prob

                            else:
                                self.transition[s][a][kappa] = (1 - self.prob) / (self.numActions[s] - 1)

                            if self.grid[kappa] == self.goal:
                                self.reward[s][a][kappa] = self.mult2

                    elif a == 24:  # b up-left
                        for kappa in self.transition[s][a]:

                            if kappa == s - 1 - (self.numZ * self.numCol) - self.numZ:
                                self.transition[s][a][kappa] = self.prob

                            else:
                                self.transition[s][a][kappa] = (1 - self.prob) / (self.numActions[s] - 1)

                            if self.grid[kappa] == self.goal:
                                self.reward[s][a][kappa] = self.mult2

                    elif a == 25:  # b down-right
                        for kappa in self.transition[s][a]:

                            if kappa == s - 1 + (self.numZ * self.numCol) + self.numZ:
                                self.transition[s][a][kappa] = self.prob

                            else:
                                self.transition[s][a][kappa] = (1 - self.prob) / (self.numActions[s] - 1)

                            if self.grid[kappa] == self.goal:
                                self.reward[s][a][kappa] = self.mult2


                    else:  # b down-left
                        for kappa in self.transition[s][a]:

                            if kappa == s - 1 + (self.numZ * self.numCol) - self.numZ:
                                self.transition[s][a][kappa] = self.prob

                            else:
                                self.transition[s][a][kappa] = (1 - self.prob) / (self.numActions[s] - 1)

                            if self.grid[kappa] == self.goal:
                                self.reward[s][a][kappa] = self.mult2
