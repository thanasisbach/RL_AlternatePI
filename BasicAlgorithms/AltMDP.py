import numpy as np
import AlternatePI as pi


class AltMDP:

    def __init__(self, numRows, numCol, wall, goal):
        self.numA = numCol
        self.numB = numRows
        self.wall = wall
        self.goal = goal

        self.numActions = []  # this variable tell us how many actions i have from a state
        self.actions = [0, 1, 2, 3, 4, 5, 6, 7, 8] #  all actions
        self.actionsA = [0, 1, 8]  # right-0, left-1, do-noth-8
        self.actionsB = [2, 3, 8]  # up-2, down-3, do-noth-8

        self.states = []  # all states
        self.statesA = []  # states of dimension A
        self.statesB = []  # states of dimension B
        self.grid = []
        self.gridStates = {}  # this is a mapping from grid positions to the states
        self.transition = {}
        # np.zeros((numRows * numCol, len(self.actionsA) * len(self.actionsB),
        #                            numRows * numCol))  # transition and reward tables are n x (k1 x k2) x n dimensions
        self.reward = {}
        # np.zeros((numRows * numCol, len(self.actionsA) * len(self.actionsB),
        #                        numRows * numCol))  # where n is the state space and k is the action space
        self.gamma = 0.9  # in range (0,1)
        self.mult = numRows * numCol
        self.prob = 0.8

    def CreateGrid(self):

        for i in range(self.numA):
            self.statesA.append(i)
        for j in range(self.numB):
            self.statesB.append(j)

        cnt = 0
        for i in range(self.numB):  # rows
            for j in range(self.numA):  # columns
                self.grid.append(self.mult * i + j)
                self.gridStates[self.mult * i + j] = cnt
                self.states.append(cnt)
                self.numActions.append(0)  # combines actions of the two dimensions
                cnt += 1
                # self.reward.append(0)

    def TnR(self):

        arStates = pi.AroundStates(self.states, self.actions, self.grid, self.numB, self.numA, self.wall, self.mult)

        for s in self.states:

            if self.grid[s] in self.wall:
                continue

            self.transition[s] = {}
            self.reward[s] = {}
            for a in self.actions:
                if pi.validAction(s, a, self.grid, self.numB, self.numA, self.wall, self.mult):
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

                    else:  # do-nothing action
                        self.transition[s][a] = {}
                        self.reward[s][a] = {}

                        for kappa in arStates[s]:
                            self.transition[s][a][kappa] = 0  # [s + 1] = 0
                            self.reward[s][a][kappa] = 0  # [s + 1] = 0


    def InitRnT(self):
        # print(self.mult)
        # for every state
        for s in self.states:

            # for every possible action
            for a in self.actions:
                # i have to check if the action is valid from this state
                if pi.validAction(s, a, self.grid, self.numB, self.numA, self.wall, self.mult):
                    self.numActions[s] += 1  # then count how many actions the state has

            # in this loop i assign equal probability in the transition matrix and the reward in final state only
            for a in self.actions:
                if pi.validAction(s, a, self.grid, self.numB, self.numA, self.wall, self.mult):

                    if a == 0:  # right

                        for kappa in self.transition[s][a]:

                            if kappa == s + 1:
                                self.transition[s][a][kappa] = self.prob

                            else:
                                self.transition[s][a][kappa] = (1 - self.prob) / (self.numActions[s] - 1)

                            if self.grid[kappa] == self.goal:
                                self.reward[s][a][kappa] = self.mult


                    elif a == 1:  # left

                        for kappa in self.transition[s][a]:

                            if kappa == s - 1:
                                self.transition[s][a][kappa] = self.prob

                            else:
                                self.transition[s][a][kappa] = (1 - self.prob) / (self.numActions[s] - 1)

                            if self.grid[kappa] == self.goal:
                                self.reward[s][a][kappa] = self.mult

                    elif a == 2:  # up

                        for kappa in self.transition[s][a]:

                            if kappa == s - self.numA:
                                self.transition[s][a][kappa] = self.prob

                            else:
                                self.transition[s][a][kappa] = (1 - self.prob) / (self.numActions[s] - 1)

                            if self.grid[kappa] == self.goal:
                                self.reward[s][a][kappa] = self.mult

                    elif a == 3:  # down

                        for kappa in self.transition[s][a]:

                            if kappa == s + self.numA:
                                self.transition[s][a][kappa] = self.prob

                            else:
                                self.transition[s][a][kappa] = (1 - self.prob) / (self.numActions[s] - 1)

                            if self.grid[kappa] == self.goal:
                                self.reward[s][a][kappa] = self.mult




                    elif a == 4:  # up-right

                        for kappa in self.transition[s][a]:

                            if kappa == s - self.numA + 1:
                                self.transition[s][a][kappa] = self.prob

                            else:
                                self.transition[s][a][kappa] = (1 - self.prob) / (self.numActions[s] - 1)

                            if self.grid[kappa] == self.goal:
                                self.reward[s][a][kappa] = self.mult

                    elif a == 5:  # up-left

                        for kappa in self.transition[s][a]:

                            if kappa == s - self.numA - 1:
                                self.transition[s][a][kappa] = self.prob

                            else:
                                self.transition[s][a][kappa] = (1 - self.prob) / (self.numActions[s] - 1)

                            if self.grid[kappa] == self.goal:
                                self.reward[s][a][kappa] = self.mult

                    elif a == 6:  # down-right

                        for kappa in self.transition[s][a]:

                            if kappa == s + self.numA + 1:
                                self.transition[s][a][kappa] = self.prob

                            else:
                                self.transition[s][a][kappa] = (1 - self.prob) / (self.numActions[s] - 1)

                            if self.grid[kappa] == self.goal:
                                self.reward[s][a][kappa] = self.mult

                    elif a == 7:  # down-left

                        for kappa in self.transition[s][a]:

                            if kappa == s + self.numA - 1:
                                self.transition[s][a][kappa] = self.prob

                            else:
                                self.transition[s][a][kappa] = (1 - self.prob) / (self.numActions[s] - 1)

                            if self.grid[kappa] == self.goal:
                                self.reward[s][a][kappa] = self.mult

                    else:  # do-nothing action

                        for kappa in self.transition[s][a]:

                            if kappa == s:
                                self.transition[s][a][kappa] = self.prob

                            else:
                                self.transition[s][a][kappa] = (1 - self.prob) / (self.numActions[s] - 1)

                            if self.grid[kappa] == self.goal:
                                self.reward[s][a][kappa] = self.mult
            # print(s, self.numActions[s])