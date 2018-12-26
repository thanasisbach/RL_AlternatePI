from typing import List, Any
import numpy as np
import PolicyIteration as pi

class MDP:

    def __init__(self, numRows, numCol, wall, goal):
        self.numRows = numRows
        self.numCol = numCol
        self.wall = wall
        self.goal = goal
        self.numActions = []  # this variable tell us how many actions i have from a state

        self.actions = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # lets make four actions right-0, left-1, up-2, down-3, upRight-4, upLeft-5, downRight-6, downLeft-7, do-noth-8
        self.states = []  # grid and states are one-by-one lists, to know which state
        self.grid = []
        self.transition = np.zeros((numRows * numCol, len(self.actions),
                                    numRows * numCol))  # transition and reward tables are n x k x n dimensions
        self.reward = np.zeros((numRows * numCol, len(self.actions),
                                numRows * numCol))  # where n is the state space and k is the action space
        self.gamma = 0.9  # in range (0,1)
        self.mult = numRows * numCol

    def CreateGrid(self):

        cnt = 0
        for i in range(0, self.numRows):
            for j in range(0, self.numCol):
                self.grid.append(self.mult * i + j)  # div gives i and mod gives j
                self.states.append(cnt)
                self.numActions.append(0)
                cnt += 1
                # self.reward.append(0)

    def InitRnT(self):
        # print(self.mult)
        # for every state
        for s in self.states:

            # for every possible action
            for a in self.actions:
                # i have to check if the action is valid from this state
                if pi.validAction(s, a, self.grid, self.numRows, self.numCol, self.wall, self.mult):
                    self.numActions[s] += 1  # then count how many actions the state has

            # in this loop i assign equal probability in the transition matrix and the reward in final state only
            for a in self.actions:
                if pi.validAction(s, a, self.grid, self.numRows, self.numCol, self.wall, self.mult):

                    if a == 0:  # right
                        self.transition[s, a, s + 1] = 1 / self.numActions[s]
                        if self.grid[s + 1] // self.mult == self.goal // self.mult and self.grid[s + 1] % self.mult == self.goal % self.mult:
                            self.reward[s, a, s + 1] = self.mult

                    elif a == 1:  # left
                        self.transition[s, a, s - 1] = 1 / self.numActions[s]
                        if self.grid[s - 1] // self.mult == self.goal // self.mult and self.grid[s - 1] % self.mult == self.goal % self.mult:
                            self.reward[s, a, s - 1] = self.mult

                    elif a == 2:  # up
                        self.transition[s, a, s - self.numCol] = 1 / self.numActions[s]
                        if self.grid[s - self.numCol] // self.mult == self.goal // self.mult and self.grid[
                            s - self.numCol] % self.mult == self.goal % self.mult:
                            self.reward[s, a, s - self.numCol] = self.mult

                    elif a == 3:  # down
                        self.transition[s, a, s + self.numCol] = 1 / self.numActions[s]
                        if self.grid[s + self.numCol] // self.mult == self.goal // self.mult and self.grid[
                            s + self.numCol] % self.mult == self.goal % self.mult:
                            self.reward[s, a, s + self.numCol] = self.mult

                    elif a == 4:  # up-right
                        self.transition[s, a, s - self.numCol + 1] = 1 / self.numActions[s]
                        if self.grid[s - self.numCol + 1] // self.mult == self.goal // self.mult and self.grid[
                            s - self.numCol + 1] % self.mult == self.goal % self.mult:
                            self.reward[s, a, s - self.numCol + 1] = self.mult

                    elif a == 5:  # up-left
                        self.transition[s, a, s - self.numCol - 1] = 1 / self.numActions[s]
                        if self.grid[s - self.numCol - 1] // self.mult == self.goal // self.mult and self.grid[
                            s - self.numCol - 1] % self.mult == self.goal % self.mult:
                            self.reward[s, a, s - self.numCol - 1] = self.mult

                    elif a == 6:  # down-right
                        self.transition[s, a, s + self.numCol + 1] = 1 / self.numActions[s]
                        if self.grid[s + self.numCol + 1] // self.mult == self.goal // self.mult and self.grid[
                            s + self.numCol + 1] % self.mult == self.goal % self.mult:
                            self.reward[s, a, s + self.numCol + 1] = self.mult

                    elif a == 7:  # down-left
                        self.transition[s, a, s + self.numCol - 1] = 1 / self.numActions[s]
                        if self.grid[s + self.numCol - 1] // self.mult == self.goal // self.mult and self.grid[
                            s + self.numCol - 1] % self.mult == self.goal % self.mult:
                            self.reward[s, a, s + self.numCol - 1] = self.mult

                    else:  # do-nothing action
                        self.transition[s, a, s] = 1 / self.numActions[s]
                        if self.grid[s] // self.mult == self.goal // self.mult and self.grid[
                            s] % self.mult == self.goal % self.mult:
                            self.reward[s, a, s] = self.mult
            # print(s, self.numActions[s])
