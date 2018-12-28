import numpy as np


def AlternatePI(states, stateC, stateR, actions, actC, actR, grid, gridStates, wall, goal, mult):

    nStates, nActions = PossibleStates(states, actions, grid, len(stateR), len(stateC), wall, mult)

    polC = InitPolicy()
    polR = InitPolicy()
    # here we must implement the new alternative PI

    changeValue = True
    while changeValue:




        changeValue = False


def PossibleStates(states, actions, grid, rows, col, wall, mult):

    nStates = {}
    nActions = {}

    for s in states:

        if (grid[s] // mult == wall // mult) and (grid[s] % mult == wall % mult):
            continue
        nStates[s] = []
        nActions[s] = []
        for a in actions:
            if validAction(s, a, grid, rows, col, wall, mult):
                if a == 0:  # right
                    nStates[s].append(s + 1)
                    nActions[s].append(a)

                elif a == 1:  # left
                    nStates[s].append(s - 1)
                    nActions[s].append(a)

                elif a == 2:  # up
                    nStates[s].append(s - col)
                    nActions[s].append(a)

                elif a == 3:  # down
                    nStates[s].append(s + col)
                    nActions[s].append(a)

                elif a == 4:  # up-right
                    nStates[s].append(s - col + 1)
                    nActions[s].append(a)

                elif a == 5:  # up-left
                    nStates[s].append(s - col - 1)
                    nActions[s].append(a)

                elif a == 6:  # down-right
                    nStates[s].append(s + col + 1)
                    nActions[s].append(a)

                elif a == 7:  # down-left
                    nStates[s].append(s + col - 1)
                    nActions[s].append(a)

                else:  # do-nothing action
                    nStates[s].append(s)
                    nActions[s].append(a)

    return nStates, nActions

def InitPolicy():  # policy (random) initialization
    pol = []

    return pol


def DecombinePolicy(policy):  # Information aliasing problem was detected here

    for i in policy:
        if i == 0:  # right
            a1 = 0
            a2 = 8

        elif i == 1:  # left
            a1 = 1
            a2 = 8

        elif i == 2:  # up
            a1 = 8
            a2 = 2

        elif i == 3:  # down
            a1 = 8
            a2 = 3
        elif i == 4:  # up-right
            a1 = 0
            a2 = 2

        elif i == 5:  # up-left
            a1 = 1
            a2 = 2

        elif i == 6:  # down-right
            a1 = 0
            a2 = 3

        elif i == 7:  # down-left
            a1 = 1
            a2 = 3

        elif i == 8:  # do-nothing action
            a1 = 8
            a2 = 8


def CombineActions(a1, a2):  # columns action - rows action

    if a1 == 0:  # right
        if a2 == 2:
            return 4  # up-right

        elif a2 == 3:
            return 6  # down-right

        else:
            return a1  # right

    elif a1 == 1:  # left
        if a2 == 2:
            return 5  # up-left

        elif a2 == 3:
            return 7  # down-left

        else:
            return a1  # left

    else:  # do nothing
        if a2 == 8:
            return 8  # do nothing

        else:
            return a2  # whatever this action is

def validAction(s, a, grid, rows, col, wall, mult):
    # 0-right 1-left 2-up 3-down
    i = grid[s] // mult
    j = grid[s] % mult

    if i == wall // mult and j == wall % mult:
        return False

    if a == 0:  # right
        if j + 1 > col - 1 or (j + 1 == wall % mult and i == wall // mult):
            return False
        else:
            return True

    elif a == 1:  # left
        if j - 1 < 0 or (j - 1 == wall % mult and i == wall // mult):
            return False
        else:
            return True

    elif a == 2:  # up
        if i - 1 < 0 or (i - 1 == wall // mult and j == wall % mult):
            return False
        else:
            return True

    elif a == 3:  # down
        if i + 1 > rows - 1 or (i + 1 == wall // mult and j == wall % mult):
            return False
        else:
            return True

    elif a == 4:  # up-right
        if (i - 1 < 0 or j + 1 > col - 1) or (i - 1 == wall // mult and j + 1 == wall % mult):
            return False
        else:
            return True

    elif a == 5:  # up-left
        if (i - 1 < 0 or j - 1 < 0) or (i - 1 == wall // mult and j - 1 == wall % mult):
            return False
        else:
            return True

    elif a == 6:  # down-right
        if (i + 1 > rows - 1 or j + 1 > col - 1) or (i + 1 == wall // mult and j + 1 == wall % mult):
            return False
        else:
            return True

    elif a == 7:  # down-left
        if (i + 1 > rows - 1 or j - 1 < 0) or (i + 1 == wall // mult and j - 1 == wall % mult):
            return False
        else:
            return True

    else:  # do nothing
        return True