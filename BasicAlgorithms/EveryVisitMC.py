import PolicyIteration as pi
import numpy as np
import random as rnd
import MDP_1D as mdp1

def EveryVMC(states, actions, reward, numR, numC, grid, wall, mult, goal):
    # nStates and nActions are 1-by-1 dictionaries with all the next states mapping with all possible actions of each state
    nStates, nActions = PossibleStates(states, actions, grid, numR, numC, wall, mult)

    policy = Initialize(states, nActions)
    print(policy)
    MonteCarloRun = 30000
    Q, Returns = InitQ(states, actions, grid, wall, mult, numR, numC)
    print(Q)
    # Returns =
    # G = 0

    for i in range(MonteCarloRun):

        episode = GenEpisode(policy, numC, reward, wall, grid, nStates, nActions)
        # print(episode)
        G = 0  # total reward for the episode
        ep_s = []  # that's the state list of the episodes
        for epL in range(len(episode), 0, -1):
            if grid[episode[epL - 1][0]] == goal:
                continue
            s = episode[epL - 1][0]  # zero for the state
            a = episode[epL - 1][1]  # one for the action
            ep_s.append(s)

            # This is the return(reward) that follows the first occurrence os s, a
            G += episode[epL - 1][2] # two is for the reward
            Returns[s][a].append(G)
            Q[s][a] = np.mean(Returns[s][a])

        # e-greedy policy improvement (we have to decrease this as the episodes pass)
        e = 0.7
        for ss in ep_s:
            a_star = maxQ(Q, ss)

            for aa in nActions[ss]:
                if aa == a_star:
                    policy[ss][aa] = 1 - e + e / len(nActions[ss])
                else:
                    policy[ss][aa] = e / len(nActions[ss])

    print(Q)
    print(policy)
    return policy, 0

    # optimal a based on Q

def PossibleStates(states, actions, grid, rows, col, wall, mult):

    arStates = AroundStates(states, actions, grid, rows, col, wall, mult)

    nStates = {}
    nActions = {}

    for s in states:

        if grid[s] in wall:
            continue
        nStates[s] = []
        nActions[s] = []
        for a in actions:
            if validAction(s, a, grid, rows, col, wall, mult):
                if a == 0:  # right
                    nStates[s] = arStates  # .append(s + 1)
                    nActions[s].append(a)

                elif a == 1:  # left
                    nStates[s] = arStates[s]  # .append(s - 1)
                    nActions[s].append(a)

                elif a == 2:  # up
                    nStates[s] = arStates[s]  # .append(s - col)
                    nActions[s].append(a)

                else:  # down
                    nStates[s] = arStates[s]  # .append(s + col)
                    nActions[s].append(a)

    return nStates, nActions

def maxQ(Q, state):
    a_star = -1

    maxV = -1000
    for a in Q[state]:

        if Q[state][a] > maxV:
            a_star = a
            maxV = Q[state][a]

    return a_star

    # This method creates the first init policy


def Initialize(states, nActions):
    policy = {}

    for s in states:
        if s in nActions:
            policy[s] = {}
            for i in range(len(nActions[s])):
                policy[s][nActions[s][i]] = 1 / len(nActions[s])

    return policy

    # This method creates the episodes for our MC


def GenEpisode(policy, numC, reward, wall, grid, nStates, nActions):
    # the length of episodes will be in range of our grid i.e numColumns
    ep = []

    # getting a random first state for the episode
    val = True
    while val:
        s = rnd.randint(1, len(policy))
        s -= 1
        if grid[s] in wall:
            val = True
        else:
            val = False

    for i in range(numC):

        prob = rnd.random()
        cnt = 0
        for j in range(len(nActions[s])):
            a = nActions[s][j]
            cnt = cnt + policy[s][a]

            if prob <= cnt:
                # state - action - reward
                ep.append((s, a, reward[s][a][nStates[s][j]]))
                ss = nStates[s][j]
                s = ss
                break

        # s = ss

    return ep


def InitQ(states, actions, grid, wall, mult, rows, col):
    Q = {}
    Returns = {}

    for s in states:

        if grid[s] in wall:
            continue
        Q[s] = {}
        Returns[s] = {}

        for a in actions:
            if pi.validAction(s, a, grid, rows, col, wall, mult):
                if a == 0:  # right
                    Q[s][a] = 0
                    Returns[s][a] = []

                elif a == 1:  # left
                    Q[s][a] = 0
                    Returns[s][a] = []

                elif a == 2:  # up
                    Q[s][a] = 0
                    Returns[s][a] = []

                else:  # down
                    Q[s][a] = 0
                    Returns[s][a] = []

    return Q, Returns

def validAction(s, a, grid, rows, col, wall, mult):
    # 0-right 1-left 2-up 3-down
    i = grid[s] // mult
    j = grid[s] % mult
    ww = grid[s]
    wa = np.array(wall)

    if ww in wall:
        return False

    if a == 0:  # right

        bb = False
        for w in wall:
            if j + 1 == w % mult and i == w // mult:
                bb = True

        if j + 1 > col - 1 or bb:
            return False
        else:
            return True

    elif a == 1:  # left

        bb = False
        for w in wall:
            if j - 1 == w % mult and i == w // mult:
                bb = True

        if j - 1 < 0 or bb:
            return False
        else:
            return True

    elif a == 2:  # up

        bb = False
        for w in wall:
            if j == w % mult and i - 1 == w // mult:
                bb = True

        if i - 1 < 0 or bb:
            return False
        else:
            return True

    else:  # down

        bb = False
        for w in wall:
            if j == w % mult and i + 1 == w // mult:
                bb = True

        if i + 1 > rows - 1 or bb:
            return False
        else:
            return True

def AroundStates(states, actions, grid, rows, col, wall, mult):
    # this method find the around states of the current state
    ArStates = {}

    for s in states:
        ArStates[s] = []

        for a in actions:

            if validAction(s, a, grid, rows, col, wall, mult):

                if a == 0:  # right
                    ArStates[s].append(s + 1)

                elif a == 1:  # left
                    ArStates[s].append(s - 1)

                elif a == 2:  # up
                    ArStates[s].append(s - col)

                else:  # down
                    ArStates[s].append(s + col)


    return ArStates


if __name__ == "__main__":
    rows = 3
    cols = 3
    w = rows * cols
    goal = (rows - 1) * w + (cols - 1)
    wall = [(rows - 2) * w + (cols - 1)]  # , (rows - 2) * w + (cols - 2)]

    mdp = mdp1.MDP(rows, cols, wall, goal)
    mdp.CreateGrid()
    mdp.TnR()
    mdp.InitRnT()

    EveryVMC(mdp.states, mdp.actions, mdp.reward, mdp.numRows, mdp.numCol, mdp.grid, mdp.wall, mdp.mult, mdp.goal)