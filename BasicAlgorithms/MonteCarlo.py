import PolicyIteration as pi
import numpy as np
import random as rnd

def FirstVisitMC(states, actions, reward, transition, gamma, numR, numC, grid, wall, goal, mult):

    # nStates and nActions are 1-by-1 dictionaries with all the next states mapping with all possible actions of each state
    nStates, nActions = pi.PossibleStates(states, actions, grid, numR, numC, wall, mult)

    policy = Initialize(states, nActions)
    print(policy)
    MonteCarloRun = 10000
    Q, Returns = InitQ(states, actions, grid, wall, mult, numR, numC)
    print(Q)
    # Returns =

    for i in range(MonteCarloRun):

        episode = GenEpisode(policy, numC, reward, wall, grid, nStates, nActions)
        #print(episode)
        G = 0  # total reward for the episode
        ep_s = []   # that's the state list of the episodes
        for epL in range(len(episode), 0, -1):
            s = episode[epL-1][0]
            a = episode[epL-1][1]
            ep_s.append(s)

            # This is the return(reward) that follows the first occurrence os s, a
            G += episode[epL-1][2]
            Returns[s][a].append(G)
            Q[s][a] = np.mean(Returns[s][a])

        # e-greedy policy improvement (we have to decrease this as the episodes pass)
        e = 0.7
        for ss in ep_s:
            a_star = maxQ(Q, ss)

            for aa in nActions[ss]:
                if aa == a_star:
                    policy[ss][aa] = 1 - e + e/len(nActions[ss])
                else:
                    policy[ss][aa] = e/len(nActions[ss])


    print(Q)
    print(policy)
    return policy, 0

# optimal a based on Q
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
                policy[s][nActions[s][i]] = 1/len(nActions[s])


    return policy


# This method creates the episodes for our MC
def GenEpisode(policy, numC, reward, wall, grid, nStates, nActions):

    # the length of episodes will be in range of our grid i.e numColumns
    ep = []

    # getting a random first state for the episode
    s = rnd.randint(1, len(policy))
    s -= 1
    if grid[s] == wall:
        s -= 1

    for i in range(numC):

        prob = rnd.random()
        cnt = 0
        for j in range(len(nActions[s])):
            a = nActions[s][j]
            cnt = cnt + policy[s][a]

            if prob <= cnt:
                # state - action - reward
                ep.append((s, a, reward[s, a, nStates[s][j]]))
                ss = nStates[s][j]
                s = ss
                break


        #s = ss

    return ep


def InitQ(states, actions, grid, wall, mult, rows, col):

    Q = {}
    Returns = {}

    for s in states:

        if (grid[s] // mult == wall // mult) and (grid[s] % mult == wall % mult):
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

                elif a == 3:  # down
                    Q[s][a] = 0
                    Returns[s][a] = []

                elif a == 4:  # up-right
                    Q[s][a] = 0
                    Returns[s][a] = []

                elif a == 5:  # up-left
                    Q[s][a] = 0
                    Returns[s][a] = []

                elif a == 6:  # down-right
                    Q[s][a] = 0
                    Returns[s][a] = []

                elif a == 7:  # down-left
                    Q[s][a] = 0
                    Returns[s][a] = []

                else:  # do-nothing action
                    Q[s][a] = 0
                    Returns[s][a] = []

    return Q, Returns