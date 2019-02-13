import numpy as np
import PolicyIteration as pi
import ValueIteration as vi
import AlternatePI as api
import MDP as m
import AltMDP as am
import collections
import PCA as pca
from sklearn.decomposition import PCA


def nextState(s, a, col):
    if a == 0:  # right
        return s + 1

    elif a == 1:  # left
        return s - 1

    elif a == 2:  # up
        return s - col

    elif a == 3:  # down
        return s + col

    elif a == 4:  # up-right
        return s - col + 1

    elif a == 5:  # up-left
        return s - col - 1

    elif a == 6:  # down-right
        return s + col + 1

    elif a == 7:  # down-left
        return s + col - 1

    else:  # do-nothing action
        return s

def neighboors(states, actions, grid, goal, wall, mult, col, rows):
    # 0-right 1-left 2-up 3-down
    neig = {}

    for s in states:

        i = grid[s] // mult
        j = grid[s] % mult

        if (i == wall // mult) and (j == wall % mult):
            continue

        if (i == goal // mult) and (j == goal % mult):
            continue

        neig[s] = []
        for a in actions:
            if pi.validAction(s, a, grid, rows, col, wall, mult):
                if a == 0:  # right
                    neig[s].append(s + 1)

                elif a == 1:  # left
                    neig[s].append(s - 1)

                elif a == 2:  # up
                    neig[s].append(s - col)

                elif a == 3:  # down
                    neig[s].append(s + col)

                elif a == 4:  # up-right
                    neig[s].append(s - col + 1)

                elif a == 5:  # up-left
                    neig[s].append(s - col - 1)

                elif a == 6:  # down-right
                    neig[s].append(s + col + 1)

                elif a == 7:  # down-left
                    neig[s].append(s + col - 1)

    return neig



def optimalPolicy(states, actions, grid, goal, wall, policy, col, rows, mult):

    nei = neighboors(states, actions, grid, goal, wall, mult, col, rows)
    print(nei)

    for s in states:
        if (grid[s] // mult == wall // mult) and (grid[s] % mult == wall % mult):
            continue

        if (grid[s] // mult == goal // mult) and (grid[s] % mult == goal % mult):
            continue

        # Here starts the Deapth First Search for the sortest path in my grid
        seen, queue = set([s]), collections.deque([s])
        depth = collections.deque([0])
        found = True
        path = 0
        while queue:  # queue
            vertex = queue.popleft()
            dep = depth.popleft()

            if grid[vertex] == goal:
                print("From node: ", s, ",", dep, " steps to goal --- DFS")
                continue

            for node in nei[vertex]:
                if node not in seen:
                    seen.add(node)
                    queue.append(node)
                    depth.append(dep + 1)

        gg = True
        cnt = 0
        cur_state = s
        while gg:
            if grid[cur_state] == goal:
                print("From node: ", s, ",", cnt, " steps to goal --- policy")
                gg = False

            a = policy[cur_state]
            if pi.validAction(cur_state, a, grid, rows, col, wall, mult):
                cur_state = nextState(cur_state, a, col)
                cnt += 1

            else:
                print("invalid policy, problem with the RL algorithms")
                print(policy[cur_state], grid[cur_state])
                gg = False

            if cnt > len(states)*len(states):
                print("Too many moves in the grid to find the goal state")
                gg = False


def runVI():

    nRows = 3
    nCols = 3
    w = nRows * nCols
    wallP = 1 * w + 2
    goalP = 2 * w + 2

    mdp = m.MDP(nRows, nRows, wallP, goalP)
    mdp.CreateGrid()
    mdp.InitRnT()

    vi.ValueIteration(mdp.states, mdp.actions, mdp.reward, mdp.transition, mdp.gamma, mdp.numRows, mdp.numCol,
                          mdp.grid, mdp.wall, mdp.goal, mdp.mult)


def runPI():

    nRows = 3
    nCols = 3
    w = nRows * nCols
    wallP = 1 * w + 2
    goalP = 2 * w + 2

    mdp = m.MDP(nRows, nRows, wallP, goalP)
    mdp.CreateGrid()
    mdp.InitRnT()


    v, p = pi.PolicyIteration(mdp.states, mdp.actions, mdp.reward, mdp.transition, mdp.gamma, mdp.numRows, mdp.numCol,
                              mdp.grid, mdp.wall, mdp.goal, mdp.mult)

    # well this is the part where we will use pca with the "optimal" Value function that PI returns
    # pca.Pca(v, p, mdp.numRows, mdp.numCol)

    optimalPolicy(mdp.states, mdp.actions, mdp.grid, mdp.goal, mdp.wall, p, mdp.numCol, mdp.numRows, mdp.mult)


def runAltPI():

    nRows = 3
    nCols = 3
    w = nRows * nCols
    wallP = 1 * w + 2
    goalP = 2 * w + 2

    mdp = am.MDP(nRows, nRows, wallP, goalP)
    mdp.CreateGrid()
    mdp.InitRnT()

    pi.PolicyIteration(mdp.states, mdp.actions, mdp.reward, mdp.transition, mdp.gamma, mdp.numRows, mdp.numCol,
                       mdp.grid, mdp.wall, mdp.goal, mdp.mult)


def main():

    runPI()
    # runVI()

if __name__ == "__main__":
    main()
