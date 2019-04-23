import numpy as np
import random


# This is the classic policy iteration algorithm
def PolicyIteration(states, actions, reward, transition, gamma, numR, numC, grid, wall, goal, mult):

    nStates, nActions = PossibleStates(states, actions, grid, numR, numC, wall, mult)
    # print(nStates,"\n", nActions)

    policy = initPolicy(states, nActions)  # [0 for s in states]

    Value = np.zeros(len(states))  # this is the value function
    print("Initial policy", policy)

    valueChange = True
    iter = 0
    pi_iter = 0

    while valueChange:
        valueChange = False
        iter += 1

        # Value iteration part
        for s in states:

            if grid[s] in wall:
                continue

            V = 0
            for s1 in nStates[s]:  # for loop for the next state
                if (s1 in transition[s][policy[s]]) and (s1 in reward[s][policy[s]]):
                    V += transition[s][policy[s]][s1] * (reward[s][policy[s]][s1] + gamma * Value[s1])
                # reward[s, policy[s], s1] + gamma * Value[s1] * transition[s, policy[s], s1]
                # reward[s, policy[s], s1] + gamma * Value[s1] * transition[s, policy[s], s1]

            Value[s] = V

        # policy evaluation part
        for s in states:

            if grid[s] in wall:
                continue

            q_best = Value[s]  # we assume that the current value is the best and we improve it
            cnt = 0
            for a in nActions[s]:  # maximize over actions
                pi_iter += 1
                s1 = nStates[s][cnt]
                q_sa = transition[s][a][s1] * (reward[s][a][s1] + gamma * Value[s1])
                if q_sa > q_best:
                    policy[s] = a
                    q_best = q_sa
                    valueChange = True

                cnt += 1

    print("Num iters", iter)
    print("Policy Improvement iters", pi_iter)
    GraphThePolicy(policy, Value, numR, numC)

    return Value, policy, iter


def PossibleStates(states, actions, grid, rows, col, wall, mult):

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


def GraphThePolicy(policy, value, row, col):
    # list, lists
    act = ""
    val = ""
    cnt = 0
    for i in range(0, row):
        for j in range(0, col):

            if policy[cnt] == 0:  # right
                act = act + "right" + "||"
            elif policy[cnt] == 1:  # left
                act = act + "left" + "||"
            elif policy[cnt] == 2:  # up
                act = act + "up" + "||"
            elif policy[cnt] == 3:  # down
                act = act + "down" + "||"
            elif policy[cnt] == 4:  # up-right
                act = act + "up-right" + "||"
            elif policy[cnt] == 5:  # up-left
                act = act + "up-left" + "||"
            elif policy[cnt] == 6:  # down-right
                act = act + "down-right" + "||"
            elif policy[cnt] == 7:  # down-left
                act = act + "down-left" + "||"
            elif policy[cnt] == 8:  # do nothing
                act = act + "do-nothing" + "||"
            else: # none
                act = act + "None" + "||"

            val = val + str(value[cnt]) + "||"
            cnt += 1

        print("||" + act)
        print("||"+ val)
        print("-----------------------")
        act = ""
        val = ""


def initPolicy(states, posActs):

    policy = [None for s in states]

    for s in states:
        if s in posActs:
            r = random.randint(1, len(posActs[s]))
            policy[s] = posActs[s][r-1]

    return policy


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

    elif a == 3:  # down

        bb = False
        for w in wall:
            if j == w % mult and i + 1 == w // mult:
                bb = True

        if i + 1 > rows - 1 or bb:
            return False
        else:
            return True

    elif a == 4:  # up-right

        bb = False
        for w in wall:
            if j + 1 == w % mult and i - 1 == w // mult:
                bb = True

        if (i - 1 < 0 or j + 1 > col - 1) or bb:
            return False
        else:
            return True

    elif a == 5:  # up-left

        bb = False
        for w in wall:
            if j - 1 == w % mult and i - 1 == w // mult:
                bb = True

        if (i - 1 < 0 or j - 1 < 0) or bb:
            return False
        else:
            return True

    elif a == 6:  # down-right

        bb = False
        for w in wall:
            if j + 1 == w % mult and i + 1 == w // mult:
                bb = True

        if (i + 1 > rows - 1 or j + 1 > col - 1) or bb:
            return False
        else:
            return True

    elif a == 7:  # down-left

        bb = False
        for w in wall:
            if j - 1 == w % mult and i + 1 == w // mult:
                bb = True

        if (i + 1 > rows - 1 or j - 1 < 0) or bb:
            return False
        else:
            return True

    else:  # do nothing
        return True
