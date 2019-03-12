import numpy as np
import random
import PolicyIteration as pi


def AlternatePI(states, stateC, stateR, actions, actC, actR, grid, gridStates, wall, goal, mult, transition, reward, gamma):

    nStates, nActions = PossibleStates(states, actions, grid, len(stateR), len(stateC), wall, mult)

    nColActions = PossibleColActions(stateC)
    nRowActions = PossibleRowActions(stateR)

    polC = initDecompPolicy(stateC, nColActions)
    polR = initDecompPolicy(stateR, nRowActions)
    polC = [8, 1, 8]
    polR = [3, 2, 2]

    Value = np.zeros(len(states))  # this is the value function
    print("Col Policy:", polC, "Row Policy:", polR)
    policy = CombinePolicy(polR, polC, gridStates, grid, wall, mult, nActions, nStates, transition, reward, Value, gamma)  # initPolicy(states, nActions)

    print("Global Policy:", policy)

    cValueR = False
    cValueC = False
    changeValue = True
    iter = 0
    while changeValue:
        # print(iter)
        changeValue = False
        iter += 1
        polR, Value, cValueR, policy = rowPI(policy, polC, polR, stateR, stateC, nStates, nActions, grid, gridStates, wall, transition, reward, gamma, mult, Value, nRowActions)

        # combine the policies after improving one dimension
        # policy = CombinePolicy(polR, polC, gridStates, grid, wall, mult, nActions, nStates, transition, reward, Value, gamma)

        polC, Value, cValueC, policy = colPI(policy, polC, polR, stateR, stateC, nStates, nActions, grid, gridStates, wall, transition, reward, gamma, mult, Value, nColActions)

        # combine the policies after improving one dimension
        # policy = CombinePolicy(polR, polC, gridStates, grid, wall, mult, nActions, nStates, transition, reward, Value, gamma)

        if cValueR or cValueC:
            changeValue = True

        # print("Individual policies are:")
        # print("Col Policy:", polC, "Row Policy:", polR)
        # print("The combined policy is:")
        # print(policy)
        # print("Value function:", Value)

    # policy = CombinePolicy(polR, polC, gridStates, grid, wall, mult, nActions, nStates, transition, reward, Value, gamma)
    print(iter)
    print("Row Policy:", polR)
    print("Col Policy:", polC)
    pi.GraphThePolicy(policy, Value, len(stateR), len(stateC))
    return policy, 0

def rowPI(policy, polC, polR, stateR, stateC, nStates, nActions, grid, gridStates, wall, transition, reward, gamma, mult, Value, nRowActions):

    #print(" oook")
    #print("olaaa", Value)
    changeValue = False
    # Value iteration part
    for sR in stateR:
        for sC in stateC:
            if sR * mult + sC == wall:
                continue

            s = gridStates[sR * mult + sC]
            V = 0
            for s1 in nStates[s]:  # for loop for the next state
                V += transition[s, policy[s], s1] * (reward[s, policy[s], s1] + gamma * Value[s1])

            Value[s] = V

    # policy evaluation part
    for sR in stateR:
        for sC in stateC:
            if sR * mult + sC == wall:
                continue

            s = gridStates[sR * mult + sC]
            q_best = Value[s]  # we assume that the current value is the best and we improve it

            for aR in nRowActions[sR]:  # maximize over actions
                aC = polC[sC]

                a = CombAction(aR, aC, sR, sC, grid, wall, mult, gridStates, len(stateR), len(stateC), nActions, nStates, transition, reward, Value, gamma)

                pp = [a]
                arr, acc = DecombinePolicy(pp)
                if arr != aR:
                    continue
                    aR = arr

                if a is None:
                    continue

                s1 = NextState(s, a, len(stateC))
                q_sa = transition[s, a, s1] * (reward[s, a, s1] + gamma * Value[s1])
                if q_sa > q_best:
                    polR[sR] = aR
                    policy[s] = a
                    q_best = q_sa
                    changeValue = True

    return polR, Value, changeValue, policy


def colPI(policy, polC, polR, stateR, stateC, nStates, nActions, grid, gridStates, wall, transition, reward, gamma, mult, Value, nColActions):

    changeValue = False
    # Value iteration part
    for sC in stateC:
        for sR in stateR:
            if sR * mult + sC == wall:
                continue

            s = gridStates[sR * mult + sC]
            V = 0
            for s1 in nStates[s]:  # for loop for the next state
                V += transition[s, policy[s], s1] * (reward[s, policy[s], s1] + gamma * Value[s1])

            Value[s] = V

    # policy evaluation part
    for sC in stateC:
        for sR in stateR:
            if sR * mult + sC == wall:
                continue

            s = gridStates[sR * mult + sC]
            q_best = Value[s]  # we assume that the current value is the best and we improve it

            for aC in nColActions[sC]:  # maximize over actions
                aR = polR[sR]

                a = CombAction(aR, aC, sR, sC, grid, wall, mult, gridStates, len(stateR), len(stateC), nActions, nStates, transition, reward, Value, gamma)

                pp = [a]
                arr, acc = DecombinePolicy(pp)
                if acc != aC:
                    continue
                    aC = acc

                if a is None:
                    continue

                s1 = NextState(s, a, len(stateC))
                q_sa = transition[s, a, s1] * (reward[s, a, s1] + gamma * Value[s1])
                if q_sa > q_best:
                    polC[sC] = aC
                    policy[s] = a
                    q_best = q_sa
                    changeValue = True


    return polC, Value, changeValue, policy


def PossibleColActions(states):


    nActions = {}

    for s in states:
        if s - 1 < 0:  # left
            nActions[s] = [0, 8]
        elif s + 1 > len(states) - 1:  # right
            nActions[s] = [1, 8]
        else:
            nActions[s] = [0, 1, 8]

    return nActions


def PossibleRowActions(states):

    nActions = {}

    for s in states:
        if s - 1 < 0:  # up
            nActions[s] = [3, 8]

        elif s + 1 > len(states) - 1:  # down
            nActions[s] = [2, 8]
        else:
            nActions[s] = [2, 3, 8]

    return nActions


def NextState(s, a, col):

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


def PossibleStates(states, actions, grid, rows, col, wall, mult):

    nStates = {}
    nActions = {}

    for s in states:

        if grid[s] == wall:
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


def initDecompPolicy(states, nActions):

    policy = []
    for s in states:
        policy.append(random.choice(nActions[s]))


    return policy


def initPolicy(states, posActs):

    policy = [None for s in states]

    for s in states:
        if s in posActs:
            r = random.randint(1, len(posActs[s]))
            policy[s] = posActs[s][r-1]

    return policy


def DecombinePolicy(policy):  # Information aliasing problem was detected here

    for i in policy:
        if i == 0:  # right
            aC = 0
            aR = 8

        elif i == 1:  # left
            aC = 1
            aR = 8

        elif i == 2:  # up
            aC = 8
            aR = 2

        elif i == 3:  # down
            aC = 8
            aR = 3

        elif i == 4:  # up-right
            aC = 0
            aR = 2

        elif i == 5:  # up-left
            aC = 1
            aR = 2

        elif i == 6:  # down-right
            aC = 0
            aR = 3

        elif i == 7:  # down-left
            aC = 1
            aR = 3

        elif i == 8:  # do-nothing action
            aC = 8
            aR = 8

    return aR, aC

def CombAction(aR, aC, sR, sC, grid, wall, mult, gridStates, lenR, lenC, nActions, nStates, transition, reward, Value, gamma):
    a = None

    if sR * mult + sC == wall:
        return None

    if aR == 2:  # up
        if aC == 0:  # right
            a = 4  # up-right

        elif aC == 1:  # left
            a = 5  # down-right

        else:
            a = aR  # right

    elif aR == 3:  # down
        if aC == 0:  # right
            a = 6  # up-left

        elif aC == 1:  # left
            a = 7  # down-left

        else:
            a = aR  # left

    else:  # do nothing
        a = aC  # whatever this action is

    # checking if the action is valid in the whole grid
    # case = True
    # while case:
    #     if validAction(gridStates[sR * mult + sC], a, grid, lenR, lenC, wall, mult):
    #         return a
    #         # case = False
    #         # print("nice", act[len(act) - 1])
    #
    #     else:
    #         # get a random action from the possible actions of the state
    #         # act[len(act) - 1] = random.choice(nActions[gridStates[cntR * mult + cntC]])
    #         return random.choice(nActions[gridStates[sR * mult + sC]])

    if validAction(gridStates[sR * mult + sC], a, grid, lenR, lenC, wall, mult):
        return a
        # print("nice", act[len(act) - 1])

    else:
        # return None
        # print(Value)
        s = gridStates[sR * mult + sC]
        # print("State:", s, "Action:", a)
        # if value is zero in all next states we cant improve
        val = []
        for sa in nActions[s]:
            s1 = NextState(s, sa, lenC)
            val.append(Value[s1])

        if len(set(val)) <= 1:
            return random.choice(nActions[gridStates[sR * mult + sC]])


        # policy improvement
        case = True
        while case:
            # instead of random action this helps value function convergence
            # policy evaluation part
            case = False

            q_best = Value[s]  # we assume that the current value is the best and we improve it
            for aa in nActions[s]:  # maximize over actions
                s1 = NextState(s, aa, lenC)  # nStates[s][cnt]  #
                q_sa = transition[s, aa, s1] * (reward[s, aa, s1] + gamma * Value[s1])
                if q_sa > q_best:
                    a = aa
                    # policy[s] = a
                    q_best = q_sa
                    case = True
                    Value[s] = q_best
        # print(a)
        return a

    # return a


def CombinePolicy(pa1, pa2, gridStates, grid, wall, mult, nActions, nStates, transition, reward, Value, gamma):  # columns action - rows action

    act = []

    cntR = 0
    for a1 in pa1:

        cntC = 0
        for a2 in pa2:

            if cntR * mult + cntC == wall:
                act.append(None)
                continue

            if a1 == 2:  # up
                if a2 == 0:  # right
                    act.append(4)

                elif a2 == 1:  # left
                    act.append(5)

                else:
                    act.append(a1)

            elif a1 == 3:  # down
                if a2 == 0:  # right
                    act.append(6)

                elif a2 == 1:  # left
                    act.append(7)

                else:
                    act.append(a1)

            else:  # do nothing
                if a2 == 8:
                    act.append(8)

                else:
                    act.append(a2)

            # checking if the action is valid in the whole grid
            if validAction(gridStates[cntR * mult + cntC], act[len(act) - 1], grid, len(pa1), len(pa2), wall, mult):
                pass
                # print("nice", act[len(act) - 1])

            else:
                # print(Value)
                s = gridStates[cntR * mult + cntC]
                # print("State:", s, "Action:", act[len(act) - 1])
                # if value is zero in all next states we cant improve
                bol = False
                val = []
                for sa in nActions[s]:
                    s1 = NextState(s, sa, len(pa2))
                    val.append(Value[s1])

                if len(set(val)) <= 1:
                    act[len(act) - 1] = random.choice(nActions[gridStates[cntR * mult + cntC]])
                    pass

                # policy improvement
                case = True
                while case:
                    case = False
                    # instead of random action this helps value function convergence
                    # policy improvement part


                    q_best = Value[s]  # we assume that the current value is the best and we improve it
                    for a in nActions[s]:  # maximize over actions
                        s1 = NextState(s, a, len(pa2))  # nStates[s][cnt]
                        q_sa = transition[s, a, s1] * (reward[s, a, s1] + gamma * Value[s1])
                        # print("q_best", q_best, "q_a", q_sa, "action", a)
                        if q_sa > q_best:
                            act[len(act) - 1] = a
                            # policy[s] = a
                            q_best = q_sa
                            Value[s] = q_best
                            case = True

                # print(act[len(act) - 1])

            cntC += 1


        cntR += 1

    return act


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