import numpy as np
import random
import PolicyIteration as pi
import time
import PI_3D as pi3

from numpy.core.multiarray import ndarray


def AlternatePI3D(states, stateC, stateR, stateZ, actions, actC, actR, actZ, grid, gridStates, wall, goal, mult, mult2,
                transition, reward, gamma):

    nStates, nActions = pi3.PossibleStates(states, actions, grid, len(stateR), len(stateC), len(stateZ), wall, mult,
                                           mult2)

    nColActions = PossibleColActions(stateC)
    nRowActions = PossibleRowActions(stateR)
    nZActions = PossibleZActions(stateR)

    polC = initDecompPolicy(stateC, nColActions, goal, mult, mult2)
    polR = initDecompPolicy(stateR, nRowActions, goal, mult, mult2)
    polZ = initDecompPolicy(stateZ, nZActions, goal, mult, mult2)

    Value = np.zeros(len(states))  # this is the value function

    policyy, it = CombinePolicy(polR, polC, polZ, gridStates, grid, wall, mult, mult2, nActions, nStates, transition,
                                reward, Value, gamma)  # initPolicy(states, nActions)

    policy = np.array(policyy)
    # print("Global Policy:", policy)

    cValueR = False
    cValueC = False
    e = 1  # something small
    changeValue = True
    iter = 0
    extraIter = it
    start_time = time.time()
    prevValue = np.copy(Value)
    prevPolicy = np.copy(policy)

    while changeValue:
        # print(iter)
        changeValue = False
        iter += 1
        polR, Value, cValueR, policy, it, polC, polZ = rowPI(policy, polC, polR, polZ, stateR, stateC, stateZ, nStates,
                                                             nActions, grid, gridStates, wall, transition, reward,
                                                             gamma,
                                                             mult, mult2, Value, nRowActions)
        extraIter += it

        if np.max(Value - prevValue) < e:  # and np.array_equal(policy, prevPolicy):
            # print("lol")
            break

        prevValue = np.copy(Value)
        prevPolicy = np.copy(policy)

        polC, Value, cValueC, policy, it, polR, polZ = colPI(policy, polC, polR, polZ, stateR, stateC, stateZ, nStates,
                                                             nActions, grid, gridStates, wall, transition, reward,
                                                             gamma,
                                                             mult, mult2, Value, nRowActions)

        polZ, Value, cValueZ, policy, it, polR, polC = ZPI(policy, polC, polR, polZ, stateR, stateC, stateZ, nStates,
                                                           nActions, grid, gridStates, wall, transition, reward, gamma,
                                                           mult, mult2, Value, nRowActions)

        extraIter += it

        if np.max(Value - prevValue) < e:  # and np.array_equal(policy, prevPolicy):
            break

        prevValue = np.copy(Value)
        prevPolicy = np.copy(policy)

        if cValueR or cValueC or cValueZ:
            # print(iter, cValueC, cValueR)
            changeValue = True

        if iter > (len(stateC) * len(stateR) * len(stateZ)):
            print("it doesnt conv")
            changeValue = False

    totTime = time.time() - start_time
    # print("AAA Run time in seconds: ", totTime)
    # print("AAA PI iter:", iter)
    # print("Policy Imp:", extraIter)

    # print("Row Policy:", polR)
    # print("Col Policy:", polC)
    # pi.GraphThePolicy(policy, Value, len(stateR), len(stateC))
    # print("col policy: ", polC)
    # print("row policyL, ", polR)
    return policy, Value, iter, extraIter, totTime


def rowPI(policy, polC, polR, polZ, stateR, stateC, stateZ, nStates, nActions, grid, gridStates, wall, transition,
          reward, gamma, mult, mult2, Value, nRowActions):
    changeValue = False
    exIt = 0
    # Value iteration part
    for sR in stateR:
        for sC in stateC:
            for sZ in stateZ:
                if sR * mult2 + sC * mult + sZ in wall:
                    continue

                s = gridStates[sR * mult2 + sC * mult + sZ]
                V = 0
                # print(nStates[s], "NExt states from state: ", s)
                # print(transition[s][policy[s]])
                for s1 in nStates[s]:  # for loop for the next state
                    # print(s, policy[s], s1)
                    # print(policy)
                    if (s1 in transition[s][policy[s]]) and (s1 in reward[s][policy[s]]):
                        V += transition[s][policy[s]][s1] * (reward[s][policy[s]][s1] + gamma * Value[s1])

                Value[s] = V

    # policy evaluation part
    for sR in stateR:
        for sC in stateC:
            for sZ in stateZ:
                if sR * mult2 + sC * mult + sZ in wall:
                    continue

                s = gridStates[sR * mult2 + sC * mult + sZ]
                q_best = Value[s]  # we assume that the current value is the best and we improve it

                for aR in nRowActions[sR]:  # maximize over actions
                    # new part
                    exIt += 1
                    aC = polC[sC]
                    aZ = polZ[sZ]

                    a, et = CombAction(aR, aC, aZ, sR, sC, sZ, grid, wall, mult, mult2, gridStates, len(stateR), len(stateC),
                                       len(stateZ), nActions, nStates, transition, reward, Value, gamma)
                    exIt += et
                    pp = [a]
                    arr, acc, azz = DecombinePolicy(pp)

                    if a is None:
                        # print("errrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrorrrrrrrrrrrrrrrrr")
                        continue

                    q_sa = 0
                    for s1 in nStates[s]:
                        q_sa += transition[s][a][s1] * (reward[s][a][s1] + gamma * Value[s1])

                    if q_sa > q_best:
                        polR[sR] = aR
                        # polC[sC] = acc  # sooo new
                        policy[s] = a
                        # print(q_sa, q_best)
                        q_best = q_sa
                        # print(policy[s])
                        changeValue = True

    return polR, Value, changeValue, policy, exIt, polC, polZ


def colPI(policy, polC, polR, polZ, stateR, stateC, stateZ, nStates, nActions, grid, gridStates, wall, transition,
          reward, gamma, mult, mult2, Value, nColActions):
    changeValue = False
    exIt = 0
    # Value iteration part
    for sC in stateC:
        for sR in stateR:
            for sZ in stateZ:
                if sR * mult2 + sC * mult + sZ in wall:
                    continue

                s = gridStates[sR * mult2 + sC * mult + sZ]
                V = 0
                for s1 in nStates[s]:  # for loop for the next state
                    if (s1 in transition[s][policy[s]]) and (s1 in reward[s][policy[s]]):
                        V += transition[s][policy[s]][s1] * (reward[s][policy[s]][s1] + gamma * Value[s1])

                Value[s] = V

    # policy evaluation part
    for sC in stateC:
        for sR in stateR:
            for sZ in stateZ:
                if sR * mult2 + sC * mult + sZ in wall:
                    continue

                s = gridStates[sR * mult2 + sC * mult + sZ]
                q_best = Value[s]  # we assume that the current value is the best and we improve it

                for aC in nColActions[sC]:  # maximize over actions
                    # new part
                    exIt += 1
                    aR = polR[sR]
                    aZ = polZ[sZ]

                    a, et = CombAction(aR, aC, aZ, sR, sC, sZ, grid, wall, mult, mult2, gridStates, len(stateR), len(stateC),
                                       len(stateZ), nActions, nStates, transition, reward, Value, gamma)
                    exIt += et
                    pp = [a]
                    arr, acc, azz = DecombinePolicy(pp)

                    if a is None:
                        # print("errrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrorrrrrrrrrrrrrrrrr")
                        continue

                    q_sa = 0
                    for s1 in nStates[s]:
                        q_sa += transition[s][a][s1] * (reward[s][a][s1] + gamma * Value[s1])

                    if q_sa > q_best:
                        polC[sC] = aC
                        # polR[sR] = arr  # sooo new
                        policy[s] = a
                        # print(q_sa, q_best, " column")
                        q_best = q_sa
                        changeValue = True

    return polC, Value, changeValue, policy, exIt, polR, polZ


def ZPI(policy, polC, polR, polZ, stateR, stateC, stateZ, nStates, nActions, grid, gridStates, wall, transition,
        reward, gamma, mult, mult2, Value, nZActions):
    changeValue = False
    exIt = 0
    # Value iteration part
    for sZ in stateZ:
        for sR in stateR:
            for sC in stateC:
                if sR * mult2 + sC * mult + sZ in wall:
                    continue

                s = gridStates[sR * mult2 + sC * mult + sZ]
                V = 0
                for s1 in nStates[s]:  # for loop for the next state
                    if (s1 in transition[s][policy[s]]) and (s1 in reward[s][policy[s]]):
                        V += transition[s][policy[s]][s1] * (reward[s][policy[s]][s1] + gamma * Value[s1])

                Value[s] = V

    # policy evaluation part
    for sZ in stateZ:
        for sR in stateR:
            for sC in stateC:
                if sR * mult2 + sC * mult + sZ in wall:
                    continue

                s = gridStates[sR * mult2 + sC * mult + sZ]
                q_best = Value[s]  # we assume that the current value is the best and we improve it

                for aZ in nZActions[sZ]:  # maximize over actions
                    # new part
                    exIt += 1
                    aR = polR[sR]
                    aC = polZ[sC]

                    a, et = CombAction(aR, aC, aZ, sR, sC, sZ, grid, wall, mult, mult2, gridStates, len(stateR), len(stateC),
                                       len(stateZ), nActions, nStates, transition, reward, Value, gamma)
                    exIt += et
                    pp = [a]
                    arr, acc, azz = DecombinePolicy(pp)

                    if a is None:
                        # print("errrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrorrrrrrrrrrrrrrrrr")
                        continue

                    q_sa = 0
                    for s1 in nStates[s]:
                        q_sa += transition[s][a][s1] * (reward[s][a][s1] + gamma * Value[s1])

                    if q_sa > q_best:
                        polZ[sZ] = aZ
                        # polR[sR] = arr  # sooo new
                        policy[s] = a
                        # print(q_sa, q_best, " column")
                        q_best = q_sa
                        changeValue = True

    return polZ, Value, changeValue, policy, exIt, polR, polC


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


def PossibleZActions(states):
    nActions = {}

    for s in states:
        if s - 1 < 0:  # b
            nActions[s] = [9, 8]

        elif s + 1 > len(states) - 1:  # f
            nActions[s] = [10, 8]
        else:
            nActions[s] = [9, 10, 8]

    return nActions


def NextState(s, a, col, numZ):
    if a == 0:  # right

        return s + numZ

    elif a == 1:  # left

        return s - numZ

    elif a == 2:  # up

        return s - (numZ * col)

    elif a == 3:  # down
        return s + (numZ * col)

    elif a == 4:  # up-right
        return s - (numZ * col) + numZ

    elif a == 5:  # up-left
        return s - (numZ * col) - numZ

    elif a == 6:  # down-right
        return s + (numZ * col) + numZ

    elif a == 7:  # down-left

        return s + (numZ * col) - numZ

    elif a == 8:  # down nothing
        return s

    elif a == 9:
        return s + 1

    elif a == 10:  # b
        return s - 1

    elif a == 11:  # f right
        return s + 1 + numZ

    elif a == 12:  # f left
        return s + 1 - numZ

    elif a == 13:  # f up
        return s + 1 - (numZ * col)

    elif a == 14:  # f down
        return s + 1 + (numZ * col)

    elif a == 15:  # b right
        return s - 1 + numZ

    elif a == 16:  # b left
        return s - 1 - numZ

    elif a == 17:  # b up
        return s - 1 - (numZ * col)

    elif a == 18:  # b down
        return s - 1 + (numZ * col)

    elif a == 19:  # f up-right
        return s + 1 - (numZ * col) + numZ

    elif a == 20:  # f up-left
        return s + 1 - (numZ * col) - numZ

    elif a == 21:  # f down-right
        return s + 1 + (numZ * col) + numZ

    elif a == 22:  # f down-left
        return s + 1 + (numZ * col) - numZ

    elif a == 23:  # b up-right
        return s - 1 - (numZ * col) + numZ

    elif a == 24:  # b up-left
        return s - 1 - (numZ * col) - numZ

    elif a == 25:  # b down-right
        return s - 1 + (numZ * col) + numZ


    else:  # b down-left
        return s - 1 + (numZ * col) - numZ


def initDecompPolicy(states, nActions, goal, mult, mult2):
    policy = []
    for s in states:
        if s == (goal % mult2) % mult:
            policy.append(8)
        else:
            policy.append(random.choice(nActions[s]))

    return policy


def initPolicy(states, posActs):
    policy = [None for s in states]

    for s in states:
        if s in posActs:
            r = random.randint(1, len(posActs[s]))
            policy[s] = posActs[s][r - 1]

    return policy


def DecombinePolicy(policy):  # Information aliasing problem was detected here

    for a in policy:

        if a == 0:  # right
            aR = 8
            aC = 0
            aZ = 8

        elif a == 1:  # left
            aR = 8
            aC = 1
            aZ = 8

        elif a == 2:  # up
            aR = 2
            aC = 8
            aZ = 8

        elif a == 3:  # down
            aR = 3
            aC = 8
            aZ = 8

        elif a == 4:  # up-right
            aR = 2
            aC = 0
            aZ = 8

        elif a == 5:  # up-left
            aR = 2
            aC = 1
            aZ = 8

        elif a == 6:  # down-right
            aR = 3
            aC = 0
            aZ = 8

        elif a == 7:  # down-left
            aR = 3
            aC = 1
            aZ = 8

        elif a == 8:  # down nothing
            aR = 8
            aC = 8
            aZ = 8

        elif a == 9:  # f
            aR = 8
            aC = 8
            aZ = 9

        elif a == 10:  # b
            aR = 8
            aC = 8
            aZ = 10

        elif a == 11:  # f right
            aR = 8
            aC = 0
            aZ = 9

        elif a == 12:  # f left
            aR = 8
            aC = 1
            aZ = 9

        elif a == 13:  # f up
            aR = 2
            aC = 8
            aZ = 9

        elif a == 14:  # f down
            aR = 3
            aC = 8
            aZ = 9

        elif a == 15:  # b right
            aR = 8
            aC = 0
            aZ = 10

        elif a == 16:  # b left
            aR = 8
            aC = 1
            aZ = 10

        elif a == 17:  # b up
            aR = 2
            aC = 8
            aZ = 10

        elif a == 18:  # b down
            aR = 3
            aC = 8
            aZ = 10

        elif a == 19:  # f up-right
            aR = 2
            aC = 0
            aZ = 9

        elif a == 20:  # f up-left
            aR = 2
            aC = 1
            aZ = 9

        elif a == 21:  # f down-right
            aR = 3
            aC = 0
            aZ = 9

        elif a == 22:  # f down-left
            aR = 3
            aC = 1
            aZ = 9

        elif a == 23:  # b up-right
            aR = 2
            aC = 0
            aZ = 10

        elif a == 24:  # b up-left
            aR = 2
            aC = 1
            aZ = 10

        elif a == 25:  # b down-right
            aR = 3
            aC = 0
            aZ = 10

        else:  # b down-left
            aR = 3
            aC = 1
            aZ = 10

    return aR, aC, aZ


def CombAction(aR, aC, aZ, sR, sC, sZ, grid, wall, mult, mult2, gridStates, lenR, lenC, lenZ, nActions, nStates, transition,
               reward, Value, gamma):
    a = None

    if sR * mult2 + sC * mult + sZ in wall:
        return None

    if aR == 2:  # up
        if aC == 0:  # right
            if aZ == 9:
                a = 19 # fur
            elif aZ == 10:
                a = 23 # bur
            else:
                a = 4  # up-right

        elif aC == 1:  # left
            if aZ == 9: # ful
                a = 20
            elif aZ == 10: # bul
                a = 24
            else:
                a = 5  # ul

        else:
            if aZ == 9: # fu
                a = 13
            elif aZ == 10: # bu
                a = 17
            else:
                a = aR  # up

    elif aR == 3:  # down
        if aC == 0:  # right
            if aZ == 9: # fdr
                a = 21
            elif aZ == 10: # bdr
                a = 25
            else:
                a = 6  # dr

        elif aC == 1:  # left
            if aZ == 9: # fdl
                a = 22
            elif aZ == 10: # bdl
                a = 26
            else:
                a = 7  # down-left

        else:
            if aZ == 9: # fd
                a = 14
            elif aZ == 10: # bd
                a = 18
            else:
                a = aR  # left

    else:  # do nothing
            a = aZ  # whatever this action is

    if pi3.validAction(gridStates[sR * mult2 + sC * mult + sZ], a, grid, lenR, lenC, lenZ, wall, mult, mult2):
        return a, 0
        # print("nice", act[len(act) - 1])

    else:

        # return None
        # print(Value)
        s = gridStates[sR * mult2 + sC * mult + sZ]
        a = random.choice(nActions[gridStates[sR * mult2 + sC * mult + sZ]])

        # policy improvement
        case = True
        while case:
            # instead of random action this helps value function convergence
            # policy evaluation part
            case = False

            q_best = Value[s]  # we assume that the current value is the best and we improve it
            for aa in nActions[s]:  # maximize over actions

                # s1 = NextState(s, aa, lenC)  # nStates[s][cnt]  #
                q_sa = 0
                for s1 in nStates[s]:
                    # print(s, transition[s][aa], s1)
                    q_sa += transition[s][aa][s1] * (reward[s][aa][s1] + gamma * Value[s1])

                if q_sa > q_best:
                    a = aa
                    # policy[s] = a
                    q_best = q_sa
                    case = True
                    Value[s] = q_best
        # print(a)
        return a, 1

    # return a


def CombinePolicy(pa1, pa2, pa3, gridStates, grid, wall, mult, mult2, nActions, nStates, transition, reward, Value,
                  gamma):  # columns action - rows action

    act = []
    ct = 0

    cntR = 0
    for a1 in pa1:

        cntC = 0
        for a2 in pa2:

            cntZ = 0
            for a3 in pa3:

                if cntR * mult2 + cntC * mult + cntZ in wall:
                    act.append(None)
                    cntZ += 1
                    continue

                if a1 == 2:  # up
                    if a2 == 0:  # right
                        if a3 == 9:
                            act.append(19)
                            # a = 19  # fur
                        elif a3 == 10:
                            act.append(23)
                            # a = 23  # bur
                        else:
                            act.append(4)
                            # a = 4  # up-right

                    elif a2 == 1:  # left
                        if a3 == 9:  # ful
                            act.append(20)
                            # a = 20
                        elif a3 == 10:  # bul
                            act.append(24)
                            # a = 24
                        else:
                            act.append(5)
                            # a = 5  # ul

                    else:
                        if a3 == 9:  # fu
                            act.append(13)
                            # a = 13
                        elif a3 == 10:  # bu
                            act.append(17)
                            # a = 17
                        else:
                            act.append(a1)
                            # a = a1  # up

                elif a1 == 3:  # down
                    if a2 == 0:  # right
                        if a3 == 9:  # fdr
                            act.append(21)
                            # a = 21
                        elif a3 == 10:  # bdr
                            act.append(25)
                            # a = 25
                        else:
                            act.append(6)
                            # a = 6  # dr

                    elif a2 == 1:  # left
                        if a3 == 9:  # fdl
                            act.append(22)
                            # a = 22
                        elif a3 == 10:  # bdl
                            act.append(26)
                            # a = 26
                        else:
                            act.append(7)
                            # a = 7  # down-left

                    else:
                        if a3 == 9:  # fd
                            act.append(14)
                            # a = 14
                        elif a3 == 10:  # bd
                            act.append(18)
                            # a = 18
                        else:
                            act.append(a1)
                            # a = a1  # left

                else:  # do nothing
                    act.append(a3)
                    # a = a3  # whatever this action is

            # checking if the action is valid in the whole grid
                if pi3.validAction(gridStates[cntR * mult2 + cntC * mult + cntZ], act[len(act) - 1], grid, len(pa1), len(pa2), len(pa3), wall, mult, mult2):
                    pass
                    # print("nice", act[len(act) - 1])

                else:

                    # print(Value)
                    s = gridStates[cntR * mult2 + cntC * mult + cntZ]
                    act[len(act) - 1] = random.choice(nActions[gridStates[cntR * mult2 + cntC * mult + cntZ]])
                    # print("State:", s, "Action:", act[len(act) - 1])
                    # if value is zero in all next states we cant improve

                    ############################## Random case for the actions
                    # bol = False
                    # val = []
                    # for sa in nActions[s]:
                    #     s1 = NextState(s, sa, len(pa2))
                    #     val.append(Value[s1])
                    #
                    # if len(set(val)) <= 1:
                    #     act[len(act) - 1] = random.choice(nActions[gridStates[cntR * mult + cntC]])
                    #     pass
                    ############################################################

                    # policy improvement
                    ct += 1
                    case = True
                    while case:
                        case = False
                        # instead of random action this helps value function convergence
                        # policy improvement part

                        q_best = Value[s]  # we assume that the current value is the best and we improve it
                        for a in nActions[s]:  # maximize over actions

                            # s1 = NextState(s, a, len(pa2))  # nStates[s][cnt]
                            q_sa = 0
                            for s1 in nStates[s]:
                                q_sa += transition[s][a][s1] * (reward[s][a][s1] + gamma * Value[s1])
                            # print("q_best", q_best, "q_a", q_sa, "action", a)
                            if q_sa > q_best:
                                act[len(act) - 1] = a
                                # policy[s] = a
                                q_best = q_sa
                                Value[s] = q_best
                                case = True

                # print(act[len(act) - 1])

                cntZ += 1

            cntC += 1

        cntR += 1

    return act, ct


def validAction(s, a, grid, rows, col, numZ, wall, mult, mult2):
    # 0-right 1-left 2-up 3-down
    i = grid[s] // mult2
    j = (grid[s] % mult2) // mult
    z = (grid[s] % mult2) % mult
    ww = grid[s]
    wa = np.array(wall)

    if ww in wall:
        return False

    if a == 0:  # right

        if j + 1 > col - 1:
            return False

        ss = s + numZ
        if grid[ss] in wall:
            return False

        return True


    elif a == 1:  # left

        if j - 1 < 0:
            return False

        ss = s - numZ
        if grid[ss] in wall:
            return False

        return True


    elif a == 2:  # up

        if i - 1 < 0:
            return False

        ss = s - (numZ * col)
        if grid[ss] in wall:
            return False

        return True


    elif a == 3:  # down

        if i + 1 > rows - 1:
            return False

        ss = s + (numZ * col)
        if grid[ss] in wall:
            return False

        return True

    elif a == 4:  # up-right

        if j + 1 > col - 1 or i - 1 < 0:
            return False

        ss = s - (numZ * col) + numZ
        if grid[ss] in wall:
            return False

        return True


    elif a == 5:  # up-left

        if j - 1 < 0 or i - 1 < 0:
            return False

        ss = s - (numZ * col) - numZ
        if grid[ss] in wall:
            return False

        return True


    elif a == 6:  # down-right

        if j + 1 > col - 1 or i + 1 > rows - 1:
            return False

        ss = s + (numZ * col) + numZ
        if grid[ss] in wall:
            return False

        return True


    elif a == 7:  # down-left

        if j - 1 < 0 or i + 1 > rows - 1:
            return False

        ss = s + (numZ * col) - numZ
        if grid[ss] in wall:
            return False

        return True


    elif a == 8:  # down nothing
        return True

    elif a == 9:  # f

        if z + 1 > numZ - 1:
            return False

        ss = s + 1
        if grid[ss] in wall:
            return False

        return True


    elif a == 10:  # b

        if z - 1 < 0:
            return False

        ss = s - 1
        if grid[ss] in wall:
            return False

        return True

    elif a == 11:  # f right

        if j + 1 > col - 1 or z + 1 > numZ - 1:
            return False

        ss = s + 1 + numZ
        if grid[ss] in wall:
            return False

        return True


    elif a == 12:  # f left

        if j - 1 < 0 or z + 1 > numZ - 1:
            return False

        ss = s + 1 - numZ
        if grid[ss] in wall:
            return False

        return True


    elif a == 13:  # f up

        if i - 1 < 0 or z + 1 > numZ - 1:
            return False

        ss = s + 1 - (numZ * col)
        if grid[ss] in wall:
            return False

        return True


    elif a == 14:  # f down

        if i + 1 > rows - 1 or z + 1 > numZ - 1:
            return False

        bb = False
        ss = s + 1 + (numZ * col)
        if grid[ss] in wall:
            return False

        return True


    elif a == 15:  # b right

        if j + 1 > col - 1 or z - 1 < 0:
            return False

        ss = s - 1 + numZ
        if grid[ss] in wall:
            return False

        return True


    elif a == 16:  # b left

        if j - 1 < 0 or z - 1 < 0:
            return False

        ss = s - 1 - numZ
        if grid[ss] in wall:
            return False

        return True


    elif a == 17:  # b up

        if i - 1 < 0 or z - 1 < 0:
            return False

        bb = False
        ss = s - 1 - (numZ * col)
        if grid[ss] in wall:
            return False

        return True


    elif a == 18:  # b down

        if i + 1 > rows - 1 or z - 1 < 0:
            return False

        ss = s - 1 + (numZ * col)
        if grid[ss] in wall:
            return False

        return True


    elif a == 19:  # f up-right

        if i - 1 < 0 or j + 1 > col - 1 or z + 1 > numZ - 1:
            return False

        ss = s + 1 - (numZ * col) + numZ
        if grid[ss] in wall:
            return False

        return True


    elif a == 20:  # f up-left

        if i - 1 < 0 or j - 1 < 0 or z + 1 > numZ - 1:
            return False

        ss = s + 1 - (numZ * col) - numZ
        if grid[ss] in wall:
            return False

        return True


    elif a == 21:  # f down-right

        if i + 1 > rows - 1 or j + 1 > col - 1 or z + 1 > numZ - 1:
            return False

        ss = s + 1 + (numZ * col) + numZ
        if grid[ss] in wall:
            return False

        return True


    elif a == 22:  # f down-left

        if i + 1 > rows - 1 or j - 1 < 0 or z + 1 > numZ - 1:
            return False

        ss = s + 1 + (numZ * col) - numZ
        if grid[ss] in wall:
            return False

        return True


    elif a == 23:  # b up-right

        if i - 1 < 0 or j + 1 > col - 1 or z - 1 < 0:
            return False

        ss = s - 1 - (numZ * col) + numZ
        if grid[ss] in wall:
            return False

        return True


    elif a == 24:  # b up-left

        if i - 1 < 0 or j - 1 < 0 or z - 1 < 0:
            return False

        ss = s - 1 - (numZ * col) - numZ
        if grid[ss] in wall:
            return False

        return True


    elif a == 25:  # b down-right

        if i + 1 > rows - 1 or j + 1 > col - 1 or z - 1 < 0:
            return False

        ss = s - 1 + (numZ * col) + numZ
        if grid[ss] in wall:
            return False

        return True


    else:  # b down-left

        if i + 1 > rows - 1 or j - 1 < 0 or z - 1 < 0:
            return False

        ss = s - 1 + (numZ * col) - numZ
        if grid[ss] in wall:
            return False

        return True
