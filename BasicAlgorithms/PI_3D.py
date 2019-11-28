import numpy as np
import random
from PIL import Image
import time


# This is the classic policy iteration algorithm
def PolicyIteration3D(states, actions, reward, transition, gamma, numR, numC, numZ, grid, wall, goal, mult, mult2):

    nStates, nActions = PossibleStates(states, actions, grid, numR, numC, numZ, wall, mult, mult2)
    # print(nStates)
    policy = initPolicy(states, nActions)  # [0 for s in states]

    Value = np.zeros(len(states))  # this is the value function
    # print("Initial policy", policy)

    valueChange = True
    iter = 0
    pi_iter = 0

    start_time = time.time()
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

            # print(q_best, "optimal")
            for a in nActions[s]:  # maximize over actions
                pi_iter += 1
                # s1 = nStates[s][cnt]
                q_sa = 0
                for s1 in nStates[s]:
                    # print(transition[s][a][s1])
                    # print("action, ", a)
                    # print("state, ", s)
                    # print("next state ", s1)
                    # print(nStates[s])
                    # print(nActions[s])
                    q_sa += transition[s][a][s1] * (reward[s][a][s1] + gamma * Value[s1])

                if q_sa > q_best:
                    policy[s] = a
                    q_best = q_sa
                    valueChange = True

    totTime = time.time() - start_time
    # print("Run time in seconds: ", totTime)
    # print("Num iters", iter)
    # print("Policy Improvement iters", pi_iter)
    # GraphThePolicy(policy, Value, numR, numC)

    return Value, policy, iter, totTime


def AroundStates(states, actions, grid, rows, col, numZ, wall, mult, mult2):
    # this method find the around states of the current state
    ArStates = {}

    for s in states:
        ArStates[s] = []

        for a in actions:

            if validAction(s, a, grid, rows, col, numZ, wall, mult, mult2):

                if a == 0:  # right
                    ArStates[s].append(s + numZ)

                elif a == 1:  # left
                    ArStates[s].append(s - numZ)

                elif a == 2:  # up
                    ArStates[s].append(s - (numZ * col))

                elif a == 3:  # down
                    ArStates[s].append(s + (numZ * col))


                elif a == 4:  # up-right
                    ArStates[s].append(s - (numZ * col) + numZ)

                elif a == 5:  # up-left
                    ArStates[s].append(s - (numZ * col) - numZ)

                elif a == 6:  # down-right
                    ArStates[s].append(s + (numZ * col) + numZ)

                elif a == 7:  # down-left
                    ArStates[s].append(s + (numZ * col) - numZ)

                elif a == 8:  # down nothing
                    ArStates[s].append(s)

                elif a == 9:  # f
                    ArStates[s].append(s + 1)

                elif a == 10:  # b
                    ArStates[s].append(s - 1)

                elif a == 11:  # f right
                    ArStates[s].append(s + 1 + numZ)

                elif a == 12:  # f left
                    ArStates[s].append(s + 1 - numZ)

                elif a == 13:  # f up
                    ArStates[s].append(s + 1 - (numZ * col))

                elif a == 14:  # f down
                    ArStates[s].append(s + 1 + (numZ * col))

                elif a == 15:  # b right
                    ArStates[s].append(s - 1 + numZ)

                elif a == 16:  # b left
                    ArStates[s].append(s - 1 - numZ)

                elif a == 17:  # b up
                    ArStates[s].append(s - 1 - (numZ * col))

                elif a == 18:  # b down
                    ArStates[s].append(s - 1 + (numZ * col))

                elif a == 19:  # f up-right
                    ArStates[s].append(s + 1 - (numZ * col) + numZ)

                elif a == 20:  # f up-left
                    ArStates[s].append(s + 1 - (numZ * col) - numZ)

                elif a == 21:  # f down-right
                    ArStates[s].append(s + 1 + (numZ * col) + numZ)

                elif a == 22:  # f down-left
                    ArStates[s].append(s + 1 + (numZ * col) - numZ)

                elif a == 23:  # b up-right
                    ArStates[s].append(s - 1 - (numZ * col) + numZ)

                elif a == 24:  # b up-left
                    ArStates[s].append(s - 1 - (numZ * col) - numZ)

                elif a == 25:  # b down-right
                    ArStates[s].append(s - 1 + (numZ * col) + numZ)

                else:  # b down-left
                    ArStates[s].append(s - 1 + (numZ * col) - numZ)

    return ArStates


def PossibleStates(states, actions, grid, rows, col, numZ, wall, mult, mult2):
    arStates = AroundStates(states, actions, grid, rows, col, numZ, wall, mult, mult2)

    nStates = {}
    nActions = {}

    for s in states:

        if grid[s] in wall:
            continue
        nStates[s] = []
        nActions[s] = []
        for a in actions:
            if validAction(s, a, grid, rows, col, numZ, wall, mult, mult2):

                if a == 0:  # right
                    nStates[s] = arStates[s]  # .append(s + 1)
                    nActions[s].append(a)


                elif a == 1:  # left
                    nStates[s] = arStates[s]  # .append(s + 1)
                    nActions[s].append(a)

                elif a == 2:  # up
                    nStates[s] = arStates[s]  # .append(s + 1)
                    nActions[s].append(a)

                elif a == 3:  # down
                    nStates[s] = arStates[s]  # .append(s + 1)
                    nActions[s].append(a)

                elif a == 4:  # up-right
                    nStates[s] = arStates[s]  # .append(s + 1)
                    nActions[s].append(a)

                elif a == 5:  # up-left
                    nStates[s] = arStates[s]  # .append(s + 1)
                    nActions[s].append(a)

                elif a == 6:  # down-right
                    nStates[s] = arStates[s]  # .append(s + 1)
                    nActions[s].append(a)

                elif a == 7:  # down-left
                    nStates[s] = arStates[s]  # .append(s + 1)
                    nActions[s].append(a)

                elif a == 8:  # down nothing
                    nStates[s] = arStates[s]  # .append(s + 1)
                    nActions[s].append(a)

                elif a == 9:  # f
                    nStates[s] = arStates[s]  # .append(s + 1)
                    nActions[s].append(a)

                elif a == 10:  # b
                    nStates[s] = arStates[s]  # .append(s + 1)
                    nActions[s].append(a)

                elif a == 11:  # f right
                    nStates[s] = arStates[s]  # .append(s + 1)
                    nActions[s].append(a)

                elif a == 12:  # f left
                    nStates[s] = arStates[s]  # .append(s + 1)
                    nActions[s].append(a)

                elif a == 13:  # f up
                    nStates[s] = arStates[s]  # .append(s + 1)
                    nActions[s].append(a)

                elif a == 14:  # f down
                    nStates[s] = arStates[s]  # .append(s + 1)
                    nActions[s].append(a)

                elif a == 15:  # b right
                    nStates[s] = arStates[s]  # .append(s + 1)
                    nActions[s].append(a)

                elif a == 16:  # b left
                    nStates[s] = arStates[s]  # .append(s + 1)
                    nActions[s].append(a)

                elif a == 17:  # b up
                    nStates[s] = arStates[s]  # .append(s + 1)
                    nActions[s].append(a)

                elif a == 18:  # b down
                    nStates[s] = arStates[s]  # .append(s + 1)
                    nActions[s].append(a)

                elif a == 19:  # f up-right
                    nStates[s] = arStates[s]  # .append(s + 1)
                    nActions[s].append(a)

                elif a == 20:  # f up-left
                    nStates[s] = arStates[s]  # .append(s + 1)
                    nActions[s].append(a)

                elif a == 21:  # f down-right
                    nStates[s] = arStates[s] # .append(s + 1)
                    nActions[s].append(a)

                elif a == 22:  # f down-left
                    nStates[s] = arStates[s]  # .append(s + 1)
                    nActions[s].append(a)

                elif a == 23:  # b up-right
                    nStates[s] = arStates[s]  # .append(s + 1)
                    nActions[s].append(a)

                elif a == 24:  # b up-left
                    nStates[s] = arStates[s]  # .append(s + 1)
                    nActions[s].append(a)

                elif a == 25:  # b down-right
                    nStates[s] = arStates[s]  # .append(s + 1)
                    nActions[s].append(a)

                else:  # b down-left
                    nStates[s] = arStates[s]  # .append(s + 1)
                    nActions[s].append(a)

    return nStates, nActions


def GraphThePolicy(policy, value, row, col, numZ):
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
            else:  # none
                act = act + "None" + "||"

            val = val + str(value[cnt]) + "||"
            cnt += 1

        print("||" + act)
        print("||" + val)
        print("-----------------------")
        act = ""
        val = ""

    # policy = np.matrix(policy)
    # imgx = 512
    # imgy = 512
    # image = Image.new("RGB", (imgx, imgy))
    # pixels = image.load()
    # for ky in range(imgy):
    #     for kx in range(imgx):
    #         m = policy[row * ky // imgy][col * kx // imgx] * 255
    #         pixels[kx, ky] = (m, m, m)
    # image.save("RandomMaze_" + str(row) + "x" + str(col) + ".png", "PNG")


def initPolicy(states, posActs):
    policy = [None for s in states]

    for s in states:
        if s in posActs:
            r = random.randint(1, len(posActs[s]))
            policy[s] = posActs[s][r - 1]

    return policy


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
