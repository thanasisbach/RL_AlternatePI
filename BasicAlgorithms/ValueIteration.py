import numpy as np
import PolicyIteration as pi
import time


# This is the classic policy iteration algorithm
def ValueIteration(states, actions, reward, transition, gamma, numR, numC, grid, wall, goal, mult):
    # list, list, 3d matrix, 3d matrix, int

    nStates, nActions = pi.PossibleStates(states, actions, grid, numR, numC, wall, mult)

    policy = pi.initPolicy(states, nActions) # [0 for s in states]  # [0, 3, 1, 0, 3, 0, 0, 0, 1]  #

    Value = np.zeros(len(states))  # this is the value function
    # print("Initial policy", policy)

    valueChange = True
    iter = 0
    start_time = time.time()
    while valueChange:
        iter += 1
        valueChange = False

        for s in states:
            if grid[s] in wall:
                continue
                
            v_best = Value[s]  # we assume that the current value is the best and we improve it
            # cnt = 0
            for a in nActions[s]:

                v_a = 0
                # s1 = nStates[s][cnt]
                for s1 in nStates[s]:
                    v_a += transition[s][a][s1] * (reward[s][a][s1] + gamma * Value[s1])

                if v_a > v_best:
                    policy[s] = a
                    v_best = v_a
                    valueChange = True

                # cnt += 1
            Value[s] = v_best

    totTime = time.time() - start_time
    print("Run time in seconds: ", totTime)
    print("Num iters", iter)

    # pi.GraphThePolicy(policy, Value, numR, numC)
    return Value, policy, iter, totTime
