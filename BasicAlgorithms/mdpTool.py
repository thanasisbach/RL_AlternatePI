import pandas as pd
import mdptoolbox
import mdptoolbox.example


def run():

    P, R = mdptoolbox.example.forest()
    vi = mdptoolbox.mdp.PolicyIteration(P, R, 0.9)
    vi.run()
    print(vi.policy,  vi.V)
    print(P, R)


if __name__ == "__main__":
    run()





def PolicyIteration(states, actions, reward, transition, gamma, numR, numC, grid, wall, goal):

    nStates = PossibleStates(states, actions, grid, numR, numC, wall)
    print(nStates)

    policy = [0 for s in states]  # [0, 3, 1, 0, 3, 0, 0, 0, 1, 0, 0, 0]  #

    Value = np.zeros(len(states))  # this is the value function
    print("Initial policy", policy)

    valueChange = True
    iter = 0

    while valueChange:
        valueChange = False
        iter += 1

        # Value = np.zeros(len(states))  # i have to initialize value function here again
        # Value iteration part
        for s in states:
            #if (grid[s] // 10 == goal // 10) and (grid[s] % 10 == goal % 10):
            #    continue

            if (grid[s] // 10 == wall // 10) and (grid[s] % 10 == wall % 10):
                continue

            Value[s] = 0  # i have to initialize value function here again
            if validAction(s, policy[s], grid, numR, numC, wall):
                for s1 in states:  # for loop for the next state
                    if (s1 // 10 != wall // 10 or s1 % 10 != wall % 10):  # (s != s1 and) i had this state before "do-nothing" action

                        Value[s] += transition[s, policy[s], s1] * (reward[s, policy[s], s1] + gamma * Value[s1])
                        # reward[s, policy[s], s1] + gamma * Value[s1] * transition[s, policy[s], s1]
                        # reward[s, policy[s], s1] + gamma * Value[s1] * transition[s, policy[s], s1]

        # policy evaluation part
        for s in states:
            #if (grid[s] // 10 == goal // 10) and (grid[s] % 10 == goal % 10):
            #    continue

            if (grid[s] // 10 == wall // 10) and (grid[s] % 10 == wall % 10):
                continue

            q_best = Value[s]  # we assume that the current value is the best and we improve it
            for a in actions:  # maximize over actions
                if validAction(s, a, grid, numR, numC, wall):
                    q_sa = 0
                    for s1 in states:  # for loop for the next state
                        if (s1 // 10 != wall // 10 or s1 % 10 != wall % 10):
                            q_sa += transition[s, a, s1] * (reward[s, a, s1] + gamma * Value[s1])
                            # reward[s, a, s1] + gamma * Value[s1] * transition[s, a, s1]
                            # transition[s, a, s1] * (reward[s, a, s1] + gamma * Value[s1])
                            # reward[s, a, s1] + gamma * Value[s1] * transition[s, a, s1]  #

                    # q_sa = sum([transition[s, a, s1] * (reward[s, a, s1] + gamma * Value[s1]) for s1 in states])
                    if q_sa > q_best:
                        # print("State 0", s, "Action:", a, "State1:", s, ": q_sa", q_sa, "q_best", q_best)
                        policy[s] = a
                        q_best = q_sa
                        valueChange = True

    print(Value, iter, policy)
    return policy, Value