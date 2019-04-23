import collections
import PolicyIteration as pi

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
        ww = grid[s]


        if ww in wall:
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
    # print(nei)

    dfs_list = {}
    policy_list = {}

    for s in states:

        dfs_list[s] = 0
        policy_list[s] = 0
        # print(grid[s], wall)
        if grid[s] in wall:
            continue

        if grid[s] == goal:
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
                dfs_list[s] = dep  # add number of steps from that state to the goal
                # print("From node: ", s, ",", dep, " steps to goal --- DFS")
                queue = None
                continue

            # print(nei)
            # print(vertex)
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
                policy_list[s] = cnt  # add number of steps from that state to the goal
                # print("From node: ", s, ",", cnt, " steps to goal --- policy")
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

    # the part we see if there are differences in optimal path and our policy path
    cc = 1
    for state in dfs_list:
        if dfs_list[state] != policy_list[state]:
            cc = 0
            print("not optimal path from state:", state)

    if cc:
        print("Optimal Policy!!!")