import stochasticdp as stc

def BackInd():

    r = 10
    c = 10
    numbStages = r * c
    states = []
    for i in range(numbStages):
        states.append(i)
    deci = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    dp = stc.StochasticDP(numbStages, states, deci, minimize=False)

    # reward table
    for i in range(numbStages):
        if i == numbStages - 1:
            dp.add_boundary(state=i, value=r*c)
        else:
            dp.add_boundary(state=i, value=0)
    # ger transition
    for t in range(numbStages):
        for s in states:
            for a in deci:
                if a == 0:  # right
                    if s + 1 in states:
                        dp.add_transition(stage=t, from_state=s, decision=a, to_state=s+1,
                                          probability=0.7, contribution=0)
                elif a == 1:  # left

                    if s - 1 in states:
                        dp.add_transition(stage=t, from_state=s, decision=a, to_state=s - 1,
                                          probability=0.7, contribution=0)
                elif a == 2:  # up
                    if s - c in states:
                        dp.add_transition(stage=t, from_state=s, decision=a, to_state=s - c,
                                          probability=0.7, contribution=0)
                elif a == 3:  # down
                    if s + c in states:
                        dp.add_transition(stage=t, from_state=s, decision=a, to_state=s + c,
                                          probability=0.7, contribution=0)
                elif a == 4:  # up-right
                    if s - c + 1 in states:
                        dp.add_transition(stage=t, from_state=s, decision=a, to_state=s - c + 1,
                                          probability=0.7, contribution=0)
                elif a == 5:  # up-left
                    if s - c - 1 in states:
                        dp.add_transition(stage=t, from_state=s, decision=a, to_state=s - c - 1,
                                          probability=0.7, contribution=0)
                elif a == 6:  # down-right
                    if s + c + 1 in states:
                        dp.add_transition(stage=t, from_state=s, decision=a, to_state=s + c + 1,
                                          probability=0.7, contribution=0)
                elif a == 7:  # down-left
                    if s + c - 1 in states:
                        dp.add_transition(stage=t, from_state=s, decision=a, to_state=s + c - 1,
                                          probability=0.7, contribution=0)
                else:  # do-nothing action
                    dp.add_transition(stage=t, from_state=s, decision=a, to_state=s,
                                      probability=0, contribution=0)

    val, pol = dp.solve()
    print(val)


if __name__ == "__main__":
    BackInd()
