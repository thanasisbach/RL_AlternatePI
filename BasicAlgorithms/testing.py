import numpy as np
import PolicyIteration as pi
import ValueIteration as vi
import AlternatePI as api
import MDP as m
import AltMDP as am


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


    pi.PolicyIteration(mdp.states, mdp.actions, mdp.reward, mdp.transition, mdp.gamma, mdp.numRows, mdp.numCol,
                              mdp.grid, mdp.wall, mdp.goal, mdp.mult)


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
    runVI()

if __name__ == "__main__":
    main()
