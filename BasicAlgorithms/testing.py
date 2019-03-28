import numpy as np
import PolicyIteration as pi
import ValueIteration as vi
import AlternatePI as api
import MonteCarlo as mc
import MDP as m
import AltMDP as am
import PCA as pca
import OptPolicy as op
from sklearn.decomposition import PCA


def runMC():
    nRows = 15
    nCols = 15
    w = nRows * nCols
    wallP = 1 * w + 2
    goalP = 2 * w + 2

    mdp = m.MDP(nRows, nRows, wallP, goalP)
    mdp.CreateGrid()
    mdp.TnR()
    mdp.InitRnT()

    v, p = mc.FirstVisitMC(mdp.states, mdp.actions, mdp.reward, mdp.transition, mdp.gamma, mdp.numRows, mdp.numCol,
                           mdp.grid, mdp.wall, mdp.goal, mdp.mult)

    # well this is the part where we will use pca with the "optimal" Value function that PI returns
    # pca.Pca(v, p, mdp.numRows, mdp.numCol)

    # op.optimalPolicy(mdp.states, mdp.actions, mdp.grid, mdp.goal, mdp.wall, p, mdp.numCol, mdp.numRows, mdp.mult)


def runVI():
    nRows = 50
    nCols = 50
    w = nRows * nCols
    wallP = 25 * w + 25
    goalP = 49 * w + 49

    mdp = m.MDP(nRows, nRows, wallP, goalP)
    mdp.CreateGrid()
    mdp.TnR()
    mdp.InitRnT()

    v, p = vi.ValueIteration(mdp.states, mdp.actions, mdp.reward, mdp.transition, mdp.gamma, mdp.numRows, mdp.numCol,
                             mdp.grid, mdp.wall, mdp.goal, mdp.mult)

    # well this is the part where we will use pca with the "optimal" Value function that PI returns
    # pca.Pca(v, p, mdp.numRows, mdp.numCol)

    op.optimalPolicy(mdp.states, mdp.actions, mdp.grid, mdp.goal, mdp.wall, p, mdp.numCol, mdp.numRows, mdp.mult)


def runPI():
    nRows = 250
    nCols = 250
    w = nRows * nCols
    wallP = 23 * w - 24
    goalP = 12 * w + 12

    mdp = m.MDP(nRows, nRows, wallP, goalP)
    mdp.CreateGrid()
    mdp.TnR()
    mdp.InitRnT()

    v, p = pi.PolicyIteration(mdp.states, mdp.actions, mdp.reward, mdp.transition, mdp.gamma, mdp.numRows, mdp.numCol,
                              mdp.grid, mdp.wall, mdp.goal, mdp.mult)

    # well this is the part where we will use pca with the "optimal" Value function that PI returns
    # pca.Pca(v, p, mdp.numRows, mdp.numCol)

    op.optimalPolicy(mdp.states, mdp.actions, mdp.grid, mdp.goal, mdp.wall, p, mdp.numCol, mdp.numRows, mdp.mult)


def runAltPI():
    nRows = 30
    nCols = 30
    w = nRows * nCols
    wallP = 1 * w + 2
    goalP = 2 * w + 2

    mdp = am.AltMDP(nRows, nRows, wallP, goalP)
    mdp.CreateGrid()
    mdp.TnR()
    mdp.InitRnT()

    p, v = api.AlternatePI(mdp.states, mdp.statesA, mdp.statesB, mdp.actions, mdp.actionsA, mdp.actionsB, mdp.grid,
                           mdp.gridStates, mdp.wall, mdp.goal, mdp.mult, mdp.transition, mdp.reward, mdp.gamma)

    op.optimalPolicy(mdp.states, mdp.actions, mdp.grid, mdp.goal, mdp.wall, p, mdp.numA, mdp.numB, mdp.mult)

def main():
    runPI()
    # runVI()
    # runMC()
    # runAltPI()


if __name__ == "__main__":
    main()
