import numpy as np
import PolicyIteration as pi
import ValueIteration as vi
import AlternatePI as api
import MonteCarlo as mc
import MDP as m
import AltMDP as am
import PCA as pca
import OptPolicy as op
import MazeGenerator as mg
from sklearn.decomposition import PCA


def runMC():
    nRows = 15
    nCols = 15
    w = nRows * nCols
    wallP = [1 * w + 2]
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
    nRows = 5
    nCols = 5
    w = nRows * nCols
    wallP = [3 * w + 4]
    goalP = 4 * w + 4

    mdp = m.MDP(nRows, nRows, wallP, goalP)
    mdp.CreateGrid()
    mdp.TnR()
    mdp.InitRnT()

    v, p, it = vi.ValueIteration(mdp.states, mdp.actions, mdp.reward, mdp.transition, mdp.gamma, mdp.numRows, mdp.numCol,
                             mdp.grid, mdp.wall, mdp.goal, mdp.mult)

    # well this is the part where we will use pca with the "optimal" Value function that PI returns
    # pca.Pca(v, p, mdp.numRows, mdp.numCol)

    # op.optimalPolicy(mdp.states, mdp.actions, mdp.grid, mdp.goal, mdp.wall, p, mdp.numCol, mdp.numRows, mdp.mult)


def runPI(rows, cols, goal, wall):
    nRows = rows
    nCols = cols
    w = nRows * nCols
    # wallP = [(nRows - 2) * w + (nCols - 1), 0 * w + 2]
    # goalP = (nRows - 1) * w + (nCols - 1)

    goalP, wallP = goal, wall # mg.MazeGrid(nRows, nCols)

    mdp = m.MDP(nRows, nRows, wallP, goalP)
    mdp.CreateGrid()
    mdp.TnR()
    mdp.InitRnT()

    v, p, it = pi.PolicyIteration(mdp.states, mdp.actions, mdp.reward, mdp.transition, mdp.gamma, mdp.numRows, mdp.numCol,
                              mdp.grid, mdp.wall, mdp.goal, mdp.mult)

    # well this is the part where we will use pca with the "optimal" Value function that PI returns
    # pca.Pca(v, p, mdp.numRows, mdp.numCol)

    op.optimalPolicy(mdp.states, mdp.actions, mdp.grid, mdp.goal, mdp.wall, p, mdp.numCol, mdp.numRows, mdp.mult)


def runAltPI(rows, cols, goal, wall):
    nRows = rows
    nCols = cols
    w = nRows * nCols
    # wallP = [1 * w + 2, 0 * w, 2]
    # goalP = 2 * w + 2

    goalP, wallP = goal, wall # mg.MazeGrid(nRows, nCols)

    mdp = am.AltMDP(nRows, nRows, wallP, goalP)
    mdp.CreateGrid()
    mdp.TnR()
    mdp.InitRnT()

    p, v, it, altit = api.AlternatePI(mdp.states, mdp.statesA, mdp.statesB, mdp.actions, mdp.actionsA, mdp.actionsB, mdp.grid,
                           mdp.gridStates, mdp.wall, mdp.goal, mdp.mult, mdp.transition, mdp.reward, mdp.gamma)

    op.optimalPolicy(mdp.states, mdp.actions, mdp.grid, mdp.goal, mdp.wall, p, mdp.numA, mdp.numB, mdp.mult)

def main():
    rows = 3
    cols = 3
    goal = 2 * 9 + 2
    wall = [0 * 9 - 1]
    # goal, wall = mg.MazeGrid(rows, cols)

    runPI(rows, cols, goal, wall)
    # runVI()
    # runMC()
    runAltPI(rows, cols, goal, wall)


if __name__ == "__main__":
    main()
