import numpy as np
import PolicyIteration as pi
import ValueIteration as vi
import AlternatePI as api
import MonteCarlo as mc
import MDP as m
import AltMDP as am
import AAAPI as aaa
import PCA as pca
import OptPolicy as op
import MazeGenerator as mg
import MDP_1D as mdp1
from sklearn.decomposition import PCA


def runVI(nRows, nCols, goalP, wallP):
    # nRows = 3
    # nCols = 3
    w = nRows * nCols
    # goalP = 2 * 9 + 2
    # wallP = [1 * 9 + 2]

    mdp = m.MDP(nRows, nCols, wallP, goalP)
    mdp.CreateGrid()
    mdp.TnR()
    mdp.InitRnT()

    v, p, it, time = vi.ValueIteration(mdp.states, mdp.actions, mdp.reward, mdp.transition, mdp.gamma, mdp.numRows, mdp.numCol,
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

    mdp = m.MDP(nRows, nCols, wallP, goalP)
    mdp.CreateGrid()
    mdp.TnR()
    mdp.InitRnT()

    v, p, it, time = pi.PolicyIteration(mdp.states, mdp.actions, mdp.reward, mdp.transition, mdp.gamma, mdp.numRows, mdp.numCol,
                              mdp.grid, mdp.wall, mdp.goal, mdp.mult)

    # print(v)
    # well this is the part where we will use pca with the "optimal" Value function that PI returns
    # pca.Pca(v, p, mdp.numRows, mdp.numCol)

    # op.optimalPolicy(mdp.states, mdp.actions, mdp.grid, mdp.goal, mdp.wall, p, mdp.numCol, mdp.numRows, mdp.mult)
    Count(p)


def runAltPI(rows, cols, goal, wall):
    nRows = rows
    nCols = cols
    w = nRows * nCols
    # wallP = [1 * w + 2, 0 * w, 2]
    # goalP = 2 * w + 2

    goalP, wallP = goal, wall # mg.MazeGrid(nRows, nCols)

    mdp = am.AltMDP(nRows, nCols, wallP, goalP)
    mdp.CreateGrid()
    mdp.TnR()
    mdp.InitRnT()

    p, v, it, altit, time = api.AlternatePI(mdp.states, mdp.statesA, mdp.statesB, mdp.actions, mdp.actionsA, mdp.actionsB, mdp.grid,
                           mdp.gridStates, mdp.wall, mdp.goal, mdp.mult, mdp.transition, mdp.reward, mdp.gamma)

    # print("value: ", v)
    op.optimalPolicy(mdp.states, mdp.actions, mdp.grid, mdp.goal, mdp.wall, p, mdp.numA, mdp.numB, mdp.mult)


def AAAPI(rows, cols, goal, wall):
    nRows = rows
    nCols = cols
    w = nRows * nCols
    # wallP = [1 * w + 2, 0 * w, 2]
    # goalP = 2 * w + 2

    goalP, wallP = goal, wall  # mg.MazeGrid(nRows, nCols)

    mdp = am.AltMDP(nRows, nCols, wallP, goalP)
    mdp.CreateGrid()
    mdp.TnR()
    mdp.InitRnT()

    p, v, it, altit, time = aaa.AlternatePI(mdp.states, mdp.statesA, mdp.statesB, mdp.actions, mdp.actionsA,
                                            mdp.actionsB, mdp.grid,
                                            mdp.gridStates, mdp.wall, mdp.goal, mdp.mult, mdp.transition, mdp.reward,
                                            mdp.gamma)

    op.optimalPolicy(mdp.states, mdp.actions, mdp.grid, mdp.goal, mdp.wall, p, mdp.numA, mdp.numB, mdp.mult)


    # op.optimalPolicy(mdp.states, mdp.actions, mdp.grid, mdp.goal, mdp.wall, p, mdp.numA, mdp.numB, mdp.mult)

def runMC(nRows, nCols, wallP, goalP):

    mdp = mdp1.MDP(nRows, nCols, goalP, wallP)
    mdp.CreateGrid()
    mdp.TnR()
    mdp.InitRnT()

    v, p = mc.FirstVisitMC(mdp.states, mdp.actions, mdp.reward, mdp.transition, mdp.gamma, mdp.numRows, mdp.numCol,
                           mdp.grid, mdp.wall, mdp.goal, mdp.mult)

    # well this is the part where we will use pca with the "optimal" Value function that PI returns
    # pca.Pca(v, p, mdp.numRows, mdp.numCol)

    # op.optimalPolicy(mdp.states, mdp.actions, mdp.grid, mdp.goal, mdp.wall, p, mdp.numCol, mdp.numRows, mdp.mult)


def Count(policy):

    cnt = 0
    for p in policy:
        if p is not None:
            cnt += 1

    print(cnt)


def main():
    rows = 3
    cols = 3
    w = rows * cols
    goal = (rows - 1) * w + (cols - 1)
    wall = [(rows - 2) * w + (cols - 1)]  # , (rows - 2) * w + (cols - 2)]
    # goal, wall = mg.MazeGrid(rows, cols)
    # goal, wall = mg.gridFigure(rows, cols)

    # AAAPI(rows, cols, goal, wall)
    runPI(rows, cols, goal, wall)
    # runVI(rows, cols, goal, wall)
    # runMC(rows, cols, goal, wall)
    # runAltPI(rows, cols, goal, wall)


if __name__ == "__main__":
    main()
