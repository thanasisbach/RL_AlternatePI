import numpy as np
import statistics as st
import MazeGenerator as mg
import MDP_3D as mdp3
import MDP_Dec_3D as mdpd3
import PI_3D as pi3
import AAAPI3D as a3pi3
import VI3D as vi3
import random


def runVI3D(rows, cols, numZ, goal, wall):
    nRows = rows
    nCols = cols
    w = nRows * nCols
    # wallP = [(nRows - 2) * w + (nCols - 1), 0 * w + 2]
    # goalP = (nRows - 1) * w + (nCols - 1)

    goalP, wallP = goal, wall

    mdp = mdp3.MDP3D(rows, cols, numZ, wall, goal)
    mdp.CreateGrid()
    mdp.TnR()
    mdp.InitRnT()

    v, p, it, time = vi3.ValueIteration3D(mdp.states, mdp.actions, mdp.reward, mdp.transition, mdp.gamma, mdp.numRows,
                                           mdp.numCol, mdp.numZ, mdp.grid, mdp.wall, mdp.goal, mdp.mult, mdp.mult2)

    # print("policy: ", p)
    # Count(p)
    return time

def runPI3D(rows, cols, numZ, goal, wall):
    nRows = rows
    nCols = cols
    w = nRows * nCols
    # wallP = [(nRows - 2) * w + (nCols - 1), 0 * w + 2]
    # goalP = (nRows - 1) * w + (nCols - 1)

    goalP, wallP = goal, wall

    mdp = mdp3.MDP3D(rows, cols, numZ, wall, goal)
    mdp.CreateGrid()
    mdp.TnR()
    mdp.InitRnT()

    v, p, it, time = pi3.PolicyIteration3D(mdp.states, mdp.actions, mdp.reward, mdp.transition, mdp.gamma, mdp.numRows,
                                           mdp.numCol, mdp.numZ, mdp.grid, mdp.wall, mdp.goal, mdp.mult, mdp.mult2)

    # print("policy: ", p)
    # Count(p)
    return time

def runAAAPI3D(rows, cols, numZ, goal, wall):
    nRows = rows
    nCols = cols
    w = nRows * nCols
    # wallP = [(nRows - 2) * w + (nCols - 1), 0 * w + 2]
    # goalP = (nRows - 1) * w + (nCols - 1)

    goalP, wallP = goal, wall

    mdp = mdpd3.AltMDP(rows, cols, numZ, wall, goal)
    mdp.CreateGrid()
    mdp.TnR()
    mdp.InitRnT()

    v, p, it, exIt, time = a3pi3.AlternatePI3D(mdp.states, mdp.statesA, mdp.statesB, mdp.statesZ, mdp.actions, mdp.actionsA,
                                       mdp.actionsB, mdp.actionsZ, mdp.grid, mdp.gridStates, mdp.wall, mdp.goal,
                                       mdp.mult, mdp.mult2, mdp.transition, mdp.reward, mdp.gamma)

    # print("policy: ", p)
    # Count(p)
    return time


def Count(policy):

    cnt = 0
    for p in policy:
        if p is not None:
            cnt += 1

    print(cnt)

def gridFigure(rlen, clen, zlen):
    imgx = 512
    imgy = 512
    # image = Image.new("RGB", (imgx, imgy))
    # pixels = image.load()
    # pixels2 = image.load()


    a = 0.01
    gridLen = rlen * clen * zlen

    wallLen = int(gridLen * a)
    if wallLen == 0:
        wallLen = 1

    goal = (rlen - 1) * gridLen + (clen - 1) * (gridLen // zlen) + (zlen - 1)
    gg = [goal, (rlen - 1) * gridLen + (clen - 1) * (gridLen // zlen) + (zlen - 2)]
    # print(gg)
    wall = []

    # print(cc[0][1])
    for i in range(wallLen):

        # wall.append(cc[i][1] * gridLen + cc[i][0])
        val = True
        while val:
            r = random.randint(0, rlen - 1)
            c = random.randint(0, clen - 1)
            z = random.randint(0, zlen - 1)
            w = r * gridLen + c * (gridLen // zlen) + z
            if (w not in wall) and (w not in gg):
                wall.append(w)
                val = False

    maze = [[[0 for x in range(rlen)] for y in range(clen)] for z in range(zlen)]
    for ii in range(rlen):
        for jj in range(clen):
            for zz in range(zlen):

                kappa = ii * gridLen + jj * (gridLen // zlen) + zz
                if kappa in wall:
                    maze[ii][jj][zz] = 0
                else:
                    maze[ii][jj][zz] = 1


    # print("wall number: ", len(wall), "grid size: ", gridLen)
    return goal, wall


def MultRuns(size):
    length = len(size)
    i = 0
    run_aaaPI = []
    # run_alt_pi = []
    run_pi = []
    MeanAAAPI = []
    stdAAAPI = []
    MeanPI = []
    stdPI = []
    MeanVI = []
    stdVI = []
    # run_vi = []
    num_iters = 10


    while i < length:
        run_aaaPI = []
        run_pi = []
        run_vi = []

        for j in range(num_iters):

            size_inner = size[i]
            multiple = size_inner
            # goal = (size[i] - 1) * (size[i] * size[i] * size[i]) + (size[i] - 1) * (size[i] * size[i]) + (size[i] - 1)
            # wall = [-1]
            # [(size[i] - 1) * (size[i] * size[i] * size[i]) + (size[i] - 1) * (size[i] * size[i]) + (size[i] - 2)]
            # goal, wall = mg.MazeGrid(size[i], size[i])
            goal, wall = gridFigure(size[i], size[i], size[i])

            p0 = runAAAPI3D(size_inner, size_inner, size_inner, goal, wall)
            run_aaaPI.append(p0)

            # p1 = runAltPI(size_inner, size_inner, wall, goal)
            # run_alt_pi.append(p1)
            p1 = runVI3D(size_inner, size_inner, size_inner, goal, wall)
            run_vi.append(p1)

            p2 = runPI3D(size_inner, size_inner, size_inner, goal, wall)
            run_pi.append(p2)

        MeanAAAPI.append(np.mean(run_aaaPI))
        stdAAAPI.append(st.stdev(run_aaaPI))
        print("AAAPI", MeanAAAPI, stdAAAPI)

        # MeanAPI.append(np.mean(run_alt_pi))
        # stdAPI.append(st.stdev(run_alt_pi))

        MeanPI.append(np.mean(run_pi))
        stdPI.append(st.stdev(run_pi))
        print("PI", MeanPI, stdPI)

        MeanVI.append(np.mean(run_vi))
        stdVI.append(st.stdev(run_vi))
        print("VI", MeanVI, stdVI)

        i += 1

    print("AVG time aaapi", MeanAAAPI)
    print("STD AAAPI", stdAAAPI)

    print("AVG time alt", MeanPI)
    print("STD alt pi", stdPI)

    print("AVG time VI", MeanVI)
    print("STD VI", stdVI)

    # print("AVG time PI", MeanPI)
    # print("STD pi", stdPI)
    #


def main():
    rows = 3
    cols = 3
    numZ = 3
    w2 = rows * cols * numZ
    w = cols * numZ
    goal = (rows - 1) * w2 + (cols - 1) * w + (numZ - 1)
    wall = [(rows - 1) * w2 + (cols - 1) * w + (numZ - 2)]  # , (rows - 2) * w + (cols - 2)]
    size = [3, 5, 10, 15, 20, 25, 30]  # , 40, 50]



    # goal, wall = mg.MazeGrid(rows, cols)
    # goal, wall = mg.gridFigure(rows, cols)
    # runPI3D(rows, cols, numZ, goal, wall)
    # runAAAPI3D(rows, cols, numZ, goal, wall)
    MultRuns(size)


if __name__ == "__main__":
    main()
