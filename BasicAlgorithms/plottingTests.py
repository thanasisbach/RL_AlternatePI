from typing import Any, Union

import numpy as np
import statistics as st
import PolicyIteration as pi
import ValueIteration as vi
import AlternatePI as api
import AAAPI as aaa
import MonteCarlo as mc
import MDP as m
import AltMDP as am
import PCA as pca
import OptPolicy as op
import matplotlib.pyplot as plt
from numpy.core.multiarray import ndarray
from scipy import optimize
import random
import math
import MazeGenerator as mg


def runAAAPI(nRows, nCols, wallP, goalP):
    mdp = am.AltMDP(nRows, nCols, wallP, goalP)
    mdp.CreateGrid()
    mdp.TnR()
    mdp.InitRnT()

    p, v, iter, extraiter, time = aaa.AlternatePI(mdp.states, mdp.statesA, mdp.statesB, mdp.actions, mdp.actionsA,
                                            mdp.actionsB, mdp.grid,
                                            mdp.gridStates, mdp.wall, mdp.goal, mdp.mult, mdp.transition, mdp.reward,
                                            mdp.gamma)
    # op.optimalPolicy(mdp.states, mdp.actions, mdp.grid, mdp.goal, mdp.wall, p, mdp.numA, mdp.numB, mdp.mult)
    return time


def runAltPI(nRows, nCols, wallP, goalP):
    mdp = am.AltMDP(nRows, nCols, wallP, goalP)
    mdp.CreateGrid()
    mdp.TnR()
    mdp.InitRnT()

    p, v, iter, extraiter, time = api.AlternatePI(mdp.states, mdp.statesA, mdp.statesB, mdp.actions, mdp.actionsA,
                                            mdp.actionsB, mdp.grid,
                                            mdp.gridStates, mdp.wall, mdp.goal, mdp.mult, mdp.transition, mdp.reward,
                                            mdp.gamma)
    # op.optimalPolicy(mdp.states, mdp.actions, mdp.grid, mdp.goal, mdp.wall, p, mdp.numA, mdp.numB, mdp.mult)
    return time


def runPI(nRows, nCols, wallP, goalP):
    mdp = m.MDP(nRows, nCols, wallP, goalP)
    mdp.CreateGrid()
    mdp.TnR()
    mdp.InitRnT()

    v, p, iter, time = pi.PolicyIteration(mdp.states, mdp.actions, mdp.reward, mdp.transition, mdp.gamma, mdp.numRows,
                                    mdp.numCol,
                                    mdp.grid, mdp.wall, mdp.goal, mdp.mult)

    # op.optimalPolicy(mdp.states, mdp.actions, mdp.grid, mdp.goal, mdp.wall, p, mdp.numCol, mdp.numRows, mdp.mult)
    return time





def runVI(nRows, nCols, wallP, goalP):
    mdp = m.MDP(nRows, nCols, wallP, goalP)
    mdp.CreateGrid()
    mdp.TnR()
    mdp.InitRnT()

    v, p, iter, time = vi.ValueIteration(mdp.states, mdp.actions, mdp.reward, mdp.transition, mdp.gamma, mdp.numRows,
                                   mdp.numCol,
                                   mdp.grid, mdp.wall, mdp.goal, mdp.mult)

    # op.optimalPolicy(mdp.states, mdp.actions, mdp.grid, mdp.goal, mdp.wall, p, mdp.numCol, mdp.numRows, mdp.mult)
    return time



def different_square_with_worst_case(size):
    length = len(size)
    i = 0
    run_aaaPI = []
    run_alt_pi = []
    run_pi = []
    run_vi = []
    MeanAAAPI = []
    stdAAAPI = []
    MeanAPI = []
    stdAPI = []
    MeanPI = []
    stdPI = []
    MeanVI = []
    stdVI = []
    repetition = 10

    while i < length:

        run_aaaPI = []
        run_alt_pi = []
        run_pi = []
        run_vi = []


        for j in range(repetition):
            size_inner = size[i]
            multiple = size_inner
            # goal = (size[i] - 1) * (size[i] * size[i]) + (size[i] - 1)
            # wall = [-1]
            goal, wall = mg.MazeGrid(size[i], size[i])
            # goal, wall = mg.gridFigure(size[i], size[i])

            p0 = runAAAPI(size_inner, size_inner, wall, goal)
            run_aaaPI.append(p0)

            # p1 = runAltPI(size_inner, size_inner, wall, goal)
            # run_alt_pi.append(p1)

            p2 = runPI(size_inner, size_inner, wall, goal)
            run_pi.append(p2)

            # runMC(size_inner,size_inner,wall,goal)

            p4 = runVI(size_inner, size_inner, wall, goal)
            run_vi.append(p4)

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

    print("AVG time alt", MeanAPI)
    print("STD alt pi", stdAPI)

    print("AVG time PI", MeanPI)
    print("STD pi", stdPI)

    print("AVG time VI", MeanVI)
    print("STD VI", stdVI)

          # return;
    # plot_test(size, run_alt_pi, run_pi, run_vi)


def plot_test(size, alt_pi, pi, vi):
    plt.figure()

    x0 = size
    y0 = alt_pi  # [11, 999, 1, 290, 599, 3200, 99999]
    plt.scatter(x0[:], y0[:], 25, "red")

    y1 = pi
    plt.scatter(x0[:], y1[:], 25, "blue")

    y2 = vi
    plt.scatter(x0[:], y2[:], 25, "green")

    # A3, B3, C3, D3 = optimize.curve_fit(f_3, x0, y0)[0]
    # x3 = np.arange(0, 550, 2)
    # y3 = (A3 * x3 * x3 * x3 * x3) + (B3 * x3 * x3 * x3) + (C3 * x3 * x3) + D3
    # plt.plot(x3, y3, "purple")

    plt.gca().legend(("Alternating PI", "PI", "VI"))
    plt.title("Convergence Rate in Maze")
    plt.xlabel('Number of Grid size, NxN')
    plt.ylabel('Time in Seconds')

    plt.show()


def main():

    # size = [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    # size = [3, 5, 10, 15, 20, 25, 30, 60, 100, 150, 200]
    # size = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    size = [60]
    # for i in size:  # 571 was the upper bound of iterations
    #     size.append(i)
    # i += 20
    # size = [3, 5, 10] # , 250, 500, 750, 1000, 5000]
    print(size)
    different_square_with_worst_case(size)


if __name__ == "__main__":
    main()
