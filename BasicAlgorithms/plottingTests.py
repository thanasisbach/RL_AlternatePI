import numpy as np
import PolicyIteration as pi
import ValueIteration as vi
import AlternatePI as api
import MonteCarlo as mc
import MDP as m
import AltMDP as am
import PCA as pca
import OptPolicy as op
import matplotlib.pyplot as plt
from scipy import optimize
import random
import math
import MazeGenerator.py as mg


def runAltPI(nRows, nCols, wallP, goalP):
    mdp = am.AltMDP(nRows, nCols, wallP, goalP)
    mdp.CreateGrid()
    mdp.TnR()
    mdp.InitRnT()

    p, v, iter, extraiter = api.AlternatePI(mdp.states, mdp.statesA, mdp.statesB, mdp.actions, mdp.actionsA,
                                            mdp.actionsB, mdp.grid,
                                            mdp.gridStates, mdp.wall, mdp.goal, mdp.mult, mdp.transition, mdp.reward,
                                            mdp.gamma)
    # op.optimalPolicy(mdp.states, mdp.actions, mdp.grid, mdp.goal, mdp.wall, p, mdp.numA, mdp.numB, mdp.mult)
    return iter


def runPI(nRows, nCols, wallP, goalP):
    mdp = m.MDP(nRows, nCols, wallP, goalP)
    mdp.CreateGrid()
    mdp.TnR()
    mdp.InitRnT()

    v, p, iter = pi.PolicyIteration(mdp.states, mdp.actions, mdp.reward, mdp.transition, mdp.gamma, mdp.numRows,
                                    mdp.numCol,
                                    mdp.grid, mdp.wall, mdp.goal, mdp.mult)

    # op.optimalPolicy(mdp.states, mdp.actions, mdp.grid, mdp.goal, mdp.wall, p, mdp.numCol, mdp.numRows, mdp.mult)
    return iter


def runMC(nRows, nCols, wallP, goalP):
    mdp = m.MDP(nRows, nCols, wallP, goalP)
    mdp.CreateGrid()
    mdp.TnR()
    mdp.InitRnT()

    v, p = mc.FirstVisitMC(mdp.states, mdp.actions, mdp.reward, mdp.transition, mdp.gamma, mdp.numRows, mdp.numCol,
                           mdp.grid, mdp.wall, mdp.goal, mdp.mult)
    return v, p


def runVI(nRows, nCols, wallP, goalP):
    mdp = m.MDP(nRows, nCols, wallP, goalP)
    mdp.CreateGrid()
    mdp.TnR()
    mdp.InitRnT()

    v, p, iter = vi.ValueIteration(mdp.states, mdp.actions, mdp.reward, mdp.transition, mdp.gamma, mdp.numRows,
                                   mdp.numCol,
                                   mdp.grid, mdp.wall, mdp.goal, mdp.mult)

    # op.optimalPolicy(mdp.states, mdp.actions, mdp.grid, mdp.goal, mdp.wall, p, mdp.numCol, mdp.numRows, mdp.mult)
    return iter


def mask(max_size, wall_num_portion):
    row_inc = 3
    col_inc = 3
    wall_inc = 0
    wall_list = []
    wall_x_list = []
    wall_y_list = []

    while col_inc < max_size:

        while row_inc <= col_inc:

            wall_num = math.ceil(row_inc * col_inc * wall_num_portion)
            wall_x_list = random.sample(range(row_inc), wall_num + 1)
            wall_y_list = random.sample(range(col_inc), wall_num + 1)
            multiple = row_inc * col_inc

            while wall_inc < wall_num:
                wall_position_as_grid = wall_x_list[wall_inc] * multiple + wall_y_list[wall_inc]
                wall_list.append(wall_position_as_grid)
                wall_inc += 1
            wall_inc = 0
            goal = wall_x_list[wall_num] * multiple + wall_y_list[wall_num]

            print("wall_num", wall_num)
            print("multiple:", multiple)
            print("row size :", row_inc, "coloumn size:", col_inc)
            print("wall_num", wall_num)
            print("wall coordiante:", wall_x_list[0], wall_y_list[0], wall_list[0])
            print("goal corrdiante:", wall_x_list[wall_num], wall_y_list[wall_num], goal)
            runAltPI(row_inc, col_inc, wall_list[0], goal)
            # runPI(row_inc, col_inc, wall_list[0],goal)
            # runMC(row_inc, col_inc, wall_list[0],goal)
            # row_inc+=1

            wall_list = []
            wall_x_list = []
            wall_y_list = []

        row_inc = 3
        col_inc += 1
    return


def different_square_with_worst_case(size):
    length = len(size)
    i = 0
    run_alt_pi = []
    run_pi = []
    run_vi = []

    while i < length:
        size_inner = size[i]
        multiple = size_inner
        goal = (size[i] - 1) * (size[i] * size[i]) + (size[i] - 1)
        wall = (size[i] - 2) * (size[i] * size[i]) + (size[i] - 1)

        p1 = runAltPI(size_inner, size_inner, wall, goal)
        run_alt_pi.append(p1)

        p2 = runPI(size_inner, size_inner, wall, goal)
        run_pi.append(p2)

        # runMC(size_inner,size_inner,wall,goal)

        p4 = runVI(size_inner, size_inner, wall, goal)
        run_vi.append(p4)
        i += 1

    print("alt", run_alt_pi)
    print("pi", run_pi)
    print("run_vi", run_vi)
    # return;
    plot_test(size, run_alt_pi, run_pi, run_vi)

def f_3(x, A, B, C, D):

    return A * x * x * x + B * x * x + C * x + D


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

    plt.gca().legend(("Alternate PI", "PI", "VI"))
    plt.title("Convergence")
    plt.xlabel('DP RL')
    plt.ylabel('number_of_iteration')

    plt.show()


def main():
    size = [1000]  # [3, 5, 10, 100, 250, 500, 750, 1000, 5000]
    different_square_with_worst_case(size)


if __name__ == "__main__":
    main()
