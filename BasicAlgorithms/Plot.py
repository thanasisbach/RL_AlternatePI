import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import random
import math



def plot_test(size, pi, aaa_pi):

    plt.figure()

    x0 = size

    y0 = aaa_pi  # [11, 999, 1, 290, 599, 3200, 99999]
    plt.plot(x0[:], y0[:], color="red", linestyle='dashed', marker= "*", markersize=8)

    y1 = pi
    plt.plot(x0[:], y1[:], color="blue", linestyle='dashed', marker= "o", markersize=8)

    # y2 = vi
    # plt.plot(x0[:], y2[:], color="green", linestyle='dashed', marker= "^", markersize=8)

    # y3 = alt_pi
    # plt.plot(x0[:], y3[:], color="grey", linestyle='dashed', marker= "s", markersize=8)

    plt.gca().legend(("AAA-PI", "PI"))  # , "VI", "Alt-PI"
    plt.title("Number of Iterations until Convergence in 3D Grid-World with One walls")
    plt.xlabel('Number of Grid size, NxN')
    plt.ylabel('Time till Convergence in Seconds')

    plt.show()


def main():

    # size = [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    size = [3, 5, 10, 15, 20, 25, 30]
    # size = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    print(size)
    # aaapi = [143, 163, 199, 225, 270, 284, 324, 351, 380, 405, 443]

    # alt = [350, 400, 478, 562, 674, 794, 903, 1090, 1205, 1311, 1452]

    # pi = [285, 327, 399, 447, 535, 564, 630, 690, 749, 803, 873]

    # vi = [289, 327, 399, 444, 522, 564, 627, 676, 739, 792, 864]

    aaapi = [0.002931356430053711, 0.014217615127563477, 0.10392594337463379, 0.2934446334838867, 0.5810866355895996, 1.0222532749176025, 1.7929415702819824]

    pi = [0.042859792709350586, 0.18460988998413086, 0.9763948917388916, 2.354036808013916, 4.5911619663238525, 7.853440523147583, 12.32552433013916]

    plot_test(size, pi, aaapi)


if __name__ == "__main__":
    main()