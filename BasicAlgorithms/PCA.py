import numpy as np
import matplotlib.pyplot as plt


def Pca(valueMatrix, r, c):

    val = Mean(valueMatrix)

    rr = shapeMat(valueMatrix, r, c)
    print("Rank of Value Func:", np.linalg.matrix_rank( np.asarray(rr) ))


    for i in range(len(valueMatrix)):
        valueMatrix[i] -= val

    Matrix = shapeMat(valueMatrix, r, c)

    # So now we are going to separate the features (rows and columns)
    dimI0 = Matrix[0]
    dimI1 = Matrix[1]
    dimI2 = Matrix[2]

    dimJ0 = [Matrix[0][0], Matrix[1][0], Matrix[2][0]]
    dimJ1 = [Matrix[0][1], Matrix[1][1], Matrix[2][1]]
    dimJ2 = [Matrix[0][2], Matrix[1][2], Matrix[2][2]]


    mm = [dimI2, dimJ1]
    a = list(map(list, zip(*mm)))
    Cov = np.dot(mm, a)

    [U, S, V] = np.linalg.svd(Cov)

    CC = np.dot(U, np.dot(np.diag(S), V))

    print("Covariance: ", Cov, "\nMatrix: ", mm)
    print("S Matrix", np.diag(S), "V Matrix", V)

    plotFeatures(dimI2, dimJ1, "Row 2 - Column 1", V[0], V[1])
    # plotFeatures(dimI2, dimJ0, "Row 2 - Column 0")


# makes a vector matrix into a 2-d matrix, given the dimensions
def shapeMat(vecMat, r, c):

    Matrix = [[0 for x in range(r)] for y in range(c)]
    cnt = 0
    for i in range(r):
        for j in range(c):
            Matrix[i][j] = vecMat[cnt]
            cnt += 1

    return Matrix


# calculates the arithmetical mean of a matrix
def Mean(matrix):

    # the matrix here is represented as a list that's the reason we use sum
    s = sum(matrix)
    m = s / len(matrix)  # [x / s for x in matrix]
    print("Mean: ", m)
    return m


def plotFeatures(dim1, dim2, title, pc1, pc2):

    plt.scatter(dim1, dim2)
    plt.title(title)
    plt.xlabel("Rows")
    plt.ylabel("Columns")

    plt.plot([0, pc1[0]], [0, pc1[1]])
    plt.plot([0, pc2[0]], [0, pc2[1]])

    plt.gca().legend(("PC1", "PC2"))

    plt.show()

