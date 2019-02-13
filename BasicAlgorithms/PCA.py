import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as pca1


def Pca(valueMatrix, policy, r, c):

    # val = Mean(valueMatrix)

    rr = shapeMat(valueMatrix, r, c)
    print("Rank of Value Func: ", np.linalg.matrix_rank(np.asarray(rr)))

    for k in range(len(policy)):
        if policy[k] is None:
            policy[k] = -1

    pp = shapeMat(policy, r, c)
    print("Rank of Policy matrix: ", np.linalg.matrix_rank(np.array(pp)))

    # for i in range(len(valueMatrix)):
    #     valueMatrix[i] -= val

    procedurePCA(valueMatrix, r, c)


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

    plt.figure()
    plt.scatter(dim1, dim2)
    plt.title(title)
    plt.xlabel("Rows")
    plt.ylabel("Columns")

    print("PC1: ", pc1, "PC2: ", pc2)
    plt.plot([0, pc1[0]], [0, pc1[1]])
    plt.plot([0, pc2[0]], [0, pc2[1]])

    plt.gca().legend(("PC1", "PC2"))

    # plt.show()

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops = dict(arrowstyle='->', linewidth=2, shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

def procedurePCA(valueMatrix, r, c):
    Matrix = shapeMat(valueMatrix, r, c)
    # TODO find a way to compute all pca in a method
    M = np.array(Matrix)
    mm = []
    for i in range(r):
        for j in range(c):
            mm.append( np.array([M[i, :], M[:, j]]).T )

    # So now we are going to separate the features (rows and columns)
    dimI0 = M[0, :]
    dimI1 = M[1, :]
    dimI2 = M[2, :]

    dimJ0 = M[:, 0]
    dimJ1 = M[:, 1]
    dimJ2 = M[:, 2]

    # mm = np.array([dimI2, dimJ2]).T
    # mm = np.array(mm).T
    mm1 = mm[20] - np.mean(mm[20], axis=0)
    n_samples = len(mm1[:, 0]) + len(mm1[:, 1])
    Cov = np.dot(mm1.T, mm1) / n_samples

    [U, S, V] = np.linalg.svd(Cov)
    # CC = np.dot(U, np.dot(np.diag(S), V))

    print("Covariance: ", Cov, "\nMatrix: ", mm1, np.mean(mm, axis=0))
    print("S Matrix", np.diag(S), "V Matrix", V)

    plotFeatures(mm1[:, 0], mm1[:, 1], "Row 0 - Column 1", V[0, :], V[1, :])
    # plotFeatures(dimI2, dimJ0, "Row 2 - Column 0")


    print(mm[20][:, 0], mm[20][:, 1])
    pca2 = pca1(n_components=2, whiten=True)
    pca2.fit(mm[20])

    plotFeatures(mm1[:, 0], mm1[:, 1], "From PCA library", pca2.components_[0, :], pca2.components_[1, :])
    plt.show()
    #### this is plotting code taken online ####
    # plt.figure()
    # fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    # fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
    #
    # # plot data
    # ax[0].scatter(mm[:, 0], mm[:, 1], alpha=0.2)
    # for length, vector in zip(pca2.explained_variance_, pca2.components_):
    #     v = vector * 3 * np.sqrt(length)
    #     draw_vector(pca2.mean_, pca2.mean_ + v, ax=ax[0])
    #
    # ax[0].axis('equal')
    # ax[0].set(xlabel='x', ylabel='y', title='input')
    #
    # # plot principal components
    # X_pca = pca2.transform(mm)
    # ax[1].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.2)
    # draw_vector([0, 0], [0, 3], ax=ax[1])
    # draw_vector([0, 0], [3, 0], ax=ax[1])
    # ax[1].axis('equal')
    # ax[1].set(xlabel='component 1', ylabel='component 2', title='principal components', xlim=(-5, 5), ylim=(-3, 3.1))
    #
    # # fig.savefig('figures/05.09-PCA-rotation.png')
    # plt.show()
    ##### it ends here #####

    print("Probably the eig vectors ", pca2.components_, "Covariance: ", pca2.get_covariance())
    print("This should be the eig values  ", pca2.explained_variance_)



