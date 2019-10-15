# Random Maze Generator using Depth-first Search (Recursive Version)
# http://en.wikipedia.org/wiki/Maze_generation_algorithm
# FB - 20121205
import random
from PIL import Image


def GenerateMaze(cx, cy, mx, my, dx, dy, maze):

    maze[cy][cx] = 1
    while True:
        # find a new cell to add
        nlst = []  # list of available neighbors
        for i in range(4):
            nx = cx + dx[i]
            ny = cy + dy[i]
            if nx >= 0 and nx < mx and ny >= 0 and ny < my:
                if maze[ny][nx] == 0:
                    # of occupied neighbors of the candidate cell must be 1
                    ctr = 0
                    for j in range(4):
                        ex = nx + dx[j]
                        ey = ny + dy[j]
                        if ex >= 0 and ex < mx and ey >= 0 and ey < my:
                            if maze[ey][ex] == 1:
                                ctr += 1
                    if ctr == 1:
                        nlst.append(i)
        # if 1 or more available neighbors then randomly select one and add
        if len(nlst) > 0:
            ir = nlst[random.randint(0, len(nlst) - 1)]
            cx += dx[ir]
            cy += dy[ir]
            maze = GenerateMaze(cx, cy, mx, my, dx, dy, maze)
        else:
            return maze
            # break


def MazeGrid(r, c):

    imgx = 512
    imgy = 512
    image = Image.new("RGB", (imgx, imgy))
    pixels = image.load()
    pixels2 = image.load()
    mx = r
    my = c  # width and height of the maze
    maze = [[0 for x in range(mx)] for y in range(my)]
    maze2 = [[0 for x in range(mx)] for y in range(my)]
    dx = [0, 1, 0, -1]
    dy = [-1, 0, 1, 0]  # directions to move in the maze

    maze = GenerateMaze(0, 0, mx, my, dx, dy, maze)
    # paint the maze
    for ky in range(imgy):
        for kx in range(imgx):
            m = maze[my * ky // imgy][mx * kx // imgx] * 255
            pixels[kx, ky] = (m, m, m)
    image.save("RandomMaze_" + str(mx) + "x" + str(my) + ".png", "PNG")
    print(maze)
    # print(len(maze))
    # print(len(maze[0]))
    mm = r * c
    goal = (r - 1) * mm + (c - 1)
    wall = []
    for i in range(r):
        for j in range(c):
            if maze[i][j] == 0:
                wall.append((r - 1 - i) * mm + (c - 1 - j))
            # break


    #########################################
    ### maze checking ######

    for ii in range(r):
        for jj in range(c):
            kappa = ii * mm + jj
            if kappa in wall:
                maze2[ii][jj] = 0
            else:
                maze2[ii][jj] = 1



    for ky in range(imgy):
        for kx in range(imgx):
            m = maze2[my * ky // imgy][mx * kx // imgx] * 255
            pixels2[kx, ky] = (m, m, m)
    image.save("testMaze_" + str(mx) + "x" + str(my) + ".png", "PNG")

    ##########################################


    print(wall)
    print("wall number: ", len(wall), "grid size: ", r * c)
    return goal, wall


def gridFigure(rlen, clen):
    imgx = 512
    imgy = 512
    image = Image.new("RGB", (imgx, imgy))
    pixels = image.load()
    pixels2 = image.load()


    a = 0.03
    gridLen = rlen * clen
    wallLen = 2  # int(gridLen * a)
    goal = (rlen - 1) * gridLen + (clen - 1)
    gg = [goal, (rlen - 2) * gridLen + (clen - 2), (rlen - 2) * gridLen + (clen - 1), (rlen - 1) * gridLen + (clen - 2)]
    # print(gg)
    wall = []
    cc = {0: [5, 11], 1: [16, 6], 2: [2, 15], 3: [9, 13], 4: [17, 11], 5: [20, 2], 6: [17, 21], 7: [2, 18], 8: [18, 2],
          9: [7, 14], 10: [24, 7], 11: [17, 13], 12: [20, 20], 13: [11, 24], 14: [23, 17], 15: [5, 20], 16: [4, 14], 17: [0, 14]}
    # print(cc[0][1])
    for i in range(wallLen):

        wall.append(cc[i][1] * gridLen + cc[i][0])
        val = True

        while val:
            r = random.randint(0, rlen - 1)
            c = random.randint(0, clen - 1)
            w = r * gridLen + c
            if (w not in wall) and (w not in gg):
                wall.append(w)
                val = False

    maze = [[0 for x in range(rlen)] for y in range(clen)]
    for ii in range(rlen):
        for jj in range(clen):

            kappa = ii * gridLen + jj
            if kappa in wall:
                maze[ii][jj] = 0
            else:
                maze[ii][jj] = 1

    # image creation
    for ky in range(imgy):
        for kx in range(imgx):
            m = maze[clen * ky // imgy][rlen * kx // imgx] * 255
            pixels2[kx, ky] = (m, m, m)
    image.save("wallGridWorld" + str(rlen) + "x" + str(clen) + ".png", "PNG")

    print("wall number: ", len(wall), "grid size: ", gridLen)
    return goal, wall


if __name__ == "__main__":
    x, y = gridFigure(25, 25)
    # x, y = MazeGrid(60, 60)
