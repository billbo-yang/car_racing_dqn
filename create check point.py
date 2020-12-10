import numpy as np
import math


def main():
    mapCheckpoints = [(300, 500), (100, 600), (0, 600), (-100, 600), (-300, 500), (-300, 0), (-300, -500), (-100, -600),
                      (0, -600), (100, -600), (300, -500), (300, -100)]
    mapCheckpoints = [(845, 120), (833, 70), (787, 29), (145, 421), (49, 605), (75, 663), (19, 765), (19, 829),
                      (113, 1013), (65, 1139), (553, 1253), (663, 1197), (855, 1265)]
    checkpoints = [(0, 451.5, 640)] #Zero rad point must include
    x_avg, y_avg = 0, 0
    for i in range(len(mapCheckpoints) - 1, 0, -1):
        x_avg += mapCheckpoints[i][0]
        y_avg += mapCheckpoints[i][1]
    x_avg /= len(mapCheckpoints)
    y_avg /= len(mapCheckpoints)
    for i in range(len(mapCheckpoints) - 1, 0, -1):
        x, y = mapCheckpoints[i]
        # x/=4
        # y/=4
        # move coordinates to rotate about 0,0
        x -= 451.5  # 910/2
        y -= 640  # 1280/2
        rad, alpha = cartesianToPolar(x, y)
        if alpha <= 0:
            alpha += 2 * math.pi
        checkpoints.append((alpha, rad * math.cos(alpha), rad * math.sin(alpha)))
    rotate(checkpoints, 1)
    print(checkpoints)


def rotate(l, n):
    return l[n:] + l[:n]


def cartesianToPolar(x, y):
    R = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return (R, phi)


main()
