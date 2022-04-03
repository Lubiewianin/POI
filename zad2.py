# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from sklearn.cluster import KMeans
import csv
import numpy as np
import matplotlib.pyplot as plt
import random


def ransac(obj):
    k = 1000  # maksymalna liczba iteracji
    t = 0.01  # threshold
    ids = np.array([0, 0, 0])
    bestFit = [0]

    for a in range(0, k):
        for i in range(0, 3):  # losowanie 3 punktow
            ids[i] = random.randint(0, obj.shape[0] - 1)

        potInliers = obj[ids]
        uA = (potInliers[0] - potInliers[2]) / np.linalg.norm(potInliers[0] - potInliers[2])
        uB = (potInliers[1] - potInliers[2]) / np.linalg.norm(potInliers[1] - potInliers[2])
        uC = np.cross(uA, uB)
        D = -np.sum(np.multiply(uC, potInliers[2]))
        distance_all_points = (uC[0] * obj[:, 0] + uC[1] * obj[:, 1] + uC[2] * obj[:, 2] + D) / np.linalg.norm(uC)
        inliers = np.where(np.abs(distance_all_points) <= t)[0]
        if (len(bestFit) < len(inliers)):
            bestFit = inliers

    new_plane = obj[bestFit]
    print(len(new_plane))
    tmp_A = []
    tmp_b = []
    for i in range(len(new_plane)):
        tmp_A.append([new_plane[i, 0], new_plane[i, 1], 1])
        tmp_b.append(new_plane[i, 2])
    b = np.matrix(tmp_b).T
    A = np.matrix(tmp_A)
    fit = (A.T * A).I * A.T * b
    print(fit)
    # print(obj[bestFit])
    return 1


def reader():  # wczytanie danych z pliku
    with open('LidarData.xyz', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for a, b, c in reader:
            yield (float(a), float(b), float(c))


points = []
for p in reader():
    points.append(p)

x, y, z = zip(*points)  # tworzenie wykresu 3d - cala chmura
plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(x, y, z)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

clusterer = KMeans(n_clusters=3)  # znajdowanie chmur rozlacznych
X = np.array(points)
y_pred = clusterer.fit_predict(X)

c1 = y_pred == 0
c2 = y_pred == 1
c3 = y_pred == 2
w1 = X[c1]
w2 = X[c2]
w3 = X[c3]

plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.scatter3D(X[c1, 0], X[c1, 1], X[c1, 2], c="red")
ax.scatter3D(X[c2, 0], X[c2, 1], X[c2, 2], c="blue")
ax.scatter3D(X[c3, 0], X[c3, 1], X[c3, 2], c="cyan")

ransac(w1)


