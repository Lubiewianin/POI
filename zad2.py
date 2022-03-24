# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from sklearn.cluster import KMeans
import csv
import numpy as np
import matplotlib.pyplot as plt


def ransac():
    n = 50
    k = 100  # liczba iteracji
    t = 300  # szumy
    d = 300

    iterations = 0
    bestFit = null
    bestErr = 99999999999

    while iterations < k:
        maybeInliers


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
w1 = zip((X[c1, 0], X[c1, 1], X[c1, 2]))
w2 = zip((X[c2, 0], X[c2, 1], X[c2, 2]))
w3 = zip((X[c3, 0], X[c3, 1], X[c3, 2]))

plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.scatter3D(X[c1, 0], X[c1, 1], X[c1, 2], c="red")
ax.scatter3D(X[c2, 0], X[c2, 1], X[c2, 2], c="blue")
ax.scatter3D(X[c3, 0], X[c3, 1], X[c3, 2], c="cyan")

