# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 06:46:20 2022

@author: zsuet
"""
from sklearn.cluster import DBSCAN
import csv
import numpy as np
import matplotlib.pyplot as plt
import pyransac3d as pyrsc


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

X = np.array(points)
clusterer = DBSCAN(eps=20, min_samples=10).fit(X)
y_pred = clusterer.labels_

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

plane1 = pyrsc.Plane()
best_eq, best_inliers = plane1.fit(w1, 0.01)
