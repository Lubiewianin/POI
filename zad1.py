from scipy.stats import norm
from csv import writer
import numpy as np



def generate_points(num_points:int=2000):
    distribution_x = norm(loc=0, scale=50)
    distribution_y = norm(loc=-500, scale=20)
    distribution_z = norm(loc=0.2, scale=0.05)

    x = distribution_x.rvs(size=num_points)
    y = distribution_y.rvs(size=num_points)
    z = distribution_z.rvs(size=num_points)

    points = zip(x, y, z)
    return points



cloud_points = generate_points(1000)
with open('LidarData.xyz', 'w', encoding='utf-8', newline='\n') as csvfile:
    csvwriter = writer(csvfile)
    # csvwriter.writerow('x', 'y', 'z')
    for p in cloud_points:
        csvwriter.writerow(p)
