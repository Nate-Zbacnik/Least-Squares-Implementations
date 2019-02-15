# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 11:11:33 2019

@author: NATE
This program creates from the ground up a least squares polynomial approximation
of a grid of points sampled on sine with a user defined amount of
measurement error. Number of points and degree of poly are also user defined.
"""

# -*- coding: utf-8 -*-

import math
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IPython import get_ipython


#plt.close('all')
get_ipython().run_line_magic('matplotlib', 'qt') #put image in new window

#User input
deg = int(input('total degree of polynomial fit: '))
num_pts = int(input('number of grid points in x and y direction: '))
std_dev = float(input('error in measurement: '))
k=2

num_terms = int((deg+1)*(deg+2)/2)

#even distribution of x values
x_values = np.linspace(0,k*math.pi,num_pts)

y_values = np.array(x_values)
x_values = np.array(x_values)

#introduce random error into sine of x
data_points = [np.sin(x+y)+random.uniform(-std_dev,std_dev) \
               for y in y_values for x in x_values]
data_points = np.array(data_points)

#actual values for plotting sine
sin_points = [np.sin(x+y) \
              for y in  y_values for x in x_values]
sin_points = np.array(sin_points)

#initialize matrix and vector
lsq_mat = np.zeros((num_terms, num_terms))
lsq_matt = np.zeros((num_terms, num_terms))
lsq_vec = np.zeros((num_terms, 1))

#Initialize arrays to help limit redundant computations
comp_mat_x = np.ones((num_terms+1, len(x_values))) 
comp_mat_y = np.zeros((num_terms+1, len(x_values)))
comp_mat = np.zeros(((num_terms+1)**2,(num_pts)**2))


for i in range(1,num_terms+1): #Calculate powers ahead preemptively
    comp_mat_x[i,] = [x*comp_mat_x[i-1,ind] for ind, x in enumerate(x_values)]

comp_mat_y = comp_mat_x
   

for i in range(0,2*deg+1):
    for j in range(0,2*deg+1): #product powers of x and y values preemptively
        comp_mat[num_terms*i+j,] = np.array([comp_mat_x[i,a]*comp_mat_y[j,b] \
                                      for a in range(num_pts) for b in range(num_pts)])

        
#build the least squares matrix
row = 0
col = 0
for i in range(0,deg+1):
    for j in range(0,deg+1-i):
        for m in range(0,deg+1):
            for n in range(0,deg+1-m):
                #take advantage of comp_mat to not have to remultiply
                lsq_mat[row,col] = np.sum(comp_mat[num_terms*(i+m)+j+n,])
                col+=1
        lsq_vec[row,0] = np.dot(data_points,comp_mat[num_terms*i+j])
        row +=1 
        col = 0
        

 
#woo linear algebra     
coeffs = np.linalg.solve(lsq_mat,lsq_vec)

#Evaluate the polynomial at all inputs. This could definitely be optimized for
# ease on the eyes, but its not slow
def polyval(coeffs,deg,x_values,y_values):
    evals = np.zeros(len(x_values)*len(y_values))
    cur_point = 0
    for a in range(num_pts):
        for b in range(num_pts):
            col = 0
            for i in range(0,deg+1):
                for j in range(0,deg+1-i):
                    evals[cur_point]= evals[cur_point]+coeffs[col]*(comp_mat_x[i,a]*comp_mat_y[j,b])
                    col +=1
            cur_point += 1
    return evals                
                 
#Evaluate our polynomial at all the right spots
evals = polyval(coeffs,deg,x_values,y_values)


print('sum of squared error from data: ' + str(np.sum((data_points-evals)**2)))
print('sum of squared error from sine: ' + str(np.sum((sin_points-evals)**2)))


#Build square arrays for 3D picture
evals = np.reshape(evals, (num_pts,num_pts))
sin_points = np.reshape(sin_points, (num_pts,num_pts))
data_points = np.reshape(data_points, (num_pts, num_pts))

x_coords = np.zeros((num_pts,num_pts))
y_coords = np.zeros((num_pts,num_pts))
for i in range(num_pts):
    for j in range(num_pts):
        x_coords[i,j] = x_values[j]
        y_coords[i,j] = y_values[i]  


#put all this into one figure        
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d') 
ax.scatter(x_coords, y_coords, data_points, s=5, c= 'k')
ax.plot_wireframe(x_coords, y_coords, evals)       


