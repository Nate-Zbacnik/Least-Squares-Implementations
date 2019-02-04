# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 21:15:36 2019

@author: NATE
This program creates from the ground up a least squares polynomial approximation
of a sequence of points sampled on sine with a user defined amount of
measurement error. Number of points and degree of poly are also user defined.
"""
import math
import numpy as np
import random
import matplotlib.pyplot as plt
from IPython import get_ipython


k=2  #How far the approximation extends

#plt.close('all')


get_ipython().run_line_magic('matplotlib', 'qt') #put image in new window

#user inputs
deg = int(input('degree of polynomial: '))
num_pts = int(input('number of sample points: '))
std_dev = float(input('error in measurement: '))

#even distribution of x values
x_values = np.linspace(0,k*math.pi,num_pts)
x_values = np.array(x_values)

#introduce random error into sine of x
data_points = [np.sin(x)+random.uniform(-std_dev,std_dev) for x in  x_values]
data_points = np.array(data_points)

#actual values for plotting sine
sin_points = [np.sin(x) for x in  x_values]
sin_points = np.array(sin_points)

#initialize matrix and vector
lsq_mat = np.zeros((deg+1,deg+1))
lsq_vec = np.zeros((deg+1, 1))

#build the least squares matrix
for i in range(0,deg+1):
    for j in range(0,deg+1):
        lsq_mat[i,j] = np.sum(x_values**(i+j))
        lsq_vec[i,0] = np.dot(data_points,x_values**i)
   
#woo linear algebra     
coeffs = np.linalg.solve(lsq_mat,lsq_vec)

# coeffs need to be a list for the poly1d for some reason..
coeffs = [i[0] for i in coeffs]
coeffs.reverse() #whoops wrong way around
poly_approx = np.poly1d(coeffs)
        
#put all this into one figure
plt.figure(1)
datapts, = plt.plot(x_values, data_points, 'bo' , label="data")
graph, = plt.plot(x_values,sin_points, 'k', label = "sin(x)")
approx, = plt.plot(x_values, poly_approx(x_values), 'r', label = 'approximation')
plt.xlabel('x'), plt.ylabel('y')
plt.legend(handles=[datapts,graph,approx])
plt.show()

print('sum of squared error from data: ' + str(np.sum((data_points-poly_approx(x_values))**2)))
print('sum of squared error from sine: ' + str(np.sum((sin_points-poly_approx(x_values))**2)))