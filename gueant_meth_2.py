# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 11:33:06 2014

@author: tower_000
"""
from __future__ import division
import numpy as np
import scipy.weave as weave
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#INPUTS
dx = 1/50 #these taken from Gueant's paper
dt = 1/2000 
xmin = 0
xmax = 1
T = 1
Niter = 35 #number of iterations

#CRUNCH
Nx = int(abs(xmax-xmin)/dx)
Nt = int(T/dt) 
I = int(Nt)
J = int(Nx)
x = np.arange(xmin,xmax+dx,dx)
t = np.arange(0,T+dt,dt)
def index(i,j): #this is a jolly source of errors, no more
	return int(j+J*(i))

#PLOT STUFF
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
Xplot = np.zeros((J,J))
Tplot = np.zeros((I,I))
for i in range (0,J):
	for j in range (0,J):
		Xplot[i,j] = x[j]
for i in range (0,I):
	for j in range (0,I):
		Tplot[i,j] = t[i]

#input functions and constants
def f(xh,xi):
	return -min(1.4,max(xi,0.7))
sigma = 0.8
m0 = 1-0.2*np.cos(np.pi*x)
fmax = 2 #this has to be chosen empirically
uT = np.square(x*(1-x))

#initialise solution matrices
u = np.empty([(I+1)*(J+1),Niter])
v = np.empty([(I+1)*(J+1),Niter+1]) #to allow for one extra iteration

#initial guess for v(0)
for i in range (0,I):
	for j in range (0,J):
		v[index(i,j),0] = max(uT) + sigma**2 * max(np.log(m0)) + 2*T*fmax
		#print i,j,index(i,j)
#weave.blitz("for i in range (0,I): for j in range (0,J):" + " v[index(i,j),0] = max(uT) + sigma**2 * max(np.log(m0))" + "+ (2*T)*fmax")


#crunch
for k in range (0,Niter):
	#known terminal conditions
	for j in range (0,J):
		u[index(I,j),k] = uT[j]
	#solve u
	for i in range (I-1,0,-1):
		for j in range (0,J):
			if j==0:
				u[index(i,j),k] = u[index(i+1,j),k] + dt * ( sigma**2/2 * ( u[index(i+1,j+1),k] + u[index(i+1,j+1),k] - 2*u[index(i+1,j),k] )/(dx**2) + f(x[j],np.exp( ( u[index(i+1,j),k] - v[index(i+1,j),k] )/(sigma**2) )) + 0.5*( max(0,(u[index(i+1,j+1),k]-u[index(i+1,j),k])/dx)**2 + min(0,(u[index(i+1,j),k]-u[index(i+1,j+1),k])/dx)**2 ) );
			elif j==J-1:
				u[index(i,j),k] = u[index(i+1,j),k] + dt * ( sigma**2/2 * ( u[index(i+1,j-1),k] + u[index(i+1,j-1),k] - 2*u[index(i+1,j),k] )/(dx**2) + f(x[j],np.exp( ( u[index(i+1,j),k] - v[index(i+1,j),k] )/(sigma**2) )) + 0.5*( max(0,(u[index(i+1,j-1),k]-u[index(i+1,j),k])/dx)**2 + min(0,(u[index(i+1,j),k]-u[index(i+1,j-1),k])/dx)**2 ) );
			else:
				u[index(i,j),k] = u[index(i+1,j),k] + dt * ( sigma**2/2 * ( u[index(i+1,j+1),k] + u[index(i+1,j-1),k] - 2*u[index(i+1,j),k] )/(dx**2) + f(x[j],np.exp( ( u[index(i+1,j),k] - v[index(i+1,j),k] )/(sigma**2) )) + 0.5*( max(0,(u[index(i+1,j+1),k]-u[index(i+1,j),k])/dx)**2 + min(0,(u[index(i+1,j),k]-u[index(i+1,j-1),k])/dx)**2 ) );
	#known initial conditions on v
	for j in range (0,J):
		v[index(0,j),k+1] = u[index(0,j),k] - sigma**2 * np.log(m0[j])

	
	#solve for v
	for i in range (1,I):
		for j in range (0,J): #these should be vectorised 
			if j==0: #j
				v[index(i,j),k+1] = v[index(i+1,j),k+1] + dt * ( sigma**2/2 * ( v[index(i+1,j+1),k+1] + v[index(i+1,j+1),k+1] - 2*v[index(i+1,j),k+1] )/(dx**2) - f(x[j],np.exp( ( u[index(i+1,j),k] - v[index(i+1,j),k] )/(sigma**2) )) + 0.5*( max(0,(v[index(i+1,j+1),k+1]-v[index(i+1,j),k+1])/dx)**2 + min(0,(v[index(i+1,j),k+1]-v[index(i+1,j+1),k+1])/dx)**2 ) );
			elif j==J-1:
				v[index(i,j),k+1] = v[index(i+1,j),k+1] + dt * ( sigma**2/2 * ( v[index(i+1,j-1),k+1] + v[index(i+1,j-1),k+1] - 2*v[index(i+1,j),k+1] )/(dx**2) - f(x[j],np.exp( ( u[index(i+1,j),k] - v[index(i+1,j),k] )/(sigma**2) )) + 0.5*( max(0,(v[index(i+1,j-1),k+1]-v[index(i+1,j),k+1])/dx)**2 + min(0,(v[index(i+1,j),k+1]-v[index(i+1,j-1),k+1])/dx)**2 ) );
			else: #source of error could be the inputs of the f function in terms of u
				v[index(i,j),k+1] = v[index(i+1,j),k+1] + dt * ( sigma**2/2 * ( v[index(i+1,j+1),k+1] + v[index(i+1,j-1),k+1] - 2*v[index(i+1,j),k+1] )/(dx**2) - f(x[j],np.exp( ( u[index(i+1,j),k] - v[index(i+1,j),k] )/(sigma**2) )) + 0.5*( max(0,(v[index(i+1,j+1),k+1]-v[index(i+1,j),k+1])/dx)**2 + min(0,(v[index(i+1,j),k+1]-v[index(i+1,j-1),k+1])/dx)**2 ) );

#resolve solutions into a mesh
v = v[:,1:Niter+1] #remove initial condition
m = np.exp((u-v)/(sigma**2))
msoln = np.empty((I,J))
usoln = np.empty((I,J))
for i in range (0,I):
	for j in range (0,J):
		msoln[i,j] = m[index(i,j),Niter-1]
		usoln[i,j] = u[index(i,j),Niter-1]

#shit attempt at plotting
print Xplot.shape,Tplot.shape,msoln.shape
ax.plot_surface(Xplot,Tplot,m)
plt.show()