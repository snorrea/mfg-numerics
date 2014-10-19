# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 11:33:06 2014

@author: tower_000
"""
from __future__ import division
import numpy as np
import scipy.weave as weave
import matplotlib.pyplot as plt
import time as time
from mpl_toolkits.mplot3d import Axes3D

#in this one we aim to not use so much fucking space

#INPUTS
dx = 1/50 #these taken from Gueant's paper
dt = 1/2000 
xmin = 0
xmax = 1
T = 1
Niter = 10 #number of iterations

#CRUNCH
Nx = int(abs(xmax-xmin)/dx)
Nt = int(T/dt) 
I = int(Nt)
J = int(Nx)
x = np.arange(xmin,xmax+dx,dx)
t = np.arange(0,T+dt,dt)
def index(i,j): #this is a jolly source of errors, no more
	return int(j+J*(i))

#PLOT STUFF, need to just use linspace or something instead, this is homosexual and fake and wrong 
Xplot = np.zeros((I+1,J+1))
Tplot = np.zeros((I+1,J+1))
for i in range (0,I+1):
	Xplot[i,:] = x
for j in range (0,J+1):
	Tplot[:,j] = t

#input functions and constants
def f(xh,xi):
	return -min(1.4,max(xi,0.7))
sigma = 0.8
m0 = 1-0.2*np.cos(np.pi*x)
fmax = 2 #this has to be chosen empirically
uT = np.square(x*(1-x))

#initialise solution VECTORS WHY WOULD YOU USE MATRICES
u = np.empty((I+1)*(J+1))
v = np.empty((I+1)*(J+1))
u_old = u; #actually this one might not be used at all
v_old = v; #the improved guesses on v is the thing that keeps this method going

print "Initialising done, now crunching."
t = time.time()

#initial guess for v(0)
for i in range (0,I+1):
	for j in range (0,J+1):
		v_old[index(i,j)] = max(uT) + sigma**2 * max(np.log(m0)) + 2*T*fmax


#crunch
for k in range (0,Niter):
	#known terminal conditions
	titer = time.time()
	for j in range (0,J+1):
		u[index(I+1,j)] = uT[j]
	#solve u
	for i in range (I,0,-1):
		for j in range (0,J+1):
			if j==0:
				u[index(i,j)] = u[index(i+1,j)] + dt * ( sigma**2/2 * ( u[index(i+1,j+1)] + u[index(i+1,j+1)] - 2*u[index(i+1,j)] )/(dx**2) + f(x[j],np.exp( ( u[index(i+1,j)] - v_old[index(i+1,j)] )/(sigma**2) )) + 0.5*( max(0,(u[index(i+1,j+1)]-u[index(i+1,j)])/dx)**2 + min(0,(u[index(i+1,j)]-u[index(i+1,j+1)])/dx)**2 ) );
			elif j==J:
				u[index(i,j)] = u[index(i+1,j)] + dt * ( sigma**2/2 * ( u[index(i+1,j-1)] + u[index(i+1,j-1)] - 2*u[index(i+1,j)] )/(dx**2) + f(x[j],np.exp( ( u[index(i+1,j)] - v_old[index(i+1,j)] )/(sigma**2) )) + 0.5*( max(0,(u[index(i+1,j-1)]-u[index(i+1,j)])/dx)**2 + min(0,(u[index(i+1,j)]-u[index(i+1,j-1)])/dx)**2 ) );
			else:
				u[index(i,j)] = u[index(i+1,j)] + dt * ( sigma**2/2 * ( u[index(i+1,j+1)] + u[index(i+1,j-1)] - 2*u[index(i+1,j)] )/(dx**2) + f(x[j],np.exp( ( u[index(i+1,j)] - v_old[index(i+1,j)] )/(sigma**2) )) + 0.5*( max(0,(u[index(i+1,j+1)]-u[index(i+1,j)])/dx)**2 + min(0,(u[index(i+1,j)]-u[index(i+1,j-1)])/dx)**2 ) );
	#known initial conditions on v
	for j in range (0,J+1):
		v[index(0,j)] = u[index(0,j)] - sigma**2 * np.log(m0[j])

	
	#solve for v
	for i in range (1,I+1):
		for j in range (0,J): #these should be vectorised 
			if j==0: #j
				v[index(i,j)] = v[index(i+1,j)] + dt * ( sigma**2/2 * ( v[index(i+1,j+1)] + v[index(i+1,j+1)] - 2*v[index(i+1,j)] )/(dx**2) - f(x[j],np.exp( ( u[index(i+1,j)] - v_old[index(i+1,j)] )/(sigma**2) )) + 0.5*( max(0,(v[index(i+1,j+1)]-v[index(i+1,j)])/dx)**2 + min(0,(v[index(i+1,j)]-v[index(i+1,j+1)])/dx)**2 ) );
			elif j==J:
				v[index(i,j)] = v[index(i+1,j)] + dt * ( sigma**2/2 * ( v[index(i+1,j-1)] + v[index(i+1,j-1)] - 2*v[index(i+1,j)] )/(dx**2) - f(x[j],np.exp( ( u[index(i+1,j)] - v_old[index(i+1,j)] )/(sigma**2) )) + 0.5*( max(0,(v[index(i+1,j-1)]-v[index(i+1,j)])/dx)**2 + min(0,(v[index(i+1,j)]-v[index(i+1,j-1)])/dx)**2 ) );
			else: #source of error could be the inputs of the f function in terms of u
				v[index(i,j)] = v[index(i+1,j)] + dt * ( sigma**2/2 * ( v[index(i+1,j+1)] + v[index(i+1,j-1)] - 2*v[index(i+1,j)] )/(dx**2) - f(x[j],np.exp( ( u[index(i+1,j)] - v_old[index(i+1,j)] )/(sigma**2) )) + 0.5*( max(0,(v[index(i+1,j+1)]-v[index(i+1,j)])/dx)**2 + min(0,(v[index(i+1,j)]-v[index(i+1,j-1)])/dx)**2 ) );


	#MIX IT UP NIGGA
	u_old = u #as stated, this is probably not used
	v_old = v
	print "Iteration number", k+1, "completed, used time", time.time()-titer



print "Crunching over. Total elapsed time (in seconds):", time.time()-t

#resolve solutions into a mesh
#v = v[:,1:Niter+1] #remove initial condition
m = np.exp((u-v)/(sigma**2))
msoln = np.empty((I+1,J+1))
usoln = msoln
for i in range (0,I+1):
	for j in range (0,J+1):
		msoln[i,j] = m[index(i,j)]
		usoln[i,j] = u[index(i,j)]

#shit attempt at plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#print Xplot, Tplot, msoln
ax.plot_wireframe(Xplot,Tplot,msoln)
plt.show()



