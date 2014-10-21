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
Niter = 500 #maximum number of iterations
tolerance = 1e-6

#CRUNCH
dx2 = dx**2
Nx = int(abs(xmax-xmin)/dx)
Nt = int(T/dt) 
I = int(Nt)
J = int(Nx)
x = np.arange(xmin,xmax+dx,dx)
t = np.arange(0,T+dt,dt)
def index(i,j): #this is a jolly source of errors, no more, probably still
	return int(j+(J)*i)

#input functions and constants
def f(xh,xi):
	#return -min(1.4,max(xi,0.7)) #gueant's original
	#return xh*(1-xh)*xi #my thing which will explode but didn't
	#return -(xi)
	#return -100*np.log(xi)
	#return -0.1 + (xh*(1-xh))**2/(1+4*xi)**(1.5)
	#return -xi - abs(xh-0.3)**2
	#return -xi - 2*abs(xh-0.3)**2/(xi+1)
	return -2*(xi**2)*abs(xh-0.5)**2
sigma2 = 0.8**2
m0 = 1-0.2*np.cos(np.pi*x) #gueant's original
fmax = 10 #this has to be chosen empirically
uT = np.square(x*(1-x))*0 #gueant's original
#uT = abs(np.sin(np.pi*x)*np.cos(np.pi*x))

#initialise solution VECTORS WHY WOULD YOU USE MATRICES
u = np.empty((I*J))
v = np.empty((I*J))
u_old = np.empty((I*J)); #actually this one might not be used at all
v_old = np.empty((I*J)); #the improved guesses on v is the thing that keeps this method going
print "Initialising done, now crunching."
t = time.time()

#initial guess for v(0)
for i in range (0,I):
	for j in range (0,J):
		v_old[index(i,j)] = max(abs(uT)) + sigma2 * max(abs(np.log(m0))) + 2*T*fmax


#crunch
for k in range (0,Niter):
	#known terminal conditions
	titer = time.time()
	u[J*I-J-2:J*I-1] = uT #this is wrong wrong wrong what the fuck is wrong
	#j+(J-1)*i
	#solve u
	for i in range (I-2,-1,-1):
		for j in range (0,J):
			if j==0:
				u[index(i,j)] = u[index(i+1,j)] + dt * ( sigma2/2 * ( u[index(i+1,j+1)] + u[index(i+1,j+1)] - 2*u[index(i+1,j)] )/(dx2) + f(x[j],np.exp( ( u[index(i+1,j)] - v_old[index(i+1,j)] )/(sigma2) )) + 0.5*( max(0,(u[index(i+1,j+1)]-u[index(i+1,j)])/dx)**2 + min(0,(u[index(i+1,j)]-u[index(i+1,j+1)])/dx)**2 ) )
			elif j==J-1:
				u[index(i,j)] = u[index(i+1,j)] + dt * ( sigma2/2 * ( u[index(i+1,j-1)] + u[index(i+1,j-1)] - 2*u[index(i+1,j)] )/(dx2) + f(x[j],np.exp( ( u[index(i+1,j)] - v_old[index(i+1,j)] )/(sigma2) )) + 0.5*( max(0,(u[index(i+1,j-1)]-u[index(i+1,j)])/dx)**2 + min(0,(u[index(i+1,j)]-u[index(i+1,j-1)])/dx)**2 ) )
			else:
				u[index(i,j)] = u[index(i+1,j)] + dt * ( sigma2/2 * ( u[index(i+1,j+1)] + u[index(i+1,j-1)] - 2*u[index(i+1,j)] )/(dx2) + f(x[j],np.exp( ( u[index(i+1,j)] - v_old[index(i+1,j)] )/(sigma2) )) + 0.5*( max(0,(u[index(i+1,j+1)]-u[index(i+1,j)])/dx)**2 + min(0,(u[index(i+1,j)]-u[index(i+1,j-1)])/dx)**2 ) )

	#known initial conditions on v
	#print m0.shape
	v[0:J+1] = np.copy(u[0:(J+1)]) - sigma2 * np.log(m0)
	
	#solve for v
	for i in range (0,I-1):
		for j in range (0,J):
			if j==0: #j
				v[index(i+1,j)] = v[index(i,j)] + dt * ( sigma2/2 * ( v[index(i,j+1)] + v[index(i,j+1)] - 2*v[index(i,j)] )/(dx2) - f(x[j],np.exp( ( u[index(i,j)] - v_old[index(i,j)] )/(sigma2) )) - 0.5*( min(0,(v[index(i,j+1)]-v[index(i,j)])/dx)**2 + max(0,(v[index(i,j)]-v[index(i,j+1)])/dx)**2 ) )
			elif j==J-1:
				v[index(i+1,j)] = v[index(i,j)] + dt * ( sigma2/2 * ( v[index(i,j-1)] + v[index(i,j-1)] - 2*v[index(i,j)] )/(dx2) - f(x[j],np.exp( ( u[index(i,j)] - v_old[index(i,j)] )/(sigma2) )) - 0.5*( min(0,(v[index(i,j-1)]-v[index(i,j)])/dx)**2 + max(0,(v[index(i,j)]-v[index(i,j-1)])/dx)**2 ) )
			else: #source of error could be the inputs of the f function in terms of u
				v[index(i+1,j)] = v[index(i,j)] + dt * ( sigma2/2 * ( v[index(i,j+1)] + v[index(i,j-1)] - 2*v[index(i,j)] )/(dx2) - f(x[j],np.exp( ( u[index(i,j)] - v_old[index(i,j)] )/(sigma2) )) - 0.5*( min(0,(v[index(i,j+1)]-v[index(i,j)])/dx)**2 + max(0,(v[index(i,j)]-v[index(i,j-1)])/dx)**2 ) )
	
	#change in stuff check; this indicates that something is very very wrong, but WHAT THE FUCK
	deltaeverything = max(abs( np.exp(( u-v )/(sigma2)) - np.exp(( u_old-v_old )/(sigma2))) )
	if deltaeverything < tolerance:
		print "Method converged with final change" , deltaeverything
		break

	#MIX IT UP NIGGA
	u_old = np.copy(u)
	v_old = np.copy(v)
	print "Iteration number", k+1, "completed, used time", time.time()-titer, "with change", deltaeverything


print "Crunching over. Total elapsed time (in seconds):", time.time()-t

#resolve solutions into a mesh
m = np.exp((u-v)/(sigma2))
msoln = np.empty((I,J))
usoln = np.empty((I,J))
for i in range (0,I):
	for j in range (0,J):
		msoln[i,j] = m[index(i,j)]
		usoln[i,j] = u[index(i,j)]
#shit attempt at plotting
xi = np.linspace(xmin,xmax,Nx)
ti = np.linspace(0,T,Nt)
Xplot, Tplot = np.meshgrid(xi,ti)
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot_wireframe(Xplot,Tplot,msoln,rstride=15,cstride=15)
ax1.set_xlabel('x')
ax1.set_ylabel('t')
ax1.set_zlabel('m(x,t)')
plt.show()
#fig2 = plt.figure()
#ax2 = fig1.add_subplot(111, projection='3d')
#ax2.plot_wireframe(Xplot,Tplot,usoln,rstride=15,cstride=5)
#plt.show()


