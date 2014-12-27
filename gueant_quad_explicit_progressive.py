# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 11:33:06 2014

@author: tower_000
"""
from __future__ import division
import numpy as np
import scipy.weave as weave
import matplotlib.pyplot as plt
from matplotlib import cm
import time as time
from mpl_toolkits.mplot3d import Axes3D
import input_functions as iF

#in this one we aim to not use so much fucking space

#INPUTS
dx = 1/100#0.0075*8#1/50 #these taken from Gueant's paper
dt = 1/5000
xmin = 0-0.2
xmax = 1+0.2
T = 1
Niter = 500 #maximum number of iterations
tolerance = 1e-4
sigma = 0.05
starting_sigma = 0.4
sigma_decrements = 3

#CRUNCH
sigma2s = np.linspace(starting_sigma,sigma,num=sigma_decrements,endpoint=True)**2
kMax=Niter
dx2 = dx**2
Nx = int(abs(xmax-xmin)/dx)
Nt = int(T/dt)
x = np.arange(xmin,xmax,dx)
t = np.arange(0,T,dt)
I = t.size
J = x.size
def index(i,j): 
	return int(j+J*i)

#INITIAL/TERMINAL CONDITIONS
m0 = iF.initial_distribution(x) #carlini's no-game
uT = iF.G(x,m0)
#Other stuff
fmax = max(abs(iF.F_global(x,m0,0))) #sort of

#check CFL
R = max(abs(uT)) + 2*T*fmax + sigma2s[-1] *max(abs(np.log(m0)))
print "Initial value for distribution:", R
KR = 0
crit = (sigma2s[0]+4*R)*(dt/dx2) + (dt/sigma2s[0])*KR*np.exp(2*R/sigma2s[0])
if crit > 1:
	print "CFL condition not satisfied, this will be weird."
	print dx2/(sigma2s[0]+4*R)
#initialise solution
u = np.empty((I*J))
v = np.empty((I*J))
u_old = np.empty((I*J)); 
#initial guess for v(0)
v_old = R*np.ones((I*J))

#initialise vectors to store l1, l2 and linfty norm errors/improvement in iterations
ul1 = -1*np.ones((Niter,1))
ul2 = -1*np.ones((Niter,1))
ulinfty = -1*np.ones((Niter,1))
ml1 = -1*np.ones((Niter,1))
ml2 = -1*np.ones((Niter,1))
mlinfty = -1*np.ones((Niter,1))
BIGZ = np.zeros(J-2)

print "Initialising done, now crunching."
timestart = time.time()
TOTAL_ITERATIONS = 0
for n in range (0,sigma2s.size):
	sigma2 = sigma2s[n]
	print "Computing solution for sigma^2 =", sigma2, "..."
	for k in range (0,Niter):
		#known terminal conditions
		titer = time.time()
		u[J*I-J:J*I] = uT
		#solve for u
		for i in range (I-2,-1,-1):
			tmp = (u[((i+1)*J):((i+1)*J+J)] - v_old[((i+1)*J):((i+1)*J+J)])/sigma2
			Fval = iF.F_global(x,np.exp(tmp),0)
			u[i*J] = u[(i+1)*J] + dt * ( sigma2/2 * ( 2*u[(i+1)*J+1] - 2*u[(i+1)*J] )/(dx2) - Fval[0] + 0.5*( max(0,(u[(i+1)*J+1]-u[(i+1)*J])/dx)**2 + min(0,(u[(i+1)*J]-u[(i+1)*J+1])/dx)**2 ) ) #first index
			u[i*J+J-1] = u[(i+1)*J+J-1] + dt * ( sigma2/2 * ( 2*u[(i+1)*J+J-2] - 2*u[(i+1)*J+J-1] )/(dx2) - Fval[-1] + 0.5*( max(0,(u[(i+1)*J+J-2]-u[(i+1)*J+J-1])/dx)**2 + min(0,(u[(i+1)*J+J-1]-u[(i+1)*J+J-2])/dx)**2 ) ) #last index
			u[(i*J+1):(i*J+J-1)] = u[((i+1)*J+1):((i+1)*J+J-1)] + dt * ( sigma2/2 * ( u[((i+1)*J+2):((i+1)*J+J)] + u[((i+1)*J):((i+1)*J+J-2)] - 2*u[((i+1)*J+1):((i+1)*J+J-1)] )/(dx2) - Fval[1:-1] + 0.5*( np.maximum(BIGZ,(u[((i+1)*J+2):((i+1)*J+J)]-u[((i+1)*J+1):((i+1)*J+J-1)])/dx)**2 + np.minimum(BIGZ,(u[((i+1)*J+1):((i+1)*J+J-1)]-u[((i+1)*J):((i+1)*J+J-2)])/dx)**2 ) ) #all other indices
		v[0:J] = np.copy(u[0:J]) - sigma2 * np.log(m0)
		#print v[0:J]
		#print ss
		for i in range (0,I-1):
			tmp = (u[(i*J):(i*J+J)] - v_old[(i*J):(i*J+J)])/sigma2
			Fval = iF.F_global(x,np.exp(tmp),0)
			v[(i+1)*J] = v[i*J] + dt * ( sigma2/2 * ( 2*v[i*J+1] - 2*v[i*J] )/(dx2) + Fval[0] - 0.5*( max(0,(v[i*J+1]-v[i*J])/dx)**2 + min(0,(v[i*J]-v[i*J+1])/dx)**2 ) ) #first index
			v[(i+1)*J+J-1] = v[i*J+J-1] + dt * ( sigma2/2 * ( 2*v[i*J+J-2] - 2*v[i*J+J-1] )/(dx2) + Fval[-1] - 0.5*( max(0,(v[i*J+J-2]-v[i*J+J-1])/dx)**2 + min(0,(v[i*J+J-1]-v[i*J+J-2])/dx)**2 ) ) #last index
			v[((i+1)*J+1):((i+1)*J+J-1)] = v[(i*J+1):(i*J+J-1)] + dt * ( sigma2/2 * ( v[(i*J+2):(i*J+J)] + v[(i*J):(i*J+J-2)] - 2*v[(i*J+1):(i*J+J-1)] )/(dx2) + Fval[1:-1] - 0.5*( np.minimum(BIGZ,(v[(i*J+2):(i*J+J)]-v[(i*J+1):(i*J+J-1)])/dx)**2 + np.maximum(BIGZ,(v[(i*J+1):(i*J+J-1)]-v[(i*J):(i*J+J-2)])/dx)**2 ) ) #all other indices

		#compute norms of stuff
		mchange = np.exp(( u-v )/(sigma2)) - np.exp(( u_old-v_old )/(sigma2))
		uchange = u-u_old
		ml1[TOTAL_ITERATIONS] = np.sum(abs(mchange))
		ml2[TOTAL_ITERATIONS] = np.sqrt(np.sum(abs(mchange)**2))
		mlinfty[TOTAL_ITERATIONS] = max(abs( mchange) ) 
		ul1[TOTAL_ITERATIONS] = np.sum(abs(uchange))
		ul2[TOTAL_ITERATIONS] = np.sqrt(np.sum(abs(uchange)**2))
		ulinfty[TOTAL_ITERATIONS] = max(abs( uchange) ) 
		if mlinfty[TOTAL_ITERATIONS] < tolerance:
			print "Converged for sigma =", sigma2s[n]**2 , "with final change" , mlinfty[TOTAL_ITERATIONS]
			break	

		#MIX IT UP NIGGA
		u_old = np.copy(u)
		v_old = np.copy(v)
		print "Iteration number", k+1, "in cycle number", n+1, "completed, used time", time.time()-titer, "with change", mlinfty[TOTAL_ITERATIONS]
		TOTAL_ITERATIONS += 1
	


print "Crunching over. Total elapsed time (in seconds):", time.time()-timestart
print "Used", TOTAL_ITERATIONS,"total iterations."

#resolve solutions into a mesh
m = np.exp((u-v)/(sigma2))
msoln = np.empty((I,J))
usoln = np.empty((I,J))
for i in range (0,I):
	for j in range (0,J):
		msoln[i,j] = m[index(i,j)]
		usoln[i,j] = u[index(i,j)]
		
#cut the change vectors with kMax
ml1 = ml1[:TOTAL_ITERATIONS]
ml2 = ml2[:TOTAL_ITERATIONS]
mlinfty = mlinfty[:TOTAL_ITERATIONS]
ul1 = ul1[:TOTAL_ITERATIONS]
ul2 = ul2[:TOTAL_ITERATIONS]
ulinfty = ulinfty[:TOTAL_ITERATIONS]

#init plotstuff
Xplot, Tplot = np.meshgrid(x,t)
#plot solution of m(x,t)
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111, projection='3d')
#ax1.plot_surface(Xplot,Tplot,msoln,rstride=5,cstride=5,cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax1.plot_surface(Xplot,Tplot,msoln,rstride=1,cstride=5,cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax1.set_xlabel('x')
ax1.set_ylabel('t')
ax1.set_zlabel('m(x,t)')
fig1.suptitle('Solution of density m(x,t)', fontsize=14)
#plot solution of u(x,t)
fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111, projection='3d')
ax2.plot_surface(Xplot,Tplot,usoln,rstride=10,cstride=10,cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax2.set_xlabel('x')
ax2.set_ylabel('t')
ax2.set_zlabel('u(x,t)')
fig2.suptitle('Solution of potential u(x,t)', fontsize=14)
#plot the norms of change on u
fig3 = plt.figure(3)
plt.plot(np.arange(1,TOTAL_ITERATIONS+1), np.log10(ml1), label="L1-norm")
plt.plot(np.arange(1,TOTAL_ITERATIONS+1), np.log10(ml2), label="L2-norm")
plt.plot(np.arange(1,TOTAL_ITERATIONS+1), np.log10(mlinfty), label="Linf-norm")
legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')
ax3 = fig3.add_subplot(111)
ax3.set_xlabel('Iteration number')
ax3.set_ylabel('Log10 of change')
fig3.suptitle('Convergence of u(x,t)', fontsize=14)
#plot the norms of change on m
fig4 = plt.figure(4)
plt.plot(np.arange(1,TOTAL_ITERATIONS+1), np.log10(ul1), label="L1-norm")
plt.plot(np.arange(1,TOTAL_ITERATIONS+1), np.log10(ul2), label="L2-norm")
plt.plot(np.arange(1,TOTAL_ITERATIONS+1), np.log10(ulinfty), label="Linf-norm")
legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')
ax4 = fig4.add_subplot(111)
ax4.set_xlabel('Iteration number')
ax4.set_ylabel('Log10 of change')
fig4.suptitle('Convergence of m(x,t)', fontsize=14)
plt.show()

