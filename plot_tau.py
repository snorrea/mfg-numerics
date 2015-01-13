from __future__ import division
import numpy as np
import scipy.weave as weave
import matplotlib.pyplot as plt
import time as time
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize as minimize
from matplotlib import cm
import quadrature_nodes as qn
import input_functions as iF
import sys

#INPUTS
dx = 0.05#075
dt = 2*dx
xmin = 0
xmax = 1
x = np.arange(xmin,xmax,dx)
I = x.size
#i = I/1.5
alphas = np.arange(-8,8,dx)
K = alphas.size
#CRUNCH
#v_sample = np.cos(8*np.pi*x) #get a sample F function
v_sample = (x-0.2)**2
#taus1 = iF.tau_first_order(alphas,i,v_sample,x,dt) #tau_first_order(alpha,i,v_array,x_array,dt)
#taus2 = np.interp(x[i]-dt*alphas,x,v_sample)

taus1soln = np.empty((K,I))
taus2soln = np.empty((K,I))
taus3soln = np.empty((K,I))
taus1minima_x = np.empty(I)
taus1minima_y = np.empty(I)
for i in range(0,I):
	#taus1soln[(i*K):((i+1)*K),:] = iF.tau_first_order(alphas,i,v_sample,x,dt) 
	#taus2soln[(i*K):((i+1)*K),:] = np.interp(x[i]-dt*alphas,x,v_sample)
	tmp = iF.tau_first_order(alphas,i,v_sample,x,dt)
	taus1minima_x[i] = alphas[np.argmin(tmp)]
	taus1minima_y[i] = tmp[np.argmin(tmp)]
	taus1soln[:,i] = tmp
	taus2soln[:,i] = np.interp(x[i]-dt*alphas,x,v_sample)
	#taus3soln[:,i] = np.interp(x[i]-dt*alphas,x,v_sample) - v_sample
	

Aplot, Xplot = np.meshgrid(x,alphas)
#
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot_surface(Aplot,Xplot,taus1soln,rstride=5,cstride=5,cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax1.plot(taus1minima_x,taus1minima_y,iF.tau_first_order(taus1minima_x,np.arange(0,I),v_sample,x,dt))
#
fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111, projection='3d')
ax2.plot_surface(Aplot,Xplot,taus2soln,rstride=5,cstride=5,cmap=cm.coolwarm,linewidth=0, antialiased=False)
#
fig3 = plt.figure(3)
plt.plot(alphas,np.interp(x[10]-dt*alphas,x,v_sample))
plt.show()
