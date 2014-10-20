
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
Niter = 50 #maximum number of iterations
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

###input functions and constants
#constants for g
c0 = 1
c1 = 1
c2 = 1
#constant for f
beta = 0.7
#the rest
sigma2 = 0.8**2
m0 = 1-0.2*np.cos(np.pi*x) #gueant's original
def cost(a):
	return a**2/2
def g(t,x,m):
	return c0*x/(c1+c2*m)
def p(t):	#price of electricity
	return 0.5
def f(t,x):
	return p(t)*(1-beta*x)
vT = np.zeros(J)

#initialise solution VECTORS WHY WOULD YOU USE MATRICES
v = np.empty((I*J)) #potential
m = np.empty((I*J)) #distribution
a = np.empty((I*J)) #control
v_old = np.empty((I*J)); 
m_old = np.empty((I*J)); 
a_old = np.empty((I*J)); 
print "Initialising done, now crunching."
t = time.time()

#initial value for a
a_old = np.random([I*J])

for k in range (0,Niter):
	titer = time.time()
	#compute v
	v[J*I-J-2:J*I-1] = vT
	for i in range (I-2,-1,-1):
		for j in range (0,J):
			if j==0:
				1
			elif j==J-1:
				1
			else:
				











