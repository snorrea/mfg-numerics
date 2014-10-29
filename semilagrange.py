from __future__ import division
import numpy as np
import scipy.weave as weave
import matplotlib.pyplot as plt
import time as time
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize as minimize

#in this one we aim to not use so much fucking space

#INPUTS
dx = 1/10 #these taken from Gueant's paper
dt = 1/20
xmin = 0
xmax = 1
T = 1
Niter = 5 #maximum number of iterations
tolerance = 1e-6
epsilon = 0.003 #for use in convolution thing

#CRUNCH
dx2 = dx**2
Nx = int(abs(xmax-xmin)/dx)
Nt = int(T/dt) 
I = int(Nx)+1 #space
K = int(Nt)+1 #time
x = np.arange(xmin,xmax+dx,dx)
t = np.arange(0,T+dt,dt)
def index(i,k): #this is a jolly source of errors, no more, probably still
	return int(i+(I)*k)
def convolution(eps,x): #this is the convolution function used in the paper, use like np.convolve(u,v)
	return 1/eps * 1/np.sqrt(2*np.pi) * exp(-(x/eps)**2/2)
def beta(x,i):
	return max(0,1-abs(x-x(i))/dx)
def tau(alpha,i,k,v_array,x_array): #the function to be minimised
	tmp = 0.5*dt*alpha**2
	for j in range (0,I):
		tmp = tmp + v_array[index(j,k+1)]*beta(x_array[i]-dt*alpha,j)
	return tmp


#FUNCTIONS
def G(xi,m_array): #this is the final cost, and is a function of the entire distribution m and each point x_i
	return -0.5*(xi+0.5)**2 * (1.5-xi)**2 #Carlini's original

def F(xi,m_array): #this is the running cost
	return 1

sigma2 = 0.8**2
m0 = 1-0.2*np.cos(np.pi*x) #gueant's original
v = np.empty((I*K)) #potential
m = np.empty((I*K)) #distribution
v_old = np.empty((I*K))
m_old = np.empty((I*K)) #I guess, but this is a fucking nasty thing

for k in range (0,K):
	m_old[(I*k):(I*k+I)] = np.copy(m0)

print m_old
#minimize(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None)
#minimize(tau,0,args=(i,k,v_array,x_array))
#minimize(tau,0,args=(i,k,m0,x))
minimize(tau,0,args=(1,2,m0,x))

#for n in range (0,Niter):
#	titer = time.time()
#	v[(I*K-I-1):(I*K)] = G(x,m_old) #think this works
#	for k in range (K-1,-1,-1):
#		for i in range (0,I):
#		v(index(i,k)) = 


