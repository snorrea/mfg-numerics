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
def beta(x,i,x_array):
	return max(0,1-abs(x-x_array[i])/dx)
def tau(alpha,i,k,v_array,x_array): #the function to be minimised
	tmp = 0.5*dt*alpha**2
	for j in range (0,I):
		tmp = tmp + v_array[j]*beta(x_array[i]-dt*alpha,j,x_array)
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
m_old = np.empty((I*K))
v_grad = np.empty((I*K))
#initial guess on the distribution
for k in range (0,K):
	m_old[(I*k):(I*k+I)] = np.copy(m0)

#HOW TO MINIMIZE
#minimize(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None)
#minimize(tau,0,args=(i,k,v_array,x_array))
#minimize(tau,0,args=(i,k,m_old,x))
#tmp = minimize(tau,0,args=(1,2,m_old,x))
#print tmp.fun #this returns the minimum value of the function 


for n in range (0,Niter):
	titer = time.time()
	v[(I*K-I):(I*K)] = G(x,m_old) #think this works
	for k in range (K-2,-1,-1):
		v_tmp = np.copy(v[((k+1)*I):((k+1)*I+I)])
		#print v_tmp
		for i in range (0,I):
			#print i,k
			tmp = minimize(tau,0,args=(i,k,v_tmp,x)) #the previous timestep
			v[index(i,k)] = dt*F(x[i],m_old) + tmp.fun
	#compute the control based on the convoluted potential v
	#convolute the potential in space
		#let's skip that for now

	#find the gradient in space, which happens to be the fucking control/state/whatever
	for k in range (0,K-1):
		#print I*k+I
		#v_grad[(I*k):(I*k+I)] = np.gradient(v[(I+k):(I*k+I)],dx)
		v_grad[(I*k)] = (v[I*k]-v[I*k+1])/dx
		v_grad[(I*k)+I] = (v[I*k+I-1]-v[I*k+I])/(dx)
		v_grad[(I*k+1):(I*k+I-1)] = (v[(I*k):(I*k+I-2)]-v[(I*k+2):(I*k+I)])/(2*dx)
#do it manually because someone is a fucking moron

	
	#initial condition on m is already set, compute the rest of them
	for k in range(0,K-1):
		for i in range (0,I):
			m[index(i,k+1)] = 0
			for j in range (0,I):
				m[index(i,k+1)] += beta(x[j]-dt*v_grad[index(j,k)],i,x)*m[index(i,k)]



