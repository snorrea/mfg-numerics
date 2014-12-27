from __future__ import division
import numpy as np
import quadrature_nodes as qn
quad_order = 15
gll_x = qn.GLL_points(quad_order) #quadrature nodes
gll_w = qn.GLL_weights(quad_order,gll_x)


###################
#MINIMISING FUNCTION
###################
def find_minimum(function, (args)): #for some reason this is twice as slow as the built-in one :(
	left = -2
	right = 2
	linesearch_decrement = 0.5
	linesearch_tolerance = 1/50
	dx = 1
	Nx = 20
	xpts = np.linspace(left,right,Nx)
	fpts = np.empty((xpts.size,1))
	fpts = function(xpts,*args)
	x0 = xpts[np.argmin(fpts)] #choose smallest
	h = dx/2
	while True:
		if function(x0+h,*args) < function(x0-h,*args): #go right
			x0 += h
			h = h*linesearch_decrement
		elif function(x0-h,*args) < function(x0+h,*args): #go left
			x0 += -h
			h = h*linesearch_decrement
		elif h > linesearch_tolerance:
			h = h*linesearch_decrement
		else:
			break
	#print "Time to minimise:",time.time()-t1
	return function(x0,*args)

def line_search(function, (args),dx,x0): #for some reason this is twice as slow as the built-in one :(
	#here dx is the distance between tested sample points, and x0 is initial guess
	linesearch_decrement = 0.5
	linesearch_tolerance = 1/50
	h = dx/2
	while True:
		if function(x0+h,*args) < function(x0-h,*args): #go right
			x0 += h
			h = h*linesearch_decrement
		elif function(x0-h,*args) < function(x0+h,*args): #go left
			x0 += -h
			h = h*linesearch_decrement
		elif h > linesearch_tolerance:
			h = h*linesearch_decrement
		else:
			break
	#print "Time to minimise:",time.time()-t1
	return function(x0,*args)

def scatter_search(function, (args),dx,x0,N,k): #here k is the number of laps going
	x_naught = x0
	dex = dx
	for i in range (0,k):
		xpts = np.linspace(x_naught-dex,x_naught+dex,N)
		fpts = function(xpts,*args)
		if i!=k-1:
			x_naught = xpts[np.argmin(fpts)]
			dex = xpts[2]-xpts[1]
	return min(fpts)

def scatter_search2(function, (args),dx,x0,N,k,alpha): #here k is the number of laps going
	x_naught = x0
	dex = dx
	for i in range (0,k):
		xpts = np.linspace(x_naught-dex,x_naught+dex,N)
		fpts = function(xpts,*args)
		if i!=k-1:
			x_naught = xpts[np.argmin(fpts)]
			dex = xpts[2]-xpts[1]
			N = alpha*N
	return min(fpts)


###################
#TAU FUNCTION
###################
def tau_first_order(alpha,i,v_array,x_array,dt):
	return 0.5*dt*alpha**2 + np.interp(x_array[i]-dt*alpha,x_array,v_array)

def tau_second_order(alpha,i,v_array,x_array,dt,noise):
	return 0.5*dt*alpha**2 + 0.5*(np.interp(x_array[i]-dt*alpha+np.sqrt(dt)*noise,x_array,v_array) + np.interp(x_array[i]-dt*alpha-np.sqrt(dt)*noise,x_array,v_array))

###################
#RUNNING COST
###################
def F_global(x_array,m_array,sigma): #more effective running cost function
	#return (x_array-0.2)**2 #Carlini's no-game
	#return np.minimum(1.4*np.ones(x_array.size),np.maximum(m_array,0.7*np.ones(x_array.size))) #Gueant's game
	#return np.minimum(1.4*np.ones(m_array.size),np.maximum(m_array,0.7*np.ones(m_array.size)))
	#tmp = mollify_array(m_array,sigma,x_array,gll_x,gll_w)
	#tmp = mollify_array(tmp,sigma,x_array,gll_x,gll_w)
	#return 0.05*mollify_array(tmp,sigma,x_array,gll_x,gll_w)
	#return 0.03*tmp
	return m_array #shyness game
	#return 1/max(m_array+0.0001)*m_array #scaled shyness game
	#return np.zeros(x_array.size) #no-game

##################
#TERMINAL COST
##################
def G(x_array,m_array): #this is the final cost, and is a function of the entire distribution m and each point x_i
	#return 0.5*(x_array+0.5)**2 * (1.5-x_array)**2 #Carlini's original
	#return 0.1*(x_array*(1-x_array))**2 #Gueant's game
	return -((x_array+0.2)*(1.2-x_array))**4 #Shyness game
	#return 0*x_array #Carlini's no-game 

##################
#TERMINAL COST
##################
def initial_distribution(x):
	#return 1-0.2*np.cos(np.pi*x) #gueant's original
	#return np.exp(-(x-0.75)**2/0.1**2) #carlini's no-game
	return np.exp(-(x-0.5)**2/0.1**2) #shyness game

###################
#AUXILIARY FUNCTIONS
###################
def mollifier(x_val): #Evans' mollifier
	if abs(x_val) < 1:
		return np.exp(1/(x_val**2 - 1))/0.443994
	else:
		return 0
def mollify_array(array,epsilon,x_array,gll_x,gll_w): 
	output = np.zeros((array.size))
	for k in range (0,array.size):
		for j in range (0,gll_x.size):
			output[k] += gll_w[j]*mollifier(gll_x[j])*np.interp(x_array[k]-epsilon*gll_x[j],x_array,array)
	return output
def restrain(trajectory,x_array):
	for i in range(0,trajectory.size):
		if trajectory[i]<x_array[1]:
			trajectory[i] = x_array[1]
		elif trajectory[i]>x_array[x_array.size-1]:
			trajectory[i] = x_array[x_array.size-1]
	return trajectory

###############THIS SHOULD GO AWAY
def beta(x_val,i,x_array):
	return np.maximum(0,1-abs(x_val-x_array[i])/(x_array[2]-x_array[1]))

def beta_array(array,i,x_array):
	output = np.empty(array.size)
	for j in range (0,array.size):
		output[j] = np.maximum(0,1-abs(array[j]-x_array[i])/(x_array[2]-x_array[1]))
	return output

def beta_left(z,x_array,dx,index):
	return (x_array[index+1]-z)/dx

def beta_right(z,x_array,dx,index):
	return (z-x_array[index])/dx

	
