from __future__ import division
import numpy as np
import quadrature_nodes as qn
quad_order = 15
gll_x = qn.GLL_points(quad_order) #quadrature nodes
gll_w = qn.GLL_weights(quad_order,gll_x)


###################
#MINIMISING FUNCTION
###################
def min_approx1(function, (args)): #for some reason this is twice as slow as the built-in one :(
	left = -2
	right = 2
	linesearch_decrement = 0.5
	linesearch_tolerance = 1/50
	dx = 1
	Nx = 10
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

###################
#TAU FUNCTION
###################
def tau_first_order(alpha,i,v_array,x_array,dt):
	return 0.5*dt*alpha**2 + np.interp(x_array[i]-dt*alpha,x_array,v_array)

def tau_second_order(alpha,i,v_array,x_array,dt):
	return 0.5*dt*alpha**2 + 0.5*(np.interp(x_array[i]-dt*alpha+np.sqrt(dt)*noise,x_array,v_array) + np.interp(x_array[i]-dt*alpha-np.sqrt(dt)*noise,x_array,v_array))

###################
#RUNNING COST
###################
def F_global(x_array,m_array,sigma): #more effective running cost function
	#return (x_array-0.2)**2 #Carlini's no-game
	output = np.zeros(x_array.size)
	for i in range (0,output.size):
		output[i] = min(1.4,max(m_array[i],0.7))
	return output
	#return min(1.4,max(m_array,0.7))
#	tmp = mollify_array(m_array,sigma,x_array,gll_x,gll_w)
#	return 0.3*mollify_array(tmp,sigma,x_array,gll_x,gll_w)

##################
#TERMINAL COST
#################
def G(xi,m_array): #this is the final cost, and is a function of the entire distribution m and each point x_i
	#return -0.5*(xi+0.5)**2 * (1.5-xi)**2 #Carlini's original
	return -(xi*(1-xi))**2 #Gueant's game
	#return 0 #Carlini's no-game


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

###############THIS SHOULD GO AWAY
def beta(x_val,i,x_array):
	return np.maximum(0,1-abs(x_val-x_array[i])/(x_array[2]-x_array[1]))

def beta_array(array,i,x_array):
	output = np.empty(array.size)
	for j in range (0,array.size):
		output[j] = np.maximum(0,1-abs(array[j]-x_array[i])/(x_array[2]-x_array[1]))
	return output
