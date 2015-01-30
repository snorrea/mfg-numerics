from __future__ import division
import numpy as np
import quadrature_nodes as qn
from scipy.optimize import minimize as minimize
quad_order = 15
gll_x = qn.GLL_points(quad_order) #quadrature nodes
gll_w = qn.GLL_weights(quad_order,gll_x)


###################
#MINIMISING FUNCTIONS
###################
def scatter_search(function, (args),dx,x0,N,k): #here k is the number of laps going
	x_naught = x0
	dex = dx
	for i in range (0,k):
		xpts = np.linspace(x_naught-dex,x_naught+dex,N)
		fpts = function(xpts,*args)
		if i!=k-1:
			x_naught = xpts[np.argmin(fpts)]
			dex = xpts[2]-xpts[1]
	return xpts[np.argmin(fpts)],min(fpts)

def scatter_search2(function, (args),dx,x0,N,k,alpha): #here k is the number of laps going, also this is bad
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
#POLICY ITERATION FUNCTIONS
###################

def best_control(x_array,u_array,m_array,dt,dx,time): #let's be happy with this for a time... but this is probably seriously flawed
	Radius = 2
	Nx = x_array.size
	dx2 = dx**2
	alphas = np.linspace(-Radius,Radius,2*round(Radius/(dx*0.1))+1)
	#alphas = np.linspace(0,Radius,2*Radius/(dx)+1)
	output = np.empty(x_array.size)
	for index in range (0,Nx):
		#stuff = hamiltonian(alphas,x_array,u_array,m_array,dt,dx,time,index)
		#output[index] = alphas[np.argmin(stuff)]
		sigma2 = Sigma_global(time,x_array[index],alphas,m_array[index])**2
		L_var = L_global(time,x_array[index],alphas,m_array[index])
		#print L_var, x_array[index], (x_array[index]-0.2)**2
		#print ss
		movement = f_global(time,x_array[index],alphas)
		if index==0: #topmost stuff is correct
			stuff = (abs(movement)/dx - sigma2/dx2)*(u_array[0] - u_array[1]) + L_var
			output[index] = alphas[np.argmin(stuff)]
		elif index==Nx-1:
			stuff = (abs(movement)/dx - sigma2/dx2)*(u_array[-1] - u_array[-2]) + L_var
			output[index] = alphas[np.argmin(stuff)]
		else:
			BIGZERO = np.zeros(alphas.size)
			stuff = u_array[index]*(abs(movement)/dx - sigma2/dx2) + u_array[index+1]*(sigma2/(2*dx2) + np.minimum(movement,BIGZERO)/dx) + u_array[index-1]*(sigma2/(2*dx2) - np.maximum(movement,BIGZERO)/dx) + L_var
			output[index] = alphas[np.argmin(stuff)]
	return output

def hamiltonian(alphas,x_array,u_array,m_array,dt,dx,time,index):
	BIGZERO = np.zeros(alphas.size)
	sigma2 = Sigma_local(time,x_array[index],alphas,m_array[index])**2
	movement = f_global(time,x_array[index],alphas)
	L_var = L_global(time,x_array[index],alphas,m_array[index])
	dx2 = dx**2
	if index==0: #topmost stuff is correct
		tmp = (abs(movement)/dx - sigma2/dx2)*(u_array[0] - u_array[1]) + L_var
	elif index==x_array.size-1:
		tmp = (abs(movement)/dx - sigma2/dx2)*(u_array[-1] - u_array[-2]) + L_var
	else:
		tmp = u_array[index]*(abs(movement)/dx - sigma2/dx2) + u_array[index+1]*(sigma2/(2*dx2) + np.minimum(movement,BIGZERO)/dx) + u_array[index-1]*(sigma2/(2*dx2) - np.maximum(movement,BIGZERO)/dx) + L_var
	#print "Returning..."
	return tmp

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
def F_global(x_array,m_array,sigma,time): #more effective running cost function
	return (x_array-0.2)**2 #Carlini's no-game
	#return np.minimum(1.4*np.ones(x_array.size),np.maximum(m_array,0.7*np.ones(x_array.size))) #Gueant's game
	#tmp = mollify_array(m_array,sigma,x_array,gll_x,gll_w)
	#tmp = mollify_array(tmp,sigma,x_array,gll_x,gll_w)
	#return 0.05*mollify_array(tmp,sigma,x_array,gll_x,gll_w)
	#return 0.03*tmp
	#return 0.1*m_array #shyness game
	#return (powerbill(time)*(1-0.8*x_array) + x_array/(0.1+m_array))
	#return 0*x_array#no-game

def powerbill(time):
	if time<=0.2:
		return 50*time
	elif time<=0.3:
		return 10
	elif time <=0.75:
		return 10+10/(0.3-0.75)*time
	else:
		return 0
#	return 0

def L_global(time,x_array,a_array,m_array): #general cost
	#return a_array + np.sqrt(x_array) + a_array**2 #Classic Robstad
	return 0.5*a_array**2 + F_global(x_array,m_array,0,time)

def f_global(time,x_array,a_array):
	#return 0.1*a_array*x_array #Classic Robstad
	return a_array #standard MFG

def Sigma_global(time,x_array,a_array,m_array):
	#return 4+a_array*x_array #Classic Robstad
	#return 0.1*x_array+(1-x_array)*0.3
	return 0.2*np.ones(x_array.size)

def Sigma_local(time,x,a,m):
	return 0*x

##################
#TERMINAL COST
##################
def G(x_array,m_array): #this is the final cost, and is a function of the entire distribution m and each point x_i
	#return -0.5*(x_array+0.5)**2 * (1.5-x_array)**2 #Carlini's original
	#return 0.1*(x_array*(1-x_array))**2 #Gueant's game
	#return -((x_array+0.2)*(1.2-x_array))**4 #Shyness game
	return np.zeros(x_array.size) #Carlini's no-game & Isolation game
	#return 0.001*m_array


##################
#INITIAL DISTRIBUTION
##################
def initial_distribution(x):
	#return 1-0.2*np.cos(np.pi*x) #gueant's original
	#return np.exp(-(x-0.75)**2/0.1**2) #carlini's no-game
	return np.exp(-(x-0.5)**2/0.1**2) #shyness game
	#return np.exp(-(x-0.3)**2/0.1**2) #isolation game
	#return 1/(3*x+1)**3 #isolation game 2

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
	trajectory = np.minimum(np.maximum(x_array[0]*np.ones(x_array.size),trajectory),x_array[-1]*np.ones(x_array.size))
	return trajectory
def restrain4isolation(trajectory,x_array):
	return np.maximum(restrain(trajectory),x_array)

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

	
