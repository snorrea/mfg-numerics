from __future__ import division
import numpy as np
import quadrature_nodes as qn
from scipy.optimize import minimize as minimize
import matplotlib.pyplot as plt
import random
quad_order = 15
gll_x = qn.GLL_points(quad_order) #quadrature nodes
gll_w = qn.GLL_weights(quad_order,gll_x)


def hmean(a,b):
	#output = 2*a*b/(a+b)
	output = np.zeros(a.size)
	for i in range(0,a.size):
		if a[i]==0 or b[i]==0:
			output[i]=0
		else:
			output[i]=2*a[i]*b[i]/(a[i]+b[i])
	return output

def hmean_scalar(a,b):
	if a==0 or b==0:
		return 0
	else:
		return 2*a*b/(a+b)

###################
#MINIMISING FUNCTIONS
###################
def scatter_search(function, (args),dx,x0,N,k,left,right): #here k is the number of laps going
	x_naught = x0
	dex = dx
	if k>0:
		for i in range (0,k):
			#define searchspace
			if x_naught+dex > right:
				xpts = np.linspace(x_naught-dex,right,N)
			elif x_naught-dex < left:
				xpts = np.linspace(left,x_naught+dex,N)
			else:
				xpts = np.linspace(x_naught-dex,x_naught+dex,N)
			#evaluate and cherry pick
			dex = xpts[2]-xpts[1]
			fpts = function(xpts,*args)
			if i!=k-1: #this if is here just to save an extra evaluation
				x_naught = xpts[np.argmin(fpts)]
		return xpts[np.argmin(fpts)],min(fpts)
	else:
		return x_naught,function(x_naught,*args)

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

def crafty_jew_search(funcprime,(args),tolerance,max_iter,x0,dx,left,right):
	x = x0
	for i in range(0,max_iter):
		x_old = x
		p = funcprime(x_old,*args)
		psign = np.sign(p)
		if i==0: 
			if psign>0 and x0==left:
				break
			if psign<0 and x0==right:
				break
		dex = -(.5)**(i+1)*dx*psign
		x = x_old + dex
		if abs(dex) < tolerance:
			return x
	return x

def hybrid_search(func,funcprime,(args),tolerance,iterations,x0,dx,N,left,right):
	x = x0
	dex = dx
	p = funcprime(x,*args)
	psign = np.sign(p)
	if psign>0 and x0==left:
		return x0
	elif psign<0 and x0==right:
		return x0
	elif psign==0:
		print "Found exact"
		return x0	
	for i in range(0,iterations):
		if i is not 0:
			p = funcprime(x,*args)
			psign = np.sign(p)
		if psign>0: #go left
			xt = np.linspace(x-dex,x,N)
		elif psign<0: #go right
			xt = np.linspace(x,x+dex,N)
		else:
			#print "Found exact"
			return x
		dex = xt[1]-xt[0]
		fpts = func(xt,*args)
		x = xt[np.argmin(fpts)]
	return x

def newton_search(func,funcprime,(args),tolerance,max_iter,x0,dx,left,right):
	x = x0
	for i in range(0,max_iter):
		x_old = x
		dd = funcprime(x_old,*args)
		x = x_old - func(x_old,*args)/dd
		if abs(x-x_old) < tolerance and dd>0:
			#print i
			return x,1
		elif x<x0-dx:
		#	print "Left,x0-dx:",x0-dx
		#	print "\tSuggested:",x
			return max(left,x0-dx),0
		elif x>x0+dx:
		#	print "Right,x0+dx:",x0+dx
		#	print "\tSuggested:",x
			return min(right,x0+dx),0
	return x,-1


def newton_search_wolfe(func,funcprime,(args),tolerance,max_iter,x0,dx,left,right):
	x = x0
	step = 1
	for i in range(0,max_iter):
		x_old = x
		d = func(x_old,*args)
		dd = funcprime(x_old,*args)
		direction = -d/dd
		for j in range(0,max_iter-i):
			wolfe1 = func(x_old + step*direction,*args)
			wolfe2 = d + 1e-4*step*direction*dd
			wolfe3 = direction*funcprime(x_old+step*direction,*args)
			wolfe4 = .9*direction*dd
			if wolfe1 <= wolfe2 and wolfe3 >= wolfe4:
				#print "Wolfe success at:", j,step
				break
			else:
				step = .1*step
		x = x_old + step*direction
		if abs(x-x_old) < tolerance and dd>0:
			#print "Found:",i
			return x
		#elif x<x0-dx:
		#	return x0-dx
		#elif x>x0+dx: 
		#	return x0+dx
	return x

def bisection_search(function, (args),dx,x0,tol):
	c,a,b = x0,x0-dx,x0+dx
	err = dx/2
	i = 0
	while err>tol:
		L = function((c+a)/2,*args)
		R = function((c+b)/2,*args)
		c_old = c
		a_old = a
		b_old = b
		if L<R:
			c = (c_old+a_old)/2
			b = c_old
		elif R<L:
			c = (c_old+b_old)/2
			a = c_old
		err = err/2
		i = i+1
	xpts = np.array([a,c,b])
	fpts = function(xpts,*args)
	return xpts[np.argmin(fpts)],min(fpts)

###################
#POLICY ITERATION FUNCTIONS
###################
def Hamiltonian(alphas,time,x_array,u_array,m_array,index,dx):
	BIGZERO = np.zeros(alphas.size)
	#print alphas,time,x_array,u_array,m_array,index,dx
	sigma2 = Sigma_global(time,x_array[index],alphas)**2
	movement = f_global(time,x_array[index],alphas)
	L_var = L_global(time,x_array[index],alphas,m_array[index],dx)
	dx2 = dx**2
	#Kushner
	if index==0: #topmost stuff is correct
		tmp = u_array[index]*(-abs(movement)/dx - sigma2/dx2) +u_array[index+1]*(abs(movement)/dx+sigma2/dx2)+ L_var
	elif index==x_array.size-1:
		tmp =  u_array[index]*(-abs(movement)/dx - sigma2/dx2) +u_array[index-1]*(abs(movement)/dx+sigma2/dx2)+ L_var
	else:
		tmp = u_array[index]*(-abs(movement)/dx - sigma2/dx2) + u_array[index+1]*(sigma2/(2*dx2) + np.maximum(movement,BIGZERO)/dx) + u_array[index-1]*(sigma2/(2*dx2) - np.minimum(movement,BIGZERO)/dx) + L_var
	#print "Returning..."
	#plt.plot(alphas,tmp)
	#plt.show()
	#print tmp
	return tmp

def Hamiltonian_array(alphas,time,x,U_up,U_down,m,dx): 
	#for every value in alphas, give a 2D array for the entire x-alpha plane
	#assumes that alphas,x are meshgrid'ed
	s2 = Sigma_global(time,x,alphas)**2
	f = f_global(time,x,alphas)
	L = L_global(time,x,alphas,m,dx)
	dx2 = dx**2
	zero = np.zeros(alphas.shape)
	return L + np.maximum(zero,f)*U_down + np.minimum(zero,f)*U_up + .5*s2*(U_up-U_down)/dx
	

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
def F_global(x_array,m_array,time,damp): #more effective running cost function
	#return (x_array-0.5)**2 #Carlini's no-game
	#dx=1
	#tau = 1
	#omega = 1
	#return 0.5*(omega**2)*np.exp(-2*time*tau)*np.cos(omega*x_array)**2 + (tau + 0.5*(omega**2)*Sigma_global(time,x_array,x_array,x_array)**2) * np.exp(-tau*time)*np.sin(omega*x_array) #HJB exact test
	#ONE = np.ones(m_array.size)
	#return (x_array-0.2)**2 + np.minimum(4*ONE,np.maximum(m_array,ONE))
	#return np.minimum(1.4*np.ones(x_array.size),np.maximum(m_array,0.7*np.ones(x_array.size))) #Gueant's game
	#tmp = mollify_array(m_array,sigma,x_array,gll_x,gll_w)
	#tmp = mollify_array(tmp,sigma,x_array,gll_x,gll_w)
	#return 0.05*mollify_array(tmp,sigma,x_array,gll_x,gll_w)
	#return 0.03*tmp
	#print "A:",sum(m_array/sum(m_array))
	#print "B:",sum(m_array)
	#print "C:",sum(m_array)*dx
	#plt.plot(x_array,m_array)
	#plt.plot(x_array,m_array/sum(m_array))
	#plt.show()
	#return m_array #shyness game
	#return .1*m_array/sum(m_array) #modified shyness game
	return 2*powerbill(time)*(1-.95*x_array)+ .4*x_array/((1+damp*m_array)**(1.))
	#return 0*x_array#no-game

def powerbill(time):
	return np.sin(np.pi*time)
	#if time<=0.2:
	#	return 50*time
	#elif time<=0.3:
	#	return 10
	#elif time <=0.75:
	#	return 10+10/(0.3-0.75)*time
	#else:
	#	return 0
#	return 0

def L_global(time,x_array,a_array,m_array,damp): #general cost
	#return a_array + np.sqrt(x_array) + a_array**2 #Classic Robstad
	#one = np.ones(x_array.size)
	#xenophobia = np.minimum( np.maximum(m_array,0.2*one), one )
	#return np.exp(-time)*abs(x_array) + 0.5*a_array**2 - 0.2*a_array# +xenophobia#brutal test
	return (0.5*a_array**2+.4*x_array)/(1+.1*m_array*damp) + 2*powerbill(time)*(1-.95*x_array)
	#return (0.5*a_array**2+.4*x_array) + 2*powerbill(time)*(1-.95*x_array)
	#return 0.5*a_array**2 + F_global(x_array,m_array,time,damp) #HJB test and "nice" MFG
	
def f_global(time,x_array,a_array):
	#return 0.1*a_array*x_array #Classic Robstad
	#return -1*np.ones(x_array.size) #FP test, constant coefficients
	#return 2.5*x_array #Ornstein FP test
	#return x_array*np.sin(a_array) #brutal test
	#print x_array.shape,a_array.shape
	#print type(a_array)
	#print a_array
	if isinstance(a_array, np.float64):
		return max(a_array,0)*np.exp(-time) + min(a_array,0)*0.1
	output = np.zeros(a_array.shape)
	#print output.shape
	pos_ind = np.nonzero(np.maximum(np.sign(a_array),output))
	neg_ind = np.nonzero(np.minimum(np.sign(a_array),output))
	output[pos_ind] = a_array[pos_ind]*np.exp(-time)
	output[neg_ind] = .1*a_array[neg_ind]
	#tmp = np.maximum(np.sign(a_array),output) 
	#for i in range(0,output.size):
	#	if a_array[i]>=0:
	#		output[i] = a_array[i]*(np.exp(-time))
	#	else:
	#		output[i] = 0.1*a_array[i]
	#print output
	#print ss
	return output
	#return a_array #standard MFG, HJB test

def Sigma_global(time,x_array,a_array): #any of these will do for the HJB test
	#return 0.*x_array
	#return 4+a_array*x_array #Classic Robstad
	#return 0.1*x_array+(1-x_array)*0.3
	#one = np.ones(x_array.size)
	#return .05*one
	return 0.1*abs(a_array)
	#xenophobia = np.minimum( np.maximum(m_array,0.2*one), 0*one )
	#return 0.5*a_array**2# + xenophobia #brutal test
	#return .1*np.ones(x_array.size)
	#return np.sqrt(2*0.1)*np.ones(x_array.size) #FP test, constant coefficients
	
def Hamiltonian_Derivative(a,t,x,u,m,i,dx):
	#print x
	if i!=0 and i!=m.size-1:
		u1 = (u[i]-u[i-1])/dx
		u2 = (u[i+1]-u[i])/dx
		u3 = (u[i+1]+u[i-1]-2*u[i])/(dx**2)
		u4 = (u[i+1]-u[i-1])/(2*dx)
	elif i==0:
		u1 = (u[i]-u[i+1])/dx
		u2 = (u[i+1]-u[i])/dx
		u3 = (u[i+1]+u[i+1]-2*u[i])/(dx**2)
		u4 = 0
	elif i==m.size-1:
		u1 = (u[i]-u[i-1])/dx
		u2 = (u[i-1]-u[i])/dx
		u3 = (u[i-1]+u[i-1]-2*u[i])/(dx**2)
		u4 = 0
	#tmp = a - 0.2 + a**3/2 * u3 + max_or_if(x*np.cos(a)*u1,x*np.sin(a))+min_or_if(x*np.cos(a)*u2,x*np.sin(a))
	#print tmp
	#print "Inputs:",a,t,x[i],u,m,i,dx
	#print a - 0.2 + a**3/2 * u3 + max_or_if(x[i]*np.cos(a)*u1,x[i]*np.sin(a))+min_or_if(x[i]*np.cos(a)*u2,x[i]*np.sin(a))
	#return a + u1*max_or_if(1,a) + u2*min_or_if(1,a) #HJB exact test
	#return a + 
	return a + u4
	#return a - 0.2 + a**3/2 * u3 + max_or_if(x[i]*np.cos(a)*u1,x[i]*np.sin(a))+min_or_if(x[i]*np.cos(a)*u2,x[i]*np.sin(a))

def Hamiltonian_Derivative_vectorised(alphas,time,x,u_up,u_down,m,dx): 
	#for every value in alphas, give a 2D array for the entire x-alpha plane
	#assumes that alphas,x are meshgrid'ed
	#s = Sigma_global(time,x,alphas)
	#f = f_global(time,x,alphas)
	#L = L_global(time,x,alphas,m,dx)
	zero = np.zeros(alphas.shape)
	return alphas/(1+0.1*m) + np.exp(-time)*u_down*np.maximum(np.sign(alphas),zero) + .1*u_up*np.minimum(np.sign(alphas),zero) + .01*alphas*(u_up-u_down)/dx 

#def max_or_if(val,valif):
#	if valif>0:
#		return val
#	else:
#		return 0

##################
#TERMINAL COST
##################
def G(x,m): #this is the final cost, and is a function of the entire distribution m and each point x_i
	#return 0*m #shyness game
	#return -(x_array+2)**2 * (x_array-2)**2 + 4
	#return 0.5*(x_array+0.5)**2 * (1.5-x_array)**2 #Carlini's original
	#return 0.1*(x_array*(1-x_array))**2 #Gueant's game
	#return -((x_array+0.2)*(1.2-x_array))**4 #Shyness game
	return x*.0 
	#return .4*x**2*(1.5-x) #isolation game cmfg1d
	#return x_array*4 #skyness
	#return np.zeros(x_array.size) #Carlini's no-game & Isolation game
	#return 0.001*m_array


##################
#INITIAL DISTRIBUTION
##################
def initial_distribution(x):
	#return 1-0.2*np.cos(np.pi*x) #gueant's original
	##return np.exp(-(x-0.75)**2/0.1**2) #carlini's no-game
	#m0 = 0.33*np.exp(-(x-0.3)**2/0.1**2)
	#m0 += 0.33*np.exp(-(x-.6)**2/0.1**2)
	#return m0
	#m0 += 0.33*np.exp(-(x+1)**2/0.1**2)
	#return m0
	#return np.exp(-(x-0.5)**2/0.1**2) #shyness game
	#return np.exp(-(x-0.3)**2/0.1**2) #isolation game
	return  (32./5.) * 1/((3*x+1)**3) #isolation game 2

############
# OPTIMAL CONTROL
#######
def opt_cmfg(u,dx):
	output = -np.gradient(u,dx)
	output[0] = 0
	output[-1] = 0
	return output

###################
#AUXILIARY FUNCTIONS
###################
def mollifier(x_val): #Evans' mollifier
	if abs(x_val) < 1:
		return np.exp(1/(x_val**2 - 1))/0.443994
	else:
		return 0

def mollifier_arr(x): #Evans' mollifier
	return np.nan_to_num(np.exp(1/(x**2-1))/0.443994) * abs(np.sign(np.minimum(np.zeros(x.size),abs(x) - 1)))
	
def mollify_array(array,epsilon,x_array,gll_x,gll_w): 
	output = np.zeros((array.size))
#	for k in range (0,array.size):
		#for j in range (0,gll_x.size):
		#	output[k] += gll_w[j]*mollifier(gll_x[j])*np.interp(x_array[k]-epsilon*gll_x[j],x_array,array)
#		output[k] = sum(gll_w*mollifier_arr(gll_x)*np.interp(x_array[k]-epsilon*gll_x,x_array,array))
	for j in range(0,gll_x.size):
		output += gll_w[j]*mollifier_arr(gll_x[j])*np.interp(x_array-epsilon*gll_x[j],x_array,array)
	return output
def restrain(trajectory,x_array):
	trajectory = np.minimum(np.maximum(x_array[0]*np.ones(x_array.size),trajectory),x_array[-1]*np.ones(x_array.size))
	return trajectory
def restrain4isolation(trajectory,x_array):
	return np.maximum(restrain(trajectory),x_array)

def max_or_if(val,valif):
	if valif>0:
		return val
	else:
		return 0

def min_or_if(val,valif):
	if valif<=0:
		return val
	else:
		return 0


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


