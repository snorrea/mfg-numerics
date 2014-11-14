
from __future__ import division
import numpy as np
from pylab import *
import time as time
from scipy.optimize import minimize as minimize

def min_approx(function, (args)):
	searchsector_left = False
	searchsector_right = False
	left = -30
	right = 30
	#backtracking linesearch parameters
	linesearch_step0 = 1
	linesearch_decrement = 0.5
	linesearch_tolerance = 1/5
	#simplex parameters
	dx = 0.5
	Nx = 5 #number of points
	###########################################################################################
	t1 = time.time()
	while searchsector_left is False: #do line search
		h = linesearch_step0
		while True:
			if function(left+h,*args) < function(left,*args):
				left += h
			elif h > linesearch_tolerance:
				h = h*linesearch_decrement
			else:
				searchsector_left = True #left search boundary found
				break
	while searchsector_right is False:
		h = linesearch_step0
		while True:
			if function(right-h,*args) < function(right,*args):
				right += -h
			elif h > linesearch_tolerance:
				h = h*linesearch_decrement
			else:
				searchsector_right = True #left search boundary found
				break
	#print left, right
	if left > right:
		print "Error! Left boundary greater than right boundary."
		return -1
	print "Time to determine sector:",time.time()-t1
	###########################################################################################
	#now the sector has been found; dish out points and do some kind of simplex method
	#xpts = np.linspace(left,right,abs(right-left)/dx) #our points
	xpts = np.linspace(left,right,Nx)
	fpts = np.empty((xpts.size,1))
	#now it gets tricky
	#let's do a linesearch for every single point! THIS WILL BE FUN and then select the best one
#	t1 = time.time()
#	dx = (xpts[1]-xpts[0])/2
#	for i in range(0,xpts.size):
#		h = dx
#		#print xpts.size
#		while True:
#			if function(xpts[i]+h,*args) < function(xpts[i]-h,*args): #go right
#				xpts[i] += h
#				h = h*linesearch_decrement
#			elif function(xpts[i]-h,*args) < function(xpts[i]+h,*args): #go left
#				xpts[i] += -h
#				h = h*linesearch_decrement
#			elif h > linesearch_tolerance:
#				h = h*linesearch_decrement
#			else:
#				break
	#for i in range (0,xpts.size):
	#	fpts[i] = function(xpts[i],*args)
	fpts = function(xpts,*args)
	x0 = x[np.argmin(fpts)]
	t1 = time.time()
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
	print "Time to minimise:",time.time()-t1
	return x0
	###########################################################################################
	#by now we should have all the points lumped in more or more minima
	#for i in range (0,xpts.size):
	#	fpts[i] = function(xpts[i],*args)
	#minindex = np.argmin(fpts) #get index of minimum
	###########################################################################################
	#return xpts[minindex]

def min_approx1(function, (args)):
	left = -10
	right = 10
	linesearch_decrement = 0.75
	linesearch_tolerance = 1/10
	dx = .2
	#Nx = 5 #number of points
	###########################################################################################
	#now the sector has been found; dish out points and do some kind of simplex method
	xpts = np.linspace(left,right,abs(right-left)/dx) #our points
	#xpts = np.linspace(left,right,Nx)
	fpts = np.empty((xpts.size,1))
	fpts = function(xpts,*args)
	x0 = xpts[np.argmin(fpts)]
	t1 = time.time()
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
	print "Time to minimise:",time.time()-t1
	return x0
	###########################################################################################
	#by now we should have all the points lumped in more or more minima
	#for i in range (0,xpts.size):
	#	fpts[i] = function(xpts[i],*args)
	#minindex = np.argmin(fpts) #get index of minimum
	###########################################################################################
	#return xpts[minindex]


###########################################################################################
def tau(alpha,i,v_array,x_array): #the function to be minimised
	tmp = 0.5*dt*alpha**2
	for j in range (0,x_array.size):
		tmp += v_array[j]*beta(x_array[i]-dt*alpha,j,x_array)
	return tmp
def beta(x,i,x_array):
	return np.maximum(0,1-abs(x-x_array[i])/dx)	
def G(xi,m_array): #this is the final cost, and is a function of the entire distribution m and each point x_i
	return -0.5*(xi+0.5)**2 * (1.5-xi)**2 #Carlini's original

dx = 0.05 #these taken from Gueant's paper
dt = 0.02
xmin = -1
xmax = 1
Iarg = 1
x = np.arange(xmin,xmax+dx,dx)
v_tmp = G(x,x)
alpha = np.arange(-10,10+dx,dx*0.1)
taus = np.empty((alpha.size,1))
for i in range (0,alpha.size):
	taus[i] = tau(alpha[i],10,v_tmp,x)

plot(alpha,taus)
t1 = time.time()
val1 = min_approx1(tau,(Iarg,v_tmp,x)) #first and second arg are indices
t2 = time.time()
val2 = minimize(tau,0,args=(Iarg,v_tmp,x))
t3 = time.time()
print val1
print val1-val2.x[0], tau(val1,Iarg,v_tmp,x)-tau(val2.x[0],Iarg,v_tmp,x)
if tau(val1,Iarg,v_tmp,x)-tau(val2.x[0],Iarg,v_tmp,x) > 0:
	print "Our method is flawed."
elif tau(val1,Iarg,v_tmp,x)-tau(val2.x[0],Iarg,v_tmp,x) == 0:
	print "Methods are exactly equal."
else:
	print "Their method sucks."
print "Comparison: mine used", t2-t1, ", their used", t3-t2
if t2-t1 < t3-t2:
	print "Mine is faster by", (t3-t2)/(t2-t1), "."
else:
	print "Theirs is faster by", (t2-t1)/(t3-t2), "."
show()





