from __future__ import division
import numpy as np
import quadrature_nodes as qn
import input_functions_2D as iF
import matrix_gen as mg
from scipy.sparse.linalg import spsolve
quad_order = 15
gll_x = qn.GLL_points(quad_order) #quadrature nodes
gll_w = qn.GLL_weights(quad_order,gll_x)

###################
#MINIMISING FUNCTIONS
###################
def scatter_search(function, (args),dx,dy,x0,y0,N,k,xmin,xmax,ymin,ymax): #here k is the number of laps going
	x_naught = x0
	y_naught = y0
	dex_x = dx
	dex_y = dy
	for i in range (0,k):
		if x_naught+dex_x > xmax:
			xpts = np.linspace(x_naught-dex_x,xmax,N)
		elif x_naught-dex_x < xmin:
			xpts = np.linspace(xmin,x_naught+dex_x,N)
		else:
			xpts = np.linspace(x_naught-dex_x,x_naught+dex_x,N)
		if y_naught+dex_y > ymax:
			ypts = np.linspace(y_naught-dex_y,ymax,N)
		elif y_naught-dex_y < ymin:
			ypts = np.linspace(ymin,y_naught+dex_y,N)
		else:
			ypts = np.linspace(y_naught-dex_y,y_naught+dex_y,N)
		xpts,ypts = np.meshgrid(xpts,ypts)
		fpts = function(xpts,ypts,*args)
		if i!=k-1:
			key = xpts.shape
			crit_ind = np.argmin(fpts)
			xi,yi = recover_index(crit_ind,key[0])
			#print xi,yi
			#print xpts.shape,ypts.shape
			x_naught = xpts[xi,yi]
			y_naught = ypts[xi,yi]
			dex_x = xpts[0,1]-xpts[0,0]
			dex_y = ypts[1,0]-ypts[0,0]
			#print dex_x,dex_y
			#print ss
	crit_ind = np.argmin(fpts)
	#print crit_ind
	xi,yi = recover_index(crit_ind,key[0])
	#print xi,yi
	#print "Indices:",xi,yi
	#print "Storage:",xpts.shape,ypts.shape
	return xpts[xi,yi],ypts[xi,yi]

def hybrid_search(function,funcprime, (args),dx,dy,x0,y0,N,max_iterations,xmin,xmax,ymin,ymax): 
	x_naught = x0
	y_naught = y0
	dex_x = dx
	dex_y = dy
	for i in range (0,max_iterations):
		[p1,p2] = funcprime(x_naught,y_naught,*args)
		psign1 = np.sign(p1)
		psign2 = np.sign(p2)
		if (psign1 > 0 and x_naught==xmin) or (psign1<0 and x_naught==xmax) or (psign1==0): #x0 is found
			if (psign2 > 0 and y_naught==ymin) or (psign2 <0 and y_naught==ymax) or (psign2==0):
				return x_naught,y_naught
			else: #search in y
				if y_naught+dex_y > ymax:
					ypts = np.linspace(y_naught-dex_y,ymax,N)
				elif y_naught-dex_y < ymin:
					ypts = np.linspace(ymin,y_naught+dex_y,N)
				else:
					ypts = np.linspace(y_naught-dex_y,y_naught+dex_y,N)
			#NEEDS TO EVAL AND ALL THAT
		elif (psign2 > 0 and y_naught==ymin) or (psign2<0 and y_naught==ymax) or (psign2==0): #y0 is found
			if (psign1 > 0 and x_naught==xmin) or (psign1 <0 and x_naught==xmax) or (psign1==0):
				return x_naught,y_naught
			else: #search in x
				if x_naught+dex_x > xmax:
					xpts = np.linspace(x_naught-dex_x,xmax,N)
				elif x_naught-dex_y < ymin:
					xpts = np.linspace(xmin,x_naught+dex_x,N)
				else:
					xpts = np.linspace(x_naught-dex_x,x_naught+dex_x,N)
			#NEEDS TO EVAL AND ALL THAT
		else:
			if x_naught+dex_x > xmax:
				xpts = np.linspace(x_naught-dex_x,xmax,N)
			elif x_naught-dex_x < xmin:
				xpts = np.linspace(xmin,x_naught+dex_x,N)
			else:
				xpts = np.linspace(x_naught-dex_x,x_naught+dex_x,N)
			if y_naught+dex_y > ymax:
				ypts = np.linspace(y_naught-dex_y,ymax,N)
			elif y_naught-dex_y < ymin:
				ypts = np.linspace(ymin,y_naught+dex_y,N)
			else:
				ypts = np.linspace(y_naught-dex_y,y_naught+dex_y,N)
		xpts,ypts = np.meshgrid(xpts,ypts)
		fpts = function(xpts,ypts,*args)
		if i!=k-1:
			key = xpts.shape
			crit_ind = np.argmin(fpts)
			xi,yi = recover_index(crit_ind,key[0])
			x_naught = xpts[xi,yi]
			y_naught = ypts[xi,yi]
			dex_x = xpts[0,1]-xpts[0,0]
			dex_y = ypts[1,0]-ypts[0,0]
			if max(abs(dex_x),abs(dex_y)) < tolerance:
				return x_naught,y_naught
			#print dex_x,dex_y
			#print ss
	crit_ind = np.argmin(fpts)
	#print crit_ind
	xi,yi = recover_index(crit_ind,key[0])
	#print xi,yi
	#print "Indices:",xi,yi
	#print "Storage:",xpts.shape,ypts.shape
	return xpts[xi,yi],ypts[xi,yi]

##################
# GRID FUNCTIONS
##################

def add_obstacle(x,y,(xmin,xmax,ymin,ymax),south,north,west,east,nulled):
	#inputs:
		#x,y are 1D vectors of the grid
		#xmin,xmax,ymin,ymax indicate the vertices of the obstacle
		#south,north,west,east are the boundaries of the grid as they are
	#outputs:
		#south,north,west,east are the updates boundaries of the grid
	#x,y = np.meshgrid(x,y)
	#find the closest indices for xmin,xmax,ymin,ymax
	#corners are (xmin,ymin), (xmin,ymax), (xmax,ymin), (xmax,ymax)
	xmin_i = find_nearest_index(x,xmin)
	xmax_i = find_nearest_index(x,xmax)
	ymin_i = find_nearest_index(y,ymin)
	ymax_i = find_nearest_index(y,ymax)
	I = x.size
	new_south = range(xmin_i+I*ymin_i,xmax_i+I*ymin_i+1)
	new_north = range(xmin_i+I*ymax_i,xmax_i+I*ymax_i+1)
	new_west = range(xmin_i+I*ymin_i,xmin_i+I*ymax_i+1,I)
	new_east = range(xmax_i+I*ymin_i,xmax_i+I*ymax_i+1,I)
	south = np.sort(np.concatenate((new_south,south),0)) 
	north = np.sort(np.concatenate((new_north,north),0)) 
	west = np.sort(np.concatenate((new_west,west),0)) 
	east = np.sort(np.concatenate((new_east,east),0)) 
	if nulled==None:
		nulled = range(xmin_i + 1 + I*(ymin_i + 1),xmax_i - 1 + I*(ymin_i + 1)+1)
		for i in range(1,ymax_i - ymin_i-1):
			tmp = range(xmin_i + 1 + I*(ymin_i + 1+i),xmax_i - 1 + I*(ymin_i + 1+i)+1)
			nulled = np.concatenate((tmp,nulled),0)
	else:
		for i in range(ymax_i - ymin_i - 2):
			tmp = range(xmin_i + 1 + I*(ymin_i + 1+i),xmax_i - 1 + I*(ymin_i + 1+i)+1)
			nulled = np.concatenate((tmp,nulled),0)
	return south,north,west,east,np.sort(nulled)

###################
#AUXILIARY FUNCTIONS
###################
def find_nearest_index(array,value):
    return (np.abs(array-value)).argmin()

def find_nearest_value(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

def spike_detector(array): #finds the spikes, assuming reflective boundary
	u_xx = np.empty(array.size)
	for i in range(0,array.size):
		if i==0:
			u_xx[i] = 2*(array[i+1]-array[i])
		elif i==array.size-1:
			u_xx[i] = 2*(array[i-1]-array[i])
		else:
			u_xx[i] = array[i+1]+array[i-1]-2*array[i]
	#now find sign change
	u_xx = np.gradient(np.sign(u_xx))
	begins = [] #stores indices where things begin
	ends = [] #stores indices where things end
	look = None #
	for i in range(0,array.size):
		if u_xx[i] < 0 and look!=True: #start of thing
			begins.append(i)
			look = True
		elif u_xx[i] > 0 and look==True:
			ends.append(i)
			look = False
	return begins,ends

def hmean(a,b): #returns harmonic mean of a,b
	return 2*a*b/(a+b)

def recover_index(crit_ind,key): #where key is as in index:=i+I*j, with I=key. this function recovers i,j
	i=0
	j=0
	while crit_ind - key >= 0:
		crit_ind = crit_ind - key
		j += 1
	while crit_ind > 0:
		crit_ind += -1
		i += 1
	#print i+j*key == crit_ind
	return i,j

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

