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


###################
#AUXILIARY FUNCTIONS
###################
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

