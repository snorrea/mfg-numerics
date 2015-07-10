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
			xpts = np.linspace(x_naught-dex_x,xmax,N[0])
		elif x_naught-dex_x < xmin:
			xpts = np.linspace(xmin,x_naught+dex_x,N[0])
		else:
			xpts = np.linspace(x_naught-dex_x,x_naught+dex_x,N[0])
		if y_naught+dex_y > ymax:
			ypts = np.linspace(y_naught-dex_y,ymax,N[1])
		elif y_naught-dex_y < ymin:
			ypts = np.linspace(ymin,y_naught+dex_y,N[1])
		else:
			ypts = np.linspace(y_naught-dex_y,y_naught+dex_y,N[1])
		xpts,ypts = np.meshgrid(xpts,ypts)
		fpts = function(xpts,ypts,*args)
		if i!=k-1:
			(xi,yi) = np.unravel_index(np.argmin(fpts),fpts.shape)
			#if fpts[xi,yi]!=np.amin(fpts):
			#	print "Picked wrong thing..."
			#print xi,yi
			#print xpts.shape,ypts.shape
			x_naught = xpts[xi,yi]
			y_naught = ypts[xi,yi]
			dex_x = xpts[0,1]-xpts[0,0]
			dex_y = ypts[1,0]-ypts[0,0]
			#print i,dex_x,dex_y
			#print ss
	#print crit_ind
	(xi,yi) = np.unravel_index(np.argmin(fpts),fpts.shape)#print xi,yi
	#print "Indices:",xi,yi
	#print "Storage:",xpts.shape,ypts.shape
	return xpts[xi,yi],ypts[xi,yi]

def hybrid_search2(function,funcprime, (args),dx,dy,x0,y0,N,max_iterations,xmin,xmax,ymin,ymax,tolerance): 
	x_naught = x0
	y_naught = y0
	dex_x = dx
	dex_y = dy
	for i in range (0,max_iterations):
		[p1,p2] = funcprime(x_naught,y_naught,*args)
		psign1 = np.sign(p1)
		psign2 = np.sign(p2)
		if (psign1 > 0 and x_naught==xmin) or (psign1<0 and x_naught==xmax) or (psign1==0): #x0 is found
			xpts = x_naught
			if (psign2 > 0 and y_naught==ymin) or (psign2 < 0 and y_naught==ymax) or (psign2==0):
				return x_naught,y_naught
			else: #search in y
				if y_naught+dex_y > ymax:
					ypts = np.linspace(y_naught-dex_y,ymax,N)
				elif y_naught-dex_y < ymin:
					ypts = np.linspace(ymin,y_naught+dex_y,N)
				else:
					ypts = np.linspace(y_naught-dex_y,y_naught+dex_y,N)
			#NEEDS TO EVAL AND ALL THAT
		elif (psign2 > 0 and y_naught==ymin) or (psign2 < 0 and y_naught==ymax) or (psign2==0): #y0 is found
			ypts = y_naught
			if (psign1 > 0 and x_naught==xmin) or (psign1 < 0 and x_naught==xmax) or (psign1==0):
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
		tmpx = np.copy(xpts)
		tmpy = np.copy(ypts)
		xpts,ypts = np.meshgrid(xpts,ypts)
		fpts = function(xpts,ypts,*args)
		if i!=max_iterations:
			(xi,yi) = np.unravel_index(np.argmin(fpts),fpts.shape)
			x_naught = xpts[xi,yi]
			y_naught = ypts[xi,yi]
			if tmpx.size==1:
				dex_x = 0
			else:
				dex_x = tmpx[1]-tmpx[0]
			if tmpy.size==1:
				dex_y = 0
			else:
				dex_y = tmpy[1]-tmpy[0]
			if max(abs(dex_x),abs(dex_y)) < tolerance:
				return x_naught,y_naught
			#print dex_x,dex_y
			#print ss
	(xi,yi) = np.unravel_index(np.argmin(fpts),fpts.shape)
	#print xi,yi
	#print "Indices:",xi,yi
	#print "Storage:",xpts.shape,ypts.shape
	return xpts[xi,yi],ypts[xi,yi]
def hybrid_search(function,funcprime, (args),dx,dy,x0,y0,N,max_iterations,xmin,xmax,ymin,ymax,tolerance): 
	x_naught = x0
	y_naught = y0
	dex_x = dx
	dex_y = dy
	for i in range(0,10*max_iterations):
		xpts = None
		ypts = None
		[p1,p2] = funcprime(x_naught,y_naught,*args)
		#print p1,p2
		psign1 = np.sign(p1)
		psign2 = np.sign(p2)
		#print psign1,psign2
		if (psign1 > 0 and x_naught==xmin) or (psign1 < 0 and x_naught==xmax) or (psign1==0) or (dex_x==0): #x0 is found
		#if (psign1 > 0 and x_naught==xmin) or (psign1 < 0 and x_naught==xmax) or (dex_x==0): #x0 is found
			xpts = x_naught
		if (psign2 > 0 and y_naught==ymin) or (psign2 < 0 and y_naught==ymax) or (psign2==0) or (dex_y==0): #y0 is found
		#if (psign2 > 0 and y_naught==ymin) or (psign2 < 0 and y_naught==ymax) or (dex_y==0): #y0 is found
			ypts = y_naught
		#set xpts
		if xpts!=None and ypts!=None: #found both
			return xpts, ypts
		if psign1 < 0 and xpts==None: #go forward
			xpts = np.linspace(x_naught,min(x_naught+dex_x,xmax),N/2)
		elif psign1 > 0 and xpts==None: #go back
			xpts = np.linspace(max(x_naught-dex_x,xmin),x_naught,N/2)
		#else:
		#	xpts = np.linspace(max(x_naught-dex_x,xmin),min(x_naught+dex_x,xmax),N)
		#set ypts
		if psign2 < 0 and ypts==None: #go forward
			ypts = np.linspace(y_naught,min(y_naught+dex_y,ymax),N/2)
		elif psign2 > 0 and ypts==None: #go back
			ypts = np.linspace(max(y_naught-dex_y,ymin),y_naught,N/2)
		#else:
		#	ypts = np.linspace(max(y_naught-dex_y,ymin),min(y_naught+dex_y,ymax),N)
		#by this time, ypts and xpts should have been set
		tmpx = np.copy(xpts)
		tmpy = np.copy(ypts)
		xpts,ypts = np.meshgrid(xpts,ypts)
		fpts = function(xpts,ypts,*args)
		if i!=max_iterations:
			(xi,yi) = np.unravel_index(np.argmin(fpts),fpts.shape)
			x_naught = xpts[xi,yi]
			y_naught = ypts[xi,yi]
			if tmpx.size==1:
				dex_x = 0
			else:
				dex_x = abs(tmpx[1]-tmpx[0])
			if tmpy.size==1:
				dex_y = 0
			else:
				dex_y = abs(tmpy[1]-tmpy[0])
			if max(abs(dex_x),abs(dex_y)) < tolerance:
				return x_naught,y_naught
			#print dex_x,dex_y
			#print ss
	(xi,yi) = np.unravel_index(np.argmin(fpts),fpts.shape)
	#print xi,yi
	#print "Indices:",xi,yi
	#print "Storage:",xpts.shape,ypts.shape
	return xpts[xi,yi],ypts[xi,yi]

def hybrid_search66(function,(args),f0,dx,dy,x0,y0,N,max_iterations,xmin,xmax,ymin,ymax,tolerance): 
	x_naught = x0
	y_naught = y0
	dex_x = dx
	dex_y = dy
	for i in range(0,10*max_iterations):
		xpts = None
		ypts = None
		psign1,psign2 = sign_of_derivative(function,x_naught,y_naught,(args),f0)
		#print psign1,psign2
		if (psign1 == 1 and x_naught==xmin) or (psign1 == -1 and x_naught==xmax) or (psign1==0) or (dex_x==0): #x0 is found
			xpts = x_naught
		if (psign2 == 1 and y_naught==ymin) or (psign2 == -1 and y_naught==ymax) or (psign2==0) or (dex_y==0): #y0 is found
			ypts = y_naught
		#set xpts
		if xpts!=None and ypts!=None: #found both
			return xpts, ypts
		if psign1 == -1 and xpts==None: #go forward
			xpts = np.linspace(x_naught,min(x_naught+dex_x,xmax),N/2)
		elif psign1 == 1 and xpts==None: #go back
			xpts = np.linspace(max(x_naught-dex_x,xmin),x_naught,N/2)
		elif psign1 == 2 and xpts==None:
			xpts = np.linspace(max(x_naught-dex_x,xmin),min(x_naught+dex_x,xmax),N)
		#set ypts
		if psign2 == -1 and ypts==None: #go forward
			ypts = np.linspace(y_naught,min(y_naught+dex_y,ymax),N/2)
		elif psign2 == 1 and ypts==None: #go back
			ypts = np.linspace(max(y_naught-dex_y,ymin),y_naught,N/2)
		elif psign2 == 2 and ypts==None:
			ypts = np.linspace(max(y_naught-dex_y,ymin),min(y_naught+dex_y,ymax),N)
		#by this time, ypts and xpts should have been set
		tmpx = np.copy(xpts)
		tmpy = np.copy(ypts)
		xpts,ypts = np.meshgrid(xpts,ypts)
		fpts = function(xpts,ypts,*args)
		if i!=max_iterations:
			(xi,yi) = np.unravel_index(np.argmin(fpts),fpts.shape)
			x_naught = xpts[xi,yi]
			y_naught = ypts[xi,yi]
			f0 = fpts[xi,yi]
			if tmpx.size==1:
				dex_x = 0
			else:
				dex_x = abs(tmpx[1]-tmpx[0])
			if tmpy.size==1:
				dex_y = 0
			else:
				dex_y = abs(tmpy[1]-tmpy[0])
			if max(abs(dex_x),abs(dex_y)) < tolerance:
				return x_naught,y_naught
	(xi,yi) = np.unravel_index(np.argmin(fpts),fpts.shape)
	return xpts[xi,yi],ypts[xi,yi]

def sign_of_derivative(function,x,y,(args),f0):
	eps = 1e-6
	x_args = np.array([x-eps, x+eps, x, x])
	y_args = np.array([y, y, y-eps, y+eps])
	x_args,y_args = np.meshgrid(x_args,y_args)
	[west, east, south, north] = np.diagonal(function(x_args,y_args,*args))
	if west > f0 and east > f0: #both bigger
		ps1 = 0
	elif west>f0 and east<f0: #east
		ps1 = -1
	elif west<f0 and east>f0: #west
		ps1 = 1
	else: #both smaller...
		ps1 = 2 #special case
	if south > f0 and north > f0: #both bigger
		ps2 = 0
	elif south>f0 and north<f0: #east
		ps2 = -1
	elif south<f0 and north>f0: #west
		ps2 = 1
	else: #both smaller...
		ps2 = 2 #special case
	return ps1, ps2
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
	new_north = range(xmin_i+I*ymin_i,xmax_i+I*ymin_i+1)
	new_south = range(xmin_i+I*ymax_i,xmax_i+I*ymax_i+1)
	new_east = range(xmin_i+I*ymin_i,xmin_i+I*ymax_i+1,I)
	new_west = range(xmax_i+I*ymin_i,xmax_i+I*ymax_i+1,I)
	south = np.sort(np.concatenate((new_south,south),0)).astype(int)
	north = np.sort(np.concatenate((new_north,north),0)).astype(int)
	west = np.sort(np.concatenate((new_west,west),0)).astype(int)
	east = np.sort(np.concatenate((new_east,east),0)).astype(int)
	if nulled==None:
		nulled = range(xmin_i + 1 + I*(ymin_i + 1),xmax_i - 1 + I*(ymin_i + 1)+1)
		for i in range(1,ymax_i - ymin_i):
			tmp = range(xmin_i + 1 + I*(ymin_i + 1+i),xmax_i - 1 + I*(ymin_i + 1+i)+1)
			nulled = np.concatenate((tmp,nulled),0)
	else:
		for i in range(ymax_i - ymin_i):
			tmp = range(xmin_i + 1 + I*(ymin_i + 1+i),xmax_i - 1 + I*(ymin_i + 1+i)+1)
			nulled = np.concatenate((tmp,nulled),0)
	se = np.intersect1d(south,east)
	sw = np.intersect1d(south,west)
	ne = np.intersect1d(north,east)
	nw = np.intersect1d(north,west)
	return south,north,west,east,np.sort(nulled).astype(int),se,sw,ne,nw

###########
#FMM
###########

def FMM(x,y,nulled,goal_x,goal_y):
	dx = x[1]-x[0]
	I = x.size
	J = y.size
	south = range(0,I) #ok
	west = range(0,I*J,I) #ok
	north = range(I*J-I,I*J)
	east = range(I-1,I*J+1,I)
	index_0 = []
	for i in range(len(goal_x)):
		index_0.append(find_nearest_index(x,goal_x[i])+I*find_nearest_index(y,goal_y[i]))
	known = list(np.unique(index_0[:]))
	trial = []
	unvisited = [ item for i,item in enumerate(range(I*J)) if i not in known ]
	cost_function = np.ones(I*J)*dx
	cost_function[nulled] = 100*dx
	known_costs = np.empty(I*J)
	known_costs[:] = np.inf
	trial_costs = np.empty(I*J)
	trial_costs[:] = np.inf
	for i in range(len(index_0)): #add known stuff
		neb = find_neighbours(index_0[i],north,south,west,east,I)
		trial.extend(neb) #expand to index_0 -> known
		trial_costs[neb] = cost_function[neb]
	known_costs[index_0] = 0
	trial_costs[index_0] = np.inf
	while len(known)!=I*J:
		idx = int(np.argmin(trial_costs)) #picks values that are already in known because of value-ties
		val = np.amin(trial_costs)
		trial_costs[idx] = np.inf #remove trial value
		trial.remove(idx) #remove trial
		known.append(idx)
		known_costs[idx] = val #store known cost
		neighbours0 = find_neighbours(idx,north,south,west,east,I)
		neighbours = [x for x in neighbours0 if (x not in known) and (x not in trial)]
		trial.extend(neighbours)
		#trial_costs[neighbours] = np.minimum(val + cost_function[neighbours],trial_costs[neighbours])
		trial_costs[neighbours] = val + cost_function[neighbours]
	return known_costs

###################
#AUXILIARY FUNCTIONS
###################

def map_2d_to_3d(arr2d,arr3d):
	#kinda broadcast the 2d-array into a 3d-shape
	#assumes that two of the dimensions match
	output = np.zeros(arr3d.shape)
	#print arr2d.shape
	#print arr3d.shape
	#print output.shape
	#output[:,0:arr3d.shape[1],:] = np.transpose(arr2d)[:]
	#print output.shape
	#print arr2d.shape
	for i in range(arr3d.shape[1]):
		if output.shape[0]==arr2d.shape[0]:
			output[:,i,:] = (arr2d)
		else:
			output[:,i,:] = np.transpose(arr2d)
	return output

def gimme_gradient_points(minis,index_max):
	my_list_x = []
	my_list_y = []	
	for v in range(minis.size):
		if minis[v]==0:
			my_list_x.append(0)
			my_list_x.append(1)
			my_list_y.append(v)
			my_list_y.append(v)
		elif minis[v]==index_max:
			my_list_x.append(index_max-1)
			my_list_x.append(index_max)
			my_list_y.append(v)
			my_list_y.append(v)
		else:
			tmp = minis[v]
			my_list_x.append(tmp-1)
			my_list_x.append(tmp)
			my_list_x.append(tmp+1)
			my_list_y.append(v)
			my_list_y.append(v)
			my_list_y.append(v)
	return my_list_x,my_list_y

def gimme_gradient_indices(minis,index_max):
	my_list_y = range(index_max+1)*2
	my_list_y.sort()
	my_list_x = []
	for v in range(minis.size):
			tmp = minis[v]
			my_list_x.append(max(tmp-1,0))
			my_list_x.append(min(tmp+1,index_max))
	return my_list_x,my_list_y
	
#def gimme_gradient_values(minis,index_max,array):
#	gradlist = []
#	for v in range(minis.size):
#		gradlist.append(array[min(minis[v]+1,index_max),v]-array[max(minis[v]-1,0),v])
#	return gradlist

def gimme_gradient_values(minis,index_max,array):
	return [array[min(minis[v]+1,index_max),v]-array[max(minis[v]-1,0),v] for v in range(minis.size)]

def find_neighbours(node_index,north,south,west,east,I):
	#print (node_index)
	neighbours = []
	if not ismember(node_index,north):
		neighbours.append(node_index + I)
	if not ismember(node_index,south):
		neighbours.append(node_index - I)
	if not ismember(node_index,west):
		neighbours.append(node_index - 1)
	if not ismember(node_index,east):
		neighbours.append(node_index + 1)
	return neighbours

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


def ismember(a,array): #assumes sorted
	for i in range(0,len(array)):
		if array[i]==a:
			return True
		elif array[i]>a:
			return False
	return False
