from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import time as time
from mpl_toolkits.mplot3d import Axes3D
import applications as app
from matplotlib import cm

#input
#origin = [0., 0.]
goal_x = [.0, 1.]
goal_y = [.0, .0]
dx = 0.01
Nx = 1/dx+1
Ny = 1/dx+1
obstacle_x_min = [.4,.1]
obstacle_x_max = [.6,.9]
obstacle_y_min = [.3,.5]
obstacle_y_max = [.4,.6]

#crunch
x = np.linspace(0,1,Nx)
y = np.linspace(0,1,Ny)
X,Y = np.meshgrid(x,y)
index_0 = []
for i in range(len(goal_x)):
	index_0.append(app.find_nearest_index(x,goal_x[i])+x.size*app.find_nearest_index(y,goal_y[i]))
#print index_0
#print ss

south = []
north = []
west = []
east = []
nulled = []
for i in range(len(obstacle_x_min)): #only need nulled from this
		south,north,west,east,nulled,se,sw,ne,nw = app.add_obstacle(x,y,(obstacle_x_min[i],obstacle_x_max[i],obstacle_y_min[i],obstacle_x_max[i]),south,north,west,east,nulled)
I = x.size
J = y.size
south = range(0,I) #ok
west = range(0,I*J,I) #ok
north = range(I*J-I,I*J)
east = range(I-1,I*J+1,I)

#initialise FMM stuff; use indices for this
#known = [index_0]#list(index_0)
#print known
known = index_0[:]
#known = []
#known.extend(index_0)
trial = []
cost_function = np.ones(x.size*y.size)*dx
cost_function[nulled] = 1
known_costs = np.empty(x.size*y.size)
trial_costs = np.empty(x.size*y.size)
trial_costs[:] = np.inf
def find_neighbours(node_index):
	#print (node_index)
	neighbours = []
	if not app.ismember(node_index,north):
		neighbours.append(node_index + I)
	if not app.ismember(node_index,south):
		neighbours.append(node_index - I)
	if not app.ismember(node_index,west):
		neighbours.append(node_index - 1)
	if not app.ismember(node_index,east):
		neighbours.append(node_index + 1)
	return neighbours

for i in range(len(index_0)): #add known stuff
	trial.extend(find_neighbours(index_0[i])) #expand to index_0 -> known
	trial_costs[find_neighbours(index_0[i])] = cost_function[find_neighbours(index_0[i])]
known_costs[index_0] = 0
trial_costs[index_0] = np.inf
#trial = [x for x in trial if x not in known]
#print trial
#print ss
while len(known)!=I*J:
	#find cheapest trial
	idx = int(np.argmin(trial_costs)) #picks values that are already in known because of value-ties
	#print trial_costs
	#print idx
	#if app.ismember(idx,known): #we're done...?
	#	print cnt,I*J+1
	#	break
	val = np.amin(trial_costs)
	trial_costs[idx] = np.inf #remove trial value
	#print idx, app.ismember(idx,known)
	trial.remove(idx) #remove trial
	#print ss
	known.append(idx)
	#known.sort()
	#print known
	#print ss
	known_costs[idx] = val #store known cost
	neighbours0 = find_neighbours(idx)
	neighbours = [x for x in neighbours0 if (x not in known) and (x not in trial)]
	#if neighbours == []:
	#	break
	trial.extend(neighbours)
	trial_costs[neighbours] = np.minimum(val + cost_function[neighbours],trial_costs[neighbours])
	#trial.sort()
	print len(trial),len(known),I*J
	#if trial_costs[index_0[1]]!=np.inf:
	#	print "Fuck"
	#	print idx
#print trial
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot_surface(X,Y,np.reshape(known_costs,(I,J)),rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=0, antialiased=False)
fig1.suptitle("a1")
ax1.set_xlabel('x')
ax1.set_ylabel('y')
print known_costs
plt.show()
