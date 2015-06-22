from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import time as time
from mpl_toolkits.mplot3d import Axes3D
import applications as app
from matplotlib import cm

#input
#origin = [0., 0.]
goal_x = [1.]
goal_y = [1.]
dx = 0.05
Nx = 1/dx+1
Ny = 1/dx+1
obstacle_x_min = [.1]
obstacle_x_max = [.9]
obstacle_y_min = [.3]
obstacle_y_max = [.6]

#crunch
x = np.linspace(0,1,Nx)
y = np.linspace(0,1,Ny)
X,Y = np.meshgrid(x,y)
index_0 = []
for i in range(len(goal_x)):
	index_0.append(int(app.find_nearest_index(x,goal_x[i])+x.size*app.find_nearest_index(y,goal_y[i])))

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

known = [index_0]
#known = index_0
print known
#print ss
trial = []
unvisited = [ item for i,item in enumerate(range(x.size*y.size)) if i not in known ]
all_nodes = len(range(x.size*y.size))
cost_function = np.ones(x.size*y.size)*dx
cost_function[nulled] = 1
known_costs = np.empty(x.size*y.size)
known_costs[:] = np.inf
trial_costs = np.empty(x.size*y.size)
trial_costs[:] = np.inf

def find_neighbours(node_index):
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

#trial.extend(find_neighbours(index_0)) #expand to index_0 -> known
#trial_costs[find_neighbours(index_0)] = cost_function[find_neighbours(index_0)]
for i in range(len(index_0)): #add known stuff
	trial.extend(find_neighbours(index_0[i])) #expand to index_0 -> known
	trial_costs[find_neighbours(index_0[i])] = cost_function[find_neighbours(index_0[i])]
known_costs[index_0] = 0
trial_costs[index_0] = np.inf
while len(known)!=I*J+1:
	#find cheapest trial
	idx = int(np.argmin(trial_costs)) #picks values that are already in known because of value-ties
	if app.ismember(idx,known):
		print "Something went wrong"
		print known
		print known_costs
		print trial
		print trial_costs
		print ss
	val = np.amin(trial_costs)
	trial_costs[idx] = np.inf #remove trial value
	trial.remove(idx) #remove trial
	known.append(idx)
	known.sort() #add to known
	known_costs[idx] = val #store known cost
	#find new neighbours
	neighbours0 = find_neighbours(idx)
	#neighbours = [ item for i,item in enumerate(neighbours0) if i not in known ]
	neighbours = [x for x in neighbours0 if x not in known]
	#print "n0:",neighbours0
	#print "k:",known
	#print "n:",neighbours
	#update trial things
	trial.extend(neighbours)
	trial.sort()
	trial_costs[neighbours] = np.minimum(val + cost_function[neighbours],trial_costs[neighbours])

fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot_surface(X,Y,np.reshape(known_costs,(I,J)),rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=0, antialiased=False)
fig1.suptitle("a1")
ax1.set_xlabel('x')
ax1.set_ylabel('y')
plt.show()
