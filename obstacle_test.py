import applications as app
import matplotlib.pyplot as plt
import numpy as np

#inputs
Nx = 1001
Ny = 1001
xmin = 0
xmax = 10
ymin = 0
ymax = 10

#crunch
x = np.linspace(xmin,xmax,Nx)
y = np.linspace(ymin,ymax,Ny)
I = x.size
J = y.size
south = range(0,I)
west = range(0,I*J,I)
north = range(I*J-I,I*J)
east = range(I-1,I*J,I)
nulled = None

#obstacles
xi = [2, 2]
xa = [6, 3]
yi = [4, 1]
ya = [6, 2]

for i in range(len(xi)):
	south,north,west,east,nulled = app.add_obstacle(x,y,(xi[i],xa[i],yi[i],ya[i]),south,north,west,east,nulled)
x,y = np.meshgrid(x,y)
x = np.ravel(x)
y = np.ravel(y)
fig1 = plt.figure(1)
plt.plot(x[south],y[south],'.b')
plt.plot(x[north],y[north],'.b')
plt.plot(x[west],y[west],'.b')
plt.plot(x[east],y[east],'.b')
plt.plot(x[nulled],y[nulled],'.r')
plt.show()
