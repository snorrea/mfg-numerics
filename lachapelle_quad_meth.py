
from __future__ import division
import numpy as np
import scipy.weave as weave
import matplotlib.pyplot as plt
import time as time
from mpl_toolkits.mplot3d import Axes3D

#in this one we aim to not use so much fucking space

#INPUTS
dx = 1/50 #these taken from Gueant's paper
dt = 1/100 
xmin = 0
xmax = 1
T = 1
Niter = 15 #maximum number of iterations
tolerance = 1e-6

#CRUNCH
dx2 = dx**2
Nx = int(abs(xmax-xmin)/dx)
Nt = int(T/dt) 
I = int(Nt)+1
J = int(Nx)+1
x = np.arange(xmin,xmax+dx,dx)
t = np.arange(0,T+dt,dt)
def index(i,j): #this is a jolly source of errors, no more, probably still
	return int(j+(J)*i)

def convection(matrix,i,j,pm,velocity): #this could be a source of errors
	if pm==1:
		if velocity < 0:
			if j==J-1:
				return matrix[index(i,j-1)]
			else:
				return matrix[index(i,j+1)]
		else:
			return matrix[index(i,j)]
	elif pm==-1:
		if velocity < 0:
			if j==0:
				return matrix[index(i,j+1)]
			else:
				return matrix[index(i,j-1)]
		else:
			return matrix[index(i,j)]

###input functions and constants
theta = 1 #used to solve for alpha and stuff
#constants for g
c0 = 1
c1 = 1
c2 = 1
#constant for f
beta = 0.7
#the rest
sigma2 = 0.14#0.8**2
m0 = 1-0.2*np.cos(np.pi*x) #gueant's original
def cost(a):
	return a**2/2
def g(t,x,m):
	return c0*x/(c1+c2*m)
def p(t):	#price of electricity
	return 0.5
def f(t,x):
	return p(t)*(1-beta*x)
vT = np.zeros(J)

#initialise solution VECTORS WHY WOULD YOU USE MATRICES
v = np.empty((I*J)) #potential
m = np.empty((I*J)) #distribution
mhat = np.empty((I*J))
mtilde = np.empty((I*J))
a = np.empty((I*J)) #control
v_old = np.empty((I*J))
a_old = np.empty((I*J)) 
bconst = np.empty((I*J))
mu = np.empty((I*J))
mutilde = np.empty((I*J))
muhat = np.empty((I*J))
xi = np.empty((I*J))
betaconst = np.empty((I*J))
cconst = np.empty((I*J))
print "Initialising done, now crunching."

#crank out some constants
delta = 2/(theta+1)
aconst = 1+theta

#initial value for a
a_old = np.ones((I*J))*0.1

#initial distribution
m_old = np.empty((I*J))
for i in range (0,I): #set initial guess
	m_old[(i*J):(J*(i+1))]=np.copy(m0);
m = np.copy(m_old)
for k in range (0,Niter):
	titer = time.time()
	#compute v
	#print v[J*I-1]
	v[J*I-J-1:J*I-1] = vT
	for i in range (I-2,-1,-1):
		for j in range (0,J): #this needs the whole cost function thingamajig
			if j==0:
				v[index(i,j)] = v[index(i+1,j)] + dt*(  sigma2/(2*dx2)*(v[index(i+1,j+1)]+v[index(i+1,j+1)]-2*v[index(i+1,j)]) + 0.25*(a_old[index(i,j)]**2 + a_old[index(i,j+1)]**2 ) + 1/dx * ( convection(v,i+1,j,1,a_old[index(i,j+1)]) + convection(v,i+1,j,-1,a_old[index(i,j)])  ) )
			elif j==J-1:
				v[index(i,j)] = v[index(i+1,j)] + dt*(  sigma2/(2*dx2)*(v[index(i+1,j-1)]+v[index(i+1,j-1)]-2*v[index(i+1,j)]) + 0.25*(a_old[index(i,j)]**2 + a_old[index(i,j-1)]**2 ) + 1/dx * ( convection(v,i+1,j,1,a_old[index(i,j-1)]) + convection(v,i+1,j,-1,a_old[index(i,j)])  ) )
			else: #everything that has to do with a_old is a possible source of errors... hohohoho
				v[index(i,j)] = v[index(i+1,j)] + dt*(  sigma2/(2*dx2)*(v[index(i+1,j+1)]+v[index(i+1,j-1)]-2*v[index(i+1,j)]) + 0.25*(a_old[index(i,j)]**2 + a_old[index(i,j+1)]**2 ) + 1/dx * ( convection(v,i+1,j,1,a_old[index(i,j+1)]) + convection(v,i+1,j,-1,a_old[index(i,j)])  ) )

	###compute m iteratively from m_old
	
	#compute mtilde and mhat
	for i in range(0,I-1):
		for j in range(0,J): #set mtilde
			if a_old[index(i,j)] < 0:
				mhat[index(i,j)] = m_old[index(i,j)]
				if j==J-1:
					mtilde[index(i,j)] = m_old[index(i,j-1)]
				else:
					mtilde[index(i,j)] = m_old[index(i,j+1)]
			else:
				mtilde[index(i,j)] = m_old[index(i,j)]
				if j==J-1:
					mhat[index(i,j)] = m_old[index(i,j-1)]
				else:
					mhat[index(i,j)] = m_old[index(i,j+1)]
		#now compute all the helping constants....	
		for j in range(0,J-1):
			#print mhat[index(i,j)],m_old[index(i,j)],m_old[index(i,j+1)]
			#print i,j,m0.size
			#print mhat[index(i,j)]
			mu[index(i,j)] = 2*mhat[index(i,j)]/(m_old[index(i,j)]+m_old[index(i,j+1)]) #explosion
			muhat[index(i,j)] = 2*mtilde[index(i,j)]/(m_old[index(i,j)]+m_old[index(i,j+1)])
			if i==I:
				bconst[index(i,j)] = -2*(theta*a_old[index(i,j)] + mu[index(i,j)]*(v[index(i-1,j+1)]-v[index(i-1,j)])/dx)
				cconst[index(i,j)] = (theta-1)*(a_old[index(i,j)])**2 + 2*a_old[index(i,j)]*muhat[index(i,j)]*(v[index(i-1,j+1)]-v[index(i-1,j)])/dx
			else:
				bconst[index(i,j)] = -2*(theta*a_old[index(i,j)] + mu[index(i,j)]*(v[index(i+1,j+1)]-v[index(i+1,j)])/dx)
				cconst[index(i,j)] = (theta-1)*(a_old[index(i,j)])**2 + 2*a_old[index(i,j)]*muhat[index(i,j)]*(v[index(i+1,j+1)]-v[index(i+1,j)])/dx

			#things
			xi[index(i,j)] = (1-delta)*a_old[index(i,j)] + delta*muhat[index(i,j)]*(v[index(i+1,j+1)]-v[index(i+1,j)])/dx
			betaconst[index(i,j)] = ( -bconst[index(i,j)] - np.sign(a_old[index(i,j)])*np.sqrt( (bconst[index(i,j)])**2 - 4*aconst*cconst[index(i,j)] ) )/(2*a_old[index(i,j)])

		for j in range(0,J):
			if a_old[index(i,j)]*xi[index(i,j)] >= 0:
				a[index(i,j)] = xi[index(i,j)]
			else:
				a[index(i,j)] = betaconst[index(i,j)]

		for j in range(0,J):
			if j==0:
				m[index(i,j)] = m_old[index(i,j)] + sigma2*dt/(2*dx2) * ( m_old[index(i,j+1)] + m_old[index(i,j+1)] - 2*m_old[index(i,j)] ) - dt/dx*(convection(m_old,i,j,1,a[index(i,j+1)]) - convection(m_old,i,j,-1,a[index(i,j+1)]))
			elif j==J-1:
				m[index(i,j)] = m_old[index(i,j)] + sigma2*dt/(2*dx2) * ( m_old[index(i,j-1)] + m_old[index(i,j-1)] - 2*m_old[index(i,j)] ) - dt/dx*(convection(m_old,i,j,1,a[index(i,j-1)]) - convection(m_old,i,j,-1,a[index(i,j-1)]))
			else:
				m[index(i,j)] = m_old[index(i,j)] + sigma2*dt/(2*dx2) * ( m_old[index(i,j+1)] + m_old[index(i,j-1)] - 2*m_old[index(i,j)] ) - dt/dx*(convection(m_old,i,j,1,a[index(i,j+1)]) - convection(m_old,i,j,-1,a[index(i,j-1)]))
	
	
	#compute other constants
	#for i in range(0,I-1): #things are going kinda bananas in this (why am I not surprised)
		#for j in range(0,J-1):
			

	#thank god that's over! compute next a
	#for i in range(0,I):
		
	
	#compute updated m
	#for i in range(0,I):
		
	#check improvement
	deltaeverything = max(abs( a-a_old ) + abs(m-m_old))

	#update values
	m_old = np.copy(m)
	a_old = np.copy(a)
	print "Iteration number", k+1, "completed, used time", time.time()-titer , "with change", deltaeverything


#resolve solutions into a mesh
#m = np.exp((u-v)/(sigma2)) #GET OUT OF HERE CRIMINAL SCUM
msoln = np.empty((I,J))
vsoln = np.empty((I,J))
for i in range (0,I):
	for j in range (0,J):
		msoln[i,j] = m[index(i,j)]
		vsoln[i,j] = v[index(i,j)]
#shit attempt at plotting
Xplot, Tplot = np.meshgrid(x,t)
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot_wireframe(Xplot,Tplot,msoln,rstride=15,cstride=5)
ax1.set_xlabel('x')
ax1.set_ylabel('t')
ax1.set_zlabel('m(x,t)')
plt.show()





