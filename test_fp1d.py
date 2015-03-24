from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import time as time
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize as minimize
from matplotlib import cm
import quadrature_nodes as qn
import input_functions as iF
import solvers_1D as solve
quad_order = 500
gll_x = qn.GLL_points(quad_order) #quadrature nodes
gll_w = qn.GLL_weights(quad_order,gll_x)

#INPUTS
dx0 = 2*0.25
REFINEMENTS = 6
X_NAUGHT = 0.0
#constant coefficient test
velocity = -0.1
epsilon = 0.3
C = 1/0.00000224497
D = 0.1 #diffusion
########
xmin = 0-3
xmax = 3
T = 1
#set dx
dx = dx0
def index(i,k): 
	return int(i+(I)*k)
def ornstein_kernel(x,time):
	sig2 = 2*D/velocity * (1-np.exp(-2*velocity*time))
	return (1/(np.sqrt(np.pi*sig2))) * np.exp(-x**2/sig2)	

def heat_kernel(x,time):
	return 1/(np.sqrt(4*D*np.pi*time)) * np.exp( -(x)**2/(4*D*time) ) #heat kernel

def delta(x,x0,dx):
	output = np.zeros(x.size)
	for i in range(0,x.size):
		if x[i]==x0:
			output[i]=1/dx
	return output
def m0_init(x):
	output = np.zeros(x.size)
	for i in range(0,output.size):
		if abs(x[i])<epsilon:
			output[i] = C*np.exp(-1/(epsilon**2 - x[i]**2))
	return output

def m0_scalar(x):
	if abs(x)>=epsilon:
		return 0
	else:
		return C*np.exp(-1/(epsilon**2 - x**2))

def convolution(x,time): #alpha is the boundary of the compact domain
	output = np.zeros(x.size)
	for i in range(0,quad_order):
		output += gll_w[i]*ornstein_kernel(x-epsilon*gll_x[i],time)*m0_scalar(epsilon*gll_x[i]) #Ornstein test
		#output += gll_w[i]*m0_scalar(epsilon*gll_x[i])*heat_kernel(x-epsilon*gll_x[i],time) #constant coefficient test
	return epsilon*output

e1 = np.zeros(REFINEMENTS)
e2 = np.zeros(REFINEMENTS)
e3 = np.zeros(REFINEMENTS)
dexes = np.zeros(REFINEMENTS)
################################
#THIS IS WHERE WE NEED THE LOOP#
################################
for N in range(0,REFINEMENTS):
	dx = dx/2 #starts at dx=0.25
	dexes[N] = dx
	dt = .1*dx**2
	#CRUNCH
	print "(",dx,",",dt,")"
	dx2 = dx**2
	Nx = int(abs(xmax-xmin)/dx)+1
	Nt = int(T/dt)+1
	x = np.linspace(xmin,xmax,Nx)
	t = np.linspace(0,T,Nt)
	I = x.size #space
	K = t.size #time

	#INITIALISE STORAGE
	m1 = np.zeros(I) #distribution
	m2 = np.zeros(I) #distribution
	m3 = np.zeros(I) #distribution
	m_exact = np.zeros(I)
	
	#INITIAL/TERMINAL CONDITIONS, INITIAL GUESS and COPYING
	#m0 = delta(x,X_NAUGHT,dx)
	m0 = m0_init(x)
	m1 = np.copy(m0)
	m2 = np.copy(m0)
	m3 = np.copy(m0)
	m_exact = np.copy(m0)
	
	#SOLVE STUFF
	for k in range(0,K-1): #COMPUTE M WHERE WE DO NOT ALLOW AGENTS TO LEAVE SUCH THAT m(-1) = m(N+1) = 1 ALWAYS
		m1 = solve.fp_fd_centered(x,(k)*dt,m1,m0,dt,dx)
		m2 = solve.fp_fd_upwind(x,(k)*dt,m2,m0,dt,dx)
		m3 = solve.fp_fv(x,(k)*dt,m3,m0,dt,dx)
		#timer?
	#compute error in 2-norm
	#m_exact = convolution(x-velocity*T,T) #CONSTANT COEFFICIENT TEST
	#m_exact = heat_kernel(x-velocity*T,T) #the other one
	#m_exact = ornstein_kernel(x,T) #the other one
	m_exact = convolution(x,T)
	e1[N] = np.linalg.norm(m1-m_exact)
	e2[N] = np.linalg.norm(m2-m_exact)
	e3[N] = np.linalg.norm(m3-m_exact)

#print e1
#print e2
#print e3
#crunch the slopes and put in the figures
slope1, intercept = np.polyfit(np.log(dexes[1:]), np.log(e1[1:]), 1)
slope2, intercept = np.polyfit(np.log(dexes[1:]), np.log(e2[1:]), 1)
slope3, intercept = np.polyfit(np.log(dexes[1:]), np.log(e3[1:]), 1)

print slope1, slope2, slope3

fig4 = plt.figure(1)
str1 = "Centered FD, slope:", "%.2f" %slope1
str2 = "Upwind FD, slope:", "%.2f" %slope2
str3 = "Finite volume, slope:", "%.2f" %slope3
plt.loglog(dexes,e1,'o-',label=str1)
plt.loglog(dexes,e2,'o-',label=str2)
plt.loglog(dexes,e3,'o-',label=str3)
legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')
ax4 = fig4.add_subplot(111)
ax4.set_xlabel('Log10 of dx')
ax4.set_ylabel('Log10 of error')
plt.grid(True,which="both",ls="-")
ax4.invert_xaxis()
fig4.suptitle('Convergence of m(x,t)', fontsize=14)


fig3 = plt.figure(2)
plt.plot(x,(m1-m_exact),label="Centered FD")
plt.plot(x,(m2-m_exact),label="Upwind FD")
plt.plot(x,(m3-m_exact),label="Finite volume FD")
#plt.plot(m_exact,label="Exact solution")
legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')

fig2 = plt.figure(3)
plt.plot(x,m1,label="Centered FD")
plt.plot(x,m2,label="Upwind FD")
plt.plot(x,m3,label="Finite volume FD")
plt.plot(x,m_exact,label="Exact solution")
legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')
plt.show()








