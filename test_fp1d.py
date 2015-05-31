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
import matrix_gen1d as mg
from scipy import sparse
quad_order = 500
gll_x = qn.GLL_points(quad_order) #quadrature nodes
gll_w = qn.GLL_weights(quad_order,gll_x)

#INPUTS
dx0 = 2*0.1
REFINEMENTS = 7
X0 = -0.5
cutoff = 0
#constant coefficient test
DT = .05
velocity = -2.5
D = 0.1 #diffusion
#for the initial condition
epsilon = 0.3
C = 1/0.00000224497
########
xmin = 0-10
xmax = 10
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
e4 = np.zeros(REFINEMENTS)
e1_1 = np.zeros(REFINEMENTS)
e2_1 = np.zeros(REFINEMENTS)
e3_1 = np.zeros(REFINEMENTS)
e4_1 = np.zeros(REFINEMENTS)
e1_inf = np.zeros(REFINEMENTS)
e2_inf = np.zeros(REFINEMENTS)
e3_inf = np.zeros(REFINEMENTS)
e4_inf = np.zeros(REFINEMENTS)
dexes = np.zeros(REFINEMENTS)
################################
#THIS IS WHERE WE NEED THE LOOP#
################################
for N in range(0,REFINEMENTS):
	dx = dx/2 #starts at dx=0.25
	dexes[N] = dx
	dt = DT*dx
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
	m4 = np.zeros(I) #distribution
	m_exact = np.zeros(I)
	
	#INITIAL/TERMINAL CONDITIONS, INITIAL GUESS and COPYING
	#m0 = delta(x,X_NAUGHT,dx)
	m0 = m0_init(x)
	m1 = np.copy(m0)
	m2 = np.copy(m0)
	m3 = np.copy(m0)
	m4 = np.copy(m0)
	
	#SOLVE STUFF
	t0 = time.time()
	for k in range(0,K-1): #COMPUTE M WHERE WE DO NOT ALLOW AGENTS TO LEAVE SUCH THAT m(-1) = m(N+1) = 1 ALWAYS
		#m1 = solve.fp_fd_centered_mod(x,(k)*dt,m1,m0,dt,dx)
		#m2 = solve.fp_fd_upwind_mod(x,(k)*dt,m2,m0,dt,dx)
		#m3 = solve.fp_fd_upwind_visc(x,(k)*dt,m2,m0,dt,dx)
		#m4 = solve.fp_fv_mod(x,(k)*dt,m4,m0,dt,dx)
		if k==0:
			LHS_centered = mg.fp_fd_centered_diffusion(k*dt,x,m1,dt,dx)
			RHS_centered = mg.fp_fd_centered_convection(k*dt,x,m1,dt,dx)
			RHS_upwind = mg.fp_fd_upwind_convection(k*dt,x,m2,dt,dx)
			LHS_fv = mg.fp_fv_diffusion(k*dt,x,m4,dt,dx)
			RHS_fv = mg.fp_fv_convection_classic(k*dt,x,m4,dt,dx)
			RHS_fv2 = mg.fp_fv_convection_interpol(k*dt,x,m4,dt,dx)
		#add direchlet boundary... kinda fake, but fuck it
		#LHS_centered,RHS_centered,m1 = mg.add_direchlet_boundary(LHS_centered,RHS_centered,m1,0)
		#LHS_centered,RHS_upwind,m2 = mg.add_direchlet_boundary(LHS_centered,RHS_centered,m2,0)
		#LHS_fv,RHS_fv,m3 = mg.add_direchlet_boundary(LHS_fv,RHS_fv,m3,0)
		#LHS_fv,RHS_fv2,m4 = mg.add_direchlet_boundary(LHS_fv,RHS_fv2,m4,0)
		m1 = sparse.linalg.spsolve(LHS_centered,RHS_centered*np.ravel(m1))
		m2 = sparse.linalg.spsolve(LHS_centered,RHS_upwind*np.ravel(m2))
		m3 = sparse.linalg.spsolve(LHS_fv,RHS_fv2*np.ravel(m3)) #interpolated
		m4 = sparse.linalg.spsolve(LHS_fv,RHS_fv*np.ravel(m4))
		m1[0] = 0
		m1[-1] = 0
		m2[0] = 0
		m2[-1] = 0
		m3[0] = 0
		m3[-1] = 0
		m4[0] = 0
		m4[-1] = 0
		#print m4
	print "Time spent:",time.time()-t0
	#compute error in 2-norm
	#print ss
	#m_exact = convolution(x-velocity*T,T) #CONSTANT COEFFICIENT TEST
	m_exact = convolution(x,T) #Ornstein test
	e1[N] = np.linalg.norm(m1-m_exact)*np.sqrt(dx)
	e2[N] = np.linalg.norm(m2-m_exact)*np.sqrt(dx)
	e3[N] = np.linalg.norm(m3-m_exact)*np.sqrt(dx)
	e4[N] = np.linalg.norm(m4-m_exact)*np.sqrt(dx)
	e1_1[N] = np.linalg.norm(m1-m_exact,ord=1)*dx
	e2_1[N] = np.linalg.norm(m2-m_exact,ord=1)*dx
	e3_1[N] = np.linalg.norm(m3-m_exact,ord=1)*dx
	e4_1[N] = np.linalg.norm(m4-m_exact,ord=1)*dx
	e1_inf[N] = np.linalg.norm(m1-m_exact,ord=np.inf)
	e2_inf[N] = np.linalg.norm(m2-m_exact,ord=np.inf)
	e3_inf[N] = np.linalg.norm(m3-m_exact,ord=np.inf)
	e4_inf[N] = np.linalg.norm(m4-m_exact,ord=np.inf)

#print e1
#print e2
#print e3
#crunch the slopes and put in the figures
print e4
print e4_1
print e4_inf
slope1, intercept = np.polyfit(np.log(dexes[cutoff:]), np.log(e1[cutoff:]), 1)
slope2, intercept = np.polyfit(np.log(dexes[cutoff:]), np.log(e2[cutoff:]), 1)
slope3, intercept = np.polyfit(np.log(dexes[cutoff:]), np.log(e3[cutoff:]), 1)
slope4, intercept = np.polyfit(np.log(dexes[cutoff:]), np.log(e4[cutoff:]), 1)
slope1_1, intercept = np.polyfit(np.log(dexes[cutoff:]), np.log(e1_1[cutoff:]), 1)
slope2_1, intercept = np.polyfit(np.log(dexes[cutoff:]), np.log(e2_1[cutoff:]), 1)
slope3_1, intercept = np.polyfit(np.log(dexes[cutoff:]), np.log(e3_1[cutoff:]), 1)
slope4_1, intercept = np.polyfit(np.log(dexes[cutoff:]), np.log(e4_1[cutoff:]), 1)
slope1_inf, intercept = np.polyfit(np.log(dexes[cutoff:]), np.log(e1_inf[cutoff:]), 1)
slope2_inf, intercept = np.polyfit(np.log(dexes[cutoff:]), np.log(e2_inf[cutoff:]), 1)
slope3_inf, intercept = np.polyfit(np.log(dexes[cutoff:]), np.log(e3_inf[cutoff:]), 1)
slope4_inf, intercept = np.polyfit(np.log(dexes[cutoff:]), np.log(e4_inf[cutoff:]), 1)



fig4 = plt.figure(6)
str1 = "Centered FD, slope:", "%.2f" %slope1
str2 = "Upwind FD, slope:", "%.2f" %slope2
str3 = "Finite volume with interpolation, slope:", "%.2f" %slope3
str4 = "Finite volume, slope:", "%.2f" %slope4
plt.loglog(dexes,e1,'o-',label=str1)
plt.loglog(dexes,e2,'o-',label=str2)
plt.loglog(dexes,e3,'o-',label=str3)
plt.loglog(dexes,e4,'o-',label=str4)
legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')
ax4 = fig4.add_subplot(111)
ax4.set_xlabel('Log10 of dx')
ax4.set_ylabel('Log10 of error')
plt.grid(True,which="both",ls="-")
ax4.invert_xaxis()
fig4.suptitle('Convergence of m(x,t) in 2-norm', fontsize=14)


fig3 = plt.figure(2)
plt.plot(x,(m1-m_exact),label="Centered FD")
plt.plot(x,(m2-m_exact),label="Upwind FD")
plt.plot(x,(m3-m_exact),label="Finite volume with interpolation")
plt.plot(x,(m4-m_exact),label="Finite volume")
legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')
fig3.suptitle('Deviation from true solution', fontsize=14)

fig2 = plt.figure(3)
plt.plot(x,m1,label="Centered FD")
plt.plot(x,m2,label="Upwind FD")
plt.plot(x,m3,label="Finite volume with interpolation")
plt.plot(x,m4,label="Finite volume")
plt.plot(x,m_exact,label="Exact solution")
fig2.suptitle('Solutions', fontsize=14)
legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')

#1-norm
fig5 = plt.figure(5)
str1 = "Centered FD, slope:", "%.2f" %slope1_1
str2 = "Upwind FD, slope:", "%.2f" %slope2_1
str3 = "Finite volume with interpolation, slope:", "%.2f" %slope3_1
str4 = "Finite volume, slope:", "%.2f" %slope4_1
plt.loglog(dexes,e1_1,'o-',label=str1)
plt.loglog(dexes,e2_1,'o-',label=str2)
plt.loglog(dexes,e3_1,'o-',label=str3)
plt.loglog(dexes,e4_1,'o-',label=str4)
legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')
ax4 = fig5.add_subplot(111)
ax4.set_xlabel('Log10 of dx')
ax4.set_ylabel('Log10 of error')
plt.grid(True,which="both",ls="-")
ax4.invert_xaxis()
fig5.suptitle('Convergence of m(x,t) in 1-norm', fontsize=14)
#inf-norm
fig6 = plt.figure(7)
str1 = "Centered FD, slope:", "%.2f" %slope1_inf
str2 = "Upwind FD, slope:", "%.2f" %slope2_inf
str3 = "Finite volume with interpolation:", "%.2f" %slope3_inf
str4 = "Finite volume, slope:", "%.2f" %slope4_inf
plt.loglog(dexes,e1_inf,'o-',label=str1)
plt.loglog(dexes,e2_inf,'o-',label=str2)
plt.loglog(dexes,e3_inf,'o-',label=str3)
plt.loglog(dexes,e4_inf,'o-',label=str4)
legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')
ax4 = fig6.add_subplot(111)
ax4.set_xlabel('Log10 of dx')
ax4.set_ylabel('Log10 of error')
plt.grid(True,which="both",ls="-")
ax4.invert_xaxis()
fig6.suptitle('Convergence of m(x,t) in inf-norm', fontsize=14)

fig9000 = plt.figure(9)
plt.plot(e3_inf/e4_inf, label="Inf norm")
plt.plot(e3_1/e4_1, label="1 norm")
plt.plot(e3/e4, label="2 norm")
legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')

plt.show()








