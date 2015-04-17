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
import scipy.sparse as sparse
import matrix_gen1d as mg
quad_order = 500
gll_x = qn.GLL_points(quad_order) #quadrature nodes
gll_w = qn.GLL_weights(quad_order,gll_x)

#INPUTS
dx0 = 2*0.1
REFINEMENTS = 6
X_NAUGHT = 0.0
omega = 10
tau = 5
DT = 1 #ratio as in dt = DT*dx(**2)
NICE_DIFFUSION = 1
xmin = 0-8
xmax = 8
cutoffmax = 6
cutoffmin = -6
T = 1
#set dx
dx = dx0
def index(i,k): 
	return int(i+(I)*k)
def exact_solution(x,t):
	return np.exp(-tau*t)*np.sin(omega*x)
e1 = np.zeros(REFINEMENTS)
e1_1 = np.zeros(REFINEMENTS)
e1_inf = np.zeros(REFINEMENTS)
dexes = np.zeros(REFINEMENTS)

#STUFF TO MINIMIZE
alpha_upper = 1
alpha_lower = -1
N = 60 #searchpoints
min_tol = 1e-6#tolerance#1e-5 #tolerance for minimum
min_left = alpha_lower #search region left
min_right = alpha_upper #search region right
relation = 2
scatters = int(np.ceil(np.log((min_right-min_left)/min_tol)/np.log(N)))
#scatters2 = int(1 + np.ceil(np.log((min_right-min_left)/(N*min_tol))/np.log(relation)))
xpts_search = np.linspace(min_left,min_right,N)
fpts = np.empty((xpts_search.size,1))
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
	cutoffindexmax = Nx-abs((xmax-cutoffmax)/dx)
	cutoffindexmin = abs((xmin-cutoffmin)/dx)
	#print cutoffindexmin,cutoffindexmax,Nx
	#INITIALISE STORAGE
	u = np.zeros(I) #distribution
	u_exact = np.zeros(I)
	
	#INITIAL/TERMINAL CONDITIONS, INITIAL GUESS and COPYING
	uT = exact_solution(x,T)
	u = exact_solution(x,T)
	#plt.plot(u)
	#plt.show()
	
	#SOLVE STUFF
	t0 = time.time()
	for k in range(K-2,-1,-1): #COMPUTE M WHERE WE DO NOT ALLOW AGENTS TO LEAVE SUCH THAT m(-1) = m(N+1) = 1 ALWAYS
		#print 1-k/K
		a_tmp = -np.gradient(u,dx)
		#a_tmp = solve.control_general(x,k*dt,u,u,dt,dx,xpts_search,N,scatters)
		#u = solve.hjb_kushner_mod(x,k*dt,u,u,a_tmp,dt,dx)
		if NICE_DIFFUSION==0:
			u = solve.hjb_kushner_mod(x,k*dt,u,u,a_tmp,dt,dx) #implicit
		else:
			if k==K-2:
				LHS_HJB = mg.hjb_diffusion(k*dt,x,a_tmp,u,dt,dx)
			RHS_HJB = mg.hjb_convection(k*dt,x,a_tmp,u,dt,dx)
			Ltmp = iF.L_global(k*dt,x,a_tmp,u)
			#print RHS_HJB*u_last
			u = sparse.linalg.spsolve(LHS_HJB,RHS_HJB*u+dt*Ltmp)
		#if k==3*(K-1)/4:
		#	plt.plot(x,u)
		#	plt.show()
	print "Time spent:",time.time()-t0
	#compute error in 2-norm
	u_exact = exact_solution(x,0)
	#trim
	u_exact = u_exact[cutoffindexmin:cutoffindexmax]
	u = u[cutoffindexmin:cutoffindexmax]
	e1[N] = np.linalg.norm(u-u_exact)*dx
	e1_1[N] = np.linalg.norm(u-u_exact,ord=1)*dx
	e1_inf[N] = np.linalg.norm(u-u_exact,ord=np.inf)

#print e1
#print e2
#print e3
#crunch the slopes and put in the figures
slope1, intercept = np.polyfit(np.log(dexes[1:]), np.log(e1[1:]), 1)
slope1_1, intercept = np.polyfit(np.log(dexes[1:]), np.log(e1_1[1:]), 1)
slope1_inf, intercept = np.polyfit(np.log(dexes[1:]), np.log(e1_inf[1:]), 1)
x = x[cutoffindexmin:cutoffindexmax]

fig4 = plt.figure(6)
str1 = "2-norm slope:", "%.2f" %slope1
str2 = "1-norm slope:", "%.2f" %slope1_1
str3 = "inf-norm slope:", "%.2f" %slope1_inf
plt.loglog(dexes,e1,'o-',label=str2)
plt.loglog(dexes,e1_1,'o-',label=str1)
plt.loglog(dexes,e1_inf,'o-',label=str3)
legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')
ax4 = fig4.add_subplot(111)
ax4.set_xlabel('Log10 of dx')
ax4.set_ylabel('Log10 of error')
plt.grid(True,which="both",ls="-")
ax4.invert_xaxis()
fig4.suptitle('Convergence rates of u(x,t)', fontsize=14)

fig3 = plt.figure(2)
plt.plot(x,(u-u_exact),label="Centered FD")
legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')
fig3.suptitle('Deviation from true solution', fontsize=14)

fig2 = plt.figure(3)
plt.plot(x,u,label="Centered FD")
plt.plot(x,u_exact,label="Exact solution")
fig2.suptitle('Solutions', fontsize=14)
legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')


plt.show()








