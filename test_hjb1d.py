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
REFINEMENTS = 5
X_NAUGHT = 0.0
n = 2
tau = 1
omega = 1
DT = 1 #ratio as in dt = DT*dx(**2)
POINTS = 20
NICE_DIFFUSION = 1
cutoff = 0
xmax = np.pi/omega*(n-0.5)
xmin = -xmax
T = 1
#set dx
dx0 = abs(xmax-xmin)/POINTS
dx = dx0
def index(i,k): 
	return int(i+(I)*k)
def exact_solution(x,t):
	return np.exp(-tau*t)*np.sin(omega*x)
e1 = np.zeros(REFINEMENTS)
e1_1 = np.zeros(REFINEMENTS)
e1_inf = np.zeros(REFINEMENTS)
dexes = np.zeros(REFINEMENTS)
e2 = np.zeros(REFINEMENTS)
e2_1 = np.zeros(REFINEMENTS)
e2_inf = np.zeros(REFINEMENTS)
eu = np.zeros(REFINEMENTS)
eu_1 = np.zeros(REFINEMENTS)
eu_inf = np.zeros(REFINEMENTS)

################################
for N in range(0,REFINEMENTS):
	dx = dx/2
	dexes[N] = dx
	dt = DT*dx
	#CRUNCH
	
	dx2 = dx**2
	Nx = int(abs(xmax-xmin)/dx)
	Nt = int(np.ceil(T/dt))
	#hybrid search stuff
	Ns = 50
	n = 20
	tol = 1e-16#dx2
	search_space = np.linspace(-1,1,Ns)
	scatters = max(int( np.ceil( np.log(2/(tol*Ns))/np.log(n) ) ),1)
	#print np.log(2/(tol*200))/np.log(n)
	#/hybrid search stuff
	print "(dx,dt,scatters)=(",dx,",",dt,",",scatters,")"
	#print ss
	x = np.linspace(xmin,xmax,Nx)
	I = x.size #space
	#INITIALISE STORAGE
	u = np.zeros(I) #distribution
	u_exact = np.zeros(I)
	#INITIAL/TERMINAL CONDITIONS, INITIAL GUESS and COPYING
	uT = exact_solution(x,T)
	u = exact_solution(x,T)
	u2 = exact_solution(x,T)
	#SOLVE STUFF
	t0 = time.time()
	for k in range(Nt-1,-1,-1): #COMPUTE M WHERE WE DO NOT ALLOW AGENTS TO LEAVE SUCH THAT m(-1) = m(N+1) = 1 ALWAYS
		a_tmp = -np.gradient(u,dx)
		a_tmp2 = solve.control_hybrid(x,k*dt,u,u,dt,dx,search_space,tol,n,scatters)
		#a_tmp2 = solve.control_hybrid(x,k*dt,u,u,dt,dx,search_space,tol,n,2)
		#a_tmp3 = solve.control_hybrid(x,k*dt,u,u,dt,dx,search_space,tol,n,15)
		print np.linalg.norm(a_tmp-a_tmp2)
		#plt.plot(x,a_tmp-a_tmp2,label="Exact")
		#plt.show()
		#if k==Nt-1:
		#	print a_tmp
		#	print ss
		if NICE_DIFFUSION==0:
			u = solve.hjb_kushner_mod(x,k*dt,u,u,a_tmp,dt,dx) #implicit
		else:
			if k==Nt-1:
				LHS_HJB = mg.hjb_diffusion(k*dt,x,a_tmp,u,dt,dx)
			RHS_HJB = mg.hjb_convection(k*dt,x,a_tmp,u,dt,dx)
			RHS_HJB2 = mg.hjb_convection(k*dt,x,a_tmp2,u2,dt,dx)
			Ltmp = iF.L_global(k*dt,x,a_tmp,u)
			Ltmp2 = iF.L_global(k*dt,x,a_tmp2,u2)
			u = sparse.linalg.spsolve(LHS_HJB,RHS_HJB*u+dt*Ltmp)
			u2 = sparse.linalg.spsolve(LHS_HJB,RHS_HJB2*u2+dt*Ltmp2)
	print "Time spent:",time.time()-t0
	#compute error in 2-norm
	u_exact = exact_solution(x,0)
	e1[N] = np.linalg.norm(u-u_exact)*np.sqrt(dx)
	e1_1[N] = np.linalg.norm(u-u_exact,ord=1)*dx
	e1_inf[N] = np.linalg.norm(u-u_exact,ord=np.inf)
	eu[N] = np.linalg.norm(u-u2)*np.sqrt(dx)
	eu_1[N] = np.linalg.norm(u-u2,ord=1)*dx
	eu_inf[N] = np.linalg.norm(u-u2,ord=np.inf)
	e2[N] = np.linalg.norm(u2-u_exact)*np.sqrt(dx)
	e2_1[N] = np.linalg.norm(u2-u_exact,ord=1)*dx
	e2_inf[N] = np.linalg.norm(u2-u_exact,ord=np.inf)

#print e1
#print e2
#print e3
#crunch the slopes and put in the figures
slope1, intercept = np.polyfit(np.log(dexes[cutoff:]), np.log(e1[cutoff:]), 1)
slope1_1, intercept = np.polyfit(np.log(dexes[cutoff:]), np.log(e1_1[cutoff:]), 1)
slope1_inf, intercept = np.polyfit(np.log(dexes[cutoff:]), np.log(e1_inf[cutoff:]), 1)
slopeu, intercept = np.polyfit(np.log(dexes[cutoff:]), np.log(eu[cutoff:]), 1)
slopeu_1, intercept = np.polyfit(np.log(dexes[cutoff:]), np.log(eu_1[cutoff:]), 1)
slopeu_inf, intercept = np.polyfit(np.log(dexes[cutoff:]), np.log(eu_inf[cutoff:]), 1)
slope2, intercept = np.polyfit(np.log(dexes[cutoff:]), np.log(e2[cutoff:]), 1)
slope2_1, intercept = np.polyfit(np.log(dexes[cutoff:]), np.log(e2_1[cutoff:]), 1)
slope2_inf, intercept = np.polyfit(np.log(dexes[cutoff:]), np.log(e2_inf[cutoff:]), 1)
#x = x[cutoffindexmin:cutoffindexmax]

fig4 = plt.figure(6)
str1 = "1-norm slope:", "%.2f" %slope1_1
str2 = "2-norm slope:", "%.2f" %slope1
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
fig4.suptitle('Convergence rates of u(x,t), exact control', fontsize=14)

fig5 = plt.figure(7)
str1 = "1-norm slope:", "%.2f" %slope2_1
str2 = "2-norm slope:", "%.2f" %slope2
str3 = "inf-norm slope:", "%.2f" %slope2_inf
plt.loglog(dexes,e2,'o-',label=str2)
plt.loglog(dexes,e2_1,'o-',label=str1)
plt.loglog(dexes,e2_inf,'o-',label=str3)
legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')
ax5 = fig5.add_subplot(111)
ax5.set_xlabel('Log10 of dx')
ax5.set_ylabel('Log10 of error')
plt.grid(True,which="both",ls="-")
ax5.invert_xaxis()
fig5.suptitle('Convergence rates of u(x,t), computed control', fontsize=14)

fig6 = plt.figure(8)
str1 = "1-norm slope:", "%.2f" %slope1_1
str2 = "2-norm slope:", "%.2f" %slope1
str3 = "inf-norm slope:", "%.2f" %slope1_inf
plt.loglog(dexes,eu,'o-',label=str2)
plt.loglog(dexes,eu_1,'o-',label=str1)
plt.loglog(dexes,eu_inf,'o-',label=str3)
legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')
ax6 = fig6.add_subplot(111)
ax6.set_xlabel('Log10 of dx')
ax6.set_ylabel('Log10 of error')
plt.grid(True,which="both",ls="-")
ax6.invert_xaxis()
fig6.suptitle('Deviation between exact and computed control', fontsize=14)


fig3 = plt.figure(2)
plt.plot(x,(u-u_exact),label="Centered FD")
legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')
fig3.suptitle('Deviation from true solution', fontsize=14)

fig2 = plt.figure(3)
plt.plot(x,u,label="Computed solution")
plt.plot(x,u_exact,label="Exact solution")
fig2.suptitle('Solutions', fontsize=18)
legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')


plt.show()








