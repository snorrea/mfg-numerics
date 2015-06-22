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
dx0 = 2*0.2
REFINEMENTS = 4
cutoff = 0

########
xmin = 0
xmax = 1
DT = 0.25 #time ratio
DM = .001 #minimiser tolerance coefficient
DM_E = 1 #minimiser tolerance exponent
T = 1
#set dx
dx = dx0
NICE_DIFFUSION = 0
iterations = 100

#########
alpha_upper = 1.5
alpha_lower = -1.5
#STUFF TO MINIMIZE
min_left = alpha_lower #search region left
min_right = alpha_upper #search region right
Ns = int(np.ceil(abs(alpha_upper-alpha_lower)/dx) + 1)
xpts_scatter = np.linspace(min_left,min_right,Ns)

#########

e1_hjb1 = np.zeros(REFINEMENTS-1)
e1_fp1 = np.zeros(REFINEMENTS-1)
e2_hjb1 = np.zeros(REFINEMENTS-1)
e2_fp1 = np.zeros(REFINEMENTS-1)
einf_hjb1 = np.zeros(REFINEMENTS-1)
einf_fp1 = np.zeros(REFINEMENTS-1)
dexes = np.zeros(REFINEMENTS)

m_solns = [None]*REFINEMENTS
u_solns = [None]*REFINEMENTS
x_solns = [None]*REFINEMENTS

################################
#THIS IS WHERE WE NEED THE LOOP#
################################
for N in range(0,REFINEMENTS):
	dx = dx/2 #starts at dx=0.25
	dexes[N] = dx
	dt = DT*dx
	#CRUNCH
	dx2 = dx**2
	Nx = int(abs(xmax-xmin)/dx)+1
	Nt = int(T/dt)+1
	x = np.linspace(xmin,xmax,Nx)
	t = np.linspace(0,T,Nt)
	I = x.size #space
	K = t.size #time
	#MINIMISE
	Ns = int(np.ceil(abs(alpha_upper-alpha_lower)/dx) + 1)
	min_tol = DM*dx**DM_E
	scatters = int(np.ceil( np.log((min_right-min_left)/(min_tol*Ns))/np.log(Ns/2) ))
	##########
	print "(",dx,",",dt,")"
	m = np.zeros((I,K))
	u = np.empty((I,K))
	a = np.empty((I,K-1))
	m_old = np.copy(m)
	#INITIAL/TERMINAL CONDITIONS, INITIAL GUESS and COPYING
	#m0 = delta(x,X_NAUGHT,dx)
	for k in range(0,Nt):
		m[:,k] = iF.initial_distribution(x)
		#m_old[:,k] = iF.initial_distribution(x)
		u[:,k] = iF.G(x,x)
	for KELL in range(iterations):
		#SOLVE STUFF
		t0 = time.time()
		for k in range(Nt-2,-1,-1): #COMPUTE M WHERE WE DO NOT ALLOW AGENTS TO LEAVE SUCH THAT m(-1) = m(N+1) = 1 ALWAYS
			a[:,k] = solve.control_general(x,k*dt,u[:,k+1],m[:,k],dt,dx,xpts_scatter,Ns,scatters)
			if NICE_DIFFUSION==0:
				u[:,k] = solve.hjb_kushner_mod(x,k*dt,u[:,k+1],m[:,k],a[:,k],dt,dx) #implicit
			else:
				if k==Nt-2:
					LHS_HJB = mg.hjb_diffusion(k*dt,x,a[:,k],dt,dx)
				RHS_HJB = mg.hjb_convection(k*dt,x,a[:,k],dt,dx)
				Ltmp = iF.L_global(k*dt,x,a[:,k],m[:,k],u)
				u[:,k] = sparse.linalg.spsolve(LHS_HJB,RHS_HJB*u[:,k+1]+dt*Ltmp)
		#SOLVE STUFF
		#t0 = time.time()
		m_old = np.copy(m)
		for k in range(0,K-1): #COMPUTE M WHERE WE DO NOT ALLOW AGENTS TO LEAVE SUCH THAT m(-1) = m(N+1) = 1 ALWAYS
			if NICE_DIFFUSION==0:
				m[:,k+1] = solve.fp_fv_mod(x,k*dt,m[:,k],a[:,k],dt,dx)
			else:
				if k==0:
					LHS_fv = mg.fp_fv_diffusion(k*dt,x,a[:,k],dt,dx)
				RHS_fv = mg.fp_fv_convection_interpol(k*dt,x,a[:,k],dt,dx)
				m[:,k+1] = sparse.linalg.spsolve(LHS_fv,RHS_fv*np.ravel(m[:,k]))
		print "Time spent:",time.time()-t0,KELL
		#compute error in 2-norm
		#print ss
		#m_exact = convolution(x-velocity*T,T) #CONSTANT COEFFICIENT TEST
		m_solns[N] = m[:,-1]
		u_solns[N] = u[:,0]
		x_solns[N] = x
		if max(sum(abs(m-m_old)))<1e-4:
			print "Iteration successful!"
			break
		else:
			print max(sum(abs(m-m_old)))
		#print a1.max()
		#print a1.min()
		#print a2.max()
		#print a2.min()


#for i in range(0,REFINEMENTS-1):

#interpolate everything
m_interpolates = np.zeros((REFINEMENTS-1,x.size))
u_interpolates = np.zeros((REFINEMENTS-1,x.size))
for i in range(REFINEMENTS-1):
	m_interpolates[i,:] = np.interp(x_solns[-1],x_solns[i],m_solns[i])
	u_interpolates[i,:] = np.interp(x_solns[-1],x_solns[i],u_solns[i])
	
e1_hjb1 = np.zeros(REFINEMENTS-1)
e2_hjb1 = np.zeros(REFINEMENTS-1)
einf_hjb1 = np.zeros(REFINEMENTS-1)
e1_fp1 = np.zeros(REFINEMENTS-1)
e2_fp1 = np.zeros(REFINEMENTS-1)
einf_fp1 = np.zeros(REFINEMENTS-1)
for i in range(REFINEMENTS-1):
	e1_hjb1[i] = np.linalg.norm(u_interpolates[i,:]-u_solns[-1],ord=1)#*dexes[i]
	e2_hjb1[i] = np.linalg.norm(u_interpolates[i,:]-u_solns[-1],ord=2)#*np.sqrt(dexes[i])
	einf_hjb1[i] = np.linalg.norm(u_interpolates[i,:]-u_solns[-1],ord=np.inf)
	e1_fp1[i] = np.linalg.norm(m_interpolates[i,:]-m_solns[-1],ord=1)#*dexes[i]
	e2_fp1[i] = np.linalg.norm(m_interpolates[i,:]-m_solns[-1],ord=2)#*np.sqrt(dexes[i])
	einf_fp1[i] = np.linalg.norm(m_interpolates[i,:]-m_solns[-1],ord=np.inf)
	

#crunch the slopes and put in the figures
slope_u1_1, intercept = np.polyfit(np.log(dexes[cutoff:-1]), np.log(e1_hjb1[cutoff:]), 1)
slope_u1_2, intercept = np.polyfit(np.log(dexes[cutoff:-1]), np.log(e2_hjb1[cutoff:]), 1)
slope_u1_inf, intercept = np.polyfit(np.log(dexes[cutoff:-1]), np.log(einf_hjb1[cutoff:]), 1)

slope_m1_1, intercept = np.polyfit(np.log(dexes[cutoff:-1]), np.log(e1_fp1[cutoff:]), 1)
slope_m1_2, intercept = np.polyfit(np.log(dexes[cutoff:-1]), np.log(e2_fp1[cutoff:]), 1)
slope_m1_inf, intercept = np.polyfit(np.log(dexes[cutoff:-1]), np.log(einf_fp1[cutoff:]), 1)



fig4 = plt.figure(1)
str11 = "1-norm: slope %.2f for precision 1" %slope_u1_1
str12 = "2-norm: slope %.2f for precision 1" %slope_u1_2
str1inf = "inf-norm: slope %.2f for precision 1" %slope_u1_inf
plt.loglog(dexes[:-1],e1_hjb1,'o-',label=str11)
plt.loglog(dexes[:-1],e2_hjb1,'o-',label=str12)
plt.loglog(dexes[:-1],einf_hjb1,'o-',label=str1inf)
legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')
ax4 = fig4.add_subplot(111)
ax4.set_xlabel('Log10 of dx')
ax4.set_ylabel('Log10 of error')
plt.grid(True,which="both",ls="-")
ax4.invert_xaxis()
fig4.suptitle('Convergence of u(x,t)', fontsize=14)


fig4 = plt.figure(2)
str11 = "1-norm: slope %.2f for precision 1" %slope_m1_1
str12 = "2-norm: slope %.2f for precision 1" %slope_m1_2
str1inf = "inf-norm: slope %.2f for precision 1" %slope_m1_inf
plt.loglog(dexes[:-1],e1_fp1,'o-',label=str11)
plt.loglog(dexes[:-1],e2_fp1,'o-',label=str12)
plt.loglog(dexes[:-1],einf_fp1,'o-',label=str1inf)
legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')
ax4 = fig4.add_subplot(111)
ax4.set_xlabel('Log10 of dx')
ax4.set_ylabel('Log10 of error')
plt.grid(True,which="both",ls="-")
ax4.invert_xaxis()
fig4.suptitle('Convergence of m(x,t)', fontsize=14)

fig4 = plt.figure(4)
plt.plot(x,m[:,-1],label="Exact")
fig4.suptitle("Solution of m")

fig5 = plt.figure(5)
for i in range(REFINEMENTS-1):
	plt.plot(x,m_interpolates[i,:])
plt.plot(x,m[:,-1])


plt.show()




