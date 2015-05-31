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
REFINEMENTS = 5
cutoff = 0

########
xmin = 0
xmax = 1
DT = 0.4 #time ratio
DM = 1e-2 #minimiser tolerance coefficient
DM_E = 1 #minimiser tolerance exponent
T = 1
#set dx
dx = dx0
NICE_DIFFUSION = 1

#########
alpha_upper = 1.5
alpha_lower = -1.5
#STUFF TO MINIMIZE
min_left = alpha_lower #search region left
min_right = alpha_upper #search region right
Ns = 100
xpts_scatter = np.linspace(min_left,min_right,Ns)

#########

e1_hjb1 = np.zeros(REFINEMENTS-1)
e1_hjb2 = np.zeros(REFINEMENTS-1)
e1_fp1 = np.zeros(REFINEMENTS-1)
e1_fp2 = np.zeros(REFINEMENTS-1)
e2_hjb1 = np.zeros(REFINEMENTS-1)
e2_hjb2 = np.zeros(REFINEMENTS-1)
e2_fp1 = np.zeros(REFINEMENTS-1)
e2_fp2 = np.zeros(REFINEMENTS-1)
einf_hjb1 = np.zeros(REFINEMENTS-1)
einf_hjb2 = np.zeros(REFINEMENTS-1)
einf_fp1 = np.zeros(REFINEMENTS-1)
einf_fp2 = np.zeros(REFINEMENTS-1)
dexes = np.zeros(REFINEMENTS)

m1_solns = [None]*REFINEMENTS
m2_solns = [None]*REFINEMENTS
u1_solns = [None]*REFINEMENTS
u2_solns = [None]*REFINEMENTS
x_solns = [None]*REFINEMENTS

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
	#MINIMISE
	min_tol = DM*dx**DM_E
	scatters = int(np.ceil( np.log((min_right-min_left)/(min_tol*Ns))/np.log(Ns/2) ))
	##########
	m1 = np.empty((I,K))
	m2 = np.empty((I,K))
	u1 = np.empty((I,K))
	u2 = np.empty((I,K))
	a1 = np.empty((I,K-1))
	a2 = np.empty((I,K-1))
	#INITIAL/TERMINAL CONDITIONS, INITIAL GUESS and COPYING
	#m0 = delta(x,X_NAUGHT,dx)
	for k in range(0,Nt):
		m1[:,k] = iF.initial_distribution(x)
		m2[:,k] = iF.initial_distribution(x)
		u1[:,k] = iF.G(x,x)
		u2[:,k] = iF.G(x,x)
	#SOLVE STUFF
	t0 = time.time()
	for k in range(Nt-2,-1,-1): #COMPUTE M WHERE WE DO NOT ALLOW AGENTS TO LEAVE SUCH THAT m(-1) = m(N+1) = 1 ALWAYS
		a1[:,k] = -np.gradient(u1[:,k+1],dx)
		a2[:,k] = solve.control_general(x,k*dt,u2[:,k+1],u2[:,k+1],dt,dx,xpts_scatter,Ns,scatters)
		if NICE_DIFFUSION==0:
			u1[:,k] = solve.hjb_kushner_mod(x,k*dt,u1[:,k],u1[:,k],a1,dt,dx) #implicit
			u2[:,k] = solve.hjb_kushner_mod(x,k*dt,u2[:,k],u2[:,k],a2,dt,dx) #implicit
		else:
			if k==Nt-2:
				LHS_HJB1 = mg.hjb_diffusion(k*dt,x,a1[:,k],dt,dx)
				LHS_HJB2 = mg.hjb_diffusion(k*dt,x,a2[:,k],dt,dx)
			RHS_HJB1 = mg.hjb_convection(k*dt,x,a1[:,k],dt,dx)
			RHS_HJB2 = mg.hjb_convection(k*dt,x,a2[:,k],dt,dx)
			Ltmp1 = iF.L_global(k*dt,x,a1[:,k],m1[:,k],u1)
			Ltmp2 = iF.L_global(k*dt,x,a2[:,k],m2[:,k],u2)
			u1[:,k] = sparse.linalg.spsolve(LHS_HJB1,RHS_HJB1*u1[:,k+1]+dt*Ltmp1)
			u2[:,k] = sparse.linalg.spsolve(LHS_HJB2,RHS_HJB2*u2[:,k+1]+dt*Ltmp2)
	#SOLVE STUFF
	#t0 = time.time()
	for k in range(0,K-1): #COMPUTE M WHERE WE DO NOT ALLOW AGENTS TO LEAVE SUCH THAT m(-1) = m(N+1) = 1 ALWAYS
		#m1 = solve.fp_fd_centered_mod(x,(k)*dt,m1,m0,dt,dx)
		#m2 = solve.fp_fd_upwind_mod(x,(k)*dt,m2,m0,dt,dx)
		#m3 = solve.fp_fd_upwind_visc(x,(k)*dt,m2,m0,dt,dx)
		#m4 = solve.fp_fv_mod(x,(k)*dt,m4,m0,dt,dx)
		if k==0:
			LHS_fv1 = mg.fp_fv_diffusion(k*dt,x,a1[:,k],dt,dx)
			#RHS_fv1 = mg.fp_fv_convection_classic(k*dt,x,a1[:,k],dt,dx)
			RHS_fv1 = mg.fp_fv_convection_interpol(k*dt,x,a1[:,k],dt,dx)
			LHS_fv2 = mg.fp_fv_diffusion(k*dt,x,a2[:,k],dt,dx)
			#RHS_fv2 = mg.fp_fv_convection_classic(k*dt,x,a2[:,k],dt,dx)
			RHS_fv2 = mg.fp_fv_convection_interpol(k*dt,x,a2[:,k],dt,dx)
		m1[:,k+1] = sparse.linalg.spsolve(LHS_fv1,RHS_fv1*np.ravel(m1[:,k]))
		m2[:,k+1] = sparse.linalg.spsolve(LHS_fv2,RHS_fv2*np.ravel(m2[:,k]))
		#print m4
	print "Time spent:",time.time()-t0
	#compute error in 2-norm
	#print ss
	#m_exact = convolution(x-velocity*T,T) #CONSTANT COEFFICIENT TEST
	m1_solns[N] = m1[:,-1]
	m2_solns[N] = m2[:,-1]
	u1_solns[N] = u1[:,0]
	u2_solns[N] = u2[:,0]
	x_solns[N] = x

	#print a1.max()
	#print a1.min()
	#print a2.max()
	#print a2.min()


#for i in range(0,REFINEMENTS-1):

#interpolate everything
m1_interpolates = np.zeros((REFINEMENTS-1,x.size))
m2_interpolates = np.zeros((REFINEMENTS-1,x.size))
u1_interpolates = np.zeros((REFINEMENTS-1,x.size))
u2_interpolates = np.zeros((REFINEMENTS-1,x.size))
for i in range(REFINEMENTS-1):
	m1_interpolates[i,:] = np.interp(x_solns[-1],x_solns[i],m1_solns[i])
	m2_interpolates[i,:] = np.interp(x_solns[-1],x_solns[i],m2_solns[i])
	u1_interpolates[i,:] = np.interp(x_solns[-1],x_solns[i],u1_solns[i])
	u2_interpolates[i,:] = np.interp(x_solns[-1],x_solns[i],u2_solns[i])

e1_hjb1 = np.zeros(REFINEMENTS-1)
e2_hjb1 = np.zeros(REFINEMENTS-1)
einf_hjb1 = np.zeros(REFINEMENTS-1)
e1_hjb2 = np.zeros(REFINEMENTS-1)
e2_hjb2 = np.zeros(REFINEMENTS-1)
einf_hjb2 = np.zeros(REFINEMENTS-1)
e1_fp1 = np.zeros(REFINEMENTS-1)
e2_fp1 = np.zeros(REFINEMENTS-1)
einf_fp1 = np.zeros(REFINEMENTS-1)
e1_fp2 = np.zeros(REFINEMENTS-1)
e2_fp2 = np.zeros(REFINEMENTS-1)
einf_fp2 = np.zeros(REFINEMENTS-1)
for i in range(REFINEMENTS-1):
	e1_hjb1[i] = np.linalg.norm(u1_interpolates[i,:]-u1_solns[-1],ord=1)
	e2_hjb1[i] = np.linalg.norm(u1_interpolates[i,:]-u1_solns[-1],ord=2)
	einf_hjb1[i] = np.linalg.norm(u1_interpolates[i,:]-u1_solns[-1],ord=np.inf)
	e1_hjb2[i] = np.linalg.norm(u2_interpolates[i,:]-u2_solns[-1],ord=1)
	e2_hjb2[i] = np.linalg.norm(u2_interpolates[i,:]-u2_solns[-1],ord=2)
	einf_hjb2[i] = np.linalg.norm(u2_interpolates[i,:]-u2_solns[-1],ord=np.inf)
	e1_fp1[i] = np.linalg.norm(m1_interpolates[i,:]-m1_solns[-1],ord=1)
	e2_fp1[i] = np.linalg.norm(m1_interpolates[i,:]-m1_solns[-1],ord=2)
	einf_fp1[i] = np.linalg.norm(m1_interpolates[i,:]-m1_solns[-1],ord=np.inf)
	e1_fp2[i] = np.linalg.norm(m2_interpolates[i,:]-m2_solns[-1],ord=1)
	e2_fp2[i] = np.linalg.norm(m2_interpolates[i,:]-m2_solns[-1],ord=2)
	einf_fp2[i] = np.linalg.norm(m2_interpolates[i,:]-m2_solns[-1],ord=np.inf)


#crunch the slopes and put in the figures
slope_u1_1, intercept = np.polyfit(np.log(dexes[cutoff:-1]), np.log(e1_hjb1[cutoff:]), 1)
slope_u2_1, intercept = np.polyfit(np.log(dexes[cutoff:-1]), np.log(e1_hjb2[cutoff:]), 1)
slope_u1_2, intercept = np.polyfit(np.log(dexes[cutoff:-1]), np.log(e2_hjb1[cutoff:]), 1)
slope_u2_2, intercept = np.polyfit(np.log(dexes[cutoff:-1]), np.log(e2_hjb2[cutoff:]), 1)
slope_u1_inf, intercept = np.polyfit(np.log(dexes[cutoff:-1]), np.log(einf_hjb1[cutoff:]), 1)
slope_u2_inf, intercept = np.polyfit(np.log(dexes[cutoff:-1]), np.log(einf_hjb2[cutoff:]), 1)

slope_m1_1, intercept = np.polyfit(np.log(dexes[cutoff:-1]), np.log(e1_fp1[cutoff:]), 1)
slope_m2_1, intercept = np.polyfit(np.log(dexes[cutoff:-1]), np.log(e1_fp2[cutoff:]), 1)
slope_m1_2, intercept = np.polyfit(np.log(dexes[cutoff:-1]), np.log(e2_fp1[cutoff:]), 1)
slope_m2_2, intercept = np.polyfit(np.log(dexes[cutoff:-1]), np.log(e2_fp2[cutoff:]), 1)
slope_m1_inf, intercept = np.polyfit(np.log(dexes[cutoff:-1]), np.log(einf_fp1[cutoff:]), 1)
slope_m2_inf, intercept = np.polyfit(np.log(dexes[cutoff:-1]), np.log(einf_fp2[cutoff:]), 1)




fig4 = plt.figure(1)
str11 = "Exact control, 1-norm", "%.2f" %slope_u1_1
str12 = "Exact control, 2-norm", "%.2f" %slope_u1_2
str1inf = "Exact control, inf-norm", "%.2f" %slope_u1_inf
str21 = "Computed control, 1-norm", "%.2f" %slope_u2_1
str22 = "Computed control, 2-norm", "%.2f" %slope_u2_2
str2inf = "Computed control, inf-norm", "%.2f" %slope_u2_inf
plt.loglog(dexes[:-1],e1_hjb1,'o-',label=str11)
plt.loglog(dexes[:-1],e2_hjb1,'o-',label=str12)
plt.loglog(dexes[:-1],einf_hjb1,'o-',label=str1inf)
plt.loglog(dexes[:-1],e1_hjb2,'o-',label=str21)
plt.loglog(dexes[:-1],e2_hjb2,'o-',label=str22)
plt.loglog(dexes[:-1],einf_hjb2,'o-',label=str2inf)
#plt.loglog(dexes[:-1],e_u1,'o-',label="Error, exact vs exact")
#plt.loglog(dexes[:-1],e_u2,'o-',label="Error, computed vs exact")
#plt.loglog(dexes[:-1],e_u3,'o-',label="Error, computed vs computed")
legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')
ax4 = fig4.add_subplot(111)
ax4.set_xlabel('Log10 of dx')
ax4.set_ylabel('Log10 of error')
plt.grid(True,which="both",ls="-")
ax4.invert_xaxis()
fig4.suptitle('Convergence of u(x,t)', fontsize=14)


fig4 = plt.figure(2)
str11 = "Exact control, 1-norm", "%.2f" %slope_m1_1
str12 = "Exact control, 2-norm", "%.2f" %slope_m1_2
str1inf = "Exact control, inf-norm", "%.2f" %slope_m1_inf
str21 = "Computed control, 1-norm", "%.2f" %slope_m2_1
str22 = "Computed control, 2-norm", "%.2f" %slope_m2_2
str2inf = "Computed control, inf-norm", "%.2f" %slope_m2_inf
plt.loglog(dexes[:-1],e1_fp1,'o-',label=str11)
plt.loglog(dexes[:-1],e2_fp1,'o-',label=str12)
plt.loglog(dexes[:-1],einf_fp1,'o-',label=str1inf)
plt.loglog(dexes[:-1],e1_fp2,'o-',label=str21)
plt.loglog(dexes[:-1],e2_fp2,'o-',label=str22)
plt.loglog(dexes[:-1],einf_fp2,'o-',label=str2inf)
#plt.loglog(dexes[:-1],e_m1,'o-',label="Error, exact vs exact")
#plt.loglog(dexes[:-1],e_m2,'o-',label="Error, computed vs exact")
#plt.loglog(dexes[:-1],e_m3,'o-',label="Error, computed vs computed")
legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')
ax4 = fig4.add_subplot(111)
ax4.set_xlabel('Log10 of dx')
ax4.set_ylabel('Log10 of error')
plt.grid(True,which="both",ls="-")
ax4.invert_xaxis()
fig4.suptitle('Convergence of m(x,t)', fontsize=14)

#fig2 = plt.figure(3)
#plt.plot(np.log(dexes[:-1]),e_m2/e_m1,'o-',label="Computed vs exact, distribution")
#plt.plot(np.log(dexes[:-1]),e_u2/e_u1,'o-',label="Computed vs exact, potential")
#legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')
#plt.grid(True,which="both",ls="-")
#ax4 = fig2.add_subplot(111)
#ax4.invert_xaxis()
#ax4.set_xlabel('Log10 of dx')
#ax4.set_ylabel('Relationship between computed/exact error')

fig4 = plt.figure(4)
plt.plot(x,m1[:,-1],label="Exact")
plt.plot(x,m2[:,-1],label="Computed")
legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')

fig5 = plt.figure(5)
for i in range(REFINEMENTS-1):
	plt.plot(x,m1_interpolates[i,:])
plt.plot(x,m1[:,-1])

plt.show()




