from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import time as time
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize as minimize
from matplotlib import cm
import quadrature_nodes as qn
import input_functions_2D as iF
import solvers_1D as solve
import scipy.sparse as sparse
import matrix_gen as mg

#INPUTS
dx0 = 2*0.1
REFINEMENTS = 2
X_NAUGHT = 0.0
alpha = 1
beta = 1
tau = 1
DT = .5#ratio as in dt = DT*dx(**2)
NICE_DIFFUSION = 1
n = 1 #must be integer greater than 0
xmax = np.pi/alpha * (n-0.5)
xmin = -xmax
ymax = np.pi/beta*n
ymin = -ymax
T = 1
#set dx
dx = dx0
def exact_solution(x,y,t):
	x,y=np.meshgrid(x,y)
	return np.exp(-tau*t)*np.sin(alpha*x)*np.cos(beta*y)
e1 = np.zeros(REFINEMENTS)
e1_1 = np.zeros(REFINEMENTS)
e1_inf = np.zeros(REFINEMENTS)
dexes = np.zeros(REFINEMENTS)


################################
#THIS IS WHERE WE NEED THE LOOP#
################################
for N in range(0,REFINEMENTS):
	t1,t2=0,0
	dx = dx/2 #starts at dx=0.25
	dexes[N] = dx
	dt = DT*dx
	#CRUNCH
	print "(",dx,",",dt,")"
	dx2 = dx**2
	Nx = int(abs(xmax-xmin)/dx)+1
	Ny = int(abs(ymax-ymin)/dx)+1
	Nt = int(np.ceil(T/dt))
	x = np.linspace(xmin,xmax,Nx)
	y = np.linspace(ymin,ymax,Nx)
	t = np.linspace(0,T,Nt)
	I = x.size #space
	J = y.size
	#print cutoffindexmin,cutoffindexmax,Nx
	#INITIALISE STORAGE
	u = np.zeros((I,J))
	u_exact = np.zeros((I,J))
	#INITIAL/TERMINAL CONDITIONS, INITIAL GUESS and COPYING
	u = exact_solution(x,y,T)
	#SOLVE STUFF
	t0 = time.time()
	for k in range(Nt-1,-1,-1): #COMPUTE M WHERE WE DO NOT ALLOW AGENTS TO LEAVE SUCH THAT m(-1) = m(N+1) = 1 ALWAYS
		#print np.gradient(u,dx)
		#print ss
		a1,a2 = np.gradient(np.reshape(u,(I,J)),dx,dx) #this ought to do it
		#a1,a2=-a1,-a2
		t1_tmp = time.time()
		if NICE_DIFFUSION==0:
			u = solve.hjb_kushner_mod(k*dt,x,y,a1_tmp,a2_tmp,m_tmp,dx,dt)
		else:
			if k==Nt-1:
				LHS_HJB = mg.HJB_diffusion_implicit(k*dt,x,y,a1,a2,u,dx,dt)
				#print LHS_HJB
				#print ss
			RHS_HJB = mg.HJB_convection_explicit(k*dt,x,y,a1,a2,u,dx,dt)
			t1 += time.time() - t1_tmp
			#print RHS_HJB
			#print ss
			Ltmp = iF.L_global(k*dt,x,y,a1,a2,u)
			#plt.spy(RHS_HJB)
			#plt.show()
			t2_tmp = time.time()
			u = sparse.linalg.spsolve(LHS_HJB,RHS_HJB*np.ravel(u)+dt*np.ravel(Ltmp))
			t2 += time.time()-t2_tmp
	t0 = time.time()-t0
	print "Time spent:",t0
	print "\tGenerating matrices:",t1/t0*100
	print "\tSolving linear systems:",t2/t0*100
	#compute error in 2-norm
	u = np.reshape(u,(I,J))
	u_exact = exact_solution(x,y,0)
	e1[N] = np.linalg.norm(u-u_exact)*dx
	e1_1[N] = np.linalg.norm(u-u_exact,ord=1)*dx
	e1_inf[N] = np.linalg.norm(u-u_exact,ord=np.inf)

#print e1
#print e2
#print e3
#crunch the slopes and put in the figures
Xplot, Yplot = np.meshgrid(x,y)
if REFINEMENTS>1:
	slope1, intercept = np.polyfit(np.log(dexes[1:]), np.log(e1[1:]), 1)
	slope1_1, intercept = np.polyfit(np.log(dexes[1:]), np.log(e1_1[1:]), 1)
	slope1_inf, intercept = np.polyfit(np.log(dexes[1:]), np.log(e1_inf[1:]), 1)
	fig4 = plt.figure(6)
	str1 = "2-norm slope:", "%.2f" %slope1
	str2 = "1-norm slope:", "%.2f" %slope1_1
	str3 = "inf-norm slope:", "%.2f" %slope1_inf
	plt.loglog(dexes,e1,'o-',label=str1)
	plt.loglog(dexes,e1_1,'o-',label=str2)
	plt.loglog(dexes,e1_inf,'o-',label=str3)
	legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')
	ax4 = fig4.add_subplot(111)
	ax4.set_xlabel('Log10 of dx')
	ax4.set_ylabel('Log10 of error')
	plt.grid(True,which="both",ls="-")
	ax4.invert_xaxis()
	fig4.suptitle('Convergence rates of u(x,t)', fontsize=14)

fig3 = plt.figure(2)
ax1 = fig3.add_subplot(111, projection='3d')
ax1.plot_surface(Xplot,Yplot,u-u_exact,rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('Delta u(x,t)')
fig3.suptitle('Deviation from the solution', fontsize=14)

fig2 = plt.figure(3)
ax1 = fig2.add_subplot(111, projection='3d')
ax1.plot_surface(Xplot,Yplot,u,rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('u(x,t)')
fig2.suptitle('Computed solution', fontsize=14)

fig22 = plt.figure(4)
ax1 = fig22.add_subplot(111, projection='3d')
ax1.plot_surface(Xplot,Yplot,u_exact,rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('u(x,t)')
fig22.suptitle('Exact solution', fontsize=14)


#fig2 = plt.figure(3)
#plt.plot(x,u,label="Centered FD")
#plt.plot(x,u_exact,label="Exact solution")
#fig2.suptitle('Solutions', fontsize=14)
#legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')


plt.show()








