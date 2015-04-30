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
POINTSx = 10 #points in x-direction
POINTSy = 20 #points in y-direction
REFINEMENTS = 3
X_NAUGHT = 0.0
DT = 0.5#ratio as in dt = DT*dx(**2)
NICE_DIFFUSION = 1
n = 2 #must be integer greater than 0
cutoff = 0 #for convergence slope
xmax = np.pi*n
xmin = 0
ymax = np.pi*n
ymin = 0
T = 1
#set dx
dx0 = abs(xmax-xmin)/POINTSx
dy0 = abs(ymax-ymin)/POINTSy
dx = dx0
dy = dy0

def exact_solution(x,y,t):
	x,y=np.meshgrid(x,y)
	return np.exp(-t)*np.cos(x)*np.cos(y)
e1 = np.zeros(REFINEMENTS)
e1_1 = np.zeros(REFINEMENTS)
e1_inf = np.zeros(REFINEMENTS)
dexes = np.zeros(REFINEMENTS)


################################
#THIS IS WHERE WE NEED THE LOOP#
################################
for N in range(0,REFINEMENTS):
	t1,t2=0,0
	dx = dx/2
	dy = dy/2
	dexes[N] = max(dx,dy)
	dt = DT*min(dx,dy)
	#CRUNCH
	print "(",dx,",",dy,",",dt,")"
	Nx = int(abs(xmax-xmin)/dx)+1
	Ny = int(abs(ymax-ymin)/dy)+1
	Nt = int(np.ceil(T/dt))
	x = np.linspace(xmin,xmax,Nx)
	y = np.linspace(ymin,ymax,Ny)
	t = np.linspace(0,T,Nt)
	I = x.size #space
	J = y.size
	u = exact_solution(x,y,T)
	#SOLVE STUFF
	t0 = time.time()
	for k in range(Nt-1,-1,-1): #COMPUTE M WHERE WE DO NOT ALLOW AGENTS TO LEAVE SUCH THAT m(-1) = m(N+1) = 1 ALWAYS
		ush = np.reshape(u,(J,I))
		#print u-ush
		#print ss
		a1,a2 = np.gradient(ush,dx,dy) #this ought to do it
		#x,y = np.meshgrid(x,y)
		#print x.shape,a1.shape
		#print ss
		a1 = -np.transpose(a1) #is this seriously enough
		a2 = -np.transpose(a2)
		t1_tmp = time.time()
		#print u_rsh.shape,a1.shape,a2.shape
		#print ss
		if NICE_DIFFUSION==0:
			u = solve.hjb_kushner_mod(k*dt,x,y,a1,a2,m_tmp,dx,dy,dt)
		else:
			if k==Nt-1:
				LHS_HJB = mg.HJB_diffusion_implicit(k*dt,x,y,a1,a2,u,dx,dy,dt)
			RHS_HJB = mg.HJB_convection_explicit(k*dt,x,y,a1,a2,dx,dy,dt)
			t1 += time.time() - t1_tmp
			Ltmp = iF.L_global(k*dt,x,y,a1,a2,u)
			t2_tmp = time.time()
			u = sparse.linalg.spsolve(LHS_HJB,RHS_HJB*np.ravel(u)+dt*np.ravel(Ltmp))
			t2 += time.time()-t2_tmp
	t0 = time.time()-t0
	print "Time spent:",t0
	print "\tGenerating matrices:",t1/t0*100
	print "\tSolving linear systems:",t2/t0*100
	u = np.reshape(u,(J,I))
	u_exact = exact_solution(x,y,0)
	e1[N] = np.linalg.norm(u-u_exact)*np.sqrt(dx*dy)
	e1_1[N] = np.linalg.norm(u-u_exact,ord=1)*dx*dy
	e1_inf[N] = np.linalg.norm(u-u_exact,ord=np.inf)

#print e1
#print e2
#print e3
#crunch the slopes and put in the figures
Xplot, Yplot = np.meshgrid(x,y)
if REFINEMENTS>1:
	slope1, intercept = np.polyfit(np.log(dexes[cutoff:]), np.log(e1[cutoff:]), 1)
	slope1_1, intercept = np.polyfit(np.log(dexes[cutoff:]), np.log(e1_1[cutoff:]), 1)
	slope1_inf, intercept = np.polyfit(np.log(dexes[cutoff:]), np.log(e1_inf[cutoff:]), 1)
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








