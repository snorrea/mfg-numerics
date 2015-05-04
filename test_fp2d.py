from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import time as time
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize as minimize
from matplotlib import cm
import quadrature_nodes as qn
import input_functions_2D as iF
import solvers_2D as solve
import scipy.sparse as sparse
import matrix_gen as mg

#INPUTS
POINTSx = 10 #points in x-direction
POINTSy = 10 #points in y-direction
REFINEMENTS = 2
X_NAUGHT = 0.0
DT = .1#ratio as in dt = DT*dx(**2)
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
	m = exact_solution(x,y,0)
	#SOLVE STUFF
	t0 = time.time()
	for k in range(1,Nt): #COMPUTE M WHERE WE DO NOT ALLOW AGENTS TO LEAVE SUCH THAT m(-1) = m(N+1) = 1 ALWAYS
		t1_tmp = time.time()
		if k==1:
			LHS = mg.FP_diffusion_implicit_Ometh(k*dt,x,y,x,y,m,dx,dt)
			#LHS = mg.FP_diffusion_implicit_Diamond(k*dt,x,y,x,y,m,dx,dt)
			#print LHS.sum(1)
			#print ss
		RHS = mg.FP_convection_explicit(k*dt,x,y,x,y,m,dx,dt)
		#print RHS
		#print ss
		t1 += time.time() - t1_tmp
		t2_tmp = time.time()
		F = iF.F_global(x,y,m,k*dt)
		m = sparse.linalg.spsolve(LHS,RHS*np.ravel(m)+dt*np.ravel(F))
		#m = sparse.linalg.spsolve(LHS,RHS*np.ravel(m))
		t2 += time.time()-t2_tmp
	t0 = time.time()-t0
	print "Time spent:",t0
	print "\tGenerating matrices:",t1/t0*100
	print "\tSolving linear systems:",t2/t0*100
	#u = np.reshape(u,(J,I))
	m_exact = np.ravel(exact_solution(x,y,T))
	e1[N] = np.linalg.norm(m-m_exact)*np.sqrt(dx*dy)
	e1_1[N] = np.linalg.norm(m-m_exact,ord=1)*dx*dy
	e1_inf[N] = np.linalg.norm(m-m_exact,ord=np.inf)

m = np.reshape(m,(J,I))
m_exact = exact_solution(x,y,T)

#print u-u_exact
print e1
print e1_1
print e1_inf
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
ax1.plot_surface(Xplot,Yplot,m-m_exact,rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('Delta u(x,t)')
fig3.suptitle('Deviation from the solution', fontsize=14)

fig2 = plt.figure(3)
ax1 = fig2.add_subplot(111, projection='3d')
ax1.plot_surface(Xplot,Yplot,m,rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('u(x,t)')
fig2.suptitle('Computed solution', fontsize=14)

fig22 = plt.figure(4)
ax1 = fig22.add_subplot(111, projection='3d')
ax1.plot_surface(Xplot,Yplot,m_exact,rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=0, antialiased=False)
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








