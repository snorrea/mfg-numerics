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
import scipy.sparse as sparse
import grand_matrix_gen1d as gmg

#INPUTS
FINITE_VOLUME = 1 #0 if FD, 1 if FV
NICE_DIFFUSION = 1 #1 if diffusion indep of t,m,alpha
#dx = 0.1**2/2
dx = 0.02
dt = .4*dx#**2
#dt = dx**2/(0.3**2 + dx*2) # dt = dx**2/(max(sigma)**2 + dx*max(f))
print dx,dt
xmin = 0#-2
xmax = 1#+.2
T = 1
Niter = 500 #maximum number of iterations
tolerance = 1e-4
alpha_upper = 1
alpha_lower = -1

#STUFF TO MINIMIZE
N = 200 #searchpoints
Nr = 5
min_tol = 1e-6#tolerance#1e-5 #tolerance for minimum
min_left = alpha_lower #search region left
min_right = alpha_upper #search region right
relation = 2
#scatters = int(np.ceil(np.log((min_right-min_left)/min_tol)/np.log(N)))
#scatters = np.ceil( np.log(min_tol*N/(min_right-min_left))/np.log(2/N) )
scatters = int(np.ceil( np.log((min_right-min_left)/(min_tol*N))/np.log(N/2) ))
#scatters2 = int(1 + np.ceil(np.log((min_right-min_left)/(N*min_tol))/np.log(relation)))
xpts_scatter = np.linspace(min_left,min_right,N)
xpts_newton = np.linspace(min_left,min_right,N*Nr)
dx_scatter = xpts_scatter[1]-xpts_scatter[0]
dx_newton = xpts_newton[1]-xpts_newton[0]
print scatters
#print ss
#CRUNCH
dx2 = dx**2
Nx = int(abs(xmax-xmin)/dx)+1
Nt = int(T/dt)+1
x = np.linspace(xmin,xmax,Nx)
t = np.linspace(0,T,Nt)
I = x.size #space
K = t.size #time

#INITIALISE STORAGE
u = np.zeros((I*K)) #potential
m = np.zeros((I*K)) #distribution
a = np.zeros((I*K)) #control
u_old = np.zeros((I*K))
m_old = np.zeros((I*K))
a_old = np.zeros((I*K))
ul1 = -1*np.ones((Niter,1))
ul2 = -1*np.ones((Niter,1))
ulinfty = -1*np.ones((Niter,1))
ml1 = -1*np.ones((Niter,1))
ml2 = -1*np.ones((Niter,1))
mlinfty = -1*np.ones((Niter,1))
al1 = -1*np.ones((Niter,1))
al2 = -1*np.ones((Niter,1))
alinfty = -1*np.ones((Niter,1))
def index(i,k): 
	return int(i+(I)*k)
Xplot, Tplot = np.meshgrid(x,t)
m0 = iF.initial_distribution(x)
m0 = m0/(sum(m0)*dx)
for k in range (0,K):
	m[k*I:k*I+I] = np.copy(m0)
#INITIAL/TERMINAL CONDITIONS, INITIAL GUESS and COPYING
m_old = np.copy(m)
u_old = np.copy(u)
a_old = np.copy(a)
a_tmp = np.copy(a)
time_total = time.time()
BIGZERO = np.zeros(x.size-2)

#matrix-generation if stuff
LHS_HJB = np.zeros(I)
LHS_FP = np.zeros(I)

for n in range (0,Niter):
	titer = time.time()
	print "Computing iteration",n+1,"of u..."
	temptime = time.time()
	#Compute u
	#make the a_grand
	u_old = np.copy(u)
	for k in range(1,K):
			#a[(I*k-I):(I*k)] = np.maximum(-np.gradient(u[(I*k-I):(I*k)],dx),np.zeros(u[(I*k-I):(I*k)].size))
			a[(I*k-I):(I*k)] = -np.gradient(u[(I*k):(I*k+I)],dx)
	for turns in range(0,50):
		#a_grand = 0.2*np.ones(I*K)
		LHS = gmg.grand_hjb_cmfg_ng(t,x,a,m,dt,dx)
		RHS = gmg.grand_hjb_vector(t,x,a,m,dt)
		#plt.spy(LHS)
		#plt.show()
		u_tmp = np.copy(u)
		u = sparse.linalg.spsolve(LHS,RHS)
		for k in range(1,K):
			#a[(I*k-I):(I*k)] = np.maximum(-np.gradient(u[(I*k-I):(I*k)],dx),np.zeros(u[(I*k-I):(I*k)].size))
			a[(I*k-I):(I*k)] = -np.gradient(u[(I*k):(I*k+I)],dx)
		if np.linalg.norm(u-u_tmp)*np.sqrt(dx) < 1e-10:
			break
		else:
			print np.linalg.norm(u-u_tmp)*np.sqrt(dx)
	print "Spent time", time.time()-temptime, "on computing u"
	#store changes in norms
	uchange = np.copy(u-u_old)
	ul1[n] = np.sum(abs(uchange))*dx
	ul2[n] = np.sqrt(np.sum(abs(uchange)**2))*np.sqrt(dx)
	ulinfty[n] = max(abs(uchange))
	achange = np.copy(a-a_old)
	al1[n] = np.sum(abs(achange))*dx
	al2[n] = np.sqrt(np.sum(abs(achange)**2))*np.sqrt(dx)
	alinfty[n] = max(abs(achange) )
	
	#GET GOING WITH M
	print "Computing iteration", n+1, "of m..."
	temptime = time.time()
	LHS = gmg.grand_fp_cmfg_ng(t,x,a,dt,dx)
	RHS = gmg.grand_fp_vector(x,K)
	#plt.spy(LHS)
	#plt.show()
	m = sparse.linalg.spsolve(LHS,RHS)
	print "Spent time", time.time()-temptime, "on computing m"
	#compute norms of stuff
	mchange = np.copy(m-m_old)
	ml1[n] = np.sum(abs(mchange))*dx
	ml2[n] = np.sqrt(np.sum(abs(mchange)**2))*np.sqrt(dx)
	mlinfty[n] = max(abs( mchange) ) 
	if (mlinfty[n] < tolerance):
		print "Method converged with final change" , mlinfty[n]
		print "Time spent:", time.time()-time_total
		kMax = n
		break
	#Evaluate iteration
	m_old = np.copy(m)
	u_old = np.copy(u)
	a_old = np.copy(a)
	print "Iteration number", n+1, "completed. \nUsed time", time.time()-titer, "\nChange in (a,u,m)= (",  alinfty[n][0], ",", ulinfty[n][0], ",", mlinfty[n][0], ")"
print "Time spent:", time.time()-time_total
kMax = n

msoln = np.empty((I,K))
usoln = np.empty((I,K))
asoln = np.empty((I,K))
for i in range (0,I):
	for k in range (0,K):
		msoln[i,k] = m[index(i,k)]
		usoln[i,k] = u[index(i,k)]
		asoln[i,k] = a[index(i,k)]
msoln = np.transpose(msoln)
usoln = np.transpose(usoln)
asoln = np.transpose(asoln)
#cut the change vectors with kMax
ml1 = ml1[:kMax]
ml2 = ml2[:kMax]
mlinfty = mlinfty[:kMax]
ul1 = ul1[:kMax]
ul2 = ul2[:kMax]
ulinfty = ulinfty[:kMax]
al1 = al1[:kMax]
al2 = al2[:kMax]
alinfty = alinfty[:kMax]
#init plotstuff
#print Xplot.shape,Tplot.shape,msoln.shape,vsoln.shape,gradsoln.shape,mollgrad.shape
#plot solution of m(x,t)
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot_surface(Xplot,Tplot,msoln,rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax1.set_xlabel('x')
ax1.set_ylabel('t')
ax1.set_zlabel('m(x,t)')
fig1.suptitle('Solution of the density m(x,t)', fontsize=14)
#plot solution of u(x,t)
fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111, projection='3d')
ax2.plot_surface(Xplot,Tplot,usoln,rstride=10,cstride=10,cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax2.set_xlabel('x')
ax2.set_ylabel('t')
ax2.set_zlabel('u(x,t)')
fig2.suptitle('Solution of the potential v(x,t)', fontsize=14)
#plot solution of a(x,t)
fig3 = plt.figure(3)
ax3 = fig3.add_subplot(111, projection='3d')
ax3.plot_surface(Xplot,Tplot,asoln,rstride=10,cstride=10,cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax3.set_xlabel('x')
ax3.set_ylabel('t')
ax3.set_zlabel('a(x,t)')
fig3.suptitle('Solution of the control a(x,t)', fontsize=14)
#plot the norms of change on m
fig4 = plt.figure(4)
plt.loglog(np.arange(1,kMax+1), ml1, label="L1-norm")
plt.loglog(np.arange(1,kMax+1), (ml2), label="L2-norm")
plt.loglog(np.arange(1,kMax+1), (mlinfty), label="Linf-norm")
legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')
ax4 = fig4.add_subplot(111)
ax4.set_xlabel('Iteration number')
ax4.set_ylabel('Log10 of change')
fig4.suptitle('Convergence of m(x,t)', fontsize=14)
#plot the norms of change on u
fig5 = plt.figure(5)
plt.loglog(np.arange(1,kMax+1),(ul1), label="L1-norm")
plt.loglog(np.arange(1,kMax+1),(ul2), label="L2-norm")
plt.loglog(np.arange(1,kMax+1),(ulinfty), label="Linf-norm")
legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')
ax5 = fig5.add_subplot(111)
ax5.set_xlabel('Iteration number')
ax5.set_ylabel('Log10 of change')
fig5.suptitle('Convergence of u(x,t)', fontsize=14)
#plot the norms of change on m
fig6 = plt.figure(6)
plt.loglog(np.arange(1,kMax+1), (al1), label="L1-norm")
plt.loglog(np.arange(1,kMax+1), (al2), label="L2-norm")
plt.loglog(np.arange(1,kMax+1), (alinfty), label="Linf-norm")
legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')
ax6 = fig6.add_subplot(111)
ax6.set_xlabel('Iteration number')
ax6.set_ylabel('Log10 of change')
fig6.suptitle('Convergence of a(x,t)', fontsize=14)
fig7 = plt.figure(7)
plt.plot(a[0:Nx])

##########PLOT
plt.show()

