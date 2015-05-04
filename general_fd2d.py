from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import time as time
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize as minimize
from matplotlib import cm
from scipy.sparse.linalg import spsolve
from numpy.linalg import cond
import input_functions_2D as iF
import applications as app
import matrix_gen as mg
import solvers_2D as solve
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import Delaunay

dx = 0.05
dt = 0.05
#dx = 0.75*0.1/2
#dx = 0.3**2/(2*0.7)
#dx = 0.5**2/2
#dx = 0.5*dx
#print 
#dt = dx**2/(0.3**2 + dx*2) # dt = dx**2/(max(sigma)**2 + dx*max(f)) #to guarantee CFL holds for dt
#dt = 0.01*dt
xmin = 0
xmax = 1
ymin = 0
ymax = 1
T = 2
Niter = 1 #maximum number of iterations
tolerance = 1e-4
alpha_upper = 2
alpha_lower = -2
quad_order = 15
#STUFF TO MINIMIZE
N = 60 #searchpoints
min_tol = 1e-6#tolerance#1e-5 #tolerance for minimum
min_left = alpha_lower #search region left
min_right = alpha_upper #search region right
relation = 2
scatters = int(np.ceil(np.log((min_right-min_left)/min_tol)/np.log(N)))
#scatters2 = int(1 + np.ceil(np.log((min_right-min_left)/(N*min_tol))/np.log(relation)))
xpts_search = np.linspace(min_left,min_right,N)
fpts = np.empty((xpts_search.size,1))
print "(dx,dt,scatters)=", "(", dx, ",",dt, "," , scatters, ")"
#CRUNCH
dx2 = dx**2
Nx = int(abs(xmax-xmin)/dx)+1
Ny = int(abs(ymax-ymin)/dx)+1
Nt = int(T/dt)+1
x = np.linspace(xmin,xmax,Nx)
y = np.linspace(ymin,ymax,Nx)
t = np.linspace(0,T,Nt)
I = x.size #space
J = y.size
K = t.size #time


#grid = np.array([[0,0],[0,1],[1,1],[1,0]])
#tri = Delaunay(grid)

#grid = np.empty((I*J,2))
#print grid
#for i in range(0,I):
#	for j in range(0,J):
#		#print np.array((x[i],y[j]))
#		grid[i+I*j] = np.array((x[i],y[j]))
#tri = Delaunay(grid)
#plt.triplot(grid[:,0], grid[:,1], tri.simplices.copy())
#plt.plot(grid[:,0], grid[:,1], 'o')
#plt.show()

#print grid
#vor = Voronoi(grid)
#voronoi_plot_2d(vor)
#plt.show()
#print ss

#INITIALISE STORAGE
u = np.zeros((I*J*K)) #potential
m = np.zeros((I*J*K)) #distribution
a1 = np.zeros((I*J*K)) #control
a2 = np.zeros((I*J*K))
u_old = np.zeros((I*J*K))
m_old = np.zeros((I*J*K))
a1_old = np.zeros((I*J*K))
a2_old = np.zeros((I*J*K))
ul1 = -1*np.ones((Niter,1))
ul2 = -1*np.ones((Niter,1))
ulinfty = -1*np.ones((Niter,1))
ml1 = -1*np.ones((Niter,1))
ml2 = -1*np.ones((Niter,1))
mlinfty = -1*np.ones((Niter,1))
a1l1 = -1*np.ones((Niter,1))
a1l2 = -1*np.ones((Niter,1))
a1linfty = -1*np.ones((Niter,1))
a2l1 = -1*np.ones((Niter,1))
a2l2 = -1*np.ones((Niter,1))
a2linfty = -1*np.ones((Niter,1))
def index(i,j,k): 
	return int(i+I*j+I*J*k)

#INITIAL/TERMINAL CONDITIONS, INITIAL GUESS and COPYING
m0 = iF.initial_distribution(x,y)
m0 = m0/(sum(sum(m0))*dx*dx) 
#print m0
for k in range (0,K):
	m[I*J*k:I*J*(k+1)] = np.ravel(m0)
m_old = np.copy(m)
u_old = np.copy(u)
a1_old = np.copy(a1)
a1_tmp = np.copy(a1)
a2_old = np.copy(a2)
a2_tmp = np.copy(a2)
time_total = time.time()
BIGZERO = np.zeros(x.size-2)
for n in range (0,Niter):
	titer = time.time()
	print "Computing iteration",n+1,"of u..."
	temptime = time.time()
	#Compute u
	u[(I*K*J-I*J):(I*K*J)] = iF.G(x,y,m[(I*J*K-I*J):(I*J*K)])
	for k in range (K-2,-1,-1): #this is extremely messy but might just work
		u_last = np.copy(u[((k+1)*I*J):((k+1)*I*J+I*J)]) #this one to keep
		m_tmp = np.copy(m[((k)*I*J):((k)*I*J+I*J)]) #only actually need this, amirite?
		a1_tmp = np.empty((x.size,y.size))
		a2_tmp = np.empty((x.size,y.size))
		#find the bestest controls for the entire grid...
		a1_tmp, a2_tmp = solve.control_general(xpts_search,x,y,u_last,m_tmp,dt,dx,k*dt,I,J,1e-4,scatters,N)
		L_var = np.ravel(iF.L_global(k*dt,x,y,a1_tmp,a2_tmp,m_tmp)) #this has been vetted
		[f1_array, f2_array] = iF.f_global(k*dt,x,y,a1_tmp,a2_tmp)
		D11 = iF.Sigma_D11_test(time,x,y,a1_tmp,a2_tmp,m_tmp)
		D12 = iF.Sigma_D12_test(time,x,y,a1_tmp,a2_tmp,m_tmp)
		D22 = iF.Sigma_D22_test(time,x,y,a1_tmp,a2_tmp,m_tmp)
		#and now generate the matrix and solve
		#U = mg.u_matrix_explicit(f1_array,f2_array,D11,D22,D12,I,J,dx,dt) #work matrix
		#u_tmp = U*np.ravel(u_last) + dt*L_var #I don't trust that this actually works, but hey
		U = mg.u_matrix_implicit(f1_array,f2_array,D11,D22,D12,I,J,dx,dt) #work matrix
		u_tmp = spsolve(U,np.ravel(u_last)+dt*L_var)
		#print U
		#print U.shape
		#U_I = 0.5*mg.u_matrix_implicit(f1_array,f2_array,D11,D22,D12,I,J,dx,dt) #work matrix
		#U_E = 0.5*mg.u_matrix_explicit(f1_array,f2_array,D11,D22,D12,I,J,dx,dt) #work matrix
		#u_tmp = spsolve(U_I,U_E*np.ravel(u_last)+dt*L_var)
		#plt.spy(U_WORK_4_DIS)
		#plt.show()
		u[(k*I*J):(k*I*J+I*J)] = np.copy(u_tmp)
		a1[(k*I*J):(k*I*J+I*J)] = np.copy(np.ravel(a1_tmp))
		a2[(k*I*J):(k*I*J+I*J)] = np.copy(np.ravel(a2_tmp))
		print (K-k-1)/(K-1)*100,"percent done..."
	print "Spent time", time.time()-temptime, "on computing u"
	#store changes in norms
	uchange = np.copy(u-u_old)
	ul1[n] = np.sum(abs(uchange))
	ul2[n] = np.sqrt(np.sum(abs(uchange)**2))
	ulinfty[n] = max(abs(uchange))
	a1change = np.copy(a1-a1_old)
	a1l1[n] = np.sum(abs(a1change))
	a1l2[n] = np.sqrt(np.sum(abs(a1change)**2))
	a1linfty[n] = max(abs(a1change) )
	a2change = np.copy(a2-a2_old)
	a2l1[n] = np.sum(abs(a2change))
	a2l2[n] = np.sqrt(np.sum(abs(a2change)**2))
	a2linfty[n] = max(abs(a2change) )
	#GET GOING WITH M
	print "Computing iteration", n+1, "of m..."
	temptime = time.time()
	for k in range(0,K-1): #COMPUTE M WHERE WE DO NOT ALLOW AGENTS TO LEAVE SUCH THAT m(-1) = m(N+1) = 1 ALWAYS
		a1_tmp = np.reshape(a1[(k*I*J):(k*I*J+I*J)],(I,J))
		a2_tmp = np.reshape(a2[(k*I*J):(k*I*J+I*J)],(I,J))
		m_tmp = m[(k*I*J):(k*I*J+I*J)]
		[f1_array, f2_array] = iF.f_global(k*dt,x,y,a1_tmp,a2_tmp)
		D11 = iF.Sigma_D11_test(time,x,y,a1_tmp,a2_tmp,m_tmp)
		D12 = iF.Sigma_D12_test(time,x,y,a1_tmp,a2_tmp,m_tmp)
		D22 = iF.Sigma_D22_test(time,x,y,a1_tmp,a2_tmp,m_tmp)
		#the actual computation
		#M = mg.m_matrix_explicit(f1_array,f2_array,D11,D22,D12,I,J,dx,dt) 
		#M = mg.m_matrix_explicit_diagonal_diffusion(f1_array,f2_array,D11,D22,I,J,dx,dt)
		#m_update = M*np.ravel(m_tmp) #computed!
		#M = mg.m_matrix_implicit(f1_array,f2_array,D11,D22,D12,I,J,dx,dt) 
		#m_update = spsolve(M,np.ravel(m_tmp))
		LHS,RHS = mg.m_matrix_iioe(f1_array,f2_array,D11,D22,D12,I,J,dx,dt) 
		m_update = spsolve(LHS,RHS*np.ravel(m_tmp))
		#print M
		m[I*J*(k+1):(I*J+I*J*(k+1))] = np.copy(m_update)
	print "Spent time", time.time()-temptime, "on computing m"
	#compute norms of stuff
	mchange = np.copy(m-m_old)
	ml1[n] = np.sum(abs(mchange))
	ml2[n] = np.sqrt(np.sum(abs(mchange)**2))
	mlinfty[n] = max(abs( mchange) ) 
	if (mlinfty[n] < tolerance):
		print "Method converged with final change" , mlinfty[n]
		print "Time spent:", time.time()-time_total
		kMax = n
		break
	#Evaluate iteration
	m_old = np.copy(m)
	u_old = np.copy(u)
	a1_old = np.copy(a1)
	a2_old = np.copy(a2)
	print "Iteration number", n+1, "completed. \nUsed time", time.time()-titer, "\nChange in (a1,a2,u,m)= (",  a1linfty[n][0], ",",a2linfty[n][0], ",", ulinfty[n][0], ",", mlinfty[n][0], ")"
print "Time spent:", time.time()-time_total
kMax = n

msoln = np.empty((I,J))
m0soln = np.empty((I,J))
usoln = np.empty((I,J))
a1soln = np.empty((I,J))
a2soln = np.empty((I,J))
for i in range (0,I):
	for j in range (0,J):
		msoln[i,j] = m[index(i,j,K-1)]
		m0soln[i,j] = m[index(i,j,0)]
		usoln[i,j] = u[index(i,j,0)]
		a1soln[i,j] = a1[index(i,j,0)]
		a2soln[i,j] = a2[index(i,j,0)]
msoln = np.transpose(msoln)
m0soln = np.transpose(m0soln)
usoln = np.transpose(usoln)
a1soln = np.transpose(a1soln)
a2soln = np.transpose(a2soln)
#cut the change vectors with kMax
ml1 = ml1[:kMax]
ml2 = ml2[:kMax]
mlinfty = mlinfty[:kMax]
ul1 = ul1[:kMax]
ul2 = ul2[:kMax]
ulinfty = ulinfty[:kMax]
#al1 = al1[:kMax]
#al2 = al2[:kMax]
#alinfty = alinfty[:kMax]
#init plotstuff
Xplot, Tplot = np.meshgrid(x,y)
#print Xplot.shape,Tplot.shape,msoln.shape,vsoln.shape,gradsoln.shape,mollgrad.shape
#plot solution of m(x,t)
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot_surface(Xplot,Tplot,msoln,rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('m(x,t)')
fig1.suptitle('Solution of the density m(x,t)', fontsize=14)
#plot solution of u(x,t)
fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111, projection='3d')
ax2.plot_surface(Xplot,Tplot,usoln,rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('u(x,t)')
fig2.suptitle('Solution of the potential u(x,t)', fontsize=14)
#initial distribution
fig3 = plt.figure(3)
ax3 = fig3.add_subplot(111, projection='3d')
ax3.plot_surface(Xplot,Tplot,m0,rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('m0(x,t)')
fig3.suptitle('Initial distribution', fontsize=14)

fig4 = plt.figure(4)
ax4 = fig4.add_subplot(111, projection='3d')
ax4.plot_surface(Xplot,Tplot,a1soln,rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax4.set_xlabel('x')
ax4.set_ylabel('y')
ax4.set_zlabel('a10(x,t)')
fig4.suptitle('Initial distribution', fontsize=14)

fig5 = plt.figure(5)
ax5 = fig5.add_subplot(111, projection='3d')
ax5.plot_surface(Xplot,Tplot,a2soln,rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax5.set_xlabel('x')
ax5.set_ylabel('y')
ax5.set_zlabel('a20(x,t)')
fig5.suptitle('Initial distribution', fontsize=14)

#plot solution of a(x,t)
#fig3 = plt.figure(3)
#ax3 = fig3.add_subplot(111, projection='3d')
#ax3.plot_surface(Xplot,Tplot,asoln,rstride=10,cstride=10,cmap=cm.coolwarm,linewidth=0, antialiased=False)
#ax3.set_xlabel('x')
#ax3.set_ylabel('t')
#ax3.set_zlabel('a(x,t)')
#fig3.suptitle('Solution of the control a(x,t)', fontsize=14)
#plot the norms of change on m
#fig4 = plt.figure(4)
#plt.plot(np.arange(1,kMax+1), np.log10(ml1), label="L1-norm")
#plt.plot(np.arange(1,kMax+1), np.log10(ml2), label="L2-norm")
#plt.plot(np.arange(1,kMax+1), np.log10(mlinfty), label="Linf-norm")
#legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')
#ax4 = fig4.add_subplot(111)
#ax4.set_xlabel('Iteration number')
#ax4.set_ylabel('Log10 of change')
#fig4.suptitle('Convergence of m(x,t)', fontsize=14)
#plot the norms of change on u
#fig5 = plt.figure(5)
#plt.plot(np.arange(1,kMax+1), np.log10(ul1), label="L1-norm")
#plt.plot(np.arange(1,kMax+1), np.log10(ul2), label="L2-norm")
#plt.plot(np.arange(1,kMax+1), np.log10(ulinfty), label="Linf-norm")
#legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')
#ax5 = fig5.add_subplot(111)
#ax5.set_xlabel('Iteration number')
#ax5.set_ylabel('Log10 of change')
#fig5.suptitle('Convergence of u(x,t)', fontsize=14)
#plot the norms of change on m
#fig6 = plt.figure(6)
#plt.plot(np.arange(1,kMax+1), np.log10(al1), label="L1-norm")
#plt.plot(np.arange(1,kMax+1), np.log10(al2), label="L2-norm")
#plt.plot(np.arange(1,kMax+1), np.log10(alinfty), label="Linf-norm")
#legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')
#ax6 = fig6.add_subplot(111)
#ax6.set_xlabel('Iteration number')
#ax6.set_ylabel('Log10 of change')
#fig6.suptitle('Convergence of a(x,t)', fontsize=14)
#fig7 = plt.figure(7)
#plt.plot(a[0:Nx])

##########PLOT
plt.show()









