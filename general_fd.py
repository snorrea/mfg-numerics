from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import time as time
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize as minimize
from matplotlib import cm
import quadrature_nodes as qn
import input_functions as iF


#in this one we aim to not use so much fucking space

#INPUTS; IF OSCILLATION AT BOUNDARY, MAKE SURE dt << dx
dx = 1/100
dt = 1/500
#dx = 0.75*0.1/2
#dx = 0.3**2/(2*0.7)
dx = 0.2**2/2
#dx = 0.1*dx
#print 
dt = dx**2/(0.3**2 + dx*2) # dt = dx**2/(max(sigma)**2 + dx*max(f))
print dx,dt
xmin = 0#-0.2
xmax = 1#+.2
T = 1
Niter = 100 #maximum number of iterations
tolerance = 1e-4
alpha_upper = 1
alpha_lower = -1
quad_order = 15
gll_x = qn.GLL_points(quad_order) #quadrature nodes
gll_w = qn.GLL_weights(quad_order,gll_x)

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

#INITIAL/TERMINAL CONDITIONS, INITIAL GUESS and COPYING
m0 = iF.initial_distribution(x)
m0 = m0/(sum(m0)*dx) #normalise
for k in range (0,K):
	m[k*I:k*I+I] = np.copy(m0)
m_old = np.copy(m)
u_old = np.copy(u)
a_old = np.copy(a)
a_tmp = np.copy(a)
time_total = time.time()
BIGZERO = np.zeros(x.size-2)
for n in range (0,Niter):
	titer = time.time()
	print "Computing iteration",n+1,"of u..."
	temptime = time.time()
	#Compute u
	u[(I*K-I):(I*K)] = iF.G(x,m[(I*K-I):(I*K)])
	for k in range (K-2,-1,-1): #this is extremely messy but might just work
		#u_tmp = np.copy(u[((k+1)*I):((k+1)*I+I)]) #this one to iterate over
		u_tmp = np.empty(x.size)
		u_last = np.copy(u[((k+1)*I):((k+1)*I+I)]) #this one to keep
		m_tmp = np.copy(m[((k)*I):((k)*I+I)]) #only actually need this, amirite?
		#a_last = np.copy(a_old[((k)*I):((k)*I+I)]) #this one to iterate over
		#a_last = np.zeros(x.size) #this one to iterate over
		################################
		#POLICY ITERATION HERE
		###############################
		#a_tmp = iF.best_control(x,u_last,m_tmp,dt,dx,k*dt)
		a_tmp = np.empty(x.size)
		#u_choice = np.empty(x.size)
		for i in range (0,Nx):
			fpts = iF.hamiltonian(xpts_search,x,u_last,m_tmp,dt,dx,k*dt,i)
			x0 = xpts_search[np.argmin(fpts)]
			tmp,tmpval = iF.scatter_search(iF.hamiltonian,(x,u_last,m_tmp,dt,dx,k*dt,i),xpts_search[2]-xpts_search[1],x0,N,scatters) 
			a_tmp[i] = tmp
			#u_choice[i] = tmpval
		sigma2 = iF.Sigma_global(k*dt,x,a_tmp,m_tmp)**2
		L_var = iF.L_global(k*dt,x,a_tmp,m_tmp)
		movement = iF.f_global(k*dt,x,a_tmp)
		#u_tmp = np.copy(u_last) + dt*u_choice
		u_tmp[1:-1] = u_last[1:-1]*(1-sigma2[1:-1]*dt/dx2 + abs(movement[1:-1])*dt/dx) + u_last[2:]*(sigma2[1:-1]*dt/(2*dx2) + np.minimum(movement[1:-1],BIGZERO)*dt/dx) +  u_last[0:-2]*(sigma2[1:-1]*dt/(2*dx2) - np.maximum(movement[1:-1],BIGZERO)*dt/dx) + dt*L_var[1:-1]
		u_tmp[0] = u_last[0] + dt*(abs(movement[0])/dx - sigma2[0]/dx2)*(u_last[0] - u_last[1]) + dt*L_var[0]
		u_tmp[-1] = u_last[-1] + dt*(abs(movement[-1])/dx - sigma2[-1]/dx2)*(u_last[-1] - u_last[-2]) + dt*L_var[-1]
		u[(k*I):(k*I+I)] = np.copy(u_tmp)
		a[(k*I):(k*I+I)] = np.copy(a_tmp)
	print "Spent time", time.time()-temptime, "on computing u"
	#store changes in norms
	uchange = np.copy(u-u_old)
	ul1[n] = np.sum(abs(uchange))
	ul2[n] = np.sqrt(np.sum(abs(uchange)**2))
	ulinfty[n] = max(abs(uchange))
	achange = np.copy(a-a_old)
	al1[n] = np.sum(abs(achange))
	al2[n] = np.sqrt(np.sum(abs(achange)**2))
	alinfty[n] = max(abs(achange) )
	#GET GOING WITH M
	print "Computing iteration", n+1, "of m..."
	temptime = time.time()
	for k in range(0,K-1): #COMPUTE M WHERE WE DO NOT ALLOW AGENTS TO LEAVE SUCH THAT m(-1) = m(N+1) = 1 ALWAYS
		a_tmp = a[(k*I):(k*I+I)]
		m_tmp = m[(k*I):(k*I+I)]
		sigma2 = iF.Sigma_global(k*dt,x,a_tmp,m_tmp)**2
		L_var = iF.L_global(k*dt,x,a_tmp,m_tmp)
		movement = iF.f_global(k*dt,x,a_tmp) #the function f
		#the actual computation
		m_update = np.empty(m_tmp.size)
		#m_update[1:-1] = m_tmp[1:-1]*(1-sigma2[1:-1]*dt/dx2-abs(movement[1:-1])*dt/dx) + m_tmp[2:]*(dt/dx)*(sigma2[1:-1]/(2*dx) - np.minimum(movement[1:-1],BIGZERO)) + m_tmp[0:-2]*(dt/dx)*(sigma2[1:-1]/(2*dx) + np.maximum(movement[1:-1],BIGZERO))
		#m_update[0] = m_tmp[0] + dt/(2*dx2) * (2*sigma2[1]*m_tmp[1] - 2*sigma2[0]*m_tmp[0] ) - dt/dx * ( max(0,movement[0])*(m_tmp[0]) + min(0,movement[0])*(m_tmp[1]-m_tmp[0]) )
		#m_update[-1] = m_tmp[-1] + dt/(2*dx2) * (2*sigma2[-2]*m_tmp[-2] - 2*sigma2[-1]*m_tmp[-1] ) - dt/dx * ( max(0,movement[-1] ) * (m_tmp[-1]-m_tmp[-2]) + min(0,movement[-1]) * ( - m_tmp[-1] ) )
		#m_update[1:-1] = m_tmp[1:-1]*(1-sigma2[1:-1]*dt/dx2) + m_tmp[2:]*(dt/(2*dx))*(sigma2[1:-1]/dx - movement[2:]) + m_tmp[0:-2]*(dt/(2*dx))*(sigma2[1:-1]/dx + movement[0:-2])
		#m_update[0] = m_tmp[0]*(1-sigma2[0]*dt/dx2) + m_tmp[1]*dt*(sigma2[1]/dx2 - 0.5*(movement[1])/dx)
		#m_update[-1] = m_tmp[0]*(1-sigma2[-1]*dt/dx2) + m_tmp[-2]*dt*(sigma2[-2]/dx2 + 0.5*(movement[-2])/dx)
		m_update[1:-1] = m_tmp[1:-1]*(1-sigma2[1:-1]*dt/dx2) + m_tmp[2:]*(dt/(2*dx))*(sigma2[2:]/dx - movement[2:]) + m_tmp[0:-2]*(dt/(2*dx))*(sigma2[0:-2]/dx + movement[0:-2])
		m_update[0] = m_tmp[0]*(1-sigma2[0]*dt/dx2) + m_tmp[1]*dt*(sigma2[1]/dx2 - 0.5*(movement[1])/dx)
		m_update[-1] = m_tmp[0]*(1-sigma2[-1]*dt/dx2) + m_tmp[-2]*dt*(sigma2[-2]/dx2 + 0.5*(movement[-2])/dx)
		#m_update[1:-1] = m_tmp[1:-1]*(1-sigma2[1:-1]*dt/dx2 - dt/(2*dx)*(movement[2:]-movement[0:-2] + 2*abs(movement[1:-1]) )) + dt*m_tmp[2:]*(sigma2[1:-1]/dx2 - np.minimum(movement[1:-1],BIGZERO)/dx) + dt*m_tmp[0:-2]*(sigma2[1:-1]/dx2 + np.maximum(movement[1:-1],BIGZERO)/dx)
		#m_update[0] = m_tmp[0]*(1-sigma2[0]*dt/dx2 - dt/(2*dx)*(movement[1]+2*abs(movement[0]))) + m_tmp[1]*dt*(sigma2[1]/dx2 + abs(movement[1])/dx)
		#m_update[-1] = m_tmp[0]*(1-sigma2[-1]*dt/dx2 - dt/(2*dx)*(-movement[-2]+2*abs(movement[-1]))) + m_tmp[-2]*dt*(sigma2[-2]/dx2 + abs(movement[-2])/dx)
		#the actual update
		#m_update = m_update/(sum(m_update)*dx) #normalise; this is a hack
		m[I*(k+1):(I+I*(k+1))] = np.copy(m_update)
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
Xplot, Tplot = np.meshgrid(x,t)
#print Xplot.shape,Tplot.shape,msoln.shape,vsoln.shape,gradsoln.shape,mollgrad.shape
#plot solution of m(x,t)
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot_surface(Xplot,Tplot,msoln,rstride=10,cstride=10,cmap=cm.coolwarm,linewidth=0, antialiased=False)
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
plt.plot(np.arange(1,kMax+1), np.log10(ml1), label="L1-norm")
plt.plot(np.arange(1,kMax+1), np.log10(ml2), label="L2-norm")
plt.plot(np.arange(1,kMax+1), np.log10(mlinfty), label="Linf-norm")
legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')
ax4 = fig4.add_subplot(111)
ax4.set_xlabel('Iteration number')
ax4.set_ylabel('Log10 of change')
fig4.suptitle('Convergence of m(x,t)', fontsize=14)
#plot the norms of change on u
fig5 = plt.figure(5)
plt.plot(np.arange(1,kMax+1), np.log10(ul1), label="L1-norm")
plt.plot(np.arange(1,kMax+1), np.log10(ul2), label="L2-norm")
plt.plot(np.arange(1,kMax+1), np.log10(ulinfty), label="Linf-norm")
legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')
ax5 = fig5.add_subplot(111)
ax5.set_xlabel('Iteration number')
ax5.set_ylabel('Log10 of change')
fig5.suptitle('Convergence of u(x,t)', fontsize=14)
#plot the norms of change on m
fig6 = plt.figure(6)
plt.plot(np.arange(1,kMax+1), np.log10(al1), label="L1-norm")
plt.plot(np.arange(1,kMax+1), np.log10(al2), label="L2-norm")
plt.plot(np.arange(1,kMax+1), np.log10(alinfty), label="Linf-norm")
legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')
ax6 = fig6.add_subplot(111)
ax6.set_xlabel('Iteration number')
ax6.set_ylabel('Log10 of change')
fig6.suptitle('Convergence of a(x,t)', fontsize=14)
fig7 = plt.figure(7)
plt.plot(a[0:Nx])

##########PLOT
plt.show()








