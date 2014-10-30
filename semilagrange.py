from __future__ import division
import numpy as np
import scipy.weave as weave
import matplotlib.pyplot as plt
import time as time
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize as minimize
from matplotlib import cm

#in this one we aim to not use so much fucking space

#INPUTS
dx = 1/10 #these taken from Gueant's paper
dt = 1/160
xmin = 0
xmax = 1
T = 1
Niter = 20 #maximum number of iterations
tolerance = 1e-6
epsilon = 0.003 #for use in convolution thing

#CRUNCH
dx2 = dx**2
Nx = int(abs(xmax-xmin)/dx)
Nt = int(T/dt) 
I = int(Nx)+1 #space
K = int(Nt)+1 #time
x = np.arange(xmin,xmax+dx,dx)
t = np.arange(0,T+dt,dt)
def index(i,k): #this is a jolly source of errors, no more, probably still
	return int(i+(I)*k)
def convolution(eps,x): #this is the convolution function used in the paper, use like np.convolve(u,v)
	return 1/eps * 1/np.sqrt(2*np.pi) * exp(-(x/eps)**2/2)
def beta(x,i,x_array):
	return max(0,1-abs(x-x_array[i])/dx)
def tau(alpha,i,k,v_array,x_array): #the function to be minimised
	tmp = 0.5*dt*alpha**2
	for j in range (0,I):
		tmp = tmp + v_array[j]*beta(x_array[i]-dt*alpha,j,x_array)
	return tmp


#FUNCTIONS
def G(xi,m_array): #this is the final cost, and is a function of the entire distribution m and each point x_i
	return -0.5*(xi+0.5)**2 * (1.5-xi)**2 #Carlini's original

def F(xi,m_array): #this is the running cost
	return 0.3*(xi-0.5)**2

sigma2 = 0.8**2
m0 = 1-0.2*np.cos(np.pi*x) #gueant's original
v = np.zeros((I*K)) #potential
m = np.zeros((I*K)) #distribution
v_old = np.zeros((I*K))
m_old = np.zeros((I*K))
v_grad = np.zeros((I*K))
#initial guess on the distribution
for k in range (0,K):
	m[(I*k):(I*k+I)] = np.copy(m0)

#HOW TO MINIMIZE
#minimize(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None)
#minimize(tau,0,args=(i,k,v_array,x_array))
#minimize(tau,0,args=(i,k,m_old,x))
#tmp = minimize(tau,0,args=(1,2,m_old,x))
#print tmp.fun #this returns the minimum value of the function 

#initialise vectors to store l1, l2 and linfty norm errors/improvement in iterations
vl1 = -1*np.ones((Niter,1))
vl2 = -1*np.ones((Niter,1))
vlinfty = -1*np.ones((Niter,1))
ml1 = -1*np.ones((Niter,1))
ml2 = -1*np.ones((Niter,1))
mlinfty = -1*np.ones((Niter,1))

for n in range (0,Niter):
	#copy old solutions to use for error comparison thing
	m_old = np.copy(m)
	v_old = np.copy(v)
	titer = time.time()

	#compute next iteration of v given m_old
	v[(I*K-I):(I*K)] = G(x,m_old) #think this works
	#v[(I*K-I):(I*K)] = G(x,m)
	for k in range (K-2,-1,-1):
		v_tmp = np.copy(v[((k+1)*I):((k+1)*I+I)])
		for i in range (0,I):
			#print i,k
			tmp = minimize(tau,0,args=(i,k,v_tmp,x))
			v[index(i,k)] = dt*F(x[i],m_old) + tmp.fun

	##compute the control based on the convoluted potential v
	#convolute the potential in space
		#let's skip that for now

	#find the gradient in space, which happens to be the fucking control/state/whatever
	#do it manually because someone is a fucking moron
	for k in range (0,K-1):
		#print I*k+I
		#v_grad[(I*k):(I*k+I)] = np.gradient(v[(I+k):(I*k+I)],dx)
		v_grad[(I*k)] = (v[I*k]-v[I*k+1])/dx
		v_grad[(I*k)+I] = (v[I*k+I-1]-v[I*k+I])/(dx)
		v_grad[(I*k+1):(I*k+I-1)] = (v[(I*k):(I*k+I-2)]-v[(I*k+2):(I*k+I)])/(2*dx)
	#print v_grad

	#initial condition on m is already set, compute the rest of them
	#this is where it fucks up
	#set initial conditions again I guess?
#	m[0:I] = np.copy(m0)
	for k in range(0,K-1):
		for i in range (0,I):
			m[index(i,k+1)] = 0
			if m[index(i,k)] > 2: #stuff accumulates in i=10 like a bitch
				print i,k,m[index(i,k)]
			for j in range (0,I):
				m[index(i,k+1)] += beta(x[j]-dt*v_grad[index(j,k)],i,x)*m[index(j,k)]

	#compute norms of stuff
	mchange = m-m_old
	vchange = v-v_old
	ml1[n] = np.sum(abs(mchange))
	ml2[n] = np.sqrt(np.sum(abs(mchange)**2))
	mlinfty[n] = max(abs( mchange) ) 
	vl1[n] = np.sum(abs(vchange))
	vl2[n] = np.sqrt(np.sum(abs(vchange)**2))
	vlinfty[n] = max(abs( vchange) ) 
	if (mlinfty[n] < tolerance) and (vlinfty[n] < tolerance):
		print "Method converged with final change" , mlinfty[n], "and", vlinfty[n]
		kMax = n
		break

	#MIX IT UP NIGGA
	print "Iteration number", n+1, "completed, used time", time.time()-titer, "with change", mlinfty[n], "and", vlinfty[n]

kMax = n
msoln = np.empty((I,K))
vsoln = np.empty((I,K))
for i in range (0,I):
	for k in range (0,K):
		msoln[i,k] = m[index(i,k)]
		vsoln[i,k] = v[index(i,k)]
msoln = np.transpose(msoln)
vsoln = np.transpose(vsoln)
#cut the change vectors with kMax
ml1 = ml1[:kMax]
ml2 = ml2[:kMax]
mlinfty = mlinfty[:kMax]
vl1 = vl1[:kMax]
vl2 = vl2[:kMax]
vlinfty = vlinfty[:kMax]

#init plotstuff
#xi = np.linspace(xmin,xmax,Nx)
#ti = np.linspace(0,T,Nt)
Xplot, Tplot = np.meshgrid(x,t)

#print msoln.shape, Xplot.shape
#print xi

#plot solution of m(x,t)
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot_surface(Xplot,Tplot,msoln,rstride=5,cstride=5,cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax1.set_xlabel('x')
ax1.set_ylabel('t')
ax1.set_zlabel('m(x,t)')
fig1.suptitle('Solution of density m(x,t)', fontsize=14)
#plot solution of u(x,t)
fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111, projection='3d')
ax2.plot_surface(Xplot,Tplot,vsoln,rstride=5,cstride=5,cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax2.set_xlabel('x')
ax2.set_ylabel('t')
ax2.set_zlabel('u(x,t)')
fig2.suptitle('Solution of potential v(x,t)', fontsize=14)
#plot the norms of change on u
fig3 = plt.figure(3)
plt.plot(np.arange(1,kMax+1), np.log10(ml1), label="L1-norm")
plt.plot(np.arange(1,kMax+1), np.log10(ml2), label="L2-norm")
plt.plot(np.arange(1,kMax+1), np.log10(mlinfty), label="Linf-norm")
legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')
ax3 = fig3.add_subplot(111)
ax3.set_xlabel('Iteration number')
ax3.set_ylabel('Log10 of change')
fig3.suptitle('Convergence of m(x,t)', fontsize=14)
#plot the norms of change on m
fig4 = plt.figure(4)
plt.plot(np.arange(1,kMax+1), np.log10(vl1), label="L1-norm")
plt.plot(np.arange(1,kMax+1), np.log10(vl2), label="L2-norm")
plt.plot(np.arange(1,kMax+1), np.log10(vlinfty), label="Linf-norm")
legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')
ax4 = fig4.add_subplot(111)
ax4.set_xlabel('Iteration number')
ax4.set_ylabel('Log10 of change')
fig4.suptitle('Convergence of v(x,t)', fontsize=14)
#plt.show()



