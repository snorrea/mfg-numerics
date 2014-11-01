from __future__ import division
import numpy as np
import scipy.weave as weave
import matplotlib.pyplot as plt
import time as time
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize as minimize
from matplotlib import cm
from scipy.special import erf as erf

#in this one we aim to not use so much fucking space

#INPUTS
dx = 1/20 #these taken from Gueant's paper
dt = 1/20
xmin = -0.25
xmax = 1.25
T = 1
Niter = 3 #maximum number of iterations
tolerance = 1e-6
epsilon = 0.03 #for use in convolution thing

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
	return 1/eps * 1/np.sqrt(2*np.pi) * np.exp(-(x/eps)**2/2)
def beta(x,i,x_array):
	return max(0,1-abs(x-x_array[i])/dx)
def tau(alpha,i,k,v_array,x_array): #the function to be minimised
	tmp = 0.5*dt*alpha**2
	for j in range (0,I):
		tmp = tmp + v_array[j]*beta(x_array[i]-dt*alpha,j,x_array)
	return tmp
def mollifyweight(epsilon,x_array,index): #need to be absolutely sure this is right...
	return 0.5 * ( erf((x_array[abs(index)]+0.5*dx)/(np.sqrt(2)*epsilon)) - erf((x_array[abs(index)]-0.5*dx)/(np.sqrt(2)*epsilon)) )
def mollify(array,epsilon,x_array): #mollifyweight spits out zero all the time, staaaahhp
	output = np.empty((array.size))
	for j in range (0,array.size):
		output[j] = 0
		w = mollifyweight(epsilon,x_array,j)
		for i in range(-j,array.size-j):
			output[j] += w*array[j+i]
	return output


#FUNCTIONS
def G(xi,m_array): #this is the final cost, and is a function of the entire distribution m and each point x_i
	return -0.5*(xi+0.5)**2 * (1.5-xi)**2 #Carlini's original

def F(x_array,m_array,space_index,time_index): #this is the running cost
	return 2*abs(x_array[space_index])#+0.5*m_array[index(space_index,time_index)]

m0 = np.empty(I)
for i in range (0,x.size):
	if x[i] >=0 and x[i]<=1:
		m0[i] = 1-0.2*np.cos(np.pi*x[i]) #gueant's original
	else:
		m0[i] = 0

v = np.zeros((I*K)) #potential
m = np.empty((I*K)) #distribution
v_old = np.empty((I*K))
m_old = np.empty((I*K))
v_grad = np.empty((I))
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


#set final value on v and copy the bitches
#tmp = G(x,m)
#print tmp.size,I
v[(I*K-I):(I*K)] = np.copy(G(x,m))
m_old = np.copy(m)
v_old = np.copy(v)
print "Initialisation done, crunching..."

#THINGS THAT ARE WRONG:
	#convolution method with its weights
	#F seems to give same value for m and m_old...?

for n in range (0,Niter):
	#copy old solutions to use for error comparison thing
	
	titer = time.time()
	#compute next iteration of v given m_old
	print "Computing iteration", n+1, "of v..."
	temptime = time.time()
	for k in range (K-2,-1,-1):
		v_tmp = np.copy(v[((k+1)*I):((k+1)*I+I)])
		for i in range (0,I):
			tmp = minimize(tau,0,args=(i,k,v_tmp,x)) #this dominates time consumption! and works
			v[index(i,k)] = dt*F(x,m,i,k) + tmp.fun
	print "Spent time", time.time()-temptime

	#initial condition on m is already set, compute the rest of them
	print "Computing iteration", n+1, "of m..."
	temptime = time.time()
	for k in range(0,K-1):
		#smoothify the thing; this is fucked
		v_grad = np.gradient(v[(I*k):(I*k+I)],dx)  #this is okay as is
		#v_grad = mollify(np.gradient(v[(I*k):(I*k+I)],dx),epsilon,x) #mollifier apparently sets everything to zero
		for i in range (0,I): 
			m[index(i,k+1)] = 0
			for j in range (0,I): 
				m[index(i,k+1)] += beta(x[j]-dt*v_grad[j],i,x)*m[index(j,k)]
	print "Spent time", time.time()-temptime
	#compute norms of stuff
	mchange = np.copy(m-m_old)
	vchange = np.copy(v-v_old)
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

	

	#Evaluateiteration
	m_old = np.copy(m)
	v_old = np.copy(v)
	print "Iteration number", n+1, "completed, used time", time.time()-titer, "with change in m", mlinfty[n], "and change in v", vlinfty[n]
	

kMax = n
msoln = np.empty((I,K))
vsoln = np.empty((I,K))
gradsoln1 = np.empty((I*K))
gradsoln = np.empty((I,K))
for k in range (0,K):
	gradsoln1[(I*k):(I*k+I)] = np.gradient(v[(I*k):(I*k+I)],dx)
for i in range (0,I):
	for k in range (0,K):
		msoln[i,k] = m[index(i,k)]
		vsoln[i,k] = v[index(i,k)]
		gradsoln[i,k] = gradsoln1[index(i,k)]
msoln = np.transpose(msoln)
vsoln = np.transpose(vsoln)
gradsoln = np.transpose(gradsoln)
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

#plot solution of m(x,t)
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot_surface(Xplot,Tplot,msoln,rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax1.set_xlabel('x')
ax1.set_ylabel('t')
ax1.set_zlabel('m(x,t)')
fig1.suptitle('Solution of density m(x,t)', fontsize=14)
#plot solution of u(x,t)
fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111, projection='3d')
ax2.plot_surface(Xplot,Tplot,vsoln,rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax2.set_xlabel('x')
ax2.set_ylabel('t')
ax2.set_zlabel('u(x,t)')
fig2.suptitle('Solution of potential v(x,t)', fontsize=14)
#plot the norms of change on u
#fig3 = plt.figure(3)
#plt.plot(np.arange(1,kMax+1), np.log10(ml1), label="L1-norm")
#plt.plot(np.arange(1,kMax+1), np.log10(ml2), label="L2-norm")
#plt.plot(np.arange(1,kMax+1), np.log10(mlinfty), label="Linf-norm")
#legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')
#ax3 = fig3.add_subplot(111)
#ax3.set_xlabel('Iteration number')
#ax3.set_ylabel('Log10 of change')
#fig3.suptitle('Convergence of m(x,t)', fontsize=14)
#plot the norms of change on m
#fig4 = plt.figure(4)
#plt.plot(np.arange(1,kMax+1), np.log10(vl1), label="L1-norm")
#plt.plot(np.arange(1,kMax+1), np.log10(vl2), label="L2-norm")
#plt.plot(np.arange(1,kMax+1), np.log10(vlinfty), label="Linf-norm")
#legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')
#ax4 = fig4.add_subplot(111)
#ax4.set_xlabel('Iteration number')
#ax4.set_ylabel('Log10 of change')
#fig4.suptitle('Convergence of v(x,t)', fontsize=14)
#plot gradient
fig5 = plt.figure(5)
ax5 = fig5.add_subplot(111, projection='3d')
ax5.plot_surface(Xplot,Tplot,gradsoln,rstride=5,cstride=5,cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax5.set_xlabel('x')
ax5.set_ylabel('t')
ax5.set_zlabel('grad v(x,t)')
fig5.suptitle('Solution of gradient of v(x,t)', fontsize=14)
plt.show()




