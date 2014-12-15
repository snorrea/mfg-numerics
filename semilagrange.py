from __future__ import division
import numpy as np
import scipy.weave as weave
import matplotlib.pyplot as plt
import time as time
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize as minimize
from matplotlib import cm
import quadrature_nodes as qn
import sys

#in this one we aim to not use so much fucking space

#INPUTS
dx = 0.0075*2#1/300 #these taken from Gueant's paper
dt = dx*2
xmin = 0-0.1
xmax = 1+.1
T = 1
Niter = 10 #maximum number of iterations
tolerance = 1e-3
epsilon = 3*dt #for use in convolution thing
sigma = 2
noise = 0.2 #noise in the MFG sense
molly = 1
quad_order = 15
second_order = 0 #1 for second-order, 0 for first order

#CRUNCH
gll_x = qn.GLL_points(quad_order) #quadrature nodes
gll_w = qn.GLL_weights(quad_order,gll_x) #quadrature weights
dx2 = dx**2
Nx = int(abs(xmax-xmin)/dx)
Nt = int(T/dt) 
#if (Nx-abs(xmax-xmin)/dx is not 0) or (Nt-T/dt is not 0):
#	print Nx-abs(xmax-xmin)/dx, N-T/dt
#	sys.exit("Grid size not possible.")
I = int(Nx)+1 #space
K = int(Nt)+1 #time
x = np.arange(xmin,xmax,dx)
t = np.arange(0,T,dt)
#print x
#print t
#print x.size
#print K,t.size
def index(i,k): 
	return int(i+(I)*k)
def convolution(eps,x): #this is the convolution function used in the paper, use like np.convolve(u,v)
	return 1/eps * 1/np.sqrt(2*np.pi) * np.exp(-(x/eps)**2/2)
def beta(x_val,i,x_array):
	return np.maximum(0,1-abs(x_val-x_array[i])/dx)
def tau(alpha,i,v_array,x_array): #the function to be minimised
	tmp = 0.5*dt*alpha**2
	if second_order==0:
		for j in range (0,I):
			tmp = tmp + v_array[j]*beta(x_array[i]-dt*alpha,j,x_array)
	else:
		for j in range (0,I):
			tmp = tmp + 0.5*v_array[j]*(beta(x_array[i]-dt*alpha+np.sqrt(dt)*noise,j,x_array)+beta(x_array[i]-dt*alpha-np.sqrt(dt)*noise,j,x_array))
	return tmp
def mollifier(x_val): #Evans' mollifier
	if abs(x_val) < 1:
		return np.exp(1/(x_val**2 - 1))/0.443994
	else:
		return 0
def mollify_array(array,epsilon,x_array,gll_x,gll_w): 
	output = np.zeros((array.size))
	for k in range (0,array.size):
		for j in range (0,gll_x.size):
			tmp = 0
			for i in range (0,array.size):
				tmp += array[i]*beta(x_array[k]-epsilon*(gll_x[j]),i,x_array)
			output[k] += tmp*gll_w[j]*mollifier(gll_x[j])
	return output
def min_approx1(function, (args)): #for some reason this is twice as slow as the built-in one :(
	left = -10
	right = 10
	linesearch_decrement = 0.5
	linesearch_tolerance = 1/50
	dx = 1
	Nx = 5
	#now the sector has been found; dish out points and do some kind of simplex method
	#xpts = np.linspace(left,right,abs(right-left)/dx) #our points
	xpts = np.linspace(left,right,Nx)
	fpts = np.empty((xpts.size,1))
	fpts = function(xpts,*args)
	x0 = xpts[np.argmin(fpts)] #choose smallest
	h = dx/2
	while True:
		if function(x0+h,*args) < function(x0-h,*args): #go right
			x0 += h
			h = h*linesearch_decrement
		elif function(x0-h,*args) < function(x0+h,*args): #go left
			x0 += -h
			h = h*linesearch_decrement
		elif h > linesearch_tolerance:
			h = h*linesearch_decrement
		else:
			break
	#print "Time to minimise:",time.time()-t1
	return function(x0,*args)
#def my_own_gradient(arr,dx):
#	siz = arr.size
#	returnarray = np.zeros(arr.size)
#	for i in range (0,arr.size):
#		if i==0:
#			returnarray[i] = 
#		returnarray[i] = (arr[i+1]-arr[i-1])/dx


#FUNCTIONS
def G(xi,m_array): #this is the final cost, and is a function of the entire distribution m and each point x_i
#	return -0.5*(xi+0.5)**2 * (1.5-xi)**2 #Carlini's original
	return (xi*(1-xi))**2 #Gueant's game
	#return 0 #Carlini's no-game

def F_global(x_array,m_array,sigma): #more effective running cost function
	#return (x-0.2)**2 #Carlini's no-game
	return min(1.4,max(m_array,0.7))
#	tmp = mollify_array(m_array,sigma,x_array,gll_x,gll_w)
#	return 2*mollify_array(tmp,sigma,x_array,gll_x,gll_w)

def F_local(x_array,m_array,sigma,index):
	return min(1.4,max(m_array[index],0.7))

m0 = np.empty(I)
for i in range (0,x.size):
	if x[i] >=0 and x[i]<=1:
		m0[i] = 1-0.2*np.cos(np.pi*x[i]) #gueant's original, carlini's game
		#m0[i] = np.exp(-(x[i]-0.75)**2/0.1**2)#/0.177209 #carlini's no-game
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
time_total = time.time()
for n in range (0,Niter):
	#copy old solutions to use for error comparison thing
	titer = time.time()
	#compute next iteration of v given m_old
	print "Computing iteration", n+1, "of v..."
	temptime = time.time()
	v[(I*K-I):(I*K)] = np.copy(G(x,m)) 
	for k in range (K-2,-1,-1):
		v_tmp = np.copy(v[((k+1)*I):((k+1)*I+I)]) #this is correct
		#F_var = F_global(x,m[((k+1)*I):((k+1)*I+I)],sigma)
		for i in range (0,I):
			tmp = minimize(tau,0,args=(i,v_tmp,x))
			#v[index(i,k)] = dt*F_var[i] + tmp.fun
			v[index(i,k)] = dt*F_local(x,m[((k+1)*I):((k+1)*I+I)],sigma,i) + tmp.fun
			#tmp = min_approx1(tau,(i,v_tmp,x)) 
			#v[index(i,k)] = dt*F(x,m,i,k) + tmp
	print "Spent time", time.time()-temptime
	vchange = np.copy(v-v_old)
	vl1[n] = np.sum(abs(vchange))
	vl2[n] = np.sqrt(np.sum(abs(vchange)**2))
	vlinfty[n] = max(abs( vchange) )
	if (vlinfty[n] < tolerance):
		print "Method converged with final change" , vl2[n]
		print "Time spent:", time.time()-time_total
		kMax = n
		break

	#initial condition on m is already set, compute the rest of them
	print "Computing iteration", n+1, "of m..."
	temptime = time.time()
	for k in range(0,K-1):
		if molly==1:
			v_grad = mollify_array(np.gradient(v[(I*k):(I*k+I)],dx),epsilon,x,gll_x,gll_w) #this is completely fine
		else:
			v_grad = np.gradient(v[(I*k):(I*k+I)],dx)  #this is okay as is, but I suspect there is something wrong just because
		for i in range (0,I): 
			m[index(i,k+1)] = 0
			if second_order==0:
				for j in range (0,I): 
					m[index(i,k+1)] += beta(x[j]-dt*v_grad[j],i,x)*m[index(j,k)]
			else:
				for j in range (0,I): 
					m[index(i,k+1)] += 0.5*(beta(x[j]-dt*v_grad[j]+np.sqrt(dt)*noise,i,x)+beta(x[j]-dt*v_grad[j]-np.sqrt(dt)*noise,i,x))*m[index(j,k)]
	print "Spent time", time.time()-temptime
	#compute norms of stuff
	mchange = np.copy(m-m_old)
	ml1[n] = np.sum(abs(mchange))
	ml2[n] = np.sqrt(np.sum(abs(mchange)**2))
	mlinfty[n] = max(abs( mchange) ) 
	#Evaluate iteration
	m_old = np.copy(m)
	v_old = np.copy(v)
	print "Iteration number", n+1, "completed, used time", time.time()-titer, "with change in m", ml2[n], "and change in v", vl2[n]
	
print "Time spent:", time.time()-time_total
kMax = n

msoln = np.empty((I,K))
vsoln = np.empty((I,K))
gradsoln1 = np.empty((I*K))
gradsoln = np.empty((I,K))
mollgrad1 = np.empty((I*K))
mollgrad = np.empty((K,I))
for k in range (0,K):
	gradsoln1[(I*k):(I*k+I)] = np.gradient(v[(I*k):(I*k+I)],dx)
	mollgrad1[(I*k):(I*k+I)] = mollify_array(np.gradient(v[(I*k):(I*k+I)],dx),epsilon,x,gll_x,gll_w)
for i in range (0,I):
	for k in range (0,K):
		msoln[i,k] = m[index(i,k)]
		vsoln[i,k] = v[index(i,k)]
		gradsoln[i,k] = gradsoln1[index(i,k)]
		mollgrad[k,i] = mollgrad1[index(i,k)]
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
Xplot, Tplot = np.meshgrid(x,t)
print Xplot.shape,Tplot.shape,msoln.shape,vsoln.shape,gradsoln.shape,mollgrad.shape
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
#plot gradient
fig5 = plt.figure(5)
ax5 = fig5.add_subplot(111, projection='3d')
ax5.plot_surface(Xplot,Tplot,gradsoln,rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax5.set_xlabel('x')
ax5.set_ylabel('t')
ax5.set_zlabel('grad v(x,t)')
fig5.suptitle('Solution of gradient of v(x,t)', fontsize=14)
#sss
fig6 = plt.figure(6)
ax6 = fig6.add_subplot(111, projection='3d')
ax6.plot_surface(Xplot,Tplot,mollgrad,rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax6.set_xlabel('x')
ax6.set_ylabel('t')
ax6.set_zlabel('mollified grad v(x,t)')
fig5.suptitle('Solution of gradient of v(x,t)', fontsize=14)
plt.show()




