from __future__ import division
import numpy as np
import scipy.weave as weave
import matplotlib.pyplot as plt
import time as time
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize as minimize
from matplotlib import cm
import quadrature_nodes as qn
import input_functions as iF
import sys

#in this one we aim to not use so much fucking space

#INPUTS
dx = 0.0075*2#/2#/2#/2#/2#1/300 #these taken from Gueant's paper
dt = dx*1.5
xmin = 0#-0.2
xmax = 1#+.2
T = 1
Niter = 500 #maximum number of iterations
tolerance = 1e-4
epsilon = 3*dt #for use in convolution thing
sigma = 10 #for use in cost function
noise = 0.9 #noise in the MFG sense
molly = 1
second_order = 0 #1 for second-order, 0 for first order
quad_order = 15
R = 10

#CRUNCH
gll_x = qn.GLL_points(quad_order) #quadrature nodes
gll_w = qn.GLL_weights(quad_order,gll_x) #quadrature weights
dx2 = dx**2
Nx = int(abs(xmax-xmin)/dx)
Nt = int(T/dt)
I = int(Nx)+1 #space
K = int(Nt)+1 #time
x = np.arange(xmin,xmax,dx)
t = np.arange(0,T,dt)
def index(i,k): 
	return int(i+(I)*k)

#INITIAL CONDITION
m0 = iF.initial_distribution(x)
for i in range (0,x.size):
	if x[i]>1 or x[i]<0:
		m0[i] = 0
m0 = m0/(sum(m0)*dx)

v = np.zeros((I*K)) #potential
m = np.empty((I*K)) #distribution
v_old = np.empty((I*K))
m_old = np.empty((I*K))
v_grad = np.empty((I))
#initial guess on the distribution
for k in range (0,K):
	m[k*I:k*I+I] = np.copy(m0)

#initialise vectors to store l1, l2 and linfty norm errors/improvement in iterations
vl1 = -1*np.ones((Niter,1))
vl2 = -1*np.ones((Niter,1))
vlinfty = -1*np.ones((Niter,1))
ml1 = -1*np.ones((Niter,1))
ml2 = -1*np.ones((Niter,1))
mlinfty = -1*np.ones((Niter,1))
#used to search for minimum of tau
N = 20 #searchpoints
min_tol = tolerance#1e-5 #tolerance for minimum
min_left = -R/2 #search region left
min_right = R/2 #search region right
relation = 2
scatters = int(np.ceil(np.log((min_right-min_left)/min_tol)/np.log(N)))
scatters2 = int(1 + np.ceil(np.log((min_right-min_left)/(N*min_tol))/np.log(relation)))
print scatters
xpts_search = np.linspace(min_left,min_right,N)
fpts = np.empty((xpts_search.size,1))
#set final value on v and copy the bitches
m_old = np.copy(m)
v_old = np.copy(v)
print "Initialisation done, crunching..."
time_total = time.time()
for n in range (0,Niter):
	titer = time.time()
	#compute next iteration of v given m_old
	print "Computing iteration", n+1, "of v..."
	temptime = time.time()
	v[(I*K-I):(I*K)] = iF.G(x,m[(I*K-I):(I*K)]) 
	for k in range (K-2,-1,-1):
		v_tmp = np.copy(v[((k+1)*I):((k+1)*I+I)])
		m_tmp = np.copy(m[((k)*I):((k)*I+I)])
		F_var = iF.F_global(x,m_tmp,sigma,k*dt)
		#print max(F_var)
		for i in range (0,I):
			if second_order==0:
				#find starting point
				if i==0:
					fpts = iF.tau_first_order(xpts_search,i,v_tmp,x,dt)
					x0 = xpts_search[np.argmin(fpts)]
				#NUMPY
				#tmp = minimize(iF.tau_first_order,x0,args=(i,v_tmp,x,dt),tol=min_tol)
				#v[index(i,k)] = dt*F_var[i] + tmp.fun
				#MY OWN WOLF
				tmp,tmpval = iF.scatter_search(iF.tau_first_order,(i,v_tmp,x,dt),xpts_search[2]-xpts_search[1],x0,N,scatters) 
				v[index(i,k)] = dt*F_var[i] + tmpval
				x0 = tmp
			else:
				tmp = iF.find_minimum(iF.tau_second_order,(i,v_tmp,x,dt,noise)) 
				v[index(i,k)] = dt*F_var[i] + tmp
	print "Spent time", time.time()-temptime
	vchange = np.copy(v-v_old)
	vl1[n] = np.sum(abs(vchange))
	vl2[n] = np.sqrt(np.sum(abs(vchange)**2))
	vlinfty[n] = max(abs( vchange) )

	#initial condition on m is already set, compute the rest of them
	print "Computing iteration", n+1, "of m..."
	temptime = time.time()
	for k in range(0,K-1):
		if molly==1:
			v_grad = iF.mollify_array(np.gradient(v[(I*k):(I*k+I)],dx),epsilon,x,gll_x,gll_w)
			#v_grad = np.minimum(v_grad,np.zeros(v_grad.size)) #isolation game
		else:
			v_grad = np.gradient(v[(I*k):(I*k+I)],dx)
		if second_order==0:
			xtraj = iF.restrain(x-dt*v_grad,x)
			#xtraj = iF.restrain4isolation(x-dt*v_grad,x) #for the isolation modelling game
		else:
			xtraj1 = iF.restrain(x-dt*v_grad+np.sqrt(dt)*noise,x)
			xtraj2 = iF.restrain(x-dt*v_grad-np.sqrt(dt)*noise,x)
		m_update = np.zeros(v_grad.size)
		for i in range (1,I-1):
			if second_order==0:
				refindex = np.floor((xtraj[i]-xmin)/dx)
				if (refindex > -1) and (refindex < x.size-1):
					if xtraj[i] <= xmin: #out of bounds to the left; THERE'S NO LEAVING THIS GAME
						#m_update[1] += iF.beta_left(xtraj[i],x,dx,refindex)*m[index(i,k)]
						m_update[1] += m[index(i,k)]
					elif xtraj[i] >= xmax: #out of bounds to the right; THERE'S NO LEAVING THIS GAME
						#m_update[-1] += iF.beta_right(xtraj[i],x,dx,refindex)*m[index(i,k)]
						m_update[-1] += m[index(i,k)]
					else:
						m_update[refindex] += iF.beta_left(xtraj[i],x,dx,refindex)*m[index(i,k)]
						m_update[refindex+1] += iF.beta_right(xtraj[i],x,dx,refindex)*m[index(i,k)]
			else:
				refindex1 = np.floor((xtraj1[i]-xmin)/dx)
				refindex2 = np.floor((xtraj2[i]-xmin)/dx)
				if xtraj1[i] < xmin: #out of bounds to the left
					m_update[1] += 0.5*iF.beta_left(xtraj1[i],x,dx,refindex1)*m[index(i,k)]
				elif xtraj1[i] > xmax: #out of bounds to the right
					m_update[I-1] += 0.5*iF.beta_right(xtraj1[i],x,dx,refindex1)*m[index(i,k)]
				else:
					m_update[refindex1] += 0.5*iF.beta_left(xtraj1[i],x,dx,refindex1)*m[index(i,k)]
					m_update[refindex1+1] += 0.5*iF.beta_right(xtraj1[i],x,dx,refindex1)*m[index(i,k)]
				if xtraj2[i] < xmin: #out of bounds to the left
					m_update[1] += 0.5*iF.beta_left(xtraj2[i],x,dx,refindex2)*m[index(i,k)]
				elif xtraj2[i] > xmax: #out of bounds to the right
					m_update[I-1] += 0.5*iF.beta_right(xtraj2[i],x,dx,refindex2)*m[index(i,k)]
				else:
					m_update[refindex2] += 0.5*iF.beta_left(xtraj2[i],x,dx,refindex2)*m[index(i,k)]
					m_update[refindex2+1] += 0.5*iF.beta_right(xtraj2[i],x,dx,refindex2)*m[index(i,k)]
		m[I*(k+1):(I+I*(k+1))] = np.copy(m_update)
	#########################################
	################ WRAP UP ITERATION ######
	#########################################
	print "Spent time", time.time()-temptime
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
	v_old = np.copy(v)
	print "Iteration number", n+1, "completed, used time", time.time()-titer, "with change in m", mlinfty[n], "and change in v", vlinfty[n]
	
print "Time spent:", time.time()-time_total
kMax = n

msoln = np.empty((I,K))
vsoln = np.empty((I,K))
gradsoln1 = np.empty((I*K))
gradsoln = np.empty((I,K))
mollgrad1 = np.empty((I*K))
mollgrad = np.empty((K,I))
molldist1 = np.empty((I*K))
molldist = np.empty((K,I))
for k in range (0,K):
	gradsoln1[(I*k):(I*k+I)] = np.gradient(v[(I*k):(I*k+I)],dx)
	mollgrad1[(I*k):(I*k+I)] = iF.mollify_array(np.gradient(v[(I*k):(I*k+I)],dx),epsilon,x,gll_x,gll_w)
	molldist1[(I*k):(I*k+I)] = iF.mollify_array(m[(I*k):(I*k+I)],sigma,x,gll_x,gll_w)
for i in range (0,I):
	for k in range (0,K):
		msoln[i,k] = m[index(i,k)]
		vsoln[i,k] = v[index(i,k)]
		gradsoln[i,k] = gradsoln1[index(i,k)]
		mollgrad[k,i] = mollgrad1[index(i,k)]
		molldist[k,i] = molldist1[index(i,k)]
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
#print Xplot.shape,Tplot.shape,msoln.shape,vsoln.shape,gradsoln.shape,mollgrad.shape
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
#plot mollified gradient
fig6 = plt.figure(6)
ax6 = fig6.add_subplot(111, projection='3d')
ax6.plot_surface(Xplot,Tplot,mollgrad,rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax6.set_xlabel('x')
ax6.set_ylabel('t')
ax6.set_zlabel('mollified grad v(x,t)')
fig6.suptitle('Solution of mollified gradient of v(x,t)', fontsize=14)
#plot mollified distribution
fig7 = plt.figure(7)
ax7 = fig7.add_subplot(111, projection='3d')
ax7.plot_surface(Xplot,Tplot,molldist,rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax7.set_xlabel('x')
ax7.set_ylabel('t')
ax7.set_zlabel('mollified m(x,t)')
fig7.suptitle('Solution of mollified m(x,t)', fontsize=14)
#plot initial and terminal distribution in the same plot
fig8 = plt.figure(8)
plt.plot(x,m0,color="blue",label="Initial distribution")
plt.plot(x,m[(I*K-I):(I*K)],color="red",label="Terminal distribution")
legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')
ax8 = fig8.add_subplot(111)
ax8.set_xlabel('x')
ax8.set_ylabel('m(x)')
fig8.suptitle('Initial and terminal distributions', fontsize=14)
##########PLOT
plt.show()




