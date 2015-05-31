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
import scipy.interpolate as intpol
import matrix_gen as mg

#INPUTS
NONLINEAR = True
POINTSx = 5 #points in x-direction
POINTSy = 5 #points in y-direction
REFINEMENTS = 5
X_NAUGHT = 0.0
DT = .8#ratio as in dt = DT*dx(**2)
NICE_DIFFUSION = 1
n = 2 #must be integer greater than 0
cutoff = 0 #for convergence slope
xmax = 1
xmin = 0
ymax = 1
ymin = 0
T = 1
#set dx
dx0 = abs(xmax-xmin)/POINTSx
dy0 = abs(ymax-ymin)/POINTSy
dx = dx0
dy = dy0
#def exact_solution(x,y,t):
#	x,y=np.meshgrid(x,y)
#	return np.exp(-t)*np.cos(x)*np.cos(y)
dexes = np.zeros(REFINEMENTS)

x_solns = [None]*(REFINEMENTS)
y_solns = [None]*(REFINEMENTS)
m_solns = [None]*(REFINEMENTS)
m_solns_dims = [None]*REFINEMENTS

Mass_devs = np.zeros(REFINEMENTS)

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
	print "(" , dx , "," , dy , "," , dt , ")"
	Nx = int(abs(xmax-xmin)/dx)+1
	Ny = int(abs(ymax-ymin)/dy)+1
	Nt = int(np.ceil(T/dt))
	x = np.linspace(xmin,xmax,Nx)
	y = np.linspace(ymin,ymax,Ny)
	t = np.linspace(0,T,Nt)
	I = x.size #space
	J = y.size
	m = iF.initial_distribution(x,y)
	m = m/(sum(sum(m))*dx*dy)#normalise
	m_solns_dims[N] = (I,J)
	#print "Initial mass:",np.sum(m/sum(sum(m)))
	#print ss
	#SOLVE STUFF
	t0 = time.time()
	print Nt
	Mass = np.zeros(Nt)
	for k in range(1,Nt): #COMPUTE M WHERE WE DO NOT ALLOW AGENTS TO LEAVE SUCH THAT m(-1) = m(N+1) = 1 ALWAYS
		t1_tmp = time.time()
		if k==1:
			LHS = mg.FP_diffusion_implicit_Ometh(k*dt,x,y,x,y,dx,dt)
			#LHS = mg.FP_diffusion_flux_Diamond(time,x,y,x,y,dx,dt)
			#print LHS
			#LHS = mg.FP_diffusion_Nonlinear(k*dt,x,y,x,y,dx,dt,m)
			#print LHS.sum(0)
			RHS = mg.FP_convection_explicit_interpol(k*dt,x,y,x,y,dx,dt)
		#RHS = mg.FP_convection_explicit_interpol(k*dt,x,y,x,y,dx,dt)
		#RHS = mg.FP_convection_explicit_classic(k*dt,x,y,x,y,dx,dt)
		#print RHS.sum(1)
		#print ss
		t1 += time.time() - t1_tmp
		t2_tmp = time.time()
		if NONLINEAR:
			m_old = np.ravel(np.copy(m))
			while True:
				m_tmp = np.ravel(np.copy(m))
				#LHS = mg.FP_diffusion_implicit_Nonlinear(k*dt,x,y,x,y,dx,dt,m)
				LHS = mg.FP_diffusion_Nonlinear(k*dt,x,y,x,y,dx,dt,m)
				#print LHS
				#print LHS.sum(0)
				#print ss
				m = sparse.linalg.spsolve(LHS,RHS*np.ravel(m_old))
				if sum(abs(m-m_tmp))*dx < 1e-6:
					print "Picard successful", k,"/",Nt
					#print LHS
					#print ss
					break
				else:
					print sum(abs(m-m_tmp))*dx
		else:
			#print LHS.sum(1)
			#print LHS
			#print ss
			m = sparse.linalg.spsolve(LHS,RHS*np.ravel(m))
			#print sum(np.ravel(m))
			#m = RHS*np.ravel(m)
			#print sum(m)
			#print ss
		Mass[k] = sum(m)*dx*dy-1
		t2 += time.time()-t2_tmp
	t0 = time.time()-t0
	Mass = Mass/np.finfo(float).eps
	Mass_devs[N] = sum(Mass)
	print "Time spent:",t0
	print "\tGenerating matrices:",t1/t0*100
	print"\tSolving linear systems:",t2/t0*100
	#m_exact = np.ravel(exact_solution(x,y,T))
	x_solns[N] = x
	y_solns[N] = y
	m_solns[N] = m


m_interpolates = np.zeros((REFINEMENTS-1,x.size*y.size))

for i in range(REFINEMENTS-1):
	tmp = intpol.RectBivariateSpline(x_solns[i], y_solns[i], np.reshape(m_solns[i],m_solns_dims[i]), bbox=[xmin, xmax, ymin, ymax], kx=1, ky=1, s=0)
	#tmp = intpol.RectBivariateSpline(x_solns[-1], y_solns[-1], np.reshape(m_solns[-1],m_solns_dims[-1]), bbox=[xmin, xmax, ymin, ymax], kx=1, ky=1, s=0)
	#x,y = np.meshgrid(x_solns[-1],y_solns[-1])
	#print x_solns[-1].shape
	#m_interpolates[i,:] = tmp.ev(x_solns[-1],y_solns[-1])
	m_interpolates[i,:] = np.ravel(tmp(x_solns[-1],y_solns[-1]))
	#m_interpolates[i,:] = tmp.ev(x_solns[i],y_solns[i])
	#m_interpolates[i,:] = intpol.interp2d(x,y,np.reshape(m_solns[i],(I,I)))

#tmp.ev(x_solns[-1],y_solns[-1])
#print m_interpolates[0,:].shape
#print m_interpolates[1,:].shape
#print m.shape
#print ss

e1 = np.zeros(REFINEMENTS-1)
e2 = np.zeros(REFINEMENTS-1)
einf = np.zeros(REFINEMENTS-1)

for i in range(REFINEMENTS-1):
	#e1[i] = np.linalg.norm(m_interpolates[i,:]-m_solns[-1],ord=1)*dexes[i]#**2
	#e2[i] = np.linalg.norm(m_interpolates[i,:]-m_solns[-1],ord=2)*np.sqrt(dexes[i])#**2
	#einf[i] = np.linalg.norm(m_interpolates[i,:]-m_solns[-1],ord=np.inf)
	e1[i] = sum(abs(m_interpolates[i,:]-m_solns[-1]))#*dexes[i]
	e2[i] = np.sqrt(sum(abs(m_interpolates[i,:]-m_solns[-1])**2))#*np.sqrt(dexes[i])
	einf[i] = max(abs(m_interpolates[i,:]-m_solns[-1]))

#print u-u_exact
#crunch the slopes and put in the figures
Xplot, Yplot = np.meshgrid(x,y)
if REFINEMENTS>1:
	slope1, intercept = np.polyfit(np.log(dexes[cutoff:-1]), np.log(e1[cutoff:]), 1)
	slope1_1, intercept = np.polyfit(np.log(dexes[cutoff:-1]), np.log(e2[cutoff:]), 1)
	slope1_inf, intercept = np.polyfit(np.log(dexes[cutoff:-1]), np.log(einf[cutoff:]), 1)
	fig1 = plt.figure(1)
	str1 = "1-norm slope:", "%.2f" %slope1
	str2 = "2-norm slope:", "%.2f" %slope1_1
	str3 = "inf-norm slope:", "%.2f" %slope1_inf
	plt.loglog(dexes[:-1],e1,'o-',label=str1)
	plt.loglog(dexes[:-1],e2,'o-',label=str2)
	plt.loglog(dexes[:-1],einf,'o-',label=str3)
	legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')
	ax1 = fig1.add_subplot(111)
	ax1.set_xlabel('Log10 of dx')
	ax1.set_ylabel('Log10 of error')
	plt.grid(True,which="both",ls="-")
	ax1.invert_xaxis()
	fig1.suptitle('Convergence rates of u(x,t)', fontsize=14)

m0 = iF.initial_distribution(x,y)
m0 = m0/sum(sum(m0)*dx*dy)
print Xplot.shape,Yplot.shape,m.shape
fig2 = plt.figure(2)
ax1 = fig2.add_subplot(111, projection='3d')
ax1.plot_surface(Xplot,Yplot,np.reshape(m,(I,J)),rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('m(x,t)')
m0_max = float(m0.max())
m0_min = float(m0.min())
m_max = float(m.max())
m_min = float(m.min())
textstr = "Initial maximum: %.5f\nFinal maximum: %.5f\nInitial minimum: %.5f\nFinal minimum: %.5f"%(m0_max,m_max,m0_min,m_min)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#fig2.hold(True)
#ax = fig2.add_subplot(111)
ax1.text2D(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=12, verticalalignment='top', bbox=props)
fig2.suptitle('Computed solution', fontsize=14)
#fig2.hold(False)

fig2 = plt.figure(3)
ax1 = fig2.add_subplot(111, projection='3d')
ax1.plot_surface(Xplot,Yplot,m0,rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('m(x,t)')
fig2.suptitle('Initial solution', fontsize=14)

#fig22 = plt.figure(4)
#ax1 = fig22.add_subplot(111, projection='3d')
#ax1.plot_surface(Xplot,Yplot,m_exact,rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=0, antialiased=False)
#ax1.set_xlabel('x')
#ax1.set_ylabel('y')
#ax1.set_zlabel('u(x,t)')
#fig22.suptitle('Exact solution', fontsize=14)


#fig2 = plt.figure(3)
#plt.plot(x,u,label="Centered FD")
#plt.plot(x,u_exact,label="Exact solution")
#fig2.suptitle('Solutions', fontsize=14)
#legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')

fig4 = plt.figure(4)
plt.plot(Mass_devs)
fig4.suptitle('Total mass deviations', fontsize=14)

fig5 = plt.figure(5)
plt.plot(Mass)
fig5.suptitle('Mass deviation over time for last solution', fontsize=14)

fig6 = plt.figure(6)
ax1 = fig6.add_subplot(111, projection='3d')
ax1.plot_surface(Xplot,Yplot,abs(np.minimum(np.reshape(m,(I,J)),np.zeros((I,J)))),rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('m(x,t)')


plt.show()






