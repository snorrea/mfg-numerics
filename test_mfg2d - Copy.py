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
import scipy.interpolate as intpol
import applications as app

#INPUTS
NONLINEAR = False
NICE_DIFFUSION = 1
POINTSx = 10 #points in x-direction
POINTSy = 10 #points in y-direction
REFINEMENTS = 2
DT = .2#ratio as in dt = DT*dx(**2
cutoff = 0 #for convergence slope
xmax = 1
xmin = 0
ymax = 1
ymin = 0
T = 1

#OBSTACLES
obstacle_x_min = [1.5]
obstacle_x_max = [2]
obstacle_y_min = [1.5]
obstacle_y_max = [2]

#Crunching
dx0 = abs(xmax-xmin)/POINTSx
dy0 = abs(ymax-ymin)/POINTSy
dx = dx0
dy = dy0
dexes = np.zeros(REFINEMENTS)

x_solns = [None]*(REFINEMENTS)
y_solns = [None]*(REFINEMENTS)
m_solns = [None]*(REFINEMENTS)
u_solns = [None]*(REFINEMENTS)
m_solns_dims = [None]*REFINEMENTS
Mass_devs = np.zeros(REFINEMENTS)

for N in range(REFINEMENTS):
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
	u = iF.G(x,y,m)
	
	#add obstacles
	south = range(0,I)
	west = range(0,I*J,I)
	north = range(I*J-I,I*J)
	east = range(I-1,I*J,I)
	nulled = None
	#for i in range(len(obstacle_x_min)):
	#	south,north,west,east,nulled = app.add_obstacle(x,y,(obstacle_x_min[i],obstacle_x_max[i],obstacle_y_min[i],obstacle_x_max[i]),south,north,west,east,nulled)

	Mass = np.zeros(Nt)
	t0 = time.time()
	a1_arr = [None]*(Nt)
	a2_arr = [None]*(Nt)
	for k in range(Nt-1,-1,-1): #COMPUTE M WHERE WE DO NOT ALLOW AGENTS TO LEAVE SUCH THAT m(-1) = m(N+1) = 1 ALWAYS
		ush = np.reshape(u,(I,J))
		a1,a2 = np.gradient(ush,dx,dy) #this ought to doit
		a1 = -np.transpose(a1)
		a2 = -np.transpose(a2)
		a1_arr[k] = np.copy(a1)
		a2_arr[k] = np.copy(a2)
		t1_tmp = time.time()
		if NICE_DIFFUSION==0:
			u = solve.hjb_kushner_mod(k*dt,x,y,a1,a2,m_tmp,dx,dy,dt)
		else:
			if k==Nt-1:
				LHS_HJB = mg.HJB_diffusion_implicit(k*dt,x,y,a1,a2,dx,dy,dt,south,north,west,east,nulled)
			RHS_HJB = mg.HJB_convection_explicit(k*dt,x,y,a1,a2,dx,dy,dt,south,north,west,east,nulled)
			t1 += time.time() - t1_tmp
			Ltmp = np.ravel(iF.L_global(k*dt,x,y,a1,a2,u))
			t2_tmp = time.time()
			if nulled!=None:
				u[nulled] = 0
				Ltmp[nulled] = 0
			u = sparse.linalg.spsolve(LHS_HJB,RHS_HJB*np.ravel(u)+dt*np.ravel(Ltmp))
			t2 += time.time()-t2_tmp
	t0 = time.time()-t0
	print "Time spent:",t0
	print "\tGenerating matrices:",t1/t0*100
	print "\tSolving linear systems:",t2/t0*100
	t0 = time.time()
	t1 = 0
	t2 = 0
	for k in range(1,Nt): #COMPUTE M WHERE WE DO NOT ALLOW AGENTS TO LEAVE SUCH THAT m(-1) = m(N+1) = 1 ALWAYS
		t1_tmp = time.time()
		a1 = a1_arr[k-1]
		a2 = a2_arr[k-1]
		if k==1 and NICE_DIFFUSION==1:
			#LHS = mg.FP_diffusion_implicit_Ometh(k*dt,x,y,a1,a2,dx,dt,south,north,west,east,nulled)
			zab = 2
			LHS = mg.FP_diffusion_flux_Diamond(time,x,y,x,y,dx,dt)
			#LHS = mg.FP_diffusion_Nonlinear(k*dt,x,y,x,y,dx,dt,m)
		RHS = mg.FP_convection_explicit_interpol(k*dt,x,y,a1,a2,dx,dt,south,north,west,east,nulled)
		t1 += time.time() - t1_tmp
		t2_tmp = time.time()
		if NONLINEAR:
			m_old = np.ravel(np.copy(m))
			while True:
				m_tmp = np.ravel(np.copy(m))
				LHS = mg.FP_diffusion_Nonlinear(k*dt,x,y,a1,a2,dx,dt,m)
				m = sparse.linalg.spsolve(LHS,RHS*np.ravel(m_old))
				if sum(abs(m-m_tmp))*dx < 1e-6:
					print "Picard successful", k,"/",Nt
					break
				else:
					print sum(abs(m-m_tmp))*dx
		else:
			if nulled!=None:
				m = np.ravel(m)
				m[nulled] = 0
			m = sparse.linalg.spsolve(LHS,RHS*np.ravel(m))
			#m = RHS*np.ravel(m)
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
	u_solns[N] = u


m_interpolates = np.zeros((REFINEMENTS-1,x.size*y.size))
u_interpolates = np.zeros((REFINEMENTS-1,x.size*y.size))

for i in range(REFINEMENTS-1):
	tmp = intpol.RectBivariateSpline(x_solns[i], y_solns[i], np.reshape(m_solns[i],m_solns_dims[i]), bbox=[xmin, xmax, ymin, ymax], kx=1, ky=1, s=0)
	m_interpolates[i,:] = np.ravel(tmp(x_solns[-1],y_solns[-1]))
	tmp = intpol.RectBivariateSpline(x_solns[i], y_solns[i], np.reshape(u_solns[i],m_solns_dims[i]), bbox=[xmin, xmax, ymin, ymax], kx=1, ky=1, s=0)
	u_interpolates[i,:] = np.ravel(tmp(x_solns[-1],y_solns[-1]))

e1_u = np.zeros(REFINEMENTS-1)
e2_u = np.zeros(REFINEMENTS-1)
einf_u = np.zeros(REFINEMENTS-1)
e1_m = np.zeros(REFINEMENTS-1)
e2_m = np.zeros(REFINEMENTS-1)
einf_m = np.zeros(REFINEMENTS-1)

for i in range(REFINEMENTS-1):
	e1_u[i] = sum(abs(u_interpolates[i,:]-u_solns[-1]))#*dexes[i]
	e2_u[i] = np.sqrt(sum(abs(u_interpolates[i,:]-u_solns[-1])**2))#*np.sqrt(dexes[i])
	einf_u[i] = max(abs(u_interpolates[i,:]-u_solns[-1]))
	e1_m[i] = sum(abs(m_interpolates[i,:]-m_solns[-1]))#*dexes[i]
	e2_m[i] = np.sqrt(sum(abs(m_interpolates[i,:]-m_solns[-1])**2))#*np.sqrt(dexes[i])
	einf_m[i] = max(abs(m_interpolates[i,:]-m_solns[-1]))

#print u-u_exact
#crunch the slopes and put in the figures
Xplot, Yplot = np.meshgrid(x,y)
if REFINEMENTS>1:
	slope1, intercept = np.polyfit(np.log(dexes[cutoff:-1]), np.log(e1_m[cutoff:]), 1)
	slope1_1, intercept = np.polyfit(np.log(dexes[cutoff:-1]), np.log(e2_m[cutoff:]), 1)
	slope1_inf, intercept = np.polyfit(np.log(dexes[cutoff:-1]), np.log(einf_m[cutoff:]), 1)
	fig1 = plt.figure(1)
	str1 = "1-norm slope:", "%.2f" %slope1
	str2 = "2-norm slope:", "%.2f" %slope1_1
	str3 = "inf-norm slope:", "%.2f" %slope1_inf
	plt.loglog(dexes[:-1],e1_m,'o-',label=str1)
	plt.loglog(dexes[:-1],e2_m,'o-',label=str2)
	plt.loglog(dexes[:-1],einf_m,'o-',label=str3)
	legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')
	ax1 = fig1.add_subplot(111)
	ax1.set_xlabel('Log10 of dx')
	ax1.set_ylabel('Log10 of error')
	plt.grid(True,which="both",ls="-")
	ax1.invert_xaxis()
	fig1.suptitle('Convergence rates of m(x,t)', fontsize=14)

	slope1, intercept = np.polyfit(np.log(dexes[cutoff:-1]), np.log(e1_u[cutoff:]), 1)
	slope1_1, intercept = np.polyfit(np.log(dexes[cutoff:-1]), np.log(e2_u[cutoff:]), 1)
	slope1_inf, intercept = np.polyfit(np.log(dexes[cutoff:-1]), np.log(einf_u[cutoff:]), 1)
	fig2 = plt.figure(2)
	str1 = "1-norm slope:", "%.2f" %slope1
	str2 = "2-norm slope:", "%.2f" %slope1_1
	str3 = "inf-norm slope:", "%.2f" %slope1_inf
	plt.loglog(dexes[:-1],e1_u,'o-',label=str1)
	plt.loglog(dexes[:-1],e2_u,'o-',label=str2)
	plt.loglog(dexes[:-1],einf_u,'o-',label=str3)
	legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')
	ax1 = fig2.add_subplot(111)
	ax1.set_xlabel('Log10 of dx')
	ax1.set_ylabel('Log10 of error')
	plt.grid(True,which="both",ls="-")
	ax1.invert_xaxis()
	fig2.suptitle('Convergence rates of u(x,t)', fontsize=14)

m0 = iF.initial_distribution(x,y)
m0 = m0/sum(sum(m0)*dx*dy)
print Xplot.shape,Yplot.shape,m.shape
fig3 = plt.figure(3)
ax1 = fig3.add_subplot(111, projection='3d')
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
fig3.suptitle('Computed solution', fontsize=14)
#fig2.hold(False)

fig2 = plt.figure(4)
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

fig4 = plt.figure(5)
plt.plot(Mass_devs)
fig4.suptitle('Total mass deviations', fontsize=14)

fig5 = plt.figure(6)
plt.plot(Mass)
fig5.suptitle('Mass deviation over time for last solution', fontsize=14)

fig6 = plt.figure(7)
ax1 = fig6.add_subplot(111, projection='3d')
ax1.plot_surface(Xplot,Yplot,abs(np.minimum(np.reshape(m,(I,J)),np.zeros((I,J)))),rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('m(x,t)')
fig6.suptitle("Negative values in solution of m")

fig7 = plt.figure(8)
ax1 = fig7.add_subplot(111, projection='3d')
ax1.plot_surface(Xplot,Yplot,np.reshape(u,(J,I)),rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('u(x,t)')
fig2.suptitle('Computed solution of u', fontsize=14)

plt.show()



