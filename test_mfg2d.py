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
scatter_test = 5
NONLINEAR = False
PLOT_DEBUG = False
NICE_DIFFUSION = 1
POINTSx = 5*2*2 #points in x-direction
POINTSy = 5*2*2 #points in y-direction
REFINEMENTS = 1
NITERATIONS = 1
DT = .1#ratio as in dt = DT*dx(**2)
cutoff = 0 #for convergence slope
grid = [0, 1, 0, 1]
xmax = grid[1]
xmin = grid[0]
ymax = grid[3]
ymin = grid[2]
T = 1.5
PLOT_GRID = [3, 4]

#OBSTACLES
obstacle_x_min = [.0, .6]#[.0]#
obstacle_x_max = [.4, 1.]#[.05]#
obstacle_y_min = [.4, .4]#[.0]#
obstacle_y_max = [.6, .6]#[.05]#
KILL = 2 #set 0 if you don't want this to matter

#GOALS
#goals_x = [.0,.1,.2,.8,.9,1.]#[.2,.4,.6,.8]
#goals_y = [.9,.9,.9,.9,.9,.9]#[.9,.9,.9,.9]
goals_x = np.linspace(.0,.2,50)
goals_x = np.concatenate([goals_x,np.linspace(.8,1.,50)])
goals_y = np.linspace(.9,1.,100)
#goals_x,goals_y = np.meshgrid(goals_x,goals_y)
goals_x = np.unique(np.ravel(goals_x))
goals_y = np.unique(np.ravel(goals_y))

print goals_x.size
print goals_y.size
	

#Crunching
dx0 = abs(xmax-xmin)/POINTSx
dy0 = abs(ymax-ymin)/POINTSy
dx = dx0
dy = dy0
dexes = np.zeros(REFINEMENTS)
TIME_PLOTS = PLOT_GRID[0]*PLOT_GRID[1]

x_solns = [None]*(REFINEMENTS)
y_solns = [None]*(REFINEMENTS)
m_solns = [None]*(REFINEMENTS)
u_solns = [None]*(REFINEMENTS)
m_solns_dims = [None]*REFINEMENTS
Mass_devs = np.zeros(REFINEMENTS)
conv_plot_m = np.zeros(NITERATIONS)

#Searching stuff
min_tol = 1e-6#tolerance#1e-5 #tolerance for minimum
alpha_upper = np.array([1, 1]) #search region left
alpha_lower = np.array([-1, -1]) #search region right

def opt_cont_cmfg(u):
	ush = np.transpose(np.reshape(u,(I,J)))
	a1,a2 = np.gradient(ush,dx,dy) #this ought to doit
	a1 = -np.transpose(a1)
	a2 = -np.transpose(a2)
	#a1 = -(a1)
	#a2 = -(a2)
	return a1,a2		

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
	m0 = iF.initial_distribution(x,y)
	m0 = m0/(sum(sum(m0))*dx*dy)#normalise
	m_solns_dims[N] = (I,J)
	u = np.zeros(I*J)
	X,Y = np.meshgrid(x,y)
	
	indices = range(0,I*J)
	south = range(0,I) #ok
	west = range(0,I*J,I) #ok
	north = range(I*J-I,I*J)
	east = range(I-1,I*J+1,I)
	nulled = None
	
	#print south
	#print west
	#print north
	#print east
	new_south = []
	new_north = []
	new_east = []
	new_west = []
	temp_east = [None]*2
	temp_west = [None]*2
	
	for i in range(len(obstacle_x_min)):
		#new_south,new_north,new_west,new_east,nulled,se,sw,ne,nw = app.add_obstacle(x,y,(obstacle_x_min[i],obstacle_x_max[i],obstacle_y_min[i],obstacle_y_max[i]),south,north,west,east,nulled)
		new_south,new_north,new_west,new_east,nulled,se,sw,ne,nw = app.add_obstacle(x,y,(obstacle_x_min[i],obstacle_x_max[i],obstacle_y_min[i],obstacle_y_max[i]),[],[],[],[],nulled)
		south = np.concatenate([south,new_south])
		north = np.concatenate([north,new_north])
		if i==0:
			west = np.concatenate([west,new_west])
			nulled = np.concatenate([nulled,new_east])
		else:
			east = np.concatenate([east,new_east])
			nulled = np.concatenate([nulled,new_west])
	south.sort()
	north.sort()
	east.sort()
	west.sort()
	if nulled is not None:
		nulled.sort()
	
#	indices = np.array([n for n in list(range(I*J)) if n not in list(nulled)])
#	not_south = np.array([n for n in indices if n not in list(south)])
#	not_north = np.array([n for n in indices if n not in list(north)])
#	not_east = np.array([n for n in indices if n not in list(east)])
#	not_west = np.array([n for n in indices if n not in list(west)])
#	print indices
#	print north
#	print not_north
#	print ss

	obstacle = np.copy(nulled)
	#obstacle = np.concatenate([obstacle,new_east])
	#obstacle = np.concatenate([obstacle,new_west])
	#obstacle = np.concatenate([obstacle,new_north])
	#obstacle = np.concatenate([obstacle,new_south])

	#south = np.concatenate([south,new_south])
	#north = np.concatenate([north,new_north])
	#east = np.concatenate([east,new_east])
	#west = np.concatenate([west,new_west])
	#obstacle = np.concatenate([obstacle,west])
	#obstacle = np.concatenate([obstacle,east])
	#obstacle = np.concatenate([obstacle,north])
	#obstacle = np.concatenate([obstacle,south])
	#print obstacle
	
	not_nulled = [ item for i,item in enumerate(indices) if i not in nulled ]
	not_obstacle = [ item for i,item in enumerate(indices) if i not in obstacle ]
	ObstacleCourse = app.FMM(x,y,obstacle,goals_x,goals_y)
	ObstacleCourse = np.reshape(ObstacleCourse,(J,I))
	fig1 = plt.figure(1)
	#ax1 = fig1.add_subplot(111, projection='3d')
	#ax1.plot_surface(X,Y,Obs,rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=0, antialiased=False)
	#levels = np.arange(np.amin(Obs),np.amax(Obs),0.5)
	levels = np.arange(0,2,0.05)
	norm = cm.colors.Normalize(vmax=abs(ObstacleCourse).max(), vmin=0)
	cmap = cm.PRGn
	#print X.shape,Y.shape,ObstacleCourse.shape
	fig1 = plt.contourf(X,Y,ObstacleCourse,levels,cmap=plt.cm.Reds)
	CS2 = plt.contour(fig1, levels=levels[:],colors = 'b')
	#cbar = plt.colorbar(fig1)
	#cbar.ax.set_ylabel('Distribution density')
	#cbar.add_lines(CS2)
	#fig1.suptitle("a1")
	#ax1.set_xlabel('x')
	#ax1.set_ylabel('y')
	#plt.show()#[]#
	


	#print ss
	#print y
	#print nulled
	#print ss
	#minimisation things
	Ns_x = int(np.ceil(abs(alpha_upper[0]-alpha_lower[0])/(dx)) + 1)#*10
	Ns_y = int(np.ceil(abs(alpha_upper[1]-alpha_lower[1])/(dy)) + 1)#*10
	xpts_scatter = np.linspace(alpha_lower[0],alpha_upper[0],Ns_x)
	ypts_scatter = np.linspace(alpha_lower[1],alpha_upper[1],Ns_y)
	Ns = [Ns_x,Ns_y]
	scatters = int(np.ceil( np.log((max(alpha_upper-alpha_lower))/(min_tol*min(Ns)))/np.log(min(Ns)/2) ))
	#min_tol = .5*abs(max(alpha_upper-alpha_lower))*(2/min(Ns))**scatters
	print "Scatters:",scatters,Ns

	Mass = np.zeros(Nt)
	t0 = time.time()
	a1_arr = [None]*(Nt)
	a2_arr = [None]*(Nt)
	m_arr = [None]*(Nt)
	#stuff initial guess into m_arr
	for k in range(0,Nt):
		m_arr[k] = m0
	m_last_solution = None

	for ITERATION in range(NITERATIONS):
		#m_last_solution = np.copy(m_arr[-1])
		m_last_solution = np.copy(m_arr)
		u = np.zeros(I*J)
		t_naive = 0
		t_3d = 0
		t_4d = 0
		t_hybrid = 0
		t_hybrido = 0
		for k in range(Nt-1,-1,-1): #COMPUTE M WHERE WE DO NOT ALLOW AGENTS TO LEAVE SUCH THAT m(-1) = m(N+1) = 1 ALWAYS
			print k
			tBALLS = time.time()
			for BALLS in range(scatter_test):
				m_tmp = np.copy(m_arr[k])
			#	a1,a2 = opt_cont_cmfg(u)
			#	t0 = time.time()
			#	a1,a2 = solve.control_general(xpts_scatter,ypts_scatter,x,y,u,m_tmp,dt,dx,dy,k*dt,I,J,min_tol,scatters,Ns,nulled,np.ravel(ObstacleCourse),south,north,west,east) #must be transposed
			#	t_naive += time.time()-t0
			#	print "Naive done"
				t0 = time.time()
				a1,a2 = solve.control_general_vectorised_3D(xpts_scatter,ypts_scatter,x,y,(u),m_tmp,dt,dx,dy,k*dt,I,J,min_tol,scatters,Ns,nulled,ObstacleCourse,south,north,west,east)
				t_3d += time.time()-t0
				#print "3D done"
			#	t0 = time.time()
			#	a1,a2 = solve.control_general_vectorised_4D(xpts_scatter,ypts_scatter,x,y,(u),m_tmp,dt,dx,dy,k*dt,I,J,min_tol,scatters,Ns,nulled,ObstacleCourse,south,north,west,east)
			#	t_4d += time.time()-t0
				#print "4D done"
				t0 = time.time()
				a1,a2 = solve.control_hybridC_vectorised_3D(xpts_scatter,ypts_scatter,x,y,(u),m_tmp,dt,dx,dy,k*dt,I,J,min_tol,scatters,Ns,nulled,ObstacleCourse,south,north,west,east)
				t_hybrid += time.time()-t0
				#print "HybridC done"
				t0 = time.time()
				a1,a2 = solve.control_hybridO_vectorised_3D(xpts_scatter,ypts_scatter,x,y,(u),m_tmp,dt,dx,dy,k*dt,I,J,min_tol,scatters,Ns,nulled,ObstacleCourse,south,north,west,east)
				t_hybrido += time.time()-t0
				#print "HybridO done"
			#t0 = time.time()
			#a1_,a2_ = solve.control_general(xpts_scatter,xpts_scatter,x,y,u,m_tmp,dt,dx,dy,k*dt,I,J,min_tol,scatters,Ns,nulled,ObstacleCourse,south,north,west,east)
			#t0 = time.time()-t0
			#t1 = time.time()
			#a1,a2 = solve.control_hybrid(xpts_scatter,xpts_scatter,x,y,u,m_tmp,dt,dx,dy,k*dt,I,J,min_tol,scatters,Ns,nulled,ObstacleCourse,south,north,west,east)
			#t1 = time.time()-t1
			#print "Scatter:", t0, "\nHybrid:", t1
			a1 = np.transpose(a1)
			a2 = np.transpose(a2)
			#a1_ = np.transpose(a1_)
			#a2_ = np.transpose(a2_)
			#print np.linalg.norm(a1-a1_)
			#print np.linalg.norm(a2-a2_)
		#	print a2#==a2_
		#	print a2_#==a1_
			#print a2==np.transpose(a2_)
			#print a2_
			#print a1.shape,a2.shape
			
			a1_arr[k] = np.copy(a1)
			a2_arr[k] = np.copy(a2)
			t1_tmp = time.time()
			if NICE_DIFFUSION==0:
				u = solve.hjb_kushner_mod(k*dt,x,y,a1,a2,m_tmp,dx,dy,dt)
			else:
				if k==Nt-1:
					LHS_HJB = mg.HJB_diffusion_implicit(k*dt,x,y,a1,a2,dx,dy,dt,south,north,west,east,nulled,se,sw,ne,nw)
					#LHS_HJB  = mg.trim_nulled(LHS_HJB,nulled)
				RHS_HJB = mg.HJB_convection_explicit(k*dt,x,y,a1,a2,dx,dy,dt,south,north,west,east,nulled)
				#RHS_HJB = mg.trim_nulled(RHS_HJB,nulled)
				t1 += time.time() - t1_tmp
				Obs = np.reshape(ObstacleCourse,(I,J))
				m_tmp = np.reshape(m_tmp,(I,J))
				Ltmp = np.ravel(iF.L_global(k*dt,x,y,a1,a2,m_tmp,Obs))
				#plt.spy(RHS_HJB)
				#plt.show()
				#print ss
				t2_tmp = time.time()
				#if nulled!=None:
				#	u[nulled] = 0
				#	Ltmp[nulled] = 0
				u = sparse.linalg.spsolve(LHS_HJB,RHS_HJB*np.ravel(u)+dt*np.ravel(Ltmp))
				#if nulled!=None:
				#	u[nulled] = 10
				#plot it plz
				if PLOT_DEBUG:# and k<140:
					X,Y = np.meshgrid(x,y)
					#print u.shape
					fig1 = plt.figure(1)
					ax1 = fig1.add_subplot(111, projection='3d')
					ax1.plot_surface(X,Y,(a1),rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=0, antialiased=False)
					fig1.suptitle("a1")
					ax1.set_xlabel('x')
					ax1.set_ylabel('y')
					fig2 = plt.figure(2)
					ax2 = fig2.add_subplot(111, projection='3d')
					ax2.plot_surface(X,Y,(a2),rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=0, antialiased=False)
					ax2.set_xlabel('x')
					ax2.set_ylabel('y')
					fig2.suptitle("a2")
					fig3 = plt.figure(3)
					ax3 = fig3.add_subplot(111, projection='3d')
					ax3.plot_surface(X,Y,np.reshape(u,(J,I)),rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=0, antialiased=False)
					ax3.set_xlabel('x')
					ax3.set_ylabel('y')
					fig3.suptitle("u")	
					plt.show()
				t2 += time.time()-t2_tmp
			tBALLS = time.time()-tBALLS
			print "Estimated remaining time in minutes:", tBALLS*k*scatter_test/60
		print "Naive:", t_naive/(Nt*scatter_test)
		print "3d:", t_3d/(Nt*scatter_test)
		print "4d:", t_4d/(Nt*scatter_test)
		print "Hybrid_c:", t_hybrid/(Nt*scatter_test)
		print "Hybrid_o:", t_hybrido/(Nt*scatter_test)
	#	print ss
		t0 = time.time()-t0
		print "Time spent:",t0
		print "\tGenerating matrices:",t1/t0*100
		print "\tSolving linear systems:",t2/t0*100
		t0 = time.time()
		t1 = 0
		t2 = 0
		for k in range(1,Nt): #COMPUTE M WHERE WE DO NOT ALLOW AGENTS TO LEAVE SUCH THAT m(-1) = m(N+1) = 1 ALWAYS
			t1_tmp = time.time()
			m = np.zeros(I*J)
			m_last = np.copy(m_arr[k-1])
			a1 = a1_arr[k-1]
			a2 = a2_arr[k-1]
			if k==1 and NICE_DIFFUSION==1:
				#LHS = mg.FP_diffusion_implicit_Ometh(k*dt,x,y,a1,a2,dx,dt)
				LHS = mg.FP_diffusion_flux_Diamond(time,x,y,x,y,dx,dt,south,north,west,east,nulled)
				#LHS = mg.FP_diffusion_Nonlinear(k*dt,x,y,x,y,dx,dt,m)
				#LHS = mg.trim_nulled(LHS,obstacle)
			RHS = mg.FP_convection_explicit_interpol(k*dt,x,y,a1,a2,dx,dt,south,north,west,east,nulled)
			#plt.spy(RHS)
			#plt.show()
			#RHS = mg.trim_nulled(RHS,obstacle)
			#plt.spy(RHS)
			#plt.show()
			t1 += time.time() - t1_tmp
			t2_tmp = time.time()
			if NONLINEAR:
				m_old = np.ravel(np.copy(m_last))
				while True:
					m_tmp = np.ravel(np.copy(m_last))
					LHS = mg.FP_diffusion_Nonlinear(k*dt,x,y,a1,a2,dx,dt,m,south,north,west,east,nulled)
					m = sparse.linalg.spsolve(LHS,RHS*np.ravel(m_old))
					if sum(abs(m-m_tmp))*dx < 1e-6:
						print "Picard successful", k,"/",Nt
						break
					else:
						print sum(abs(m-m_tmp))*dx
			else:
				m_last = np.ravel(m_last)
				#if nulled!=None:
				#	m_last[nulled] = 0
				m = sparse.linalg.spsolve(LHS,RHS*np.ravel(m_last))
				#m = m/(sum((m))*dx*dy)
				#m[not_nulled] = sparse.linalg.spsolve(LHS,RHS*m_last[not_nulled])
				#m[not_obstacle] = sparse.linalg.spsolve(LHS,RHS*m_last[not_obstacle])
				#m = RHS*np.ravel(m)
			m_arr[k] = np.copy(m)
			Mass[k] = sum(m)*dx*dy-1
			t2 += time.time()-t2_tmp
		
		t0 = time.time()-t0
		#Mass = Mass/np.finfo(float).eps
		Mass_devs[N] = sum(Mass)
		print "Time spent:",t0
		print "\tGenerating matrices:",t1/t0*100
		print"\tSolving linear systems:",t2/t0*100
		#m_exact = np.ravel(exact_solution(x,y,T))
		x_solns[N] = x
		y_solns[N] = y
		m_solns[N] = m
		u_solns[N] = u
		#now compare iterations
		#temp = np.linalg.norm(m_arr[-1]-np.ravel(m_last_solution))
		temp = 0
		for i in range(Nt):
			temp += np.linalg.norm(np.ravel(m_arr[i])-np.ravel(m_last_solution[i]))
		conv_plot_m[ITERATION] = temp
		kMax = ITERATION
		if temp < 1e-3:
			print "Converged! Used iterations:", ITERATION
			break
		else:
			print "Still crunching:", temp, ITERATION, "/", NITERATIONS
	#evaluate iteration

conv_plot_m = conv_plot_m[:kMax]
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

fig8 = plt.figure(8)
SOLN_NUMBER = range(1,Nt,int(round(Nt/TIME_PLOTS)))
print SOLN_NUMBER
for tullball in range(TIME_PLOTS):
	plt_number = int(str(PLOT_GRID[0]) + str(PLOT_GRID[1]) + str(tullball))
	#ax8 = fig8.add_subplot(plt_number, projection='3d')
	ax8 = fig8.add_subplot(PLOT_GRID[0], PLOT_GRID[1], tullball, projection='3d')
	#get SOLN_NUMBER
	ax8.plot_surface(Xplot,Yplot,np.reshape(m_arr[SOLN_NUMBER[tullball]],(I,J)),rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=0, antialiased=False)

fig9 = plt.figure(9)
plt.loglog(range(kMax),conv_plot_m)

plt.show()



