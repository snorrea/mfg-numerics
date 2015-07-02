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
import applications as app
import math
import scipy.sparse as sparse
import scipy.interpolate as intpol
import glob,os,sys

#INPUTS
NICE_DIFFUSION = 0 #1 if diffusion indep of t,m,alpha
TESTNAME = "mfg1d#finaltest#damp"
LOAD_WRITE = True#True
BETTER = False
#dx = 0.1**2/2
dx = 1/40#0.01#0.025
DT = .25
dt = DT*dx
#dt = dx**2/(0.3**2 + dx*2) # dt = dx**2/(max(sigma)**2 + dx*max(f))
print dx,dt
xmin = 0#-2
xmax = 1#+.2
T = 1
Niter = 400#5000 #maximum number of iterations
tolerance = 1e-4
quad_order = 20
alpha_upper = 3
alpha_lower = -3
start_eps = dx
deps = dx*2


if LOAD_WRITE: #load best solution with parameters
	best_dx = None
	best_DT = None
	best_eps = -1
	for file in glob.glob("*.txt"):
		pop = file.split("_")
		pop_dx = float(pop[1])
		pop_DT = float(pop[2])
		pop_eps = float(pop[3])
		if abs(pop_dx-dx)<1e-6 and pop_DT==DT and pop[0]==TESTNAME:
			print "Pepe",pop_eps
			if pop_eps > best_eps:
				best_dx = pop_dx
				best_DT = pop_DT
				best_eps = pop_eps
	if best_eps is not -1 and start_eps < best_eps:
		start_eps = min(1,best_eps+deps)
		BETTER = True
		dx_string = "%.8f" % best_dx
		DT_string = "%.8f" % best_DT
		eps_string = "%.8f" % best_eps
		print "Loading previous computation result..."
		m = np.loadtxt("./" + TESTNAME + "_" + dx_string + "_" + DT_string + "_" + eps_string + "_" + ".txt")
		print "Loading successful! Plotting solution..."
		x = np.linspace(xmin,xmax,round(abs(xmax-xmin)/best_dx+1))
		t = np.linspace(0,T,round(abs(T)/(best_DT*best_dx)+1))
		Xplot,Tplot = np.meshgrid(x,t)
		print best_dx,best_DT,best_eps
		print (1/best_dx+1), (1/(1/40)+1), (1/(best_DT*best_dx)+1)
		print x.size,t.size,x.size*t.size,m.shape
		print Xplot.shape, Tplot.shape, np.reshape(m,(t.size,x.size)).shape
		fig1 = plt.figure(1)
		ax1 = fig1.add_subplot(111, projection='3d')
		ax1.plot_surface(Xplot,Tplot,np.reshape(m,(t.size,x.size)),rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=0, antialiased=False)
		ax1.set_xlabel('x')
		ax1.set_ylabel('t')
		ax1.set_zlabel('m(x,t)')
		fig1.suptitle('Solution of the density m(x,t)', fontsize=14)
		plt.show()
		print "Best eps:", best_eps
	if best_eps == 0:
		print "Nothing to do here, buddy!"
		print ss



epses = int(np.round((1-start_eps)/deps))+1
#epses = 40
epsel = np.linspace(start_eps,1,epses)**1
#epsel = np.linspace(start_eps,0,epses)
print epsel.size
print epsel
#print ss

#STUFF TO MINIMIZE
min_tol = 1e-6#0.01*dx**2#tolerance#1e-5 #tolerance for minimum
min_left = alpha_lower #search region left
min_right = alpha_upper #search region right
Ns = int(np.ceil(abs(alpha_upper-alpha_lower)/(.5*dx)) + 1)
xpts_scatter = np.linspace(alpha_lower,alpha_upper,Ns)
scatters = int(np.ceil( np.log((alpha_upper-alpha_lower)/(min_tol*Ns))/np.log(Ns/2) ))
#scatters = 4

dx_scatter = xpts_scatter[1]-xpts_scatter[0]
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
gll_x = qn.GLL_points(quad_order) #quadrature nodes
gll_w = qn.GLL_weights(quad_order,gll_x) #quadrature weights

#INITIALISE STORAGE
u = np.zeros((I*K)) #potential
a = np.zeros((I*K)) #control
u_old = np.zeros((I*K))
a_old = np.zeros((I*K))
ul1 = -1*np.ones((Niter*epses,1))
ul2 = -1*np.ones((Niter*epses,1))
ulinfty = -1*np.ones((Niter*epses,1))
ml1 = -1*np.ones((Niter*epses,1))
ml2 = -1*np.ones((Niter*epses,1))
mlinfty = -1*np.ones((Niter*epses,1))
al1 = -1*np.ones((Niter*epses,1))
al2 = -1*np.ones((Niter*epses,1))
alinfty = -1*np.ones((Niter*epses,1))
def index(i,k): 
	return int(i+(I)*k)
Xplot, Tplot = np.meshgrid(x,t)
#INITIAL/TERMINAL CONDITIONS, INITIAL GUESS and COPYING
m0 = iF.initial_distribution(x)
m0 = m0/(sum(m0)*dx) #normalise
if not BETTER:
	m = np.zeros((I*K)) #distribution
	m_old = np.zeros((I*K))
	for k in range (0,K):
		m[k*I:k*I+I] = np.copy(m0)
m_old = np.copy(m)
u_old = np.copy(u)
a_old = np.copy(a)
a_tmp = np.copy(a)
time_total = time.time()
BIGZERO = np.zeros(x.size-2)

#matrix-generation if stuff
LHS_HJB = np.zeros(I)
LHS_FP = np.zeros(I)
Success = True

Mass = np.zeros(K)

M_ends = np.empty((I,epses))
kMax = 0

#damp_thing = []*Nt

m_best = np.copy(m)

total_time2 = time.time()
for NN in range(epses):
	#damp_thing = np.linspace(1,epsel[NN],Nt)#**4
	#print damp_thing
	#print ss
	for n in range (0,Niter):
		titer = time.time()
		#print "Computing iteration",n+1,"of u..."
		temptime = time.time()
		#Compute u
		u[(I*K-I):(I*K)] = iF.G(x,m[(I*K-I):(I*K)])
		#print u[]
		for k in range (K-1,0,-1):  #this is how it has to be...
			u_last = np.copy(u[((k+1)*I-I):((k+1)*I)]) #this one to keep
			m_last = np.copy(m[((k+1)*I-I):((k+1)*I)]) #only actually need this, amirite?
			a_tmp = solve.control_general_vectorised(x,k*dt,u_last,epsel[NN]*m_last,dx,xpts_scatter,Ns,scatters)
			#a_tmp = solve.control_general_vectorised(x,k*dt,u_last,m_last,dx,xpts_scatter,Ns,scatters)
			#a_tmp = -np.gradient(u[((k+1)*I-I):((k+1)*I)],dx)
			#a_tmp = iF.opt_cmfg(u[((k+1)*I-I):((k+1)*I)],dx)
			#a_tmp = iF.mollify_array(a_tmp,epsel[NN],x,gll_x,gll_w)
			#a_tmp = np.maximum(-np.gradient(u_last,dx),np.zeros(u_last.size))
			#a_tmp = solve.control_general(x,k*dt,u_last,m_last,dt,dx,xpts_scatter,Ns,scatters) #hybrid
			if NICE_DIFFUSION==0:
				#LHS_HJB = mg.hjb_diffusion_av(k*dt,x,a_tmp,dt,dx,1-epsel[NN])
				LHS_HJB = mg.hjb_diffusion_av(k*dt,x,a_tmp,dt,dx,0)
				RHS_HJB = mg.hjb_convection(k*dt,x,a_tmp,dt,dx)
				Ltmp = iF.L_global(k*dt,x,a_tmp,m_last,epsel[NN])
				u_tmp = sparse.linalg.spsolve(LHS_HJB,RHS_HJB*u_last+dt*Ltmp)
				#u_tmp = solve.hjb_kushner_mod(x,k*dt,u_last,a_tmp,dt,dx) #implicit
			else:
				#if n==0 and k==K-1:
				#	LHS_HJB = mg.hjb_diffusion(k*dt,x,a_tmp,dt,dx)
				#	LHS_HJB = mg.hjb_diffusion_av(k*dt,x,a_tmp,dt,dx,1-epsel[NN])
				RHS_HJB = mg.hjb_convection(k*dt,x,a_tmp,dt,dx)
				Ltmp = iF.L_global(k*dt,x,a_tmp,m_last,epsel[NN])
				#Ltmp = iF.L_global(k*dt,x,a_tmp,m_last,damp_thing[k])
			#u_tmp = sparse.linalg.spsolve(LHS_HJB,RHS_HJB*u_last+dt*Ltmp)
			u_tmp = RHS_HJB*u_last+dt*Ltmp
		
			u[(k*I-I):(k*I)] = np.copy(u_tmp)
			a[(k*I-I):(k*I)] = np.copy(a_tmp)
		
		#print "Spent time", time.time()-temptime, "on computing u"
		#store changes in norms
		uchange = np.copy(u-u_old)
		ul1[kMax+n] = np.sum(abs(uchange))*dx
		ul2[kMax+n] = np.sqrt(np.sum(abs(uchange)**2))*np.sqrt(dx)
		ulinfty[kMax+n] = max(abs(uchange))
		achange = np.copy(a-a_old)
		al1[kMax+n] = np.sum(abs(achange))*dx
		al2[kMax+n] = np.sqrt(np.sum(abs(achange)**2))*np.sqrt(dx)
		alinfty[kMax+n] = max(abs(achange) )
	#GET GOING WITH M
		#print "Computing iteration", n+1, "of m..."
		m[0:I] = np.copy(m0)
		#print m0
		#print ss
		temptime = time.time()
		for k in range(0,K-1): #COMPUTE M WHERE WE DO NOT ALLOW AGENTS TO LEAVE SUCH THAT m(-1) = m(N+1) = 1 ALWAYS
			a_tmp = a[(k*I):(k*I+I)]
			m_tmp = m[(k*I):(k*I+I)]
			if NICE_DIFFUSION==0:
				#LHS_FP = mg.fp_fv_diffusion_av(k*dt,x,a_tmp,dt,dx,1-epsel[NN])
				LHS_FP = mg.fp_fv_diffusion_av(k*dt,x,a_tmp,dt,dx,0)
				RHS_FP = mg.fp_fv_convection_interpol(k*dt,x,a_tmp,dt,dx)
				m_update = sparse.linalg.spsolve(LHS_FP,RHS_FP*m_tmp)
			else:
				#if n==0 and k==0:
				#	LHS_FP = mg.fp_fv_diffusion(0,x,a_tmp,dt,dx)
				#	LHS_FP = mg.fp_fv_diffusion_av(k*dt,x,a_tmp,dt,dx,1-epsel[NN])
				#RHS_FP = mg.fp_fv_convection_classic(k*dt,x,a_tmp,dt,dx)
				RHS_FP = mg.fp_fv_convection_interpol(k*dt,x,a_tmp,dt,dx)
				#m_update = sparse.linalg.spsolve(LHS_FP,RHS_FP*m_tmp)
				m_update = RHS_FP*m_tmp
			m[I*(k+1):(I+I*(k+1))] = np.copy(m_update)
		#print "Spent time", time.time()-temptime, "on computing m"
	#compute norms of stuff
		mchange = np.copy(m-m_old)
		ml1[kMax+n] = np.sum(abs(mchange))*dx
		ml2[kMax+n] = np.sqrt(np.sum(abs(mchange)**2))*np.sqrt(dx)
		mlinfty[kMax+n] = max(abs( mchange) ) 
		if (ml1[n+kMax] < tolerance):
			print "Method converged with final change" , ml1[n+kMax]
			print "Time spent:", time.time()-time_total
			m_best = np.copy(m)
			break
	#Evaluate iteration
		m_old = np.copy(m)
		u_old = np.copy(u)
		a_old = np.copy(a)
		print "Iteration number", n+1, "completed. Damping:",epsel[NN], "-", NN, "/", epsel.size, "\nUsed time", time.time()-titer, "\nChange in (a,u,m)=(",  alinfty[n+kMax][0], ",", ulinfty[n+kMax][0], ",", mlinfty[n+kMax][0], ")"
	#check if iteration was unsuccessful
	print "Time spent:", time.time()-time_total
	#store info on the last thing
	M_ends[:,NN] = m[(I*K-I):(I*K)]
	if (ml1[n+kMax] >= tolerance):	
		Success = False #this will only fall out of the loop if the complete thing was a success
		break
	kMax += n
	#print kMax
print "Total time spent:", time.time()-total_time2
#NN = NN + 1
if LOAD_WRITE: #load best solution with parameters
	if NN==0:
		sys.error("No convergence reached")
	print "Writing to file..."
	if Success: #write NN to file
		dx_string = "%.8f" % dx
		dt_string = "%.8f" % DT
		eps_string = "0"
		#filename = "damp_" + dx_string + "_" + dt_string + "_" + eps_string + "_" + ".txt"
		filename = "./" + TESTNAME + "_" + dx_string + "_" + dt_string + "_" + eps_string + "_" + ".txt"
		np.savetxt(filename, m_best, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ')
	else: #write NN-1 to file
		#check if other files with these parameters exist
		dx_string = "%.8f" % dx
		dt_string = "%.8f" % DT
		eps_string = "%.8f" % epsel[NN-1]
		#filename = "damp_" + dx_string + "_" + dt_string + "_" + eps_string + "_" + ".txt"
		filename = "./" + TESTNAME + "_" + dx_string + "_" + dt_string + "_" + eps_string + "_" + ".txt"
		np.savetxt(filename, m_best, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ')

M_ends = M_ends[:,:(NN+1)]
#for k in range (0,K):
#	Mass[k] = np.sum(m[(I*k):(I*k+I)])*dx-1
#Mass = Mass/np.finfo(float).eps

if Success:
	M_final = M_ends[:,NN]
else:
	M_final = M_ends[:,NN-1]
Ml1 = np.empty(NN)
Ml2 = np.empty(NN)
Mlinf = np.empty(NN)
if Success:
	for i in range(NN):
		Ml1[i] = np.linalg.norm(M_ends[:,i]-M_final,ord=1)
		Ml2[i] = np.linalg.norm(M_ends[:,i]-M_final)
		Mlinf[i] = np.linalg.norm(M_ends[:,i]-M_final,ord=np.inf)
else:
	for i in range(NN-1):
		Ml1[i] = np.linalg.norm(M_ends[:,i]-M_final,ord=1)
		Ml2[i] = np.linalg.norm(M_ends[:,i]-M_final)
		Mlinf[i] = np.linalg.norm(M_ends[:,i]-M_final,ord=np.inf)
print Ml1
print Ml2
print Mlinf

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
ax1.plot_surface(Xplot,Tplot,msoln,rstride=5,cstride=1,cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax1.set_xlabel('x')
ax1.set_ylabel('t')
ax1.set_zlabel('m(x,t)')
fig1.suptitle('Solution of the density m(x,t)', fontsize=14)
#levels = np.arange(min(m),max(m),0.5)
#norm = cm.colors.Normalize(vmax=abs(m).max(), vmin=0)
#cmap = cm.PRGn
#fig1 = plt.contourf(Xplot,Tplot,msoln,levels,cmap=plt.cm.Greys)
#CS2 = plt.contour(fig1, levels=levels[::2],colors = 'b')
#cbar = plt.colorbar(fig1)
#cbar.ax.set_ylabel('Distribution density')
#cbar.add_lines(CS2)
#ax1 = fig1.add_subplot(111)
#ax1.plot_surface(Xplot,Tplot,msoln,rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=0, antialiased=False)
#ax1.set_xlabel('x')
#ax1.set_ylabel('t')
#ax1.set_zlabel('m(x,t)')
#fig1.suptitle('Solution of the density m(x,t)', fontsize=14)
#plot solution of u(x,t)
fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111, projection='3d')
ax2.plot_surface(Xplot,Tplot,usoln,rstride=2,cstride=2,cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax2.set_xlabel('x')
ax2.set_ylabel('t')
ax2.set_zlabel('u(x,t)')
fig2.suptitle('Solution of the potential v(x,t)', fontsize=14)
#plot solution of a(x,t)
fig3 = plt.figure(3)
ax3 = fig3.add_subplot(111, projection='3d')
ax3.plot_surface(Xplot,Tplot,asoln,rstride=2,cstride=2,cmap=cm.coolwarm,linewidth=0, antialiased=False)
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
plt.plot(np.arange(1,kMax+1),np.log10(ul1), label="L1-norm")
plt.plot(np.arange(1,kMax+1),np.log10(ul2), label="L2-norm")
plt.plot(np.arange(1,kMax+1),np.log10(ulinfty), label="Linf-norm")
legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')
ax5 = fig5.add_subplot(111)
ax5.set_xlabel('Iteration number')
ax5.set_ylabel('Log10 of change')
fig5.suptitle('Convergence of u(x,t)', fontsize=14)
#plot the norms of change on a
fig6 = plt.figure(6)
plt.plot(np.arange(1,kMax+1), np.log10(al1), label="L1-norm")
plt.plot(np.arange(1,kMax+1), np.log10(al2), label="L2-norm")
plt.plot(np.arange(1,kMax+1), np.log10(alinfty), label="Linf-norm")
legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')
ax6 = fig6.add_subplot(111)
ax6.set_xlabel('Iteration number')
ax6.set_ylabel('Log10 of change')
fig6.suptitle('Convergence of a(x,t)', fontsize=14)
#fig7 = plt.figure(7)
#for NNs in range(NN+1):
#	string_thing = "Solution with artificial viscosity:", "%.6f" %epsel[NNs]
#	plt.plot(x,M_ends[:,NNs],label=string_thing)
#plt.grid()
#legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')

fig17 = plt.figure(17)
if Success:
	for NNs in range(NN+1):
		plt.plot(x,M_ends[:,NNs])
else:
	if NN>9:
		for NNs in range(NN-9,NN):
			plt.plot(x,M_ends[:,NNs])
	else:
		for NNs in range(NN):
			plt.plot(x,M_ends[:,NNs])
plt.grid()


##########PLOT

fig8 = plt.figure(8)
plt.plot(x,M_ends[:,NN])
plt.grid()
fig8.suptitle('Final solution',fontsize=14)

fig9 = plt.figure(9)
plt.loglog(epsel[:NN], (Ml1), label="L1-norm")
plt.loglog(epsel[:NN], (Ml2), label="L2-norm")
plt.loglog(epsel[:NN], (Mlinf), label="Linf-norm")
legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')
ax6 = fig9.add_subplot(111)
ax6.set_xlabel('Log of viscosity')
ax6.set_ylabel('Log10 of change')
plt.grid(True,which="both",ls="-")
ax6.invert_xaxis()
fig9.suptitle('Convergence rate of final solution',fontsize=14)

mtmp = m[(I*K-I):(I*K)]
begins,ends = app.spike_detector(mtmp)
#fig10 = plt.figure(10)
print x[begins],x[ends]
slope1, intercept = np.polyfit(np.log(epsel[:NN]), np.log(Ml1), 1)
slope2, intercept = np.polyfit(np.log(epsel[:NN]), np.log(Ml2), 1)
slopeinf, intercept = np.polyfit(np.log(epsel[:NN]), np.log(Mlinf), 1)

print "Slopes:"
print "\t",slope1
print "\t",slope2
print "\t",slopeinf

plt.show()


