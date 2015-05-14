from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import time as time
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize as minimize
from matplotlib import cm
from matplotlib import animation
from scipy.sparse.linalg import spsolve
import input_functions_2D as iF
import applications as app
import matrix_gen as mg
from scipy import sparse

animate = False
EXPLICIT = 0
OMETH = False#False#True
dx = 0.05
dt = .1*dx#.1*dx
#dt = 0.1*dx**2 #this is the finest CFL we can find for (D11,D22,D12)=(1,1,0); but not necessarily enough for monotonicity
xmin = 0
xmax = 1
ymin = 0
ymax = 1
T = .05
Niter = 1 #maximum number of iterations
D11 = 1
D12 = .99
D22 = 1
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
Xplot, Tplot = np.meshgrid(x,y)

#INITIALISE STORAGE
u = np.zeros((I*J)) #potential
def index(i,j): 
	return int(i+I*j)

def hmean(a,b):
	return a*b/(a+b)

#a = D11*dt/dx2
#print a

#INITIAL/TERMINAL CONDITIONS, INITIAL GUESS and COPYING
#u0 = iF.known_diffusion2(x,y,D11,D12,D22,1)
u0 = iF.initial_distribution(x,y)
u[0:I*J] = np.ravel(u0)
time_total = time.time()
BIGZERO = np.zeros(x.size-2)
lamb = dt/dx2
#EXPLICIT DIAMOND CELL
#M = np.identity(I*J)/lamb
#M = mg.add_diffusion_flux_DIAMOND(M,D11*np.ones(I*J),D22*np.ones(I*J),D12*np.ones(I*J),I,J,dx,dt)
#M = sparse.csr_matrix(M)*lamb
#IIOE
#LHS,RHS = mg.diffusion_flux_iioe(D11*np.ones(I*J),D22*np.ones(I*J),D12*np.ones(I*J),I,J,dx,dt)
#O-method
if OMETH:
	M = sparse.eye(I*J)
	M = sparse.lil_matrix(M)
	M = mg.add_diffusion_flux_Ometh(M,D11*np.ones(I*J),D22*np.ones(I*J),D12*np.ones(I*J),I,J,dx,dt,EXPLICIT)
	M = sparse.csr_matrix(M)
else:
	M = sparse.eye(I*J)
	M = sparse.lil_matrix(M)
	M = mg.add_diffusion_flux_nonlinear(M,D11*np.ones(I*J),D22*np.ones(I*J),D12*np.ones(I*J),I,J,dx,dt,u,EXPLICIT)
	M = sparse.csr_matrix(M)

#plt.spy(M)
#plt.show()
####CHECK THE SUMS, EACH SHOULD GO TO 1
#print M
#print ss
#print M.sum(1)
#sums = M.sum(1)
#fucks = np.empty(0)
#for i in range(0,sums.size):
#	if abs(sums[i]-1) > 1e-10:
#		fucks = np.append(fucks,i)
#		#print abs(sums[i]-1)
#print fucks
#xbound1 = range(0,I)
#ybound1 = range(0,I*J,I)
#xbound2 = range(I*J-I,I*J)
#ybound2 = range(I-1,I*J,I)
#print xbound1
#print xbound2
#print ybound1
#print ybound2
#print ss
#print M[fucks,:]
#print ss
mass0 = sum(u)*dx**2
######################################################################################
######################################################################################
#################################ANIMATION STUFF######################################
######################################################################################
######################################################################################
if animate:
	film = plt.figure(1)
	ax1 = film.add_subplot(111, projection='3d')
	ax1.set_zlim(0,1)
	filmstar = ax1.plot_surface(Xplot,Tplot,np.reshape(u,(I,J)),rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=0, antialiased=False)
	def generate(u,k):
		return M*np.ravel(u),
	def update(i, ax, fig,u):
		ax.cla()
		u = generate(u,i)
		filmstar = ax.plot_surface(Xplot,Tplot,np.reshape(u,(I,J)),rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=0, antialiased=False)
		ax.set_zlim(0,1)
		return filmstar,
	print "Attempting animation"
	ani = animation.FuncAnimation(film, update, frames=100, fargs=(ax1, film,u), interval=100,blit=True)
	print "Animation attempted"
	#mywriter = animation.FFMpegWriter()
	#ani.save('mymovie.mp4',writer=mywriter)
	#ani.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
	plt.show()
######################################################################################
######################################################################################

#ACTUALLY SOLVE
print "Solving commences"
for k in range(0,K-1):
	#u = spsolve(LHS,RHS*np.ravel(u)) #IIOE
	if EXPLICIT==1:
		u = M*np.ravel(u) #O-method
	else:
		if OMETH:
			u = spsolve(M,u)
		else:
			u_last = u
			count = 0
			t1 = time.time()
			t2 = 0
			while True:
				M = sparse.lil_matrix(sparse.eye(I*J))
				t3 = time.time()
				M = mg.add_diffusion_flux_nonlinear(M,D11*np.ones(I*J),D22*np.ones(I*J),D12*np.ones(I*J),I,J,dx,dt,u,EXPLICIT)
				t2 += time.time()-t3
				M = sparse.csr_matrix(M)
				u_old = u
				u = spsolve(M,u_last) #implicit O-method
				if np.linalg.norm(u-u_old) < 1e-6:
					print np.linalg.norm(u-u_old)
					break
				print np.linalg.norm(u-u_old)
				count += 1
			print "Iteration succesful,", k,"/",K-2
			print "\tPicard iteration successful on iteration number:",count
			print "\tTime spent:",time.time()-t1
			print "\tGeneration time:",t2/(time.time()-t1)*100
	print "Iteration succesful,", k,"/",K-2
	print "\tMass deviation:", (mass0-sum(u)*dx**2)
	#print "Max vs min:", abs(max(u)-min(u))
	print "\tMinimum value:", min(u)
print "Time spent:", time.time()-time_total

#PLOT STUFF
#plot solution of u(x,t)
fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111, projection='3d')
ax2.plot_surface(Xplot,Tplot,np.reshape(u,(I,J)),rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('u(x,t)')
fig2.suptitle('Solution of the potential u(x,t)', fontsize=14)
#initial solution
fig3 = plt.figure(3)
ax3 = fig3.add_subplot(111, projection='3d')
ax3.plot_surface(Xplot,Tplot,u0,rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('m0(x,t)')
fig3.suptitle('Initial distribution', fontsize=14)

##########PLOT
plt.show()









