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
import scipy.sparse as sparse
import scipy.interpolate as intpol
import glob,os,sys

dexes = [1/10, 1/20, 1/40, 1/80, 1/160, 1/320]
DT = .25
DEX_LEN = len(dexes)
epses = [None]*DEX_LEN
solutions = [None]*DEX_LEN
TEST_NAME = "mfg1d#moar0"

for i in range(len(dexes)): #load best solution with parameters
	best_eps = 42
	for file in glob.glob("*.txt"):
		pop = file.split("_")
		if pop[0]==TEST_NAME:
			pop_dx = float(pop[1])
			pop_DT = float(pop[2])
			pop_eps = float(pop[3])
			if abs(pop_dx-dexes[i])<1e-8 and pop_DT==DT: #if file found, read it		
				dx_string = "%.8f" % dexes[i]
				DT_string = "%.8f" % DT
				eps_string = "0"
				print "Loading previous computation result..."
				solutions[i] = np.loadtxt("./" + TEST_NAME + "_" + dx_string + "_" + DT_string + "_" + eps_string + "_" + ".txt")
				print "Loading successful!"
				break

#now all solutions are loaded, interpolate

x_finest = np.linspace(0,1,1/dexes[-1]+1)
t_finest = np.linspace(0,1,1/(DT*dexes[-1])+1)
finest = np.reshape(solutions[-1],(t_finest.size,x_finest.size))
Xplot,Tplot = np.meshgrid(x_finest,t_finest)


interpolates = [None]*(DEX_LEN-1)
#x_finest,t_finest = np.meshgrid(x_finest,t_finest)
for i in range(DEX_LEN-1):
	xt = np.linspace(0,1,1/dexes[i]+1)
	tt = np.linspace(0,1,1/(DT*dexes[i])+1)
	#xt,tt = np.meshgrid(xt,tt)
	tmp = intpol.RectBivariateSpline(tt, xt, np.reshape(solutions[i],(tt.size,xt.size)), bbox=[0, 1, 0, 1], kx=1, ky=1, s=0)
	interpolates[i] = tmp(t_finest,x_finest)

Xplot,Tplot = np.meshgrid(x_finest,t_finest)

print Xplot.shape,Tplot.shape,finest.shape
#now compare
e1 = np.zeros(DEX_LEN-1)
e2 = np.zeros(DEX_LEN-1)
einf = np.zeros(DEX_LEN-1)
for i in range(DEX_LEN-1):
	e1[i] = np.linalg.norm(finest[-1,:]-interpolates[i][-1,:],ord=1)
	e2[i] = np.linalg.norm(finest[-1,:]-interpolates[i][-1,:],ord=2)
	einf[i] = np.linalg.norm(finest[-1,:]-interpolates[i][-1,:],ord=np.inf)

slope1, intercept = np.polyfit(np.log(dexes[:-1]), np.log(e1[:]), 1)
slope1_1, intercept = np.polyfit(np.log(dexes[:-1]), np.log(e2[:]), 1)
slope1_inf, intercept = np.polyfit(np.log(dexes[:-1]), np.log(einf[:]), 1)
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
fig1.suptitle('Convergence rates of m(x,t)', fontsize=14)

fig3 = plt.figure(3)
for i in range(DEX_LEN-1):
	plt.plot(x_finest,interpolates[i][-1,:])
	#print interpolates[i][-1,:].shape
plt.plot(x_finest,finest[-1,:])


plt.show()

CRAZY=False
if CRAZY:
	fig2 = plt.figure(2)
	for i in range(DEX_LEN-1):
		ax1 = fig2.add_subplot(2, 3, i+1, projection='3d')
		ax1.plot_surface(Xplot,Tplot,(interpolates[i]),rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=0, antialiased=False)
	ax1 = fig2.add_subplot(2, 3, 6, projection='3d')
	ax1.plot_surface(Xplot,Tplot,(finest),rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=0, antialiased=False)
	ax1.set_xlabel('x')
	ax1.set_ylabel('y')
	ax1.set_zlabel('m(x,t)')
else:
	fig2,ax2 = plt.subplots(2,3)
	for ax in ax2.ravel():
		levels = max(m)*np.arange(0,1,.01)**2
		#print levels
		#print ss
		norm = cm.colors.Normalize(vmax=abs(m).max(), vmin=0)
		cmap = cm.PRGn
		fig1 = plt.contourf(Xplot,Tplot,np.reshape(m,(t.size,x.size)),levels,cmap=plt.cm.pink)
		#CS2 = plt.contour(fig1, levels=levels[::2],colors = 'b')
		cbar = plt.colorbar(fig1)


