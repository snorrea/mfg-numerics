from __future__ import division
import numpy as np
import quadrature_nodes as qn
import input_functions_2D as iF
import matrix_gen as mg
from scipy.sparse.linalg import spsolve
quad_order = 15
gll_x = qn.GLL_points(quad_order) #quadrature nodes
gll_w = qn.GLL_weights(quad_order,gll_x)

###################
#MINIMISING FUNCTIONS
###################
def scatter_search(function, (args),dx,x0,y0,N,k): #here k is the number of laps going
	x_naught = x0
	y_naught = y0
	dex = dx
	for i in range (0,k):
		xpts = np.linspace(x_naught-dex,x_naught+dex,N)
		ypts = np.linspace(y_naught-dex,y_naught+dex,N)
		fpts = function(xpts,ypts,*args)
		if i!=k-1:
			crit_ind = np.argmin(fpts)
			xi,yi = recover_index(crit_ind,xpts.size)
			x_naught = xpts[xi]
			y_naught = ypts[yi]
			dex = xpts[2]-xpts[1]
	crit_ind = np.argmin(fpts)
	#print crit_ind
	xi,yi = recover_index(crit_ind,xpts.size)
	#print xi,yi
	return xpts[xi],ypts[yi]

###################
#POLICY ITERATION
###################
def policy_implicit(search,x,y,u0,m,dt,dx,time,I,J,tol,scatters,N):
	count = 0
	err = 1
	u = u0
	a1 = np.zeros((I,J))
	a2 = np.zeros((I,J))
	a1old = np.zeros((I,J))
	a2old = np.zeros((I,J))
	#print "Search",search
	while err > tol:
		count +=1
		#compute a guess of a1,a2
		for i in range (0,I):
			for j in range (0,J):
				fpts = iF.hamiltonian(search,search,x,y,u,m,dt,dx,time,i,j,I,J)
				xi,yi = recover_index(np.argmin(fpts),search.size)
				tmp_x,tmp_y = scatter_search(iF.hamiltonian,(x,y,u,m,dt,dx,time,i,j,I,J),search[2]-search[1],search[xi],search[yi],N,scatters)
				a1[i,j] = tmp_x
				a2[i,j] = tmp_y
		err = 0.5*(abs(max(np.ravel(a1old-a1)))+abs(max(np.ravel(a2old-a2))))
		if err > tol:
			L_var = np.ravel(iF.L_global(time,x,y,a1,a2,m))
			[f1_array, f2_array] = iF.f_global(time,x,y,a1,a2)
			D11 = iF.Sigma_D11_test(time,x,y,a1,a2,m)
			D12 = iF.Sigma_D12_test(time,x,y,a1,a2,m)
			D22 = iF.Sigma_D22_test(time,x,y,a1,a2,m)
			U_WORK_4_DIS = mg.u_matrix_implicit(f1_array,f2_array,D11,D22,D12,I,J,dx,dt)
			u = spsolve(U_WORK_4_DIS,np.ravel(u0)+dt*L_var)
			a1old = a1
			a2old = a2
		else:
			break
	#print count
	return a1,a2

###################
#AUXILIARY FUNCTIONS
###################
def hmean(a,b): #returns harmonic mean of a,b
	return 2*a*b/(a+b)

def recover_index(crit_ind,key): #where key is as in index:=i+I*j, with I=key. this function recovers i,j
	i=0
	j=0
	while crit_ind - key >= 0:
		crit_ind = crit_ind - key
		j += 1
	while crit_ind > 0:
		crit_ind += -1
		i += 1
	#print i+j*key == crit_ind
	return i,j

def mollifier(x_val): #Evans' mollifier
	if abs(x_val) < 1:
		return np.exp(1/(x_val**2 - 1))/0.443994
	else:
		return 0

def mollify_array(array,epsilon,x_array,gll_x,gll_w): 
	output = np.zeros((array.size))
	for k in range (0,array.size):
		for j in range (0,gll_x.size):
			output[k] += gll_w[j]*mollifier(gll_x[j])*np.interp(x_array[k]-epsilon*gll_x[j],x_array,array)
	return output
def restrain(trajectory,x_array):
	trajectory = np.minimum(np.maximum(x_array[0]*np.ones(x_array.size),trajectory),x_array[-1]*np.ones(x_array.size))
	return trajectory
def restrain4isolation(trajectory,x_array):
	return np.maximum(restrain(trajectory),x_array)

