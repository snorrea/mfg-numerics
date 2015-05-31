from __future__ import division
import numpy as np
from scipy import sparse
import input_functions as iF
import matrix_gen1d as mg
##################
# MATRIX GENERATION: HJB
##################

def grand_hjb_cmfg_ng(times,x,a_grand,m_grand,dt,dx):
	K = times.size
	I = x.size
	all_store = [[0 for foo in range(K)] for bar in range(K)] #makes a KxK array
	for k in range(0,K):
		temp_store = [None]*K
		if k==K-1:
			temp_store[K-1] = sparse.eye(I)
			#for i in range(0,K-1):
			#	temp_store[i] = None
		else:
			#for i in range(0,k):
			#	temp_store[i] = None
			temp_store[k] = mg.hjb_diffusion(times[k],x,a_grand[(I*k):(I+I*k)],dt,dx)
			temp_store[k+1] = -mg.hjb_convection(times[k],x,a_grand[(I*k):(I+I*k)],dt,dx)
			#for i in range(k+2,K):
			#	temp_store[i] = None
		all_store[k] = temp_store
	return sparse.bmat(all_store,format="csr")


def grand_hjb_vector(times,x,a,m,dt):
	I = x.size
	K = times.size
	dx = x[1]-x[0]
	output = np.zeros((I*K))
	output[(I*K-I):(I*K)] = iF.G(x,m[(I*K-I):(I*K)])
	for k in range(0,K-1):
		output[(I*k):(I*k+I)] = dt*iF.L_global(times[k],x,a[(I*k):(I*k+I)],m[(I*k):(I*k+I)],dx)
	#print output
	#print ss
	return output

##################
# MATRIX GENERATION: FP
##################

def grand_fp_cmfg_ng(times,x,a_grand,dt,dx):
	K = times.size
	I = x.size
	all_store = [[0 for foo in range(K)] for bar in range(K)] #makes a KxK array
	for k in range(0,K):
		temp_store = [None]*K
		if k==0:
			temp_store[0] = sparse.eye(I)
			#for i in range(1,K):
			#	temp_store[i] = None
		else:
			#for i in range(0,k-1):
			#	temp_store[i] = None
			temp_store[k-1] = -mg.fp_fv_convection_classic(times[k],x,a_grand[(I*k):(I+I*k)],dt,dx)
			temp_store[k] = mg.fp_fv_diffusion(times[k],x,a_grand[(I*k):(I+I*k)],dt,dx)
			#for i in range(k+1,K):
			#	temp_store[i] = None
		all_store[k] = temp_store
	return sparse.bmat(all_store,format="csr")

def grand_fp_vector(x,K):
	I = x.size
	dx = x[1]-x[0]
	output = np.zeros((I*K))
	m0 = iF.initial_distribution(x)
	output[0:I] = m0/(sum(m0)*dx)
	return output
	

