from __future__ import division
import numpy as np
from scipy import sparse

	#center = np.zeros(I*J) #(i,j)
	#off1pi = np.zeros(I*J-2) #(i+1,j)
	#off1mi = np.zeros(I*J-2) #(i-1,j)
	#off1pj = np.zeros(I*J-2*I) #(i,j+1)
	#off1mj = np.zeros(I*J-2*I) #(i,j+1)
	#center = 1/lamb*np.ones(I*J) + dx*(f1_array+f2_array) + D12 - D11 - D22
	#need some tricks for the others; this will do for now
	#movement = f_global
	#[D11 D12 D22] = sigma_stuff
	#flatten out D12, D11, D22, f1_array, f2_array

##################
# MATRIX GENERATION
##################

def u_matrix_implicit(f1_array,f2_array,D11,D22,D12,I,J,dx,dt): #THIS IS NOT IMPLEMENTED
	output = np.zeros((I*J,I*J))
	lamb = dt/(dx**2)
	xbound1 = range(0,I)
	ybound1 = range(0,I*J,I)
	xbound2 = range(I*J-I,I*J)
	ybound2 = range(I-1,I*J,I)
	#flatten out D12, D11, D22, f1_array, f2_array
	D12 = np.ravel(D12)
	D11 = np.ravel(D11)
	D22 = np.ravel(D22)
	f1_array = np.ravel(f1_array)
	f2_array = np.ravel(f2_array)
	for i in range(0,I*J):
		d12 = D12[i]
		d11 = D11[i]
		d22 = D22[i]
		f1 = f1_array[i]
		f2 = f2_array[i]
		output[i,i] += 1/lamb + dx*(f1+f2) - d12 + d11 + d22
		#avoid segfaults
		if not ismember_sorted(i,xbound1): #allows (i,j-1)
			output[i,i-I] += -(d22-d12)/2 - dx*min(f2,0)
		if not ismember_sorted(i,xbound2): #allows (i,j+1)
			output[i,i+I] += -(d22-d12)/2 - dx*max(f2,0)
		if not ismember_sorted(i,ybound1): #allows (i-1,j) 
			output[i,i-1] += -(d11-d12)/2 - dx*min(f1,0)
		if not ismember_sorted(i,ybound2): #allows (i+1,j)
			output[i,i+1] += -(d11-d12)/2 - dx*max(f1,0)
		if not ismember_sorted(i,xbound1) and not ismember_sorted(i,ybound1): #allows (i-1,j-1)
			output[i,i-1-I] += - max(d12,0)
		if not ismember_sorted(i,xbound2) and not ismember_sorted(i,ybound2): #allows (i+1,j+1)
			output[i,i+1+I] += - max(d12,0)
		if not ismember_sorted(i,xbound1) and not ismember_sorted(i,ybound2): #allows (i+1,j-1)
			output[i,i+1-I] += - min(d12,0)
		if not ismember_sorted(i,ybound1) and not ismember_sorted(i,xbound2): #allows (i-1,j+1)
			output[i,i-1+I] += - min(d12,0)
		#then add boundary conditions
		if ismember_sorted(i,xbound1): #allows (i,j-1)
			output[i,i+I] += -(d22-d12)/2 - dx*min(f2,0)
		if ismember_sorted(i,xbound2): #allows (i,j+1)
			output[i,i-I] += -(d22-d12)/2 - dx*max(f2,0)
		if ismember_sorted(i,ybound1): #allows (i-1,j) 
			output[i,i+1] += -(d11-d12)/2 - dx*min(f1,0)
		if ismember_sorted(i,ybound2): #allows (i+1,j)
			output[i,i-1] += -(d11-d12)/2 - dx*max(f1,0)
		if ismember_sorted(i,xbound1) and ismember_sorted(i,ybound1): #allows (i-1,j-1)
			output[i,i+1+I] += -max(d12,0)
		if ismember_sorted(i,xbound2) and ismember_sorted(i,ybound2): #allows (i+1,j+1)
			output[i,i-1-I] += -max(d12,0)
		if ismember_sorted(i,xbound1) and ismember_sorted(i,ybound2): #allows (i+1,j-1)
			output[i,i-1+I] += - min(d12,0)
		if ismember_sorted(i,ybound1) and ismember_sorted(i,xbound2): #allows (i-1,j+1)
			output[i,i+1-I] += - min(d12,0)
	return lamb*sparse.csr_matrix(output)

#########################
def u_matrix_explicit(f1_array,f2_array,D11,D22,D12,I,J,dx,dt):
	output = np.zeros((I*J)**2)
	lamb = dt/(dx**2)
	xbound1 = range(0,I)
	ybound1 = range(0,I*J,I)
	xbound2 = range(I*J-I,I*J)
	ybound2 = range(I-1,I*J,I)
	D12 = np.ravel(D12)
	D11 = np.ravel(D11)
	D22 = np.ravel(D22)
	f1_array = np.ravel(f1_array)
	f2_array = np.ravel(f2_array)
	for i in range(0,I*J):
		#indx = i+I*j
		d12 = D12[i]
		d11 = D11[i]
		d22 = D22[i]
		f1 = f1_array[i]
		f2 = f2_array[i]
		output[i] = 1/lamb + dx*(f1+f2) + d12 - d11 - d22
		#avoid segfaults
		if not ismember_sorted(i,xbound1): #allows (i,j-1)
			output[i+I*(i-I)] += -dx*max(f2,0) + d22/2 - max(d12,0)
			#output[i+I*(i-I)] += -dx*min(f2,0) + d22/2 - min(d12,0)
		if not ismember_sorted(i,xbound2): #allows (i,j+1)
			output[i+I*(i+I)] += dx*min(f2,0) + d22/2 - min(d12,0)
			#output[i+I*(i+I)] += dx*max(f2,0) + d22/2 - max(d12,0)
		if not ismember_sorted(i,ybound1): #allows (i-1,j)
			output[i+I*(i-1)] += -dx*max(f1,0) + d11/2 - max(d12,0)
			#output[i+I*(i-1)] += -dx*min(f1,0) + d11/2 - min(d12,0)
		if not ismember_sorted(i,ybound2): #allows (i+1,j)
			output[i+I*(i+1)] += dx*min(f1,0) + d11/2 - min(d12,0)
			#output[i+I*(i+1)] += dx*max(f1,0) + d11/2 - max(d12,0)
		if not ismember_sorted(i,xbound1) and not ismember_sorted(i,ybound1): #allows (i-1,j-1)
			output[i+I*(i-1-I)] += max(d12,0)
		if not ismember_sorted(i,xbound2) and not ismember_sorted(i,ybound2): #allows (i+1,j+1)
			output[i+I*(i+1+I)] += min(d12,0)
		#then add boundary conditions
		if ismember_sorted(i,xbound1): 
			output[i+I*(i+I)] += -dx*max(f2,0) + d22/2 - max(d12,0)
		if ismember_sorted(i,xbound2):
			output[i+I*(i-I)] += dx*min(f2,0) + d22/2 - min(d12,0)
		if ismember_sorted(i,ybound1):
			output[i+I*(i+1)] += -dx*max(f1,0) + d11/2 - max(d12,0)
		if ismember_sorted(i,ybound2):	
			output[i+I*(i-1)] += dx*min(f1,0) + d11/2 - min(d12,0)
		if ismember_sorted(i,xbound1) and ismember_sorted(i,ybound1): #allows (i-1,j-1)
			output[i+I*(i+1+I)] += max(d12,0)
		if ismember_sorted(i,xbound2) and ismember_sorted(i,ybound2): #allows (i-1,j-1)
			output[i+I*(i-1-I)] += min(d12,0)
	#make output into matrix
	output = np.reshape(output, (I*J,I*J))
	return lamb*sparse.csr_matrix(output)

def m_matrix_explicit(f1_array,f2_array,D11,D22,D12,I,J,dx,dt):
	output = np.zeros((I*J)**2)
	lamb = dt/(dx**2)
	xbound1 = range(0,I)
	ybound1 = range(0,I*J,I)
	xbound2 = range(I*J-I,I*J)
	ybound2 = range(I-1,I*J,I)
	D12 = np.ravel(D12)
	D11 = np.ravel(D11)
	D22 = np.ravel(D22)
	f1_array = np.ravel(f1_array)
	f2_array = np.ravel(f2_array)
	for i in range(0,I*J):
		#indx = i+I*j
		output[i] = 1/lamb + D12[i] - D11[i] - D22[i] #done
		#avoid segfaults
		if not ismember_sorted(i,xbound1): #allows (i,j-1)
			output[i+I*(i-I)] += dx*f2_array[i-I]/2 + D22[i-I]/2 - max(D12[i-I],0) #done
		if not ismember_sorted(i,xbound2): #allows (i,j+1)
			output[i+I*(i+I)] += -dx*f2_array[i+I]/2 + D22[i+I]/2 - min(D12[i+I],0) #done
		if not ismember_sorted(i,ybound1): #allows (i-1,j)
			output[i+I*(i-1)] += dx*f1_array[i-1]/2 + D11[i-1]/2 - max(D12[i-1],0) #done 
		if not ismember_sorted(i,ybound2): #allows (i+1,j)
			output[i+I*(i+1)] += -dx*f1_array[i+1]/2 + D11[i+1]/2 - min(D12[i+1],0) #done
		if not ismember_sorted(i,xbound1) and not ismember_sorted(i,ybound1): #allows (i-1,j-1)
			output[i+I*(i-1-I)] += max(D12[i-1-I],0)
		if not ismember_sorted(i,xbound2) and not ismember_sorted(i,ybound2): #allows (i+1,j+1)
			output[i+I*(i+1+I)] += min(D12[i+1+I],0)
		#then add boundary conditions (NOT TRIVIAL)
		if ismember_sorted(i,xbound1): 
			output[i+I*(i+I)] += D22[i+I]/2 - max(D12[i+I],0)
		if ismember_sorted(i,xbound2):
			output[i+I*(i-I)] += D22[i-I]/2 - min(D12[i-I],0)
		if ismember_sorted(i,ybound1):
			output[i+I*(i+1)] += D11[i+1]/2 - max(D12[i+1],0)
		if ismember_sorted(i,ybound2):	
			output[i+I*(i-1)] += D11[i-1]/2 - min(D12[i-1],0)
	#make output into matrix
	output = np.reshape(output, (I*J,I*J))
	return lamb*sparse.csr_matrix(output)

def m_matrix_implicit(f1_array,f2_array,D11,D22,D12,I,J,dx,dt):
	output = np.zeros((I*J)**2)
	lamb = dt/(dx**2)
	xbound1 = range(0,I)
	ybound1 = range(0,I*J,I)
	xbound2 = range(I*J-I,I*J)
	ybound2 = range(I-1,I*J,I)
	D12 = np.ravel(D12)
	D11 = np.ravel(D11)
	D22 = np.ravel(D22)
	f1_array = np.ravel(f1_array)
	f2_array = np.ravel(f2_array)
	for i in range(0,I*J):
		#indx = i+I*j
		d12 = D12[i]
		d11 = D11[i]
		d22 = D22[i]
		f1 = f1_array[i]
		f2 = f2_array[i]
		output[i] = 1/lamb - d12 + d11 + d22 #done
		#avoid segfaults
		if not ismember_sorted(i,xbound1): #allows (i,j-1)
			output[i+I*(i-I)] += -dx*f2_array[i-I]/2 - D22[i-I]/2 + max(D12[i-I],0) #done
		if not ismember_sorted(i,xbound2): #allows (i,j+1)
			output[i+I*(i+I)] += dx*f2_array[i+I]/2 - D22[i+I]/2 + min(D12[i+I],0) #done
		if not ismember_sorted(i,ybound1): #allows (i-1,j)
			output[i+I*(i-1)] += -dx*f1_array[i-1]/2 - D11[i-1]/2 + max(D12[i-1],0) #done 
		if not ismember_sorted(i,ybound2): #allows (i+1,j)
			output[i+I*(i+1)] += dx*f1_array[i+1]/2 - D11[i+1]/2 + min(D12[i+1],0) #done
		#then add boundary conditions (NOT TRIVIAL)
		if ismember_sorted(i,xbound1): 
			output[i+I*(i+I)] += -D22[i+I]/2 + max(D12[i+I],0)
		if ismember_sorted(i,xbound2):
			output[i+I*(i-I)] += -D22[i-I]/2 + min(D12[i-I],0)
		if ismember_sorted(i,ybound1):
			output[i+I*(i+1)] += -D11[i+1]/2 + max(D12[i+1],0)
		if ismember_sorted(i,ybound2):	
			output[i+I*(i-1)] += -D11[i-1]/2 + min(D12[i-1],0)
	#make output into matrix
	output = np.reshape(output, (I*J,I*J))
	return lamb*sparse.csr_matrix(output)


def ismember_sorted(a,array):
	for i in range(0,len(array)):
		if array[i]==a:
			return True
		elif array[i]>a:
			return False
	return False

