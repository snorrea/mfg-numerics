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

def u_matrix_implicit(f1_array,f2_array,D11,D22,D12,I,J,dx,dt): #THIS JUST WORKS
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
			output[i,i-1-I] += - max(d12,0)/2
		if not ismember_sorted(i,xbound2) and not ismember_sorted(i,ybound2): #allows (i+1,j+1)
			output[i,i+1+I] += - max(d12,0)/2
		if not ismember_sorted(i,xbound1) and not ismember_sorted(i,ybound2): #allows (i+1,j-1)
			output[i,i+1-I] += - min(d12,0)/2
		if not ismember_sorted(i,ybound1) and not ismember_sorted(i,xbound2): #allows (i-1,j+1)
			output[i,i-1+I] += - min(d12,0)/2
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
			output[i,i+1+I] += -max(d12,0)/2
		if ismember_sorted(i,xbound2) and ismember_sorted(i,ybound2): #allows (i+1,j+1)
			output[i,i-1-I] += -max(d12,0)/2
		if ismember_sorted(i,xbound1) and ismember_sorted(i,ybound2): #allows (i+1,j-1)
			output[i,i-1+I] += - min(d12,0)/2
		if ismember_sorted(i,ybound1) and ismember_sorted(i,xbound2): #allows (i-1,j+1)
			output[i,i+1-I] += - min(d12,0)/2
	return lamb*sparse.csr_matrix(output)

#########################
def u_matrix_explicit(f1_array,f2_array,D11,D22,D12,I,J,dx,dt):
	output = np.zeros((I*J,I*J))
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
		output[i] = 1/lamb - dx*(f1+f2) + d12 - d11 - d22
		#avoid segfaults
		if not ismember_sorted(i,xbound1): #allows (i,j-1)
			output[i,i-I] += (d22-d12)/2 + dx*min(f2,0)
		if not ismember_sorted(i,xbound2): #allows (i,j+1)
			output[i,i+I] += (d22-d12)/2 + dx*max(f2,0)
		if not ismember_sorted(i,ybound1): #allows (i-1,j) 
			output[i,i-1] += (d11-d12)/2 + dx*min(f1,0)
		if not ismember_sorted(i,ybound2): #allows (i+1,j)
			output[i,i+1] += (d11-d12)/2 + dx*max(f1,0)
		if not ismember_sorted(i,xbound1) and not ismember_sorted(i,ybound1): #allows (i-1,j-1)
			output[i,i-1-I] += max(d12,0)/2
		if not ismember_sorted(i,xbound2) and not ismember_sorted(i,ybound2): #allows (i+1,j+1)
			output[i,i+1+I] += max(d12,0)/2
		if not ismember_sorted(i,xbound1) and not ismember_sorted(i,ybound2): #allows (i+1,j-1)
			output[i,i+1-I] += min(d12,0)/2
		if not ismember_sorted(i,ybound1) and not ismember_sorted(i,xbound2): #allows (i-1,j+1)
			output[i,i-1+I] += min(d12,0)/2
		#then add boundary conditions
		if ismember_sorted(i,xbound1): #allows (i,j-1)
			output[i,i+I] += (d22-d12)/2 + dx*min(f2,0)
		if ismember_sorted(i,xbound2): #allows (i,j+1)
			output[i,i-I] += (d22-d12)/2 + dx*max(f2,0)
		if ismember_sorted(i,ybound1): #allows (i-1,j) 
			output[i,i+1] += (d11-d12)/2 + dx*min(f1,0)
		if ismember_sorted(i,ybound2): #allows (i+1,j)
			output[i,i-1] += (d11-d12)/2 + dx*max(f1,0)
		if ismember_sorted(i,xbound1) and ismember_sorted(i,ybound1): #allows (i-1,j-1)
			output[i,i+1+I] += max(d12,0)/2
		if ismember_sorted(i,xbound2) and ismember_sorted(i,ybound2): #allows (i+1,j+1)
			output[i,i-1-I] += max(d12,0)/2
		if ismember_sorted(i,xbound1) and ismember_sorted(i,ybound2): #allows (i+1,j-1)
			output[i,i-1+I] += min(d12,0)/2
		if ismember_sorted(i,ybound1) and ismember_sorted(i,xbound2): #allows (i-1,j+1)
			output[i,i+1-I] += min(d12,0)/2
	return lamb*sparse.csr_matrix(output)

#####################
def u_matrix_implicit(f1_array,f2_array,D11,D22,D12,I,J,dx,dt): #THIS JUST WORKS
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
#		d12 = D12[i]
#		d11 = D11[i]
#		d22 = D22[i]
#		f1 = f1_array[i]
#		f2 = f2_array[i]
		output[i,i] += 1/lamb + dx*(max(0.5*(f1_array[i]+f1_array[i+1]),0)+max(0.5*(f2_array[i]+f2_array[i+1]),0)-min(0.5*(f1_array[i]+f1_array[i-1]),0)-min(0.5*(f2_array[i]+f2_array[i-1]),0)) #term i)
		output[i,i] += -max(0,D11[i+1]-D11[i]+0.25*(D12[i+1+I]+D12[i+I]-D12[i-I]-D12[i+1-I])) - max(0,D22[i+I]-D22[i]+0.25*(D12[i+1+I]+D12[i+1]-D12[i-1]-D12[i-1+I])) #term ii) part 1
		output[i,i] += min(0,D11[i]-D11[i-1]+0.25*(D12[i+I]+D12[i-1+I]-D12[i-I]-D12[i-1-I])) + min(0,D22[i]-D22[i-I]+0.25*(D12[i+1]+D12[i+1-I]-D12[i-1]-D12[i-1-I])) #term ii) part 2
		output[i,i] += 0.5*(D11[i+1]+D22[i+I]-D11[i-1]-D22[i-I]) #term iii)
		#avoid segfaults
		if not ismember_sorted(i,xbound1): #allows (i,j-1)
			output[i,i-I] += max(0.5*(f2_array[i-I]+f2_array[i]),0)/h #term i)
			output[i,i-I] += max(0,D22[i]-D22[i-I]+0.25*(D12[i+1]+D12[i+1-I]-D12[i-1]-D12[i-1-I])) #term ii), conflict
			output[i,i-I] += -(-0.125*(D12[i+1]+D12[i]) + 0.5*(D22[i]+D22[i-I]) + 0.125*(D12[i]+D12[i-1]) )) #term iii), conflict
		if not ismember_sorted(i,xbound2): #allows (i,j+1)
			output[i,i+I] += min(0.5*(f2_array[i+I]+f2_array[i]),0)/h #term i)
			output[i,i+I] += -min(0,D22[i+I]-D22[i]+0.25*(D12[i+1+I]+D12[i+1]-D12[i-1]-D12[i-1+I])) #term ii), conflict
			output[i,i+I] += -(0.125*(D12[i+1]+D12[i])+0.5*(D22[i]+D22[i+I]) - 0.125*(D12[i]+D12[i-1]) ) #term iii), conflict
		if not ismember_sorted(i,ybound1): #allows (i-1,j) 
			output[i,i-1] += max(0.5*(f1_array[i-1]+f1_array[i]),0)/h #term i)
			output[i,i-1] += max(0,D11[i]-D11[i-1]+0.25*(D12[i+I]+D12[i-1+I]-D12[i-I]-D12[i-1-I])) #term ii), conflict
			output[i,i-1] += -(-0.125*(D12[i]+D12[i+I])+0.5*(D11[i]+D11[i-1])+0.125*(D12[i]+D12[i-I])) #term iii), conflict
		if not ismember_sorted(i,ybound2): #allows (i+1,j)
			output[i,i+1] += min(0.5*(f1_array[i+1]+f1_array[i]),0)/h #term i)
			output[i,i+1] += -min(0,D11[i+1]-D11[i]+0.25*(D12[i+1+I]+D12[i+I]-D12[i-I]-D12[i+1-I])) #term ii), conflict
			output[i,i+1] += -(0.5*(D11[i]+D11[i+1]) + 0.125*(D12[i]+D12[i+I]) - 0.125*(D12[i]+D12[i-I])) #term iii), conflict
		if not ismember_sorted(i,xbound1) and not ismember_sorted(i,ybound1): #allows (i-1,j-1), no conflicts
			output[i,i-1-I] += -0.125*(D12[i]+D12[i-1]+D12[i]+D12[i-I])
		if not ismember_sorted(i,xbound2) and not ismember_sorted(i,ybound2): #allows (i+1,j+1), no conflicts
			output[i,i+1+I] += -0.125*(D12[i]+D12[i+1]+D12[i]+D12[i+I])
		if not ismember_sorted(i,xbound1) and not ismember_sorted(i,ybound2): #allows (i+1,j-1), no conflicts
			output[i,i+1-I] += 0.125*(D12[i]+D12[i+1]+D12[i]+D12[i-I])
		if not ismember_sorted(i,ybound1) and not ismember_sorted(i,xbound2): #allows (i-1,j+1), no conflicts
			output[i,i-1+I] += 0.125*(D12[i]+D12[i-1]+D12[i]+D12[i+I])
		#then add boundary conditions; this is mostly guesswork
		if ismember_sorted(i,xbound1): #allows (i,j-1)
			output[i,i+I] += 0
		if ismember_sorted(i,xbound2): #allows (i,j+1)
			output[i,i-I] += 0
		if ismember_sorted(i,ybound1): #allows (i-1,j) 
			output[i,i+1] += 0
		if ismember_sorted(i,ybound2): #allows (i+1,j)
			output[i,i-1] += 0
		if ismember_sorted(i,xbound1) and ismember_sorted(i,ybound1): #allows (i-1,j-1)
			output[i,i+1+I] += 0
		if ismember_sorted(i,xbound2) and ismember_sorted(i,ybound2): #allows (i+1,j+1)
			output[i,i-1-I] += 0
		if ismember_sorted(i,xbound1) and ismember_sorted(i,ybound2): #allows (i+1,j-1)
			output[i,i-1+I] += 0
		if ismember_sorted(i,ybound1) and ismember_sorted(i,xbound2): #allows (i-1,j+1)
			output[i,i+1-I] += 0
	return lamb*sparse.csr_matrix(output)



def ismember_sorted(a,array):
	for i in range(0,len(array)):
		if array[i]==a:
			return True
		elif array[i]>a:
			return False
	return False

