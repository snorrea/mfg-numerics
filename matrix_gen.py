from __future__ import division
import numpy as np
from scipy import sparse
import applications as app
import assembly as ass

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
		output[i,i] += 1/lamb + dx*(abs(f1)+abs(f2)) - abs(d12) + d11 + d22
		#avoid segfaults
		if not ismember_sorted(i,xbound1): #allows (i,j-1)
			output[i,i-I] += -(d22-abs(d12))/2 + dx*min(f2,0)
		if not ismember_sorted(i,xbound2): #allows (i,j+1)
			output[i,i+I] += -(d22-abs(d12))/2 - dx*max(f2,0)
		if not ismember_sorted(i,ybound1): #allows (i-1,j) 
			output[i,i-1] += -(d11-abs(d12))/2 + dx*min(f1,0)
		if not ismember_sorted(i,ybound2): #allows (i+1,j)
			output[i,i+1] += -(d11-abs(d12))/2 - dx*max(f1,0)
		if not ismember_sorted(i,xbound1) and not ismember_sorted(i,ybound1): #allows (i-1,j-1)
			output[i,i-1-I] += -max(d12,0)/2
		if not ismember_sorted(i,xbound2) and not ismember_sorted(i,ybound2): #allows (i+1,j+1)
			output[i,i+1+I] += -max(d12,0)/2
		if not ismember_sorted(i,xbound1) and not ismember_sorted(i,ybound2): #allows (i+1,j-1)
			output[i,i+1-I] += min(d12,0)/2
		if not ismember_sorted(i,ybound1) and not ismember_sorted(i,xbound2): #allows (i-1,j+1)
			output[i,i-1+I] += min(d12,0)/2
		#then add boundary conditions
		if ismember_sorted(i,xbound1): #allows (i,j-1)
			output[i,i+I] += -(d22-abs(d12))/2 + dx*min(f2,0)
		if ismember_sorted(i,xbound2): #allows (i,j+1)
			output[i,i-I] += -(d22-abs(d12))/2 - dx*max(f2,0)
		if ismember_sorted(i,ybound1): #allows (i-1,j) 
			output[i,i+1] += -(d11-abs(d12))/2 + dx*min(f1,0)
		if ismember_sorted(i,ybound2): #allows (i+1,j)
			output[i,i-1] += -(d11-abs(d12))/2 - dx*max(f1,0)
		if ismember_sorted(i,xbound1) and ismember_sorted(i,ybound1): #allows (i-1,j-1)
			output[i,i+1+I] += -max(d12,0)/2
		if ismember_sorted(i,xbound2) and ismember_sorted(i,ybound2): #allows (i+1,j+1)
			output[i,i-1-I] += -max(d12,0)/2
		if ismember_sorted(i,xbound1) and ismember_sorted(i,ybound2): #allows (i+1,j-1)
			output[i,i-1+I] += min(d12,0)/2
		if ismember_sorted(i,ybound1) and ismember_sorted(i,xbound2): #allows (i-1,j+1)
			output[i,i+1-I] += min(d12,0)/2
	return lamb*sparse.csr_matrix(output)

#########################
def u_matrix_explicit(f1_array,f2_array,D11,D22,D12,I,J,dx,dt):
	output = np.zeros((I*J,I*J))
	lamb = dt/(dx**2)
	xbound1 = range(0,I)
	ybound1 = range(0,I*J,I)
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
		output[i,i] += 1/lamb - dx*(abs(f1)+abs(f2)) + abs(d12) - d11 - d22
		#avoid segfaults
		if not ismember_sorted(i,xbound1): #allows (i,j-1)
			output[i,i-I] += (d22-abs(d12))/2 - dx*min(f2,0)
		if not ismember_sorted(i,xbound2): #allows (i,j+1)
			output[i,i+I] += (d22-abs(d12))/2 + dx*max(f2,0)
		if not ismember_sorted(i,ybound1): #allows (i-1,j) 
			output[i,i-1] += (d11-abs(d12))/2 - dx*min(f1,0)
		if not ismember_sorted(i,ybound2): #allows (i+1,j)
			output[i,i+1] += (d11-abs(d12))/2 + dx*max(f1,0)
		if not ismember_sorted(i,xbound1) and not ismember_sorted(i,ybound1): #allows (i-1,j-1)
			output[i,i-1-I] += max(d12,0)/2
		if not ismember_sorted(i,xbound2) and not ismember_sorted(i,ybound2): #allows (i+1,j+1)
			output[i,i+1+I] += max(d12,0)/2
		if not ismember_sorted(i,xbound1) and not ismember_sorted(i,ybound2): #allows (i+1,j-1)
			output[i,i+1-I] += -min(d12,0)/2
		if not ismember_sorted(i,ybound1) and not ismember_sorted(i,xbound2): #allows (i-1,j+1)
			output[i,i-1+I] += -min(d12,0)/2
		#then add boundary conditions
		if ismember_sorted(i,xbound1): #allows (i,j-1)
			output[i,i+I] += (d22-abs(d12))/2 - dx*min(f2,0)
		if ismember_sorted(i,xbound2): #allows (i,j+1)
			output[i,i-I] += (d22-abs(d12))/2 + dx*max(f2,0)
		if ismember_sorted(i,ybound1): #allows (i-1,j) 
			output[i,i+1] += (d11-abs(d12))/2 - dx*min(f1,0)
		if ismember_sorted(i,ybound2): #allows (i+1,j)
			output[i,i-1] += (d11-abs(d12))/2 + dx*max(f1,0)
		if ismember_sorted(i,xbound1) and ismember_sorted(i,ybound1): #allows (i-1,j-1)
			output[i,i+1+I] += max(d12,0)/2
		if ismember_sorted(i,xbound2) and ismember_sorted(i,ybound2): #allows (i+1,j+1)
			output[i,i-1-I] += max(d12,0)/2
		if ismember_sorted(i,xbound1) and ismember_sorted(i,ybound2): #allows (i+1,j-1)
			output[i,i-1+I] += -min(d12,0)/2
		if ismember_sorted(i,ybound1) and ismember_sorted(i,xbound2): #allows (i-1,j+1)
			output[i,i+1-I] += -min(d12,0)/2
	return lamb*sparse.csr_matrix(output)

#####################
def m_matrix_iioe(f1_array,f2_array,D11,D22,D12,I,J,dx,dt):
	LHS = np.zeros((I*J,I*J)) #for the implicit stuff
	RHS = np.zeros((I*J,I*J)) #for the explicit stuff
	lamb = dt/(dx**2)
	h = dx
	xbound1 = range(0,I)
	ybound1 = range(0,I*J,I)
	xbound2 = range(I*J-I,I*J)
	ybound2 = range(I-1,I*J,I)
	#flatten out D12, D11, D22, f1_array, f2_array
	D11 = np.ravel(D11)
	D12 = np.ravel(D12)
	D22 = np.ravel(D22)
	f1 = np.ravel(f1_array)
	f2 = np.ravel(f2_array)
	for i in range (0,I*J):
		LHS[i,i] += 1/lamb
		RHS[i,i] += 1/lamb
		if not ismember_sorted(i,xbound1) and not ismember_sorted(i,xbound2) and not ismember_sorted(i,ybound1) and not ismember_sorted(i,ybound2):
			#RHS
			RHS[i,i] += 0.125*( D12[i+1] + D12[i-1] + D12[i+I] + D12[i-I] )
			RHS[i,i+1-I] += -0.125*( D12[i+1] + D12[i-I] )
			RHS[i,i-1+I] += -0.125*( D12[i-1] + D12[i+I] )
			#LHS convection
			north = 0.125*(4*(D22[i+I]-D22[i]) - 4*dx*(f2[i]+f2[i+I]) + (D12[i+1+I]+D12[i+I]-D12[i-I]-D12[i-1-I]) ) #okay ito signs
			south = 0.125*(4*(D22[i-I]-D22[i]) - 4*dx*(f2[i]+f2[i-I]) - (D12[i+1-I]+D12[i+1]-D12[i-1]-D12[i-1-I]) ) #okay ito signs
			west = 0.125*(4*(D11[i-1]-D11[i]) - 4*dx*(f1[i]+f1[i-1]) - (D12[i+I]+D12[i+I-1]-D12[i-I]-D12[i-I-1])) #okay ito signs
			east = 0.125*(4*(D11[i+1]-D11[i]) - 4*dx*(f1[i]+f1[i+1] - (D12[i+I]+D12[i+I+1]-D12[i-I]-D12[i-I+1]))) #okay ito signs
			LHS[i,i] += -(max(0,east)+max(0,north)-min(0,west)-min(0,south))
			LHS[i,i+1] += -min(0,east)
			LHS[i,i-1] += max(0,west)
			LHS[i,i+I] += -min(0,north)
			LHS[i,i-I] += max(0,south)
			#LHS diffusion
			pe = app.hmean(D11[i],D11[i+1])/2
			pn = app.hmean(D22[i],D22[i+I])/2
			pw = app.hmean(D11[i],D11[i-1])/2
			ps = app.hmean(D22[i],D22[i-I])/2
			LHS[i,i] += (pe+pn+pw+ps) + 0.125*(D12[i+1]+D12[i-1]+D12[i+I]+D12[i-I])
			LHS[i,i+1] += -pe
			LHS[i,i+I] += -pn
			LHS[i,i-1] += -pw
			LHS[i,i-I] += -ps
			LHS[i,i+I+1] += -0.125*(D12[i+1]+D12[i+I])
			LHS[i,i-I-1] += -0.125*(D12[i-1]+D12[i-I])
		elif ismember_sorted(i,xbound1) and ismember_sorted(i,ybound1): #SOUTH-WEST
			#RHS
			RHS[i,i] += 0.125*( D12[i+1] + D12[i+I])
			#LHS convection
			north = 0.125*(4*(D22[i+I]-D22[i]) - 4*dx*(f2[i]+f2[i+I]) + (D12[i+1+I]+D12[i+I]-D12[i-I]-D12[i-1-I]) ) #okay ito signs
			east = 0.125*(4*(D11[i+1]-D11[i]) - 4*dx*(f1[i]+f1[i+1] - (D12[i+I]+D12[i+I+1]-D12[i-I]-D12[i-I+1]))) #okay ito signs
			LHS[i,i] += -(max(0,east)+max(0,north))
			LHS[i,i+1] += -min(0,east)
			LHS[i,i+I] += -min(0,north)
			#LHS diffusion
			pe = app.hmean(D11[i],D11[i+1])/2
			pn = app.hmean(D22[i],D22[i+I])/2
			LHS[i,i] += (pe+pn) + 0.125*(D12[i+1]+D12[i+I])
			LHS[i,i+1] += -pe
			LHS[i,i+I] += -pn
			LHS[i,i+I+1] += -0.125*(D12[i+1]+D12[i+I])
		elif ismember_sorted(i,xbound2) and ismember_sorted(i,ybound2): #NORTH-EAST
			#RHS
			RHS[i,i] += 0.125*(D12[i-1] + D12[i-I] )
			#LHS convection
			south = 0.125*(4*(D22[i-I]-D22[i]) - 4*dx*(f2[i]+f2[i-I]) - (D12[i+1-I]+D12[i+1]-D12[i-1]-D12[i-1-I]) ) #okay ito signs
			west = 0.125*(4*(D11[i-1]-D11[i]) - 4*dx*(f1[i]+f1[i-1]) - (D12[i+I]+D12[i+I-1]-D12[i-I]-D12[i-I-1])) #okay ito signs
			LHS[i,i] += -(-min(0,west)-min(0,south))
			LHS[i,i-1] += max(0,west)
			LHS[i,i-I] += max(0,south)
			#LHS diffusion
			pw = app.hmean(D11[i],D11[i-1])/2
			ps = app.hmean(D22[i],D22[i-I])/2
			LHS[i,i] += (pw+ps) + 0.125*(D12[i-1]+D12[i-I])
			LHS[i,i-1] += -pw
			LHS[i,i-I] += -ps
			LHS[i,i-I-1] += -0.125*(D12[i-1]+D12[i-I])
		elif ismember_sorted(i,xbound1) and ismember_sorted(i,ybound2): #SOUTH-EAST
			#RHS
			RHS[i,i] += 0.125*( D12[i-1] + D12[i+I] )
			#LHS convection
			north = 0.125*(4*(D22[i+I]-D22[i]) - 4*dx*(f2[i]+f2[i+I]) + (D12[i+1+I]+D12[i+I]-D12[i-I]-D12[i-1-I]) ) #okay ito signs
			west = 0.125*(4*(D11[i-1]-D11[i]) - 4*dx*(f1[i]+f1[i-1]) - (D12[i+I]+D12[i+I-1]-D12[i-I]-D12[i-I-1])) #okay ito signs
			LHS[i,i] += -(max(0,north)-min(0,west))
			LHS[i,i-1] += max(0,west)
			LHS[i,i+I] += -min(0,north)
			#LHS diffusion
			pn = app.hmean(D22[i],D22[i+I])/2
			pw = app.hmean(D11[i],D11[i-1])/2
			LHS[i,i] += (pn+pw) + 0.125*(D12[i-1]+D12[i+I])
			LHS[i,i+I] += -pn
			LHS[i,i-1] += -pw
		elif ismember_sorted(i,xbound2) and ismember_sorted(i,ybound1): #NORTH-WEST
			#RHS
			RHS[i,i] += 0.125*( D12[i+1] + D12[i-I] )
			#LHS convection
			south = 0.125*(4*(D22[i-I]-D22[i]) - 4*dx*(f2[i]+f2[i-I]) - (D12[i+1-I]+D12[i+1]-D12[i-1]-D12[i-1-I]) ) #okay ito signs
			east = 0.125*(4*(D11[i+1]-D11[i]) - 4*dx*(f1[i]+f1[i+1] - (D12[i+I]+D12[i+I+1]-D12[i-I]-D12[i-I+1]))) #okay ito signs
			LHS[i,i] += -(max(0,east)-min(0,south))
			LHS[i,i+1] += -min(0,east)
			LHS[i,i-I] += max(0,south)
			#LHS diffusion
			pe = app.hmean(D11[i],D11[i+1])/2
			ps = app.hmean(D22[i],D22[i-I])/2
			LHS[i,i] += (pe+ps) + 0.125*(D12[i+1]+D12[i-I])
			LHS[i,i+1] += -pe
			LHS[i,i-I] += -ps
		elif ismember_sorted(i,xbound1): #SOUTH
			#RHS
			RHS[i,i] += 0.125*( D12[i+1] + D12[i-1] + D12[i+I])
			RHS[i,i-1+I] += -0.125*( D12[i-1] + D12[i+I] )
			#LHS convection
			north = 0.125*(4*(D22[i+I]-D22[i]) - 4*dx*(f2[i]+f2[i+I]) + (D12[i+1+I]+D12[i+I]-D12[i-I]-D12[i-1-I]) ) #okay ito signs
			west = 0.125*(4*(D11[i-1]-D11[i]) - 4*dx*(f1[i]+f1[i-1]) - (D12[i+I]+D12[i+I-1]-D12[i-I]-D12[i-I-1])) #okay ito signs
			east = 0.125*(4*(D11[i+1]-D11[i]) - 4*dx*(f1[i]+f1[i+1] - (D12[i+I]+D12[i+I+1]-D12[i-I]-D12[i-I+1]))) #okay ito signs
			LHS[i,i] += -(max(0,east)+max(0,north)-min(0,west))
			LHS[i,i+1] += -min(0,east)
			LHS[i,i-1] += max(0,west)
			LHS[i,i+I] += -min(0,north)
			#LHS diffusion
			pe = app.hmean(D11[i],D11[i+1])/2
			pn = app.hmean(D22[i],D22[i+I])/2
			pw = app.hmean(D11[i],D11[i-1])/2
			LHS[i,i] += (pe+pn+pw) + 0.125*(D12[i+1]+D12[i-1]+D12[i+I]+D12[i-I])
			LHS[i,i+1] += -pe
			LHS[i,i+I] += -pn
			LHS[i,i-1] += -pw
			LHS[i,i+I+1] += -0.125*(D12[i+1]+D12[i+I])
		elif ismember_sorted(i,xbound2): #NORTH
			#RHS
			RHS[i,i] += 0.125*( D12[i+1] + D12[i-1] + D12[i-I] )
			RHS[i,i+1-I] += -0.125*( D12[i+1] + D12[i-I] )
			#LHS convection
			south = 0.125*(4*(D22[i-I]-D22[i]) - 4*dx*(f2[i]+f2[i-I]) - (D12[i+1-I]+D12[i+1]-D12[i-1]-D12[i-1-I]) ) #okay ito signs
			west = 0.125*(4*(D11[i-1]-D11[i]) - 4*dx*(f1[i]+f1[i-1]) - (D12[i+I]+D12[i+I-1]-D12[i-I]-D12[i-I-1])) #okay ito signs
			east = 0.125*(4*(D11[i+1]-D11[i]) - 4*dx*(f1[i]+f1[i+1] - (D12[i+I]+D12[i+I+1]-D12[i-I]-D12[i-I+1]))) #okay ito signs
			LHS[i,i] += -(max(0,east)-min(0,west)-min(0,south))
			LHS[i,i+1] += -min(0,east)
			LHS[i,i-1] += max(0,west)
			LHS[i,i-I] += max(0,south)
			#LHS diffusion
			pe = app.hmean(D11[i],D11[i+1])/2
			pw = app.hmean(D11[i],D11[i-1])/2
			ps = app.hmean(D22[i],D22[i-I])/2
			LHS[i,i] += (pe+pw+ps) + 0.125*(D12[i+1]+D12[i-1]+D12[i-I])
			LHS[i,i+1] += -pe
			LHS[i,i-1] += -pw
			LHS[i,i-I] += -ps
			LHS[i,i-I-1] += -0.125*(D12[i-1]+D12[i-I])
		elif ismember_sorted(i,ybound1): #WEST
			#RHS
			RHS[i,i] += 0.125*( D12[i+1] + D12[i+I] + D12[i-I] )
			RHS[i,i+1-I] += -0.125*( D12[i+1] + D12[i-I] )
			#LHS convection
			north = 0.125*(4*(D22[i+I]-D22[i]) - 4*dx*(f2[i]+f2[i+I]) + (D12[i+1+I]+D12[i+I]-D12[i-I]-D12[i-1-I]) ) #okay ito signs
			south = 0.125*(4*(D22[i-I]-D22[i]) - 4*dx*(f2[i]+f2[i-I]) - (D12[i+1-I]+D12[i+1]-D12[i-1]-D12[i-1-I]) ) #okay ito signs
			east = 0.125*(4*(D11[i+1]-D11[i]) - 4*dx*(f1[i]+f1[i+1] - (D12[i+I]+D12[i+I+1]-D12[i-I]-D12[i-I+1]))) #okay ito signs
			LHS[i,i] += -(max(0,east)+max(0,north)-min(0,south))
			LHS[i,i+1] += -min(0,east)
			LHS[i,i+I] += -min(0,north)
			LHS[i,i-I] += max(0,south)
			#LHS diffusion
			pe = app.hmean(D11[i],D11[i+1])/2
			pn = app.hmean(D22[i],D22[i+I])/2
			ps = app.hmean(D22[i],D22[i-I])/2
			LHS[i,i] += (pe+pn+ps) + 0.125*(D12[i+1]+D12[i+I]+D12[i-I])
			LHS[i,i+1] += -pe
			LHS[i,i+I] += -pn
			LHS[i,i-I] += -ps
			LHS[i,i-I-1] += -0.125*(D12[i-I])
		elif ismember_sorted(i,ybound2): #EAST
			#RHS
			RHS[i,i] += 0.125*( D12[i+1] + D12[i-1] + D12[i-I] )
			RHS[i,i-1+I] += -0.125*( D12[i-1] + D12[i+I] )
			#LHS convection
			north = 0.125*(4*(D22[i+I]-D22[i]) - 4*dx*(f2[i]+f2[i+I]) + (D12[i+1+I]+D12[i+I]-D12[i-I]-D12[i-1-I]) ) #okay ito signs
			south = 0.125*(4*(D22[i-I]-D22[i]) - 4*dx*(f2[i]+f2[i-I]) - (D12[i+1-I]+D12[i+1]-D12[i-1]-D12[i-1-I]) ) #okay ito signs
			west = 0.125*(4*(D11[i-1]-D11[i]) - 4*dx*(f1[i]+f1[i-1]) - (D12[i+I]+D12[i+I-1]-D12[i-I]-D12[i-I-1])) #okay ito signs
			LHS[i,i] += -(max(0,north)-min(0,west)-min(0,south))
			LHS[i,i-1] += max(0,west)
			LHS[i,i+I] += -min(0,north)
			LHS[i,i-I] += max(0,south)
			#LHS diffusion
			pn = app.hmean(D22[i],D22[i+I])/2
			pw = app.hmean(D11[i],D11[i-1])/2
			ps = app.hmean(D22[i],D22[i-I])/2
			LHS[i,i] += (pn+pw+ps) + 0.125*(D12[i-1]+D12[i+I]+D12[i-I])
			LHS[i,i+I] += -pn
			LHS[i,i-1] += -pw
			LHS[i,i-I] += -ps
			LHS[i,i-I-1] += -0.125*(D12[i-1]+D12[i-I])
	return lamb*sparse.csr_matrix(LHS),lamb*sparse.csr_matrix(RHS)
	
def m_matrix_explicit(f1_array,f2_array,D11,D22,D12,I,J,dx,dt): #THIS JUST WORKS
	output = np.zeros((I*J,I*J))
	lamb = dt/(dx**2)
	h = dx
	xbound1 = range(0,I)
	ybound1 = range(0,I*J,I)
	xbound2 = range(I*J-I,I*J)
	ybound2 = range(I-1,I*J,I)
	#flatten out D12, D11, D22, f1_array, f2_array
	D11 = np.ravel(D11)
	D12 = np.ravel(D12)
	D22 = np.ravel(D22)
	f1 = np.ravel(f1_array)
	f2 = np.ravel(f2_array)
	for i in range (0,I*J):
		output[i,i] += 1/lamb
		if not ismember_sorted(i,xbound1) and not ismember_sorted(i,xbound2) and not ismember_sorted(i,ybound1) and not ismember_sorted(i,ybound2):
			#convection terms
			#north = 0.125*(4*(D22[i+I]-D22[i]) - 4*dx*(f2[i]+f2[i+I]) + (D12[i+1+I]+D12[i+I]-D12[i-I]-D12[i-1-I]) ) #okay ito signs
			#south = 0.125*(4*(D22[i-I]-D22[i]) - 4*dx*(f2[i]+f2[i-I]) - (D12[i+1-I]+D12[i+1]-D12[i-1]-D12[i-1-I]) ) #okay ito signs
			#west = 0.125*(4*(D11[i-1]-D11[i]) - 4*dx*(f1[i]+f1[i-1]) - (D12[i+I]+D12[i+I-1]-D12[i-I]-D12[i-I-1])) #okay ito signs
			#east = 0.125*(4*(D11[i+1]-D11[i]) - 4*dx*(f1[i]+f1[i+1] - (D12[i+I]+D12[i+I+1]-D12[i-I]-D12[i-I+1]))) #okay ito signs
			#here1 = 0.5*(2*f1[index]-
			output[i,i] += max(0,east)+max(0,north)-min(0,west)-min(0,south)
			output[i,i+1] += min(0,east)
			output[i,i-1] += -max(0,west)
			output[i,i+I] += min(0,north)
			output[i,i-I] += -max(0,south)
			#diffusion terms
			pe = app.hmean(D11[i],D11[i+1])/2
			pn = app.hmean(D22[i],D22[i+I])/2
			pw = app.hmean(D11[i],D11[i-1])/2
			ps = app.hmean(D22[i],D22[i-I])/2
			output[i,i] += (-pe-pn-pw-ps)
			output[i,i+1] += pe
			output[i,i+I] += pn
			output[i,i-1] += pw
			output[i,i-I] += ps
			#cross diffusion terms #these guys are coefficients for the respective things
			ne = 0.25*(D12[i+1]+D12[i+I])/2
			se = -0.25*(D12[i-I]+D12[i+1])/2
			sw = 0.25*(D12[i-1]+D12[i-I])/2
			nw = -0.25*(D12[i+I]+D12[i-1])/2
			output[i,i+1+I] += ne
			output[i,i+1-I] += se
			output[i,i-I-1] += sw
			output[i,i-1+I] += nw
		elif ismember_sorted(i,xbound1) and ismember_sorted(i,ybound1): #SOUTH-WEST
			#convection terms
			north = 0.125*(4*(D22[i+I]-D22[i]) - 4*dx*(f2[i]+f2[i+I]) + (D12[i+1+I]+D12[i+I]-D12[i-I]-D12[i-1-I]) ) #okay ito signs
			east = 0.125*(4*(D11[i+1]-D11[i]) - 4*dx*(f1[i]+f1[i+1] - (D12[i+I]+D12[i+I+1]-D12[i-I]-D12[i-I+1]))) #okay ito signs
			output[i,i] += max(0,east)+max(0,north)
			output[i,i+1] += min(0,east)
			output[i,i+I] += min(0,north)
			#diffusion terms
			pe = app.hmean(D11[i],D11[i+1])/2
			pn = app.hmean(D22[i],D22[i+I])/2
			output[i,i] += (-pe-pn)
			output[i,i+1] += pe
			output[i,i+I] += pn
			#cross diffusion terms #these guys are coefficients for the respective things
			ne = 0.25*(D12[i+1]+D12[i+I])/2
			output[i,i+1+I] += ne
		elif ismember_sorted(i,xbound2) and ismember_sorted(i,ybound2): #NORTH-EAST
			#convection terms
			south = 0.125*(4*(D22[i-I]-D22[i]) - 4*dx*(f2[i]+f2[i-I]) - (D12[i+1-I]+D12[i+1]-D12[i-1]-D12[i-1-I]) ) #okay ito signs
			west = 0.125*(4*(D11[i-1]-D11[i]) - 4*dx*(f1[i]+f1[i-1]) - (D12[i+I]+D12[i+I-1]-D12[i-I]-D12[i-I-1])) #okay ito signs
			output[i,i] += -min(0,west)-min(0,south)
			output[i,i-1] += -max(0,west)
			output[i,i-I] += -max(0,south)
			#diffusion terms
			pw = app.hmean(D11[i],D11[i-1])/2
			ps = app.hmean(D22[i],D22[i-I])/2
			output[i,i] += (-pw-ps)
			output[i,i-1] += pw
			output[i,i-I] += ps
			#cross diffusion terms #these guys are coefficients for the respective things
			sw = 0.25*(D12[i-1]+D12[i-I])/2
			output[i,i-I-1] += sw
		elif ismember_sorted(i,xbound1) and ismember_sorted(i,ybound2): #SOUTH-EAST
			#convection terms
			north = 0.125*(4*(D22[i+I]-D22[i]) - 4*dx*(f2[i]+f2[i+I]) + (D12[i+1+I]+D12[i+I]-D12[i-I]-D12[i-1-I]) ) #okay ito signs
			west = 0.125*(4*(D11[i-1]-D11[i]) - 4*dx*(f1[i]+f1[i-1]) - (D12[i+I]+D12[i+I-1]-D12[i-I]-D12[i-I-1])) #okay ito signs
			output[i,i] += max(0,north)-min(0,west)
			output[i,i-1] += -max(0,west)
			output[i,i+I] += min(0,north)
			#diffusion terms
			pn = app.hmean(D22[i],D22[i+I])/2
			pw = app.hmean(D11[i],D11[i-1])/2
			output[i,i] += (-pn-pw)
			output[i,i+I] += pn
			output[i,i-1] += pw
			#cross diffusion terms #these guys are coefficients for the respective things
			nw = -0.25*(D12[i+I]+D12[i-1])/2
			output[i,i-1+I] += nw
		elif ismember_sorted(i,xbound2) and ismember_sorted(i,ybound1): #NORTH-WEST
			#convection terms
			south = 0.125*(4*(D22[i-I]-D22[i]) - 4*dx*(f2[i]+f2[i-I]) - (D12[i+1-I]+D12[i+1]-D12[i-1]-D12[i-1-I]) ) #okay ito signs
			east = 0.125*(4*(D11[i+1]-D11[i]) - 4*dx*(f1[i]+f1[i+1] - (D12[i+I]+D12[i+I+1]-D12[i-I]-D12[i-I+1]))) #okay ito signs
			output[i,i] += max(0,east)-min(0,south)
			output[i,i+1] += min(0,east)
			output[i,i-I] += -max(0,south)
			#diffusion terms
			pe = app.hmean(D11[i],D11[i+1])/2
			ps = app.hmean(D22[i],D22[i-I])/2
			output[i,i] += (-pe-pn-pw-ps)
			output[i,i+1] += pe
			output[i,i-I] += ps
			#cross diffusion terms #these guys are coefficients for the respective things
			se = -0.25*(D12[i-I]+D12[i+1])/2
			output[i,i+1-I] += se
		elif ismember_sorted(i,xbound1): #SOUTH
			#convection terms
			north = 0.125*(4*(D22[i+I]-D22[i]) - 4*dx*(f2[i]+f2[i+I]) + (D12[i+1+I]+D12[i+I]-D12[i-I]-D12[i-1-I]) ) #okay ito signs
			west = 0.125*(4*(D11[i-1]-D11[i]) - 4*dx*(f1[i]+f1[i-1]) - (D12[i+I]+D12[i+I-1]-D12[i-I]-D12[i-I-1])) #okay ito signs
			east = 0.125*(4*(D11[i+1]-D11[i]) - 4*dx*(f1[i]+f1[i+1] - (D12[i+I]+D12[i+I+1]-D12[i-I]-D12[i-I+1]))) #okay ito signs
			output[i,i] += max(0,east)+max(0,north)-min(0,west)
			output[i,i+1] += min(0,east)
			output[i,i-1] += -max(0,west)
			output[i,i+I] += min(0,north)
			#diffusion terms
			pe = app.hmean(D11[i],D11[i+1])/2
			pn = app.hmean(D22[i],D22[i+I])/2
			pw = app.hmean(D11[i],D11[i-1])/2
			output[i,i] += (-pe-pn-pw)
			output[i,i+1] += pe
			output[i,i+I] += pn
			output[i,i-1] += pw
			#cross diffusion terms #these guys are coefficients for the respective things
			ne = 0.25*(D12[i+1]+D12[i+I])/2
			nw = -0.25*(D12[i+I]+D12[i-1])/2
			output[i,i+1+I] += ne
			output[i,i-1+I] += nw
		elif ismember_sorted(i,xbound2): #NORTH
			#convection terms
			south = 0.125*(4*(D22[i-I]-D22[i]) - 4*dx*(f2[i]+f2[i-I]) - (D12[i+1-I]+D12[i+1]-D12[i-1]-D12[i-1-I]) ) #okay ito signs
			west = 0.125*(4*(D11[i-1]-D11[i]) - 4*dx*(f1[i]+f1[i-1]) - (D12[i+I]+D12[i+I-1]-D12[i-I]-D12[i-I-1])) #okay ito signs
			east = 0.125*(4*(D11[i+1]-D11[i]) - 4*dx*(f1[i]+f1[i+1] - (D12[i+I]+D12[i+I+1]-D12[i-I]-D12[i-I+1]))) #okay ito signs
			output[i,i] += max(0,east)-min(0,west)-min(0,south)
			output[i,i+1] += min(0,east)
			output[i,i-1] += -max(0,west)
			output[i,i-I] += -max(0,south)
			#diffusion terms
			pe = app.hmean(D11[i],D11[i+1])/2
			pw = app.hmean(D11[i],D11[i-1])/2
			ps = app.hmean(D22[i],D22[i-I])/2
			output[i,i] += (-pe-pw-ps)
			output[i,i+1] += pe
			output[i,i-1] += pw
			output[i,i-I] += ps
			#cross diffusion terms #these guys are coefficients for the respective things
			se = -0.25*(D12[i-I]+D12[i+1])/2
			sw = 0.25*(D12[i-1]+D12[i-I])/2
			output[i,i+1-I] += se
			output[i,i-I-1] += sw
		elif ismember_sorted(i,ybound1): #WEST
			#convection terms
			north = 0.125*(4*(D22[i+I]-D22[i]) - 4*dx*(f2[i]+f2[i+I]) + (D12[i+1+I]+D12[i+I]-D12[i-I]-D12[i-1-I]) ) #okay ito signs
			south = 0.125*(4*(D22[i-I]-D22[i]) - 4*dx*(f2[i]+f2[i-I]) - (D12[i+1-I]+D12[i+1]-D12[i-1]-D12[i-1-I]) ) #okay ito signs
			east = 0.125*(4*(D11[i+1]-D11[i]) - 4*dx*(f1[i]+f1[i+1] - (D12[i+I]+D12[i+I+1]-D12[i-I]-D12[i-I+1]))) #okay ito signs
			output[i,i] += max(0,east)+max(0,north)-min(0,south)
			output[i,i+1] += min(0,east)
			output[i,i+I] += min(0,north)
			output[i,i-I] += -max(0,south)
			#diffusion terms
			pe = app.hmean(D11[i],D11[i+1])/2
			pn = app.hmean(D22[i],D22[i+I])/2
			ps = app.hmean(D22[i],D22[i-I])/2
			output[i,i] += (-pe-pn-pw-ps)
			output[i,i+1] += pe
			output[i,i+I] += pn
			output[i,i-I] += ps
			#cross diffusion terms #these guys are coefficients for the respective things
			ne = 0.25*(D12[i+1]+D12[i+I])/2
			se = -0.25*(D12[i-I]+D12[i+1])/2
			output[i,i+1+I] += ne
			output[i,i+1-I] += se
		elif ismember_sorted(i,ybound2): #EAST
			#convection terms
			north = 0.125*(4*(D22[i+I]-D22[i]) - 4*dx*(f2[i]+f2[i+I]) + (D12[i+1+I]+D12[i+I]-D12[i-I]-D12[i-1-I]) ) #okay ito signs
			south = 0.125*(4*(D22[i-I]-D22[i]) - 4*dx*(f2[i]+f2[i-I]) - (D12[i+1-I]+D12[i+1]-D12[i-1]-D12[i-1-I]) ) #okay ito signs
			west = 0.125*(4*(D11[i-1]-D11[i]) - 4*dx*(f1[i]+f1[i-1]) - (D12[i+I]+D12[i+I-1]-D12[i-I]-D12[i-I-1])) #okay ito signs
			output[i,i] += max(0,north)-min(0,west)-min(0,south)
			output[i,i-1] += -max(0,west)
			output[i,i+I] += min(0,north)
			output[i,i-I] += -max(0,south)
			#diffusion terms
			pn = app.hmean(D22[i],D22[i+I])/2
			pw = app.hmean(D11[i],D11[i-1])/2
			ps = app.hmean(D22[i],D22[i-I])/2
			output[i,i] += (-ps-pn-pw)
			output[i,i-I] += ps
			output[i,i+I] += pn
			output[i,i-1] += pw
			#cross diffusion terms #these guys are coefficients for the respective things
			sw = 0.25*(D12[i-1]+D12[i-I])/2
			nw = -0.25*(D12[i+I]+D12[i-1])/2
			output[i,i-I-1] += sw
			output[i,i-1+I] += nw
			#Quoth The Hound, "Fuck the boundary!"
	return lamb*sparse.csr_matrix(output)

def add_diffusion_flux_Ometh(output,D11,D22,D12,I,J,dx,dt): #there is no quick-fix to fixing the signs
	h = dx
	dx2 = h**2
	xbound1 = range(0,I)
	ybound1 = range(0,I*J,I)
	xbound2 = range(I*J-I,I*J)
	ybound2 = range(I-1,I*J,I)
	D11 = np.ravel(D11)
	D12 = np.ravel(D12)
	D22 = np.ravel(D22)
	for i in range(0,I*J-I-1):
		#make matrices
		a1,a2,a3,a4 = D11[i],D11[i+1],D11[i+I],D11[i+I+1]
		b1,b2,b3,b4 = D22[i],D22[i+1],D22[i+I],D22[i+I+1]
		c1,c2,c3,c4 = D12[i],D12[i+1],D12[i+I],D12[i+I+1]
		if ismember_sorted(i,xbound1): #south
			a1,b1,c1 = a1*2,b1*2,c1*2
			a2,b2,c2 = a2*2,b2*2,c2*2
		if ismember_sorted(i,ybound1): #west
			a1,b1,c1 = a1*2,b1*2,c1*2
			a3,b3,c3 = a3*2,b3*2,c3*2
		if ismember_sorted(i+I,xbound2): #north
			a3,b3,c3 = a3*2,b3*2,c3*2
			a4,b4,c4 = a4*2,b4*2,c4*2
		if ismember_sorted(i+1,ybound2): #east
			a2,b2,c2 = a2*2,b2*2,c2*2
			a4,b4,c4 = a4*2,b4*2,c4*2
		#as we believe it to be the diffusion tensor equation
		A = np.array([[a1+a2,0,c1,-c2],[0,a3+a4,-c3,c4],[c1,-c3,b1+b3,0],[-c2,c4,0,b2+b4]])
		B = np.array([[a1+c2,a2-c2,0,0],[0,0,a3-c3,a4+c4],[b1+c1,0,b3-c3,0],[0,b2-c2,0,b4+c4]])
		C = -np.array([[-a1,0,-c1,0],[0,a4,0,c4],[0,-c3,b3,0],[c2,0,0,-b2]])
		F = -np.array([[a1+c1,0,0,0],[0	,0,0,-a4-c4],[0,0,c3-b3,0],[0,-c2+b2,0,0]])
		#finish up
		tmp = np.dot(np.linalg.inv(A),B)
		T = np.dot(C,tmp)+F #transmission coefficient matrix
		R = np.array([[1,0,1,0],[-1,0,0,1],[0,1,-1,0],[0,-1,0,-1]]) #the contribution mapping extravaganza
#		if ismember_sorted(i,xbound1): #SOUTH
#			R += np.array([[],[],[],[]])
#		if ismember_sorted(i,ybound1): #WEST
#			R += np.array([[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]])
#		if ismember_sorted(i,xbound2): #NORTH
#			R += np.array([[],[],[],[]])
#		if ismember_sorted(i,ybound2): #EAST
#			R += np.array([[],[],[],[]])


		output = ass.FVL2G(np.dot(R,T),output,i,I,J)
	return output

def add_diffusion_flux_DIAMOND(output,D11,D22,D12,I,J,dx,dt): #boundary is no bueno
	#output = np.zeros((I*J,I*J))
	h = dx
	xbound1 = range(0,I)
	ybound1 = range(0,I*J,I)
	xbound2 = range(I*J-I,I*J)
	ybound2 = range(I-1,I*J,I)
	#flatten out D12, D11, D22, f1_array, f2_array
	D11 = np.ravel(D11)
	D12 = np.ravel(D12)
	D22 = np.ravel(D22)
	for i in range (0,I*J):
		if not ismember_sorted(i,xbound1) and not ismember_sorted(i,xbound2) and not ismember_sorted(i,ybound1) and not ismember_sorted(i,ybound2):
			#diffusion terms
			pe = app.hmean(D11[i],D11[i+1])/2
			pn = app.hmean(D22[i],D22[i+I])/2
			pw = app.hmean(D11[i],D11[i-1])/2
			ps = app.hmean(D22[i],D22[i-I])/2
			output[i,i] += (-pe-pn-pw-ps)
			output[i,i+1] += pe
			output[i,i+I] += pn
			output[i,i-1] += pw
			output[i,i-I] += ps
			#cross diffusion terms #these guys are coefficients for the respective things
			ne = 0.25*(D12[i+1]+D12[i+I])/2
			se = -0.25*(D12[i-I]+D12[i+1])/2
			sw = 0.25*(D12[i-1]+D12[i-I])/2
			nw = -0.25*(D12[i+I]+D12[i-1])/2
			output[i,i+1+I] += ne
			output[i,i+1-I] += se
			output[i,i-I-1] += sw
			output[i,i-1+I] += nw
		elif ismember_sorted(i,xbound1) and ismember_sorted(i,ybound1): #SOUTH-WEST
			#diffusion terms
			pe = app.hmean(D11[i],D11[i+1])/2
			pn = app.hmean(D22[i],D22[i+I])/2
			output[i,i] += (-pe-pn)
			output[i,i+1] += pe
			output[i,i+I] += pn
			#cross diffusion terms #these guys are coefficients for the respective things
			ne = 0.25*(D12[i+1]+D12[i+I])/2
			output[i,i+1+I] += ne
		elif ismember_sorted(i,xbound2) and ismember_sorted(i,ybound2): #NORTH-EAST
			#diffusion terms
			pw = app.hmean(D11[i],D11[i-1])/2
			ps = app.hmean(D22[i],D22[i-I])/2
			output[i,i] += (-pw-ps)
			output[i,i-1] += pw
			output[i,i-I] += ps
			#cross diffusion terms #these guys are coefficients for the respective things
			sw = 0.25*(D12[i-1]+D12[i-I])/2
			output[i,i-I-1] += sw
		elif ismember_sorted(i,xbound1) and ismember_sorted(i,ybound2): #SOUTH-EAST
			#diffusion terms
			pn = app.hmean(D22[i],D22[i+I])/2
			pw = app.hmean(D11[i],D11[i-1])/2
			output[i,i] += (-pn-pw)
			output[i,i+I] += pn
			output[i,i-1] += pw
			#cross diffusion terms #these guys are coefficients for the respective things
			nw = -0.25*(D12[i+I]+D12[i-1])/2
			output[i,i-1+I] += nw
		elif ismember_sorted(i,xbound2) and ismember_sorted(i,ybound1): #NORTH-WEST
			#diffusion terms
			pe = app.hmean(D11[i],D11[i+1])/2
			ps = app.hmean(D22[i],D22[i-I])/2
			output[i,i] += (-pe-pn-pw-ps)
			output[i,i+1] += pe
			output[i,i-I] += ps
			#cross diffusion terms #these guys are coefficients for the respective things
			se = -0.25*(D12[i-I]+D12[i+1])/2
			output[i,i+1-I] += se
		elif ismember_sorted(i,xbound1): #SOUTH
			#diffusion terms
			pe = app.hmean(D11[i],D11[i+1])/2
			pn = app.hmean(D22[i],D22[i+I])/2
			pw = app.hmean(D11[i],D11[i-1])/2
			output[i,i] += (-pe-pn-pw)
			output[i,i+1] += pe
			output[i,i+I] += pn
			output[i,i-1] += pw
			#cross diffusion terms #these guys are coefficients for the respective things
			ne = 0.25*(D12[i+1]+D12[i+I])/2
			nw = -0.25*(D12[i+I]+D12[i-1])/2
			output[i,i+1+I] += ne
			output[i,i-1+I] += nw
		elif ismember_sorted(i,xbound2): #NORTH
			#diffusion terms
			pe = app.hmean(D11[i],D11[i+1])/2
			pw = app.hmean(D11[i],D11[i-1])/2
			ps = app.hmean(D22[i],D22[i-I])/2
			output[i,i] += (-pe-pw-ps)
			output[i,i+1] += pe
			output[i,i-1] += pw
			output[i,i-I] += ps
			#cross diffusion terms #these guys are coefficients for the respective things
			se = -0.25*(D12[i-I]+D12[i+1])/2
			sw = 0.25*(D12[i-1]+D12[i-I])/2
			output[i,i+1-I] += se
			output[i,i-I-1] += sw
		elif ismember_sorted(i,ybound1): #WEST
			#diffusion terms
			pe = app.hmean(D11[i],D11[i+1])/2
			pn = app.hmean(D22[i],D22[i+I])/2
			ps = app.hmean(D22[i],D22[i-I])/2
			output[i,i] += (-pe-pn-pw-ps)
			output[i,i+1] += pe
			output[i,i+I] += pn
			output[i,i-I] += ps
			#cross diffusion terms #these guys are coefficients for the respective things
			ne = 0.25*(D12[i+1]+D12[i+I])/2
			se = -0.25*(D12[i-I]+D12[i+1])/2
			output[i,i+1+I] += ne
			output[i,i+1-I] += se
		elif ismember_sorted(i,ybound2): #EAST
			#diffusion terms
			pn = app.hmean(D22[i],D22[i+I])/2
			pw = app.hmean(D11[i],D11[i-1])/2
			ps = app.hmean(D22[i],D22[i-I])/2
			output[i,i] += (-ps-pn-pw)
			output[i,i-I] += ps
			output[i,i+I] += pn
			output[i,i-1] += pw
			#cross diffusion terms #these guys are coefficients for the respective things
			sw = 0.25*(D12[i-1]+D12[i-I])/2
			nw = -0.25*(D12[i+I]+D12[i-1])/2
			output[i,i-I-1] += sw
			output[i,i-1+I] += nw
			#Quoth The Hound, "Fuck the boundary!"
	return sparse.csr_matrix(output)

def diffusion_flux_iioe(D11,D22,D12,I,J,dx,dt):
	LHS = np.zeros((I*J,I*J)) #for the implicit stuff
	RHS = np.zeros((I*J,I*J)) #for the explicit stuff
	lamb = dt/(dx**2)
	h = dx
	xbound1 = range(0,I)
	ybound1 = range(0,I*J,I)
	xbound2 = range(I*J-I,I*J)
	ybound2 = range(I-1,I*J,I)
	#flatten out D12, D11, D22, f1_array, f2_array
	D11 = np.ravel(D11)
	D12 = np.ravel(D12)
	D22 = np.ravel(D22)
	for i in range (0,I*J):
		LHS[i,i] += 1/lamb
		RHS[i,i] += 1/lamb
		if not ismember_sorted(i,xbound1) and not ismember_sorted(i,xbound2) and not ismember_sorted(i,ybound1) and not ismember_sorted(i,ybound2):
			#RHS
			RHS[i,i] += 0.125*( D12[i+1] + D12[i-1] + D12[i+I] + D12[i-I] )
			RHS[i,i+1-I] += -0.125*( D12[i+1] + D12[i-I] )
			RHS[i,i-1+I] += -0.125*( D12[i-1] + D12[i+I] )
			#LHS diffusion
			pe = app.hmean(D11[i],D11[i+1])/2
			pn = app.hmean(D22[i],D22[i+I])/2
			pw = app.hmean(D11[i],D11[i-1])/2
			ps = app.hmean(D22[i],D22[i-I])/2
			LHS[i,i] += (pe+pn+pw+ps) + 0.125*(D12[i+1]+D12[i-1]+D12[i+I]+D12[i-I])
			LHS[i,i+1] += -pe
			LHS[i,i+I] += -pn
			LHS[i,i-1] += -pw
			LHS[i,i-I] += -ps
			LHS[i,i+I+1] += -0.125*(D12[i+1]+D12[i+I])
			LHS[i,i-I-1] += -0.125*(D12[i-1]+D12[i-I])
		elif ismember_sorted(i,xbound1) and ismember_sorted(i,ybound1): #SOUTH-WEST
			#RHS
			RHS[i,i] += 0.125*( D12[i+1] + D12[i+I])
			#LHS diffusion
			pe = app.hmean(D11[i],D11[i+1])/2
			pn = app.hmean(D22[i],D22[i+I])/2
			LHS[i,i] += (pe+pn) + 0.125*(D12[i+1]+D12[i+I])
			LHS[i,i+1] += -pe
			LHS[i,i+I] += -pn
			LHS[i,i+I+1] += -0.125*(D12[i+1]+D12[i+I])
		elif ismember_sorted(i,xbound2) and ismember_sorted(i,ybound2): #NORTH-EAST
			#RHS
			RHS[i,i] += 0.125*(D12[i-1] + D12[i-I] )
			#LHS diffusion
			pw = app.hmean(D11[i],D11[i-1])/2
			ps = app.hmean(D22[i],D22[i-I])/2
			LHS[i,i] += (pw+ps) + 0.125*(D12[i-1]+D12[i-I])
			LHS[i,i-1] += -pw
			LHS[i,i-I] += -ps
			LHS[i,i-I-1] += -0.125*(D12[i-1]+D12[i-I])
		elif ismember_sorted(i,xbound1) and ismember_sorted(i,ybound2): #SOUTH-EAST
			#RHS
			RHS[i,i] += 0.125*( D12[i-1] + D12[i+I] )
			#LHS diffusion
			pn = app.hmean(D22[i],D22[i+I])/2
			pw = app.hmean(D11[i],D11[i-1])/2
			LHS[i,i] += (pn+pw) + 0.125*(D12[i-1]+D12[i+I])
			LHS[i,i+I] += -pn
			LHS[i,i-1] += -pw
		elif ismember_sorted(i,xbound2) and ismember_sorted(i,ybound1): #NORTH-WEST
			#RHS
			RHS[i,i] += 0.125*( D12[i+1] + D12[i-I] )
			#LHS diffusion
			pe = app.hmean(D11[i],D11[i+1])/2
			ps = app.hmean(D22[i],D22[i-I])/2
			LHS[i,i] += (pe+ps) + 0.125*(D12[i+1]+D12[i-I])
			LHS[i,i+1] += -pe
			LHS[i,i-I] += -ps
		elif ismember_sorted(i,xbound1): #SOUTH
			#RHS
			RHS[i,i] += 0.125*( D12[i+1] + D12[i-1] + D12[i+I])
			RHS[i,i-1+I] += -0.125*( D12[i-1] + D12[i+I] )
			#LHS diffusion
			pe = app.hmean(D11[i],D11[i+1])/2
			pn = app.hmean(D22[i],D22[i+I])/2
			pw = app.hmean(D11[i],D11[i-1])/2
			LHS[i,i] += (pe+pn+pw) + 0.125*(D12[i+1]+D12[i-1]+D12[i+I]+D12[i-I])
			LHS[i,i+1] += -pe
			LHS[i,i+I] += -pn
			LHS[i,i-1] += -pw
			LHS[i,i+I+1] += -0.125*(D12[i+1]+D12[i+I])
		elif ismember_sorted(i,xbound2): #NORTH
			#RHS
			RHS[i,i] += 0.125*( D12[i+1] + D12[i-1] + D12[i-I] )
			RHS[i,i+1-I] += -0.125*( D12[i+1] + D12[i-I] )
			#LHS diffusion
			pe = app.hmean(D11[i],D11[i+1])/2
			pw = app.hmean(D11[i],D11[i-1])/2
			ps = app.hmean(D22[i],D22[i-I])/2
			LHS[i,i] += (pe+pw+ps) + 0.125*(D12[i+1]+D12[i-1]+D12[i-I])
			LHS[i,i+1] += -pe
			LHS[i,i-1] += -pw
			LHS[i,i-I] += -ps
			LHS[i,i-I-1] += -0.125*(D12[i-1]+D12[i-I])
		elif ismember_sorted(i,ybound1): #WEST
			#RHS
			RHS[i,i] += 0.125*( D12[i+1] + D12[i+I] + D12[i-I] )
			RHS[i,i+1-I] += -0.125*( D12[i+1] + D12[i-I] )
			#LHS diffusion
			pe = app.hmean(D11[i],D11[i+1])/2
			pn = app.hmean(D22[i],D22[i+I])/2
			ps = app.hmean(D22[i],D22[i-I])/2
			LHS[i,i] += (pe+pn+ps) + 0.125*(D12[i+1]+D12[i+I]+D12[i-I])
			LHS[i,i+1] += -pe
			LHS[i,i+I] += -pn
			LHS[i,i-I] += -ps
			LHS[i,i-I-1] += -0.125*(D12[i-I])
		elif ismember_sorted(i,ybound2): #EAST
			#RHS
			RHS[i,i] += 0.125*( D12[i+1] + D12[i-1] + D12[i-I] )
			RHS[i,i-1+I] += -0.125*( D12[i-1] + D12[i+I] )
			#LHS diffusion
			pn = app.hmean(D22[i],D22[i+I])/2
			pw = app.hmean(D11[i],D11[i-1])/2
			ps = app.hmean(D22[i],D22[i-I])/2
			LHS[i,i] += (pn+pw+ps) + 0.125*(D12[i-1]+D12[i+I]+D12[i-I])
			LHS[i,i+I] += -pn
			LHS[i,i-1] += -pw
			LHS[i,i-I] += -ps
			LHS[i,i-I-1] += -0.125*(D12[i-1]+D12[i-I])
	return lamb*sparse.csr_matrix(LHS),lamb*sparse.csr_matrix(RHS)

def add_direchlet_boundary(output,sol_vector,I,J,lamb,val):
	xbound1 = range(0,I)
	ybound1 = range(0,I*J,I)
	xbound2 = range(I*J-I,I*J)
	ybound2 = range(I-1,I*J,I)
	for i in range(0,I*J):
		if ismember_sorted(i,xbound1) or ismember_sorted(i,ybound1) or ismember_sorted(i,xbound2) or ismember_sorted(i,ybound2):
			#set output to exactly one
			output[i,:] = unit(i,I*J)/lamb
			sol_vector[i] = val
	return output,sol_vector

def unit(index,length):
	output = np.zeros(length)
	output[index] = 1
	return output

def ismember_sorted(a,array):
	for i in range(0,len(array)):
		if array[i]==a:
			return True
		elif array[i]>a:
			return False
	return False


#if not ismember_sorted(i,xbound1) and not ismember_sorted(i,xbound2) and not ismember_sorted(i,ybound1) and not ismember_sorted(i,ybound2):
#elif ismember_sorted(i,xbound1) and ismember_sorted(i,ybound1): #SOUTH-WEST
#elif ismember_sorted(i,xbound2) and ismember_sorted(i,ybound2): #NORTH-EAST
#elif ismember_sorted(i,xbound1) and ismember_sorted(i,ybound2): #SOUTH-EAST
#elif ismember_sorted(i,xbound2) and ismember_sorted(i,ybound1): #NORTH-WEST
#elif ismember_sorted(i,xbound1): #SOUTH
#elif ismember_sorted(i,xbound2): #NORTH
#elif ismember_sorted(i,ybound1): #WEST	
#elif ismember_sorted(i,ybound2): #EAST		
