from __future__ import division
import numpy as np
from scipy import sparse
import applications as app
import assembly as ass
import input_functions as iF
##################
# MATRIX GENERATION: HJB
##################
def hjb_convection(time,x,a_tmp,m_tmp,dt,dx): #for explicit
	#functions
	movement = iF.f_global(time,x,a_tmp) #the function f
	I = x.size
	zerohero = np.zeros(I)
	m_east = np.maximum(movement,zerohero)*dt/dx
	m_east[0] = abs(movement[0])*dt/dx #reflective
	m_west = -np.minimum(movement,zerohero)*dt/dx
	m_west[-1] = abs(movement[-1])*dt/dx #reflective
	here = 1-dt/dx*abs(movement)
	output = sparse.diags([here, m_east[0:-1], m_west[1:]],[0, 1, -1])
	return sparse.csr_matrix(output)

def hjb_diffusion(time,x,a_tmp,m_tmp,dt,dx): #for implicit
	#functions
	sigma = iF.Sigma_global(time,x,a_tmp,m_tmp)
	sigma2 = sigma**2
	dx2 = dx**2
	I = x.size
	s_east = -sigma2*dt/(2*dx2)
	s_east[0] = 2*s_east[0] #reflective
	s_west = -sigma2*dt/(2*dx2)
	s_west[-1] = 2*s_west[-1] #reflective
	here = 1+sigma2*dt/dx2
	output = sparse.diags([here, s_east[0:-1], s_west[1:]],[0, 1, -1])
	return sparse.csr_matrix(output)
##################
# MATRIX GENERATION: FINITE VOLUME
##################
def fp_fv_convection(time,x,a_tmp,m_tmp,dt,dx): #for explicit
	#functions
	sigma = iF.Sigma_global(time,x,a_tmp,m_tmp)
	movement = iF.f_global(time,x,a_tmp) #the function f
	#generate the flux vectors
	Fi = (dx*movement[1:-1]-sigma[1:-1]*( sigma[2:]-sigma[0:-2] ))/(dx)
	Fi = np.append(Fi,(dx*movement[-1])/dx )
	Fi = np.insert(Fi,0,(dx*movement[0])/dx)
	#Fi = np.append(Fi,(dx*movement[-1]-sigma[-1]*(-sigma[-2]))/dx )
	#Fi = np.insert(Fi,0,(dx*movement[0]-sigma[0]*sigma[1])/dx)
	#print Fi
	#print ss
	zerohero = np.zeros(x.size)
	here = 1-dt/dx*abs(Fi)
	east = -dt/dx*np.minimum(Fi[1:],zerohero[1:])
	west = dt/dx*np.maximum(Fi[0:-1],zerohero[0:-1])
	output = sparse.diags([here, east, west],[0, 1, -1])
	return sparse.csr_matrix(output)

def fp_fv_diffusion(time,x,a_tmp,m_tmp,dt,dx): #for implicit
	sigma = iF.Sigma_global(time,x,a_tmp,m_tmp)
	sigma2 = sigma**2
	dx2 = dx**2
	I = x.size
	#generate the flux vectors; this is a little messy
	D_e = iF.hmean_scalar(sigma2[-1],sigma2[-2])#/2
	D_w = iF.hmean_scalar(sigma2[0],sigma2[1])#/2
	D_up = iF.hmean(sigma2[1:-1],sigma2[2:])/2
	D_up = np.append(D_up,0)
	D_up = np.insert(D_up,0,D_e)
	D_down = iF.hmean(sigma2[1:-1],sigma2[0:-2])/2
	D_down = np.append(D_down,D_w)
	D_down = np.insert(D_down,0,0)
	#make the matrix
	here = 1+dt/dx2*(D_up+D_down)
	#print here
	#print ss
	east = -dt/dx2*D_up
	west = -dt/dx2*D_down
	output = sparse.diags([here, east[0:-1], west[1:]],[0, 1, -1])
	return sparse.csr_matrix(output)
############
# MATRIX GENERATION OF THE SHIT
############

def fp_fd_centered_convection(time,x,a_tmp,m_tmp,dt,dx): #for explicit
	#functions
	movement = iF.f_global(time,x,a_tmp) #the function f
	I = x.size
	#the actual computation
	zero = np.zeros(movement[1:-1].size)
	here = np.ones(I)
	east = -dt/dx*movement[1:]/2
	west = dt/dx*movement/2
	output = sparse.diags([here, east, west[0:-1]],[0, 1, -1])
	return sparse.csr_matrix(output)

def fp_fd_centered_diffusion(time,x,a_tmp,m_tmp,dt,dx): #for implicit
	sigma = iF.Sigma_global(time,x,a_tmp,m_tmp)
	sigma2 = sigma**2
	dx2 = dx**2
	I = x.size
	here = 1+dt/dx2*sigma2
	east = -dt/dx2*sigma2[1:]/2
	west = -dt/dx2*sigma2/2
	output = sparse.diags([here, east, west[0:-1]],[0, 1, -1])
	return sparse.csr_matrix(output)

def fp_fd_upwind_convection(time,x,a_tmp,m_tmp,dt,dx): #for explicit
	#functions
	movement = iF.f_global(time,x,a_tmp) #the function f
	I = x.size
	#generate the flux vectors
	zerohero = np.zeros(x.size)
	here = 1-dt/dx*abs(movement) - dt/dx*np.append(movement[1:],0)/2+dt/dx*np.insert(movement[0:-1],0,0)/2
	east = -dt/dx*np.minimum(movement,zerohero)
	west = dt/dx*np.maximum(movement,zerohero)
	output = sparse.diags([here, east[0:-1], west[1:]],[0, 1, -1])
	return sparse.csr_matrix(output)

#helpful stuff
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


