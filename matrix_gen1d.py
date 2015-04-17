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
	output = sparse.lil_matrix((I,I))
	m_east = np.maximum(movement,zerohero)*dt/dx
	m_east[0] = abs(movement[0])*dt/dx #reflective
	m_west = -np.minimum(movement[1:],zerohero[1:])*dt/dx
	m_west[-1] = abs(movement[-1])*dt/dx #reflective
	output.setdiag(1-dt/dx*abs(movement),0) #here
	output.setdiag(m_east,1)
	output.setdiag(m_west,-1)
	return sparse.csr_matrix(output)

def hjb_diffusion(time,x,a_tmp,m_tmp,dt,dx): #for implicit
	#functions
	sigma = iF.Sigma_global(time,x,a_tmp,m_tmp)
	sigma2 = sigma**2
	dx2 = dx**2
	I = x.size
	s_east = -sigma2*dt/(2*dx2)
	s_east[0] = 2*s_east[0] #reflective
	s_west = -sigma2[1:]*dt/(2*dx2)
	s_west[-1] = 2*s_west[-1] #reflective
	output = sparse.lil_matrix((I,I))
	output.setdiag(1+sigma2*dt/dx2,0)
	output.setdiag(s_east,1)
	output.setdiag(s_west,-1)
	return sparse.csr_matrix(output)
##################
# MATRIX GENERATION: FINITE VOLUME
##################
def fp_fv_convection(time,x,a_tmp,m_tmp,dt,dx): #for explicit
	#functions
	sigma = iF.Sigma_global(time,x,a_tmp,m_tmp)
	sigma2 = sigma**2
	movement = iF.f_global(time,x,a_tmp) #the function f
	I = x.size
	#generate the flux vectors
	Fi = (dx*movement[1:-1]-sigma[1:-1]*( sigma[2:]-sigma[0:-2] ))/(dx)
	Fi = np.append(Fi,(dx*movement[-1]-sigma[-1]*(-sigma[-2]))/dx )
	Fi = np.insert(Fi,0,(dx*movement[0]-sigma[0]*sigma[1])/dx)
	zerohero = np.zeros(x.size-1)
	output = sparse.lil_matrix((I,I))
	output.setdiag(1-dt/dx*abs(Fi),0)
	output.setdiag(-dt/dx*np.minimum(Fi[1:],zerohero),1)
	output.setdiag(dt/dx*np.maximum(Fi[0:-1],zerohero),-1)
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
	output = sparse.lil_matrix((I,I))
	output.setdiag(1+dt/dx2*(D_up+D_down),0)
	output.setdiag(-dt/dx2*D_up[1:],1)
	output.setdiag(-dt/dx2*D_down[0:-1],-1)
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


