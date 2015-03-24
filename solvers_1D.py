from __future__ import division
import numpy as np
import input_functions as iF
#these functions complete 1 iteration of the explicit schemes

###################
#HAMILTON-JACOBI-BELLMAN
###################

def hjb_kushner(x,time,u_last,m_last,a_tmp,dt,dx): #monotone
	u_tmp = np.empty(x.size)
	sigma2 = iF.Sigma_global(time,x,a_tmp,m_tmp)**2
	L_var = iF.L_global(time,x,a_tmp,m_tmp)
	movement = iF.f_global(time,x,a_tmp)
	#Kushner
	u_tmp[1:-1] = u_last[1:-1]*(1-sigma2[1:-1]*dt/dx2 - abs(movement[1:-1])*dt/dx) + u_last[2:]*(sigma2[1:-1]*dt/(2*dx2) + np.maximum(movement[1:-1],BIGZERO)*dt/dx) +  u_last[0:-2]*(sigma2[1:-1]*dt/(2*dx2) - np.minimum(movement[1:-1],BIGZERO)*dt/dx) + dt*L_var[1:-1]
	u_tmp[0] = u_last[0]*(1-sigma2[0]*dt/dx2 - abs(movement[0])*dt/dx)+u_last[1]*dt/dx2*(dx*abs(movement[0])+sigma2[0]) + dt*L_var[0]
	u_tmp[-1] = u_last[-1]*(1-sigma2[-1]*dt/dx2 - abs(movement[-1])*dt/dx)+u_last[-2]*dt/dx2*(dx*abs(movement[-1])+sigma2[-1]) + dt*L_var[-1]
	return u_tmp

def hjb_wrong(x,time,u_last,m_last,a_tmp,dt,dx):
	u_tmp = np.empty(x.size)
	sigma2 = iF.Sigma_global(time,x,a_tmp,m_tmp)**2
	L_var = iF.L_global(time,x,a_tmp,m_tmp)
	movement = iF.f_global(time,x,a_tmp)
	u_tmp[1:-1] = u_last[1:-1]*(1-sigma2[1:-1]*dt/dx2 + abs(movement[1:-1])*dt/dx) + u_last[2:]*(sigma2[1:-1]*dt/(2*dx2) + np.minimum(movement[1:-1],BIGZERO)*dt/dx) +  u_last[0:-2]*(sigma2[1:-1]*dt/(2*dx2) - np.maximum(movement[1:-1],BIGZERO)*dt/dx) + dt*L_var[1:-1]
	u_tmp[0] = u_last[0] + dt*(abs(movement[0])/dx - sigma2[0]/dx2)*(u_last[0] - u_last[1]) + dt*L_var[0]
	u_tmp[-1] = u_last[-1] + dt*(abs(movement[-1])/dx - sigma2[-1]/dx2)*(u_last[-1] - u_last[-2]) + dt*L_var[-1]
	return u_tmp

###################
#FOKKER-PLANCK
###################

def fp_fd_centered(x,time,m_tmp,a_tmp,dt,dx):
	sigma = iF.Sigma_global(time,x,a_tmp,m_tmp)
	sigma2 = sigma**2
	movement = iF.f_global(time,x,a_tmp) #the function f
	dx2 = dx**2
	#the actual computation
	m_update = np.empty(m_tmp.size)
	zero = np.zeros(movement[1:-1].size)
	m_update[1:-1] = m_tmp[1:-1]*(1-sigma2[1:-1]*dt/dx2) + m_tmp[2:]*(dt/(2*dx))*(sigma2[2:]/dx - movement[2:]) + m_tmp[0:-2]*(dt/(2*dx))*(sigma2[0:-2]/dx + movement[0:-2])
	#reflective
	m_update[0] = m_tmp[0]*(1-sigma2[0]*dt/dx2) + m_tmp[1]*dt*(sigma2[1]/dx2 - 0.5*(movement[1])/dx)
	m_update[-1] = m_tmp[-1]*(1-sigma2[-1]*dt/dx2) + m_tmp[-2]*dt*(sigma2[-2]/dx2 + 0.5*(movement[-2])/dx)
	#zero boundary
	#m_update[0] = m_tmp[0]*(1-sigma2[0]*dt/dx2) + m_tmp[1]*(dt/(2*dx))*(sigma2[1]/dx - movement[1])
	#m_update[-1]= m_tmp[-1]*(1-sigma2[1:-1]*dt/dx2) + m_tmp[-2]*(dt/(2*dx))*(sigma2[-2]/dx + movement[-2])
	return m_update 

def fp_fd_upwind(x,time,m_tmp,a_tmp,dt,dx):
	sigma = iF.Sigma_global(time,x,a_tmp,m_tmp)
	sigma2 = sigma**2
	movement = iF.f_global(time,x,a_tmp) #the function f
	dx2 = dx**2
	#the actual computation
	m_update = np.empty(m_tmp.size)
	zero = np.zeros(movement[1:-1].size)
	m_update[1:-1] = m_tmp[1:-1]*( 1-dt/dx2*sigma2[1:-1] - dt/(2*dx)*(movement[2:]-movement[0:-2]) - dt/dx *abs(movement[1:-1]) )
	m_update[1:-1] += m_tmp[2:]*dt/dx2*(sigma2[2:]/2 - dx*np.minimum(movement[1:-1],zero))
	m_update[1:-1] += m_tmp[0:-2]*dt/dx2*(sigma2[0:-2]/2 + dx*np.maximum(movement[1:-1],zero))
	#reflective
	m_update[0] = m_tmp[0]*( 1-dt/dx2*sigma2[0] - dt/(2*dx)*(movement[1]) - dt/dx *abs(movement[0]) ) + m_tmp[1]*dt/dx2*( sigma2[1]/2 + dx*abs(movement[0]) )
	m_update[-1] = m_tmp[-1]*( 1-dt/dx2*sigma2[-1] - dt/(2*dx)*(-movement[-2]) - dt/dx *abs(movement[-1]) ) + m_tmp[-2]*dt/dx2*( sigma2[-2]/2 + dx*abs(movement[-1]) )
	return m_update

def fp_fv(x,time,m_tmp,a_tmp,dt,dx):
	sigma = iF.Sigma_global(time,x,a_tmp,m_tmp)
	sigma2 = sigma**2
	movement = iF.f_global(time,x,a_tmp) #the function f
	dx2 = dx**2
	#the actual computation
	m_update = np.empty(m_tmp.size)
	zero = np.zeros(movement[1:-1].size)
	#generate the flux vectors
	Fi = (dx*movement[1:-1]-sigma[1:-1]*( sigma[2:]-sigma[0:-2] ))/(dx)
	Fi = np.append(Fi,(dx*movement[-1]-sigma[-1]*(-sigma[-2]))/dx )
	Fi = np.insert(Fi,0,(dx*movement[0]-sigma[0]*sigma[1])/dx)
	D_up = iF.hmean(sigma2[1:-1],sigma2[2:])/2
	D_down = iF.hmean(sigma2[1:-1],sigma2[0:-2])/2
	#regular upwinding
	m_update[1:-1] = m_tmp[1:-1]*(1 - dt/dx2*(D_up+D_down+dx*abs(Fi[1:-1])) )  
	m_update[1:-1] += m_tmp[2:]*dt/dx2*(  D_up - dx*np.minimum(Fi[2:],zero)  )
	m_update[1:-1] += m_tmp[0:-2]*dt/dx2*( D_down + dx*np.maximum(Fi[0:-2],zero) )
	D_w = iF.hmean_scalar(sigma2[0],sigma2[1])/2 #west boundary diffusion
	D_e = iF.hmean_scalar(sigma2[-1],sigma2[-2])/2#east boundary diffusion
	#reflective
	m_update[0] = (m_tmp[0]*(1-dt/dx2*(D_w+dx*abs(Fi[0]))) + m_tmp[1]*dt/dx2*(D_w-dx*np.min(Fi[1],0)))
	m_update[-1]= (m_tmp[-1]*(1-dt/dx2*(D_e+dx*abs(Fi[-1]))) + m_tmp[-2]*dt/dx2*(D_e+dx*np.max(Fi[-2],0)))
	return m_update

###################
#POLICY ITERATION FUNCTIONS
###################

def control_general(x,time,u_last,m_last,dt,dx,xpts_search,N,scatters):
	a_tmp = np.empty(x.size)
	for i in range (0,Nx):
		fpts = iF.hamiltonian(xpts_search,x,u_last,m_tmp,dt,dx,k*dt,i)
		x0 = xpts_search[np.argmin(fpts)]
		tmp,tmpval = iF.scatter_search(iF.hamiltonian,(x,u_last,m_tmp,dt,dx,k*dt,i),xpts_search[2]-xpts_search[1],x0,N,scatters) 
		a_tmp[i] = tmp
	return a_tmp

###################
#TAU FUNCTION
###################

###################
#RUNNING COST
###################

##################
#TERMINAL COST
##################

