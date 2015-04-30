from __future__ import division
import numpy as np
import input_functions as iF
import matrix_gen1d as mg
import scipy.sparse as sparse
import scipy.optimize as optimize
#these functions complete 1 iteration of the explicit schemes

###################
#HAMILTON-JACOBI-BELLMAN
###################

def hjb_kushner(x,time,u_last,m_tmp,a_tmp,dt,dx): #monotone, explicit
	u_tmp = np.empty(x.size)
	dx2 = dx**2
	BIGZERO = np.zeros(x.size-2)
	sigma2 = iF.Sigma_global(time,x,a_tmp,m_tmp)**2
	L_var = iF.L_global(time,x,a_tmp,m_tmp)
	movement = iF.f_global(time,x,a_tmp)
	#Kushner
	u_tmp[1:-1] = u_last[1:-1]*(1-sigma2[1:-1]*dt/dx2 - abs(movement[1:-1])*dt/dx) + u_last[2:]*(sigma2[1:-1]*dt/(2*dx2) + np.maximum(movement[1:-1],BIGZERO)*dt/dx) +  u_last[0:-2]*(sigma2[1:-1]*dt/(2*dx2) - np.minimum(movement[1:-1],BIGZERO)*dt/dx) + dt*L_var[1:-1]
	u_tmp[0] = u_last[0]*(1-sigma2[0]*dt/dx2 - abs(movement[0])*dt/dx)+u_last[1]*dt/dx2*(dx*abs(movement[0])+sigma2[0]) + dt*L_var[0]
	u_tmp[-1] = u_last[-1]*(1-sigma2[-1]*dt/dx2 - abs(movement[-1])*dt/dx)+u_last[-2]*dt/dx2*(dx*abs(movement[-1])+sigma2[-1]) + dt*L_var[-1]
	return u_tmp

def hjb_kushner_mod(x,time,u_last,m_tmp,a_tmp,dt,dx): #monotone, implicit
	LHS = mg.hjb_diffusion(time,x,a_tmp,m_tmp,dt,dx)
	RHS = mg.hjb_convection(time,x,a_tmp,m_tmp,dt,dx)
	Ltmp = iF.L_global(time,x,a_tmp,m_tmp)
	return sparse.linalg.spsolve(LHS,RHS*u_last+dt*Ltmp)
	#return sparse.linalg.spsolve(LHS,RHS*u_last)

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

def fp_fd_centered_mod(x,time,m_tmp,a_tmp,dt,dx):
	LHS = mg.fp_fd_centered_diffusion(time,x,m_tmp,a_tmp,dt,dx)
	RHS = mg.fp_fd_centered_convection(time,x,m_tmp,a_tmp,dt,dx)
	return sparse.linalg.spsolve(LHS,RHS*m_tmp)

def fp_fd_upwind(x,time,m_tmp,a_tmp,dt,dx):
	sigma = iF.Sigma_global(time,x,a_tmp,m_tmp)
	sigma2 = sigma**2
	movement = iF.f_global(time,x,a_tmp) #the function f
	dx2 = dx**2
	#the actual computation
	m_update = np.empty(m_tmp.size)
	zero = np.zeros(movement[1:-1].size)
	m_update[1:-1] = m_tmp[1:-1]*( 1-dt/dx2*sigma2[1:-1] - dt/(2*dx)*(movement[2:]-movement[0:-2]) - dt/dx *abs(movement[1:-1])  )
	m_update[1:-1] += m_tmp[2:]*dt/dx2*(sigma2[2:]/2 - dx*np.minimum(movement[1:-1],zero))
	m_update[1:-1] += m_tmp[0:-2]*dt/dx2*(sigma2[0:-2]/2 + dx*np.maximum(movement[1:-1],zero))
	#reflective
	m_update[0] = m_tmp[0]*( 1-dt/dx2*sigma2[0] - dt/(2*dx)*(movement[1]) - dt/dx *abs(movement[0]) ) + m_tmp[1]*dt/dx2*( sigma2[1]/2 + dx*abs(movement[0]) )
	m_update[-1] = m_tmp[-1]*( 1-dt/dx2*sigma2[-1] - dt/(2*dx)*(-movement[-2]) - dt/dx *abs(movement[-1]) ) + m_tmp[-2]*dt/dx2*( sigma2[-2]/2 + dx*abs(movement[-1]) )
	return m_update

def fp_fd_upwind_mod(x,time,m_tmp,a_tmp,dt,dx):
	LHS = mg.fp_fd_centered_diffusion(time,x,m_tmp,a_tmp,dt,dx)
	RHS = mg.fp_fd_upwind_convection(time,x,m_tmp,a_tmp,dt,dx)
	#print RHS
	#print ss
	return sparse.linalg.spsolve(LHS,RHS*m_tmp)

def fp_fd_upwind_visc(x,time,m_tmp,a_tmp,dt,dx):
	sigma = iF.Sigma_global(time,x,a_tmp,m_tmp)
	sigma2 = sigma**2
	movement = iF.f_global(time,x,a_tmp) #the function f
	dx2 = dx**2
	#the actual computation
	m_update = np.empty(m_tmp.size)
	zero = np.zeros(movement[1:-1].size)
	m_update[1:-1] = m_tmp[1:-1]*( 1-dt/dx2*sigma2[1:-1] - dt/(2*dx)*(movement[2:]-movement[0:-2]))#- dt/dx *abs(movement[1:-1])*2   ) 
	m_update[1:-1] += m_tmp[2:]*dt/dx2*(sigma2[2:]/2 - dx*np.minimum(movement[1:-1],zero) -dx*abs(movement[1:-1]) ) 
	m_update[1:-1] += m_tmp[0:-2]*dt/dx2*(sigma2[0:-2]/2 + dx*np.maximum(movement[1:-1],zero)-dx*abs(movement[1:-1]))
	#reflective
	m_update[0] = m_tmp[0]*( 1-dt/dx2*sigma2[0] - dt/(2*dx)*(movement[1]) + dt/dx *abs(movement[0]) ) + m_tmp[1]*dt/dx2*( sigma2[1]/2)
	m_update[-1] = m_tmp[-1]*( 1-dt/dx2*sigma2[-1] - dt/(2*dx)*(-movement[-2]) - dt/dx *abs(3*movement[-1]) ) + m_tmp[-2]*dt/dx2*( sigma2[-2]/2)
	return m_update

def fp_fv_mod(x,time,m_tmp,a_tmp,dt,dx):
	LHS = mg.fp_fv_diffusion(time,x,a_tmp,m_tmp,dt,dx)
	RHS = mg.fp_fv_convection(time,x,a_tmp,m_tmp,dt,dx)
	return sparse.linalg.spsolve(LHS,RHS*m_tmp)
def fp_fv(x,time,m_tmp,a_tmp,dt,dx):
	I = x.size
	sigma = iF.Sigma_global(time,x,a_tmp,m_tmp)
	sigma2 = sigma**2
	movement = iF.f_global(time,x,a_tmp) #the function f
	dx2 = dx**2
	#the fluxes
	Fi = (dx*movement[1:-1]-sigma[1:-1]*( sigma[2:]-sigma[0:-2] ))/(dx)
	Fi = np.append(Fi,(dx*movement[-1]-sigma[-1]*(-sigma[-2]))/dx )
	Fi = np.insert(Fi,0,(dx*movement[0]-sigma[0]*sigma[1])/dx)
	D_up = iF.hmean(sigma2[1:-1],sigma2[2:])/2
	D_down = iF.hmean(sigma2[1:-1],sigma2[0:-2])/2
	D_west = iF.hmean_scalar(sigma2[0],sigma2[1])
	D_east = iF.hmean_scalar(sigma2[-1],sigma2[-2])
	#the actual computation
	m_update = np.empty(m_tmp.size)
	zero = np.zeros(movement[1:-1].size)
	m_update[1:-1] = m_tmp[1:-1]*( 1-dt/dx2*(D_up+D_down + dx*(abs(Fi[1:-1]))))
	m_update[1:-1] += m_tmp[2:]*dt/dx2*(D_up-dx*np.minimum(Fi[2:],zero))
	m_update[1:-1] += m_tmp[0:-2]*dt/dx2*(D_down+dx*np.maximum(Fi[0:-2],zero))
	#reflective
	m_update[0] = m_tmp[0]*( 1-dt/dx2*(D_west + dx*abs(Fi[0]))) + m_tmp[1]*dt/dx2*( D_west - dx*min(Fi[1],0))
	m_update[-1] = m_tmp[-1]*( 1-dt/dx2*(D_east + dx*abs(Fi[-1]))) + m_tmp[-2]*dt/dx2*( D_east + dx*max(0,Fi[-2]))
	return m_update

###################
#POLICY ITERATION FUNCTIONS
###################

def control_general(x,time,u_last,m_last,dt,dx,xpts_search,N,scatters):
	a_tmp = np.empty(x.size)
	for i in range (0,x.size):
		fpts = iF.hamiltonian(xpts_search,x,u_last,m_last,dt,dx,time,i)
		x0 = xpts_search[np.argmin(fpts)]
		tmp,tmpval = iF.scatter_search(iF.hamiltonian,(x,u_last,m_last,dt,dx,time,i),xpts_search[2]-xpts_search[1],x0,N,scatters,xpts_search[0],xpts_search[-1])
		a_tmp[i] = tmp
	return a_tmp

def control_newton(x,time,u_last,m_last,dt,dx,xpts_search,tol):
	a_tmp = np.empty(x.size)
	for i in range (0,x.size):
		fpts = iF.hamiltonian(xpts_search,x,u_last,m_last,dt,dx,time,i)
		x0 = xpts_search[np.argmin(fpts)]
		zero = iF.newton_search(iF.Hamiltonian_Derivative,iF.Hamiltonian_Derivative2,(time,x[i],u_last,m_last,i,dx),tol,20,x0,xpts_search[2]-xpts_search[1],xpts_search[0],xpts_search[-1])
		a_tmp[i] = zero
	return a_tmp	

def control_bisect(x,time,u_last,m_last,dt,dx,xpts_search,N,tol):
	a_tmp = np.empty(x.size)
	dx
	for i in range (0,x.size):
		fpts = iF.hamiltonian(xpts_search,x,u_last,m_last,dt,dx,time,i)
		x0 = xpts_search[np.argmin(fpts)]
		tmp,tmpval = iF.bisection_search(iF.hamiltonian,(x,u_last,m_last,dt,dx,time,i),xpts_search[2]-xpts_search[1],x0,tol) 
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

