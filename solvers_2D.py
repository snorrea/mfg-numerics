from __future__ import division
import numpy as np
import input_functions_2D as iF
import matrix_gen as mg
import scipy.sparse as sparse
import applications as app
#these functions complete 1 iteration of the explicit schemes

###################
#HAMILTON-JACOBI-BELLMAN
###################

def hjb_kushner_mod(time,x,y,a1_tmp,a2_tmp,m_tmp,dx,dy,dt):
	#generate, solve
	LHS = mg.HJB_diffusion_implicit(time,x,y,a1,a2,m_tmp,dx,dy,dt)
	RHS = mg.HJB_convection_explicit(time,x,y,a1,a2,m_tmp,dx,dy,dt)
	Ltmp = iF.L_global(time,x,y,a1,a2,m_tmp)
	return sparse.linalg.spsolve(LHS,RHS*u_last+dt*Ltmp)
###################
#FOKKER-PLANCK
###################

def fp_fv_mod(x,y,time,m_tmp,a1_tmp,a2_tmp,D11,D22,D12,dt,dx):
	I = x.size
	J = y.size
	eye = sparse.identity((I*J),format='lil')
	#get function values
	[f1_array, f2_array] = iF.f_global(time,x,y,a1_tmp,a2_tmp)
	D11 = iF.Sigma_D11_test(time,x,y,a1_tmp,a2_tmp,m_tmp)
	D12 = iF.Sigma_D12_test(time,x,y,a1_tmp,a2_tmp,m_tmp)
	D22 = iF.Sigma_D22_test(time,x,y,a1_tmp,a2_tmp,m_tmp)
	#make matrices
	LHS = mg.add_diffusion_flux_Ometh(eye,D11,D22,D12,I,J,dx,dt)
	LHS = sparse.csr(LHS)
	RHS = mg.fp_fv_convection(time,x,a_tmp,m_tmp,dt,dx)
	#print LHS
	#print RHS
	#print ss
	#LHS = sparse.csr(sparse.eye(I)-mg.fp_fv_diffusion(time,x,a_tmp,m_tmp,dt,dx))
	#RHS = sparse.csr(sparse.eye(I)+mg.fp_fv_convection(time,x,a_tmp,m_tmp,dt,dx))
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

def control_general(search_x,search_y,x,y,u0,m,dt,dx,dy,time,I,J,tol,scatters,N):
	u = u0
	a1 = np.zeros((I,J))
	a2 = np.zeros((I,J))
	a1old = np.zeros((I,J))
	a2old = np.zeros((I,J))
	#print "Search",search
	key = search_x.size
	dxs = search_x[1]-search_x[0]
	dys = search_y[1]-search_y[0]
	xmin = search_x[0]
	xmax = search_x[-1]
	ymin = search_y[0]
	ymax = search_y[-1]
	search_x,search_y = np.meshgrid(search_x,search_y)
	for i in range (0,I):
		for j in range (0,J):
			fpts = iF.hamiltonian(search_x,search_y,x,y,np.ravel(u),np.ravel(m),dt,dx,dy,time,i,j,I,J)
			xi,yi = app.recover_index(np.argmin(fpts),key)
			tmp_x,tmp_y = app.scatter_search(iF.hamiltonian,(x,y,np.ravel(u),np.ravel(m),dt,dx,dy,time,i,j,I,J),dxs,dys,search_x[xi,yi],search_y[xi,yi],N,scatters,xmin,xmax,ymin,ymax)
			#print tmp_x,tmp_y
			#print ss
			a1[i,j] = tmp_x
			a2[i,j] = tmp_y
	print "Found optimal control!"
	return a1,a2

def control_hybrid(search_x,search_y,x,y,u0,m,dt,dx,dy,time,I,J,tol,scatters,N):
	u = u0
	a1 = np.zeros((I,J))
	a2 = np.zeros((I,J))
	a1old = np.zeros((I,J))
	a2old = np.zeros((I,J))
	#print "Search",search
	key = search_x.size
	dxs = search_x[1]-search_x[0]
	dys = search_y[1]-search_y[0]
	xmin = search_x[0]
	xmax = search_x[-1]
	ymin = search_y[0]
	ymax = search_y[-1]
	search_x,search_y = np.meshgrid(search_x,search_y)
	for i in range (0,I):
		for j in range (0,J):
			fpts = iF.hamiltonian(search_x,search_y,x,y,np.ravel(u),np.ravel(m),dt,dx,dy,time,i,j,I,J)
			xi,yi = app.recover_index(np.argmin(fpts),key)
			tmp_x,tmp_y = app.hybrid_search(iF.hamiltonian,iF.hamiltonian_derivative,(x,y,np.ravel(u),np.ravel(m),dt,dx,dy,time,i,j,I,J),dxs,dys,search_x[xi,yi],search_y[xi,yi],N,scatters,xmin,xmax,ymin,ymax)
			#print tmp_x,tmp_y
			#print ss
			a1[i,j] = tmp_x
			a2[i,j] = tmp_y
	print "Found optimal control!"
	return a1,a2

