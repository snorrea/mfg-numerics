from __future__ import division
import numpy as np
import input_functions_2D as iF
import matrix_gen as mg
import scipy.sparse as sparse
import applications as app
import scipy.optimize as opt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time as time
import matplotlib.pyplot as plt
#these functions complete 1 iteration of the explicit schemes

###################
#HAMILTON-JACOBI-BELLMAN
###################

def hjb_kushner_mod(time,x,y,a1_tmp,a2_tmp,m_tmp,dx,dy,dt):
	#generate, solve
	LHS = mg.HJB_diffusion_implicit(time,x,y,a1,a2,dx,dy,dt)
	RHS = mg.HJB_convection_explicit(time,x,y,a1,a2,dx,dy,dt)
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

def control_general(search_x,search_y,x,y,u0,m,dt,dx,dy,time,I,J,tol,scatters,N,nulled,Obstacles,south,north,west,east):
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
			#if not iF.ismember(i+I*j,nulled):
			#print search_x
			#print search_y
			fpts = iF.hamiltonian(search_x,search_y,x,y,np.ravel(u),np.ravel(m),dt,dx,dy,time,i,j,I,J,Obstacles,south,north,west,east)
			(xi,yi) = np.unravel_index(np.argmin(fpts),fpts.shape)
			tmp_x,tmp_y = app.scatter_search(iF.hamiltonian,(x,y,np.ravel(u),np.ravel(m),dt,dx,dy,time,i,j,I,J,Obstacles,south,north,west,east),dxs,dys,search_x[xi,yi],search_y[xi,yi],N,scatters,xmin,xmax,ymin,ymax)
			a1[i,j] = tmp_x
			a2[i,j] = tmp_y
	#print "Found optimal control!"
	return a1,a2
#(xpts_scatter,xpts_scatter,x,y,u,m_tmp,dt,dx,dy,k*dt,I,J,min_tol,scatters,Ns,nulled,0*ObstacleCourse,south,north,west,east)
def control_general_vectorised(search_x,search_y,x,y,u,m,dt,dx,dy,timez,I,J,tol,scatters,N,nulled,Obstacles,south,north,west,east):
	ax = search_x[1]-search_x[0]
	ay = search_y[1]-search_y[0]
	#search_x,search_y = np.meshgrid(search_x,search_y)
	Search_x,Search_y,X,Y = np.meshgrid(search_x,search_y,x,y) #pls be 3D
	bounds = [search_x[0], search_x[-1], search_y[0], search_y[-1]]
	A1_read,A2_read = np.meshgrid(search_x,search_y)
	#Search_y,X,Y = np.meshgrid(search_y,x,y) #pls be 3D
	#make the things
	u_south = np.zeros((J,I))
	u_north = np.zeros((J,I))
	u_west = np.zeros((J,I))
	u_east = np.zeros((J,I))
	u_crossup = np.zeros((J,I))
	u_crossdown = np.zeros((J,I))
	u = (np.reshape(u,(J,I)))
	#Obstacles = np.transpose(Obstacles)
	#this does not pay heed to obstacles... fuck it; also, east/west behave like north/south, and vica versa, this is bad
#	u_east[:-1,:] = (u[1:,:]-u[:-1,:])/dx
#	u_west[1:,:] = (u[1:,:]-u[:-1,:])/dx
#	u_north[:,:-1] = (u[:,1:]-u[:,:-1])/dy
#	u_south[:,1:] = (u[:,1:]-u[:,:-1])/dy
#	u_crossup[1:-1,1:-1] = .5*( u[2:,1:-1] + u[:-2,1:-1] + u[1:-1,2:] + u[1:-1,:-2] - 2*u[1:-1,1:-1] - u[2:,:-2] - u[:-2,2:] )/(dx*dy)
#	u_crossdown[1:-1,1:-1] = .5*( 2*u[1:-1,1:-1] + u[2:,2:] + u[:-2,:-2] - u[2:,1:-1] - u[:-2,1:-1] - u[1:-1,2:] - u[1:-1,:-2] )/(dx*dy)
	u_north[:-1,:] = (u[1:,:]-u[:-1,:])/dx
	u_south[1:,:] = (u[1:,:]-u[:-1,:])/dx
	u_east[:,:-1] = (u[:,1:]-u[:,:-1])/dy
	u_west[:,1:] = (u[:,1:]-u[:,:-1])/dy
	u_crossup[1:-1,1:-1] = .5*( u[1:-1,2:] + u[1:-1,:-2] + u[2:,1:-1] + u[:-2,1:-1] - 2*u[1:-1,1:-1] - u[:-2,2:] - u[2:,:-2] )/(dx*dy)
	u_crossdown[1:-1,1:-1] = .5*( 2*u[1:-1,1:-1] + u[2:,2:] + u[:-2,:-2] - u[1:-1,2:] - u[1:-1,:-2] - u[2:,1:-1] - u[:-2,1:-1] )/(dx*dy)
	#boundary condition on u
#	u_east[-1,:] = (u[-2,:]-u[-1,:])/dx
#	u_west[0,:] = (u[0,:]-u[1,:])/dx
#	u_north[:,-1] = (u[:,-2]-u[:,-1])/dy
#	u_south[:,0] = (u[:,0]-u[:,1])/dy
#	u_crossup[0,1:-1] = .5*( 2*u[1,1:-1] + u[0,2:] + u[0,:-2] - 2*u[0,1:-1] - u[1,:-2] - u[1,2:] )/(dx*dy) #west
#	u_crossup[-1:,1:-1] = .5*( 2*u[-2,1:-1] + u[-1,2:] + u[-1,:-2] - 2*u[-1,1:-1] - u[-2,:-2] - u[-2,2:] )/(dx*dy) #east
#	u_crossup[1:-1,0] = .5*( u[2:,0] + u[:-2,0] + 2*u[1:-1,1] - 2*u[1:-1,0] - u[2:,1] - u[:-2,1] )/(dx*dy) #south
#	u_crossup[1:-1,-1] = .5*( u[2:,-1] + u[:-2,-1] + 2*u[1:-1,-2] - 2*u[1:-1,-1] - u[2:,-2] - u[:-2,-2] )/(dx*dy) #north
#	u_crossdown[0,1:-1] = .5*( 2*u[0,1:-1] + u[1,2:] + u[1,:-2] - 2*u[1,1:-1] - u[0,2:] - u[0,:-2] )/(dx*dy) #west
#	u_crossdown[-1,1:-1] = .5*( 2*u[-1,1:-1] + u[-2,2:] + u[-2,:-2] - 2*u[-2,1:-1] - u[-1,2:] - u[-1,:-2] )/(dx*dy)#east
#	u_crossdown[1:-1,0] = .5*( 2*u[1:-1,0] + u[2:,1] + u[:-2,1] - u[2:,0] - u[:-2,0] - 2*u[1:-1,1])/(dx*dy)#south
#	u_crossdown[1:-1,-1] = .5*( 2*u[1:-1,-1] + u[2:,-2] + u[:-2,-2] - u[2:,-1] - u[:-2,-1] - 2*u[1:-1,-2])/(dx*dy) #north
	u_north[-1,:] = (u[-2,:]-u[-1,:])/dx
	u_south[0,:] = (u[0,:]-u[1,:])/dx
	u_east[:,-1] = (u[:,-2]-u[:,-1])/dy
	u_west[:,0] = (u[:,0]-u[:,1])/dy
	u_crossup[1:-1,0] = .5*( 2*u[1:-1,1] + u[2:,0] + u[:-2,0] - 2*u[1:-1,0] - u[:-2,1] - u[2:,1] )/(dx*dy) #west
	u_crossup[1:-1,-1] = .5*( 2*u[1:-1,-2] + u[2:,-1] + u[:-2,-1] - 2*u[1:-1,-1] - u[:-2,-2] - u[2:,-2] )/(dx*dy) #east
	u_crossup[0,1:-1] = .5*( u[0,2:] + u[0,:-2] + 2*u[1,1:-1] - 2*u[0,1:-1] - u[1,2:] - u[1,:-2] )/(dx*dy) #south
	u_crossup[-1,1:-1] = .5*( u[-1,2:] + u[-1,:-2] + 2*u[-2,1:-1] - 2*u[-1,1:-1] - u[-2,2:] - u[-2,:-2] )/(dx*dy) #north
	u_crossdown[1:-1,0] = .5*( 2*u[1:-1,0] + u[2:,1] + u[:-2,1] - 2*u[1:-1,1] - u[2:,0] - u[:-2,0] )/(dx*dy) #west
	u_crossdown[1:-1,-1] = .5*( 2*u[1:-1,-1] + u[2:,-2] + u[:-2,-2] - 2*u[1:-1,-2] - u[2:,-1] - u[:-2,-1] )/(dx*dy)#east
	u_crossdown[0,1:-1] = .5*( 2*u[0,1:-1] + u[1,2:] + u[1,:-2] - u[0,2:] - u[0,:-2] - 2*u[1,1:-1])/(dx*dy)#south
	u_crossdown[-1,1:-1] = .5*( 2*u[-1,1:-1] + u[-2,2:] + u[-2,:-2] - u[-1,2:] - u[-1,:-2] - 2*u[-2,1:-1])/(dx*dy) #north
	#continued boundary conditions; four corners of crossterms
#	u_crossup[0,0] = .5*( 2*u[1,0] + 2*u[0,1] - 2*u[0,0] - 2*u[1,1])/(dx*dy) #southwest
#	u_crossup[0,-1] = .5*( 2*u[1,-1] + 2*u[0,-2] - 2*u[0,-1] - 2*u[1,-2])/(dx*dy) #northwest
#	u_crossup[-1,0] = .5*( 2*u[-2,0] + 2*u[-1,1] - 2*u[-1,-1] - 2*u[-2,1])/(dx*dy) #southeast
#	u_crossup[-1,-1] = .5*( 2*u[-2,-1] + 2*u[-1,-2] - 2*u[-1,-1] - 2*u[-2,-2])/(dx*dy)#northeast
#	u_crossdown[0,0] = .5*( 2*u[0,0] + 2*u[1,1] - 2*u[1,0] - 2*u[0,1])/(dx*dy)
#	u_crossdown[0,-1] = .5*( 2*u[0,-1] + 2*u[1,-2] - 2*u[1,-1] - 2*u[0,-2])/(dx*dy)
#	u_crossdown[-1,0] = .5*( 2*u[-1,0] + 2*u[-2,1] - 2*u[-2,0] - 2*u[-1,1])/(dx*dy)
#	u_crossdown[-1,-1] = .5*( 2*u[-1,-1] + 2*u[-2,-2] - 2*u[-2,-1] - 2*u[-1,-2])/(dx*dy)
	u_crossup[0,0] = .5*( 2*u[1,0] + 2*u[0,1] - 2*u[0,0] - 2*u[1,1])/(dx*dy) #southwest
	u_crossup[-1,0] = .5*( 2*u[-1,1] + 2*u[-2,0] - 2*u[-1,0] - 2*u[-2,1])/(dx*dy) #northwest
	u_crossup[0,-1] = .5*( 2*u[0,-2] + 2*u[1,-1] - 2*u[-1,-1] - 2*u[1,-2])/(dx*dy) #southeast
	u_crossup[-1,-1] = .5*( 2*u[-1,-2] + 2*u[-2,-1] - 2*u[-1,-1] - 2*u[-2,-2])/(dx*dy)#northeast
	u_crossdown[0,0] = .5*( 2*u[0,0] + 2*u[1,1] - 2*u[1,0] - 2*u[0,1])/(dx*dy)
	u_crossdown[-1,0] = .5*( 2*u[-1,0] + 2*u[-2,1] - 2*u[-1,1] - 2*u[-2,0])/(dx*dy)
	u_crossdown[0,-1] = .5*( 2*u[0,-1] + 2*u[1,-2] - 2*u[0,-2] - 2*u[1,-1])/(dx*dy)
	u_crossdown[-1,-1] = .5*( 2*u[-1,-1] + 2*u[-2,-2] - 2*u[-2,-1] - 2*u[-1,-2])/(dx*dy)
	for i in range(scatters):
		t0 = time.time()
		VALUEGRID = iF.Hamiltonian_vectorised(Search_x,Search_y,x,y,m,dx,dy,timez,I,J,Obstacles,u_south,u_north,u_west,u_east,u_crossup,u_crossdown,True)
		t1 = time.time()-t0
		t0 = time.time()
		ind0 = np.argmin(VALUEGRID,axis=0)
		ind1,ind2,ind3 = np.indices(ind0.shape)
		ind00 = Search_y[ind0,ind1,ind2,ind3].argmin(axis=0)
		ind11,ind22 = np.indices(ind00.shape)
		BESTBUYS_Y = Search_y[ind0,ind1,ind2,ind3][ind00,ind11,ind22]
		ind1 = np.argmin(VALUEGRID,axis=1)
		ind0,ind2,ind3 = np.indices(ind1.shape)
		ind00 = Search_x[ind0,ind1,ind2,ind3].argmin(axis=0)
		ind11,ind22 = np.indices(ind00.shape)
		BESTBUYS_X = Search_x[ind0,ind1,ind2,ind3][ind00,ind11,ind22]
		if i-1 is not scatters:
			for j in range(Search_x.shape[2]):
				for k in range(Search_x.shape[3]):
					Search_x[:,:,j,k] , Search_y[:,:,j,k] = np.meshgrid( np.linspace(BESTBUYS_X[j,k]-ax,BESTBUYS_X[j,k]+ay,N[0]), np.linspace(BESTBUYS_Y[j,k]-ay,BESTBUYS_Y[j,k]+ay,N[1]) )
			ay = Search_y[1,0,0,0]-Search_y[0,0,0,0]
			ax = Search_y[0,1,0,0]-Search_y[0,0,0,0]
		t2 = time.time()-t0
		print "Function call:", t1
		print "The rest:", t2
	return BESTBUYS_X,BESTBUYS_Y

#def control_general_vectorised_optimised(x,time,u,m,dx,xpts_search,N,scatters):
#	for i in range(scatters):
#		VALUEGRID = iF.Hamiltonian_array(Xpts_search,time,x,u_up,u_down,m,dx) #evaluate
#		BEST_BUYS = Xpts_search[np.argmin(VALUEGRID,axis=0),range(0,VALUEGRID.shape[1])] #pick least values by row
#		if i-1 is not scatters:
#			for j in range(BEST_BUYS.size):
#				Xpts_search[:,j] = np.linspace(BEST_BUYS[j]-ax,BEST_BUYS[j]+ax,N)
#			ax = Xpts_search[1,0]-Xpts_search[0,0]
#	return BEST_BUYS

def control_hybrid(search_x,search_y,x,y,u0,m,dt,dx,dy,cock_zeit,I,J,tol,scatters,N,nulled,ObstacleCourse,south,north,west,east):
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
			fpts = iF.hamiltonian(search_x,search_y,x,y,np.ravel(u),np.ravel(m),dt,dx,dy,cock_zeit,i,j,I,J,ObstacleCourse,south,north,west,east)
			#if cock_zeit<1.4:
			#	fig1 = plt.figure(1)
			#	ax1 = fig1.add_subplot(111, projection='3d')
			#	ax1.plot_surface(search_x,search_y,fpts,rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=0, antialiased=False)
			#	plt.show()
			(xi,yi) = np.unravel_index(np.argmin(fpts),fpts.shape)
			#tmp_x,tmp_y = app.hybrid_search(iF.hamiltonian,iF.hamiltonian_derivative,(x,y,np.ravel(u),np.ravel(m),dt,dx,dy,cock_zeit,i,j,I,J,ObstacleCourse,south,north,west,east),dxs,dys,search_x[xi,yi],search_y[xi,yi],N,scatters,xmin,xmax,ymin,ymax,tol)
			#tmp_x,tmp_y = app.hybrid_search66(iF.hamiltonian,(x,y,np.ravel(u),np.ravel(m),dt,dx,dy,cock_zeit,i,j,I,J,ObstacleCourse,south,north,west,east),fpts[xi,yi],dxs,dys,search_x[xi,yi],search_y[xi,yi],N,scatters,xmin,xmax,ymin,ymax,tol)
			x0 = np.array([search_x[xi,yi],search_y[xi,yi]])
			tmp_x,tmp_y = opt.minimize(iF.hamiltonian,x0,(x,y,np.ravel(u),np.ravel(m),dt,dx,dy,cock_zeit,i,j,I,J,ObstacleCourse,south,north,west,east))
			#print tmp_x,tmp_y
			#print ss
			a1[i,j] = tmp_x
			a2[i,j] = tmp_y
	#print "Found optimal control!"
	return a1,a2

