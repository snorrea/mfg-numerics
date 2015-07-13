from __future__ import division
import numpy as np
import input_functions_2D as iF
import matrix_gen as mg
import scipy.sparse as sparse
import applications as app
import scipy.optimize as opt
import itertools as itertools
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
def control_general_vectorised_4D(search_x,search_y,x,y,u,m,dt,dx,dy,timez,I,J,tol,scatters,N,nulled,Obstacles,south,north,west,east):
	ax = search_x[1]-search_x[0]
	ay = search_y[1]-search_y[0]
	bounds = [search_x[0], search_x[-1], search_y[0], search_y[-1]]
	Search_x,Search_y,X,Y = np.meshgrid(search_x,search_y,x,y) #this is 4D
	#make the things
	u_south = np.zeros((J,I))
	u_north = np.zeros((J,I))
	u_west = np.zeros((J,I))
	u_east = np.zeros((J,I))
	u_crossup = np.zeros((J,I))
	u_crossdown = np.zeros((J,I))
	u = (np.reshape(u,(J,I)))
	u_north[:-1,:] = (u[1:,:]-u[:-1,:])/dx
	u_south[1:,:] = (u[1:,:]-u[:-1,:])/dx
	u_east[:,:-1] = (u[:,1:]-u[:,:-1])/dy
	u_west[:,1:] = (u[:,1:]-u[:,:-1])/dy
	u_crossup[1:-1,1:-1] = .5*( u[1:-1,2:] + u[1:-1,:-2] + u[2:,1:-1] + u[:-2,1:-1] - 2*u[1:-1,1:-1] - u[:-2,2:] - u[2:,:-2] )/(dx*dy)
	u_crossdown[1:-1,1:-1] = .5*( 2*u[1:-1,1:-1] + u[2:,2:] + u[:-2,:-2] - u[1:-1,2:] - u[1:-1,:-2] - u[2:,1:-1] - u[:-2,1:-1] )/(dx*dy)
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
		#	indx = range(Search_x.shape[2])
		#	indy = range(Search_x.shape[3])
		#	mytest = np.array([np.meshgrid(np.linspace(BESTBUYS_X[j,k]-ax,BESTBUYS_X[j,k]+ax,N[0]), np.linspace(BESTBUYS_Y[j,k]-ay,BESTBUYS_Y[j,k]+ay,N[1])) for j,k in zip(range(Search_x.shape[2]),range(Search_x.shape[3]))])
		#	Search_x[:,:,indx,indy] , Search_y[:,:,indx,indy] = np.transpose(mytest[:,0,:,:]),np.transpose(mytest[:,1,:,:])
			for j in range(Search_x.shape[2]):
				for k in range(Search_x.shape[3]):
					Search_x[:,:,j,k] , Search_y[:,:,j,k] = np.meshgrid( np.linspace(BESTBUYS_X[j,k]-ax,BESTBUYS_X[j,k]+ax,N[0]), np.linspace(BESTBUYS_Y[j,k]-ay,BESTBUYS_Y[j,k]+ay,N[1]) )
			#		print Search_x[:,:,j,k].shape
			#		print ss
			ay = Search_y[1,0,0,0]-Search_y[0,0,0,0]
			ax = Search_y[0,1,0,0]-Search_y[0,0,0,0]
	return BESTBUYS_X,BESTBUYS_Y

def control_general_vectorised_3D(search_x,search_y,x,y,u,m,dt,dx,dy,timez,I,J,tol,scatters,N,nulled,Obstacles,south,north,west,east):
	ax = search_x[1]-search_x[0]
	ay = search_y[1]-search_y[0]
	bounds = [search_x[0], search_x[-1], search_y[0], search_y[-1]]
	Search_x,X,Y = np.meshgrid(search_x,x,y) #this is 3D
	#Search_x,X,Y = np.meshgrid(search_x,x,y,indexing='ij') #this is 3D
#	print X
#	print Search_x.shape
#	print ss
	#print X
	Search_y,X,Y = np.meshgrid(search_y,x,y) #this is also 3D
	#print X
#	print ss
	#make the things
	u_south = np.zeros((J,I))
	u_north = np.zeros((J,I))
	u_west = np.zeros((J,I))
	u_east = np.zeros((J,I))
	u_crossup = np.zeros((J,I))
	u_crossdown = np.zeros((J,I))
	u = (np.reshape(u,(J,I)))
	m = (np.reshape(m,(J,I)))
	u_north[:-1,:] = (u[1:,:]-u[:-1,:])/dx
	u_south[1:,:] = (u[1:,:]-u[:-1,:])/dx
	u_east[:,:-1] = (u[:,1:]-u[:,:-1])/dy
	u_west[:,1:] = (u[:,1:]-u[:,:-1])/dy
	u_crossup[1:-1,1:-1] = .5*( u[1:-1,2:] + u[1:-1,:-2] + u[2:,1:-1] + u[:-2,1:-1] - 2*u[1:-1,1:-1] - u[:-2,2:] - u[2:,:-2] )/(dx*dy)
	u_crossdown[1:-1,1:-1] = .5*( 2*u[1:-1,1:-1] + u[2:,2:] + u[:-2,:-2] - u[1:-1,2:] - u[1:-1,:-2] - u[2:,1:-1] - u[:-2,1:-1] )/(dx*dy)
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
	u_crossup[0,0] = .5*( 2*u[1,0] + 2*u[0,1] - 2*u[0,0] - 2*u[1,1])/(dx*dy) #southwest
	u_crossup[-1,0] = .5*( 2*u[-1,1] + 2*u[-2,0] - 2*u[-1,0] - 2*u[-2,1])/(dx*dy) #northwest
	u_crossup[0,-1] = .5*( 2*u[0,-2] + 2*u[1,-1] - 2*u[-1,-1] - 2*u[1,-2])/(dx*dy) #southeast
	u_crossup[-1,-1] = .5*( 2*u[-1,-2] + 2*u[-2,-1] - 2*u[-1,-1] - 2*u[-2,-2])/(dx*dy)#northeast
	u_crossdown[0,0] = .5*( 2*u[0,0] + 2*u[1,1] - 2*u[1,0] - 2*u[0,1])/(dx*dy)
	u_crossdown[-1,0] = .5*( 2*u[-1,0] + 2*u[-2,1] - 2*u[-1,1] - 2*u[-2,0])/(dx*dy)
	u_crossdown[0,-1] = .5*( 2*u[0,-1] + 2*u[1,-2] - 2*u[0,-2] - 2*u[1,-1])/(dx*dy)
	u_crossdown[-1,-1] = .5*( 2*u[-1,-1] + 2*u[-2,-2] - 2*u[-2,-1] - 2*u[-1,-2])/(dx*dy)
	#FixedPoint = False
	BESTBUYS_X = np.zeros((I,J))
	BESTBUYS_Y = np.zeros((I,J))
	BBX_old = np.zeros((I,J))
	BBY_old = np.zeros((I,J))
	#stuff
	#m = app.map_2d_to_3d(m,Search_x)
	for i in range(scatters):
		cnt = 0
		while cnt < 10:
			#VALUEGRID = iF.Hamiltonian_vectorised(Search_x,app.map_2d_to_3d(BESTBUYS_Y,Search_x),x,y,m,dx,dy,timez,I,J,Obstacles,u_south,u_north,u_west,u_east,u_crossup,u_crossdown,True)
			#print Search_x.shape,x.shape,y.shape,m.shape
		#	VALUEGRID = iF.Hamiltonian_vectorised(Search_x,app.map_2d_to_3d(BESTBUYS_Y,Search_x),x,y,app.map_2d_to_3d(m,Search_x),dx,dy,timez,I,J,Obstacles,u_south,u_north,u_west,u_east,u_crossup,u_crossdown,True)
			VALUEGRID = iF.Hamiltonian_vectorised(Search_x,app.map_2d_to_3d(BESTBUYS_Y,Search_x),X,Y,app.map_2d_to_3d(m,Search_x),dx,dy,timez,I,J,Obstacles,u_south,u_north,u_west,u_east,u_crossup,u_crossdown,True)
		#	print VALUEGRID.shape
		#	print Search_x
		#	print ss
			#print Search_x
			#print Search_x.shape
			#print VALUEGRID.shape
			ind1 = VALUEGRID.argmin(axis=1)
		#	print VALUEGRID
		#	print ind1
		#	print ss
			ind0,ind2 = np.indices(ind1.shape)
			#print VALUEGRID[ind0,ind1,ind2]==np.amin(VALUEGRID,axis=1)
			#print ss
			BBX_old = np.copy(BESTBUYS_X)
			BESTBUYS_X = Search_x[ind0,ind1,ind2]
			#print Search_x[ind0,ind1,ind2]
			#print np.where(Search_x == Search_x.min())
			#print ss
		#	print ind1
			#VALUEGRID = iF.Hamiltonian_vectorised(app.map_2d_to_3d(BESTBUYS_X,Search_y),Search_y,x,y,m,dx,dy,timez,I,J,Obstacles,u_south,u_north,u_west,u_east,u_crossup,u_crossdown,True)
			#VALUEGRID = iF.Hamiltonian_vectorised(app.map_2d_to_3d(BESTBUYS_X,Search_y),Search_y,x,y,app.map_2d_to_3d(m,Search_y),dx,dy,timez,I,J,Obstacles,u_south,u_north,u_west,u_east,u_crossup,u_crossdown,True)
			VALUEGRID = iF.Hamiltonian_vectorised(app.map_2d_to_3d(BESTBUYS_X,Search_y),Search_y,X,Y,app.map_2d_to_3d(m,Search_y),dx,dy,timez,I,J,Obstacles,u_south,u_north,u_west,u_east,u_crossup,u_crossdown,True)
			ind1 = VALUEGRID.argmin(axis=1)
			ind0,ind2 = np.indices(ind1.shape)
			BBY_old = np.copy(BESTBUYS_Y)
			BESTBUYS_Y = Search_y[ind0,ind1,ind2]
			dev = np.linalg.norm(BBX_old-BESTBUYS_X)+np.linalg.norm(BBY_old-BESTBUYS_Y)
			if dev < 1e-6:
				#print "Dev:",dev
				break
			else:
				cnt += 1
				#print BBX_old
				#print BESTBUYS_X
				#print BESTBUYS_X==BBX_old
		#	if cnt>10:
		#		print cnt#,dev
		if cnt==10:
			print "Bad things happened"
		if i-1 is not scatters:
			x_ind = range(Search_x.shape[0])
			y_ind = range(Search_x.shape[2])
		#	print "Yolo"
		#	print zip(x_ind,y_ind)
		#	print BESTBUYS_X.shape
		#	print BESTBUYS_Y.shape
		#	print np.array([np.linspace(BESTBUYS_X[j,k]-ax,BESTBUYS_X[j,k]+ax,N[0]) for j,k in zip(x_ind,y_ind)]).shape
		#	print np.array([np.linspace(BESTBUYS_Y[j,k]-ay,BESTBUYS_Y[j,k]+ay,N[1]) for j,k in zip(x_ind,y_ind)]).shape
			#print Search_x.shape
			#print Search_x[y_ind,:,x_ind].shape
			#Search_x[j,:,x_ind] = np.array([np.linspace(BESTBUYS_X[j,k]-ax,BESTBUYS_X[j,k]+ax,N[0]) for j,k in zip(y_ind,x_ind)])
			#Search_y[y_ind,:,k] = np.array([np.linspace(BESTBUYS_Y[j,k]-ay,BESTBUYS_Y[j,k]+ay,N[1]) for j,k in zip(y_ind,x_ind)])
			#Search_x[x_ind,:,y_ind] = np.array([np.linspace(BESTBUYS_X[j,k]-ax,BESTBUYS_X[j,k]+ax,N[0]) for j,k in zip(x_ind,y_ind)])
			#Search_y[x_ind,:,y_ind] = np.array([np.linspace(BESTBUYS_Y[j,k]-ay,BESTBUYS_Y[j,k]+ay,N[1]) for j,k in zip(x_ind,y_ind)])
			for j in range(Search_x.shape[0]):
				for k in range(Search_x.shape[2]):
					Search_x[j,:,k] = np.linspace(BESTBUYS_X[j,k]-ax,BESTBUYS_X[j,k]+ax,N[0])
					Search_y[j,:,k] = np.linspace(BESTBUYS_Y[j,k]-ay,BESTBUYS_Y[j,k]+ay,N[1])
			#Search_x[j,:,k] = np.array([np.linspace(BESTBUYS_X[j,k]-ax,BESTBUYS_X[j,k]+ax,N[0]) for j,k in zip(x_ind,y_ind)])
			#Search_y[j,:,k] = np.array([np.linspace(BESTBUYS_Y[j,k]-ay,BESTBUYS_Y[j,k]+ay,N[1]) for j,k in zip(x_ind,y_ind)])
			ay = Search_y[0,1,0]-Search_y[0,0,0]
			ax = Search_x[0,1,0]-Search_x[0,0,0]
#	print ss		
	return BESTBUYS_X,BESTBUYS_Y


def control_general_vectorised_3Dij(search_x,search_y,x,y,u,m,dt,dx,dy,timez,I,J,tol,scatters,N,nulled,Obstacles,south,north,west,east):
	ax = search_x[1]-search_x[0]
	ay = search_y[1]-search_y[0]
	bounds = [search_x[0], search_x[-1], search_y[0], search_y[-1]]
	Search_x,Xx,Yx = np.meshgrid(search_x,x,y,indexing='ij') #this is 3D
	Search_y,Xy,Yy = np.meshgrid(search_y,x,y,indexing='ij')
	#print X #this is also 3D
	#print X
#	print ss
	#make the things
	u_south = np.zeros((J,I))
	u_north = np.zeros((J,I))
	u_west = np.zeros((J,I))
	u_east = np.zeros((J,I))
	u_crossup = np.zeros((J,I))
	u_crossdown = np.zeros((J,I))
	u = (np.reshape(u,(J,I)))
	m = (np.reshape(m,(J,I)))
	u_north[:-1,:] = (u[1:,:]-u[:-1,:])/dx
	u_south[1:,:] = (u[1:,:]-u[:-1,:])/dx
	u_east[:,:-1] = (u[:,1:]-u[:,:-1])/dy
	u_west[:,1:] = (u[:,1:]-u[:,:-1])/dy
	u_crossup[1:-1,1:-1] = .5*( u[1:-1,2:] + u[1:-1,:-2] + u[2:,1:-1] + u[:-2,1:-1] - 2*u[1:-1,1:-1] - u[:-2,2:] - u[2:,:-2] )/(dx*dy)
	u_crossdown[1:-1,1:-1] = .5*( 2*u[1:-1,1:-1] + u[2:,2:] + u[:-2,:-2] - u[1:-1,2:] - u[1:-1,:-2] - u[2:,1:-1] - u[:-2,1:-1] )/(dx*dy)
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
	u_crossup[0,0] = .5*( 2*u[1,0] + 2*u[0,1] - 2*u[0,0] - 2*u[1,1])/(dx*dy) #southwest
	u_crossup[-1,0] = .5*( 2*u[-1,1] + 2*u[-2,0] - 2*u[-1,0] - 2*u[-2,1])/(dx*dy) #northwest
	u_crossup[0,-1] = .5*( 2*u[0,-2] + 2*u[1,-1] - 2*u[-1,-1] - 2*u[1,-2])/(dx*dy) #southeast
	u_crossup[-1,-1] = .5*( 2*u[-1,-2] + 2*u[-2,-1] - 2*u[-1,-1] - 2*u[-2,-2])/(dx*dy)#northeast
	u_crossdown[0,0] = .5*( 2*u[0,0] + 2*u[1,1] - 2*u[1,0] - 2*u[0,1])/(dx*dy)
	u_crossdown[-1,0] = .5*( 2*u[-1,0] + 2*u[-2,1] - 2*u[-1,1] - 2*u[-2,0])/(dx*dy)
	u_crossdown[0,-1] = .5*( 2*u[0,-1] + 2*u[1,-2] - 2*u[0,-2] - 2*u[1,-1])/(dx*dy)
	u_crossdown[-1,-1] = .5*( 2*u[-1,-1] + 2*u[-2,-2] - 2*u[-2,-1] - 2*u[-1,-2])/(dx*dy)
	#FixedPoint = False
	BESTBUYS_X = np.zeros((I,J))
	BESTBUYS_Y = np.zeros((I,J))
	BBX_old = np.zeros((I,J))
	BBY_old = np.zeros((I,J))
	#stuff
	#m = app.map_2d_to_3d(m,Search_x)
	for i in range(scatters):
		cnt = 0
		while cnt < 10:
			### XXXX
			VALUEGRID = iF.Hamiltonian_vectorisedij(Search_x,app.map_2d_to_3dij(BESTBUYS_Y,Search_x),Xx,Yx,app.map_2d_to_3dij(m,Search_x),dx,dy,timez,I,J,Obstacles,u_south,u_north,u_west,u_east,u_crossup,u_crossdown,True)
			ind0 = VALUEGRID.argmin(axis=0)
			ind1,ind2 = np.indices(ind0.shape)
			BBX_old = np.copy(BESTBUYS_X)
			BESTBUYS_X = Search_x[ind0,ind1,ind2]
			### YYYY
			VALUEGRID = iF.Hamiltonian_vectorisedij(app.map_2d_to_3dij(BESTBUYS_X,Search_y),Search_y,Xy,Yy,app.map_2d_to_3dij(m,Search_y),dx,dy,timez,I,J,Obstacles,u_south,u_north,u_west,u_east,u_crossup,u_crossdown,True)
			ind0 = VALUEGRID.argmin(axis=0)
			ind1,ind2 = np.indices(ind0.shape)
			BBY_old = np.copy(BESTBUYS_Y)
			BESTBUYS_Y = Search_y[ind0,ind1,ind2]
			dev = np.linalg.norm(BBX_old-BESTBUYS_X)+np.linalg.norm(BBY_old-BESTBUYS_Y)
			if dev < 1e-6:
				#print "Dev:",dev
				break
			else:
				#print cnt, dev
				cnt += 1
				#print BBX_old
				#print BESTBUYS_X
				#print BESTBUYS_X==BBX_old
		#	if cnt>10:
		#		print cnt#,dev
		if cnt==10:
			print "Bad things happened"
		#	print dev
		if i-1 is not scatters:
		#	print "Yolo"
		#	print zip(x_ind,y_ind)
		#	print BESTBUYS_X.shape
		#	print BESTBUYS_Y.shape
		#	print np.array([np.linspace(BESTBUYS_X[j,k]-ax,BESTBUYS_X[j,k]+ax,N[0]) for j,k in zip(x_ind,y_ind)]).shape
		#	print np.array([np.linspace(BESTBUYS_Y[j,k]-ay,BESTBUYS_Y[j,k]+ay,N[1]) for j,k in zip(x_ind,y_ind)]).shape
			#print Search_x.shape
			#print Search_x[y_ind,:,x_ind].shape
			#Search_x[j,:,x_ind] = np.array([np.linspace(BESTBUYS_X[j,k]-ax,BESTBUYS_X[j,k]+ax,N[0]) for j,k in zip(y_ind,x_ind)])
			#Search_y[y_ind,:,k] = np.array([np.linspace(BESTBUYS_Y[j,k]-ay,BESTBUYS_Y[j,k]+ay,N[1]) for j,k in zip(y_ind,x_ind)])
			#Search_x[x_ind,:,y_ind] = np.array([np.linspace(BESTBUYS_X[j,k]-ax,BESTBUYS_X[j,k]+ax,N[0]) for j,k in zip(x_ind,y_ind)])
			#Search_y[x_ind,:,y_ind] = np.array([np.linspace(BESTBUYS_Y[j,k]-ay,BESTBUYS_Y[j,k]+ay,N[1]) for j,k in zip(x_ind,y_ind)])
		#	print Search
			for j in range(x.size):
				for k in range(y.size):
					Search_x[:,j,k] = np.linspace(BESTBUYS_X[j,k]-ax,BESTBUYS_X[j,k]+ax,N[0])
					Search_y[:,j,k] = np.linspace(BESTBUYS_Y[j,k]-ay,BESTBUYS_Y[j,k]+ay,N[1])
			#Search_x[j,:,k] = np.array([np.linspace(BESTBUYS_X[j,k]-ax,BESTBUYS_X[j,k]+ax,N[0]) for j,k in zip(x_ind,y_ind)])
			#Search_y[j,:,k] = np.array([np.linspace(BESTBUYS_Y[j,k]-ay,BESTBUYS_Y[j,k]+ay,N[1]) for j,k in zip(x_ind,y_ind)])
			ay = Search_y[1,0,0]-Search_y[0,0,0]
			ax = Search_x[1,0,0]-Search_x[0,0,0]
#	print ss		
	return BESTBUYS_X,BESTBUYS_Y


def control_hybridC_vectorised_3D(search_x,search_y,x,y,u,m,dt,dx,dy,timez,I,J,tol,scatters,N,nulled,Obstacles,south,north,west,east):
	ax = search_x[1]-search_x[0]
	ay = search_y[1]-search_y[0]
	bounds = [search_x[0], search_x[-1], search_y[0], search_y[-1]]
	Search_x,X,Y = np.meshgrid(search_x,x,y) #this is 3D
	Search_y,X,Y = np.meshgrid(search_y,x,y) #this is also 3D
	FOUR_DEE = False
	#make the things
	u_south = np.zeros((J,I))
	u_north = np.zeros((J,I))
	u_west = np.zeros((J,I))
	u_east = np.zeros((J,I))
	u_crossup = np.zeros((J,I))
	u_crossdown = np.zeros((J,I))
	u = (np.reshape(u,(J,I)))
	m = (np.reshape(m,(J,I)))
	u_north[:-1,:] = (u[1:,:]-u[:-1,:])/dx
	u_south[1:,:] = (u[1:,:]-u[:-1,:])/dx
	u_east[:,:-1] = (u[:,1:]-u[:,:-1])/dy
	u_west[:,1:] = (u[:,1:]-u[:,:-1])/dy
	u_crossup[1:-1,1:-1] = .5*( u[1:-1,2:] + u[1:-1,:-2] + u[2:,1:-1] + u[:-2,1:-1] - 2*u[1:-1,1:-1] - u[:-2,2:] - u[2:,:-2] )/(dx*dy)
	u_crossdown[1:-1,1:-1] = .5*( 2*u[1:-1,1:-1] + u[2:,2:] + u[:-2,:-2] - u[1:-1,2:] - u[1:-1,:-2] - u[2:,1:-1] - u[:-2,1:-1] )/(dx*dy)
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
	u_crossup[0,0] = .5*( 2*u[1,0] + 2*u[0,1] - 2*u[0,0] - 2*u[1,1])/(dx*dy) #southwest
	u_crossup[-1,0] = .5*( 2*u[-1,1] + 2*u[-2,0] - 2*u[-1,0] - 2*u[-2,1])/(dx*dy) #northwest
	u_crossup[0,-1] = .5*( 2*u[0,-2] + 2*u[1,-1] - 2*u[-1,-1] - 2*u[1,-2])/(dx*dy) #southeast
	u_crossup[-1,-1] = .5*( 2*u[-1,-2] + 2*u[-2,-1] - 2*u[-1,-1] - 2*u[-2,-2])/(dx*dy)#northeast
	u_crossdown[0,0] = .5*( 2*u[0,0] + 2*u[1,1] - 2*u[1,0] - 2*u[0,1])/(dx*dy)
	u_crossdown[-1,0] = .5*( 2*u[-1,0] + 2*u[-2,1] - 2*u[-1,1] - 2*u[-2,0])/(dx*dy)
	u_crossdown[0,-1] = .5*( 2*u[0,-1] + 2*u[1,-2] - 2*u[0,-2] - 2*u[1,-1])/(dx*dy)
	u_crossdown[-1,-1] = .5*( 2*u[-1,-1] + 2*u[-2,-2] - 2*u[-2,-1] - 2*u[-1,-2])/(dx*dy)
	zero = np.zeros((x.size,y.size))
	first = True
	BESTBUYS_X = np.zeros((I,J))
	BESTBUYS_Y = np.zeros((I,J))
	BBX_old = np.zeros((I,J))
	BBY_old = np.zeros((I,J))
	for i in range(scatters):
		cnt = 0
		while cnt<10:
			argument = app.map_2d_to_3d(BESTBUYS_Y,Search_x)
			#VALUEGRID = iF.Hamiltonian_vectorised(Search_x,argument,x,y,m,dx,dy,timez,I,J,Obstacles,u_south,u_north,u_west,u_east,u_crossup,u_crossdown,True)
			VALUEGRID_X = iF.Hamiltonian_vectorised(Search_x,argument,x,y,app.map_2d_to_3d(m,Search_x),dx,dy,timez,I,J,Obstacles,u_south,u_north,u_west,u_east,u_crossup,u_crossdown,True)
			ind1x = VALUEGRID_X.argmin(axis=1)
			ind0x,ind2x = np.indices(ind1x.shape)
			BBX_old = np.copy(BESTBUYS_X)
			BESTBUYS_X = Search_x[ind0x,ind1x,ind2x]
			argument = app.map_2d_to_3d(BESTBUYS_X,Search_y)
			VALUEGRID_Y = iF.Hamiltonian_vectorised(argument,Search_y,x,y,app.map_2d_to_3d(m,Search_y),dx,dy,timez,I,J,Obstacles,u_south,u_north,u_west,u_east,u_crossup,u_crossdown,True)
			ind1y = VALUEGRID_Y.argmin(axis=1)
			ind0y,ind2y = np.indices(ind1y.shape)
			BBY_old = np.copy(BESTBUYS_Y)
			BESTBUYS_Y = Search_y[ind0y,ind1y,ind2y]
			dev = np.linalg.norm(BBX_old-BESTBUYS_X)+np.linalg.norm(BBY_old-BESTBUYS_Y)
			if dev < 1e-6:
				break
			else:
				cnt+=1
				#print dev
		if first:
			search_x = np.empty((.5*search_x.size))
			Search_x,X,Y = np.meshgrid(search_x,x,y)
			search_y = np.empty((.5*search_y.size))
			Search_y,X,Y = np.meshgrid(search_y,x,y)
			first = False
		if i-1 is not scatters:
			Gradx = np.gradient(VALUEGRID_X)[1][ind0x,ind1x,ind2x] #gradients of control1
			Grady = np.gradient(VALUEGRID_Y)[1][ind0y,ind1y,ind2y] #gradients of control1
			for j in range(x.size):
				for k in range(y.size):
					if Gradx[j,k] > 0:
						Search_x[j,:,k] = np.linspace(BESTBUYS_X[j,k]-ax,BESTBUYS_X[j,k],N[0]/2)
					elif Gradx[j,k] < 0:
						Search_x[j,:,k] = np.linspace(BESTBUYS_X[j,k]-ax/2,BESTBUYS_X[j,k]+ax/2,N[0]/2)
					else:
						Search_x[j,:,k] = np.linspace(BESTBUYS_X[j,k]-ax/2,BESTBUYS_X[j,k]+ax/2,N[0]/2)
					if Grady[j,k] > 0:
						Search_y[j,:,k] = np.linspace(BESTBUYS_Y[j,k]-ay,BESTBUYS_Y[j,k],N[0]/2)
					elif Grady[j,k] < 0:
						Search_y[j,:,k] = np.linspace(BESTBUYS_Y[j,k]-ay/2,BESTBUYS_Y[j,k]+ay/2,N[0]/2)
					else:
						Search_y[j,:,k] = np.linspace(BESTBUYS_Y[j,k]-ay/2,BESTBUYS_Y[j,k]+ay/2,N[0]/2)
			ax = Search_x[0,1,0]-Search_x[0,0,0]
			ay = Search_y[0,1,0]-Search_y[0,0,0]
	return BESTBUYS_X,BESTBUYS_Y

def control_hybridO_vectorised_3D(search_x,search_y,x,y,u,m,dt,dx,dy,timez,I,J,tol,scatters,N,nulled,Obstacles,south,north,west,east):
	ax = search_x[1]-search_x[0]
	ay = search_y[1]-search_y[0]
	bounds = [search_x[0], search_x[-1], search_y[0], search_y[-1]]
	Search_x,X,Y = np.meshgrid(search_x,x,y) #this is 3D
	Search_y,X,Y = np.meshgrid(search_y,x,y) #this is also 3D
	FOUR_DEE = False
	#make the things
	u_south = np.zeros((J,I))
	u_north = np.zeros((J,I))
	u_west = np.zeros((J,I))
	u_east = np.zeros((J,I))
	u_crossup = np.zeros((J,I))
	u_crossdown = np.zeros((J,I))
	u = (np.reshape(u,(J,I)))
	m = (np.reshape(m,(J,I)))
	u_north[:-1,:] = (u[1:,:]-u[:-1,:])/dx
	u_south[1:,:] = (u[1:,:]-u[:-1,:])/dx
	u_east[:,:-1] = (u[:,1:]-u[:,:-1])/dy
	u_west[:,1:] = (u[:,1:]-u[:,:-1])/dy
	u_crossup[1:-1,1:-1] = .5*( u[1:-1,2:] + u[1:-1,:-2] + u[2:,1:-1] + u[:-2,1:-1] - 2*u[1:-1,1:-1] - u[:-2,2:] - u[2:,:-2] )/(dx*dy)
	u_crossdown[1:-1,1:-1] = .5*( 2*u[1:-1,1:-1] + u[2:,2:] + u[:-2,:-2] - u[1:-1,2:] - u[1:-1,:-2] - u[2:,1:-1] - u[:-2,1:-1] )/(dx*dy)
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
	u_crossup[0,0] = .5*( 2*u[1,0] + 2*u[0,1] - 2*u[0,0] - 2*u[1,1])/(dx*dy) #southwest
	u_crossup[-1,0] = .5*( 2*u[-1,1] + 2*u[-2,0] - 2*u[-1,0] - 2*u[-2,1])/(dx*dy) #northwest
	u_crossup[0,-1] = .5*( 2*u[0,-2] + 2*u[1,-1] - 2*u[-1,-1] - 2*u[1,-2])/(dx*dy) #southeast
	u_crossup[-1,-1] = .5*( 2*u[-1,-2] + 2*u[-2,-1] - 2*u[-1,-1] - 2*u[-2,-2])/(dx*dy)#northeast
	u_crossdown[0,0] = .5*( 2*u[0,0] + 2*u[1,1] - 2*u[1,0] - 2*u[0,1])/(dx*dy)
	u_crossdown[-1,0] = .5*( 2*u[-1,0] + 2*u[-2,1] - 2*u[-1,1] - 2*u[-2,0])/(dx*dy)
	u_crossdown[0,-1] = .5*( 2*u[0,-1] + 2*u[1,-2] - 2*u[0,-2] - 2*u[1,-1])/(dx*dy)
	u_crossdown[-1,-1] = .5*( 2*u[-1,-1] + 2*u[-2,-2] - 2*u[-2,-1] - 2*u[-1,-2])/(dx*dy)
	zero = np.zeros((x.size,y.size))
	BESTBUYS_X = np.zeros((I,J))
	BESTBUYS_Y = np.zeros((I,J))
	BBX_old = np.zeros((I,J))
	BBY_old = np.zeros((I,J))
	first = True
	for i in range(scatters):
		cnt = 0
		while cnt<10:
			argument = app.map_2d_to_3d(BESTBUYS_Y,Search_x)
			VALUEGRID_X = iF.Hamiltonian_vectorised(Search_x,argument,x,y,app.map_2d_to_3d(m,Search_x),dx,dy,timez,I,J,Obstacles,u_south,u_north,u_west,u_east,u_crossup,u_crossdown,True)
			ind1x = VALUEGRID_X.argmin(axis=1)
			ind0x,ind2x = np.indices(ind1x.shape)
			BBX_old = np.copy(BESTBUYS_X)
			BESTBUYS_X = Search_x[ind0x,ind1x,ind2x]
			argument = app.map_2d_to_3d(BESTBUYS_X,Search_y)
			VALUEGRID_Y = iF.Hamiltonian_vectorised(argument,Search_y,x,y,app.map_2d_to_3d(m,Search_y),dx,dy,timez,I,J,Obstacles,u_south,u_north,u_west,u_east,u_crossup,u_crossdown,True)
			ind1y = VALUEGRID_Y.argmin(axis=1)
			ind0y,ind2y = np.indices(ind1y.shape)
			BBY_old = np.copy(BESTBUYS_Y)
			BESTBUYS_Y = Search_y[ind0y,ind1y,ind2y]
			dev = np.linalg.norm(BBX_old-BESTBUYS_X)+np.linalg.norm(BBY_old-BESTBUYS_Y)
			if dev < 1e-6:
				break
			else:
				cnt+=1
				#print dev
		if first:
			search_x = np.empty((.5*search_x.size))
			Search_x,X,Y = np.meshgrid(search_x,x,y)
			search_y = np.empty((.5*search_y.size))
			Search_y,X,Y = np.meshgrid(search_y,x,y)
			first = False
		if i-1 is not scatters:
			Gradx = iF.Hamiltonian_derivative_vectorised(BESTBUYS_X,BESTBUYS_Y,x,y,m,dx,dy,timez,I,J,Obstacles,u_south,u_north,u_west,u_east,u_crossup,u_crossdown,False)
			Grady = iF.Hamiltonian_derivative_vectorised(BESTBUYS_X,BESTBUYS_Y,x,y,m,dx,dy,timez,I,J,Obstacles,u_south,u_north,u_west,u_east,u_crossup,u_crossdown,False)
			for j in range(x.size):
				for k in range(y.size):
					if Gradx[j,k] > 0:
						Search_x[j,:,k] = np.linspace(BESTBUYS_X[j,k]-ax,BESTBUYS_X[j,k],N[0]/2)
					elif Gradx[j,k] < 0:
						Search_x[j,:,k] = np.linspace(BESTBUYS_X[j,k]-ax/2,BESTBUYS_X[j,k]+ax/2,N[0]/2)
					else:
						Search_x[j,:,k] = np.linspace(BESTBUYS_X[j,k]-ax/2,BESTBUYS_X[j,k]+ax/2,N[0]/2)
					if Grady[j,k] > 0:
						Search_y[j,:,k] = np.linspace(BESTBUYS_Y[j,k]-ay,BESTBUYS_Y[j,k],N[0]/2)
					elif Grady[j,k] < 0:
						Search_y[j,:,k] = np.linspace(BESTBUYS_Y[j,k]-ay/2,BESTBUYS_Y[j,k]+ay/2,N[0]/2)
					else:
						Search_y[j,:,k] = np.linspace(BESTBUYS_Y[j,k]-ay/2,BESTBUYS_Y[j,k]+ay/2,N[0]/2)
			ax = Search_x[0,1,0]-Search_x[0,0,0]
			ay = Search_y[0,1,0]-Search_y[0,0,0]
	return BESTBUYS_X,BESTBUYS_Y


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

