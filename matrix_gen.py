from __future__ import division
import numpy as np
from scipy import sparse
import applications as app
import assembly as ass
import input_functions_2D as iF
import time as timer
import matplotlib.pyplot as plt
##################
# MATRIX GENERATION: HJB
##################

def HJB_diffusion_implicit(time,x,y,a1,a2,dx,dy,dt,south,north,west,east,nulled,se,sw,ne,nw): #this works, at least for constant diffusion and without D12; CORNERS ARE BAD FOR D12
	I,J = x.size,y.size
	output = sparse.lil_matrix((I*J,I*J))
	dx2 = dx**2
	dy2 = dy**2
	dxy = dx*dy
	#indices = np.delete(range(I*J),nulled)
	#indices = [n for n in list(range(I*J)) if n not in list(nulled)]
	indices = list(range(I*J))
	not_south = [n for n in indices if n not in list(south)]
	not_north = [n for n in indices if n not in list(north)]
	not_east = [n for n in indices if n not in list(east)]
	not_west = [n for n in indices if n not in list(west)]
	not_sw = np.intersect1d(not_south,not_west)
	not_se = np.intersect1d(not_south,not_east)
	not_nw = np.intersect1d(not_north,not_west)
	not_ne = np.intersect1d(not_north,not_east)
	se = np.intersect1d(np.intersect1d(south,east),se)
	sw = np.intersect1d(np.intersect1d(south,west),sw)
	ne = np.intersect1d(np.intersect1d(north,east),ne)
	nw = np.intersect1d(np.intersect1d(north,west),nw)
	D11 = iF.Sigma_D11_test(time,x,y,a1,a2)
	D22 = iF.Sigma_D22_test(time,x,y,a1,a2)
	D12 = iF.Sigma_D12_test(time,x,y,a1,a2)
	D11 = np.ravel(D11)
	D22 = np.ravel(D22)
	D12 = np.ravel(D12)
	zero = np.zeros(I*J)
	#generate the vectors
	F_here = np.zeros(I*J) 
	F_here[indices] = 1+dt/dx2*D11[indices] + dt/dy2*D22[indices]- dt/dxy*abs(D12[indices]) #[i,i]
	F_north = np.zeros(I*J) #[i,i+I]
	F_south = np.zeros(I*J) #[i,i-I]
	F_west = np.zeros(I*J) #[i,i-1]
	F_east = np.zeros(I*J) #[i,i+1]
	F_south_east = np.zeros(I*J) #[i,i-I+1]
	F_south_west = np.zeros(I*J) #[i,i-I-1]
	F_north_west = np.zeros(I*J) #[i,i+I-1]
	F_north_east = np.zeros(I*J) #[i,i+I+1]
	F_south[not_south] += -dt*(D22[not_south]/dy2-abs(D12[not_south])/dxy)/2
	F_north[not_north] += -dt*(D22[not_north]/dy2-abs(D12[not_north])/dxy)/2
	F_west[not_west] += -dt*(D11[not_west]/dx2-abs(D12[not_west])/dxy)/2
	F_east[not_east] += -dt*(D11[not_east]/dx2-abs(D12[not_east])/dxy)/2
	F_south_west[not_sw] += -dt/dxy*np.maximum(D12[not_sw],zero[not_sw])/2
	F_north_east[not_ne] += -dt/dxy*np.maximum(D12[not_ne],zero[not_ne])/2
	F_north_west[not_se] += dt/dxy*np.minimum(D12[not_se],zero[not_se])/2
	F_south_east[not_nw] += dt/dxy*np.minimum(D12[not_nw],zero[not_nw])/2
	#boundary stuff
	F_north[south] += -dt*(D22[south]/dy2-abs(D12[south])/dxy)/2
	F_north_east[np.intersect1d(south,not_east)] += dt/dxy*np.minimum(D12[np.intersect1d(south,not_east)],zero[np.intersect1d(south,not_east)])/2
	F_north_west[np.intersect1d(south,not_west)] += -dt/dxy*np.maximum(D12[np.intersect1d(south,not_west)],zero[np.intersect1d(south,not_west)])/2
	F_south[north] += -dt*(D22[north]/dy2-abs(D12[north])/dxy)/2
	F_south_west[np.intersect1d(north,not_west)] += dt/dxy*np.minimum(D12[np.intersect1d(north,not_west)],zero[np.intersect1d(north,not_west)])/2
	F_south_east[np.intersect1d(north,not_east)] += -dt/dxy*np.maximum(D12[np.intersect1d(north,not_east)],zero[np.intersect1d(north,not_east)])/2
	F_east[west] += -dt*(D11[west]/dx2-abs(D12[west])/dxy)/2
	F_north_east[np.intersect1d(west,not_north)] += dt/dxy*np.minimum(D12[np.intersect1d(west,not_north)],zero[np.intersect1d(west,not_north)])/2
	F_south_east[np.intersect1d(west,not_south)] += -dt/dxy*np.maximum(D12[np.intersect1d(west,not_south)],zero[np.intersect1d(west,not_south)])/2
	F_west[east] += -dt*(D11[east]/dx2-abs(D12[east])/dxy)/2
	F_north_west[np.intersect1d(east,not_north)] += -dt/dxy*np.maximum(D12[np.intersect1d(east,not_north)],zero[np.intersect1d(east,not_north)])/2
	F_south_west[np.intersect1d(east,not_south)] += dt/dxy*np.minimum(D12[np.intersect1d(east,not_south)],zero[np.intersect1d(east,not_south)])/2
	F_north_east[sw] += -dt/dxy*np.maximum(D12[sw],zero[sw])/2
	F_south_west[ne] += -dt/dxy*np.maximum(D12[ne],zero[ne])/2
	F_south_east[nw] += dt/dxy*np.minimum(D12[nw],zero[nw])/2
	F_north_west[se] += dt/dxy*np.minimum(D12[se],zero[se])/2
	#remove nulled things
#	if nulled!=None:
#		F_here[nulled] = 1
#		F_north[nulled] = 0
#		F_south[nulled] = 0
#		F_west[nulled] = 0
#		F_east[nulled] = 0
#		F_south_east[nulled] = 0
#		F_south_west[nulled] = 0
#		F_north_west[nulled] = 0
#		F_north_east[nulled] = 0
	output = sparse.diags([F_here, F_north[0:-I], F_south[I:], F_west[1:], F_east[0:-1], F_north_east[0:-I-1], F_south_west[(I+1):], F_north_west[0:-I+1], F_south_east[(I-1):]],[0, I, -I, -1, 1,I+1,-I-1,I-1,-I+1])
	return sparse.csr_matrix(output)


def HJB_convection_explicit(time,x,y,a1,a2,dx,dy,dt,south,north,west,east,nulled): #this should work, but also needs BC
	I,J = x.size,y.size
	[f1_array, f2_array] = iF.f_global(time,x,y,a1,a2)
	#xbound1 = range(0,I) 
	#ybound1 = range(0,I*J,I)
	#xbound2 = range(I*J-I,I*J)
	#ybound2 = range(I-1,I*J,I)
	f1_array = np.ravel(f1_array)
	f2_array = np.ravel(f2_array)
	zero = np.zeros(f1_array.size)
	F_here = np.zeros(I*J)
	F_north = np.zeros(I*J) #[i,i+I]
	F_south = np.zeros(I*J) #[i,i-I]
	F_west = np.zeros(I*J) #[i,i-1]
	F_east = np.zeros(I*J) #[i,i+1]
	f1min = np.minimum(f1_array,zero)
	f1max = np.maximum(f1_array,zero)
	f2min = np.minimum(f2_array,zero)
	f2max = np.maximum(f2_array,zero)
	#indices = np.array([ item for i,item in enumerate(range(I*J)) if i not in nulled ])
	#not_south = [ item for i,item in enumerate(indices) if i not in south ] 
	#not_north = [ item for i,item in enumerate(indices) if i not in north ]
	#not_west = [ item for i,item in enumerate(indices) if i not in west ]
	#not_east = [ item for i,item in enumerate(indices) if i not in east ]
	#indices = [n for n in list(range(I*J)) if n not in list(nulled)]
	indices = list(range(I*J))
	not_south = [n for n in indices if n not in list(south)]
	not_north = [n for n in indices if n not in list(north)]
	not_east = [n for n in indices if n not in list(east)]
	not_west = [n for n in indices if n not in list(west)]
	
	#finalise
	F_here[indices] = 1-dt*(abs(f1_array[indices])/dx+abs(f2_array[indices])/dy) #[i,i]
	F_south[not_south] += -dt/dy * f2min[not_south]
	F_north[not_north] += dt/dy * f2max[not_north]
	F_west[not_west] += - dt/dx * f1min[not_west]
	F_east[not_east] += dt/dx * f1max[not_east]
	#boundary stuff
	F_north[south] += -dt/dy*f2min[south]
	F_south[north] += dt/dy*f2max[north]
	F_east[west] += -dt/dx*f1min[west]
	F_west[east] += dt/dx*f1max[east]
	#remove nulled things
#	if nulled!=None:
#		F_here[nulled] = 1
#		F_north[nulled] = 0
#		F_south[nulled] = 0
#		F_west[nulled] = 0
#		F_east[nulled] = 0
	output = sparse.diags([F_here, F_north[0:-I], F_south[I:], F_west[1:], F_east[0:-1]],[0, I, -I, -1, 1])
	return sparse.csr_matrix(output)

#####################
#MATRIX GENERATION: FOKKER-PLANCK
#####################

def FP_convection_explicit_classic(time,x,y,a1,a2,dx,dt):
	I,J = x.size,y.size
	[f1, f2] = iF.f_global(time,x,y,a1,a2)
	xbound1 = range(0,I) 
	ybound1 = range(0,I*J,I)
	xbound2 = range(I*J-I,I*J)
	ybound2 = range(I-1,I*J,I)
	D11 = iF.Sigma_D11_test(time,x,y,a1,a2)
	D22 = iF.Sigma_D22_test(time,x,y,a1,a2)
	D12 = iF.Sigma_D12_test(time,x,y,a1,a2)
	D11x,D11y = np.gradient(D11,dx,dx)
	D22x,D22y = np.gradient(D22,dx,dx)
	D12x,D12y = np.gradient(D12,dx,dx)
	#make the fluxes
	F1 = -np.ravel(f1 - 0.5*D11x-0.5*D12y) #may be too simple... but it sure feels delish
	F2 = -np.ravel(f2 - 0.5*D12x-0.5*D22y)
	#make vectors
	#print f1
	#print ss
	zero = np.zeros(F1.size)
	here = np.zeros(I*J)
	here = 1-dt/dx*(abs(F1)+abs(F2)) #[i,i]
	north = np.zeros(I*J) #[i,i+I]
	south = np.zeros(I*J) #[i,i-I]
	west = np.zeros(I*J) #[i,i-1]
	east = np.zeros(I*J) #[i,i+1]
	F1min = np.minimum(F1,zero)
	F1max = np.maximum(F1,zero)
	F2min = np.minimum(F2,zero)
	F2max = np.maximum(F2,zero)
	indices = range(0,I*J)
	not_south = np.delete(indices,xbound1)
	not_north = np.delete(indices,xbound2)
	not_west = np.delete(indices,ybound1)
	not_east = np.delete(indices,ybound2)
	south[not_south] = -dt/dx*F2min[not_south-I]
	north[not_north] = dt/dx*F2max[not_north+I]
	east[not_east] = dt/dx*F1max[not_east+1]
	west[not_west] = -dt/dx*F1min[not_west-1]
	#for i in range(0,I*J):
	#	#then add boundary conditions #NO SUCH THINGS HERE, NO SIR
	#	if ismember(i,xbound1): #allows (i,j-1)
	#		north[i] += - dt/dx*F2min[i]
	#	if ismember(i,xbound2): #allows (i,j+1)
	#		south[i] += dt/dx*F2max[i]
	#	if ismember(i,ybound1): #allows (i-1,j) 
	#		east[i] += - dt/dx*F1min[i]
	#	if ismember(i,ybound2): #allows (i+1,j)
	#		west[i] += dt/dx*F1max[i]
	output = sparse.diags([here, north[0:-I], south[I:], west[1:], east[0:-1]],[0, I, -I, -1, 1])
	#print sparse.csr_matrix(output).sum(0)
	#print ss
	return sparse.csr_matrix(output)
def FP_convection_explicit_interpol(time,x,y,a1,a2,dx,dt,south,north,west,east,nulled):
	I,J = x.size,y.size
	[f1, f2] = iF.f_global(time,x,y,a1,a2)
	D11 = iF.Sigma_D11_test(time,x,y,a1,a2)
	D22 = iF.Sigma_D22_test(time,x,y,a1,a2)
	D12 = iF.Sigma_D12_test(time,x,y,a1,a2)
	D11x,D11y = np.gradient(D11,dx,dx)
	D22x,D22y = np.gradient(D22,dx,dx)
	D12x,D12y = np.gradient(D12,dx,dx)
	#make the fluxes
	
	#print f1.shape,f2.shape
	#print D11.shape,D11x.shape,D11y.shape
	#print D22.shape,D22x.shape,D22y.shape

	F1 = np.ravel(f1 - 0.5*D11x-0.5*D12y)#,order='F')
	F2 = np.ravel(f2 - 0.5*D12x-0.5*D22y)#,order='F')
	zero = np.zeros(F1.size)
	#indices = np.delete(range(I*J),nulled)
	#not_south = np.array([ item for i,item in enumerate(range(I*J)) if i not in south ])
	#not_north = np.array([ item for i,item in enumerate(range(I*J)) if i not in north ])
	#not_west = np.array([ item for i,item in enumerate(range(I*J)) if i not in west ])
	#not_east = np.array([ item for i,item in enumerate(range(I*J)) if i not in east ])
	#indices = np.array([ item for i,item in enumerate(range(I*J)) if i not in nulled ])
	indices = np.array([n for n in list(range(I*J)) if n not in list(nulled)])
	not_south = np.array([n for n in indices if n not in list(south)])
	not_north = np.array([n for n in indices if n not in list(north)])
	not_east = np.array([n for n in indices if n not in list(east)])
	not_west = np.array([n for n in indices if n not in list(west)])
	
	F_east = np.zeros(I*J)
	F_west = np.zeros(I*J)
	F_north = np.zeros(I*J)
	F_south = np.zeros(I*J)
	zero = np.zeros(I*J)
	F_east[not_east] = .5*(F1[not_east]+F1[not_east+1])
	F_west[not_west] = .5*(F1[not_west]+F1[not_west-1])
	F_north[not_north] = .5*(F2[not_north]+F2[not_north+I])
	F_south[not_south] = .5*(F2[not_south]+F2[not_south-I])
	Fl_here = np.zeros(I*J)
	Fl_here[indices] = 1-dt/dx*(-np.minimum(F_south[indices],zero[indices])-np.minimum(F_west[indices],zero[indices])+np.maximum(F_north[indices],zero[indices])+np.maximum(F_east[indices],zero[indices]))
	Fl_east = -dt/dx*np.minimum(F_east[:-1],zero[:-1])
	Fl_west = dt/dx*np.maximum(F_west[1:],zero[1:])
	Fl_north = -dt/dx*np.minimum(F_north[:-I],zero[:-I])
	Fl_south = dt/dx*np.maximum(F_south[I:],zero[I:])
#	if nulled!=None:
#		Fl_here[nulled] = 1
#		Fl_north[nulled] = 0
#		Fl_south[nulled] = 0
#		Fl_west[nulled] = 0
#		Fl_east[nulled] = 0
	output = sparse.diags([Fl_here, Fl_north, Fl_south, Fl_west, Fl_east],[0, I, -I, -1, 1])
	return sparse.csr_matrix(output)

def FP_diffusion_implicit_Ometh(time,x,y,a1,a2,dx,dt,south,north,west,east,nulled):
	I,J = x.size,y.size
	D11 = iF.Sigma_D11_test(time,x,y,a1,a2)
	D22 = iF.Sigma_D22_test(time,x,y,a1,a2)
	D12 = iF.Sigma_D12_test(time,x,y,a1,a2)
	lamb = dt/(dx**2)
	LHS = sparse.eye(I*J)#/lamb
	LHS = sparse.lil_matrix(LHS)
	LHS = add_diffusion_flux_Ometh(LHS,D11*np.ones((I,J)),D22*np.ones((I,J)),D12*np.ones((I,J)),I,J,dx,dt,0,south,north,west,east,nulled)
	return sparse.csr_matrix(LHS)#*lamb

def FP_diffusion_implicit_Nonlinear(time,x,y,a1,a2,dx,dt,m,south,north,west,east,nulled): #implicit
	I,J = x.size,y.size
	D11 = iF.Sigma_D11_test(time,x,y,a1,a2)
	D22 = iF.Sigma_D22_test(time,x,y,a1,a2)
	D12 = iF.Sigma_D12_test(time,x,y,a1,a2)
	lamb = dt/(dx**2)
	LHS = sparse.eye(I*J)
	LHS = sparse.lil_matrix(LHS)
	#AKu,ALu,AKr,ALr,LHS = add_diffusion_flux_nonlinear(LHS,D11*np.ones((I,J)),D22*np.ones((I,J)),D12*np.ones((I,J)),I,J,dx,dt,m,0)
	LHS = add_diffusion_flux_nonlinear(LHS,D11*np.ones((I,J)),D22*np.ones((I,J)),D12*np.ones((I,J)),I,J,dx,dt,m,0,south,north,west,east,nulled)
	#return AKu,ALu,AKr,ALr,sparse.csr_matrix(LHS)
	return sparse.csr_matrix(LHS)


def FP_diffusion_Nonlinear(time,x,y,a1,a2,dx,dt,m,south,north,west,east,nulled): #tried to optimise
	dx2 = dx**2
	I,J = x.size,y.size
	D11 = np.ravel(iF.Sigma_D11_test(time,x,y,a1,a2))
	D22 = np.ravel(iF.Sigma_D22_test(time,x,y,a1,a2))
	D12 = np.ravel(iF.Sigma_D12_test(time,x,y,a1,a2))
	m = np.ravel(m)
	#initialise shitloads of stuff
	indices = [n for n in list(range(I*J)) if n not in list(nulled)]
	#south = range(0,I) #in the future, we load these guys
	#west = range(0,I*J,I)#in the future, we load these guys
	#north = range(I*J-I,I*J)#in the future, we load these guys
	#east = range(I-1,I*J,I)#in the future, we load these guys
	not_south = np.array([ item for i,item in enumerate(range(I*J)) if i not in south ])
	not_north = np.array([ item for i,item in enumerate(range(I*J)) if i not in north ])
	not_west = np.array([ item for i,item in enumerate(range(I*J)) if i not in west ])
	not_east = np.array([ item for i,item in enumerate(range(I*J)) if i not in east ])
	not_ne = np.intersect1d(not_north,not_east)
	not_new = np.intersect1d(not_west,not_ne)
	not_nes = np.intersect1d(not_south,not_ne)
	m_p1_updown = np.zeros(I*J)
	m_p2_updown = np.zeros(I*J)
	m_p1_rightleft = np.zeros(I*J)
	m_p2_rightleft = np.zeros(I*J)
	mu1_updown = np.zeros(I*J)
	mu2_updown = np.zeros(I*J)
	mu1_rightleft = np.zeros(I*J)
	mu2_rightleft = np.zeros(I*J)
	AK_updown = np.zeros(I*J)
	AL_updown = np.zeros(I*J)
	AK_rightleft = np.zeros(I*J)
	AL_rightleft = np.zeros(I*J)
	a1_updown = np.zeros(I*J)
	a1_rightleft = np.zeros(I*J)
	a2_updown = np.zeros(I*J)
	a2_rightleft = np.zeros(I*J)
	#ratios
	ratio_up = np.minimum(D12,D22)/np.maximum(D12,D22)
	ratio_down = np.maximum(-D12,-D22)/np.minimum(-D12,-D22)
	ratio_right = np.minimum(D11,D12)/np.maximum(D11,D12) #this is vetted
	ratio_left = np.maximum(-D11,-D12)/np.minimum(-D11,-D12)
	#length of diffusion vector, ex D(K)*nK
	length_updown = np.sqrt(D12**2 + D22**2)
	length_rightleft = np.sqrt(D11**2 + D12**2)
	#length of intersection vectors, ex KO1
	intersect_up = .5*dx*np.sqrt(1+ratio_up) #|OP1| for upwards
	intersect_right = .5*dx*np.sqrt(1+ratio_right) #|OP1| for rightwards
	intersect_down = .5*dx*np.sqrt(1+ratio_down) #|OP2| for downwards
	intersect_left = .5*dx*np.sqrt(1+ratio_left) #|OP2| for leftwards
	#cell areas
	mK = np.ravel(np.ones(I*J)*dx2)
	mK[south] = mK[south]#*2
	mK[north] = mK[north]#*2
	mK[west] = mK[west]#*2
	mK[east] = mK[east]#*2
	#cell dimensions
	length_east = np.ones(I*J)*dx
	length_west = np.ones(I*J)*dx
	length_north = np.ones(I*J)*dx
	length_south  = np.ones(I*J)*dx
	length_east[south] = 0.5*length_east[south]
	length_east[north] = 0.5*length_east[north]
	length_west[south] = 0.5*length_west[south]
	length_west[north] = 0.5*length_west[north]
	length_north[east] = 0.5*length_north[east]
	length_north[west] = 0.5*length_north[west]
	length_south[east] = 0.5*length_south[east]
	length_south[west] = 0.5*length_south[west]
	
	#northbound coefficients
	m_p1_updown[not_new] = .25*(m[not_new]+m[not_new+I]+m[not_new-1]+m[not_new+I-1])
	m_p2_updown[not_new] = .25*(m[not_new]+m[not_new+I]+m[not_new+1]+m[not_new+I+1])
	west_not_north = np.intersect1d(west,not_north)
	east_not_north = np.intersect1d(east,not_north)
	m_p1_updown[west_not_north] = .25*(m[west_not_north]+m[west_not_north+1]+m[west_not_north+I]+m[west_not_north+I+1])
	m_p2_updown[west_not_north] = np.copy(m_p1_updown[west_not_north])
	m_p1_updown[east_not_north] = .25*(m[east_not_north]+m[east_not_north-1]+m[east_not_north+I]+m[east_not_north-I+1])
	m_p2_updown[east_not_north] = np.copy(m_p1_updown[east_not_north])
	
	#eastbound coefficients
	m_p1_rightleft[not_nes] = .25*(m[not_nes]+m[not_nes+1]+m[not_nes-I]+m[not_nes+1-I])
	m_p2_rightleft[not_nes] = .25*(m[not_nes]+m[not_nes+1]+m[not_nes+I]+m[not_nes+1+I])
	south_not_east = np.intersect1d(south,not_east)
	north_not_east = np.intersect1d(north,not_east)
	m_p1_rightleft[south_not_east] = .25*(m[south_not_east]+m[south_not_east+1]+m[south_not_east+I]+m[south_not_east+I+1])
	m_p2_rightleft[south_not_east] = np.copy(m_p1_rightleft[south_not_east])
	m_p1_rightleft[north_not_east] = .25*(m[north_not_east]+m[north_not_east+1]+m[north_not_east-I]+m[north_not_east-I+1])
	m_p2_rightleft[north_not_east] = np.copy(m_p1_rightleft[south_not_east])

	#coefficient things
	a1_updown[not_north] = length_north[not_north]*length_updown[not_north]/intersect_up[not_north] * ( (1+ratio_up[not_north])*m_p2_updown[not_north] + (1-ratio_up[not_north])*m_p1_updown[not_north] ) 
	a2_updown[not_north] = length_south[not_north+I]*length_updown[not_north+I]/intersect_down[not_north+I] * ( (1-ratio_down[not_north+I])*m_p2_updown[not_north] + (1+ratio_down[not_north+I])*m_p1_updown[not_north] )
	a1_rightleft[not_east] = length_east[not_east]*length_rightleft[not_east]/intersect_right[not_east] * ( (1+ratio_right[not_east])*m_p2_rightleft[not_east] + (1-ratio_right[not_east])*m_p1_rightleft[not_east] )
	a2_rightleft[not_east] = length_west[not_east+1]*length_rightleft[not_east+1]/intersect_left[not_east+1] * ((1-ratio_left[not_east+1])*m_p2_rightleft[not_east]+(1+ratio_left[not_east+1])*m_p1_rightleft[not_east]) 
	
	#rightleft
	temp2 = np.zeros(I*J)
	temp2[not_east] = a1_rightleft[not_east]+a2_rightleft[not_east]
	temp2i = np.where(temp2!=0)[0]
	temp2zeros = np.where(temp2==0)[0]
	mu1_rightleft[temp2i] = a2_rightleft[temp2i]/temp2[temp2i]
	mu2_rightleft[temp2i] = a1_rightleft[temp2i]/temp2[temp2i]
	mu1_rightleft[temp2zeros] = 0.5
	mu2_rightleft[temp2zeros] = 0.5
	#updown
	temp1 = np.zeros(I*J)
	temp1[not_north] = a1_updown[not_north]+a2_updown[not_north]
	temp1i = np.where(temp1!=0)[0]
	temp1zeros = np.where(temp1==0)[0]
	mu1_updown[temp1i] = a2_updown[temp1i]/temp1[temp1i]
	mu2_updown[temp1i] = a1_updown[temp1i]/temp1[temp1i]
	mu1_updown[temp1zeros] = 0.5
	mu2_updown[temp1zeros] = 0.5

	AK_updown[not_north] = length_north[not_north]*mu1_updown[not_north]*length_updown[not_north]/intersect_up[not_north] #*length_n
	AL_updown[not_north] = length_south[not_north]*mu2_updown[not_north]*length_updown[not_north+I]/intersect_down[not_north+I] #*length_n, remember [i+I] when with AK_updown
	AK_rightleft[not_east] = length_east[not_east]*mu1_rightleft[not_east]*length_rightleft[not_east]/intersect_right[not_east]
	AL_rightleft[not_east] = length_west[not_east]*mu2_rightleft[not_east]*length_rightleft[not_east+1]/intersect_left[not_east+1] #with [i+1]
	
	F_here = np.ones(I*J)
	F_north = np.zeros(I*J)
	F_south = np.zeros(I*J)
	F_east = np.zeros(I*J)
	F_west = np.zeros(I*J)
	F_here[not_north] += AK_updown[not_north]*dt/mK[not_north]
	F_here[not_north+I] += AL_updown[not_north]*dt/mK[not_north]
	F_here[not_east] += AK_rightleft[not_east]*dt/mK[not_east]
	F_here[not_east+1] += AL_rightleft[not_east]*dt/mK[not_east]
	F_north[not_north] += -AL_updown[not_north]*dt/mK[not_north]
	F_south[not_north] += -AK_updown[not_north]*dt/mK[not_north]
	F_east[not_east] += -AL_rightleft[not_east]*dt/mK[not_east]
	F_west[not_east] += -AK_rightleft[not_east]*dt/mK[not_east]
	
	#print ss
	output = sparse.diags([F_here, F_north[:-I], F_south[:-I], F_west[:-1], F_east[:-1]],[0, I, -I, -1, 1])
	#output = sparse.diags([F_here, F_north[:-I], F_south[I:], F_west[1:], F_east[:-1]],[0, I, -I, -1, 1])
	#output = sparse.diags([F_here, F_north[I:], F_south[:-I], F_west[:-1], F_east[1:]],[0, I, -I, -1, 1]) #stupid fucking idea
	#return AK_updown,AL_updown,AK_rightleft,AL_rightleft,sparse.csr_matrix(output)
	return sparse.csr_matrix(output)


def add_diffusion_flux_Ometh(output,D11,D22,D12,I,J,dx,dt,EXPLICIT,south,north,west,east,nulled):
	dx2 = dx**2
	#xbound1 = range(0,I)
	#ybound1 = range(0,I*J,I)
	#xbound2 = range(I*J-I,I*J)
	#ybound2 = range(I-1,I*J,I)
	D11 = np.ravel(D11)*dt/dx2
	D22 = np.ravel(D22)*dt/dx2
	D12 = np.ravel(D12)*dt/dx2
	indices = np.array([ item for i,item in enumerate(range(I*J)) if i not in nulled ])
	not_north = [ item for i,item in enumerate(indices) if i not in north ]
	not_east = [ item for i,item in enumerate(indices) if i not in east ]
	for i in np.intersect1d(not_north,not_east):
		a1,a2,a3,a4 = D11[i],D11[i+1],D11[i+I],D11[i+I+1]
		b1,b2,b3,b4 = D22[i],D22[i+1],D22[i+I],D22[i+I+1]
		c1,c2,c3,c4 = D12[i],D12[i+1],D12[i+I],D12[i+I+1]
		if ismember(i,south): #south
			a1,b1,c1 = a1*2,b1*2,c1*2
			a2,b2,c2 = a2*2,b2*2,c2*2
		if ismember(i,west): #west
			a1,b1,c1 = a1*2,b1*2,c1*2
			a3,b3,c3 = a3*2,b3*2,c3*2
		if ismember(i+I,north): #north
			a3,b3,c3 = a3*2,b3*2,c3*2
			a4,b4,c4 = a4*2,b4*2,c4*2
		if ismember(i+1,east): #east
			a2,b2,c2 = a2*2,b2*2,c2*2
			a4,b4,c4 = a4*2,b4*2,c4*2
		#as we believe it to be the diffusion tensor equation
		A = np.array([[a1+a2,0,c1,-c2],[0,a3+a4,-c3,c4],[c1,-c3,b1+b3,0],[-c2,c4,0,b2+b4]])
		B = np.array([[a1+c1,a2-c2,0,0],[0,0,a3-c3,a4+c4],[b1+c1,0,b3-c3,0],[0,b2-c2,0,b4+c4]])
		#print A
		if EXPLICIT==0:
			C = -np.array([[a1,0,c1,0],[0,-a4,0,-c4],[0,c3,-b3,0],[-c2,0,0,b2]]) #should be fine
			F = -np.array([[-a1-c1,0,0,0],[0,0,0,a4+c4],[0,0,-c3+b3,0],[0,c2-b2,0,0]]) #should be fine
		else:
			C = np.array([[a1,0,c1,0],[0,-a4,0,-c4],[0,c3,-b3,0],[-c2,0,0,b2]])
			F = np.array([[-a1-c1,0,0,0],[0,0,0,a4+c4],[0,0,-c3+b3,0],[0,c2-b2,0,0]])
		#finish up
		T = np.dot(C,np.dot(np.linalg.inv(A),B))+F #transmission coefficient matrix
		R = np.array([[1,0,1,0],[-1,0,0,1],[0,1,-1,0],[0,-1,0,-1]]) #contribution matrix
		output = ass.FVL2G(np.dot(R,T),output,i,I,J)
	return output

def FP_diffusion_flux_Diamond(time,x,y,a1,a2,dx,dy,dt,south,north,west,east,nulled): #this is implicit
	I,J = x.size,y.size
	D11 = iF.Sigma_D11_test(time,x,y,a1,a2)
	D22 = iF.Sigma_D22_test(time,x,y,a1,a2)
	D12 = iF.Sigma_D12_test(time,x,y,a1,a2)
	dx2 = dx**2
	dy2 = dy**2
	dxy = dx*dy
	#flatten out D12, D11, D22, f1_array, f2_array
	D11 = np.ravel(D11)
	D12 = np.ravel(D12)
	D22 = np.ravel(D22)
	#print D11
	#print ss
	indices = np.array([n for n in list(range(I*J)) if n not in list(nulled)])
	not_south = np.array([n for n in indices if n not in list(south)])
	not_north = np.array([n for n in indices if n not in list(north)])
	not_east = np.array([n for n in indices if n not in list(east)])
	not_west = np.array([n for n in indices if n not in list(west)])
	#not_south = np.array([ item for i,item in enumerate(range(I*J)) if i not in south ])
	#not_north = np.array([ item for i,item in enumerate(range(I*J)) if i not in north ])
	#not_west = np.array([ item for i,item in enumerate(range(I*J)) if i not in west ])
	#not_east = np.array([ item for i,item in enumerate(range(I*J)) if i not in east ])
	#indices = np.array([ item for i,item in enumerate(range(I*J)) if i not in nulled ])
	
	not_ne = np.array([n for n in indices if (n not in north) and (n not in east)])
	not_se = np.array([n for n in indices if (n not in south) and (n not in east)])
	not_nw = np.array([n for n in indices if (n not in north) and (n not in west)])
	not_sw = np.array([n for n in indices if (n not in south) and (n not in west)])
	
#	print not_north
#	print not_east
#	print not_ne
#	print ss
	#not_ne = np.intersect1d(not_north,not_east)
	#not_se = np.intersect1d(not_south,not_east)
	#not_nw = np.intersect1d(not_north,not_west)
	#not_sw = np.intersect1d(not_south,not_west)
	D_east = np.zeros(I*J)
	D_west = np.zeros(I*J)
	D_north = np.zeros(I*J)
	D_south = np.zeros(I*J)
	D_ne = np.zeros(I*J)
	D_nw = np.zeros(I*J)
	D_se = np.zeros(I*J)
	D_sw = np.zeros(I*J)
	here = np.zeros(I*J)
	#need the intersection between not_north and not_east, etc...
	D_ne[not_ne] += .25*(D12[not_ne+I]+D12[not_ne+1])*dt/dxy
	D_se[not_se] += -.25*(D12[not_se-I]+D12[not_se+1])*dt/dxy
	D_sw[not_sw] += .25*(D12[not_sw-I]+D12[not_sw-1])*dt/dxy
	D_nw[not_nw] += -.25*(D12[not_nw+I]+D12[not_nw-1])*dt/dxy
	D_north[not_north] += app.hmean(D22[not_north],D22[not_north+I])*dt/dy2
	D_south[not_south] += app.hmean(D22[not_south],D22[not_south-I])*dt/dy2
	D_west[not_west] += app.hmean(D11[not_west],D11[not_west-1])*dt/dx2
	D_east[not_east] += app.hmean(D11[not_east],D11[not_east+1])*dt/dx2

	#print D_north
	#print ss
	#boundary stuff
	#D_north[south] += app.hmean(D22[south],D22[south+I])*dt/dx2
	#D_south[north] += app.hmean(D22[north],D22[north-I])*dt/dx2
	#D_east[west] += app.hmean(D11[west],D11[west+1])*dt/dx2
	#D_west[east] += app.hmean(D11[east],D11[east-1])*dt/dx2
	here[indices] = 1+(D_north[indices]+D_south[indices]+D_west[indices]+D_east[indices]+D_ne[indices]+D_se[indices]+D_sw[indices]+D_nw[indices])
#	if nulled!=None:
#		here[nulled] = 1
#		D_north[nulled] = 0
#		D_south[nulled] = 0
#		D_west[nulled] = 0
#		D_east[nulled] = 0
#		D_ne[nulled] = 0
#		D_se[nulled] = 0
#		D_nw[nulled] = 0
#		D_sw[nulled] = 0
	output = sparse.diags([here, -D_north[:-I], -D_south[I:], -D_west[1:], -D_east[:-1], -D_ne[:-I-1], -D_sw[I+1:], -D_se[I-1:], -D_nw[:1-I]],[0, I, -I, -1, 1, I+1, -I-1, -I+1, I-1])
	return sparse.csr_matrix(output)


def add_direchlet_boundary(output,sol_vector,I,J,lamb,val):
	xbound1 = range(0,I)
	ybound1 = range(0,I*J,I)
	xbound2 = range(I*J-I,I*J)
	ybound2 = range(I-1,I*J,I)
	for i in range(0,I*J):
		if ismember(i,xbound1) or ismember(i,ybound1) or ismember(i,xbound2) or ismember(i,ybound2):
			#set output to exactly one
			output[i,:] = unit(i,I*J)/lamb
			sol_vector[i] = val
	return output,sol_vector

def trim_nulled(output,nulled):
	cols_to_keep = np.where(np.logical_not(np.in1d(np.arange(output.shape[1]), nulled)))[0]
	output = output[:, cols_to_keep]
	output = sparse.lil_matrix(output)
	for i in range(len(nulled)): #kill rows
		delete_row_lil(output,nulled[i])
	return sparse.csr_matrix(output)


def delete_row_lil(mat, i):
    if not isinstance(mat, sparse.lil_matrix):
        raise ValueError("works only for LIL format -- use .tolil() first")
    mat.rows = np.delete(mat.rows, i)
    mat.data = np.delete(mat.data, i)
    mat._shape = (mat._shape[0] - 1, mat._shape[1])

def unit(index,length):
	output = np.zeros(length)
	output[index] = 1
	return output

def ismember(a,array):
	for i in range(0,len(array)):
		if array[i]==a:
			return True
		elif array[i]>a:
			return False
	return False


#if not ismember(i,xbound1) and not ismember(i,xbound2) and not ismember(i,ybound1) and not ismember(i,ybound2):
#elif ismember(i,xbound1) and ismember(i,ybound1): #SOUTH-WEST
#elif ismember(i,xbound2) and ismember(i,ybound2): #NORTH-EAST
#elif ismember(i,xbound1) and ismember(i,ybound2): #SOUTH-EAST
#elif ismember(i,xbound2) and ismember(i,ybound1): #NORTH-WEST
#elif ismember(i,xbound1): #SOUTH
#elif ismember(i,xbound2): #NORTH
#elif ismember(i,ybound1): #WEST	
#elif ismember(i,ybound2): #EAST		
