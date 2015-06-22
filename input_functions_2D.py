from __future__ import division
import numpy as np
import math
from scipy import sparse

###################
#POLICY ITERATION FUNCTIONS
###################
def hamiltonian(ax_array,ay_array,x_array,y_array,u_array,m_array,dt,dx,dy,time,index_x,index_y,I,J,Obstacles,south,north,west,east): #spits out a 2D array of Hamiltonian values given arrays of inputs ax_array \times ay_array
	ind = index_x+index_y*I
	#print "Hamiltonian:", ax_array.shape,ay_array.shape
	#print ss
	zero = np.zeros(ax_array.shape)
	d11 = Sigma_D11_test(time,x_array[index_x],y_array[index_y],ax_array,ay_array) #all these bitches return matrices now
	d12 = Sigma_D12_test(time,x_array[index_x],y_array[index_y],ax_array,ay_array) #all these bitches
	d22 = Sigma_D22_test(time,x_array[index_x],y_array[index_y],ax_array,ay_array)
	f1,f2 = f_global(time,x_array[index_x],y_array[index_y],ax_array,ay_array) #and these also
	L = L_global(time,x_array[index_x],y_array[index_y],ax_array,ay_array,m_array[index_x+I*index_y],Obstacles[index_x+I*index_y])
	#print "Stuff:", d11.shape,d22.shape,f1.shape,L.shape,zero.shape,u_array.shape
	#print ss
	dx2 = dx**2
	dy2 = dy**2
	dxy = dx*dy
	xbound1 = south#range(0,I)
	ybound1 = west#range(0,I*J,I)
	xbound2 = north#range(I*J-I,I*J)
	ybound2 = east#range(I-1,I*J,I)
	tmp = u_array[ind]*( -(abs(f1)/dx+abs(f2)/dy) + ( abs(d12)/dxy-d11/dx2-d22/dy2 ) ) + L
	#avoid segfaults
	if not ismember(ind,xbound1): #allows (i,j-1)
		tmp += u_array[ind-I]*( -np.minimum(f2,zero)/dy + (d22/dy2-abs(d12)/dxy)/(2) )
	if not ismember(ind,xbound2): #allows (i,j+1)
		tmp += u_array[ind+I]*( np.maximum(f2,zero)/dy + (d22/dy2-abs(d12)/dxy)/(2) )
	if not ismember(ind,ybound1): #allows (i-1,j)
		tmp += u_array[ind-1]*( -np.minimum(f1,zero)/dx + (d11/dx2-abs(d12)/dxy)/(2) )
	if not ismember(ind,ybound2): #allows (i+1,j)
		tmp += u_array[ind+1]*( np.maximum(f1,zero)/dx + (d11/dx2-abs(d12)/dxy)/(2) )
	if not ismember(ind,xbound1) and not ismember(ind,ybound1): #allows (i-1,j-1)
		tmp += u_array[ind-1-I]*np.maximum(d12,0)/(2*dxy)
	if not ismember(ind,xbound2) and not ismember(ind,ybound2): #allows (i+1,j+1)
		tmp += u_array[ind+1+I]*np.maximum(d12,0)/(2*dxy)
	if not ismember(ind,xbound1) and not ismember(ind,ybound2): #allows (i+1,j-1)
		tmp += -u_array[ind+1-I]*np.minimum(d12,0)/(2*dxy)
	if not ismember(ind,ybound1) and not ismember(ind,xbound2): #allows (i-1,j+1)
		tmp += -u_array[ind-1+I]*np.minimum(d12,0)/(2*dxy)
	#then add boundary conditions
	if ismember(ind,xbound1): 
		tmp += u_array[ind+I]*( -np.minimum(f2,zero)/dy + (d22/dy2-d12/dxy)/(2) )
		if not ismember(ind,ybound1):
			tmp += np.maximum(d12,0)/(2*dxy)*u_array[ind+I-1]
		if not ismember(ind,ybound2):
			tmp += -np.minimum(d12,0)/(2*dxy)*u_array[ind+I+1]
	if ismember(ind,xbound2):
		tmp += u_array[ind-I]*( np.maximum(f2,zero)/dy + (d22/dy2-d12/dxy)/(2) )
		if not ismember(ind,ybound1):
			tmp += -np.minimum(d12,0)/(2*dxy)*u_array[ind-I-1]
		if not ismember(ind,ybound2):
			tmp += np.maximum(d12,0)/(2*dxy)*u_array[ind-I+1]
	if ismember(ind,ybound1):
		tmp += u_array[ind+1]*( -np.minimum(f1,zero)/dx + (d11/dx2-d12/dxy)/(2) )
		if not ismember(ind,xbound1):
			tmp+= np.maximum(d12,0)/(2*dxy)*u_array[ind-I+1] 
		if not ismember(ind,xbound2):
			tmp += -np.minimum(d12,0)/(2*dxy)*u_array[ind+I+1]
	if ismember(ind,ybound2):	
		tmp += u_array[ind-1]*( np.maximum(f1,zero)/dx + (d11/dx2-d12/dxy)/(2) )
		if not ismember(ind,xbound1):
			 tmp+= -np.minimum(d12,0)/(2*dxy)*u_array[ind-I-1]
		if not ismember(ind,xbound2):
			tmp+= np.maximum(d12,0)/(2*dxy)*u_array[ind+I-1]
	if ismember(ind,xbound1) and ismember(ind,ybound1): #allows (i-1,j-1)
		tmp += u_array[ind+1+I]*np.maximum(d12,0)/(2*dxy)
	if ismember(ind,xbound2) and ismember(ind,ybound2): #allows (i+1,j+1)
		tmp += u_array[ind-1-I]*np.maximum(d12,0)/(2*dxy)
	if ismember(ind,xbound1) and ismember(ind,ybound2): #allows (i+1,j-1)
		tmp += -u_array[ind-1+I]*np.minimum(d12,0)/(2*dxy)
	if ismember(ind,ybound1) and ismember(ind,xbound2): #allows (i-1,j+1)
		tmp += -u_array[ind+1-I]*np.minimum(d12,0)/(2*dxy)
	return tmp

def hamiltonian_derivative(ax,ay,x,y,u,m,dt,dx,dy,time,i,j,I,J,ObstacleCourse,south,north,west,east):
	ind = i+I*j
	if i!=0 and i!=I-1:
		u1_x = (u[ind]-u[ind-1])/dx
		u2_x = (u[ind+1]-u[ind])/dx
		u3_x = (u[ind+1]+u[ind-1]-2*u[ind])/(dx**2)
		u4_x = (u[ind+1]-u[ind-1])/(2*dx)
	elif i==0:
		u1_x = (u[ind]-u[ind+1])/dx
		u2_x = (u[ind+1]-u[ind])/dx
		u3_x = (u[ind+1]+u[ind+1]-2*u[ind])/(dx**2)
		u4_x = 0
	elif i==I-1:
		u1_x = (u[ind]-u[ind-1])/dx
		u2_x = (u[ind-1]-u[ind])/dx
		u3_x = (u[ind-1]+u[ind-1]-2*u[ind])/(dx**2)
		u4_x = 0
	if j!=0 and j!=J-1:
		u1_y = (u[ind]-u[ind-I])/dx
		u2_y = (u[ind+I]-u[ind])/dx
		u3_y = (u[ind+I]+u[ind-I]-2*u[ind])/(dx**2)
		u4_y = (u[ind+I]-u[ind-I])/(2*dx)
	elif j==0:
		u1_y = (u[ind]-u[ind+I])/dx
		u2_y = (u[ind+I]-u[ind])/dx
		u3_y = (u[ind+I]+u[ind+I]-2*u[ind])/(dx**2)
		u4_y = 0
	elif j==J-1:
		u1_y = (u[ind]-u[ind-I])/dx
		u2_y = (u[ind-I]-u[ind])/dx
		u3_y = (u[ind-I]+u[ind-I]-2*u[ind])/(dx**2)
		u4_y = 0
	return u4_x+ax, u4_y+ay

def Hamiltonian_vectorised(ax,ay,x,y,u,m,dx,dy,timez,I,J,Obstacles,U_south,U_north,U_west,U_east,U_crossup,U_crossdown): #spits out a 2D array of Hamiltonian values given arrays of inputs ax_array \times ay_array
	#all arguments except the obvious are assumed to be two- or three-dimensional
	zero = np.zeros(ax_array.shape)
	d11 = Sigma_D11_test(timez,x,y,ax,ay) #all these bitches return matrices now
	d12 = Sigma_D12_test(timez,x,y,ax,ay) #all these bitches
	d22 = Sigma_D22_test(timez,x,y,ax,ay)
	f1,f2 = f_global(timez,x,y,ax,ay) #and these also
	L = L_global(timez,x,y,ax,ay,m,Obstacles)
	return L + np.maximum(f1,zero)*U_west + np.minimum(f1,zero)*U_east + np.maximum(f2,zero)*U_south + np.minimum(f2,zero)*U_north + .5*d11*(U_east-U_west)/dx + .5*d22*(U_north-U_south)/dy + .5*np.maximum(d12,zero)*U_crossup + .5*np.minimum(d12,zero)*U_crossdown

###################
#RUNNING COST
###################
#def Obstacle(x,y,nulled):


def F_global(x,y,m_array,time): #more effective running cost function
	x,y = np.meshgrid(x,y)
	return np.sqrt((x-.8)**2 + (y-.4)**2)
	#return 0.1*m_array #shyness game
	#return (powerbill(time)*(1-0.8*x_array) + x_array/(0.1+m_array))
	#return 0*x_array#no-game

def L_global(time,x,y,a1,a2,m_array,G): #general cost
	#print a1
	#print a2
	#print ss
	b1 = .00
	b2 = 4
	b3 = .5
	c1 = .00
	c2 = 4
	c3 = .5
	x1 = xenophobia(b1,b2,b3,m_array)
	x2 = xenophobia(c1,c2,c3,m_array)
	#print a1.shape
	#print m_array.shape
	#(I,J) = a1.shape
	#m_array = np.reshape(m_array,(I,J))
	#G = np.reshape(G,(I,J))
	#a1 = np.ravel(a1)
	#a2 = np.ravel(a2)
#	print "Calling"
#	print x1.shape
#	print x2.shape
#	print a1.shape
#	print G.shape
	#return 0.5*(a1**2 + a2**2)*(1+x1) + G*(5 +x2) # + F_global(x,y,x,time)
	return 0.5*(a1**2+a2**2) + G#F_global(x_array,y_array,m_array,time)

def xenophobia(a,b,c,m):
	return a*(1+b*m)**c


def f_global(time,x_array,y_array,ax_array,ay_array):
	#return 0.1*a_array*x_array #Classic Robstad
	#x,y = np.meshgrid(x_array,y_array)
	#return [.1*x_array*y_array,.1*x_array*y_array] #FP test
	#c = .5
	#return [c*x,c*y]
	#return [c*x*np.exp(-time),c*y*np.exp(-time)] #FP test
	return [ax_array, ay_array] #standard MFG
	#return [ax_array*np.exp(-time), ay_array*np.exp(-time)] #standard MFG

def Sigma_D11_test(time,x,y,ax_array,ay_array):
	I = x.size
	J = y.size
	x,y = np.meshgrid(x,y)
	#return .5**2*np.ones(x.shape)
	out = .01*np.ones(x.shape)
	#for i in range(0,I):
	#	for j in range(0,J):
	#		if x[i,j] >= .5:
	#			out[i,j] = 2*out[i,j]
	#return abs(ax_array)*(.01 + h1(x))
	return out
def Sigma_D22_test(time,x,y,ax_array,ay_array):
	I = x.size
	J = y.size
	x,y = np.meshgrid(x,y)
	#return .5**2*np.ones(x.shape)
	out = .01*np.ones(x.shape)
	#for i in range(0,I):
	#	for j in range(0,J):
	#		if y[i,j] >= .5:
	#			out[i,j] = 2*out[i,j]
	return out
	#return abs(ay_array)*(.01 + h2(y))
def Sigma_D12_test(time,x,y,ax_array,ay_array):
	#I,J = x.size,y.size
	D11 = Sigma_D11_test(time,x,y,ax_array,ay_array)
	D22 = Sigma_D22_test(time,x,y,ax_array,ay_array)
	return 0*np.sqrt(D11*D22)
	#out = 0.00125*(2+x+y)#+1e-3 #FP test mean
	#out = 0.0001*np.ones(x.shape)
	#for i in range(0,I):
	#	for j in range(0,J):
	#		if y[i,j] >= .5 and x[i,j] >= .5:
	#			out[i,j] = 2*out[i,j]
	#return out
	#return .0**2*np.ones(x.shape) #FP test nice

def h1(x):
	out = np.zeros(x.shape)
	for i in range(x.shape[0]):
		for j in range(x.shape[1]):
			if x[i,j] < .5:
				out[i,j] = .01
	return out

def h2(x):
	out = np.zeros(x.shape)
	for i in range(x.shape[0]):
		for j in range(x.shape[1]):
			if x[i,j] < .5:
				out[i,j] = .01
	return out

##################
#TERMINAL COST
##################
def terminal_cost(x_array,y_array,m_array): #this is the final cost, and is a function of the entire distribution m and each point x_i
	#return -0.5*(x_array+0.5)**2 * (1.5-x_array)**2 #Carlini's original
	#return 0.1*(x_array*(1-x_array))**2 #Gueant's game
	#return -((x_array+0.2)*(1.2-x_array))**4 #Shyness game
	return np.zeros(x_array.size**2) #Carlini's no-game & Isolation game
	#return 0.001*m_array

##################
#INITIAL DISTRIBUTION
##################
def initial_distribution(x,y): #this now works
	#x,y = np.meshgrid(x,y)
	#return np.exp( -(x-0.)**2/(0.2**2) - (y-0.)**2/(0.2**2) )
	m0 = np.zeros((x.size,y.size))
	for i in range(x.size):
		for j in range(y.size):
			if x[i] <= .9 and y[j] <= .3 and x[i] >= 0.1 and y[j] >=0.1:
				m0[i,j] = .1
	return np.transpose(m0)
	#return np.exp( -(x-0.5)**2/(0.1**2) - (y-0.5)**2/(0.1**2) )
	#return 1-0.2*np.cos(np.pi*x) #gueant's original
	#return np.exp(-(x-0.75)**2/0.1**2) #carlini's no-game
	#return np.exp(-(np.outer(x,y)-0.5)**2/0.1**2) #shyness game
	#return np.exp(-(x-0.3)**2/0.1**2) #isolation game
	#return 1/(3*x+1)**3 #isolation game 2

def known_diffusion1(x,y,D11,D12,D22,time):
	x,y = np.meshgrid(x,y)
	det = D11*D22-D12**2
	#print "Culprit"
	#print x*y
	#print ss
	print np.exp( ( D22*x**2+D11*y**2-2*D12*x*y )/(4*det*time) )
	return 1/(4*math.pi*np.sqrt(det)*time) * np.exp( ( D22*x**2+D11*y**2-2*D12*x*y )/(4*det*time) )

def known_diffusion2(x,y,D11,D12,D22,time):
	I = len(x)
	J = len(y)
	output = np.empty((I,J))
	for i in range(0,I):
		for j in range(0,J):
			if abs(x[i])>0.25 or abs(y[j])>0.25:
				output[i,j] = 1
			else:
				output[i,j] = 0
	return output

#OTHER STUFF
#def ismember(a, b):
#   bind = {}
#    for i, elt in enumerate(b):
#        if elt not in bind:
#            bind[elt] = i
#    return [bind.get(itm, None) for itm in a]  # None can be replaced by any other "not in b" value

def ismember(a,array): #assumes array is sorted
	for i in range(0,len(array)):
		if array[i]==a:
			return True
		elif array[i]>a:
			return False
	return False

