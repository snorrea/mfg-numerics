from __future__ import division
import numpy as np
import math
from scipy import sparse

###################
#POLICY ITERATION FUNCTIONS
###################
def hamiltonian(ax_array,ay_array,x_array,y_array,u_array,m_array,dt,dx,dy,time,index_x,index_y,I,J): #spits out a 2D array of Hamiltonian values given arrays of inputs ax_array \times ay_array
	ind = index_x+index_y*I
	#print "Hamiltonian:", ax_array.shape,ay_array.shape
	#print ss
	zero = np.zeros(ax_array.shape)
	d11 = Sigma_D11_test(time,x_array[index_x],y_array[index_y],ax_array,ay_array,m_array[ind]) #all these bitches return matrices now
	d12 = Sigma_D12_test(time,x_array[index_x],y_array[index_y],ax_array,ay_array,m_array[ind]) #all these bitches
	d22 = Sigma_D22_test(time,x_array[index_x],y_array[index_y],ax_array,ay_array,m_array[ind])
	f1,f2 = f_global(time,x_array[index_x],y_array[index_y],ax_array,ay_array) #and these also
	L = L_global(time,x_array[index_x],y_array[index_y],ax_array,ay_array,m_array)
	#print "Stuff:", d11.shape,d22.shape,f1.shape,L.shape,zero.shape,u_array.shape
	#print ss
	dx2 = dx**2
	dy2 = dy**2
	dxy = dx*dy
	xbound1 = range(0,I)
	ybound1 = range(0,I*J,I)
	xbound2 = range(I*J-I,I*J)
	ybound2 = range(I-1,I*J,I)
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

###################
#RUNNING COST
###################
def F_global(x,y,m_array,time): #more effective running cost function
	#return (x-0.2)**2 + (y-0.2)**2
	#return 0.1*m_array #shyness game
	#return (powerbill(time)*(1-0.8*x_array) + x_array/(0.1+m_array))
	D11 = Sigma_D11_test(time,x,y,x,x,x)
	D22 = Sigma_D22_test(time,x,y,x,x,x)
	D12 = Sigma_D12_test(time,x,y,x,x,x)
	[f1, f2] = f_global(time,x,y,x,y)
	x,y = np.meshgrid(x,y)
	#return 0*x
	sinx = np.sin(x)
	cosx = np.cos(x)
	siny = np.sin(y)
	cosy = np.cos(y)
	et = np.exp(-time)
	#return (1+D11/2+D22/2)*et*cosx*cosy + 0.5*et**2 * ( (sinx*cosy)**2 + (cosx*siny)**2) - D12*et*sinx*siny #HJB exact test
	return et*cosx*cosy*(-1-.5*D11-.5*D22 + .1*x+.1*y ) - et*sinx*cosy*f1 - et*cosx*siny*f2 + D12*et*sinx*siny
	#return 0*x_array#no-game

def L_global(time,x,y,a1,a2,m_array): #general cost
	return 0.5*(a1**2 + a2**2) + F_global(x,y,x,time)
	#return 0.5*(ax_array+ay_array)**2 + F_global(x_array,y_array,m_array,time)


def f_global(time,x_array,y_array,ax_array,ay_array):
	#return 0.1*a_array*x_array #Classic Robstad
	x,y = np.meshgrid(x_array,y_array)
	return [.1*x_array*y_array,.1*x_array*y_array] #FP test
	#return [ax_array, ay_array] #standard MFG

def Sigma_D11_test(time,x,y,ax_array,ay_array,m_array):
	x,y = np.meshgrid(x,y)
	return .5**2*np.ones(x.shape)
def Sigma_D22_test(time,x,y,ax_array,ay_array,m_array):
	x,y = np.meshgrid(x,y)
	return .5**2*np.ones(x.shape)
def Sigma_D12_test(time,x,y,ax_array,ay_array,m_array):
	x,y = np.meshgrid(x,y)
	return .0**2*np.ones(x.shape)

##################
#TERMINAL COST
##################
def G(x_array,y_array,m_array): #this is the final cost, and is a function of the entire distribution m and each point x_i
	#return -0.5*(x_array+0.5)**2 * (1.5-x_array)**2 #Carlini's original
	#return 0.1*(x_array*(1-x_array))**2 #Gueant's game
	#return -((x_array+0.2)*(1.2-x_array))**4 #Shyness game
	return np.zeros(x_array.size**2) #Carlini's no-game & Isolation game
	#return 0.001*m_array

##################
#INITIAL DISTRIBUTION
##################
def initial_distribution(x,y): #this now works
	x,y = np.meshgrid(x,y)
	#return np.exp( -(x-0.)**2/(0.2**2) - (y-0.)**2/(0.2**2) )
	return np.exp( -(x-0.5)**2/(0.1**2) - (y-0.5)**2/(0.1**2) )
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

