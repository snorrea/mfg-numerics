from __future__ import division
import numpy as np
import math
from scipy import sparse

###################
#POLICY ITERATION FUNCTIONS
###################

def hamiltonian(ax_array,ay_array,x_array,y_array,u_array,m_array,dt,dx,time,index_x,index_y,I,J): #spits out a 2D array of Hamiltonian values given arrays of inputs ax_array \times ay_array
	ind = index_x+index_y*I
	#print ax_array.size,ay_array.size
	zero = np.zeros((ax_array.size,ay_array.size))
	d11 = Sigma_D11(time,x_array[index_x],y_array[index_y],ax_array,ay_array,m_array[ind]) #all these bitches return matrices now
	d12 = Sigma_D12(time,x_array[index_x],y_array[index_y],ax_array,ay_array,m_array[ind]) #all these bitches
	d22 = Sigma_D22(time,x_array[index_x],y_array[index_y],ax_array,ay_array,m_array[ind])
	f1,f2 = f_test(time,x_array[index_x],y_array[index_y],ax_array,ay_array) #and these also
	L = L_test(time,x_array[index_x],y_array[index_y],ax_array,ay_array,m_array)
	dx2 = dx**2
	xbound1 = range(0,I)
	ybound1 = range(0,I*J,I)
	xbound2 = range(I*J-I,I*J)
	ybound2 = range(I-1,I*J,I)
	tmp = u_array[ind]*( -(abs(f1)+abs(f2))/dx + ( abs(d12)-d11-d22 )/dx2 ) + L
	#avoid segfaults
	if not ismember_sorted(ind,xbound1): #allows (i,j-1)
		tmp += u_array[ind-I]*( -np.minimum(f2,zero)/dx + (d22-abs(d12))/(2*dx2) )
	if not ismember_sorted(ind,xbound2): #allows (i,j+1)
		tmp += u_array[ind+I]*( np.maximum(f2,zero)/dx + (d22-abs(d12))/(2*dx2) )
	if not ismember_sorted(ind,ybound1): #allows (i-1,j)
		tmp += u_array[ind-1]*( -np.minimum(f1,zero)/dx + (d11-abs(d12))/(2*dx2) )
	if not ismember_sorted(ind,ybound2): #allows (i+1,j)
		tmp += u_array[ind+1]*( np.maximum(f1,zero)/dx + (d11-abs(d12))/(2*dx2) )
	if not ismember_sorted(ind,xbound1) and not ismember_sorted(ind,ybound1): #allows (i-1,j-1)
		tmp += u_array[ind-1-I]*np.maximum(d12,0)/(2*dx2)
	if not ismember_sorted(ind,xbound2) and not ismember_sorted(ind,ybound2): #allows (i+1,j+1)
		tmp += u_array[ind+1+I]*np.maximum(d12,0)/(2*dx2)
	if not ismember_sorted(ind,xbound1) and not ismember_sorted(ind,ybound2): #allows (i+1,j-1)
		tmp += -u_array[ind+1-I]*np.minimum(d12,0)/(2*dx2)
	if not ismember_sorted(ind,ybound1) and not ismember_sorted(ind,xbound2): #allows (i-1,j+1)
		tmp += -u_array[ind-1+I]*np.minimum(d12,0)/(2*dx2)
	#then add boundary conditions
	if ismember_sorted(ind,xbound1): 
		tmp += u_array[ind+I]*( -np.minimum(f2,zero)/dx + (d22-d12)/(2*dx2) )
	if ismember_sorted(ind,xbound2):
		tmp += u_array[ind-I]*( np.maximum(f2,zero)/dx + (d22-d12)/(2*dx2) )
	if ismember_sorted(ind,ybound1):
		tmp += u_array[ind+1]*( -np.minimum(f1,zero)/dx + (d11-d12)/(2*dx2) )
	if ismember_sorted(ind,ybound2):	
		tmp += u_array[ind-1]*( np.maximum(f1,zero)/dx + (d11-d12)/(2*dx2) )
	if ismember_sorted(ind,xbound1) and ismember_sorted(ind,ybound1): #allows (i-1,j-1)
		tmp += u_array[ind+1+I]*np.maximum(d12,0)/(2*dx2)
	if ismember_sorted(ind,xbound2) and ismember_sorted(ind,ybound2): #allows (i+1,j+1)
		tmp += u_array[ind-1-I]*np.maximum(d12,0)/(2*dx2)
	if ismember_sorted(ind,xbound1) and ismember_sorted(ind,ybound2): #allows (i+1,j-1)
		tmp += -u_array[ind-1+I]*np.minimum(d12,0)/(2*dx2)
	if ismember_sorted(ind,ybound1) and ismember_sorted(ind,xbound2): #allows (i-1,j+1)
		tmp += -u_array[ind+1-I]*np.minimum(d12,0)/(2*dx2)
	#print tmp
	#print ss
	#if np.amax(tmp) > 100:
	#	print "(i,j,k)=(",index_x,index_y,time/dt,")"
	#	print "u_array",u_array
	#	print "L",L
	#	print "Tmp",tmp
	#	print ss
	return tmp

###################
#RUNNING COST
###################
def F_global(x,y,m_array,time): #more effective running cost function
	x,y = np.meshgrid(x,y)
	return (x-0.2)**2 + (y-0.2)**2
	#return 0.1*m_array #shyness game
	#return (powerbill(time)*(1-0.8*x_array) + x_array/(0.1+m_array))
	#return 0*x_array#no-game

def L_global(time,x,y,ax_array,ay_array,m_array): #general cost
	x,y = np.meshgrid(x,y)
	return 0.5*(ax_array**2 + ay_array**2) + (x-0.0)**2 + (y-0.0)**2
	#output = np.empty(x_array.size,y_array.size)
	#for i in range (0,y_array.size):
	#	output[:,i] = 0.5*(ax_array**2 + ay_array**2) + (x-0.2)**2 * (y-0.2)**2 #location bias
	#return output
	#return 0.5*(ax_array+ay_array)**2 + F_global(x_array,y_array,m_array,time)

def L_test(time,x,y,ax_array,ay_array,m): #only dof is alphas
	a1,a2 = np.meshgrid(ax_array,ay_array)
	return 0.5*(a1**2+a2**2) + (x-0.0)**2 + (y-0.0)**2
	
def f_test(time,x,y,ax_array,ay_array):
	a1,a2 = np.meshgrid(ax_array,ay_array)
	return a1,a2
	#output1 = np.empty((ax_array.size,ay_array.size))
	#output2 = np.empty((ay_array.size,ax_array.size))
	#for i in range (0,ay_array.size):
	#	output1[:,i] = ax_array #location bias
	#	output2[i,:] = ay_array #location bias
	#return output1, output2

def f_global(time,x_array,y_array,ax_array,ay_array):
	#return 0.1*a_array*x_array #Classic Robstad
	return [ax_array, ay_array] #standard MFG

def Sigma_D11_test(time,x,y,ax_array,ay_array,m_array):
	return 0.3**2*np.ones((ax_array.size,ay_array.size))
def Sigma_D22_test(time,x,y,ax_array,ay_array,m_array):
	return 0.3**2*np.ones((ax_array.size,ay_array.size))
def Sigma_D12_test(time,x,y,ax_array,ay_array,m_array):
	return 0.1**2*np.ones((ax_array.size,ay_array.size))

def Sigma_D11(time,x,y,ax,ay,m):
	return 0.3**2
def Sigma_D22(time,x,y,ax,ay,m):
	return 0.3**2
def Sigma_D12(time,x,y,ax,ay,m):
	return 0.00**2
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
def ismember(a, b):
    bind = {}
    for i, elt in enumerate(b):
        if elt not in bind:
            bind[elt] = i
    return [bind.get(itm, None) for itm in a]  # None can be replaced by any other "not in b" value

def ismember_sorted(a,array):
	for i in range(0,len(array)):
		if array[i]==a:
			return True
		elif array[i]>a:
			return False
	return False

