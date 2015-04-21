from __future__ import division
import numpy as np
import quadrature_nodes as qn
import input_functions_2D as iF
import matrix_gen as mg
from scipy.sparse.linalg import spsolve

#want a function that maps a 4x4 matrix into the domain... ish
def FVL2G(Local,Global,i,I,J):
	Local = -Local #this for implicit
	#Local is 4x4 array
	#Global is I*JxI*J
	#F1;
	Global[i,i] += Local[0,0]
	Global[i,i+1] += Local[0,1]
	Global[i,i+I] += Local[0,2]
	Global[i,i+I+1] += Local[0,3]
	#F2;  
	Global[i+1,i] += Local[1,0]
	Global[i+1,i+1] += Local[1,1]
	Global[i+1,i+I] += Local[1,2]
	Global[i+1,i+I+1] += Local[1,3]
	#F3; 
	Global[i+I,i] += Local[2,0]
	Global[i+I,i+1] += Local[2,1]
	Global[i+I,i+I] += Local[2,2]
	Global[i+I,i+I+1] += Local[2,3]
	#F4; 
	Global[i+I+1,i] += Local[3,0]
	Global[i+I+1,i+1] += Local[3,1]
	Global[i+I+1,i+I] += Local[3,2]
	Global[i+I+1,i+I+1] += Local[3,3]
	return Global

