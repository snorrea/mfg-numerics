from __future__ import division
import numpy as np
import glob, os



dx = 0.02
DT = 0.2
dt = DT*dx
x = np.linspace(0,1,int(1/dx+1))
t = np.linspace(0,1,int(1/dt+1))
xm,tm = np.meshgrid(x,t)
soln = np.exp(-tm)*xm*(1-x)
eps = 0.003001



######file stuff
####
#ADD A STATEMENT TO CHECK THE LATEST THING WITH THESE PARAMETERS
####


dx_string = "%.6f" % dx
dt_string = "%.6f" % DT
eps_string = "%.6f" % eps

filename = "test_" + dx_string + "_" + dt_string + "_" + eps_string + "_" + ".txt"


os.chdir("./")

#print ss
DO = True
for file in glob.glob("*.txt"):
	pop = file.split("_")
	pop_dx = float(pop[1])
	pop_DT = float(pop[2])
	pop_eps = float(pop[3])
	print pop_dx,pop_DT,pop_eps
	print pop_dx==dx,pop_DT==DT,pop_eps==eps
	#print ss
	if file==filename:
		print "File with these parameters already exist! Aborting file storage."
		DO = False

if DO:
	print "Saving solution in txt-file with name:", filename
	np.savetxt(filename, soln, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ')
	print "Success!"

#print ss
soln = None
print "Solution deleted..."
print "Loading solution from file"
soln = np.loadtxt(filename)
print "Success!"
