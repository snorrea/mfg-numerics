#A collection of functions for getting GLL points and weights.
import numpy as np
import scipy.linalg as la

def Legendre(x, n):
    """Computes the Legendre polynomial
    of order n at point x.
    """
    Ln = np.zeros(n+1)
    Ln[0] = 1
    Ln[1] = x
    if (n>1):
        for i in range(1,n):
            Ln[i+1] = (2*i+1)/float(i+1)*x*Ln[i] - i/float(i+1) * Ln[i-1]

    return Ln[n]


def LegendreDerivative(x,n):
    """Computes the derivative of the Legendre
    polynomial of order n at x.
    """
    Ln = np.zeros(n+1)
    if (n==0):
        return 0.
    elif (n==1):
        return 1.
    elif (n>1):
        return n/(1-x**2)*Legendre(x,n-1) - n*x/(1-x**2)*Legendre(x,n)
    else:
        return -1

def GL_points_and_weights(n):
    """Calculates the GL nodes and weights.
    """
    A = np.zeros( (n,n) )
    p = np.zeros(n)
    w = np.zeros(n)

    #Assembling the A-matrix:
    A[0,1] = 1.
    if (n>2):
        for i in range(1,n-1):
            A[i,i-1] = i/float(2*i+1)
            A[i,i+1] = (i+1)/float(2*i+1)
    A[-1,-2] = (n-1)/float(2*n-1)

    #The array of the sorted eigenvalues:
    p = np.sort ( la.eig(A, right=False) ).real
    
    for i in range(n):
        w[i] = 2./( (1-p[i]**2)*(LegendreDerivative( p[i], n ))**2 )

    return p, w

def GLL_points(n):
    """Calculates the GLL nodes and weights.
    """
    tol = 10.**(-12)

    p = np.zeros(n)
    p[0] = -1.
    p[-1] = 1.

    w = np.zeros(n)

    if (n<3):
        return p

    #These points aare needed for the startvalues in the Newton iteration.
    GLpoints, GLweights = GL_points_and_weights(n-1)

    startvalues = np.zeros(n)
    startvalues[1:n-1] = 0.5 * ( GLpoints[:n-2] + GLpoints[1:n-1] )

    #This loop executes the Newton-iteration to find GLL points.
    for i in range(1,n-1):
        p[i] = startvalues[i]
        p_old = 0.
        while (abs(p[i]-p_old) > tol):
            p_old = p[i]
            L = Legendre(p_old, n-1)
            Ld = LegendreDerivative(p_old, n-1)
            p[i] = p_old + ( (1-p_old**2)*Ld)/( (n-1)*n*L )

    return p

def GLL_weights(n,p):
    """Returns GLL weights for points p.
    """
    w = np.zeros(n)
    for i in range(n):
        L = Legendre(p[i], n-1)
        w[i] = 2./( (n-1)*n*L**2)

    return w
