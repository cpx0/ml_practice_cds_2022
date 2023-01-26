
import numpy as np


# Implementation of the first eleven Legendre polynomials.

def leg_0(x):
    return np.ones(x.shape, dtype=x.dtype)

def leg_1(x):
    return x

def leg_2(x):
    return (3*x**2 - 1)/2

def leg_3(x):
    return (5*x**3 - 3*x)/2

def leg_4(x):
    return (35*x**4 - 30*x**2 + 3)/8

def leg_5(x):
    return (63*x**5 - 70*x**3 + 15*x)/8

def leg_6(x):
    return (231*x**6 - 315*x**4 + 105*x**2 - 5)/16

def leg_7(x):
    return (429*x**7 - 693*x**5 + 315*x**3 - 35*x)/16

def leg_8(x):
    return (6435*x**8 - 12012*x**6 + 6930*x**4 - 1260*x**2 + 35)/128

def leg_9(x):
    return (12155*x**9 - 25740*x**7 + 18018*x**5 - 4620*x**3 + 315*x)/128

def leg_10(x):
    return (46189*x**10 - 109395*x**8 + 90090*x**6 - 30030*x**4 + 3465*x**2 - 63)/256


def leg(x, deg):
    '''
    General-purpose function for computing Legendre
    polynomials of degree=deg.
    
    Input: x is assumed to be an (n x 1) numpy array.
    '''
    return np.hstack((leg_0(x=x), leg_1(x=x), leg_2(x=x),
                      leg_3(x=x), leg_4(x=x), leg_5(x=x),
                      leg_6(x=x), leg_7(x=x), leg_8(x=x),
                      leg_9(x=x), leg_10(x=x)))[:,0:(deg+1)]





