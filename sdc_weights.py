# from sympy import *
import numpy as np
from fractions import Fraction
import scipy.integrate as integrate
from quadrature_weights import get_quadrature_weights
# x = symbols('x')

# num_x_vals = 3
# P = 1/2*(3*x**2-1)
# x_vals = [-1, 0, 1]

# num_x_vals = 4
# P = 0.5*(5.0*x**3.0-3.0*x) # define
# x_vals = [-1.0,-sqrt(1.0/5.0),sqrt(1.0/5.0),1.0] # define

# num_x_vals = 5

# x_vals = np.array([-1.0,-np.sqrt(3.0/7.0),0.0,np.sqrt(3.0/7.0),1.0])

# def P(x):
# 	return 1.0/8.0*(35.0*x**4.0-30.0*x**2.0+3.0)
# def P_prime(x):
# 	return 17.5*x**3.0-7.5*x
# def P_2prime(x):
# 	return 52.5*x**2-7.5

num_x_vals = 6

x_vals = np.array([-1.0,-np.sqrt(1.0/3.0+2.0*np.sqrt(7)/21),-np.sqrt(1.0/3.0-2.0*np.sqrt(7)/21),np.sqrt(1.0/3.0-2.0*np.sqrt(7)/21),np.sqrt(1.0/3.0+2.0*np.sqrt(7)/21),1.0])

def P(x):
	return 1.0/8.0*(63.0*x**5.0-70.0*x**3.0+15.0*x)
def P_prime(x):
	return -26.25*x**2.0 + 39.375*x**4.0 + 1.875
def P_2prime(x):
	return -52.5*x**1.0 + 157.5*x**3.0


# num_x_vals = 7

# x_vals = np.array([-1.0, -np.sqrt(5/11+2/11*np.sqrt(5/3)), -np.sqrt(5/11-2/11*np.sqrt(5/3)), 0, np.sqrt(5/11-2/11*np.sqrt(5/3)), np.sqrt(5/11+2/11*np.sqrt(5/3)), 1.0])
# def P(x):
# 	return 1/16*(231*x**6-315*x**4+105*x**2-5)
# def P_prime(x):
# 	return 1/16*(210*x-1260*x**3+1386*x**5)
# def P_2prime(x):
# 	return 1/16*(210-3780*x**2+6930*x**4)


def integral0(x0, xf):
	return integrate.quad(lambda x: (1-x)*P_prime(x), x0, xf)[0]

def integralM(x0, xf):
	return integrate.quad(lambda x: (1+x)*P_prime(x), x0, xf)[0]

def integrali(x0, xf, x_i):
	return integrate.quad(lambda x: (x**2-1.0)*P_prime(x)/(x-x_i), x0, xf)[0]

q = np.zeros((num_x_vals-1,num_x_vals))

# print(1.0/(4.0*P_prime(-1.0)))
# print(integrate.quad(integrand0, x_vals[0], x_vals[1]))
for m in range(0,num_x_vals-1):
	# print(x_vals[m], x_vals[m+1])
	# print(integral0(x_vals[m], x_vals[m+1]))
	q[m,0] = 1.0/(4.0*P_prime(-1.0)) * integral0(x_vals[m], x_vals[m+1])
	q[m,num_x_vals-1] = 1.0/(4.0*P_prime(1.0)) * integralM(x_vals[m], x_vals[m+1])
	
	for i in range(1,num_x_vals-1):
		q[m,i] = 0.5/((x_vals[i]**2-1.0)*P_2prime(x_vals[i])) * integrali(x_vals[m], x_vals[m+1], x_vals[i])

# for m in range(0,num_x_vals-1):
	# q[m,0] = 1.0/(4.0*P_prime.subs(x,-1.0))*integrate((1.0-x)*P_prime, (x, x_vals[m], x_vals[m+1])) 
	# q[m,num_x_vals-1] = 1.0/(4.0*P_prime.subs(x,1.0))*integrate((1.0+x)*P_prime, (x, x_vals[m], x_vals[m+1]))
	# for i in range(1,num_x_vals-1):
		# print(x_vals[i])
		# print(((x**2-1)*P_prime)/(x-x_vals[i]))
		# q[m,i] = 0.5/((x_vals[i]**2.0-1.0)*P_2prime.subs(x,x_vals[i]))
		# print(lh)
		# print(lh*integrate(((x**2.0-1.0)*P_prime)/(x-x_vals[i]), (x, x_vals[m], x_vals[m+1])))
		# print(integrate(((x**2.0-1.0)*P_prime)/(x-x_vals[i]), (x, x_vals[m], x_vals[m+1])))
		# q[m,i] = lh*rh
		# print(integrate(((x**2.0-1.0)*P_prime)/(x-x_vals[i]), (x, x_vals[m], x_vals[m+1])))
		# q[m,i] = 0.5/((x_vals[i]**2.0-1.0)*P_2prime.subs(x,x_vals[i]))*integrate(((x**2.0-1.0)*P_prime)/(x-x_vals[i]), (x, x_vals[m], x_vals[m+1]))
		# print(q[m,i])
# quads = get_quadrature_weights(6)
# print(q-quads)
# print(quads)
# print(quads[0][0])
# print(np.sum(quads))
# print(np.sum(q))