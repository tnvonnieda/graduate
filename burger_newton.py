import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

def f(u, x, t):
    return u - 1/2 - np.sin(np.pi*(x-u*t))

def fprime(u, x, t):
    return 1+t*np.pi*np.cos(np.pi*(x-u*t))

# while abs(delta) >= epsilon and iteration < max_iterations:
	# 	delta = (g_1(P,T) - g_2(P,T)) / (g_1_prime(P,T) - g_2_prime(P,T))
	# 	T = T - delta
	# 	iteration = iteration + 1
	# return T
def get_exact(x, t, guess):
    u_old = 0
    u_new = guess

    while np.max(abs(u_new-u_old)) > 10**(-15):
        u_old = u_new
        u_new = u_old - f(u_old,x,t)/fprime(u_old,x,t)

    return u_new

# print(get_exact())

# def f(u, x, t):
# 	return 1/2 + np.sin(np.pi * (x - u*t)) - u

# def f_prime(u, x, t)
# 	return 
x = np.linspace(0, 2, 5000)
u_guess = np.piecewise(x, [x < 1+1/np.pi , x >= 1+1/np.pi], [lambda x: 0.158 + x/(1+1/np.pi), lambda x: -1.3 + x/(1+1/np.pi)])
u_exact = np.empty(np.shape(x))
t = 0.7
for i in range(len(x)):
	u_exact[i] = optimize.newton(f, u_guess[i], args=(x[i], t))