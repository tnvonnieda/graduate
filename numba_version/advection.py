import numpy as np
import sys
import math
import matplotlib.pyplot as plt
from get_interpolation_coefficients import get_interpolation_coefficients
from get_derivative_coefficients import get_derivative_coefficients
from get_optimal_weights import get_optimal_weights
from reconstruct import calculate_beta_no_characteristic
from quadrature_weights import get_quadrature_weights, get_quadrature_points
from numba import njit

def shift(array, shift_value, fill_value=np.nan):
	shifted_array = np.empty_like(array)
	if shift_value > 0:
		shifted_array[:shift_value] = fill_value
		shifted_array[shift_value:] = array[:-shift_value]
	elif shift_value < 0:
		shifted_array[shift_value:] = fill_value
		shifted_array[:shift_value] = array[-shift_value:]
	else:
		shifted_array = array
	return shifted_array

class advection:
	def __init__(x_0, x_f, t_f, k, advection_velocity, CFL, r, p, time_int, boundary_type, problem_type, mapped):
		x_0 = x_0 # domain left bound
		x_f = x_f # domain right bound
		x = np.linspace(x_0, x_f, k)
		mapped = mapped
		t = 0
		t_f = t_f # final time
		k = k # number of spatial grid points
		r = r # polynomial degree
		CFL = CFL
		R = 2*r+1
		p = p
		a = get_interpolation_coefficients(r)
		d = get_derivative_coefficients(R)
		b = get_optimal_weights(r).reshape(r+1,1,1)
		h = (x_f-x_0)/(k-1)
		v_a = advection_velocity
		time_int = time_int
		boundary_type = boundary_type
		Pi = 0
		tau = h/abs(v_a)*CFL
		epsilon = 10**(-40)
		if time_int == 'sdc':
			q = get_quadrature_weights(p)
			quad_points = get_quadrature_points(p)
		if problem_type == 'discontinuous':
			# u = np.array([np.piecewise(x, [x < 1/4, (x >= 1/4) & (x <= 3/4), x > 3/4], [0, 1, 0])]).T
			u = np.array([np.piecewise(x, [x < -1/2, (x >= -1/2) & (x <= 1/2), x > 1/2], [0, 1, 0])]).T
		else:
			# u = np.sin(2*np.pi*x).T
			# u = np.array([np.sin(np.pi*x)**2]).T
			u = np.array([np.sin(np.pi*x)], dtype=np.longdouble).T
		u_0 = np.copy(u)
		num_vars = 1

	def weno_left(u_extended):
		u = np.pad(u[0:-1], ((r, r), (0,0)), mode=boundary_type)
		
		P = np.zeros(np.append((r+1), u.shape))
		u_reconstructed = np.zeros(u.shape)
		beta = calculate_beta_no_characteristic(u, r, shift)
	
		alpha = b/(beta+epsilon)**(r+1)
		
		omega = alpha / alpha.sum(axis=0)
	
		# if mapped:
		# 	alpha = omega*(b + b**2 - 3*b*omega + omega**2) / (b**2 + omega*(1 - 2*b))
		# 	omega = alpha / alpha.sum(axis=0)

		for k_s in range(r+1): # for each stencil
			# calculate half point polynomial interpolation, P_{r,k_s,i+1/2}		
			for l in range(r+1):
				P[k_s,:,:] += a[k_s,l]*shift(u, k_s-l).reshape((len(u), num_vars, 1)).reshape((len(u), num_vars))
			u_reconstructed = u_reconstructed + omega[k_s]*P[k_s]

		return u_reconstructed[r:k+r-1]

	def weno_right(): 
		u = np.flip(np.pad(u[1:], ((r, r), (0,0)), mode=boundary_type))
		
		P = np.zeros(np.append((r+1), u.shape))
		u_reconstructed = np.zeros(u.shape)		
		beta = calculate_beta_no_characteristic(u, r, shift)
		# alpha = b/(beta+epsilon)**2
		alpha = b/(beta+epsilon)**(r+1)
		omega = alpha / alpha.sum(axis=0)

		for k_s in range(r+1):
			for l in range(r+1):
				P[k_s,:,:] += a[k_s,l]*shift(u, k_s-l).reshape((len(u), num_vars, 1)).reshape((len(u), num_vars))
			u_reconstructed = u_reconstructed + omega[k_s]*P[k_s]

		return np.flip(u_reconstructed, axis=0)[r:k+r-1]

	def get_dudx():
		u_extended_l = np.pad(u[0:-1], ((r, r), (0,0)), mode=boundary_type)
		u_reconstructed = weno_left() 

		u_upwind = np.pad(u_reconstructed, [(r+1, r+1), (0,0)], mode=boundary_type)
		
		# x_half_extended = np.linspace(x_0 + h / 2 - (r+1)*h, x_f - h/2+(r+1)*h, k - 1 + 2*(r+1))

		dudx = np.zeros(np.shape(u_upwind))
		for i in range(len(d)):
			dudx += d[i]*(shift(u_upwind, -i)-shift(u_upwind, i+1))
		
		return (dudx/h)[r+1:k+r+1]		

	def rk4():
		u = u
		
		k1 = -v_a*get_dudx()
		u = u + tau*1/2*k1

		k2 = -v_a*get_dudx()
		u = u + tau*1/2*k2

		k3 = -v_a*get_dudx()
		u = u + tau*k3

		k4 = -v_a*get_dudx()
		u = u + tau*(1/6*k1+1/3*k2+1/3*k3+1/6*k4)

		t = t + tau

	def sdc():
		w = np.empty(np.append(p, np.append(2*p-1, np.shape(u))))
		dudx = np.empty(np.append(p, np.append(2*p-1, np.shape(u))))

		w[:] = u
		dudx[:] = get_dudx()

		for k in range(1,2*p-1):
			w[1,k] = w[0,k]
			for m in range(2,p):
				w[m,k] = 0 
			
			for j in range(p):
				for m in range (1,p):
					w[m,k] += -tau*v_a*q[m-1,j]*dudx[j,k-1]
				
			for j in range(2,p):
				w[j,k] += w[j-1,k]
				u = w[j-1,k]
				dudx[j-1,k] = get_dudx()
				w[j,k] += tau*(quad_points[j]-quad_points[j-1])/2*(-dudx[j-1,k] + dudx[j-1,k-1])
			
			u = w[p-1,k]
		
			if k < 2*p-2:				
				dudx[p-1,k] = get_dudx()	

	def run():
		# plt.ion()
		# fig = plt.figure(1)
		# ax = fig.add_subplot(111)
		# line1, = ax.plot(x, u,'r-')
		while t < t_f:
			if t + tau > t_f:
				tau = t_f - t
			if time_int == 'rk4':
				rk4()
			elif time_int == 'sdc':
				sdc()
			elif time_int == 'mod_sdc':
				sdc_modified()
			t = t + tau
			# line1.set_ydata(u)
			# fig.canvas.draw()
			# fig.canvas.flush_events()
		max_error = np.amax(np.absolute(u-u_0))
		# plt.plot(x, u, marker='.')
# 		# plt.show()

# # print(np.log((5.6215*10**-7)/(3.5121*10**-7)) / np.log(0.9/0.8))

# # print(np.finfo(np.longdouble).eps)
# # 18
# # sys.exit()
# # sdc3 = advection(0, 1, 5, 101, 1, 0.5, 1, 3, 'sdc', 'wrap', 'smooth')
# # sdc5 = advection(0, 1, 5, 101, 1, 0.5, 2, 3, 'sdc', 'wrap', 'smooth')
# # mod_sdc = advection(0, 1, 5, 101, 1, 0.5, 2, 3, 'mod_sdc', 'wrap', 'smooth')

a1 = advection(-1, 1, 26, 21, 1, 1.0, 6, 4, 'rk4', 'wrap', 'discontinuous', True)
