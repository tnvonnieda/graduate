import numpy as np
import sys
import math
import matplotlib.pyplot as plt
from get_interpolation_coefficients import get_interpolation_coefficients
from get_derivative_coefficients import get_derivative_coefficients
from get_optimal_weights import get_optimal_weights
from reconstruct import calculate_beta_characteristic
from quadrature_weights import get_quadrature_weights, get_quadrature_points
from numba import njit
# from numba.experimental import jitclass
import time

'''
Shifts an array. Passing a positive shift value implies a negative index, 
ie. for x_{i-1}, the shift value is 1, and for x_{i+1}, the shift value is -1
'''
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

def euler(x_0, x_f, t_f, k, CFL, r, p, characteristic, time_int):
	# start_time = time.clock()
	elapsed_time = 0
	x = np.linspace(x_0, x_f, k)
	
	t = 0.0
	tau = 0.0
	R = 2*r+1
	
	a = get_interpolation_coefficients(r)
	d = get_derivative_coefficients(R)
	b = get_optimal_weights(r).reshape(r+1, 1)
	h = (x_f-x_0)/(k-1)

	x_half = np.linspace(x_0 + h/2, x_f-h/2, k-1)

	gamma = 1.4
	Pi = 0.0
	epsilon = 10.0**-40
	tiny = np.finfo(1.0).tiny

	if time_int == 'sdc':
		q = get_quadrature_weights(p)
		quad_points = get_quadrature_points(p)

	conds = [(x >= 0) & (x < 0.1), (x >= 0.1) & (x < 0.9), x >= 0.9]
	u_p_global = np.array([
		np.piecewise(x, conds, [1.0, 1.0, 1.0]),
		np.piecewise(x, conds, [0.0, 0.0, 0.0]),
		np.piecewise(x, conds, [1000.0, 0.01, 100.0])
		]).T

	num_vars = u_p_global.shape[1]
	def get_characteristic_transform(u):
		u = 1/2*(shift(u,-1)+u)
		c = np.sqrt(abs(gamma*(u[:,2]+Pi)/abs(u[:,0])))

		Q = np.array([
			[np.ones(len(u)), np.ones(len(u)), np.ones(len(u))],
			[-c/u[:,0], np.zeros(len(u)),c/u[:,0]], 
			[c**2, np.zeros(len(u)), c**2]]).transpose((2,0,1))
		Q_inverse = np.array([
			[np.zeros(len(u)), -u[:,0]/(2*c), 1/(2*c**2)], 
			[np.ones(len(u)), np.zeros(len(u)), -1/c**2], 
			[np.zeros(len(u)), u[:,0]/(2*c), 1/(2*c**2)]]).transpose((2,0,1))
		
		return Q[:-1], Q_inverse[:-1]


	def get_maximum_characteristic(u):	
		return np.max(np.sqrt(abs(gamma*u[:,2]+Pi)/abs(u[:,0])) + abs(u[:,1]))

	def get_flux(u):
		e=(u[:,2]+gamma*Pi)/(u[:,0]*(gamma-1))
		return np.array([
			u[:,0]*u[:,1], 
			u[:,0]*u[:,1]**2+u[:,2], 
			(u[:,0]*e+1/2*u[:,0]*u[:,1]**2+u[:,2])*u[:,1]]).T

	def get_primitive_vars(u_c):
		e = u_c[:,2]/u_c[:,0]-1/2*(u_c[:,1]/u_c[:,0])**2
		return np.array([u_c[:,0], u_c[:,1]/u_c[:,0], (gamma-1)*u_c[:,0]*e-gamma*Pi]).T

	def get_conservative_vars(u_p):
		e=(u_p[:,2]+gamma*Pi)/(u_p[:,0]*(gamma-1))
		return np.array([u_p[:,0], u_p[:,0]*u_p[:,1], u_p[:,0]*(e+1/2*u_p[:,1]**2)]).T

	@njit
	def build_weno_reconstructions(u_p_extended_l, u_p_extended_r, Q_l, Q_inverse_l, Q_r, Q_inverse_r):				
		u_p_reconstructed_l = np.zeros((k-1,3)) 
		u_p_reconstructed_r = np.zeros((k-1,3))

		P_l = np.zeros((r+1, k-1, 3))
		P_r = np.zeros((r+1, k-1, 3))

		for i in range(k-1):

			''' VERIFIED AGAINST OLD CODE '''
			# beta_l = np.zeros((r+1,3))
			# beta_l[0] = (np.dot(Q_inverse_l[i],u_p_extended_l[i+r+1])-np.dot(Q_inverse_l[i],u_p_extended_l[i+r]))**2
			# beta_l[1] = (np.dot(Q_inverse_l[i],u_p_extended_l[i+r])-np.dot(Q_inverse_l[i],u_p_extended_l[i+r-1]))**2

			beta_l = calculate_beta_characteristic(u_p_extended_l[i:i+2*r+1], r, Q_inverse_l[i])
			beta_r = calculate_beta_characteristic(u_p_extended_r[i:i+2*r+1], r, Q_inverse_r[i])
		
			# beta_r = np.zeros((r+1,3))			
			# beta_r[0] = (np.dot(Q_inverse_r[i],u_p_extended_r[i+r+1])-np.dot(Q_inverse_r[i],u_p_extended_r[i+r]))**2
			# beta_r[1] = (np.dot(Q_inverse_r[i],u_p_extended_r[i+r])-np.dot(Q_inverse_r[i],u_p_extended_r[i+r-1]))**2
			
			alpha_l =  b / (tiny + beta_l**(r+1))
			omega_l = alpha_l / alpha_l.sum(axis=0)
			alpha_l = omega_l*(b + b**2 - 3*b*omega_l + omega_l**2) / (b**2 + omega_l*(1 - 2*b))
			omega_l = alpha_l / alpha_l.sum(axis=0)

			alpha_r = b / (tiny + beta_r**(r+1))
			omega_r = alpha_r / alpha_r.sum(axis=0)
			alpha_r = omega_r*(b + b**2 - 3*b*omega_r + omega_r**2) / (b**2 + omega_r*(1 - 2*b))
			omega_r = alpha_r / alpha_r.sum(axis=0)

			for k_s in range(r+1):
				for l in range(r+1):					
					P_l[k_s,i,:] = P_l[k_s,i,:] + a[k_s,l]*u_p_extended_l[i+r-k_s+l,:]
					P_r[k_s,i,:] = P_r[k_s,i,:] + a[k_s,l]*u_p_extended_r[i+r-k_s+l,:]

				P_l[k_s,i,:] = np.dot(Q_inverse_l[i],P_l[k_s,i,:])
				P_r[k_s,i,:] = np.dot(Q_inverse_r[i],P_r[k_s,i,:])

				u_p_reconstructed_l[i] = u_p_reconstructed_l[i] + omega_l[k_s]*P_l[k_s,i]
				u_p_reconstructed_r[i] = u_p_reconstructed_r[i] + omega_r[k_s]*P_r[k_s,i]
			
			u_p_reconstructed_l[i] = np.dot(Q_l[i], u_p_reconstructed_l[i])
			u_p_reconstructed_r[i] = np.dot(Q_r[i], u_p_reconstructed_r[i]) 	

		return u_p_reconstructed_l, u_p_reconstructed_r

	def get_dudx(u_p):	
		max_characteristic = get_maximum_characteristic(u_p)
		'''
		Inititalize padded array for WENO reconstructions
		np.ascontiguousarray() is necessary for improved numba efficiency
		'''
		u_p_extended = np.pad(u_p, ((r, r), (0,0)), mode='reflect', reflect_type='odd')
		u_p_extended_l = np.ascontiguousarray(u_p_extended[0:-1])
		u_p_extended_r = np.ascontiguousarray(np.flip(u_p_extended[1:], axis=0))
		''' 
		Need a transformation matrix for each boundary value in extended array.
		The tranformation matrices are computed as cell-centered values.
		Thus, we need one extra right cell on the original domain. We can reference this
		extension in the way shown. The caluclation of Q and Q_inverse is performed
		outisde of the build_reconstructed_arrays() function because it is independent
		of r and of the direction of the reconstruction. 

		Note that the left and right Q matrices are the same, just flipped for the
		respective reconstructions.

		np.flip() is not supported by numba, so we must perform all of the array flipping
		outside of the numba box
		'''

		Q, Q_inverse = get_characteristic_transform(u_p_extended[r:k+r+1])

		Q_l = np.ascontiguousarray(Q[0:-1])
		Q_inverse_l = np.ascontiguousarray(Q_inverse[0:-1])
		Q_r = np.ascontiguousarray(np.flip(Q[0:-1], axis=0))
		Q_inverse_r = np.ascontiguousarray(np.flip(Q_inverse[0:-1], axis=0))
		'''
		Build the right and left reconstructions using the extended array.
		The overall array has r additional points to the left of the original domain,
		and r additional points to the right of the original domain. Additionally,
		each grid point within the domain has a pair of associated Q and Q_inverse matrices. 
		These are sent into the reconstruction to correctly perform the characteristic reconstruction.
		'''
		u_p_reconstructed_l, u_p_reconstructed_r = build_weno_reconstructions(u_p_extended_l, u_p_extended_r, Q_l, Q_inverse_l, Q_r, Q_inverse_r)
		u_p_reconstructed_r = np.flip(u_p_reconstructed_r, axis=0) 
	
		u_c_reconstructed_l = get_conservative_vars(u_p_reconstructed_l)
		u_c_reconstructed_r = get_conservative_vars(u_p_reconstructed_r)

		flux_left = get_flux(u_p_reconstructed_l) + max_characteristic*u_c_reconstructed_l
		flux_right = get_flux(u_p_reconstructed_r) - max_characteristic*u_c_reconstructed_r
		# plt.plot(x_half, flux_right)
		# plt.show()
		# sys.exit()
		u_split = np.pad(1/2*(flux_left + flux_right), [(r+1, r+1), (0,0)], mode='reflect', reflect_type='odd')

		dudx = np.zeros(np.shape(u_split))
	
		for i in range(len(d)):
			dudx += d[i]*(shift(u_split, -i)-shift(u_split, i+1))
	
		return (dudx/h)[r+1:-r, :]

	def rk4(u_p, u_c):
		tau = CFL * h / get_maximum_characteristic(u_p) 
		u_c_0 = np.copy(u_c)
		
		k1 = -get_dudx(u_p)
		
		u_c = u_c_0 + tau*1/2*k1		
		u_p = get_primitive_vars(u_c)	

		k2 = -get_dudx(u_p)
		u_c = u_c_0 + tau*1/2*k2
		u_p = get_primitive_vars(u_c)
		
		k3 = -get_dudx(u_p)
		u_c = u_c_0 + tau*k3
		u_p = get_primitive_vars(u_c)
		
		k4 = -get_dudx(u_p)
		u_c = u_c_0 + tau*(1/6*k1+1/3*k2+1/3*k3+1/6*k4)
		u_p = get_primitive_vars(u_c)

		return u_p, u_c, tau

	def ssprk3(u_p, u_c):
		tau = CFL * h / get_maximum_characteristic(u_p) 
		u_c_0 = np.copy(u_c)
		
		k1 = -get_dudx(u_p)
		# plt.plot(x, k1)
		# plt.show()
		# sys.exit()
		u_c = u_c_0 + tau*k1
		u_p = get_primitive_vars(u_c)		
		
		k2 = -get_dudx(u_p)
		u_c = 3/4*u_c_0 + 1/4*(u_c_0 + tau*k1) + 1/4*tau*k2
		u_p = get_primitive_vars(u_c)
		
		k3 = -get_dudx(u_p)
		u_c = u_c_0 + tau*(1/6*k1 + 1/6*k2 + 2/3*k3)
		u_p = get_primitive_vars(u_c)

		return u_p, u_c, tau

	def sdc(u_p, u_c):		
		tau = CFL * h / get_maximum_characteristic(u_p)
		w = np.empty(np.append(p, np.append(2*p-1, np.shape(u_c))))
		dudx = np.empty(np.append(p, np.append(2*p-1, np.shape(u_c))))

		w[:] = u_c
		dudx[:] = get_dudx()

		for k in range(1,2*p-1):
			w[1,k] = w[0,k]
			for m in range(2,p):
				w[m,k] = 0 
			
			for j in range(p):
				for m in range (1,p):
					w[m,k] += -tau*q[m-1,j]*dudx[j,k-1]
				
			for j in range(2,p):
				w[j,k] += w[j-1,k]
				u_c = w[j-1,k]
				u_p = get_primitive_vars(u_c)
				dudx[j-1,k] = get_dudx()
				w[j,k] += tau*(quad_points[j]-quad_points[j-1])/2*(-dudx[j-1,k] + dudx[j-1,k-1])
			u_c = w[p-1,k]
			u_p = get_primitive_vars(u_c)
			if k < 2*p-2:				
				dudx[p-1,k] = get_dudx()
	
	# plt.ion()
	# fig = plt.figure(1)
	# ax = fig.add_subplot(111)
	# line1, = ax.plot(x,u_p[:,0],'r-')
	# line2, = ax.plot(x,u_p[:,1],'b-')
	# line3, = ax.plot(x,u_p[:,2],'g-')
	u_c_global = get_conservative_vars(u_p_global)
	while t < t_f:
	# for i in range(2):
		if time_int == 'rk4':
			u_p_global, u_c_global, tau = rk4(u_p_global, u_c_global)
		elif time_int == 'ssprk3':
			u_p_global, u_c_global, tau = ssprk3(u_p_global, u_c_global)
		elif time_int == 'sdc': 
			u_p_global, u_c_global, tau = sdc(u_p, u_c)
		t = t + tau
		# print(tau)
		# sys.exit()
		
		# line1.set_ydata(u_p[:,0])
		# line2.set_ydata(u_p[:,1])
		# line3.set_ydata(u_p[:,2])
		# fig.canvas.draw()
		# fig.canvas.flush_events()
	# plt.show()
	# sys.exit()
	# elapsed_time = time.clock() - start_time
	# print('done, time: ', elapsed_time)	
	return x, u_p_global

x0 = 0
xf = 1.0
tf = 1.3
N = 301
# CFL = 0.5
characteristic = True

x, u_p = euler(x0, xf, tf, N, 0.4, 5, 4, characteristic, 'rk4')
plt.plot(x, u_p[:,0], marker='.')

plt.show()