import numpy as np
import sys
import math
import matplotlib.pyplot as plt
from get_interpolation_coefficients import get_interpolation_coefficients
from get_derivative_coefficients import get_derivative_coefficients
from get_optimal_weights import get_optimal_weights
from reconstruct import calculate_beta_characteristic, calculate_beta_no_characteristic
from quadrature_weights import get_quadrature_weights, get_quadrature_points
from calculate_error import calculate_error
# import time


# shifts an array. Passing a positive shift value implies a negative index, 
# ie. for x_{i-1}, the shift value is 1, and for x_{i+1}, the shift value is -1
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

class euler:
	def __init__(self, x_0, x_f, t_f, k, CFL, r, p, characteristic, boundary_type, time_int, problem_type, shuffle_shock):
		# self.start_time = time.clock()
		self.elapsed_time = 0
		self.x_0 = x_0 # domain left bound
		self.x_f = x_f # domain right bound
		self.x = np.linspace(x_0, x_f, k)
		
		self.t = 0.0
		self.t_f = t_f # final time
		self.k = k # number of spatial grid points
		self.r = r # polynomial degree
		self.CFL = CFL
		self.R = 2*r+1
		self.p = p
		self.a = get_interpolation_coefficients(r)
		self.d = get_derivative_coefficients(self.R)
		self.b = get_optimal_weights(r).reshape(r+1, 1, 1)
		self.h = (x_f-x_0)/(k-1)

		self.x_half = np.linspace(x_0 + self.h/2, self.x_f-self.h/2, self.k-1)

		self.gamma = 1.4
		self.boundary_type = boundary_type
		self.should_shuffle_shock = shuffle_shock
		self.characteristic = characteristic
		self.Pi = 0.0
		self.tau = 0.0
		self.epsilon = 10.0**-40
		self.entropy = None

		if time_int == 'sdc':
			self.q = get_quadrature_weights(p)
			self.quad_points = get_quadrature_points(p)

		# A_rho = 0.2
		# kappa_rho = 5.0
		# x_sw = -4
		# conds = [self.x <= x_sw, self.x > x_sw]

		# self.u_p = np.array([
		# 	np.piecewise(self.x, conds, [lambda x: 27/7, lambda x: 1.0 + A_rho*np.sin(kappa_rho * x)]),
		# 	np.piecewise(self.x, conds, [4*np.sqrt(35)/9, 0.0]),
		# 	np.piecewise(self.x, conds, [31/3, 1.0])
		# 	]).T
		conds = [(self.x >= 0) & (self.x < 0.1), (self.x >= 0.1) & (self.x < 0.9), self.x >= 0.9]
		self.u_p = np.array([
			np.piecewise(self.x, conds, [1.0, 1.0, 1.0]),
			np.piecewise(self.x, conds, [0.0, 0.0, 0.0]),
			np.piecewise(self.x, conds, [1000.0, 0.01, 100.0])
			]).T
		# if shuffle_shock:
		# 	if problem_type == 'moving-shock':
		# 		conds = [self.x <= -9.5, self.x > -9.5]
		# 		self.u_p = np.array([
		# 			np.piecewise(self.x, conds, [27/7, 1.0]),
		# 			np.piecewise(self.x, conds, [4*np.sqrt(35)/9, 0.0]),
		# 			np.piecewise(self.x, conds, [31/3, 1.0])
		# 			]).T
		# 		self.u_p0 = self.u_p
		# 		self.u_c0 = self.get_conservative_vars(self.u_p0)
		# 	elif problem_type == 'shock-entropy-interaction':
		# 		# For shock shuffling
		# 		conds = [self.x <= -9.5, self.x > -9.5]
		# 		self.u_p = np.array([
		# 			np.piecewise(self.x, conds, [27/7, 1.0]),
		# 			np.piecewise(self.x, conds, [4*np.sqrt(35)/9, 0.0]),
		# 			np.piecewise(self.x, conds, [31/3, 1.0])
		# 			]).T

		# 		# Set initial condition as shock-entropy
		# 		conds = [self.x < -9.5, (self.x >= -9.5) & (self.x <= -8.80), self.x > -8.80]			
		# 		self.u_p0 = np.array([
		# 			np.piecewise(self.x, conds, [lambda x: 27/7, lambda x: 1.0, lambda x: np.exp(-0.01*np.sin(13*(x-8.80)))]),
		# 			np.piecewise(self.x, conds, [4*np.sqrt(35)/9, 0.0, 0.0]),
		# 			np.piecewise(self.x, conds, [31/3, 1.0, 1.0])
		# 			]).T
		# 		self.u_c0 = self.get_conservative_vars(self.u_p0)
		# else:
		# 	if problem_type == 'moving-shock':
		# 		conds = [self.x < -9.5, self.x >= -9.5]
		# 		self.u_p = np.array([
		# 			np.piecewise(self.x, conds, [27/7, 1.0]),
		# 			np.piecewise(self.x, conds, [4*np.sqrt(35)/9, 0.0]),
		# 			np.piecewise(self.x, conds, [31/3, 1.0])
		# 			]).T
		# 	elif problem_type == 'shock-entropy-interaction':
		# 		conds = [self.x < -9.5, (self.x >= -9.5) & (self.x <= -8.80), self.x > -8.80]			
		# 		self.u_p = np.array([
		# 			np.piecewise(self.x, conds, [lambda x: 27/7, lambda x: 1.0, lambda x: np.exp(-0.01*np.sin(13*(x-8.80)))]),
		# 			np.piecewise(self.x, conds, [4*np.sqrt(35)/9, 0.0, 0.0]),
		# 			np.piecewise(self.x, conds, [31/3, 1.0, 1.0])
		# 			]).T
		# 	elif problem_type == 'test':
		# 		conds = [self.x < 0, self.x >= 0]
		# 		self.u_p = np.array([
		# 			np.piecewise(self.x, conds, [932, 0.6]), # kg/m^3
		# 			np.piecewise(self.x, conds, [0.0, 0.0]), # m/s
		# 			np.piecewise(self.x, conds, [100000, 100000]), # kPa
		# 			]).T
		self.u_p0 = self.u_p
		self.u_c0 = self.get_conservative_vars(self.u_p0)

		# print(self.u_p)
		# print(self.x)
		# sys.exit()
			# conds = [self.x < -4.0, self.x >= -4.0]			
			# self.u_p = np.array([
			# 	np.piecewise(self.x, conds, [lambda x: 27/7, lambda x: 1.0+1/5*np.sin(5*x)]),
			# 	np.piecewise(self.x, conds, [4*np.sqrt(35)/9, 0.0]),
			# 	np.piecewise(self.x, conds, [31/3, 1.0])
			# 	]).T

		self.u_c = self.get_conservative_vars(self.u_p)
		# self.u_p0 = self.u_p
		# self.u_c0 = self.u_c
		self.num_vars = self.u_p.shape[1]
		self.time_int = time_int
		self.run()
		# self.PLOT_TYPE = REAL_TIME # REAL_TIME/END_TIME

	def get_characteristic_transform_boundary(self, u):
		u = 1/2*(shift(u,-1)+u)
		c = np.sqrt(abs(self.gamma*(u[:,2]+self.Pi)/abs(u[:,0])))

		Q = np.array([
			[np.ones(len(u)), np.ones(len(u)), np.ones(len(u))],
			[-c/u[:,0], np.zeros(len(u)),c/u[:,0]], 
			[c**2, np.zeros(len(u)), c**2]]).transpose((2,0,1))
		Q_inverse = np.array([
			[np.zeros(len(u)), -u[:,0]/(2*c), 1/(2*c**2)], 
			[np.ones(len(u)), np.zeros(len(u)), -1/c**2], 
			[np.zeros(len(u)), u[:,0]/(2*c), 1/(2*c**2)]]).transpose((2,0,1))
		
		return Q, Q_inverse

	def get_maximum_characteristic(self, u):	
		return np.max(np.sqrt(abs(self.gamma*u[:,2]+self.Pi)/abs(u[:,0])) + abs(u[:,1]))

	def get_flux(self, u):
		e=(u[:,2]+self.gamma*self.Pi)/(u[:,0]*(self.gamma-1))
		return np.array([
			u[:,0]*u[:,1], 
			u[:,0]*u[:,1]**2+u[:,2], 
			(u[:,0]*e+1/2*u[:,0]*u[:,1]**2+u[:,2])*u[:,1]]).T

	def get_primitive_vars(self, u_c):
		e = u_c[:,2]/u_c[:,0]-1/2*(u_c[:,1]/u_c[:,0])**2
		return np.array([u_c[:,0], u_c[:,1]/u_c[:,0], (self.gamma-1)*u_c[:,0]*e-self.gamma*self.Pi]).T

	def get_conservative_vars(self, u_p):
		e=(u_p[:,2]+self.gamma*self.Pi)/(u_p[:,0]*(self.gamma-1))
		return np.array([u_p[:,0], u_p[:,0]*u_p[:,1], u_p[:,0]*(e+1/2*u_p[:,1]**2)]).T

	def weno_left(self):
		u = np.pad(self.u_p, ((self.r, self.r-1), (0,0)), mode='reflect', reflect_type='odd')
		u_p_reconstructed = np.zeros(u.shape) 
		P = np.zeros(np.append((self.r+1), u.shape))
		beta = calculate_beta_no_characteristic(u, self.r, shift)

		alpha = self.b/(beta+self.epsilon)**(self.r+1)
		omega = alpha / alpha.sum(axis=0)

		for k_s in range(self.r+1): # for each stencil
			# calculate half point polynomial interpolation, P_{r,k_s,i+1/2}		
			for l in range(self.r+1):
				P[k_s,:,:] += self.a[k_s,l]*shift(u, k_s-l).reshape((len(u), self.num_vars, 1)).reshape((len(u), self.num_vars))	
			u_p_reconstructed = u_p_reconstructed + omega[k_s]*P[k_s]

		return u_p_reconstructed[self.r:-self.r]

	def weno_right(self):
		u = np.flip(np.pad(self.u_p, ((self.r-1, self.r), (0,0)), mode='reflect', reflect_type='odd'), axis=0)
		u_p_reconstructed = np.zeros(u.shape) 
		P = np.zeros(np.append((self.r+1), u.shape))
		
		beta = calculate_beta_no_characteristic(u, self.r, shift)
		alpha = self.b/(beta+self.epsilon)**(self.r+1)
		omega = alpha / alpha.sum(axis=0)

		for k_s in range(self.r+1):
			for l in range(self.r+1):
				P[k_s,:,:] += self.a[k_s,l]*shift(u, k_s-l).reshape((len(u), self.num_vars, 1)).reshape((len(u), self.num_vars))
			u_p_reconstructed = u_p_reconstructed + omega[k_s]*P[k_s]
		return np.flip(u_p_reconstructed, axis=0)[self.r:-self.r]

	def weno_characteristic_left(self):
		u = np.pad(self.u_p, ((self.r, self.r-1), (0,0)), mode='reflect', reflect_type='odd')
		u_p_reconstructed = np.zeros(u.shape) 
		P = np.zeros(np.append((self.r+1), u.shape))
		Q, Q_inverse = self.get_characteristic_transform_boundary(u)

		beta = calculate_beta_characteristic(u, self.r, Q, Q_inverse, shift)

		# print(beta[0]**(self.r+1))
		# sys.exit()
		# print(beta)
		alpha = self.b/(beta+self.epsilon)**(self.r+1)

		# print("alpha: ", alpha)
		omega = alpha / alpha.sum(axis=0)
		# print("omega: ", omega)
		
		for k_s in range(self.r+1): # for each stencil
			# calculate half point polynomial interpolation, P_{r,k_s,i+1/2}		
			for l in range(self.r+1):
				P[k_s,:,:] += self.a[k_s,l]*np.matmul(Q_inverse, shift(u, k_s-l).reshape((len(u), self.num_vars, 1))).reshape((len(u), self.num_vars))	
			u_p_reconstructed = u_p_reconstructed + omega[k_s]*P[k_s]
		# print(np.matmul(u_p_reconstructed.reshape(len(u),self.num_vars,1), Q))
		return (np.matmul(Q, u_p_reconstructed.reshape(len(u),self.num_vars,1))).reshape((len(u),self.num_vars))[self.r:-self.r]

	def weno_characteristic_right(self):
		u = np.flip(np.pad(self.u_p, ((self.r-1, self.r), (0,0)), mode='reflect', reflect_type='odd'), axis=0)
		u_p_reconstructed = np.zeros(u.shape) 
		P = np.zeros(np.append((self.r+1), u.shape))
		Q, Q_inverse = self.get_characteristic_transform_boundary(u)	
		
		beta = calculate_beta_characteristic(u, self.r, Q, Q_inverse, shift)
		
		alpha = self.b/(beta+self.epsilon)**(self.r+1)
		omega = alpha / alpha.sum(axis=0)

		for k_s in range(self.r+1):
			for l in range(self.r+1):
				P[k_s,:,:] += self.a[k_s,l]*np.matmul(Q_inverse, shift(u, k_s-l).reshape((len(u), self.num_vars, 1))).reshape((len(u), self.num_vars))
			u_p_reconstructed = u_p_reconstructed + omega[k_s]*P[k_s]
		# print(u_p_reconstructed)
		# print(Q)
		# sys.exit()
		return np.flip((np.matmul(Q, u_p_reconstructed.reshape(len(u),self.num_vars,1))).reshape((len(u),self.num_vars)), axis=0)[self.r:-self.r]

	def flux_split(self):
		max_characteristic = self.get_maximum_characteristic(self.u_p)	
		
		if self.characteristic:
			u_p_reconstructed_l = self.weno_characteristic_left()
			u_p_reconstructed_r = self.weno_characteristic_right()
		else:
			u_p_reconstructed_l = self.weno_left()
			u_p_reconstructed_r = self.weno_right()		
		
		u_c_reconstructed_l = self.get_conservative_vars(u_p_reconstructed_l)
		u_c_reconstructed_r = self.get_conservative_vars(u_p_reconstructed_r)
		
		# print(u_p_reconstructed_l)
		# sys.exit()
		flux_left = self.get_flux(u_p_reconstructed_l) + max_characteristic*u_c_reconstructed_l
		# print(flux_left)
		# sys.exit()
		flux_right = self.get_flux(u_p_reconstructed_r) - max_characteristic*u_c_reconstructed_r
		
		return 1/2*(flux_left + flux_right)

	def get_dudx(self):
		u_split = np.pad(self.flux_split(), [(self.r+1, self.r+1), (0,0)], mode='reflect', reflect_type='odd')
		# print(u_split)
		# sys.exit()
		# print(u_split)
		# sys.exit()
		dudx = np.zeros(np.shape(u_split))
	  
		for i in range(len(self.d)):
			dudx += self.d[i]*(shift(u_split, -i)-shift(u_split, i+1))
	
		return (dudx/self.h)[self.r+1:-self.r, :]
	
	def calculate_entropy(self):
		e=(self.u_p[:,2]+self.gamma*self.Pi)/(self.u_p[:,0]*(self.gamma-1))
		self.entropy = np.log(e)-(self.gamma-1)*np.log(self.u_p[:,0])

	def rk4(self):
		self.tau = self.CFL * self.h / self.get_maximum_characteristic(self.u_p) 
		u_c = self.u_c
		# print(self.tau)
		k1 = -self.get_dudx()
		# print(k1)  
		self.u_c = u_c + self.tau*1/2*k1
		self.u_p = self.get_primitive_vars(self.u_c)
		# print(self.u_c)
		# sys.exit()
		k2 = -self.get_dudx()
		self.u_c = u_c + self.tau*1/2*k2
		self.u_p = self.get_primitive_vars(self.u_c)
		
		k3 = -self.get_dudx()
		self.u_c = u_c + self.tau*k3
		self.u_p = self.get_primitive_vars(self.u_c)
		
		k4 = -self.get_dudx()
		self.u_c = u_c + self.tau*(1/6*k1+1/3*k2+1/3*k3+1/6*k4)
		self.u_p = self.get_primitive_vars(self.u_c)
		
		# plt.plot(self.x, self.u_p[:,0])
		# plt.show()
		# sys.exit()
		# self.t = self.t + self.tau

	def sdc(self):
		self.tau = self.CFL * self.h / self.get_maximum_characteristic(self.u_p)
		w = np.empty(np.append(self.p, np.append(2*self.p-1, np.shape(self.u_c))))
		dudx = np.empty(np.append(self.p, np.append(2*self.p-1, np.shape(self.u_c))))

		w[:] = self.u_c
		dudx[:] = self.get_dudx()

		for k in range(1,2*self.p-1):
			w[1,k] = w[0,k]
			for m in range(2,self.p):
				w[m,k] = 0 
			
			for j in range(self.p):
				for m in range (1,self.p):
					w[m,k] += -self.tau*self.q[m-1,j]*dudx[j,k-1]
				
			for j in range(2,self.p):
				w[j,k] += w[j-1,k]
				self.u_c = w[j-1,k]
				self.u_p = self.get_primitive_vars(self.u_c)
				dudx[j-1,k] = self.get_dudx()
				w[j,k] += self.tau*(self.quad_points[j]-self.quad_points[j-1])/2*(-dudx[j-1,k] + dudx[j-1,k-1])
			self.u_c = w[self.p-1,k]
			self.u_p = self.get_primitive_vars(self.u_c)
			if k < 2*self.p-2:				
				dudx[self.p-1,k] = self.get_dudx()	
		# self.t = self.t + self.tau

	def shuffle_shock(self):
		shuffle_time = 1.0
		t = 0
		original_shock_location = int(0.5/self.h)
		# print(original_shock_location)
		# plt.plot(self.x, self.u_p[:,0], marker='.')
		# plt.show()
		# sys.exit()
		while t < shuffle_time:
			if self.time_int == 'rk4':
				self.rk4()
			elif self.time_int == 'sdc': 
				self.sdc()
			t = t + self.tau
		
		# new_shock_location = original_shock_location + int(self.k * 0.19 * shuffle_time)
		new_shock_location = (np.abs(self.u_p[:,0] - 1.5)).argmin()
		# print(idx)
		# sys.exit()
		shock_pad = 12
		shock_left = new_shock_location - shock_pad
		shock_right = new_shock_location + shock_pad
		shock_profile = self.u_p[shock_left:shock_right]
		# print(shock_profile)
		x_shuffle = self.x[shock_left:shock_right]
		# plt.plot(self.x[new_shock_location], self.u_p[new_shock_location,0], marker='o')
		# plt.plot(x_shuffle, shock_profile[:,0], marker='.')
		# plt.show()
		# sys.exit()
		# num_shuffle_points = int(self.k * 0.17)

		# print(np.shape(self.u_p))
		# self.u_p = np.pad(self.u_p[num_shuffle_points:], ((0, num_shuffle_points), (0,0)), mode='reflect', reflect_type='odd')
		# self.u_c = self.get_conservative_vars(self.u_p)
		# print(np.shape(self.u_p))
		# print(self.u_p)

		# sys.exit()
		# plt.plot(self.x, self.u_p0[:,0], marker='.')
		# print(shock_profile)
		# print(np.shape(shock_profile))
		# print(original_shock_location - shock_pad + 1, original_shock_location + shock_pad + 1)
		self.u_p0[original_shock_location - shock_pad + 1 : original_shock_location + shock_pad + 1] = shock_profile
		self.u_p = self.u_p0
		# plt.plot(self.x, self.u_p[:,0], marker='o')
		# plt.show()
		# sys.exit()
		self.u_c = self.get_conservative_vars(self.u_p)
	def run(self):
		if self.should_shuffle_shock:
			self.shuffle_shock()
			# self.shuffle_shock()
		# plt.plot(self.x, self.u_p[:,0])
		
		# plt.show()
		# sys.exit()
		plt.ion()
		fig = plt.figure(1)
		ax = fig.add_subplot(111)
		line1, = ax.plot(self.x,self.u_p[:,0],'r-')
		# line2, = ax.plot(self.x,self.u_p[:,1],'b-')
		# line3, = ax.plot(self.x,self.u_p[:,2],'g-')
		while self.t < self.t_f:
			if self.time_int == 'rk4':
				self.rk4()
				# plt.plot(self.x, self.u_p)
				# plt.show()
				# sys.exit()
				# print(self.t)
			elif self.time_int == 'sdc': 
				self.sdc()
			self.t = self.t + self.tau
			# elif self.time_int == 'sdc6':
			# 	self.sdc6()
			line1.set_ydata(self.u_p[:,0])
			# line2.set_ydata(self.u_p[:,1])
			# line3.set_ydata(self.u_p[:,2])
			fig.canvas.draw()
			fig.canvas.flush_events()
		self.calculate_entropy()
		# self.elapsed_time = time.clock() - self.start_time
		print('done, time: ', self.elapsed_time)		

x0 = 0
xf = 1
tf = 0.038
N = 401
# CFL = 0.5
problem_type = 'shock-entropy'
characteristic = False

rk4 = euler(x0, xf, tf, N, 0.5, 2, 3, characteristic, 'reflect', 'rk4', problem_type, False)
plt.plot(rk4.x, rk4.u_p[:,0], marker='.')

# sdc4 = euler(x0, xf, tf, N, 0.5, 5, 3, characteristic, 'edge', 'sdc', problem_type)
# plt.plot(sdc4.x, sdc4.u_p[:,0], marker='.')

# sdc6 = euler(x0, xf, tf, N, 0.5, 5, 4, characteristic, 'edge', 'sdc', problem_type)
# plt.plot(sdc6.x, sdc6.u_p[:,0], marker='.')

# sdc8 = euler(x0, xf, tf, N, 0.5, 5, 5, characteristic, 'edge', 'sdc', problem_type)
# plt.plot(sdc8.x, sdc8.u_p[:,0], marker='.')

# sdc10 = euler(x0, xf, tf, N, 0.5, 5, 6, characteristic, 'edge', 'sdc', problem_type)
# plt.plot(sdc10.x, sdc10.u_p[:,0], marker='.')

# sdc8 = euler(x0, xf, tf, N, 0.1, 3, 4, characteristic, 'edge', 'rk4', problem_type, False)
# no_shuffle = euler(x0, xf, tf, N, 0.4, 3, 7, characteristic, 'edge', 'rk4', problem_type, False)
# print(sdc12.u_p[:,0])
# np.savetxt('shock_entropy_exact_t5_500.out', (sdc8.u_p[:,0], sdc8.u_p[:,1], sdc8.u_p[:,2]), delimiter=', ')

# plt.plot(sdc8.x, sdc8.u_p[:,0], marker='.')
# plt.plot(sdc12.x, sdc12.u_p[:,1], marker='.')
# plt.plot(sdc12.x, sdc12.u_p[:,2], marker='.')
# plt.plot(no_shuffle.x, no_shuffle.u_p[:,0], marker='')
# plt.plot(no_shuffle.x, no_shuffle.u_p[:,1], marker='.')
# plt.plot(no_shuffle.x, no_shuffle.u_p[:,1], marker='.')
plt.show()
# plt.legend(["rk4", "sdc4", "sdc6", "sdc8", "sdc10", "sdc12"], loc="lower left")
# plt.title("Density Profile at t=5.0, Moving Shock, N=500")

# fig, axs = plt.subplots(3,2)
# axs[0,0].plot(rk4.x, rk4.u_p[:,0], marker='.')
# axs[0,0].set_title("rk4, CFL=0.2")

# axs[0,1].plot(sdc4.x, sdc4.u_p[:,0], marker='.')
# axs[0,1].set_title("SDC4, CFL=0.5")

# axs[1,0].plot(sdc6.x, sdc6.u_p[:,0], marker='.')
# axs[1,0].set_title("SDC6, CFL=0.5")

# axs[1,1].plot(sdc8.x, sdc8.u_p[:,0], marker='.')
# axs[1,1].set_title("SDC8, CFL=0.5")

# axs[2,0].plot(sdc10.x, sdc10.u_p[:,0], marker='.')
# axs[2,0].set_title("SDC10, CFL=0.5")

# axs[2,1].plot(sdc12.x, sdc12.u_p[:,0], marker='.')
# axs[2,1].set_title("SDC12, CFL=0.5")

# for ax in axs.flat:
# 	ax.set(xlabel="x", ylabel="u")
# 	ax.label_outer()
# plt.setp(axs, xlim=(0,8.2), ylim=(3.84, 3.88))


# plt.show()

# plt.plot(sdc4.x, sdc4.u_p[:,0], marker='.')
# plt.plot(sdc6.x, sdc6.u_p[:,0], marker='.')
# plt.plot(sdc8.x, sdc8.u_p[:,0], marker='.')
# plt.plot(sdc10.x, sdc10.u_p[:,0], marker='.')
# plt.plot(sdc10.x, sdc10.u_p[:,0], marker='.')
# plt.plot(sdc10.x, sdc10.u_p[:,2], marker='.')
# plt.plot(sdc10.x, sdc10.entropy, marker='.')
# plt.plot(rk4.x, rk4.u_p[:,0], marker='.')
# plt.plot(rk4.x, rk4.u_p[:,2], marker='.')
# plt.plot(rk4.x, rk4.entropy, marker='.')

# L1, L2, Linf = calculate_error(rk4.u_p[:,0].T[0:400], exact.u_p[:,0].T[0:400])
# print(L1, L2, Linf)
# L1, L2, Linf = calculate_error(sdc12.u_p[:,0].T[0:400], exact.u_p[:,0].T[0:400])
# print(L1, L2, Linf)


# plt.title("Moving Shock Comparison, RK4 vs SDC10, 11th Order WENO, N=500, CFL=0.5, t=5.0s")
# plt.legend(["rho, sdc12", "p, sdc12", "e, sdc12", "rho, rk4", "p, rk4", "e, rk4"], loc="lower left")
# plt.legend(["sdc8", "0.2"], loc="lower left")
# plt.show()