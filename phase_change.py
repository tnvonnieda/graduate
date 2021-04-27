import numpy as np
from scipy.optimize import root
import sys
import math
import matplotlib.pyplot as plt
from get_interpolation_coefficients import get_interpolation_coefficients
from get_derivative_coefficients import get_derivative_coefficients
from get_optimal_weights import get_optimal_weights
from reconstruct import calculate_beta_characteristic, calculate_beta_no_characteristic
from quadrature_weights import get_quadrature_weights, get_quadrature_points
from calculate_error import calculate_error
import time

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
	def __init__(self, x_0, x_f, t_f, k, CFL, r, p, characteristic, boundary_type, time_int):
		self.start_time = time.clock()
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

		# self.x_half = np.linspace(x_0 + self.h/2, self.x_f-self.h/2, self.k-1)
		
		self.boundary_type = boundary_type
		# self.should_shuffle_shock = shuffle_shock
		self.characteristic = characteristic

		# Material properties
		# liquid water
		self.gamma_1 = 2.62 
		self.Pi_1 = 9058.29*10**5 # Pa 
		self.c_v_1 = 1606.97 # J/(kg K)
		self.Lambda_1 = 0.6788 # w/(m K)
		self.q_1 = -1.150975*10**6 # J
		self.q_1_prime = None

		# water vapor
		self.gamma_2 = 1.38 
		self.Pi_2 = 0 # Pa  
		self.c_v_2 = 1192.51 # J/(kg K)
		self.Lambda_2 = 249.97 # w/(m K)
		self.q_2 = 2.060759*10**6 # J
		self.q_2_prime = 0

		self.tau = 0.0
		self.epsilon = 10.0**-40
		self.entropy = None

		if time_int == 'sdc':
			self.q = get_quadrature_weights(p)
			self.quad_points = get_quadrature_points(p)

		# SET INITIAL CONDITION

		# Liquid water - water vapor shock tube
		conds = [self.x < 0, self.x >= 0]
		self.u_p = np.array([
			np.piecewise(self.x, conds, [1180, 1.11]), # kg/m^3
			np.piecewise(self.x, conds, [0.0, 0.0]), # m/s
			np.piecewise(self.x, conds, [1000, 1.0]), # bar
			np.piecewise(self.x, conds, [1.0, 0.0]) # left side liquid, right side vapor
			]).T
		
		self.T = None # initialize T, should update
		# T_in = T_sat = 372.79 K, T_out = 1000 K ?
		# P(x) = P_0 = 1 Bar (constant across x)
		self.u_c = self.get_conservative_vars(self.u_p)

		self.u_p0 = self.u_p
		self.u_c0 = self.u_c
		self.num_vars = self.u_p.shape[1]
		self.time_int = time_int
		self.run()
		# self.PLOT_TYPE = REAL_TIME # REAL_TIME/END_TIME

	def get_characteristic_transform(self, u):
		u = 1/2*(shift(u,-1)+u)
		c = self.speed_of_sound(u)

		Q = np.array([
			[np.ones(len(u)), np.ones(len(u)), np.zeros(len(u)), np.ones(len(u))],
			[-c/u[:,0], np.zeros(len(u)), np.zeros(len(u)), c/u[:,0]], 
			[c**2, np.zeros(len(u)), np.zeros(len(u)), c**2],
			[np.zeros(len(u)), np.zeros(len(u)), np.ones(len(u)), np.zeros(len(u))]]).transpose((2,0,1))
		
		Q_inverse = np.array([
			[np.zeros(len(u)), -u[:,0]/(2*c), 1/(2*c**2), np.zeros(len(u))], 
			[np.ones(len(u)), np.zeros(len(u)), -1/c**2, np.zeros(len(u))],
			[np.zeros(len(u)), np.zeros(len(u)), np.zeros(len(u)), np.ones(len(u))],
			[np.zeros(len(u)), u[:,0]/(2*c), 1/(2*c**2), np.zeros(len(u))]]).transpose((2,0,1))
		
		return Q, Q_inverse

	def e_from_primitive(self, u_p):	
		return u_p[:,3]*self.e_1()
		# return (u_p[:,3]*self.c_v_1 + (1-u_p[:,3])*self.c_v_2)*self.get_temperature_from_primitive(u_p)
		# return (u_p[:,2]+self.gamma_1*self.Pi_1)/((self.gamma_1 - 1)*u_p[:,0])

	def e_from_conservative(self, u_c):
		return u_c[:,2]/u_c[:,0]-1/2*(u_c[:,1]/u_c[:,0])**2

	# Maybe some errors
	def speed_of_sound(self, u_p):
		T = self.get_temperature_from_primitive(u_p)

		rho_1 = (u_p[:,2] + self.gamma_1*self.Pi_1)/((self.gamma_1 - 1)*self.c_v_1*T)
		rho_2 = (u_p[:,2] + self.gamma_2*self.Pi_2)/((self.gamma_2 - 1)*self.c_v_2*T)
	
		c_1 = np.sqrt(abs(self.gamma_1*(u_p[:,2]+self.Pi_1)/rho_1))
		c_2 = np.sqrt(abs(self.gamma_2*(u_p[:,2]+self.Pi_2)/rho_2))

		alpha_1 = u_p[:,0]*u_p[:,3]/rho_1
		alpha_2 = 1 - alpha_1
		# print(volume_fraction_1)

		c = np.sqrt(1/u_p[:,0]*rho_1*c_1**2*rho_2*c_2**2/(alpha_1*rho_2*c_2**2 + alpha_2*rho_1*c_1**2))

		return c

	def get_maximum_characteristic(self, u_p):	
		return np.max(self.speed_of_sound(u_p) + abs(u_p[:,1]))

	def get_flux(self, u):
		return np.array([
			u[:,0]*u[:,1], 
			u[:,0]*u[:,1]**2+u[:,2], 
			(u[:,0]*self.e_from_primitive(u)+1/2*u[:,0]*u[:,1]**2+u[:,2])*u[:,1],
			u[:,0]*u[:,1]*u[:,3]]).T

	def get_primitive_vars(self, u_c):
		T = self.get_temperature_from_conservative(u_c)
		B = -u_c[:,0]*(self.c_v_1*T*u_c[:,3]/u_c[:,0]*(self.gamma_1-1) + self.c_v_2*T*(1-u_c[:,3]/u_c[:,0])*(self.gamma_2-1)) + self.gamma_1*self.Pi_1 + self.gamma_2*self.Pi_2
		C = -u_c[:,0]*(self.gamma_2*self.Pi_2*u_c[:,3]/u_c[:,0]*self.c_v_1*T*(self.gamma_1 - 1) + self.gamma_1*self.Pi_1*(1-u_c[:,3]/u_c[:,0])*self.c_v_2*T*(self.gamma_2 - 1)) + self.gamma_1*self.Pi_1*self.gamma_2*self.Pi_2
		return np.array([
			u_c[:,0], 
			u_c[:,1]/u_c[:,0],
			1/2*(-B + np.sqrt(B**2-4*C)),
			u_c[:,3]/self.u_c[:,0]]).T

	def get_conservative_vars(self, u_p):
		return np.array([
			u_p[:,0], 
			u_p[:,0]*u_p[:,1],
			u_p[:,0]*(self.e_from_primitive(u_p)+1/2*u_p[:,1]**2),
			u_p[:,0]*u_p[:,3]]).T

	def weno_left(self, f):
		u = np.pad(f, ((self.r, self.r-1), (0,0)), mode=self.boundary_type)
		u_p_reconstructed = np.zeros(u.shape) 
		P = np.zeros(np.append((self.r+1), u.shape))
		beta = calculate_beta_no_characteristic(u, self.r, shift)

		alpha = self.b/(beta+self.epsilon)**(self.r+1)
		omega = alpha / alpha.sum(axis=0)

		for k_s in range(self.r+1): # for each stencil
			# calculate half point polynomial interpolation, P_{r,k_s,i+1/2}		
			for l in range(self.r+1):
				P[k_s,:,:] += self.a[k_s,l]*shift(u, k_s-l).reshape((len(u), np.shape(u)[1], 1)).reshape((len(u), np.shape(u)[1]))	# changed 2nd argument of reshape
			u_p_reconstructed = u_p_reconstructed + omega[k_s]*P[k_s]

		return u_p_reconstructed[self.r:-self.r]

	def weno_right(self, f):
		u = np.flip(np.pad(f, ((self.r-1, self.r), (0,0)), mode=self.boundary_type), axis=0)
		u_p_reconstructed = np.zeros(u.shape) 
		P = np.zeros(np.append((self.r+1), u.shape))
		
		beta = calculate_beta_no_characteristic(u, self.r, shift)
		alpha = self.b/(beta+self.epsilon)**(self.r+1)
		omega = alpha / alpha.sum(axis=0)

		for k_s in range(self.r+1):
			for l in range(self.r+1):
				P[k_s,:,:] += self.a[k_s,l]*shift(u, k_s-l).reshape((len(u), np.shape(u)[1], 1)).reshape((len(u), np.shape(u)[1]))
			u_p_reconstructed = u_p_reconstructed + omega[k_s]*P[k_s]
		return np.flip(u_p_reconstructed, axis=0)[self.r:-self.r]

	def weno_characteristic_left(self, f):
		u = np.pad(f, ((self.r, self.r-1), (0,0)), mode=self.boundary_type)
		u_p_reconstructed = np.zeros(u.shape) 
		P = np.zeros(np.append((self.r+1), u.shape))
		Q, Q_inverse = self.get_characteristic_transform(u)

		beta = calculate_beta_characteristic(u, self.r, Q, Q_inverse, shift)
		alpha = self.b/(beta+self.epsilon)**(self.r+1)
		omega = alpha / alpha.sum(axis=0)

		for k_s in range(self.r+1): # for each stencil
			# calculate half point polynomial interpolation, P_{r,k_s,i+1/2}		
			for l in range(self.r+1):
				P[k_s,:,:] += self.a[k_s,l]*np.matmul(Q_inverse, shift(u, k_s-l).reshape((len(u), self.num_vars, 1))).reshape((len(u), self.num_vars))	
			u_p_reconstructed = u_p_reconstructed + omega[k_s]*P[k_s]
		return (np.matmul(Q, u_p_reconstructed.reshape(len(u),self.num_vars,1))).reshape((len(u),self.num_vars))[self.r:-self.r]

	def weno_characteristic_right(self, f):
		u = np.flip(np.pad(f, ((self.r-1, self.r), (0,0)), mode=self.boundary_type), axis=0)
		u_p_reconstructed = np.zeros(u.shape) 
		P = np.zeros(np.append((self.r+1), u.shape))
		Q, Q_inverse = self.get_characteristic_transform(u)	
		
		beta = calculate_beta_characteristic(u, self.r, Q, Q_inverse, shift)
		alpha = self.b/(beta+self.epsilon)**(self.r+1)
		omega = alpha / alpha.sum(axis=0)

		for k_s in range(self.r+1):
			for l in range(self.r+1):
				P[k_s,:,:] += self.a[k_s,l]*np.matmul(Q_inverse, shift(u, k_s-l).reshape((len(u), self.num_vars, 1))).reshape((len(u), self.num_vars))
			u_p_reconstructed = u_p_reconstructed + omega[k_s]*P[k_s]
		return np.flip((np.matmul(Q, u_p_reconstructed.reshape(len(u),self.num_vars,1))).reshape((len(u),self.num_vars)), axis=0)[self.r:-self.r]

	def flux_split(self):
		max_characteristic = self.get_maximum_characteristic(self.u_p)	
		
		if self.characteristic:
			u_p_reconstructed_l = self.weno_characteristic_left(self.u_p)
			u_p_reconstructed_r = self.weno_characteristic_right(self.u_p)
		else:
			u_p_reconstructed_l = self.weno_left(self.u_p)
			u_p_reconstructed_r = self.weno_right(self.u_p)

		u_c_reconstructed_l = self.get_conservative_vars(u_p_reconstructed_l)
		u_c_reconstructed_r = self.get_conservative_vars(u_p_reconstructed_r)

		flux_left = self.get_flux(u_p_reconstructed_l) + max_characteristic*u_c_reconstructed_l
		flux_right = self.get_flux(u_p_reconstructed_r) - max_characteristic*u_c_reconstructed_r
	
		return 1/2*(flux_left + flux_right)

	def get_dudx(self):
		u_split = np.pad(self.flux_split(), [(self.r+1, self.r+1), (0,0)], mode=self.boundary_type)
	
		dudx = np.zeros(np.shape(u_split))
	
		for i in range(len(self.d)):
			dudx += self.d[i]*(shift(u_split, -i)-shift(u_split, i+1))
		# print(dudx)
		return (dudx/self.h)[self.r+1:-self.r, :]
	
	def get_dudx_centered(self, u):
		u_reconstucted = np.pad((self.weno_left(u)+self.weno_right(u))/2, [(self.r+1, self.r+1), (0,0)], mode=self.boundary_type) # central reconstruction
		dudx = np.zeros(np.shape(u_reconstucted))
		for i in range(len(self.d)):
			dudx += self.d[i]*(shift(u_reconstucted, -i)-shift(u_reconstucted, i+1))
		# print((dudx/self.h)[self.r+1:-self.r, :])
		return (dudx/self.h)[self.r+1:-self.r, :]

	def h(self, gamma, C_v, q): # specific enthalpy of given phase
		return gamma*C_v*self.T + q

	def e(self, gamma, Pi, q): # specific energy of given phase
		return (self.P+gamma*Pi)/(gamma-1)*self.nu(gamma, C_v, Pi) + q

	def nu(self, gamma, C_v, Pi): # specific volume of given phase
		return (gamma-1)*C_v*self.T / (self.p+Pi)

	def g(self, gamma, C_v, q_prime, Pi, q): # Gibbs free energy of given phase
		return (gamma*C_v - q_prime)*self.T - C_v*self.T*np.log(self.T**gamma/(P+Pi)**(gamma-1)) + q

	def get_heat_flux(self, u_p):
		T = np.array([self.get_temperature_from_primitive(u_p)]).T
		dTdx = self.get_dudx_centered(T)
		alpha_1 = (self.u_p[:,0]*self.u_p[:,3]) / ((self.u_p[:,2] + self.gamma_1*self.Pi_1) / ((self.gamma_1 - 1)*self.c_v_1*T.T))
		alpha_2 = (self.u_p[:,0]*(1-self.u_p[:,3])) / ((self.u_p[:,2] + self.gamma_2*self.Pi_2) / ((self.gamma_2 - 1)*self.c_v_2*T.T))
		q = (alpha_1*self.Lambda_1 + alpha_2*self.Lambda_2)*dTdx.T
		dqdx = np.zeros(np.shape(u_p))
		# print(dqdx)

		dqdx[:,2] = self.get_dudx_centered(q.T).T
		# print(dqdx)
		# print(dqdx)

		# print(np.array([np.zeros(len(T)), np.zeros(len(T)), dqdx, np.zeros(len(T))]))
		return dqdx 

	# def calculate_entropy(self): # modify this
	# 	e=(self.u_p[:,2]+self.gamma*self.Pi)/(self.u_p[:,0]*(self.gamma-1))
	# 	self.entropy = np.log(e)-(self.gamma-1)*np.log(self.u_p[:,0])

	# Definitely wrong, probably don't need
	# def get_temperature_from_conservative(self, u_c):
	# 	return self.e_from_conservative(u_c) / (u_c[:,3]/u_c[:,0] * self.c_v_1 + (1 - u_c[:,3]/u_c[:,0]) * self.c_v_2)

	# Probably wrong
	def get_temperature_from_primitive(self, u_p):
	

	def rk4(self):
		self.tau = self.CFL * self.h / self.get_maximum_characteristic(self.u_p) 
		u_c = self.u_c
		k1 = -self.get_dudx() - self.get_heat_flux(self.u_p)
		self.u_c = u_c + self.tau*1/2*k1
		# self.T = self.get_temperature_from_conservative(self.u_c)
		self.u_p = self.get_primitive_vars(self.u_c)
		print("1")
		print(self.u_p)
		# self.u_p = self.get_primitive_vars(self.u_c)		
		
		k2 = -self.get_dudx() - self.get_heat_flux(self.u_p)
		self.u_c = u_c + self.tau*1/2*k2
		self.u_p = self.get_primitive_vars(self.u_c)
		print("2")
		k3 = -self.get_dudx() - self.get_heat_flux(self.u_p)
		self.u_c = u_c + self.tau*k3
		self.u_p = self.get_primitive_vars(self.u_c)
		print("3")
		k4 = -self.get_dudx() - self.get_heat_flux(self.u_p)
		self.u_c = u_c + self.tau*(1/6*k1+1/3*k2+1/3*k3+1/6*k4)
		self.u_p = self.get_primitive_vars(self.u_c)	
		print("4")

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
	
	
	def run(self):
		# print(self.T)
		# sys.exit()
		plt.ion()
		fig = plt.figure(1)
		ax = fig.add_subplot(111)
		line1, = ax.plot(self.x,self.u_p[:,0],'r-')
		line2, = ax.plot(self.x,self.u_p[:,1],'b-')
		line3, = ax.plot(self.x,self.u_p[:,2],'g-')
		while self.t < self.t_f:
			if self.time_int == 'rk4':
				self.rk4()
				# print(self.t)
			elif self.time_int == 'sdc': 
				self.sdc()
			self.t = self.t + self.tau
			# elif self.time_int == 'sdc6':
			# 	self.sdc6()
			line1.set_ydata(self.u_p[:,0])
			line2.set_ydata(self.u_p[:,1])
			line3.set_ydata(self.u_p[:,2])
			fig.canvas.draw()
			fig.canvas.flush_events()
		# self.calculate_entropy()
		self.elapsed_time = time.clock() - self.start_time
		print('done, time: ', self.elapsed_time)		


x0 = -5
xf = 5
tf = 2.0
N = 100
# CFL = 0.5
# problem_type = 'moving-shock'
characteristic = True

result = euler(x0, xf, tf, N, 0.1, 3, 4, characteristic, 'edge', 'rk4')

plt.plot(result.x, result.u_p[:,0], marker='.')
plt.show()