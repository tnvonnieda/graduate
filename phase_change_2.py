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

class phase_change:
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

		self.x_half = np.linspace(x_0 + self.h/2, self.x_f-self.h/2, self.k-1)
		
		self.boundary_type = boundary_type
		self.characteristic = characteristic

		# self.tau = 0.0
		self.epsilon = 10.0**-40
		self.entropy = None

		if time_int == 'sdc':
			self.q = get_quadrature_weights(p)
			self.quad_points = get_quadrature_points(p)

		# SET INITIAL CONDITION

		# Liquid water - water vapor Quasi-isobaric flow
		conds = [self.x < 0, self.x >= 0]
		self.u_p = np.array([
			np.piecewise(self.x, conds, [932, 0.6]), # kg/m^3
			np.piecewise(self.x, conds, [5.0, 5.0]), # m/s
			np.piecewise(self.x, conds, [100000, 100000]), # kPa
			np.piecewise(self.x, conds, [1.0, 0]) # left side liquid, right side vapor
			]).T
		
		self.T = self.get_temperature_from_primitive(self.u_p) # initialize T, should update
		# T_in = T_sat = 372.79 K, T_out = 1000 K ?
		# P(x) = P_0 = 1 Bar (constant across x)
		self.u_c = self.get_conservative_vars(self.u_p)

		self.u_p0 = self.u_p
		self.u_c0 = self.u_c
		self.num_vars = self.u_p.shape[1]
		self.time_int = time_int
		self.run()
		# self.PLOT_TYPE = REAL_TIME # REAL_TIME/END_TIME

	# STIFFENED GAS EQUATIONS
	# def h_k(self, gamma_k, C_v_k, q_k): # specific enthalpy of given phase
	# 	return gamma_k*C_v_k*self.T + q_k

	# def e_k(self, gamma_k, Pi_k, q_k): # specific energy of given phase
	# 	return (self.P+gamma_k*Pi_k)/(gamma_k-1)*self.nu_k(gamma_k, C_v_k, Pi_k) + q_k

	# def nu_k(self, gamma_k, C_v_k, Pi_k): # specific volume of given phase
	# 	return (gamma_k-1)*C_v_k*self.T / (self.P+Pi_k)

	# def g_k(self, gamma_k, C_v_k, q_prime_k, Pi_k, q_k): # Gibbs free energy of given phase
	# 	return (gamma_k*C_v_k - q_prime_k)*self.T - C_v_k*self.T*np.log(self.T**gamma_k/(self.P+Pi_k)**(gamma_k-1)) + q_k
	def get_temperature_from_primitive(self, u_p):
		T_reciprocal = u_p[:,3]*(self.gamma_1 - 1)*self.C_v_1/(1/u_p[:,0]*(u_p[:,2] + self.Pi_1)) + (1-u_p[:,3])*(self.gamma_2 - 1)*self.C_v_2/(1/u_p[:,0]*(u_p[:,2] + self.Pi_2))
		return 1/T_reciprocal

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

	def speed_of_sound(self, u_p):
		T = self.get_temperature_from_primitive(u_p)
		rho_1 = (u_p[:,2] + self.Pi_1)/((self.gamma_1 - 1)*self.C_v_1*T)
		rho_2 = (u_p[:,2] + self.Pi_2)/((self.gamma_2 - 1)*self.C_v_2*T)
		# print(rho_1)
		# print("T:")
		# print(T)
		# print("P:")
		# print(u_p[:,2])
		c_1 = np.sqrt(abs(self.gamma_1*u_p[:,2]+self.Pi_1)/rho_1)
		c_2 = np.sqrt(abs(self.gamma_2*u_p[:,2]+self.Pi_2)/rho_2)
		# print(c_1)
		alpha_1 = u_p[:,0]*u_p[:,3]/rho_1
		alpha_2 = 1 - alpha_1

		# c = np.sqrt(1/u_p[:,0]*rho_1*c_1**2*rho_2*c_2**2/(alpha_1*rho_2*c_2**2 + alpha_2*rho_1*c_1**2))
		c = np.sqrt(1/u_p[:,0] * 1/(alpha_1/(rho_1*c_1**2) + alpha_2/(rho_2*c_2**2)))
		return c

	def get_maximum_characteristic(self, u_p):
		return np.max(abs(self.speed_of_sound(u_p)) + abs(u_p[:,1]))

	def get_flux(self, u_p):
		e = self.get_specific_energy(u_p)
		return np.array([
			u_p[:,0]*u_p[:,1], 
			u_p[:,0]*u_p[:,1]**2+u_p[:,2], 
			(u_p[:,0]*e+1/2*u_p[:,0]*u_p[:,1]**2+u_p[:,2])*u_p[:,1],
			u_p[:,0]*u_p[:,1]*u_p[:,3]]).T

	def get_specific_energy(self, u_p):
		T = self.get_temperature_from_primitive(u_p)
		e = u_p[:,3]*self.e_1(u_p[:,2], T) + (1-u_p[:,3])*self.e_2(u_p[:,2], T)
		return e

	def get_conservative_vars(self, u_p):
		# T = self.get_temperature_from_primitive(u_p)
		e = self.get_specific_energy(u_p)
		return np.array([
			u_p[:,0], 
			u_p[:,0]*u_p[:,1], 
			u_p[:,0]*(e+1/2*u_p[:,1]**2),
			u_p[:,0]*u_p[:,3]]).T

	def weno_left(self):
		u = np.pad(self.u_p, ((self.r, self.r-1), (0,0)), mode=self.boundary_type)
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
		u = np.flip(np.pad(self.u_p, ((self.r-1, self.r), (0,0)), mode=self.boundary_type), axis=0)
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
		u = np.pad(self.u_p, ((self.r, self.r-1), (0,0)), mode=self.boundary_type)
		u_p_reconstructed = np.zeros(u.shape) 
		P = np.zeros(np.append((self.r+1), u.shape))
		print("get characteristic left")
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

	def weno_characteristic_right(self):
		u = np.flip(np.pad(self.u_p, ((self.r-1, self.r), (0,0)), mode=self.boundary_type), axis=0)
		u_p_reconstructed = np.zeros(u.shape) 
		P = np.zeros(np.append((self.r+1), u.shape))
		print("get characteristic right")
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
		# print("get characteristic in flux")
		max_characteristic = self.get_maximum_characteristic(self.u_p)
		# max_characteristic = 5
		# print(max_characteristic)	
		# print("characteristic returned")
		if self.characteristic:
			u_p_reconstructed_l = self.weno_characteristic_left()
			u_p_reconstructed_r = self.weno_characteristic_right()
		else:
			u_p_reconstructed_l = self.weno_left()
			u_p_reconstructed_r = self.weno_right()		

		u_c_reconstructed_l = self.get_conservative_vars(u_p_reconstructed_l)
		u_c_reconstructed_r = self.get_conservative_vars(u_p_reconstructed_r)
		# plt.plot(self.x, self.u_c[:,0], marker='.')
		# plt.plot(self.x_half, u_p_reconstructed_l[:,0])
		# plt.plot(self.x_half, u_p_reconstructed_r[:,0])
		
		# plt.show()
		# sys.exit()
		flux_left = self.get_flux(u_p_reconstructed_l) + max_characteristic*u_c_reconstructed_l
		flux_right = self.get_flux(u_p_reconstructed_r) - max_characteristic*u_c_reconstructed_r
		
		return 1/2*(flux_left + flux_right)

	def get_dudx(self):
		u_split = np.pad(self.flux_split(), [(self.r+1, self.r+1), (0,0)], mode=self.boundary_type)
		dudx = np.zeros(np.shape(u_split))
	
		for i in range(len(self.d)):
			dudx += self.d[i]*(shift(u_split, -i)-shift(u_split, i+1))
	
		return (dudx/self.h)[self.r+1:-self.r, :]
	
	# def calculate_entropy(self):
	# 	e=(self.u_p[:,2]+self.gamma*self.Pi)/(self.u_p[:,0]*(self.gamma-1))
	# 	self.entropy = np.log(e)-(self.gamma-1)*np.log(self.u_p[:,0])

	def g_1(self, p, T):
		return (self.gamma_1*self.C_v_1 - self.q_1_prime)*T - self.C_v_1*T*np.log(T**self.gamma_1/((p+self.Pi_1)**(self.gamma_1-1)))+self.q_1

	def g_2(self, p, T):
		return (self.gamma_2*self.C_v_2 - self.q_2_prime)*T - self.C_v_2*T*np.log(T**self.gamma_2/((p+self.Pi_2)**(self.gamma_2-1)))+self.q_2

	def e_1(self, p, T):
		return (p+self.gamma_1*self.Pi_1)/(self.gamma_1 - 1)*self.nu_1(p, T) + self.q_1

	def e_2(self, p, T):
		return (p+self.gamma_2*self.Pi_2)/(self.gamma_2 - 1)*self.nu_2(p, T) + self.q_2

	def nu_1(self, p, T):
		return (self.gamma_1 - 1)*self.C_v_1*T/(p+self.Pi_1)

	def nu_2(self, p, T):
		return (self.gamma_2 - 1)*self.C_v_2*T/(p+self.Pi_2)

	# def f2(self, T, p):
	# 	return self.g_2(T, p) - self.g_1(T, p)

	# def f(self, p, e_0, nu_0):
	# 	self.T = optimize.newton(self.f2, self.T, args=(p,))
	# 	return ( (e_0 - self.e_2(p)) / (self.e_1(p)-self.e_2(p)) ) - (nu_0 - self.nu_2(p)) / (self.nu_1(p) - self.nu_2(p))

	def enforce_equilibrium(self):
		nu_0 = 1/self.u_c[0] # specific volume
		e_0 = u_c[:,2]/u_c[:,0]-1/2*(u_c[:,1]/u_c[:,0])**2 # specific energy
		P_relaxed, T_relaxed = optimize.newton(self.f, self.u_p[:,2], args=(nu_0, e_0))

	def get_primitive_vars(self, u_c):
		e = u_c[:,2]/u_c[:,0] - 1/2*(u_c[:,1]/u_c[:,0])**2
		rho = u_c[:,0]
		v = u_c[:,1]/u_c[:,0]
		Y_1 = u_c[:,1]/u_c[:,2]
		q = Y_1*self.q_1 + (1-Y_1)*self.q_2
		A_1 = Y_1 * (self.gamma_1 - 1)*self.C_v_1/(Y_1*self.C_v_1 + (1-Y_1)*self.C_v_2)*(rho*(e-q)-self.Pi_1)
		A_2 = (1-Y_1) * (self.gamma_2 - 1)*self.C_v_2/(Y_1*self.C_v_1 + (1-Y_1)*self.C_v_2)*(rho*(e-q)-self.Pi_2)
		p = 1/2 * (A_1 + A_2 - (self.Pi_1 + self.Pi_2)) + np.sqrt(1/4 * (A_2 - A_1 - (self.Pi_2 - self.Pi_1))**2 + A_1*A_2)
		return np.array([rho, v, p, Y_1]).T

	def rk4(self):
		self.tau = self.CFL * self.h / self.get_maximum_characteristic(self.u_p) 
		u_c = self.u_c

		k1 = -self.get_dudx()
		self.u_c = u_c + self.tau*1/2*k1
		plt.plot(self.x, self.u_c[:,0])
		
		plt.show()
		sys.exit()
		# self.u_p = self.get_primitive_vars(self.u_c)
		# self.u_p = self.enforce_equilibrium(self.u_p)
		
		# enforce equilibrium here
		# self.u_p = self.get_primitive_vars(self.u_c)		
		
		k2 = -self.get_dudx()
		self.u_c = u_c + self.tau*1/2*k2
		self.u_p = self.get_primitive_vars(self.u_c)

		k3 = -self.get_dudx()
		self.u_c = u_c + self.tau*k3
		self.u_p = self.get_primitive_vars(self.u_c)

		k4 = -self.get_dudx()
		self.u_c = u_c + self.tau*(1/6*k1+1/3*k2+1/3*k3+1/6*k4)
		self.u_p = self.get_primitive_vars(self.u_c)
	
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
				plt.plot(self.x, self.u_c[:,0])
				plt.show()
				sys.exit()
				dudx[j-1,k] = self.get_dudx()
				w[j,k] += self.tau*(self.quad_points[j]-self.quad_points[j-1])/2*(-dudx[j-1,k] + dudx[j-1,k-1])
			self.u_c = w[self.p-1,k]
			self.u_p = self.get_primitive_vars(self.u_c)

			if k < 2*self.p-2:				
				dudx[self.p-1,k] = self.get_dudx()	

		# self.t = self.t + self.tau

	def run(self):
		
		# plt.plot(self.x, self.u_p[:,0])
		
		# plt.show()
		# sys.exit()
		# plt.ion()
		# fig = plt.figure(1)
		# ax = fig.add_subplot(111)
		# line1, = ax.plot(self.x,self.u_p[:,0],'r-')
		# line2, = ax.plot(self.x,self.u_p[:,1],'b-')
		# line3, = ax.plot(self.x,self.u_p[:,2],'g-')
		while self.t < self.t_f:
			if self.time_int == 'rk4':
				self.rk4()
				# print(self.t)
			elif self.time_int == 'sdc': 
				self.sdc()
			self.t = self.t + self.tau
			# elif self.time_int == 'sdc6':
			# 	self.sdc6()
			# line1.set_ydata(self.u_p[:,0])
			# line2.set_ydata(self.u_p[:,1])
			# line3.set_ydata(self.u_p[:,2])
			# fig.canvas.draw()
			# fig.canvas.flush_events()
		# self.calculate_entropy()
		self.elapsed_time = time.clock() - self.start_time
		print('done, time: ', self.elapsed_time)		

x0 = -5
xf = 5
tf = 2.0
N = 200
CFL = 0.01
characteristic = True

rk4 = phase_change(x0, xf, tf, N, CFL, 2, 3, characteristic, 'edge', 'rk4')
# plt.plot(rk4.x, rk4.u_p[:,0], marker='.')


# sdc8 = euler(x0, xf, tf, N, 0.1, 3, 4, characteristic, 'edge', 'rk4', problem_type, False)


# plt.plot(sdc8.x, sdc8.u_p[:,0], marker='.')

# plt.show()
