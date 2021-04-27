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
from scipy import optimize

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
		self.characteristic = characteristic
		self.boundary_type = boundary_type
		self.epsilon = 10**-40
		self.zero_equivalent = 10**-7
		self.one_equivalent = 1 - 10**-7
		# self.newton_epsilon = 10**-10
		# self.max_newton_iterations = 100
		
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
		# Material properties
		# liquid water
		# self.Pi_1 = 7028*10**5
		# self.C_v_1 = 3610
		# self.C_p_1 = 4285
		# self.gamma_1 = self.C_p_1/self.C_v_1
		# self.q_1 = -1177788
		# self.q_1_prime = 0 # J/(kg K)
		# self.lambda_1 = 0.6788 # w/(m K)

		# # water vapor
		# self.Pi_2 = 0 # Pa  
		# self.C_v_2 = 955
		# self.C_p_2 = 1401
		# self.gamma_2 = self.C_p_2 / self.C_v_2
		# self.q_2 = 2077616
		# self.q_2_prime = 14370 # J/(kg K)
		# self.lambda_2 = 249.97 # w/(m K)

		self.Pi_1 = 10**9
		self.C_v_1 = 1816
		self.C_p_1 = 4267
		self.gamma_1 = self.C_p_1/self.C_v_1
		self.q_1 = -1167*10**3
		self.q_1_prime = 0 # J/(kg K)
		self.lambda_1 = 0.6788 # w/(m K)

		# water vapor
		self.Pi_2 = 0 # Pa  
		self.C_v_2 = 1040
		self.C_p_2 = 1487
		self.gamma_2 = self.C_p_2 / self.C_v_2
		self.q_2 = 2030*10**3
		self.q_2_prime = -23.26*10**3 # J/(kg K)
		self.lambda_2 = 249.97 # w/(m K)

		if time_int == 'sdc':
			self.q = get_quadrature_weights(p)
			self.quad_points = get_quadrature_points(p)		

		# DODECANE #
		# Liquid
		# self.Pi_1 = 4*10**8
		# self.C_v_1 = 1077
		# self.C_p_1 = 2534
		# self.gamma_1 = self.C_p_1/self.C_v_1
		# self.q_1 = -755*10**3
		# self.q_1_prime = 0 # J/(kg K)
		# self.lambda_1 = 0.140 # w/(m K)

		# # Vapor
		# self.Pi_2 = 0 # Pa  
		# self.C_v_2 = 1953
		# self.C_p_2 = 2005
		# self.gamma_2 = self.C_p_2 / self.C_v_2
		# self.q_2 = -237*10**3
		# self.q_2_prime = -24.483*10**3 # J/(kg K)
		# self.lambda_2 = 200 # w/(m K)
		# ----- #


		# Expansion tube
		# rho = 1150 # kg/m^3
		# v_l = -2 # m/s
		# Y_1 = 1-0.00005555
		# p = 101325 # Pa
		# v_r = 2 # m/s
		# conds = [self.x <= 0.5, self.x > 0.5]

		# self.u_p = np.array([
		# 	np.piecewise(self.x, conds, [rho, rho]),
		# 	np.piecewise(self.x, conds, [v_l, v_r]),
		# 	np.piecewise(self.x, conds, [p, p]),
		# 	np.piecewise(self.x, conds, [Y_1, Y_1])]).T

		# Two-phase shock tube
		# p_l = 10**8 # Pa
		# rho_l = 500 # kg/m^3

		# p_r = 101325 # Pa
		# rho_r = 2 # kg/m^3
		# conds = [self.x <= 0.75, self.x > 0.75]

		# # rho_1 = 1/self.nu_1(p_l,T)
		# # rho_2 = 1/self.nu_2(p_r,T)

		# self.u_p = np.array([
		# 	np.piecewise(self.x, conds, [rho_l, rho_r]),
		# 	np.piecewise(self.x, conds, [0, 0]),
		# 	np.piecewise(self.x, conds, [p_l, p_r]),
		# 	np.piecewise(self.x, conds, [0.99999, 0.00001])]).T
		
		# Liquid water - water vapor Quasi-isobaric flow
		# Q_0 = -3.76*10**6
		conds = [self.x <= 0, self.x > 0]		
		p = 100000
		T_sat = self.get_saturation_temperature(p,1)
		T_LI = T_sat
		T_VO = 1000		
		X_v = 0.05 

		m_dot = self.lambda_2/(self.C_p_2*X_v)*np.log((self.C_p_2*T_VO + self.q_2 - self.C_p_1*T_LI - self.q_1)/(self.C_p_2*T_sat + self.q_2 - self.C_p_1*T_LI - self.q_1))
		Q_0 = m_dot*(self.C_p_1*T_LI + self.q_1 - self.C_p_2*T_VO - self.q_2)

		# T_left = T_LI + (T_sat - T_LI)*np.exp(m_dot*self.C_p_2*x/self.lambda_2) # temperature left of interface
		# T_right = self.C_p_1/self.C_p_2 * T_LI + (self.q_1 - self.q_2)/self.C_p_2 + (T_sat - self.C_p_1/self.C_p_2 * T_LI - (self.q_1 - self.q_2)/self.C_p_2)*np.exp(m_dot*self.C_p_2*x/self.lambda_2)
		p = np.piecewise(self.x, conds, [p, p])
		T = np.piecewise(self.x, conds, [lambda x: T_LI + (T_sat - T_LI)*np.exp(m_dot*self.C_p_2*x/self.lambda_2), lambda x: self.C_p_1/self.C_p_2 * T_LI + (self.q_1 - self.q_2)/self.C_p_2 + (T_sat - self.C_p_1/self.C_p_2 * T_LI - (self.q_1 - self.q_2)/self.C_p_2)*np.exp(m_dot*self.C_p_2*x/self.lambda_2)])
		
		alpha_1 = np.piecewise(self.x, conds, [0.999999, 0.000001])
		alpha_2 = 1 - alpha_1

		rho = 1/self.nu_1(p,T)*alpha_1 + 1/self.nu_2(p,T)*alpha_2
		Y_1 = alpha_1 * 1/self.nu_1(p,T) / rho

		self.u_p = np.array([
			rho, # kg/m^3
			m_dot*1/rho, # m/s
			p, # kPa
			Y_1 # left side liquid, right side vapor
			]).T
		
		self.T = self.get_temperature_from_primitive(self.u_p)
		self.e_0 = self.u_p[:,3]*self.e_1(self.u_p[:,2],self.T) + (1-self.u_p[:,3])*self.e_2(self.u_p[:,2],self.T)
		self.nu_0 = self.u_p[:,3]*self.nu_1(self.u_p[:,2],self.T) + (1-self.u_p[:,3])*self.nu_2(self.u_p[:,2],self.T)
		
		self.u_c = self.get_conservative_vars(self.u_p)
		self.u_p0 = self.u_p
		self.u_c0 = self.u_c
		print(self.get_normal_pressure(Y_1, rho, self.e_0))
		# plt.plot(self.x, 1/specific_volume)
		# plt.plot(self.x, self.get_normal_pressure(Y_1, 1/specific_volume))
		# plt.show()
		# sys.exit()
		# self.u_p0 = self.u_p
		# self.u_c0 = self.u_c
		self.num_vars = self.u_p.shape[1]
		self.time_int = time_int
		self.run()
		# self.PLOT_TYPE = REAL_TIME # REAL_TIME/END_TIME

	def pad(self, u, left_pad, right_pad):
		if left_pad >= 0 and right_pad >= 0:
			return np.pad(u, ((left_pad, right_pad), (0,0)), mode=self.boundary_type)
		elif left_pad < 0 and right_pad >=0:
			return u[1:self.k]
		elif left_pad >= 0 and right_pad < 0:
			return u[0:self.k-1]

	# ---------------- EQUATION OF STATE ---------------- #
	def h_1(self, T):
		return self.gamma_1*self.C_v_1*T + self.q_1

	def h_2(self, T):
		return self.gamma_2*self.C_v_2*T + self.q_2

	def g_1(self, p, T):
		return (self.gamma_1*self.C_v_1 - self.q_1_prime)*T - self.C_v_1*T*np.log(T**self.gamma_1/((p+self.Pi_1)**(self.gamma_1-1)))+self.q_1

	def g_2(self, p, T):
		return (self.gamma_2*self.C_v_2 - self.q_2_prime)*T - self.C_v_2*T*np.log(T**self.gamma_2/((p+self.Pi_2)**(self.gamma_2-1)))+self.q_2

	def g_1_prime(self, p, T): # d/dT
		return self.gamma_1*self.C_v_1 - self.q_1_prime - self.C_v_1*(np.log(T**self.gamma_1*(p+self.Pi_1)**(1-self.gamma_1)) + self.gamma_1)

	def g_2_prime(self, p, T): # d/dT
		return self.gamma_2*self.C_v_2 - self.q_2_prime - self.C_v_2*(np.log(T**self.gamma_2*(p+self.Pi_2)**(1-self.gamma_2)) + self.gamma_2)

	# def e_1(self, p, T):
	# 	return (p+self.gamma_1*self.Pi_1)/(self.gamma_1 - 1)*self.nu_1(p, T) + self.q_1

	def e_1(self, p, T):
		return (p+self.gamma_1*self.Pi_1)/(p+self.Pi_1)*self.C_v_1*T + self.q_1

	# def e_1_prime(self, p, T):
	# 	return self.C_v_1*T*self.Pi_1*(1 - self.gamma_1)/(p+self.Pi_1)**2

	# def e_2(self, p, T):
	# 	return (p+self.gamma_2*self.Pi_2)/(self.gamma_2 - 1)*self.nu_2(p, T) + self.q_2

	def e_2(self, p, T):
		return (p+self.gamma_2*self.Pi_2)/(p+self.Pi_2)*self.C_v_2*T + self.q_2

	# def e_2_prime(self, p, T):
	# 	return self.C_v_2*T*self.Pi_2*(1 - self.gamma_2)/(p+self.Pi_2)**2

	def nu_1(self, p, T):
		return (self.gamma_1 - 1)*self.C_v_1*T/(p+self.Pi_1)

	def nu_1_prime(self, P, T):
		return self.C_v_1*(1-self.gamma_1)*T/(P + self.Pi_1)**2

	def nu_2(self, p, T):
		return (self.gamma_2 - 1)*self.C_v_2*T/(p+self.Pi_2)

	def nu_2_prime(self, P, T):
		return self.C_v_2*(1-self.gamma_2)*T/(P + self.Pi_2)**2
	# --------------------------------------------------- #

	# ---------------- SATURATION CURVE ---------------- #
	def f1(self, T, p):
		# print(T)
		return self.g_1(p,T) - self.g_2(p,T)

	def f1_prime(self, T, p):
		return self.g_1_prime(p,T) - self.g_2_prime(p,T)
	
	def get_saturation_temperature(self, p, T):
		# optimize.newton(self.f1, T, args=(p,), full_output=True)
		T_sat = optimize.newton(self.f1, T, args=(p,))
		return T_sat
		# print(T_sat)
		# if type(T_sat) == tuple:
		# 	# print(T_sat[1].converged)
		# 	if T_sat[1].converged == False:
		# 		print("Non-converging solution:")
		# 		print("p: ", p)
		# 		print("T: ", T)
		# 	return T_sat[1].root
		# else:
		# 	if np.any(T_sat.converged == False):
		# 		print("Non-converging solution:")
		# 		print("p: ", p)
		# 		print("T: ", T)
		# 	return T_sat.root

		# return T_sat
	# -------------------------------------------------- #

	# ----- FOR HEAT FLUX CALCULATION ----- #
	def get_heat_flux(self, u_p):
		T = np.array([self.get_temperature_from_primitive(u_p)]).T
		dTdx = self.get_dudx_centered(T)
		alpha_1 = (u_p[:,3]*u_p[:,0]*self.nu_1(u_p[:,2],T.T))
		alpha_2 = 1 - alpha_1
		# alpha_1 = (self.u_p[:,0]*self.u_p[:,3]) / ((self.u_p[:,2] + self.gamma_1*self.Pi_1) / ((self.gamma_1 - 1)*self.C_v_1*T.T))
		# alpha_2 = (self.u_p[:,0]*(1-self.u_p[:,3])) / ((self.u_p[:,2] + self.gamma_2*self.Pi_2) / ((self.gamma_2 - 1)*self.C_v_2*T.T))
		q = (alpha_1*self.lambda_1 + alpha_2*self.lambda_2)*dTdx.T
		dqdx = np.zeros(np.shape(self.u_p))
		dqdx[:,2] = self.get_dudx_centered(q.T).T
		return dqdx 

	def get_dudx_centered(self, u):
		u_reconstucted = np.pad((self.weno_left(u)+self.weno_right(u))/2, ((self.r+1, self.r+1), (0,0)), mode='reflect', reflect_type='odd') # central reconstruction
		dudx = np.zeros(np.shape(u_reconstucted))
		for i in range(len(self.d)):
			dudx += self.d[i]*(shift(u_reconstucted, -i)-shift(u_reconstucted, i+1))
		# print((dudx/self.h)[self.r+1:-self.r, :])
		return (dudx/self.h)[self.r+1:self.k+self.r+1, :]
	# -------------------------------------- #

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
		c_1 = np.sqrt(abs(self.gamma_1*u_p[:,2]+self.Pi_1)/rho_1)

		c_2 = np.sqrt(abs(self.gamma_2*u_p[:,2]+self.Pi_2)/rho_2)
		alpha_1 = u_p[:,0]*u_p[:,3]/rho_1
		alpha_2 = 1 - alpha_1

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
		e = self.get_specific_energy(u_p)
		return np.array([
			u_p[:,0], 
			u_p[:,0]*u_p[:,1], 
			u_p[:,0]*(e+1/2*u_p[:,1]**2),
			u_p[:,0]*u_p[:,3]]).T

	def f2(self, p, e_0, nu_0):
		# T = self.get_saturation_temperature(p, np.ones(self.k))
		
		self.T = self.get_saturation_temperature(p, np.ones(len(p)))
		return ( (e_0 - self.e_2(p,self.T)) / (self.e_1(p,self.T)-self.e_2(p,self.T)) ) - (nu_0 - self.nu_2(p,self.T)) / (self.nu_1(p,self.T) - self.nu_2(p,self.T))

	def f2_prime(self, p, T, i):
		return ((self.e_2(p,T) - self.e_0)*self.e_1_prime(p,T) - (self.e_0 - self.e_1(p,T))*self.e_2_prime(p,T)) / (self.e_2(p,T) - self.e_1(p,T))**2 + ((self.nu_2(p,T)-self.nu_0)*self.nu_1_prime(p,T) - (self.nu_0 - self.nu_1(p,T))*self.nu_2_prime(p,T)) / (self.nu_2(p,T) - self.nu_1(p,T))**2

	def f3(self, p, e_0, nu_0): # element wise 
		# print(p)
		T = self.get_saturation_temperature(p, 1)
		return ( (e_0 - self.e_2(p,T)) / (self.e_1(p,T)-self.e_2(p,T)) ) - (nu_0 - self.nu_2(p,T)) / (self.nu_1(p,T) - self.nu_2(p,T))

	def solve_phase_change(self, p, e_0, nu_0, Y_1, T_): # 
		# T = np.ones(len(p))
		p_relaxed = np.empty_like(p)
		for i in range(len(p)):
			e_0_cell = e_0[i]
			nu_0_cell = nu_0[i]
			# print("p: ", p[i])
			# print("T: ", T_[i])
			# print(Y_1[i])
			# print(e_0_cell)
			# print(nu_0_cell)
			p_result = optimize.newton(self.f3, 1, args=(e_0_cell, nu_0_cell), tol=10**-7, full_output=True)
			# print(p_result[1])
			if p_result[1].converged == False:
				print("Non-converging solution:")
				print("p: ", p[i])
				print("T: ", T_[i])
				print("Y_1: ", Y_1[i])
				sys.exit()
			p_relaxed[i] = p_result[1].root
		# p_relaxed = optimize.newton(self.f2, np.ones(len(p)), args=(e_0, nu_0,), full_output=True)
		return p_relaxed
		# if type(p_relaxed) == tuple:
		# 	if p_relaxed[1].converged == False:
		# 		print("Non-converging solution:")
		# 		print("p: ", p)
		# 		print("T: ", T)
		# 		sys.exit()
		# 	return p_relaxed[1].root
		# else:
		# 	if np.any(p_relaxed.converged == False):
		# 		condition = (p_relaxed.converged == False)
		# 		print("Non-converging solution:")
		# 		print("p: ", p[condition])
		# 		print("T: ", T_[condition])
		# 		print("Y_1: ", Y_1[condition])
		# 		print("e_0: ", e_0[condition])
		# 		print("nu_0: ", nu_0[condition])
		# 		for i in range(len(p[condition])):
		# 			# print(e_0[i])
		# 			# print(nu_0[i])
		# 			# print(p[i])
		# 			# print(T_[i])
		# 			# print(Y_1[i])
		# 			e_0_cell = e_0[condition][i]
		# 			nu_0_cell = nu_0[condition][i]
		# 			p_relaxed_ind = optimize.newton(self.f3, 1, args=(e_0_cell, nu_0_cell))
		# 			print(p_relaxed_ind)
		# 		# print(p_relaxed.converged)
				
		# 		sys.exit()
		# 	return p_relaxed.root

		# for i in range(len(p)):
			# P = p[i]

			# T = self.get_saturation_temperature(P, 1)
			# # delta = self.f2(P,T,i) / self.f2_prime(P,T,i)
			# # iteration = 0
			# # while abs(delta) >= self.newton_epsilon and iteration < self.max_newton_iterations:
			# # 	delta = self.f2(P,T,i) / self.f2_prime(P,T,i)
			# # 	P = P - delta
			# # 	T = self.get_saturation_temperature(P, T)
			# # 	++ iteration
			# # p_relaxed[i] = P
			# p_relaxed[i] = optimize.newton(self.f2, T, args=(P,i))
		# return p_relaxed

	def get_normal_pressure(self, Y_1, rho, e_0):
		# rho_1 = 1/self.nu_1(p,T)
		# rho_2 = 1/self.nu_2(p,T)
		# alpha_1 = Y_1*rho/rho_1
		# alpha_2 = 1 - alpha_1
		# p = (rho*e_0 - (alpha_1 * self.gamma_1*self.Pi_1/(self.gamma_1 - 1) + alpha_2*self.gamma_2*self.Pi_2/(self.gamma_2 - 1) - alpha_1*rho_1*self.q_2 - alpha_2*rho_2*self.q_2 ) ) / (alpha_1/(self.gamma_1 - 1) + alpha_2/(self.gamma_2 - 1))

		# q_bar = Y_1*self.q_1 + (1-Y_1)*self.q_2
		# Q = (e_0 - q_bar) / (nu_0 - b)
		# a_1 = Y_1*self.C_v_1*(self.Pi_2 + self.gamma_1*self.Pi_1 - (self.gamma_1 - 1)*Q)
		# a_2 = Y_1*self.q_1 + (1-Y_1)*self.q_2
		# p = (-a_1 + np.sqrt(a_1**2 - 4*a_0*a_2)) / (2*a_2)

		q = Y_1*self.q_1 + (1-Y_1)*self.q_2
		A_1 = Y_1 * (self.gamma_1 - 1)*self.C_v_1/(Y_1*self.C_v_1 + (1-Y_1)*self.C_v_2)*(rho*(e_0-q)-self.Pi_1)
		A_2 = (1-Y_1) * (self.gamma_2 - 1)*self.C_v_2/(Y_1*self.C_v_1 + (1-Y_1)*self.C_v_2)*(rho*(e_0-q)-self.Pi_2)
		p = 1/2 * (A_1 + A_2 - (self.Pi_1 + self.Pi_2)) + np.sqrt(1/4 * (A_2 - A_1 - (self.Pi_2 - self.Pi_1))**2 + A_1*A_2)
		return p

	def get_primitive_vars(self, u_c):
		self.e_0 = u_c[:,2]/u_c[:,0] - 1/2*(u_c[:,1]/u_c[:,0])**2
		self.nu_0 = 1/u_c[:,0]

		# we have specific energy, e_0, and specific volume, nu_0, perform phase change solver
		rho = u_c[:,0]
		v = u_c[:,1]/u_c[:,0]
		Y_1 = u_c[:,3]/u_c[:,0]
		p = self.get_normal_pressure(Y_1, rho, self.e_0)

		# u_p = np.array([rho, v, p, Y_1]).T
		# T = self.get_temperature_from_primitive(u_p)
		# print(Y_1*self.e_1(p,T) + (1-Y_1)*self.e_2(p,T))
		# print(self.e_0)

		# pressure_bound = 12*10**5

		# for i in range(len(p)):
		# 	if p[i] < pressure_bound:



		
		# p_relevant = p[(p < pressure_bound)] # eliminate any cells with a pressure outside of saturation curve bounds
		# nu_0_relevant = self.nu_0[(p < pressure_bound)]
		# e_0_relevant = self.e_0[(p < pressure_bound)]
		# Y_1_relevant = Y_1[(p < pressure_bound)]
		# T_relevant = T[(p < pressure_bound)]

		# T_sat = self.get_saturation_temperature(p_relevant, np.ones(len(p_relevant)))

		# nu_1_sat = self.nu_1(p_relevant, T_sat)
		# nu_2_sat = self.nu_2(p_relevant, T_sat)

		# condition = (nu_1_sat < nu_0_relevant) & (nu_0_relevant < nu_2_sat)
		# # get array of truthies and send to phase change solver
		# p_satisfied = p_relevant[condition]
		# nu_0_satisfied = nu_0_relevant[condition]
		# e_0_satisfied = e_0_relevant[condition]
		# Y_satisfied = Y_1_relevant[condition]
		# T_satisfied = T_relevant[condition]
		# # print(p_satisfied)
		# # self.T = self.get_saturation_temperature(p_satisfied, T_satisfied)
		# p_relaxed = self.solve_phase_change(p_satisfied, e_0_satisfied, nu_0_satisfied, Y_satisfied, T_satisfied)
		# T_relaxed = self.get_saturation_temperature(p_relaxed, np.ones(len(p_relaxed)))
		# Y_1_relaxed = ( (e_0_satisfied - self.e_2(p_relaxed,T_relaxed)) / (self.e_1(p_relaxed,T_relaxed)-self.e_2(p_relaxed,T_relaxed) ) )

		# # I think that we need to set this to 0 or 1 if not satisfied and update props from there.
		# Y_1_relaxed_final = np.where((Y_1_relaxed >= 0) & (Y_1_relaxed <= 1), Y_1_relaxed, Y_1[(p < pressure_bound)][(nu_1_sat < self.nu_0[(p < pressure_bound)]) & (self.nu_0[(p < pressure_bound)] < nu_2_sat)])
		# # Y_1_relaxed_final = np.where((Y_1_relaxed >= 0) & (Y_1_relaxed <= 1), Y_1_relaxed, (np.where(Y_1_relaxed < 0, self.zero_equivalent, self.one_equivalent)))
		# # print(Y_1_relaxed_final)
		# # print(Y_1_relaxed)
		# # print(np.where(Y_1_relaxed < 0, self.zero_equivalent, self.one_equivalent))
		# # print(len(1/nu_0_satisfied[(Y_1_relaxed < 0) | (Y_1_relaxed > 1)]))
		# # print(self.get_normal_pressure(Y_1_relaxed[(Y_1_relaxed < 0) | (Y_1_relaxed > 1)], 1/nu_0_satisfied[(Y_1_relaxed < 0) | (Y_1_relaxed > 1)]))

		# # p_relaxed_final = np.where((Y_1_relaxed >= 0) & (Y_1_relaxed <= 1), p_relaxed, self.get_normal_pressure(np.where(Y_1_relaxed < 0, self.zero_equivalent, self.one_equivalent), 1/nu_0_satisfied[(Y_1_relaxed < 0) | (Y_1_relaxed > 1)]))
		# p_relaxed_final = np.where((Y_1_relaxed >= 0) & (Y_1_relaxed <= 1), p_relaxed, p[(p < pressure_bound)][(nu_1_sat < self.nu_0[(p < pressure_bound)]) & (self.nu_0[(p < pressure_bound)] < nu_2_sat)])
		# p[(p < pressure_bound)][(nu_1_sat < self.nu_0[(p < pressure_bound)]) & (self.nu_0[(p < pressure_bound)] < nu_2_sat)] = p_relaxed_final
		# Y_1[(p < pressure_bound)][(nu_1_sat < self.nu_0[(p < pressure_bound)]) & (self.nu_0[(p < pressure_bound)] < nu_2_sat)] = Y_1_relaxed_final

		return np.array([rho, v, p, Y_1]).T

	def get_conservative_vars(self, u_p):
		e = self.get_specific_energy(u_p)
		return np.array([
			u_p[:,0], 
			u_p[:,0]*u_p[:,1], 
			u_p[:,0]*(e+1/2*u_p[:,1]**2),
			u_p[:,0]*u_p[:,3]]).T

	def weno_left(self, u):
		u = self.pad(u, self.r, self.r-1)
		u_p_reconstructed = np.zeros(u.shape) 
		P = np.zeros(np.append((self.r+1), u.shape))
		beta = calculate_beta_no_characteristic(u, self.r, shift)

		alpha = self.b/(beta+self.epsilon)**(self.r+1)
		omega = alpha / alpha.sum(axis=0)

		for k_s in range(self.r+1): # for each stencil
			# calculate half point polynomial interpolation, P_{r,k_s,i+1/2}		
			for l in range(self.r+1):
				P[k_s,:,:] += self.a[k_s,l]*shift(u, k_s-l).reshape((len(u), np.shape(u)[1], 1)).reshape((len(u), np.shape(u)[1]))	
			u_p_reconstructed = u_p_reconstructed + omega[k_s]*P[k_s]

		return u_p_reconstructed[self.r:self.k+self.r-1]

	def weno_right(self, u):
		u = np.flip(self.pad(u, self.r-1, self.r), axis=0)
		u_p_reconstructed = np.zeros(u.shape) 
		P = np.zeros(np.append((self.r+1), u.shape))
		
		beta = calculate_beta_no_characteristic(u, self.r, shift)
		alpha = self.b/(beta+self.epsilon)**(self.r+1)
		omega = alpha / alpha.sum(axis=0)

		for k_s in range(self.r+1):
			for l in range(self.r+1):
				P[k_s,:,:] += self.a[k_s,l]*shift(u, k_s-l).reshape((len(u), np.shape(u)[1], 1)).reshape((len(u), np.shape(u)[1]))
			u_p_reconstructed = u_p_reconstructed + omega[k_s]*P[k_s]
		return np.flip(u_p_reconstructed, axis=0)[self.r:self.k+self.r-1]

	def weno_characteristic_left(self):
		u = self.pad(self.u_p, self.r, self.r-1)
		u_p_reconstructed = np.zeros(u.shape) 
		P = np.zeros(np.append((self.r+1), u.shape))
		Q, Q_inverse = self.get_characteristic_transform(u)

		beta = calculate_beta_characteristic(u, self.r, Q, Q_inverse, shift)
		alpha = self.b/(beta+self.epsilon)**(self.r+1)
		omega = alpha / alpha.sum(axis=0)

		for k_s in range(self.r+1): # for each stencil
			# calculate half point polynomial interpolation, P_{r,k_s,i+1/2}		
			for l in range(self.r+1):
				P[k_s,:,:] += self.a[k_s,l]*np.matmul(Q_inverse, shift(u, k_s-l).reshape((len(u), np.shape(u)[1], 1))).reshape((len(u), np.shape(u)[1]))	
			u_p_reconstructed = u_p_reconstructed + omega[k_s]*P[k_s]
		return (np.matmul(Q, u_p_reconstructed.reshape(len(u),np.shape(u)[1],1))).reshape((len(u), np.shape(u)[1]))[self.r:self.k+self.r-1]

	def weno_characteristic_right(self):
		# u = np.flip(np.pad(self.u_p, ((self.r-1, self.r), (0,0)), mode=self.boundary_type), axis=0)
		u = np.flip(self.pad(self.u_p, self.r-1, self.r), axis=0)
		u_p_reconstructed = np.zeros(u.shape) 
		P = np.zeros(np.append((self.r+1), u.shape))
		Q, Q_inverse = self.get_characteristic_transform(u)	
		
		beta = calculate_beta_characteristic(u, self.r, Q, Q_inverse, shift)
		alpha = self.b/(beta+self.epsilon)**(self.r+1)
		omega = alpha / alpha.sum(axis=0)

		for k_s in range(self.r+1):
			for l in range(self.r+1):
				P[k_s,:,:] += self.a[k_s,l]*np.matmul(Q_inverse, shift(u, k_s-l).reshape((len(u), np.shape(u)[1], 1))).reshape((len(u), np.shape(u)[1]))
			u_p_reconstructed = u_p_reconstructed + omega[k_s]*P[k_s]
		return np.flip((np.matmul(Q, u_p_reconstructed.reshape(len(u),np.shape(u)[1],1))).reshape((len(u),np.shape(u)[1])), axis=0)[self.r:self.k+self.r-1]

	def flux_split(self):
		max_characteristic = self.get_maximum_characteristic(self.u_p)	
		if self.characteristic:
			u_p_reconstructed_l = self.weno_characteristic_left()
			u_p_reconstructed_r = self.weno_characteristic_right()
		else:
			u_p_reconstructed_l = self.weno_left(self.u_p)
			u_p_reconstructed_r = self.weno_right(self.u_p)

		u_c_reconstructed_l = self.get_conservative_vars(u_p_reconstructed_l)
		u_c_reconstructed_r = self.get_conservative_vars(u_p_reconstructed_r)

		flux_left = self.get_flux(u_p_reconstructed_l) + max_characteristic*u_c_reconstructed_l
		flux_right = self.get_flux(u_p_reconstructed_r) - max_characteristic*u_c_reconstructed_r
		
		return 1/2*(flux_left + flux_right)

	def get_dudx(self):
		u_split = np.pad(self.flux_split(), ((self.r+1, self.r+1), (0,0)), 'reflect', reflect_type='odd')
	
		dudx = np.zeros(np.shape(u_split))
	
		for i in range(len(self.d)):
			dudx += self.d[i]*(shift(u_split, -i)-shift(u_split, i+1))
		# plt.plot(self.x, (dudx/self.h)[1:self.k + 1, :])
		# plt.show()
		# sys.exit()
		return (dudx/self.h)[self.r+1:self.k+self.r+1, :]
	
	# def calculate_entropy(self):
	# 	e=(self.u_p[:,2]+self.gamma*self.Pi)/(self.u_p[:,0]*(self.gamma-1))
	# 	self.entropy = np.log(e)-(self.gamma-1)*np.log(self.u_p[:,0])

	def rk4(self):
		self.tau = self.CFL * self.h / self.get_maximum_characteristic(self.u_p) 
		u_c = self.u_c
		k1 = -self.get_dudx() - self.get_heat_flux(self.u_p)
		self.u_c = u_c + self.tau*1/2*k1		
		self.u_p = self.get_primitive_vars(self.u_c)		
		
		# plt.plot(self.x, self.u_p[:,0])
		# plt.plot(self.x, self.u_p[:,1])
		plt.plot(self.x, self.u_p[:,2])
		# plt.plot(self.x, self.u_p[:,3])
		plt.show()
		sys.exit()

		k2 = -self.get_dudx() - self.get_heat_flux(self.u_p)
		self.u_c = u_c + self.tau*1/2*k2
		self.u_p = self.get_primitive_vars(self.u_c)

		k3 = -self.get_dudx() - self.get_heat_flux(self.u_p)
		self.u_c = u_c + self.tau*k3

		self.u_p = self.get_primitive_vars(self.u_c)

		k4 = -self.get_dudx() - self.get_heat_flux(self.u_p)
		self.u_c = u_c + self.tau*(1/6*k1+1/3*k2+1/3*k3+1/6*k4)
		self.u_p = self.get_primitive_vars(self.u_c)		

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

	def run(self):
		# plt.ion()
		# fig = plt.figure(1)
		# ax = fig.add_subplot(111)
		# line1, = ax.plot(self.x,self.u_p[:,0],'r-')
		# line2, = ax.plot(self.x,self.u_p[:,1],'b-')
		# line3, = ax.plot(self.x,self.u_p[:,2],'g-')
		while self.t < self.t_f:
			if self.time_int == 'rk4':
				self.rk4()
			elif self.time_int == 'sdc': 
				self.sdc()
			self.t = self.t + self.tau
			print(self.t)
			# line1.set_ydata(self.u_p[:,0])
			# line2.set_ydata(self.u_p[:,1])
			# line3.set_ydata(self.u_p[:,2])
			# fig.canvas.draw()
			# fig.canvas.flush_events()
		# self.calculate_entropy()
		self.elapsed_time = time.clock() - self.start_time
		print('done, time: ', self.elapsed_time)		

x0 = -0.05
xf = 0.05
tf = 0.0001
N = 200
CFL = 0.1
characteristic = False

rk4 = euler(x0, xf, tf, N, CFL, 0, None, characteristic, 'edge', 'rk4')
plt.plot(rk4.x, rk4.u_p[:,2], marker='.')
# plt.yscale("log")
# plt.show()
# plt.plot(rk4.x, rk4.u_p[:,1], marker='.')
# plt.show()
# plt.plot(rk4.x, rk4.u_p[:,2], marker='.')
# plt.show()
# plt.plot(rk4.x, rk4.u_p[:,3], marker='.')
plt.show()
