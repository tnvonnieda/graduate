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

		# self.Pi_1 = 10**9
		# self.C_v_1 = 1816
		# self.C_p_1 = 4267
		# self.gamma_1 = self.C_p_1/self.C_v_1
		# self.q_1 = -1167*10**3
		# self.q_1_prime = 0 # J/(kg K)
		# self.lambda_1 = 0.6788 # w/(m K)

		# # water vapor
		# self.Pi_2 = 0 # Pa  
		# self.C_v_2 = 1040
		# self.C_p_2 = 1487
		# self.gamma_2 = self.C_p_2 / self.C_v_2
		# self.q_2 = 2030*10**3
		# self.q_2_prime = -23.26*10**3 # J/(kg K)
		# self.lambda_2 = 249.97 # w/(m K)

		self.Pi_1 = 9058.29*10**5
		self.C_v_1 = 1606.97
		self.gamma_1 = 2.62
		self.C_p_1 = self.C_v_1*self.gamma_1		
		self.q_1 = -1.150975*10**6
		self.q_1_prime = 0 # J/(kg K)
		self.lambda_1 = 0.6788 # w/(m K)

		# water vapor
		self.Pi_2 = 0 # Pa  
		self.C_v_2 = 1192.51
		self.gamma_2 = 1.38
		self.C_p_2 = self.C_v_2*self.gamma_2
		self.q_2 = 2.060759*10**6
		self.q_2_prime = -27.2386*10**3 # J/(kg K)
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
		# print(Q_0)
		# T_left = T_LI + (T_sat - T_LI)*np.exp(m_dot*self.C_p_2*x/self.lambda_2) # temperature left of interface
		# T_right = self.C_p_1/self.C_p_2 * T_LI + (self.q_1 - self.q_2)/self.C_p_2 + (T_sat - self.C_p_1/self.C_p_2 * T_LI - (self.q_1 - self.q_2)/self.C_p_2)*np.exp(m_dot*self.C_p_2*x/self.lambda_2)
		p = np.piecewise(self.x, conds, [p, p])
		T = np.piecewise(self.x, conds, [lambda x: T_LI + (T_sat - T_LI)*np.exp(m_dot*self.C_p_2*x/self.lambda_2), lambda x: self.C_p_1/self.C_p_2 * T_LI + (self.q_1 - self.q_2)/self.C_p_2 + (T_sat - self.C_p_1/self.C_p_2 * T_LI - (self.q_1 - self.q_2)/self.C_p_2)*np.exp(m_dot*self.C_p_2*x/self.lambda_2)])
		
		alpha_1 = np.piecewise(self.x, conds, [0.999999, 0.000001])
		# print(alpha_1[100])
		alpha_2 = 1 - alpha_1

		rho = 1/self.nu_1(p,T)*alpha_1 + 1/self.nu_2(p,T)*alpha_2

		Y_1 = alpha_1 * 1/self.nu_1(p,T) / rho
		
		v = m_dot*1/rho

		self.u_p = np.array([
			rho, # kg/m^3
			v, # m/s
			p, # kPa
			Y_1 # left side liquid, right side vapor
			]).T
		
		self.boundary_left = np.array([[rho[0], v[0], p[0], Y_1[0]]])
		self.boundary_right = np.array([[rho[-1], v[-1], p[-1], Y_1[-1]]])
		# print(self.boundary_right)
		
		self.T = self.get_temperature_from_primitive(self.u_p)
		self.e_0 = self.u_p[:,3]*self.e_1(self.u_p[:,2],self.T) + (1-self.u_p[:,3])*self.e_2(self.u_p[:,2],self.T)

		
		self.nu_0 = self.u_p[:,3]*self.nu_1(self.u_p[:,2],self.T) + (1-self.u_p[:,3])*self.nu_2(self.u_p[:,2],self.T)
		
		self.u_c = self.get_conservative_vars(self.u_p)
		self.u_p0 = self.u_p
		self.u_c0 = self.u_c

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

	# ---------------- EQUATION OF STATE ---------------- #
	def h_1(self, T):
		return self.gamma_1*self.C_v_1*T + self.q_1

	def h_2(self, T):
		return self.gamma_2*self.C_v_2*T + self.q_2

	def g_1(self, p, T):
		return (self.gamma_1*self.C_v_1 - self.q_1_prime)*T - self.C_v_1*T*np.log(T**self.gamma_1/((p+self.Pi_1)**(self.gamma_1-1)))+self.q_1

	def g_2(self, p, T):
		return (self.gamma_2*self.C_v_2 - self.q_2_prime)*T - self.C_v_2*T*np.log(T**self.gamma_2/((p+self.Pi_2)**(self.gamma_2-1)))+self.q_2

	def e_1(self, p, T):
		return (p+self.gamma_1*self.Pi_1)/(p+self.Pi_1)*self.C_v_1*T + self.q_1

	def e_2(self, p, T):
		return (p+self.gamma_2*self.Pi_2)/(p+self.Pi_2)*self.C_v_2*T + self.q_2

	def nu_1(self, p, T):
		return (self.gamma_1 - 1)*self.C_v_1*T/(p+self.Pi_1)

	def nu_2(self, p, T):
		return (self.gamma_2 - 1)*self.C_v_2*T/(p+self.Pi_2)

	# --------------------------------------------------- #

	# ---------------- SATURATION CURVE ---------------- #
	def f1(self, T, p):
		# print(T)
		return self.g_1(p,T) - self.g_2(p,T)
	
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
		T = self.get_temperature_from_primitive(u_p)
		# print(T)
		# print((shift(T, -1) - shift(T, 1)))
		dTdx = (shift(T, -1) - shift(T, 1))/(2*self.h)
		# print(np.shape(T))
		plt.plot(self.x, T[2:-2], marker='.')
		plt.plot(self.x, dTdx[2:-2], marker='.')
		plt.show()
		# sys.exit()
		# print(dTdx)
		# dTdx = self.get_dudx_centered(T)
		# dTdx = 
		alpha_1 = u_p[:,3]*u_p[:,0]*self.nu_1(u_p[:,2],T)

		alpha_2 = 1 - alpha_1
		# alpha_12 = (u_p[:,0]*u_p[:,3]) / ((u_p[:,2] + self.gamma_1*self.Pi_1) / ((self.gamma_1 - 1)*self.C_v_1*T.T))
		# print(alpha_12)
		# print(alpha_1)
		# print(alpha_1, alpha_12)
		# alpha_2 = (u_p[:,0]*(1-u_p[:,3])) / ((u_p[:,2] + self.gamma_2*self.Pi_2) / ((self.gamma_2 - 1)*self.C_v_2*T.T))
		Lambda = alpha_1*self.lambda_1 + alpha_2*self.lambda_2
		
		q_i_plus_one_half = -Lambda*shift(Lambda, -1)/(Lambda + shift(Lambda, -1))*(shift(T,-1)-T)/(1/2*self.h) 
		q_i_minus_one_half = -shift(Lambda, 1)*Lambda/(shift(Lambda, 1)+Lambda)*(T-shift(T, 1))/(1/2*self.h)
		
		# print(dqdx2)
		# sys.exit()
		q = - Lambda*dTdx
		# dqdx = np.zeros(np.shape(u_p))
		# print(((shift(q, -1) + shift(q,1))/(2*self.h))[2:-2])
		# sys.exit()
		dqdx = np.zeros_like(u_p)
		# dqdx[:,2] = ((shift(q, -1) - shift(q,1))/(2*self.h))
		dqdx[:,2] = 1/self.h*(q_i_plus_one_half - q_i_minus_one_half)
		# print(dqdx[110,2])
		sys.exit()
		return dqdx[2:-2, :]
		# return ((shift(q, -1) - shift(q,1))/(2*self.h))[1:-1]
		# return dqdx
	# -------------------------------------- #

	def get_temperature_from_primitive(self, u_p):
		T_reciprocal = u_p[:,3]*(self.gamma_1 - 1)*self.C_v_1/(1/u_p[:,0]*(u_p[:,2] + self.Pi_1)) + (1-u_p[:,3])*(self.gamma_2 - 1)*self.C_v_2/(1/u_p[:,0]*(u_p[:,2] + self.Pi_2))
		return 1/T_reciprocal

	def speed_of_sound(self, u_p):
		T = self.get_temperature_from_primitive(u_p)
		# rho_1 = (u_p[:,2] + self.Pi_1)/((self.gamma_1 - 1)*self.C_v_1*T)
		rho_1 = 1/self.nu_1(u_p[:,2], T)
		# rho_2 = (u_p[:,2] + self.Pi_2)/((self.gamma_2 - 1)*self.C_v_2*T)
		rho_2 = 1/self.nu_2(u_p[:,2], T)
		c_1_squared = (self.gamma_1*u_p[:,2]+self.Pi_1)/rho_1

		c_2_squared = (self.gamma_2*u_p[:,2]+self.Pi_2)/rho_2
		
		alpha_1 = u_p[:,0]*u_p[:,3]/rho_1
		alpha_2 = 1 - alpha_1

		c = np.sqrt(1/u_p[:,0] * 1/(alpha_1/(rho_1*c_1_squared) + alpha_2/(rho_2*c_2_squared)))
		# print(c)
		# sys.exit()
		return c

	def get_maximum_characteristic(self, u_p):	
		return np.max(self.speed_of_sound(u_p) + abs(u_p[:,1]))

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

	# def solve_phase_change(self, p, e_0, nu_0, Y_1, T_): # 
	# 	# T = np.ones(len(p))
	# 	p_relaxed = np.empty_like(p)
	# 	for i in range(len(p)):
	# 		e_0_cell = e_0[i]
	# 		nu_0_cell = nu_0[i]
	# 		# print("p: ", p[i])
	# 		# print("T: ", T_[i])
	# 		# print(Y_1[i])
	# 		# print(e_0_cell)
	# 		# print(nu_0_cell)
	# 		p_result = optimize.newton(self.f3, 1, args=(e_0_cell, nu_0_cell), tol=10**-7, full_output=True)
	# 		# print(p_result[1])
	# 		if p_result[1].converged == False:
	# 			print("Non-converging solution:")
	# 			print("p: ", p[i])
	# 			print("T: ", T_[i])
	# 			print("Y_1: ", Y_1[i])
	# 			sys.exit()
	# 		p_relaxed[i] = p_result[1].root
	# 	# p_relaxed = optimize.newton(self.f2, np.ones(len(p)), args=(e_0, nu_0,), full_output=True)
	# 	return p_relaxed
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
		q = Y_1*self.q_1 + (1-Y_1)*self.q_2
		A_1 = Y_1 * (self.gamma_1 - 1)*self.C_v_1/(Y_1*self.C_v_1 + (1-Y_1)*self.C_v_2)*(rho*(e_0-q)-self.Pi_1)
		A_2 = (1-Y_1) * (self.gamma_2 - 1)*self.C_v_2/(Y_1*self.C_v_1 + (1-Y_1)*self.C_v_2)*(rho*(e_0-q)-self.Pi_2)
		p = 1/2 * (A_1 + A_2 - (self.Pi_1 + self.Pi_2)) + np.sqrt(1/4 * (A_2 - A_1 - (self.Pi_2 - self.Pi_1))**2 + A_1*A_2)

		# inner = np.array(self.C_v_1**2.0*self.Pi_1**2.0*self.gamma_1**2.0*Y_1**2.0 - 2.0*self.C_v_1**2.0*self.Pi_1*self.Pi_2*self.gamma_1*Y_1**2.0 - 2.0*self.C_v_1**2.0*self.Pi_1*e_0*self.gamma_1**2.0*rho*Y_1**2.0 + 2.0*self.C_v_1**2.0*self.Pi_1*e_0*self.gamma_1*rho*Y_1**2.0 + 2.0*self.C_v_1**2.0*self.Pi_1*self.gamma_1**2.0*self.q_1*rho*Y_1**3 + 2.0*self.C_v_1**2.0*self.Pi_1*self.gamma_1**2.0*self.q_2*rho*Y_1**2.0*(1.0-Y_1) - 2.0*self.C_v_1**2.0*self.Pi_1*self.gamma_1*self.q_1*rho*Y_1**3 - 2.0*self.C_v_1**2.0*self.Pi_1*self.gamma_1*self.q_2*rho*Y_1**2.0*(1.0-Y_1) + self.C_v_1**2.0*self.Pi_2**2.0*Y_1**2.0 + 2.0*self.C_v_1**2.0*self.Pi_2*e_0*self.gamma_1*rho*Y_1**2.0 - 2.0*self.C_v_1**2.0*self.Pi_2*e_0*rho*Y_1**2.0 - 2.0*self.C_v_1**2.0*self.Pi_2*self.gamma_1*self.q_1*rho*Y_1**3 - 2.0*self.C_v_1**2.0*self.Pi_2*self.gamma_1*self.q_2*rho*Y_1**2.0*(1.0-Y_1) + 2.0*self.C_v_1**2.0*self.Pi_2*self.q_1*rho*Y_1**3 + 2.0*self.C_v_1**2.0*self.Pi_2*self.q_2*rho*Y_1**2.0*(1.0-Y_1) + self.C_v_1**2.0*e_0**2.0*self.gamma_1**2.0*rho**2.0*Y_1**2.0 - 2.0*self.C_v_1**2.0*e_0**2.0*self.gamma_1*rho**2.0*Y_1**2.0 + self.C_v_1**2.0*e_0**2.0*rho**2.0*Y_1**2.0 - 2.0*self.C_v_1**2.0*e_0*self.gamma_1**2.0*self.q_1*rho**2.0*Y_1**3 - 2.0*self.C_v_1**2.0*e_0*self.gamma_1**2.0*self.q_2*rho**2.0*Y_1**2.0*(1.0-Y_1) + 4*self.C_v_1**2.0*e_0*self.gamma_1*self.q_1*rho**2.0*Y_1**3 + 4*self.C_v_1**2.0*e_0*self.gamma_1*self.q_2*rho**2.0*Y_1**2.0*(1.0-Y_1) - 2.0*self.C_v_1**2.0*e_0*self.q_1*rho**2.0*Y_1**3 - 2.0*self.C_v_1**2.0*e_0*self.q_2*rho**2.0*Y_1**2.0*(1.0-Y_1) + self.C_v_1**2.0*self.gamma_1**2.0*self.q_1**2.0*rho**2.0*Y_1**4 + 2.0*self.C_v_1**2.0*self.gamma_1**2.0*self.q_1*self.q_2*rho**2.0*Y_1**3*(1.0-Y_1) + self.C_v_1**2.0*self.gamma_1**2.0*self.q_2**2.0*rho**2.0*Y_1**2.0*(1.0-Y_1)**2.0 - 2.0*self.C_v_1**2.0*self.gamma_1*self.q_1**2.0*rho**2.0*Y_1**4 - 4*self.C_v_1**2.0*self.gamma_1*self.q_1*self.q_2*rho**2.0*Y_1**3*(1.0-Y_1) - 2.0*self.C_v_1**2.0*self.gamma_1*self.q_2**2.0*rho**2.0*Y_1**2.0*(1.0-Y_1)**2.0 + self.C_v_1**2.0*self.q_1**2.0*rho**2.0*Y_1**4 + 2.0*self.C_v_1**2.0*self.q_1*self.q_2*rho**2.0*Y_1**3*(1.0-Y_1) + self.C_v_1**2.0*self.q_2**2.0*rho**2.0*Y_1**2.0*(1.0-Y_1)**2.0 + 2.0*self.C_v_1*self.C_v_2*self.Pi_1**2.0*self.gamma_1*Y_1*(1.0-Y_1) + 2.0*self.C_v_1*self.C_v_2*self.Pi_1*self.Pi_2*self.gamma_1*self.gamma_2*Y_1*(1.0-Y_1) - 4*self.C_v_1*self.C_v_2*self.Pi_1*self.Pi_2*self.gamma_1*Y_1*(1.0-Y_1) - 4*self.C_v_1*self.C_v_2*self.Pi_1*self.Pi_2*self.gamma_2*Y_1*(1.0-Y_1) + 2.0*self.C_v_1*self.C_v_2*self.Pi_1*self.Pi_2*Y_1*(1.0-Y_1) - 2.0*self.C_v_1*self.C_v_2*self.Pi_1*e_0*self.gamma_1*self.gamma_2*rho*Y_1*(1.0-Y_1) + 4*self.C_v_1*self.C_v_2*self.Pi_1*e_0*self.gamma_2*rho*Y_1*(1.0-Y_1) - 2.0*self.C_v_1*self.C_v_2*self.Pi_1*e_0*rho*Y_1*(1.0-Y_1) + 2.0*self.C_v_1*self.C_v_2*self.Pi_1*self.gamma_1*self.gamma_2*self.q_1*rho*Y_1**2.0*(1.0-Y_1) + 2.0*self.C_v_1*self.C_v_2*self.Pi_1*self.gamma_1*self.gamma_2*self.q_2*rho*Y_1*(1.0-Y_1)**2.0 - 4*self.C_v_1*self.C_v_2*self.Pi_1*self.gamma_2*self.q_1*rho*Y_1**2.0*(1.0-Y_1) - 4*self.C_v_1*self.C_v_2*self.Pi_1*self.gamma_2*self.q_2*rho*Y_1*(1.0-Y_1)**2.0 + 2.0*self.C_v_1*self.C_v_2*self.Pi_1*self.q_1*rho*Y_1**2.0*(1.0-Y_1) + 2.0*self.C_v_1*self.C_v_2*self.Pi_1*self.q_2*rho*Y_1*(1.0-Y_1)**2.0 + 2.0*self.C_v_1*self.C_v_2*self.Pi_2**2.0*self.gamma_2*Y_1*(1.0-Y_1) - 2.0*self.C_v_1*self.C_v_2*self.Pi_2*e_0*self.gamma_1*self.gamma_2*rho*Y_1*(1.0-Y_1) + 4*self.C_v_1*self.C_v_2*self.Pi_2*e_0*self.gamma_1*rho*Y_1*(1.0-Y_1) - 2.0*self.C_v_1*self.C_v_2*self.Pi_2*e_0*rho*Y_1*(1.0-Y_1) + 2.0*self.C_v_1*self.C_v_2*self.Pi_2*self.gamma_1*self.gamma_2*self.q_1*rho*Y_1**2.0*(1.0-Y_1) + 2.0*self.C_v_1*self.C_v_2*self.Pi_2*self.gamma_1*self.gamma_2*self.q_2*rho*Y_1*(1.0-Y_1)**2.0 - 4*self.C_v_1*self.C_v_2*self.Pi_2*self.gamma_1*self.q_1*rho*Y_1**2.0*(1.0-Y_1) - 4*self.C_v_1*self.C_v_2*self.Pi_2*self.gamma_1*self.q_2*rho*Y_1*(1.0-Y_1)**2.0 + 2.0*self.C_v_1*self.C_v_2*self.Pi_2*self.q_1*rho*Y_1**2.0*(1.0-Y_1) + 2.0*self.C_v_1*self.C_v_2*self.Pi_2*self.q_2*rho*Y_1*(1.0-Y_1)**2.0 + 2.0*self.C_v_1*self.C_v_2*e_0**2.0*self.gamma_1*self.gamma_2*rho**2.0*Y_1*(1.0-Y_1) - 2.0*self.C_v_1*self.C_v_2*e_0**2.0*self.gamma_1*rho**2.0*Y_1*(1.0-Y_1) - 2.0*self.C_v_1*self.C_v_2*e_0**2.0*self.gamma_2*rho**2.0*Y_1*(1.0-Y_1) + 2.0*self.C_v_1*self.C_v_2*e_0**2.0*rho**2.0*Y_1*(1.0-Y_1) - 4*self.C_v_1*self.C_v_2*e_0*self.gamma_1*self.gamma_2*self.q_1*rho**2.0*Y_1**2.0*(1.0-Y_1) - 4*self.C_v_1*self.C_v_2*e_0*self.gamma_1*self.gamma_2*self.q_2*rho**2.0*Y_1*(1.0-Y_1)**2.0 + 4*self.C_v_1*self.C_v_2*e_0*self.gamma_1*self.q_1*rho**2.0*Y_1**2.0*(1.0-Y_1) + 4*self.C_v_1*self.C_v_2*e_0*self.gamma_1*self.q_2*rho**2.0*Y_1*(1.0-Y_1)**2.0 + 4*self.C_v_1*self.C_v_2*e_0*self.gamma_2*self.q_1*rho**2.0*Y_1**2.0*(1.0-Y_1) + 4*self.C_v_1*self.C_v_2*e_0*self.gamma_2*self.q_2*rho**2.0*Y_1*(1.0-Y_1)**2.0 - 4*self.C_v_1*self.C_v_2*e_0*self.q_1*rho**2.0*Y_1**2.0*(1.0-Y_1) - 4*self.C_v_1*self.C_v_2*e_0*self.q_2*rho**2.0*Y_1*(1.0-Y_1)**2.0 + 2.0*self.C_v_1*self.C_v_2*self.gamma_1*self.gamma_2*self.q_1**2.0*rho**2.0*Y_1**3*(1.0-Y_1) + 4*self.C_v_1*self.C_v_2*self.gamma_1*self.gamma_2*self.q_1*self.q_2*rho**2.0*Y_1**2.0*(1.0-Y_1)**2.0 + 2.0*self.C_v_1*self.C_v_2*self.gamma_1*self.gamma_2*self.q_2**2.0*rho**2.0*Y_1*(1.0-Y_1)**3 - 2.0*self.C_v_1*self.C_v_2*self.gamma_1*self.q_1**2.0*rho**2.0*Y_1**3*(1.0-Y_1) - 4*self.C_v_1*self.C_v_2*self.gamma_1*self.q_1*self.q_2*rho**2.0*Y_1**2.0*(1.0-Y_1)**2.0 - 2.0*self.C_v_1*self.C_v_2*self.gamma_1*self.q_2**2.0*rho**2.0*Y_1*(1.0-Y_1)**3 - 2.0*self.C_v_1*self.C_v_2*self.gamma_2*self.q_1**2.0*rho**2.0*Y_1**3*(1.0-Y_1) - 4*self.C_v_1*self.C_v_2*self.gamma_2*self.q_1*self.q_2*rho**2.0*Y_1**2.0*(1.0-Y_1)**2.0 - 2.0*self.C_v_1*self.C_v_2*self.gamma_2*self.q_2**2.0*rho**2.0*Y_1*(1.0-Y_1)**3 + 2.0*self.C_v_1*self.C_v_2*self.q_1**2.0*rho**2.0*Y_1**3*(1.0-Y_1) + 4*self.C_v_1*self.C_v_2*self.q_1*self.q_2*rho**2.0*Y_1**2.0*(1.0-Y_1)**2.0 + 2.0*self.C_v_1*self.C_v_2*self.q_2**2.0*rho**2.0*Y_1*(1.0-Y_1)**3 + self.C_v_2**2.0*self.Pi_1**2.0*(1.0-Y_1)**2.0 - 2.0*self.C_v_2**2.0*self.Pi_1*self.Pi_2*self.gamma_2*(1.0-Y_1)**2.0 + 2.0*self.C_v_2**2.0*self.Pi_1*e_0*self.gamma_2*rho*(1.0-Y_1)**2.0 - 2.0*self.C_v_2**2.0*self.Pi_1*e_0*rho*(1.0-Y_1)**2.0 - 2.0*self.C_v_2**2.0*self.Pi_1*self.gamma_2*self.q_1*rho*Y_1*(1.0-Y_1)**2.0 - 2.0*self.C_v_2**2.0*self.Pi_1*self.gamma_2*self.q_2*rho*(1.0-Y_1)**3 + 2.0*self.C_v_2**2.0*self.Pi_1*self.q_1*rho*Y_1*(1.0-Y_1)**2.0 + 2.0*self.C_v_2**2.0*self.Pi_1*self.q_2*rho*(1.0-Y_1)**3 + self.C_v_2**2.0*self.Pi_2**2.0*self.gamma_2**2.0*(1.0-Y_1)**2.0 - 2.0*self.C_v_2**2.0*self.Pi_2*e_0*self.gamma_2**2.0*rho*(1.0-Y_1)**2.0 + 2.0*self.C_v_2**2.0*self.Pi_2*e_0*self.gamma_2*rho*(1.0-Y_1)**2.0 + 2.0*self.C_v_2**2.0*self.Pi_2*self.gamma_2**2.0*self.q_1*rho*Y_1*(1.0-Y_1)**2.0 + 2.0*self.C_v_2**2.0*self.Pi_2*self.gamma_2**2.0*self.q_2*rho*(1.0-Y_1)**3 - 2.0*self.C_v_2**2.0*self.Pi_2*self.gamma_2*self.q_1*rho*Y_1*(1.0-Y_1)**2.0 - 2.0*self.C_v_2**2.0*self.Pi_2*self.gamma_2*self.q_2*rho*(1.0-Y_1)**3 + self.C_v_2**2.0*e_0**2.0*self.gamma_2**2.0*rho**2.0*(1.0-Y_1)**2.0 - 2.0*self.C_v_2**2.0*e_0**2.0*self.gamma_2*rho**2.0*(1.0-Y_1)**2.0 + self.C_v_2**2.0*e_0**2.0*rho**2.0*(1.0-Y_1)**2.0 - 2.0*self.C_v_2**2.0*e_0*self.gamma_2**2.0*self.q_1*rho**2.0*Y_1*(1.0-Y_1)**2.0 - 2.0*self.C_v_2**2.0*e_0*self.gamma_2**2.0*self.q_2*rho**2.0*(1.0-Y_1)**3 + 4*self.C_v_2**2.0*e_0*self.gamma_2*self.q_1*rho**2.0*Y_1*(1.0-Y_1)**2.0 + 4*self.C_v_2**2.0*e_0*self.gamma_2*self.q_2*rho**2.0*(1.0-Y_1)**3 - 2.0*self.C_v_2**2.0*e_0*self.q_1*rho**2.0*Y_1*(1.0-Y_1)**2.0 - 2.0*self.C_v_2**2.0*e_0*self.q_2*rho**2.0*(1.0-Y_1)**3 + self.C_v_2**2.0*self.gamma_2**2.0*self.q_1**2.0*rho**2.0*Y_1**2.0*(1.0-Y_1)**2.0 + 2.0*self.C_v_2**2.0*self.gamma_2**2.0*self.q_1*self.q_2*rho**2.0*Y_1*(1.0-Y_1)**3 + self.C_v_2**2.0*self.gamma_2**2.0*self.q_2**2.0*rho**2.0*(1.0-Y_1)**4 - 2.0*self.C_v_2**2.0*self.gamma_2*self.q_1**2.0*rho**2.0*Y_1**2.0*(1.0-Y_1)**2.0 - 4*self.C_v_2**2.0*self.gamma_2*self.q_1*self.q_2*rho**2.0*Y_1*(1.0-Y_1)**3 - 2.0*self.C_v_2**2.0*self.gamma_2*self.q_2**2.0*rho**2.0*(1.0-Y_1)**4 + self.C_v_2**2.0*self.q_1**2.0*rho**2.0*Y_1**2.0*(1.0-Y_1)**2.0 + 2.0*self.C_v_2**2.0*self.q_1*self.q_2*rho**2.0*Y_1*(1.0-Y_1)**3 + self.C_v_2**2.0*self.q_2**2.0*rho**2.0*(1.0-Y_1)**4, dtype='float')
		# p = (-self.C_v_1*self.Pi_1*self.gamma_1*Y_1 - self.C_v_1*self.Pi_2*Y_1 + self.C_v_1*e_0*self.gamma_1*rho*Y_1 - self.C_v_1*e_0*rho*Y_1 - self.C_v_1*self.gamma_1*self.q_1*rho*Y_1**2.0 - self.C_v_1*self.gamma_1*self.q_2*rho*Y_1*(1.0-Y_1) + self.C_v_1*self.q_1*rho*Y_1**2.0 + self.C_v_1*self.q_2*rho*Y_1*(1.0-Y_1) - self.C_v_2*self.Pi_1*(1.0-Y_1) - self.C_v_2*self.Pi_2*self.gamma_2*(1.0-Y_1) + self.C_v_2*e_0*self.gamma_2*rho*(1.0-Y_1) - self.C_v_2*e_0*rho*(1.0-Y_1) - self.C_v_2*self.gamma_2*self.q_1*rho*Y_1*(1.0-Y_1) - self.C_v_2*self.gamma_2*self.q_2*rho*(1.0-Y_1)**2.0 + self.C_v_2*self.q_1*rho*Y_1*(1.0-Y_1) + self.C_v_2*self.q_2*rho*(1.0-Y_1)**2.0 + np.sqrt(inner))/(2*self.C_v_1*Y_1 + 2.0*self.C_v_2*(1.0-Y_1))
		
		return p


	def get_primitive_vars(self, u_c):
		# print(u_c[100,2])

		# e_0 = 652450
		# u_c_2 = (e_0+1/2*(u_c[100,1]/u_c[100,0])**2)*u_c[100,0]
		# print(u_c_2)
		# print(u_c_2 - u_c[100,2])
		self.e_0 = u_c[:,2]/u_c[:,0] - 1/2*(u_c[:,1]/u_c[:,0])**2
		# print(self.e_0[100])
		# sys.exit()
		self.nu_0 = 1/u_c[:,0]

		# we have specific energy, e_0, and specific volume, nu_0, perform phase change solver
		rho = u_c[:,0]
		v = u_c[:,1]/u_c[:,0]
		Y_1 = u_c[:,3]/u_c[:,0]
		p = self.get_normal_pressure(Y_1, rho, self.e_0)
		
		u_p = np.array([rho, v, p, Y_1]).T
		for i in range(99, 101):
			T_sat = self.get_saturation_temperature(p[i], 1)
			nu_1_sat = self.nu_1(p[i], T_sat)
			nu_2_sat = self.nu_2(p[i], T_sat)
			if nu_1_sat < self.nu_0[i] and self.nu_0[i] < nu_2_sat:
				e_0_cell = self.e_0[i]
				nu_0_cell = self.nu_0[i]
				print(p[i], Y_1[i], self.get_temperature_from_primitive(u_p)[i])
				p_relaxed = optimize.newton(self.f3, 1, args=(e_0_cell, nu_0_cell), tol=10**-8)
				T_relaxed = self.get_saturation_temperature(p_relaxed, 1)
				Y_1_relaxed = ( (e_0_cell - self.e_2(p_relaxed,T_relaxed)) / (self.e_1(p_relaxed,T_relaxed)-self.e_2(p_relaxed,T_relaxed) ) )
				print(p_relaxed)
				print(T_relaxed)
				print(Y_1_relaxed)
				
				# print(p_relaxed)
				# print(T_relaxed)
				# print(Y_1_relaxed)
				p[i] = p_relaxed
				Y_1[i] = Y_1_relaxed
				# print(p[i] - p_result)
				# print(p_result)
			# else:
			# 	print('no result')
				print("")

		# sys.exit()
		# plt.plot(self.x, p)
		# plt.show()
		# sys.exit()
		# print(Y_1[99])
		# print(p[99])
		
		# u_p = np.array([rho, v, p, Y_1]).T
		# T = self.get_temperature_from_primitive(u_p)
		# plt.plot(self.x, T)
		# plt.show()
		# sys.exit()
		# print(T[99])
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
	
	# def calculate_entropy(self):
	# 	e=(self.u_p[:,2]+self.gamma*self.Pi)/(self.u_p[:,0]*(self.gamma-1))
	# 	self.entropy = np.log(e)-(self.gamma-1)*np.log(self.u_p[:,0])	

	def run(self):
		# plt.ion()
		# fig = plt.figure(1)
		# ax = fig.add_subplot(111)
		# line1, = ax.plot(self.x,self.u_p[:,0],'r-')
		# line2, = ax.plot(self.x,self.u_p[:,1],'b-', marker='.')
		# line3, = ax.plot(self.x,self.u_p[:,2],'g-')
		while self.t < self.t_f:
			self.tau = self.CFL * self.h / self.get_maximum_characteristic(self.u_p)
			u_p_extended = np.vstack((self.boundary_left, self.u_p, self.boundary_right))
			u_c_extended = self.get_conservative_vars(u_p_extended)
			flux = self.get_flux(u_p_extended)
			heat_flux = self.get_heat_flux(np.vstack((self.boundary_left, u_p_extended, self.boundary_right)))
			# print(heat_flux[100,2]*self.tau)
			# print(heat_flux*self.tau)
			# sys.exit()
			self.u_c = (u_c_extended - self.tau / (2*self.h)*(shift(flux, -1) - shift(flux, 1) - self.get_maximum_characteristic(self.u_p)*(shift(u_c_extended, 1) - 2*u_c_extended + shift(u_c_extended, -1))))[1:-1, :] - self.tau*heat_flux
			# self.u_c[:,2] += - heat_flux	

			self.u_p = self.get_primitive_vars(self.u_c)
			print(self.u_p[100,2])
			sys.exit()
			# plt.plot(self.x, self.u_p[:,0], marker='.')
			# plt.plot(self.x, self.u_p[:,1], marker='.')
			# plt.plot(self.x, heat_flux, marker='.')
			# plt.plot(self.x, self.u_p[:,2])
			# plt.show()
			# sys.exit()
			# self.tau = self.CFL * self.h / self.get_maximum_characteristic(self.u_p)
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
CFL = 0.01
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
