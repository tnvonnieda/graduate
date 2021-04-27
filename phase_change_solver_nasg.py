import numpy as np
import sys
from scipy import optimize
import matplotlib.pyplot as plt
import math

class phase_change_solver:
	def __init__(self, P_0, T_0, Y_1_0):
		self.P_0 = P_0
		self.T = T_0
		self.Y_1 = Y_1_0

		self.newton_epsilon = 0.000001

		# ----- MATERIAL PROPERTIES ----- #

		self.Pi_1 = 7028*10**5
		self.C_v_1 = 3610
		self.C_p_1 = 4285
		self.gamma_1 = self.C_p_1/self.C_v_1
		self.b_1 = 6.61*10**-4
		self.q_1 = -1177788
		self.q_1_prime = 0 # J/(kg K)

		# water vapor

		self.Pi_2 = 0 # Pa  
		self.C_v_2 = 955
		self.C_p_2 = 1401
		self.gamma_2 = self.C_p_2 / self.C_v_2
		self.b_2 = 0
		self.q_2 = 2077616
		self.q_2_prime = 14317 # J/(kg K)

		# DODECANE #
		# # Liquid
		# self.Pi_1 = 1159*10**5
		# self.C_v_1 = 2393
		# self.C_p_1 = 2608
		# self.gamma_1 = self.C_p_1/self.C_v_1
		# self.b_1 = 7.51*10**-4
		# self.q_1 = -794696
		# self.q_1_prime = 0 # J/(kg K)

		# # Vapor
		# self.Pi_2 = 0 # Pa  
		# self.C_v_2 = 2016
		# self.C_p_2 = 2063
		# self.gamma_2 = self.C_p_2 / self.C_v_2
		# self.b_2 = 0
		# self.q_2 = -268561
		# self.q_2_prime = 471 # J/(kg K)
		# ----- #

		# -------------------- #

		self.nu_0 = self.Y_1*self.nu_1(self.P_0, self.T) + (1-self.Y_1)*self.nu_2(self.P_0, self.T)
		self.e_0 = self.Y_1*self.e_1(self.P_0, self.T) + (1-self.Y_1)*self.e_2(self.P_0, self.T)

		self.solve_phase_change()

	# ----- STIFFENED GAS EOS (and relevant derivatives, for Newton solver only) ----- #
	def h_1(self, p, T):
		return self.gamma_1*self.C_v_1*T + self.b_1*p + self.q_1

	def h_2(self, p, T):
		return self.gamma_2*self.C_v_2*T + self.b_2*p + self.q_2

	def g_1(self, p, T):
		return (self.gamma_1*self.C_v_1 - self.q_1_prime)*T - self.C_v_1*T*np.log(T**self.gamma_1/((p+self.Pi_1)**(self.gamma_1-1)))+self.b_1*p+self.q_1

	def g_2(self, p, T):
		return (self.gamma_2*self.C_v_2 - self.q_2_prime)*T - self.C_v_2*T*np.log(T**self.gamma_2/((p+self.Pi_2)**(self.gamma_2-1)))+self.b_2*p+self.q_2

	def g_1_prime(self, p, T): # d/dT update needed
		return self.gamma_1*self.C_v_1 - self.q_1_prime - self.C_v_1*(np.log(T**self.gamma_1*(p+self.Pi_1)**(1-self.gamma_1)) + self.gamma_1)

	def g_2_prime(self, p, T): # d/dT
		return self.gamma_2*self.C_v_2 - self.q_2_prime - self.C_v_2*(np.log(T**self.gamma_2*(p+self.Pi_2)**(1-self.gamma_2)) + self.gamma_2)

	# def e_1(self, p, T):
	# 	return (p+self.gamma_1*self.Pi_1)/(self.gamma_1 - 1)*self.nu_1(p, T) + self.q_1
	# def e_1(self, p, T):
	# 	return (p+self.gamma_1*self.Pi_1)*self.C_v_1*T/(p + self.Pi_1) + self.q_1
	
	def e_1(self, p, T):
		return self.C_v_1*T*(1+(self.gamma_1-1)*self.Pi_1/(p+self.Pi_1)) + self.q_1

	def e_2(self, p, T):
		return self.C_v_2*T*(1+(self.gamma_2-1)*self.Pi_2/(p+self.Pi_2)) + self.q_2
	
	def e_1_prime(self, p, T):
		return self.C_v_1*T*self.Pi_1*(1 - self.gamma_1)/(p+self.Pi_1)**2

	# def e_2(self, p, T):
	# 	return (p+self.gamma_2*self.Pi_2)/(self.gamma_2 - 1)*self.nu_2(p, T) + self.q_2

	# def e_2(self, p, T):
	# 	return (p+self.gamma_2*self.Pi_2)*self.C_v_2*T/(p + self.Pi_2) + self.q_2

	def e_2_prime(self, p, T):
		return self.C_v_2*T*self.Pi_2*(1 - self.gamma_2)/(p+self.Pi_2)**2

	def nu_1(self, p, T):
		return (self.gamma_1 - 1)*self.C_v_1*T/(p+self.Pi_1) + self.b_1

	def nu_1_prime(self, P, T):
		return self.C_v_1*(1-self.gamma_1)*T/(P + self.Pi_1)**2

	def nu_2(self, p, T):
		return (self.gamma_2 - 1)*self.C_v_2*T/(p+self.Pi_2) + self.b_2

	def nu_2_prime(self, p, T):
		return self.C_v_2*(1-self.gamma_2)*T/(p + self.Pi_2)**2
	# ----------------------------------- #

	# ----- FUNCTIONS FOR SATURATION CURVE ----- #
	def gibbs_equilibrium(self, T, P):
		return self.g_1(P,T) - self.g_2(P,T)

	def gibbs_equilibrium_prime(self, T, P):
		return self.g_1_prime(P,T) - self.g_2_prime(P,T)

	def get_saturation_temperature(self, P, T): # Gets a saturation temperature for a given pressure. T is an initial guess for the temperature.
		# --- Newton method --- #
		# delta = self.gibbs_equilibrium(T, P) / self.gibbs_equilibrium_prime(T, P)
		# max_iterations = 100
		# iteration = 0
		# while abs(delta) >= self.newton_epsilon and iteration < max_iterations:
		# 	delta = self.gibbs_equilibrium(T, P) / self.gibbs_equilibrium_prime(T, P)
		# 	T = T - delta
		# 	++ iteration
		# --------------------- #
		
		# --- Numpy built in solver --- #
		T = optimize.newton(self.gibbs_equilibrium, T, fprime=self.gibbs_equilibrium_prime, args=(P,)) # Newton method
		# T = optimize.newton(self.gibbs_equilibrium, T, args=(P,)) # Secant method
		# ----------------------------- #

		return T
	# ------------------------------------------ #

	# --- EQUATIONS FOR SOLVING PHASE CHANGE --- #
	def f(self, p):
		self.T = self.get_saturation_temperature(p, self.T)
		return ( (self.e_0 - self.e_2(p,self.T)) / (self.e_1(p,self.T)-self.e_2(p,self.T)) ) - (self.nu_0 - self.nu_2(p,self.T)) / (self.nu_1(p,self.T) - self.nu_2(p,self.T))

	def f_prime(self, p): # For Newton method only
		T = self.T
		return ((self.e_2(p,T) - self.e_0)*self.e_1_prime(p,T) - (self.e_0 - self.e_1(p,T))*self.e_2_prime(p,T)) / (self.e_2(p,T) - self.e_1(p,T))**2 + ((self.nu_2(p,T)-self.nu_0)*self.nu_1_prime(p,T) - (self.nu_0 - self.nu_1(p,T))*self.nu_2_prime(p,T)) / (self.nu_2(p,T) - self.nu_1(p,T))**2
	# ------------------------------------------ #

	def solve_phase_change(self):
		# print(self.get_saturation_temperature(2.8*10**7, 5000))
		T_sat = self.get_saturation_temperature(self.P_0, 1)
		print(T_sat)
		v_liq_sat = self.nu_1(self.P_0, T_sat)
		v_vap_sat = self.nu_2(self.P_0, T_sat)

		print(v_liq_sat, "<", self.nu_0, "<", v_vap_sat)

		if v_liq_sat > self.nu_0 or self.nu_0 > v_vap_sat:
			print("Condition for phase change is not satisfied.")
			sys.exit()

		# --- Newton Method --- # 
		# max_iterations = 100
		# iteration = 0
		# P = self.P_0
		# self.T = self.get_saturation_temperature(P, self.T)
		# delta = self.f(P) / self.f_prime(P)		
		# while abs(delta) >= self.newton_epsilon and iteration < max_iterations:
		# 	self.T = self.get_saturation_temperature(P, self.T)
		# 	delta = self.f(P) / self.f_prime(P)					
		# 	P = P - delta
		# 	++ iteration
		# P_star = P
		# ------------ #

		# --- Numpy built in solver --- # follow this link for more info: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.newton.html
		P_star = optimize.newton(self.f, 1, tol=10**-7)	# Secant method
		# P_star = optimize.newton(self.f, self.P_0, fprime=self.f_prime) # Newton method	
		# ----------------------------- #

		T_star = self.T
		Y_liq = ( (self.e_0 - self.e_2(P_star,T_star)) / (self.e_1(P_star,T_star)-self.e_2(P_star,T_star) ) )
		e_0_f = Y_liq*self.e_1(P_star,T_star) + (1-Y_liq)*self.e_2(P_star,T_star)
		v_0_f = Y_liq*self.nu_1(P_star,T_star) + (1-Y_liq)*self.nu_2(P_star,T_star)
		
		# --- VERIFY RESULTS --- #
		print("Relaxed pressure: ", P_star)
		print("Relaxed temperature: ", T_star)
		print("Relaxed mass fraction, liquid: ", Y_liq)
		print("")
		print("Y_liq equilibrium convergence: ", ( (self.e_0 - self.e_2(P_star,T_star)) / (self.e_1(P_star,T_star)-self.e_2(P_star,T_star)) ) - (self.nu_2(P_star,T_star) - self.nu_0) / (self.nu_2(P_star,T_star) - self.nu_1(P_star,T_star)))
		print("Gibbs equilibrium convergence: ", self.g_2(P_star, T_star) - self.g_1(P_star,T_star))
		print("")
		print("Specific energy condition satisfied: ", self.e_1(P_star, T_star) < e_0_f and e_0_f < self.e_2(P_star, T_star))
		print(self.e_1(P_star, T_star), "<", e_0_f, "<", self.e_2(P_star, T_star))
		print("")
		print("Specific volume condition satisfied: ", self.nu_1(P_star, T_star) < v_0_f and v_0_f < self.nu_2(P_star, T_star))
		print(self.nu_1(P_star, T_star), "<", v_0_f, "<", self.nu_2(P_star, T_star))
		print("")
		print("e_0, before: ", self.e_0)
		print("e_0, after: ", e_0_f)
		print("")
		print("nu_0, before: ", self.nu_0)
		print("nu_0, after: ", v_0_f)
		# --------------------- #

# Y_1_0 = 0.5972429980494464 # initial liquid mass fraction
# P_0 = 131205.5144877433 # initial pressure, Pa
# T_0 = 1269.1613232109564 # initial temperature, K

Y_1_0 = 0.7
P_0 = 100000 
T_0 = 600

# P = 101325
# rho = 2
# Pi = 0
# b = 0
# C_v = 2016
# C_p = 2063

# T = (P+Pi)*(1/rho-b)/((C_p/C_v-1)*C_v)
# print(T)
phase_change_solver(P_0, T_0, Y_1_0)

		


