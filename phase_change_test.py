# # PHASE CHANGE TEST
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import math

# Pi_1 = 7028*10**5
# C_v_1 = 3610
# C_p_1 = 4285
# gamma_1 = C_p_1/C_v_1
# q_1 = -1177788
# q_1_prime = 0 # J/(kg K)

# # water vapor

# Pi_2 = 0 # Pa  
# C_v_2 = 955
# C_p_2 = 1401
# gamma_2 = C_p_2 / C_v_2
# q_2 = 2077616
# q_2_prime = 14370 # J/(kg K)

# liquid water
Pi_1 = 10**9
C_v_1 = 1816
C_p_1 = 4267
gamma_1 = C_p_1 / C_v_1
q_1 = -1167 * 10**3
q_1_prime = 0 # J/(kg K)

# water vapor

Pi_2 = 0 # Pa  
C_v_2 = 1040
C_p_2 = 1487
gamma_2 = C_p_2/C_v_2
q_2 = 2030*10**3
q_2_prime = -23.26*10**3 # J/(kg K)

# liquid dodecane
# Pi_1 = 4*10**8
# C_v_1 = 1077
# C_p_1 = 2534
# gamma_1 = C_p_1/C_v_1
# q_1 = -755*10**3
# q_1_prime = 0 # J/(kg K)
# lambda_1 = 0.140 # w/(m K)

# # dodecane vapor
# Pi_2 = 0 # Pa  
# C_v_2 = 1953
# C_p_2 = 2005
# gamma_2 = C_p_2 / C_v_2
# q_2 = -237*10**3
# q_2_prime = -24.483*10**3 # J/(kg K)
# lambda_2 = 200 # w/(m K)

# Stiffened gas EOS
def h_1(T):
	return gamma_1*C_v_1*T + q_1

def h_2(T):
	return gamma_2*C_v_2*T + q_2

def e_1(P, T):
	return (P+gamma_1*Pi_1)/(gamma_1 - 1)*v_1(P, T) + q_1

def e_1_prime(P, T):
	return C_v_1*T*Pi_1*(1 - gamma_1)/(P+Pi_1)**2

def e_2(P, T):
	return (P+gamma_2*Pi_2)/(gamma_2 - 1)*v_2(P, T) + q_2

def e_2_prime(P, T):
	return C_v_2*T*Pi_2*(1 - gamma_2)/(P+Pi_2)**2

def v_1(P, T):
	return (gamma_1 - 1)*C_v_1*T/(P+Pi_1)

def v_1_prime(P, T):
	return C_v_1*(1-gamma_1)*T/(P + Pi_1)**2

def v_2(P, T):
	return (gamma_2 - 1)*C_v_2*T/(P+Pi_2)

def v_2_prime(P, T):
	return C_v_2*(1-gamma_2)*T/(P + Pi_2)**2

def g_1(P, T):
	return (gamma_1*C_v_1 - q_1_prime)*T - C_v_1*T*np.log(T**gamma_1/((P+Pi_1)**(gamma_1-1)))+q_1

def g_2(P, T):
	return (gamma_2*C_v_2 - q_2_prime)*T - C_v_2*T*np.log(T**gamma_2/((P+Pi_2)**(gamma_2-1)))+q_2

def g_1_prime(P, T): # d/dT
	return gamma_1*C_v_1 - q_1_prime - C_v_1*(np.log(T**gamma_1*(P+Pi_1)**(1-gamma_1)) + gamma_1)

def g_2_prime(P, T): # d/dT
	return gamma_2*C_v_2 - q_2_prime - C_v_2*(np.log(T**gamma_2*(P+Pi_2)**(1-gamma_2)) + gamma_2)

Y_1_0 = 0.99 # initial liquid mass fraction
P_0 = 101325 # initial pressure, Pa
T_0 = 322.577 # initial temperature, K

v_0 = Y_1_0*v_1(P_0, T_0) + (1-Y_1_0)*v_2(P_0, T_0)
e_0 = Y_1_0*e_1(P_0, T_0) + (1-Y_1_0)*e_2(P_0, T_0)
# print(e_0)
print(e_0)
# print(e_0)
def f(P, T):
	T = newton_saturation_curve(P,)
	print(T)
	return ( (e_0 - e_2(P,T)) / (e_1(P,T)-e_2(P,T)) ) - (v_0 - v_2(P,T)) / (v_1(P,T) - v_2(P,T))

def f_prime(P, T):
	T = newton_saturation_curve(P,T)
	return ((e_2(P,T) - e_0)*e_1_prime(P,T) - (e_0 - e_1(P,T))*e_2_prime(P,T)) / (e_2(P,T) - e_1(P,T))**2 + ((v_2(P,T)-v_0)*v_1_prime(P,T) - (v_0 - v_1(P,T))*v_2_prime(P,T)) / (v_2(P,T) - v_1(P,T))**2

# epsilon = 0.00001

def f2(T,P):
	return g_1(P,T) - g_2(P,T)

def f2_prime(T,P):
	return g_1_prime(P,T) - g_2_prime(P,T)

def newton_saturation_curve(P, T):
	# delta = (g_1(P,T) - g_2(P,T)) / (g_1_prime(P,T) - g_2_prime(P,T))
	# max_iterations = 100
	# iteration = 0
	# while abs(delta) >= epsilon and iteration < max_iterations:
	# 	delta = (g_1(P,T) - g_2(P,T)) / (g_1_prime(P,T) - g_2_prime(P,T))
	# 	T = T - delta
	# 	iteration = iteration + 1
	# return T
	print(P,T)
	return optimize.newton(f2, T, fprime=f2_prime, args=(P,))

# def f(P, T):
# 	T = optimize.newton(f2, T, args=(P,))
# 	# return ( (e_0 - e_2(P)) / (e_1(P)-e_2(P)) ) - (v_2(P) - v_0) / (v_2(P) - v_1(P))
# 	# return ( h_2(T) - (e_0 - P*v_0) ) / ( h_2(T) - h_1(T) ) - (v_2(P,T) - v_0) / (v_2(P,T) - v_1(P,T))
# 	return ( (e_0 - e_2(P, T)) / (e_1(P, T)-e_2(P,T)) ) - (v_0 - v_2(P,T)) / (v_1(P,T) - v_2(P,T))

# print(optimize.newton(f, 1, args=(T,)))
print(newton_saturation_curve(P_0, T_0))
P = P_0
# T = newton_saturation_curve(P, T_0)
T = T_0
# delta = f(P,T) / f_prime(P,T)
# print(delta)

# max_iterations = 100
# iteration = 0

# P = optimize.newton(f, P, fprime=f_prime, args=(T,))
# while abs(delta) >= epsilon and iteration < max_iterations:
# 	T = newton_saturation_curve(P,T)
# 	delta = f(P,T) / f_prime(P,T)
		
# 	# print(T)
# 	# print(delta)
# 	P = P - delta

# 	++ iteration
# P_star = P 
# T_star = T

P_star = optimize.newton(f, P_0, args=(T,))
T_star = optimize.newton(f2, T, args=(P_star,))

# Y_water = ( h_2(P_star,) - (e_0 - P*v_0) ) / ( h_2(P_star) - h_1(P_star) )
# Y_water = (e_0-e_2(P_star))/(e_1(P_star)-e_2(P_star))
Y_water = ( (e_0 - e_2(P_star,T_star)) / (e_1(P_star,T_star)-e_2(P_star,T_star) ) )
# print(f2(P_star))
# print(T_sat, P_star)
print("Relaxed pressure:")
print(P_star)
print("")
print("Relaxed temperature")
print(T_star)
print("")
# print(T, optimize.newton(f2, T))
print("Y_liq equilibrium:")
# print(( h_2(P_star) - (e_0 - P*v_0) ) / ( h_2(P_star) - h_1(P_star) ) - (v_2(P_star) - v_0) / (v_2(P_star) - v_1(P_star)))
print(( (e_0 - e_2(P_star,T_star)) / (e_1(P_star,T_star)-e_2(P_star,T_star)) ) - (v_2(P_star,T_star) - v_0) / (v_2(P_star,T_star) - v_1(P_star,T_star)))
print("")
print("Gibbs equilibrium:")
print(g_2(P_star, T_star) - g_1(P_star,T_star))
print("")
print("mass fraction, water:")
print(Y_water)
print("")
# print(g_1(T_sat), g_2(T_sat))
e_0_f = Y_water*e_1(P_star,T_star) + (1-Y_water)*e_2(P_star,T_star)
v_0_f = Y_water*v_1(P_star,T_star) + (1-Y_water)*v_2(P_star,T_star)
print("Specific energy condition satisfied: ", e_1(P_star, T_star) < e_0_f and e_0_f < e_2(P_star, T_star))
print(e_1(P_star, T_star), "<", e_0_f, "<", e_2(P_star, T_star))
print("")
print("Specific volume condition satisfied: ", v_1(P_star, T_star) < v_0_f and v_0_f < v_2(P_star, T_star))
print(v_1(P_star, T_star), "<", v_0_f, "<", v_2(P_star, T_star))
# print(e_0)
# print(e_2(P_star,))
print("")
print("delta(e_0)")
print(e_0)
print(e_0_f)
# print(v_1(P_star))
print("")
print("delta(v_0)")
print(v_0)
print(v_0_f)
# print(v_2(P_star))
# print(v_0)




# 	def calculate_phase_change(self):
# 		# optimize.newton(f2, T)
# 		print(g_1(101400, 373.15))
# 		print(h_1(101400, 373.15))
# 		print(s_1(101400, 373.15))
# 		print(h_1(101400, 373.15) - 373.15*s_1(101400, 373.15))
		
# 		T = np.linspace(298, 503.15, 50)
# 		P = np.zeros(len(T))
# 		for i in range(len(T)):
# 			fder = lambda self, P: -C_v_2*T[i]*((1-gamma_2)/(P + Pi_2)) - C_v_1*T[i]*(1-gamma_1)/(P+Pi_1)
# 			P[i] = optimize.newton(f2, 1, fprime=fder, args=(T[i],)) / 100000
# 			# print(f2(P[i], T[i]))
# 		# print(P)
# 		plt.plot(T,P)
# 		plt.xlim([298,503.15])
# 		plt.xlabel("T (K)")
# 		plt.ylabel("$P_{sat} (Bar)$")
# 		T_exp = np.array([323.15, 348.15, 373.15, 393.15, 423.15, 443.15, 463.15, 473.15, 483.15, 493.15, 503.15])
# 		P_exp = np.array([0.1235, 0.3858, 1.014, 1.985, 4.758, 7.917, 12.54, 15.54, 19.06, 23.18, 27.95])
# 		plt.scatter(T_exp, P_exp)
# 		plt.show()
# 		# P_star = optimize.newton(f, P)

# 		# print(root)

