import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import math

# liquid water

Pi_1 = 7028*10**5
C_v_1 = 3610
C_p_1 = 4285
gamma_1 = C_p_1/C_v_1
b_1 = 6.61*10**-4
q_1 = -1177788
q_1_prime = 0 # J/(kg K)

# water vapor

Pi_2 = 0 # Pa  
C_v_2 = 955
C_p_2 = 1401
gamma_2 = C_p_2 / C_v_2
b_2 = 0
q_2 = 2077616
q_2_prime = 14317 # J/(kg K)

def v(P, T, gamma, C_v, Pi, b):
	return (gamma - 1)*C_v*T/(P+Pi) + b

def enthalpy(T, gamma, C_v, q):
	return gamma*C_v*T + q

def g_1(P, T):
	return (gamma_1*C_v_1 - q_1_prime)*T - C_v_1*T*np.log(T**gamma_1/((P+Pi_1)**(gamma_1-1)))+b_1+q_1

def g_1_prime(P, T): # d/dp
	return (- C_v_1*T*(1-gamma_1)/(P+Pi_1))

def g_2(P, T):
	return (gamma_2*C_v_2 - q_2_prime)*T - C_v_2*T*np.log(T**gamma_2/((P+Pi_2)**(gamma_2-1)))+b_2+q_2	

def g_2_prime(P,T):
	return (-C_v_2*T*(1-gamma_2)/(P + Pi_2))

def f2(P,T):
	return np.log(P+Pi_2) - A - B/T - C*np.log(T) - D*np.log(P+Pi_1)

def f2_prime(P,T):
	return (1/(P+Pi_2) - D/(P+Pi_1))

def f(P, T):
	return g_2(P, T) - g_1(P, T)

# def g_1_prime(P, T): # d/dT
# 	return gamma_1*C_v_1 - q_1_prime - C_v_1*(np.log(T**gamma_1*(P+Pi_1)**(1-gamma_1)) + gamma_1)

# def g_2_prime(P, T): # d/dT
# 	return gamma_2*C_v_2 - q_2_prime - C_v_2*(np.log(T**gamma_2*(P+Pi_2)**(1-gamma_2)) + gamma_2)
epsilon = 0.00001

def newton(P, T):
	h = (g_1(P,T) - g_2(P,T)) / (g_1_prime(P,T) - g_2_prime(P,T))
	# h = f2(P,T) / f2_prime(P,T)
	max_iterations = 100
	iteration = 0
	while abs(h) >= epsilon and iteration < max_iterations:
		# print(P / 100000)
		h = (g_1(P,T) - g_2(P,T)) / (g_1_prime(P,T) - g_2_prime(P,T))
		# h = f2(P,T) / f2_prime(P,T)
		P = P - h
		iteration = iteration + 1
	return P

T = np.linspace(273.15, 573.15, 100)
P = np.zeros(len(T))
# h_l = np.zeros(len(T))
# h_g = np.zeros(len(T))
# v_l = np.zeros(len(T))
# v_g = np.zeros(len(T))

# P = np.linspace(100000, 1000000, 50)
# T = np.zeros(len(P))
h_l = np.zeros(len(T))
h_g = np.zeros(len(T))
v_l = np.zeros(len(T))
v_g = np.zeros(len(T))

for i in range(len(T)):
	# fder = lambda P, T: -C_v_2*T*(1-gamma_2)/(P + Pi_2) - C_v_1*T*(1-gamma_1)/(P+Pi_1)
	# fder = lambda P, T: 1/(P+Pi_2) - D/(P+Pi_1)
	# P[i] = optimize.newton(f2, 1, fprime=f2_prime, args=(T[i],))
	
	P[i] = newton(1, T[i])
	# T[i] = newton(P[i], 1)
	v_l[i] = v(P[i], T[i], gamma_1, C_v_1, Pi_1, b_1) 
	v_g[i] = v(P[i], T[i], gamma_2, C_v_2, Pi_2, b_2)
	h_l[i] = enthalpy(T[i], gamma_1, C_v_1, q_1) / 10**3
	h_g[i] = enthalpy(T[i], gamma_2, C_v_2, q_2) / 10**3

# plt.ylim([2000, 2800])
plt.xlabel("T (K)")
plt.xlim([273,573])

T_exp =          np.array([273.16,  288.15,  300.15,  323.15,  348.15, 373.15, 393.15, 423.15, 443.15, 463.15, 473.15, 483.15, 493.15,  503.15,  513.15,  523.15,  533.15,  543.15,  553.15,  563.15,  573.15])
P_exp =    10**5*np.array([0.00611, 0.01705, 0.03567, 0.1235,  0.3858, 1.014,  1.985,  4.758,  7.917,  12.54,  15.54,  19.06,  23.18,   27.95,   33.44,   39.73,   46.88,   54.99,   64.12,   74.36,   85.81])
v_g_exp =        np.array([206.136, 77.926,  38.774,  12.032,  4.131,  1.673,  0.8919, 0.3928, 0.2428, 0.1565, 0.1274, 0.1044, 0.08619, 0.07158, 0.05976, 0.05013, 0.04221, 0.03564, 0.03017, 0.02557, 0.02167])
v_l_exp = 10**-3*np.array([1.0002,  1.0009,  1.0035,  1.0121,  1.0259, 1.0435, 1.0603, 1.0905, 1.1143, 1.1414, 1.1565, 1.1726, 1.1900,  1.2088,  1.2291,  1.2512,  1.2755,  1.3023,  1.3321,  1.3656,  1.4036])
h_l_exp =        np.array([0.01,    62.99,   113.25,  209.33,  313.93, 419.04, 503.71, 632.2,  719.21, 807.62, 852.45, 897.76, 943.62,  990.12,  1037.3,  1085.4,  1134.4,  1184.5,  1236.0,  1289.1,  1344.0])
h_g_exp =        np.array([2501.4,  2528.9,  2550.8,  2592.1,  2635.3, 2676.1, 2706.3, 2746.5, 2768.7, 2786.4, 2793.2, 2798.5, 2802.1,  2804.0,  2803.8,  2801.5,  2796.6,  2789.7,  2779.6,  2766.2,  2749.0])
L_v_exp = h_g_exp - h_l_exp

plt.plot(T,P)
plt.scatter(T_exp, P_exp)
plt.ylabel("$p_{sat}, (Pa)$")
# plt.ylim([0, 80])

# plt.plot(T, v_g)
# plt.scatter(T_exp, v_g_exp)
# plt.ylabel("$\\nu_{vap}$ $(m^{3}/kg)$")
# plt.yscale("log")
# plt.ylim([0, 250])

# plt.plot(T, v_l)
# plt.scatter(T_exp, v_l_exp)
# plt.ylabel("$\\nu_{liq}$ $(m^{3}/kg)$")
# plt.yscale("log")

# plt.plot(T, h_g)
# plt.scatter(T_exp, h_g_exp)
# plt.ylabel("$h_{vap}$ $(J/kg)$")
# plt.ylim([2100, 3000])

# plt.plot(T, h_l)
# plt.scatter(T_exp, h_l_exp)
# plt.ylabel("$h_{liq}$ $(J/kg)$")
# plt.ylim([0, 1400])

# L_v = h_g - h_l
# plt.plot(T, L_v)
# plt.scatter(T_exp, L_v_exp)
# plt.ylabel("$L_{v}$ $(J/kg)$")
# plt.ylim([1400, 2600])

plt.show()

# 	def s_1(self, P, T):
# 		return C_v_1*math.log(T**gamma_1/(P+Pi_1)**(gamma_1 - 1)) + q_1_prime

# 	def s_2(self, P, T):
# 		return C_v_2*math.log(T**gamma_2/(P+Pi_2)**(gamma_2 - 1)) + q_2_prime