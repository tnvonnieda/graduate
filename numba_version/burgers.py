import numpy as np
import sys
import math
import matplotlib.pyplot as plt
from get_interpolation_coefficients import get_interpolation_coefficients
from get_derivative_coefficients import get_derivative_coefficients
from get_optimal_weights import get_optimal_weights
from reconstruct import calculate_beta_no_characteristic
from calculate_error import calculate_error
from calculate_convergence_rate import calculate_convergence_rate
from quadrature_weights import get_quadrature_weights, get_quadrature_points
from burger_newton import get_exact
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

class burgers:
	def __init__(self, x_0, x_f, t_f, k, CFL, r, p, time_int, boundary_type, a, b):
		self.x_0 = x_0 # domain left bound
		self.x_f = x_f # domain right bound
		self.x = np.linspace(x_0, x_f, k)
		self.t = 0
		self.t_f = t_f # final time
		self.k = k # number of spatial grid points
		self.r = r # polynomial degree
		self.CFL = CFL
		self.R = 2*r+1
		self.p = p
		self.a = get_interpolation_coefficients(r)
		self.d = get_derivative_coefficients(self.R)
		self.b = get_optimal_weights(r).reshape(r+1,1,1)
		self.h = (x_f-x_0)/(k-1)
		self.x_half = np.linspace(x_0+self.h/2-self.h*(self.r+1), x_f-self.h/2+self.h*(self.r+1), k+1+2*self.r)
		self.time_int = time_int
		self.boundary_type = boundary_type
		self.epsilon = 10**(-40)
		if time_int == 'sdc':
			self.q = get_quadrature_weights(p)
			self.quad_points = get_quadrature_points(p)
		# self.u = np.array([(1 + np.sin(np.pi*self.x))/2]).T
		# self.u = np.array([a*self.x+b]).T
		self.u = np.array([1/2 + np.sin(np.pi*self.x)]).T
		self.u_0 = self.u
		self.num_vars = 1
		self.max_characteristic = 0
		self.run()

	# def weno_left_test(self):
	# 	u = np.pad(self.u, ((2*self.r+1, 2*self.r), (0,0)), mode=self.boundary_type, reflect_type='odd')
	# 	u_test = np.pad(self.u, ((2*self.r, 2*self.r-1), (0,0)), mode=self.boundary_type, reflect_type='odd')
	# 	u_reconstructed = np.zeros(u.shape) 
	# 	P = np.zeros(np.append((self.r+1), u.shape))
	# 	beta = calculate_beta_no_characteristic(u, self.r, shift)

	# 	alpha = self.b/(beta+self.epsilon)**(self.r+1)
	# 	omega = alpha / alpha.sum(axis=0)

	# 	for k_s in range(self.r+1): # for each stencil
	# 		# calculate half point polynomial interpolation, P_{r,k_s,i+1/2}		
	# 		for l in range(self.r+1):
	# 			P[k_s,:,:] += self.a[k_s,l]*shift(u, k_s-l).reshape((len(u), self.num_vars, 1)).reshape((len(u), self.num_vars))	
	# 		u_reconstructed = u_reconstructed + omega[k_s]*P[k_s]

	# 	return u_reconstructed[self.r:-self.r]
	
	# def weno_left(self):
	# 	u = np.pad(self.u[0:-1], ((self.r, self.r), (0,0)), mode=self.boundary_type)
	# 	# u = np.pad(self.u, ((self.r, self.r-1), (0,0)), mode=self.boundary_type, reflect_type='odd')
	# 	u_test = np.pad(self.u, ((2*self.r, 2*self.r-1), (0,0)), mode=self.boundary_type, reflect_type='odd')
	# 	u_reconstructed = np.zeros(u.shape) 
	# 	P = np.zeros(np.append((self.r+1), u.shape))
	# 	beta = calculate_beta_no_characteristic(u, self.r, shift)

	# 	alpha = self.b/(beta+self.epsilon)**(self.r+1)
	# 	omega = alpha / alpha.sum(axis=0)

	# 	for k_s in range(self.r+1): # for each stencil
	# 		# calculate half point polynomial interpolation, P_{r,k_s,i+1/2}		
	# 		for l in range(self.r+1):
	# 			P[k_s,:,:] += self.a[k_s,l]*shift(u, k_s-l).reshape((len(u), self.num_vars, 1)).reshape((len(u), self.num_vars))	
	# 		u_reconstructed = u_reconstructed + omega[k_s]*P[k_s]

	# 	return u_reconstructed[self.r:-self.r]

	def weno_left(self):
		u = np.pad(self.u[0:-1], ((self.r, self.r), (0,0)), mode='wrap')


		P = np.zeros(np.append((self.r+1), u.shape))
		u_reconstructed = np.zeros(u.shape)
		beta = calculate_beta_no_characteristic(u, self.r, shift)
		alpha = self.b/(beta+self.epsilon)**(self.r+1)
		omega = alpha / alpha.sum(axis=0)
		
		alpha_m = omega*(self.b + self.b**2 - 3*self.b*omega + omega**2) / (self.b**2 + omega*(1 - 2*self.b))
		omega_m = alpha_m / alpha_m.sum(axis=0)
		for k_s in range(self.r+1): # for each stencil
			# calculate half point polynomial interpolation, P_{r,k_s,i+1/2}		
			for l in range(self.r+1):
				P[k_s,:,:] += self.a[k_s,l]*shift(u, k_s-l).reshape((len(u), self.num_vars, 1)).reshape((len(u), self.num_vars))
			u_reconstructed = u_reconstructed + omega_m[k_s]*P[k_s]
		return u_reconstructed[self.r:self.k+self.r-1]

	# def weno_right_test(self):
	# 	u = np.flip(np.pad(self.u, ((2*self.r, 2*self.r+1), (0,0)), mode=self.boundary_type, reflect_type='odd'), axis=0)
	# 	u_reconstructed = np.zeros(u.shape) 
	# 	P = np.zeros(np.append((self.r+1), u.shape))
		
	# 	beta = calculate_beta_no_characteristic(u, self.r, shift)
	# 	alpha = self.b/(beta+self.epsilon)**(self.r+1)
	# 	omega = alpha / alpha.sum(axis=0)

	# 	alpha_m = omega*(self.b + self.b**2 - 3*self.b*omega + omega**2) / (self.b**2 + omega*(1 - 2*self.b))
	# 	omega_m = alpha_m / alpha_m.sum(axis=0)

	# 	for k_s in range(self.r+1):
	# 		for l in range(self.r+1):
	# 			P[k_s,:,:] += self.a[k_s,l]*shift(u, k_s-l).reshape((len(u), self.num_vars, 1)).reshape((len(u), self.num_vars))
	# 		u_reconstructed = u_reconstructed + omega_m[k_s]*P[k_s]
		
	# 	return np.flip(u_reconstructed, axis=0)[self.r:-self.r]

	def weno_right(self):
		# u = np.flip(np.pad(self.u, ((self.r-1, self.r), (0,0)), mode=self.boundary_type, reflect_type='odd'), axis=0)
		u = np.flip(np.pad(self.u[1:], ((self.r, self.r), (0,0)), mode='wrap'))
		x_extended = np.linspace(self.x_0 - self.r * self.h, self.x_f + (self.r-1)*self.h, 2*self.r + self.k - 1)
		u_reconstructed = np.zeros(u.shape) 
		P = np.zeros(np.append((self.r+1), u.shape))
		
		beta = calculate_beta_no_characteristic(u, self.r, shift)
		alpha = self.b/(beta+self.epsilon)**(self.r+1)
		omega = alpha / alpha.sum(axis=0)
		alpha_m = omega*(self.b + self.b**2 - 3*self.b*omega + omega**2) / (self.b**2 + omega*(1 - 2*self.b))
		omega_m = alpha_m / alpha_m.sum(axis=0)
		for k_s in range(self.r+1):
			for l in range(self.r+1):
				P[k_s,:,:] += self.a[k_s,l]*shift(u, k_s-l).reshape((len(u), self.num_vars, 1)).reshape((len(u), self.num_vars))
			u_reconstructed = u_reconstructed + omega_m[k_s]*P[k_s]
		
		return np.flip(u_reconstructed, axis=0)[self.r:self.k+self.r-1]

	def get_dudx(self):
		# u_upwind = np.pad(self.weno_left(), [(self.r+1, self.r+1), (0,0)], mode=self.boundary_type, reflect_type='odd')
		# u_split = self.flux_split_test()
		# u_split = self.flux_split()
		u_split = np.pad(self.flux_split(), [(self.r+1, self.r+1), (0,0)], mode='wrap')
		# x_half_extended = np.linspace(self.x_0 + self.h / 2 - (self.r+1)*self.h, self.x_f - self.h/2+(self.r+1)*self.h, self.k - 1 + 2*(self.r+1))
		# plt.plot(x_half_extended, u_upwind)
		# print(u_split)
		# sys.exit()
		# print(len(u_split_test))
		# print(len(u_split))
		# plt.plot(self.x_half, u_split_test)
		# plt.plot(self.x_half, u_split)
		# plt.show()
		# sys.exit()
		# dudx = np.pad(u_split, [(self.r+1, self.r+1), (0,0)], mode='wrap')
		dudx = np.zeros(np.shape(u_split))
		
		for i in range(len(self.d)):
			dudx += self.d[i]*(shift(u_split, -i)-shift(u_split, i+1))
		
		return (dudx/self.h)[self.r+1:self.k+self.r+1]		
 
	def flux_split_test(self):
		self.max_characteristic = np.max(abs(self.u))

		u_reconstructed_l = self.weno_left_test()
		u_reconstructed_r = self.weno_right_test()

		flux_left = u_reconstructed_l**2/2 + self.max_characteristic*u_reconstructed_l
		flux_right = u_reconstructed_r**2/2 - self.max_characteristic*u_reconstructed_r
		
		return 1/2*(flux_left + flux_right)

	def flux_split(self):
		self.max_characteristic = np.max(abs(self.u))

		u_reconstructed_l = self.weno_left()
		# print(np.shape(u_reconstructed_l))
		u_reconstructed_r = self.weno_right()
		# print(np.shape(u_reconstructed_r))
		# x_half = np.linspace(self.x_0 + self.h / 2, self.x_f - self.h / 2, self.k - 1)
		# plt.plot(x_half, u_reconstructed_l, marker='.')
		# plt.plot(x_half, u_reconstructed_r)
		# plt.plot(self.x, self.u)
		# plt.show()
		# sys.exit() 
		flux_left = u_reconstructed_l**2/2 + self.max_characteristic*u_reconstructed_l
		flux_right = u_reconstructed_r**2/2 - self.max_characteristic*u_reconstructed_r
		
		return 1/2*(flux_left + flux_right)

	def rk4(self):
		u = self.u
		
		k1 = -self.get_dudx()
		self.u = u + self.tau*1/2*k1

		k2 = -self.get_dudx()
		self.u = u + self.tau*1/2*k2

		k3 = -self.get_dudx()
		self.u = u + self.tau*k3

		k4 = -self.get_dudx()
		self.u = u + self.tau*(1/6*k1+1/3*k2+1/3*k3+1/6*k4)

	def sdc(self):	
		
		# w = np.empty(np.append(self.p, np.append(2*self.p-1, np.shape(self.u))))
		w = np.empty(np.append(self.p, np.append(2*self.p-1, np.shape(self.u))))
		dudx = np.empty(np.shape(w))

		w[:] = self.u
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
				self.u = w[j-1,k]
				dudx[j-1,k] = self.get_dudx()
				w[j,k] += self.tau*(self.quad_points[j]-self.quad_points[j-1])/2*(-dudx[j-1,k] + dudx[j-1,k-1])
			self.u = w[self.p-1,k]
			if k < 2*self.p-2:				
				dudx[self.p-1,k] = self.get_dudx()
		

	def run(self):
		# plt.ion()
		# fig = plt.figure(1)
		# ax = fig.add_subplot(111)
		# line1, = ax.plot(self.x, self.u,'r-')
		while self.t < self.t_f:
			self.max_characteristic = np.max(abs(self.u))
			self.tau = self.h * self.CFL/self.max_characteristic
			if self.t + self.tau > self.t_f:
				self.tau = self.t_f - self.t
			if self.time_int == 'rk4':
				self.rk4()
			elif self.time_int == 'sdc':
				self.sdc()
			self.t = self.t + self.tau
			# line1.set_ydata(self.u)
			# fig.canvas.draw()
			# fig.canvas.flush_events()

# N = 161
t_s = 1/np.pi
t_f = 0.7

x_0 = 0.
x_f = 2.
CFL = 1.0

a = burgers(x_0, x_f, t_f, 21, CFL, 2, 4, 'sdc', 'wrap', None, None)
b = burgers(x_0, x_f, t_f, 41, CFL, 2, 4, 'sdc', 'wrap', None, None)
c = burgers(x_0, x_f, t_f, 81, CFL, 2, 4, 'sdc', 'wrap', None, None)
d = burgers(x_0, x_f, t_f, 161, CFL, 2, 4, 'sdc', 'wrap', None, None)
# e = burgers(x_0, x_f, t_f, 321, CFL, 7, 9, 'sdc', 'wrap', None, None)
# f = burgers(x_0, x_f, t_f, 641, CFL, 7, 9, 'sdc', 'wrap', None, None)

# print(a.t)
def func(u, x, t):
    return u - 1/2 - np.sin(np.pi*(x-u*t))

x = np.linspace(a.x_0, a.x_f, 5000)
# u_exact = 1/2 + np.sin(np.pi * (x_exact - u*t))
# x_exact = np.linspace(0, 2, 5000)
x_s = 1/(2*np.pi) + 1 + (t_f - t_s)*1/2
print(x_s)
# print(x_s)
u_guess = np.piecewise(x, [x < x_s, x >= x_s], [lambda x: 0.158 + x/x_s, lambda x: -1.3 + x/x_s])
u_exact = np.empty(np.shape(x))
t = t_f
for i in range(len(x)):
	u_exact[i] = optimize.newton(func, u_guess[i], args=(x[i], t))
plt.plot(x, u_exact, color='maroon', linewidth=2)

plt.title('WENO5')
plt.xlabel(r'$x$')
plt.ylabel(r'$u(x,t)$')
# plt.vlines(x_s, -5, 5)
plt.plot(a.x, a.u, color='forestgreen', linestyle='dashed', linewidth=2) # 21
plt.plot(b.x, b.u, color='navy', linestyle='dashdot', linewidth=2, marker='o') # 41
plt.plot(c.x, c.u, color='red', linestyle='dashed', linewidth=2) # 81
plt.plot(d.x, d.u, color='mediumorchid', linestyle='dashdot', linewidth=2) # 161
# plt.plot(e.x, e.u, color='teal', linestyle='dashed', linewidth=2) # 321
# plt.plot(f.x, f.u, color='black', linestyle='dashdot', linewidth=2) # 641
plt.show()


# for SDC convergence testing
# N = 10
# t_f = 10
# x_0 = -100
# x_f = 100
# a = 1
# b = 1
# x = np.linspace(x_0, x_f, N)
# CFL_array = np.array([2.5, 2.4, 2.3, 2.2, 2.1, 2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0])
# L1_array = np.empty_like(CFL_array)
# L2_array = np.empty_like(CFL_array)
# Linf_array = np.empty_like(CFL_array)
# sdc_results = np.zeros((len(CFL_array), N))
# u_exact = (a*x + b)/(a*t_f + 1)

# for i in range(len(CFL_array)):
# 	sdc_results[i] = burgers(x_0, x_f, t_f, N, CFL_array[i], 3, 7, 'sdc', 'reflect', a, b).u.T[0,:]
# 	L1, L2, Linf = calculate_error(sdc_results[i], u_exact)
# 	L1_array[i] = L1
# 	L2_array[i] = L2
# 	Linf_array[i] = Linf
# 	# print(sdc_results[i])
# 	print('done')
# r_c_L1 = calculate_convergence_rate(L1_array, CFL_array)
# r_c_L2 = calculate_convergence_rate(L2_array, CFL_array)
# r_c_L_inf = calculate_convergence_rate(Linf_array, CFL_array)

# # print(Linf_array)
# # print(r_c_L_inf)
# # print(L1_array)
# # print(r_c_L1)
# # print(L2_array)
# # print(r_c_L2)

# # error = 
# # plt.plot(x, u_exact, marker='.')
# # plt.plot(x, sdc_results[0])

# # plt.show()
# def round_to_sigfigs(x, n):
#     n = n-1
#     if x == 0:
#     	return 0
#     else:
#     	return round(x, -int(np.floor(np.sign(x) * np.log10(abs(x)))) + n) 

# print("\\begin{table}[H]\n\\centering\n\\caption{}\n\\begin{tabular}{ c | c c | c c | c c }\n CFL & \\(L_{\\infty}\\) Norm & \\(r_c\\) & \\(L_1\\) Norm & \\(r_c\\) & \\(L_2\\) Norm & \\(r_c\\) \\\\\n\\hline")
# for i in range(len(CFL_array)):
# 	print(CFL_array[i], "&", round_to_sigfigs(Linf_array[i],5), "&", round_to_sigfigs(r_c_L_inf[i],5), "&", round_to_sigfigs(L1_array[i],5), "&", round_to_sigfigs(r_c_L1[i],5), "&", round_to_sigfigs(L2_array[i],5), "&", round_to_sigfigs(r_c_L2[i],5), "\\\\")
# print("\\end{tabular}\n\\end{table}")
# \begin{table}[H]
# \centering
# \resizebox{\textwidth}{!}{\begin{tabular}{ c | c c | c c | c c }
# & \multicolumn{6}{c}{Polynomial Error} \\
# \hline
# CFL & \(L_{\infty}\) Norm & \(r_c\) & \(L_1\) Norm & \(r_c\) & \(L_2\) Norm & \(r_c\) \\
# \hline
# 0.5 & 0.1194532 &  & 0.00173965 & & 0.00123589 &  \\
# 0.25 & 0.11919595 &  & 0.00173455 & & 0.00123303 &  \\
# 0.125 & 0.11914326 &  & 0.00173368 &  & 0.00123242 &  \\
# 0.0625 & 0.11914099 &  & 0.00173364 &  & 0.0012324 & \\
# 0.03125 & 0.11914086 &  & 0.0012324 &  & 0.0012324 &
# \end{tabular}}
# \end{table}