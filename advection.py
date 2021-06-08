import numpy as np
import sys
import math
import matplotlib.pyplot as plt
from get_interpolation_coefficients import get_interpolation_coefficients
from get_derivative_coefficients import get_derivative_coefficients
from get_optimal_weights import get_optimal_weights
from reconstruct import calculate_beta_no_characteristic
from quadrature_weights import get_quadrature_weights, get_quadrature_points

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

class advection:
	def __init__(self, x_0, x_f, t_f, k, advection_velocity, CFL, r, p, time_int, boundary_type, problem_type, mapped):
		self.x_0 = x_0 # domain left bound
		self.x_f = x_f # domain right bound
		self.x = np.linspace(x_0, x_f, k, dtype=np.longdouble)
		self.mapped = mapped
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
		self.v_a = advection_velocity
		self.time_int = time_int
		self.boundary_type = boundary_type
		self.Pi = 0
		self.tau = self.h/abs(self.v_a)*self.CFL
		self.epsilon = 10**(-40)
		self.tiny = np.finfo(1.0).tiny
		if time_int == 'sdc':
			self.q = get_quadrature_weights(p)
			self.quad_points = get_quadrature_points(p)
		if problem_type == 'discontinuous':
			# self.u = np.array([np.piecewise(self.x, [self.x < 1/4, (self.x >= 1/4) & (self.x <= 3/4), self.x > 3/4], [0, 1, 0])]).T
			self.u = np.array([np.piecewise(self.x, [self.x < -1/2, (self.x >= -1/2) & (self.x <= 1/2), self.x > 1/2], [0, 1, 0])]).T
		else:
			# self.u = np.array([np.sin(np.pi*self.x)**4]).T
			self.u = np.array([np.sin(np.pi*self.x - np.sin(np.pi*self.x)/np.pi)]).T
			# self.u = np.sin(2*np.pi*self.x).T
			# self.u = np.array([np.sin(np.pi*self.x)**2]).T
			# self.u = np.array([np.sin(np.pi*self.x)], dtype=np.longdouble).T
		self.u_0 = np.copy(self.u)
		self.num_vars = 1
		# self.weno_left()
		# self.get_dudx()
		self.run()
		# self.N = u_p0.shape[0] 
		# self.PLOT_TYPE = REAL_TIME # REAL_TIME/END_TIME

	def weno_left(self):
		u = np.pad(self.u[0:-1], ((self.r, self.r), (0,0)), mode=self.boundary_type)
		# x_extended = np.linspace(self.x_0 - self.r * self.h, self.x_f + (self.r-1)*self.h, 2*self.r + self.k - 1)
		# print(np.shape(u))
		# print(np.shape(x_extended))
		# plt.plot(x_extended, u, marker='o')
		# plt.show()
		# sys.exit()
		P = np.zeros(np.append((self.r+1), u.shape))
		u_reconstructed = np.zeros(u.shape)
		beta = calculate_beta_no_characteristic(u, self.r, shift)
		# print(beta)
		alpha = self.b/(self.tiny + beta**(self.r+1))
		# print(alpha)
		# sys.exit()
		# alpha = self.b/(beta+self.epsilon)**(r+1)
		omega = alpha / alpha.sum(axis=0)
		# print(omega)
		# sys.exit()
		if self.mapped:
			alpha = omega*(self.b + self.b**2 - 3*self.b*omega + omega**2) / (self.b**2 + omega*(1 - 2*self.b))
			omega = alpha / alpha.sum(axis=0)

		for k_s in range(self.r+1): # for each stencil
			# calculate half point polynomial interpolation, P_{r,k_s,i+1/2}		
			for l in range(self.r+1):
				P[k_s,:,:] += self.a[k_s,l]*shift(u, k_s-l).reshape((len(u), self.num_vars, 1)).reshape((len(u), self.num_vars))
			u_reconstructed = u_reconstructed + omega[k_s]*P[k_s]

		return u_reconstructed[self.r:self.k+self.r-1]

	def weno_right(self): 
		u = np.flip(np.pad(self.u[1:], ((self.r, self.r), (0,0)), mode=self.boundary_type))
		
		P = np.zeros(np.append((self.r+1), u.shape))
		u_reconstructed = np.zeros(u.shape)		
		beta = calculate_beta_no_characteristic(u, self.r, shift)
		# alpha = self.b/(beta+self.epsilon)**2
		alpha = self.b/(beta+self.epsilon)**(self.r+1)
		omega = alpha / alpha.sum(axis=0)

		for k_s in range(self.r+1):
			for l in range(self.r+1):
				P[k_s,:,:] += self.a[k_s,l]*shift(u, k_s-l).reshape((len(u), self.num_vars, 1)).reshape((len(u), self.num_vars))
			u_reconstructed = u_reconstructed + omega[k_s]*P[k_s]

		return np.flip(u_reconstructed, axis=0)[self.r:self.k+self.r-1]

	def get_dudx(self):
		# print(len(self.weno_left()))
		# print(self.x)
		# print(self.u)
		# x_half = np.linspace(self.x_0 + self.h/2, self.x_f - self.h/2, self.k -1)
		# x_exact = np.linspace(self.x_0, self.x_f, 10001)
		# u_exact = np.array([np.sin(np.pi*x_exact)]).T
		# plt.plot(self.x, self.u)
		# plt.plot(x_half, self.weno_left())
		# plt.plot(x_exact, u_exact)
		# plt.show()
		# sys.exit()
		u_upwind = np.pad(self.weno_left(), [(self.r+1, self.r+1), (0,0)], mode=self.boundary_type)
		
		x_half_extended = np.linspace(self.x_0 + self.h / 2 - (self.r+1)*self.h, self.x_f - self.h/2+(self.r+1)*self.h, self.k - 1 + 2*(self.r+1))
		# plt.plot(x_half_extended, u_upwind)
		# plt.plot(self.x, self.u)
		# plt.show()
		# sys.exit()

		dudx = np.zeros(np.shape(u_upwind))
		for i in range(len(self.d)):
			dudx += self.d[i]*(shift(u_upwind, -i)-shift(u_upwind, i+1))
		# print(len((dudx/self.h)[self.r+1:self.k+self.r+1]))
		# print(len((dudx/self.h)[self.r+1:-self.r]))
		# sys.exit()
		return (dudx/self.h)[self.r+1:self.k+self.r+1]		

	def rk4(self):
		u = self.u
		# plt.plot(self.x, self.u, marker='o')
		# plt.plot(self.x, self.get_dudx(), marker='o')
		# plt.plot(self.x, np.array([np.pi*np.cos(np.pi*self.x)]).T)
		# plt.show()
		# sys.exit()
		k1 = -self.v_a*self.get_dudx()
		self.u = u + self.tau*1/2*k1

		k2 = -self.v_a*self.get_dudx()
		self.u = u + self.tau*1/2*k2

		k3 = -self.v_a*self.get_dudx()
		self.u = u + self.tau*k3

		k4 = -self.v_a*self.get_dudx()
		self.u = u + self.tau*(1/6*k1+1/3*k2+1/3*k3+1/6*k4)

		self.t = self.t + self.tau

	# def sdc4(self):
	# 	w = np.empty(np.append(self.p, np.append(self.p+2, np.shape(self.u))))
	# 	dudx = np.empty(np.append(self.p, np.append(self.p+2, np.shape(self.u))))
	# 	q = np.array([[5/24, 1/3, -1/24], [-1/24,1/3,5/24]])

	# 	w[:] = self.u
	# 	dudx[:] = self.get_dudx()

	# 	for k in range(1,self.p+2):
	# 		w[1,k] = w[0,k]
	# 		w[2,k] = 0 
			
	# 		for j in range(self.p):
	# 			w[1,k] += -self.tau*self.v_a*q[0,j]*dudx[j,k-1]
	# 			w[2,k] += -self.tau*self.v_a*q[1,j]*dudx[j,k-1]

	# 		w[2,k] += w[1,k]
	# 		self.u = w[1,k]
	# 		dudx[1,k] = self.get_dudx()
	# 		w[2,k] += self.tau/2*self.v_a*(-dudx[1,k] + dudx[1,k-1])
			
	# 		if k < self.p+1:
	# 			self.u = w[2,k]
	# 			dudx[2,k] = self.get_dudx()

	# 	self.u = w[self.p-1, self.p+1]	
	# 	self.t = self.t + self.tau

	def sdc(self):
		# self.tau = self.CFL * self.h / self.get_maximum_characteristic(self.u_p)
		w = np.empty(np.append(self.p, np.append(2*self.p-1, np.shape(self.u))))
		dudx = np.empty(np.append(self.p, np.append(2*self.p-1, np.shape(self.u))))

		w[:] = self.u
		dudx[:] = self.get_dudx()

		for k in range(1,2*self.p-1):
			w[1,k] = w[0,k]
			for m in range(2,self.p):
				w[m,k] = 0 
			
			for j in range(self.p):
				for m in range (1,self.p):
					w[m,k] += -self.tau*self.v_a*self.q[m-1,j]*dudx[j,k-1]
				
			for j in range(2,self.p):
				w[j,k] += w[j-1,k]
				self.u = w[j-1,k]
				dudx[j-1,k] = self.get_dudx()
				w[j,k] += self.tau*(self.quad_points[j]-self.quad_points[j-1])/2*(-dudx[j-1,k] + dudx[j-1,k-1])
			
			self.u = w[self.p-1,k]
		
			if k < 2*self.p-2:				
				dudx[self.p-1,k] = self.get_dudx()

		

	# def sdc_modified(self):
	# 	w = np.empty(np.append(self.p, np.append(self.p+2, np.shape(self.u))))
	# 	dudx = np.empty(np.append(self.p, np.append(self.p+2, np.shape(self.u))))
	# 	q = np.array([[5/24, 1/3, -1/24], [-1/24,1/3,5/24]])
	# 	r = [1, 1, 1, 2]
	# 	self.r = r[0]
	# 	self.a = get_interpolation_coefficients(self.r)
	# 	self.d = get_derivative_coefficients(2*self.r+1)
	# 	self.b = get_optimal_weights(self.r).reshape(self.r+1,1,1)
	# 	w[:] = self.u
	# 	dudx[:] = self.get_dudx()

	# 	for k in range(1,self.p+2):
	# 		w[1,k] = w[0,k]
	# 		w[2,k] = 0 
			
	# 		for j in range(self.p):
	# 			w[1,k] += -self.tau*self.v_a*q[0,j]*dudx[j,k-1]
	# 			w[2,k] += -self.tau*self.v_a*q[1,j]*dudx[j,k-1]

	# 		w[2,k] += w[1,k]
	# 		self.u = w[1,k]
	# 		dudx[1,k] = self.get_dudx()
	# 		w[2,k] += self.tau/2*self.v_a*(-dudx[1,k] + dudx[1,k-1])
			
	# 		# plt.plot(w[2,k])
	# 		if k < self.p+1:
	# 			self.r = r[k]
	# 			self.a = get_interpolation_coefficients(self.r)
	# 			self.d = get_derivative_coefficients(2*self.r+1)
	# 			self.b = get_optimal_weights(self.r).reshape(self.r+1,1,1)
	# 			self.u = w[2,k]
	# 			dudx[2,k] = self.get_dudx()
	# 	# plt.legend(["k=1", "k=2", "k=3", "k=4"])
	# 	# plt.show()
	# 	# sys.exit()
	# 	self.u = w[self.p-1, self.p+1]	
	# 	self.t = self.t + self.tau
	
	###
	

	def run(self):
		# plt.ion()
		# fig = plt.figure(1)
		# ax = fig.add_subplot(111)
		# line1, = ax.plot(self.x, self.u,'r-')
		while self.t < self.t_f:
			if self.t + self.tau > self.t_f:
				self.tau = self.t_f - self.t
			if self.time_int == 'rk4':
				self.rk4()
			elif self.time_int == 'sdc':
				self.sdc()
			elif self.time_int == 'mod_sdc':
				self.sdc_modified()
			self.t = self.t + self.tau
			# line1.set_ydata(self.u)
			# fig.canvas.draw()
			# fig.canvas.flush_events()
		self.max_error = np.amax(np.absolute(self.u-self.u_0))
		# plt.plot(self.x, self.u, marker='.')
# 		# plt.show()

# # print(np.log((5.6215*10**-7)/(3.5121*10**-7)) / np.log(0.9/0.8))

# # print(np.finfo(np.longdouble).eps)
# # 18
# # sys.exit()
# # sdc3 = advection(0, 1, 5, 101, 1, 0.5, 1, 3, 'sdc', 'wrap', 'smooth')
# # sdc5 = advection(0, 1, 5, 101, 1, 0.5, 2, 3, 'sdc', 'wrap', 'smooth')
# # mod_sdc = advection(0, 1, 5, 101, 1, 0.5, 2, 3, 'mod_sdc', 'wrap', 'smooth')

# a = advection(-1, 1, 16, 21, 1, 1.0, 7, 9, 'sdc', 'wrap', 'discontinuous', False)
# b = advection(-1, 1, 16, 41, 1, 1.0, 7, 9, 'sdc', 'wrap', 'discontinuous', False)
# c = advection(-1, 1, 16, 81, 1, 1.0, 7, 9, 'sdc', 'wrap', 'discontinuous', False)
# d = advection(-1, 1, 16, 161, 1, 1.0, 7, 9, 'sdc', 'wrap', 'discontinuous', False)
# e = advection(-1, 1, 16, 321, 1, 1.0, 7, 9, 'sdc', 'wrap', 'discontinuous', False)
# f = advection(-1, 1, 16, 641, 1, 1.0, 7, 9, 'sdc', 'wrap', 'discontinuous', False)

# plt.title("WENO 15", fontsize=16)
# plt.xlabel(r'$x$', fontsize=14)
# plt.ylabel(r'$u(x,t)$', fontsize=14)
# plt.xlim((-1,1))
# plt.plot(a.x, a.u, color='forestgreen', linestyle='dashed', linewidth=2) # 21
# plt.plot(b.x, b.u, color='navy', linestyle='dashdot', linewidth=2) # 41
# plt.plot(c.x, c.u, color='red', linestyle='dashed', linewidth=2) # 81
# plt.plot(d.x, d.u, color='mediumorchid', linestyle='dashdot', linewidth=2) # 161
# plt.plot(e.x, e.u, color='teal', linestyle='dashed', linewidth=2) # 321
# plt.plot(f.x, f.u, color='black', linestyle='dashdot', linewidth=2) # 641
# x_exact = np.linspace(a.x_0, a.x_f, 5000)
# u_exact = np.array([np.piecewise(x_exact, [x_exact < -1/2, (x_exact >= -1/2) & (x_exact <= 1/2), x_exact > 1/2], [0, 1, 0])]).T

# plt.plot(x_exact, u_exact, color='maroon', linewidth=2)

# plt.show()
t_f = 20.0
CFL = 1.0
a1 = advection(-1, 1, t_f, 11, 1, CFL, 5, 7, 'sdc', 'wrap', 'smooth', True)
b1 = advection(-1, 1, t_f, 21, 1, CFL, 5, 7, 'sdc', 'wrap', 'smooth', True)
c1 = advection(-1, 1, t_f, 31, 1, CFL, 5, 7, 'sdc', 'wrap', 'smooth', True)
d1 = advection(-1, 1, t_f, 41, 1, CFL, 5, 7, 'sdc', 'wrap', 'smooth', True)
e1 = advection(-1, 1, t_f, 51, 1, CFL, 5, 7, 'sdc', 'wrap', 'smooth', True)
f1 = advection(-1, 1, t_f, 61, 1, CFL, 5, 7, 'sdc', 'wrap', 'smooth', True)
g1 = advection(-1, 1, t_f, 71, 1, CFL, 5, 7, 'sdc', 'wrap', 'smooth', True)
h1 = advection(-1, 1, t_f, 81, 1, CFL, 5, 7, 'sdc', 'wrap', 'smooth', True)

print('done')

a2 = advection(-1, 1, t_f, 11, 1, CFL, 5, 7, 'sdc', 'wrap', 'smooth', False)
b2 = advection(-1, 1, t_f, 21, 1, CFL, 5, 7, 'sdc', 'wrap', 'smooth', False)
c2 = advection(-1, 1, t_f, 31, 1, CFL, 5, 7, 'sdc', 'wrap', 'smooth', False)
d2 = advection(-1, 1, t_f, 41, 1, CFL, 5, 7, 'sdc', 'wrap', 'smooth', False)
e2 = advection(-1, 1, t_f, 51, 1, CFL, 5, 7, 'sdc', 'wrap', 'smooth', False)
f2 = advection(-1, 1, t_f, 61, 1, CFL, 5, 7, 'sdc', 'wrap', 'smooth', False)
g2 = advection(-1, 1, t_f, 71, 1, CFL, 5, 7, 'sdc', 'wrap', 'smooth', False)
h2 = advection(-1, 1, t_f, 81, 1, CFL, 5, 7, 'sdc', 'wrap', 'smooth', False)

print('done')

a3 = advection(-1, 1, t_f, 11, 1, CFL, 6, 8, 'sdc', 'wrap', 'smooth', True)
b3 = advection(-1, 1, t_f, 21, 1, CFL, 6, 8, 'sdc', 'wrap', 'smooth', True)
c3 = advection(-1, 1, t_f, 31, 1, CFL, 6, 8, 'sdc', 'wrap', 'smooth', True)
d3 = advection(-1, 1, t_f, 41, 1, CFL, 6, 8, 'sdc', 'wrap', 'smooth', True)
e3 = advection(-1, 1, t_f, 51, 1, CFL, 6, 8, 'sdc', 'wrap', 'smooth', True)
f3 = advection(-1, 1, t_f, 61, 1, CFL, 6, 8, 'sdc', 'wrap', 'smooth', True)
g3 = advection(-1, 1, t_f, 71, 1, CFL, 6, 8, 'sdc', 'wrap', 'smooth', True)
h3 = advection(-1, 1, t_f, 81, 1, CFL, 6, 8, 'sdc', 'wrap', 'smooth', True)

print('done')

a4 = advection(-1, 1, t_f, 11, 1, CFL, 6, 8, 'sdc', 'wrap', 'smooth', False)
b4 = advection(-1, 1, t_f, 21, 1, CFL, 6, 8, 'sdc', 'wrap', 'smooth', False)
c4 = advection(-1, 1, t_f, 31, 1, CFL, 6, 8, 'sdc', 'wrap', 'smooth', False)
d4 = advection(-1, 1, t_f, 41, 1, CFL, 6, 8, 'sdc', 'wrap', 'smooth', False)
e4 = advection(-1, 1, t_f, 51, 1, CFL, 6, 8, 'sdc', 'wrap', 'smooth', False)
f4 = advection(-1, 1, t_f, 61, 1, CFL, 6, 8, 'sdc', 'wrap', 'smooth', False)
g4 = advection(-1, 1, t_f, 71, 1, CFL, 6, 8, 'sdc', 'wrap', 'smooth', False)
h4 = advection(-1, 1, t_f, 81, 1, CFL, 6, 8, 'sdc', 'wrap', 'smooth', False)
print('done')
a5 = advection(-1, 1, t_f, 11, 1, CFL, 7, 9, 'sdc', 'wrap', 'smooth', True)
b5 = advection(-1, 1, t_f, 21, 1, CFL, 7, 9, 'sdc', 'wrap', 'smooth', True)
c5 = advection(-1, 1, t_f, 31, 1, CFL, 7, 9, 'sdc', 'wrap', 'smooth', True)
d5 = advection(-1, 1, t_f, 41, 1, CFL, 7, 9, 'sdc', 'wrap', 'smooth', True)
# e5 = advection(-1, 1, t_f, 51, 1, CFL, 7, 9, 'sdc', 'wrap', 'smooth', True)

print('done')
a6 = advection(-1, 1, t_f, 11, 1, CFL, 7, 9, 'sdc', 'wrap', 'smooth', False)
b6 = advection(-1, 1, t_f, 21, 1, CFL, 7, 9, 'sdc', 'wrap', 'smooth', False)
c6 = advection(-1, 1, t_f, 31, 1, CFL, 7, 9, 'sdc', 'wrap', 'smooth', False)
d6 = advection(-1, 1, t_f, 41, 1, CFL, 7, 9, 'sdc', 'wrap', 'smooth', False)
e6 = advection(-1, 1, t_f, 51, 1, CFL, 7, 9, 'sdc', 'wrap', 'smooth', False)
f6 = advection(-1, 1, t_f, 61, 1, CFL, 7, 9, 'sdc', 'wrap', 'smooth', False)
print('done')
a7 = advection(-1, 1, t_f, 11, 1, CFL, 8, 10, 'sdc', 'wrap', 'smooth', True)
b7 = advection(-1, 1, t_f, 21, 1, CFL, 8, 10, 'sdc', 'wrap', 'smooth', True)
c7 = advection(-1, 1, t_f, 31, 1, CFL, 8, 10, 'sdc', 'wrap', 'smooth', True)
d7 = advection(-1, 1, t_f, 41, 1, CFL, 8, 10, 'sdc', 'wrap', 'smooth', True)
# e7 = advection(-1, 1, t_f, 51, 1, CFL, 8, 10, 'sdc', 'wrap', 'smooth', True)
print('done')
a8 = advection(-1, 1, t_f, 11, 1, CFL, 8, 10, 'sdc', 'wrap', 'smooth', False)
b8 = advection(-1, 1, t_f, 21, 1, CFL, 8, 10, 'sdc', 'wrap', 'smooth', False)
c8 = advection(-1, 1, t_f, 31, 1, CFL, 8, 10, 'sdc', 'wrap', 'smooth', False)
d8 = advection(-1, 1, t_f, 41, 1, CFL, 8, 10, 'sdc', 'wrap', 'smooth', False)
# e8 = advection(-1, 1, t_f, 51, 1, CFL, 8, 10, 'sdc', 'wrap', 'smooth', False)

plt.xlabel(r'$N$', fontsize=14)
plt.ylabel(r'$L_{\infty}$-Error', fontsize=14)

# points_1 = np.array([a1.k, b1.k, c1.k, d1.k, e1.k, f1.k, g1.k, h1.k])
# max_errors_1 = np.array([a1.max_error, b1.max_error, c1.max_error, d1.max_error, e1.max_error, f1.max_error, g1.max_error, h1.max_error])

# points_2 = np.array([a2.k, b2.k, c2.k, d2.k, e2.k, f2.k, g2.k])
# max_errors_2 = np.array([a2.max_error, b2.max_error, c2.max_error, d2.max_error, e2.max_error, f2.max_error, g2.max_error])

# points_3 = np.array([a3.k, b3.k, c3.k, d3.k, e3.k, f3.k])
# max_errors_3 = np.array([a3.max_error, b3.max_error, c3.max_error, d3.max_error, e3.max_error, f3.max_error])

# plt.xscale("log")
# plt.yscale("log")

# plt.plot(points_1, max_errors_1, marker='o', linestyle='dotted', color='black')
# plt.text(points_1[-1] + 200, max_errors_1[-1], r'$r=2$', fontsize='large', color='black')
# plt.plot(points_2, max_errors_2, marker='o', linestyle='dotted', color='red')
# plt.text(points_2[-1] + 100, max_errors_2[-1], r'$r=3$', fontsize='large', color='red')
# plt.plot(points_3, max_errors_3, marker='o', linestyle='dotted', color='navy')
# plt.text(points_3[-1] + 50, max_errors_3[-1], r'$r=4$', fontsize='large', color='navy')
# plt.show()
# max_error = np.amax(np.absolute(a.u-a_exact))
# plt.plot(a.x, a.u)
# plt.plot(a.x, a.u_0)
# plt.show()
# print(a.u-a_exact)
# print(max_error)

# plot convergence 

results_1 = np.array([a1, b1, c1, d1, e1, f1, g1, h1])

points_1 = np.empty_like(results_1)
steps_1 = np.empty_like(results_1)
max_errors_1 = np.empty_like(results_1)
for i in range(len(results_1)):
	points_1[i] = results_1[i].k
	steps_1[i] = results_1[i].h
	max_errors_1[i] = results_1[i].max_error
convergence_1 = np.zeros(len(points_1) - 1)

results_2 = np.array([a2, b2, c2, d2, e2, f2, g2, h2])

points_2 = np.empty_like(results_2)
steps_2 = np.empty_like(results_2)
max_errors_2 = np.empty_like(results_2)
for i in range(len(results_2)):
	points_2[i] = results_2[i].k
	steps_2[i] = results_2[i].h
	max_errors_2[i] = results_2[i].max_error
convergence_2 = np.zeros(len(points_2) - 1)

results_3 = np.array([a3, b3, c3, d3, e3, f3, g3, h3])
points_3 = np.empty_like(results_3)
steps_3 = np.empty_like(results_3)
max_errors_3 = np.empty_like(results_3)
for i in range(len(results_3)):
	points_3[i] = results_3[i].k
	steps_3[i] = results_3[i].h
	max_errors_3[i] = results_3[i].max_error
convergence_3 = np.zeros(len(points_3) - 1)

results_4 = np.array([a4, b4, c4, d4, e4, f4, g4, h4])
points_4 = np.empty_like(results_4)
steps_4 = np.empty_like(results_4)
max_errors_4 = np.empty_like(results_4)
for i in range(len(results_4)):
	points_4[i] = results_4[i].k
	steps_4[i] = results_4[i].h
	max_errors_4[i] = results_4[i].max_error
convergence_4 = np.zeros(len(points_4) - 1)

results_5 = np.array([a5, b5, c5, d5])
points_5 = np.empty_like(results_5)
steps_5 = np.empty_like(results_5)
max_errors_5 = np.empty_like(results_5)
for i in range(len(results_5)):
	points_5[i] = results_5[i].k
	steps_5[i] = results_5[i].h
	max_errors_5[i] = results_5[i].max_error
convergence_5 = np.zeros(len(points_5) - 1)

results_6 = np.array([a6, b6, c6, d6, e6])
points_6 = np.empty_like(results_6)
steps_6 = np.empty_like(results_6)
max_errors_6 = np.empty_like(results_6)
for i in range(len(results_6)):
	points_6[i] = results_6[i].k
	steps_6[i] = results_6[i].h
	max_errors_6[i] = results_6[i].max_error
convergence_6 = np.zeros(len(points_6) - 1)

results_7 = np.array([a7, b7, c7, d7])
points_7 = np.empty_like(results_7)
steps_7 = np.empty_like(results_7)
max_errors_7 = np.empty_like(results_7)
for i in range(len(results_7)):
	points_7[i] = results_7[i].k
	steps_7[i] = results_7[i].h
	max_errors_7[i] = results_7[i].max_error
convergence_7 = np.zeros(len(points_7) - 1)

results_8 = np.array([a8, b8, c8, d8])
points_8 = np.empty_like(results_8)
steps_8 = np.empty_like(results_8)
max_errors_8 = np.empty_like(results_8)
for i in range(len(results_8)):
	points_8[i] = results_8[i].k
	steps_8[i] = results_8[i].h
	max_errors_8[i] = results_8[i].max_error
convergence_8 = np.zeros(len(points_8) - 1)

plt.plot(points_1, max_errors_1, linestyle="dashed", marker='o', markersize=8, color='forestgreen')
plt.plot(points_2, max_errors_2, linestyle="dashed", marker='^', markersize=8, color='forestgreen')

plt.plot(points_3, max_errors_3, linestyle="dashed", marker='o', markersize=8, color='mediumorchid')
plt.plot(points_4, max_errors_4, linestyle="dashed", marker='^', markersize=8, color='mediumorchid')

plt.plot(points_5, max_errors_5, linestyle="dashed", marker='o', markersize=8, color='teal')
plt.plot(points_6, max_errors_6, linestyle="dashed", marker='^', markersize=8, color='teal')

plt.plot(points_7, max_errors_7, linestyle="dashed", marker='o', markersize=8, color='silver')
plt.plot(points_8, max_errors_8, linestyle="dashed", marker='^', markersize=8, color='silver')

# for i in range(len(points_1) - 1):
# 	convergence_1[i] = np.log(max_errors_1[i]/max_errors_1[i+1]) / np.log(steps_1[i]/steps_1[i+1])
# for i in range(len(points_2) - 1):	
# 	convergence_2[i] = np.log(max_errors_2[i]/max_errors_2[i+1]) / np.log(steps_2[i]/steps_2[i+1])
# for i in range(len(points_3) - 1):
# 	convergence_3[i] = np.log(max_errors_3[i]/max_errors_3[i+1]) / np.log(steps_3[i]/steps_3[i+1])
# for i in range(len(points_4) - 1):
# 	convergence_4[i] = np.log(max_errors_4[i]/max_errors_4[i+1]) / np.log(steps_4[i]/steps_4[i+1])
# for i in range(len(points_5) - 1):
# 	convergence_5[i] = np.log(max_errors_5[i]/max_errors_5[i+1]) / np.log(steps_5[i]/steps_5[i+1])
# for i in range(len(points_6) - 1):
# 	convergence_6[i] = np.log(max_errors_6[i]/max_errors_6[i+1]) / np.log(steps_6[i]/steps_6[i+1])

plt.xscale("log")
plt.yscale("log")
# plt.plot(points_1[1:], convergence_1, linestyle="dashed", marker='o', markersize=8, color='forestgreen')
# plt.plot(points_2[1:], convergence_2, linestyle="dashed", marker='^', markersize=8, color='forestgreen')
# # plt.text(points_1[-1] + 200, 4, r'$r=2$', fontsize='x-large', color='black')

# plt.plot(points_3[1:], convergence_3, linestyle="dashed", marker='o', markersize=8, color='mediumorchid')
# plt.plot(points_4[1:], convergence_4, linestyle="dashed", marker='^', markersize=8, color='mediumorchid')
# # plt.text(points_3[-1] + 100, 6, r'$r=3$', fontsize='x-large', color='red')

# plt.plot(points_5[1:], convergence_5, linestyle="dashed", marker='o', markersize=8, color='teal')
# plt.plot(points_6[1:], convergence_6, linestyle="dashed", marker='^', markersize=8, color='teal')
# plt.text(points_5[-1] + 25, 8, r'$r=4$', fontsize='x-large', color='navy')

#plt.ylabel(r'$r_{c,L_\infty}$', fontsize=14)
#plt.xlabel(r'$N$', fontsize=14)

# plt.xticks(fontsize=24)
# plt.yticks(fontsize=24)
# plt.grid(axis='y')
# # plt.plot(points_4[1:], convergence_4, linestyle="dashed", marker='o', color='forestgreen')
# # plt.text(points_4[-1] + 3, convergence_4[-1], r'$r=5$', fontsize='x-large', color='forestgreen')
# # plt.plot(points_5[1:], convergence_5, linestyle="dashed", marker='o', color='mediumorchid')
# # plt.text(points_5[-1] + 3, convergence_5[-1], r'$r=6$', fontsize='x-large', color='mediumorchid')
# # plt.plot(points_6[1:], convergence_6, linestyle="dashed", marker='o', color='teal')
# # plt.text(points_6[-1] + 3, convergence_6[-1], r'$r=7$', fontsize='x-large', color='teal')
plt.show()
# print(convergence_1)
