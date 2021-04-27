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

		self.gamma = 1.4
		self.boundary_type = boundary_type
		self.characteristic = characteristic
		self.Pi = 0.0
		self.tau = 0.0
		self.epsilon = 10.0**-40
		self.entropy = None

		if time_int == 'sdc':
			self.q = get_quadrature_weights(p)
			self.quad_points = get_quadrature_points(p)


		# SHU OSHER
		A_rho = 0.2
		kappa_rho = 5.0
		x_sw = -4
		conds = [self.x <= x_sw, self.x > x_sw]

		self.u_p = np.array([
			np.piecewise(self.x, conds, [lambda x: 27/7, lambda x: 1.0 + A_rho*np.sin(kappa_rho * x)]),
			np.piecewise(self.x, conds, [4*np.sqrt(35)/9, 0.0]),
			np.piecewise(self.x, conds, [31/3, 1.0])
			]).T

		self.u_p0 = self.u_p
		self.u_c0 = self.get_conservative_vars(self.u_p0)

		self.u_c = self.get_conservative_vars(self.u_p)
		# self.u_p0 = self.u_p
		# self.u_c0 = self.u_c
		self.num_vars = self.u_p.shape[1]
		self.time_int = time_int
		self.run()
		# self.PLOT_TYPE = REAL_TIME # REAL_TIME/END_TIME

	def get_characteristic_transform(self, u):
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

	# def weno_left(self):
	# 	u = np.pad(self.u_p, ((self.r, self.r-1), (0,0)), mode=self.boundary_type)
	# 	u_p_reconstructed = np.zeros(u.shape) 
	# 	P = np.zeros(np.append((self.r+1), u.shape))
	# 	beta = calculate_beta_no_characteristic(u, self.r, shift)

	# 	alpha = self.b/(beta+self.epsilon)**(self.r+1)
	# 	omega = alpha / alpha.sum(axis=0)

	# 	for k_s in range(self.r+1): # for each stencil
	# 		# calculate half point polynomial interpolation, P_{r,k_s,i+1/2}		
	# 		for l in range(self.r+1):
	# 			P[k_s,:,:] += self.a[k_s,l]*shift(u, k_s-l).reshape((len(u), self.num_vars, 1)).reshape((len(u), self.num_vars))	
	# 		u_p_reconstructed = u_p_reconstructed + omega[k_s]*P[k_s]

	# 	return u_p_reconstructed[self.r:-self.r]

	# def weno_right(self):
	# 	u = np.flip(np.pad(self.u_p, ((self.r-1, self.r), (0,0)), mode=self.boundary_type), axis=0)
	# 	u_p_reconstructed = np.zeros(u.shape) 
	# 	P = np.zeros(np.append((self.r+1), u.shape))
		
	# 	beta = calculate_beta_no_characteristic(u, self.r, shift)
	# 	alpha = self.b/(beta+self.epsilon)**(self.r+1)
	# 	omega = alpha / alpha.sum(axis=0)

	# 	for k_s in range(self.r+1):
	# 		for l in range(self.r+1):
	# 			P[k_s,:,:] += self.a[k_s,l]*shift(u, k_s-l).reshape((len(u), self.num_vars, 1)).reshape((len(u), self.num_vars))
	# 		u_p_reconstructed = u_p_reconstructed + omega[k_s]*P[k_s]
	# 	return np.flip(u_p_reconstructed, axis=0)[self.r:-self.r]

	def weno_characteristic_left(self):
		u = np.pad(self.u_p, ((self.r, self.r-1), (0,0)), mode=self.boundary_type)
		u_p_reconstructed = np.zeros(u.shape) 
		P = np.zeros(np.append((self.r+1), u.shape))
		Q, Q_inverse = self.get_characteristic_transform(u)

		beta = calculate_beta_characteristic(u, self.r, Q, Q_inverse, shift)

		alpha = self.b/(beta+self.epsilon)**2
		omega = alpha / alpha.sum(axis=0)

		# Mapped
		# alpha_m = omega*(self.b + self.b**2 - 3*self.b*omega + omega**2) / (self.b**2 + omega*(1 - 2*self.b))
		# omega_m = alpha_m / alpha_m.sum(axis=0)

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
		Q, Q_inverse = self.get_characteristic_transform(u)	
		
		beta = calculate_beta_characteristic(u, self.r, Q, Q_inverse, shift)
		# print(beta)
		# sys.exit()
		alpha = self.b/(beta+self.epsilon)**2
		omega = alpha / alpha.sum(axis=0)

		# alpha_m = omega*(self.b + self.b**2 - 3*self.b*omega + omega**2) / (self.b**2 + omega*(1 - 2*self.b))
		# omega_m = alpha_m / alpha_m.sum(axis=0)

		for k_s in range(self.r+1):
			for l in range(self.r+1):
				P[k_s,:,:] += self.a[k_s,l]*np.matmul(Q_inverse, shift(u, k_s-l).reshape((len(u), self.num_vars, 1))).reshape((len(u), self.num_vars))
			u_p_reconstructed = u_p_reconstructed + omega[k_s]*P[k_s]
		return np.flip((np.matmul(Q, u_p_reconstructed.reshape(len(u),self.num_vars,1))).reshape((len(u),self.num_vars)), axis=0)[self.r:-self.r]

	def weno_characteristic(self):

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

		flux_left = self.get_flux(u_p_reconstructed_l) + max_characteristic*u_c_reconstructed_l
		flux_right = self.get_flux(u_p_reconstructed_r) - max_characteristic*u_c_reconstructed_r
	
		return 1/2*(flux_left + flux_right)

	def get_dudx(self):
		u_split = np.pad(self.flux_split(), [(self.r+1, self.r+1), (0,0)], mode=self.boundary_type)
	
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

		k1 = -self.get_dudx()
		self.u_c = u_c + self.tau*1/2*k1
		self.u_p = self.get_primitive_vars(self.u_c)		
		
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
		self.calculate_entropy()
		self.elapsed_time = time.clock() - self.start_time
		print('done, time: ', self.elapsed_time)		

x0 = -5
xf = 5
tf = 0.1
N = 6401
CFL = 0.1
# problem_type = 'moving-shock'
characteristic = True

a = euler(x0, xf, tf, 6401, CFL, 7, None, characteristic, 'edge', 'rk4')
# b = euler(x0, xf, tf, 101, CFL, 7, 9, characteristic, 'edge', 'sdc')
# c = euler(x0, xf, tf, 201, CFL, 7, 9, characteristic, 'edge', 'sdc')
# d = euler(x0, xf, tf, 401, CFL, 7, 9, characteristic, 'edge', 'sdc')
# e = euler(x0, xf, tf, 801, CFL, 7, 9, characteristic, 'edge', 'sdc')
# f = euler(x0, xf, tf, 1601, CFL, 7, 9, characteristic, 'edge', 'sdc')
# plt.title("5th order 6401 no map")
plt.plot(a.x, a.u_p[:,0], linestyle='dashed', linewidth=2, color='black')
# plt.plot(b.x, b.u_p[:,0], linestyle='dashed', linewidth=2, color='mediumorchid')
# plt.plot(c.x, c.u_p[:,0], linestyle='dashed', linewidth=2, color='teal')
# plt.plot(d.x, d.u_p[:,0], linestyle='dashed', linewidth=2, color='navy')
# plt.plot(e.x, e.u_p[:,0], linestyle='dashed', linewidth=2, color='forestgreen')
# plt.plot(f.x, f.u_p[:,0], linestyle='dashed', linewidth=2, color='red')

plt.show()
