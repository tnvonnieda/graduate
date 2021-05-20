import numpy as np
import test_reconstruct as rec
import matplotlib.pyplot as plt

def shift(arr, num, fill_value=np.nan):
	result = np.empty_like(arr)
	if num > 0:
		result[:num] = fill_value
		result[num:] = arr[:-num]
	elif num < 0:
		result[num:] = fill_value
		result[:num] = arr[-num:]
	else:
		result[:] = arr
	return result

def get_transform(u):
	gamma = 1.4
	Pi = 0

	u = 1/2*(shift(u,-1) + u)

	c = np.sqrt(abs(gamma*(u[:,2] + Pi)/u[:,0]))

	Q = np.array([
		[np.ones(len(u)), np.ones(len(u)), np.ones(len(u))],
		[-c/u[:,0], 0*u[:,0], c/u[:,0]], 
		[c**2, 0*u[:,0], c**2]]).transpose((2,0,1))
	Q_inv = np.array([
		[0*u[:,0], -u[:,0]/(2*c), 1/(2*c**2)], 
		[np.ones(len(u)), np.zeros(len(u)), -1/(c**2)], 
		[np.zeros(len(u)), u[:,0]/(2*c), 1/(2*c**2)]]).transpose((2,0,1))
	
	return Q, Q_inv

class euler_system:

	def __init__(self, x, up0, h, CFL, r, time_scaling=False, pad_mode='reflect', reflect_type='odd', mapping=False, eigenspace=True, power='r', boundary_velocity=False):
		self.CFL = CFL
		self.h = h
		self.m = up.shape[1]
		self.N = up0.shape[0]
		self.up = self.up0 = up
		self.x = x
		self.t = 0
		self.set_interpolation_constants(r)
		self.uc = self.get_cons(self.up)
		self.time_scaling = time_scaling
		self.mapping = mapping
		self.pad_mode = pad_mode
		self.eigenspace = eigenspace
		self.boundary_velocity = boundary_velocity
		self.gamma = 1.4
		self.Pi = 0
		self.left_boundary = np.array([up0[0,0],0,up0[0,2]])
		self.right_boundary = np.array([up0[-1,0],0,up0[-1,2]])

		if power == 'r':
			self.power = r
		elif type(power == int):
			self.power = power
		else:
			raise ValueError("Failed")

	def set_interpolation_constants(self,r):
		self.r = r
		self.a_vals = rec.get_interpolation_weights(r)
		self.b_vals = (rec.get_optimal_weights(r)).reshape((r,1,1))
		self.d_vals = rec.get_derivative_weights(2*r)

	def f(self,u):
		Pi = 0
		gamma = 1.4
		e = (u[:,2] + gamma*Pi) / (u[:,0]*(gamma - 1))

		return np.array([
			u[:,0]*u[:,1], 
			u[:,0]*u[:,1]**2+u[:,2], 
			(u[:,0]*e+1/2*u[:,0]*u[:,1]**2+u[:,2])*u[:,1]]).T
		
	def set_alpha(self, u):
		Pi = 0
		gamma = 1.4
		return (np.sqrt(abs(gamma*(u[:,2]+Pi)/u[:,0])) + abs(u[:,1])).max()

	def get_prim(self, uc):
		Pi = 0
		gamma = 1.4

		e = uc[:,2]/uc[:,0] - 1/2*(uc[:,1]/uc[:,0])**2

		return np.array([
			uc[:,0],
			uc[:,1]/uc[:,0],
			(gamma-1)*uc[:,0]*e - gamma*Pi]).T

	def get_cons(self, up):
		Pi = 0
		gamma = 1.4

		e=(up[:,2]+gamma*Pi)/(up[:,0]*(gamma-1))
		return np.array([
			up[:,0], 
			up[:,0]*up[:,1], 
			up[:,0]*(e+1/2*up[:,1]**2)]).T

	def characteristic_weno(self):
		# if self.boundary_velocity:
		# 	left_boundary = np.tile(self.left_boundary, (r,1))
		# 	right_boundary = np.tile(self.right_boundary, (r,1))
		# 	u = np.vstack((self.left_boundary, self.up, self.right_boundary))
		# 	# u = np.pad(self.up,((self.r,self.r),(0,0)),
		# 	# 	mode='constant',
		# 	# 	constant_values=(((self.up0[0,0],0,self.up0[0,2]),(self.up0[-1,0],0,self.up0[-1,2])),(0,0)))
		# else:
		u = np.pad(self.up,((self.r, self.r),(0,0)),mode='reflect', reflect_type='odd')
			
		P = np.zeros(np.append((self.r+1),u.shape))
		Q,Q_inv = get_transform(u)

		beta = rec.get_beta(u,self.r,get_transform)
		beta = self.b_vals/(beta+10**(-40))**self.power

		omega = beta/beta.sum(axis=0)

		if self.mapping:
			alpha_star = omega*(self.b_vals + self.b_vals**2 - 3*self.b_vals*omega + omega**2)/\
				(self.b_vals**2 + omega*(1-2*self.b_vals))
			omega = alpha_star / alpha_star.sum(axis=0)

		for k in range(0,self.r):
			for l in range(0,self.r):
				P[l+1,:,:] = self.a_vals[k][l]*(np.matmul(Q_inv,\
					shift(u,k-l).reshape((len(u),self.m,1))).\
					reshape((len(u),self.m)))
			P[0,:,:] = P[0,:,:] + omega[k]*(P[1:,:,:].sum(axis=0))
		return (np.matmul(Q,P[0,:].reshape(len(u),self.m,1))).\
			reshape((len(u),self.m))[self.r:-self.r,:]

	def characteristic_weno_right(self):
		# if self.boundary_velocity:
		# 	left_boundary = np.tile(self.left_boundary, (r,1))
		# 	right_boundary = np.tile(self.right_boundary, (r+1,1))
		# 	# print(left_boundary)
		# 	# print(right_boundary)
		# 	# sys.exit()
		# 	u = np.vstack((self.left_boundary, self.up, self.right_boundary))
		# 	# u = np.pad(self.up,((self.r,self.r+1),(0,0)),
		# 	# 	mode='constant',
		# 	# 	constant_values=(((self.up0[0,0],0,self.up0[0,2]),(self.up0[-1,0],0,self.up0[-1,2])),(0,0)))
		# else:
		u = np.pad(self.up,((self.r, self.r+1),(0,0)),mode='reflect', reflect_type='odd')
		u = np.flip(u,axis=0)
		u = shift(u,1)
		# print()
		P = np.zeros(np.append((self.r+1),u.shape))
		Q,Q_inv = get_transform(u)

		beta = rec.get_beta(u,self.r,get_transform)
		beta = self.b_vals/(beta+10**(-40))**self.power

		omega = beta/beta.sum(axis=0)

		if self.mapping:
			alpha_star = omega*(self.b_vals + self.b_vals**2 - 3*self.b_vals*omega + omega**2)/\
				(self.b_vals**2 + omega*(1-2*self.b_vals))
			omega = alpha_star / alpha_star.sum(axis=0)

		for k in range(0,self.r):
			for l in range(0,self.r):
				P[l+1,:,:] = self.a_vals[k][l]*(np.matmul(Q_inv,\
					shift(u,k-l).reshape((len(u),self.m,1))).\
					reshape((len(u),self.m)))
			P[0,:,:] = P[0,:,:] + omega[k]*(P[1:,:,:].sum(axis=0))
		return np.flip((np.matmul(Q,P[0,:].reshape(len(u),self.m,1))).\
			reshape((len(u),self.m)),axis=0)[self.r:-self.r-1,:]

	def higher_order_lf_split(self, optimal=False):
		alpha = self.set_alpha(self.up)

		if optimal:
			P_l = self.linear_optimal()
			P_r = self.linear_optimal_right()

		elif self.eigenspace:
			P_l = self.characteristic_weno()
			P_r = self.characteristic_weno_right()
		else:
			P_l = self.weno()
			P_r = self.weno_right()
		# plt.plot(self.x, P_r)
		# plt.show()
		# sys.exit()

		C_l = self.get_cons(P_l)
		C_r = self.get_cons(P_r)

		return np.array([
			1/2*(self.f(P_l)+alpha*C_l),
			1/2*(self.f(P_r)-alpha*C_r)]).sum(axis=0)

	def spatial_grad(self, optimal=False):
		u_split = self.higher_order_lf_split(optimal)
		u_split = np.pad(u_split, [(self.r,self.r),(0,0)], mode='reflect', reflect_type='odd')

		grad = np.zeros(np.append((len(self.d_vals)),u_split.shape))

		q = 0
		for d in self.d_vals:
			grad[q,:,:] = d*(shift(u_split,-q) - shift(u_split,q+1))
			q+=1
		return (grad.sum(axis=0)/self.h)[self.r:-self.r,:]

	def RK4(self, optimal=False):
		self.set_tau(4)
		# print(self.tau)
		# sys.exit()
		uc = self.uc
		
		k1 = -self.spatial_grad(optimal)

		self.uc = self.uc + self.tau*1/2*k1
		self.up = self.get_prim(self.uc)

		k2 = -self.spatial_grad(optimal)
		self.uc = uc + self.tau*1/2*k2
		self.up = self.get_prim(self.uc)
		
		k3 = -self.spatial_grad(optimal)
		self.uc = uc + self.tau*k3
		self.up = self.get_prim(self.uc)
		
		k4 = -self.spatial_grad(optimal)
		self.uc = uc + self.tau*(1/6*k1+1/3*k2+1/3*k3+1/6*k4)
		self.up = self.get_prim(self.uc)
		# print(self.tau + self.t)
		# sys.exit()
		self.t = self.t + self.tau
		# print(self.t)

	def set_tau(self,t_order):
		if self.time_scaling:
			self.tau = self.CFL*(self.h/((abs(self.set_alpha(self.up)))))**((2*self.r-1)/(t_order))
		else:
			self.tau = self.CFL*self.h/((abs(self.set_alpha(self.up))))

	def run(self):
		# print(self.uc)
		# sys.exit()
		# plt.ion()
		# fig = plt.figure(1)
		# ax = fig.add_subplot(111)
		# line1, = ax.plot(self.x,self.up[:,0],'r-')
		# line2, = ax.plot(self.x,self.up[:,1],'b-')
		# line3, = ax.plot(self.x,self.up[:,2],'g-')
		while self.t < 0.038:
			self.RK4()
			# plt.plot(self.x, self.up)
			# plt.show()
			# line1.set_ydata(self.up[:,0])
			# line2.set_ydata(self.up[:,1])
			# line3.set_ydata(self.up[:,2])
			# fig.canvas.draw()
			# fig.canvas.flush_events()
			# print(self.t)
			# print(self.up)
			# sys.exit()
			# print(self.t)
			# plt.plot(self.x,self.up)
			# plt.show()
			# sys.exit()
		return self.up
		# self.calculate_entropy()
		# self.elapsed_time = time.clock() - self.start_time
		# print('done, time: ', self.elapsed_time)

# DEFINE SYSTEM PROPS
x0 = 0
xf = 1
N = 501
CFL = 0.1
r = 2
x = np.linspace(x0,xf,N)
h = (xf - x0)/(N-1)
conds = [(x >= 0) & (x < 0.1), (x >= 0.1) & (x < 0.9), x >= 0.9]
up = np.array([
	np.piecewise(x, conds, [1.0, 1.0, 1.0]),
	np.piecewise(x, conds, [0.0, 0.0, 0.0]),
	np.piecewise(x, conds, [1000, 0.01, 100.0])
	]).T

euler = euler_system(x, up, h, CFL, r, time_scaling=False, pad_mode='reflect', reflect_type='odd', mapping=True, eigenspace=True, power='r', boundary_velocity=False)
up = euler.run()
plt.plot(x, up[:,0])
plt.show()
# euler.run()





