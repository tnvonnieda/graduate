import numpy as np
import pandas as pd
import scipy.special as sp
# from numba import jit
# import sys
# print(sys.version)

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

def get_interpolation_weights(r: 'error order', k_s=None):
	R = r-1
	if k_s == None:
		k_s = np.arange(0,r,1)
	else:
		k_s = np.array([k_s])
	k_s = R - k_s

	results = np.zeros([len(k_s),r])
	i = 0
	for k in k_s:
		q = np.arange(0,r,1)
		denominators = (-1)**q*sp.factorial(q)*sp.factorial(R-q)
		numerators = np.zeros(r)
		for p in q:
			numerators[p] = np.prod(1/2-k+q[(q!=p)])
		results[i] = np.flip(numerators/denominators, 0)
		i+=1
	return results

def get_derivative_weights(n: 'Approximation Order'):
	if n < 2:
		n = 2
	elif (n%2)!=0:
		n = n+1
	r = n/2
	k_max = 2*r - 1
	a = np.arange(1,k_max+1,2)
	A = [a]
	for power in a[1:]:
		A = np.append(A, [a**power], axis=0)
	b = np.zeros(int(r))
	b[0] = 1
	return np.linalg.solve(A,b)

def get_optimal_weights(r):
	A = get_interpolation_weights(r)
	b = get_interpolation_weights(2*r-1,r-1)[0]
	b = np.flip(np.append(b[:(r-1)],b[-1]))

	C = np.zeros([r,r])
	C[0,0] = A[0,1]
	for diag in list(range(1,r,1)):
		C[diag] = np.pad(np.diagonal(A,-diag), pad_width=(diag,0), mode='constant')
	return np.linalg.solve(C,b)

def poly(x: 'x values', y: 'y values', r: 'error order', k: 'stencil'):
	df = pd.DataFrame({'x': x, 'y': y})
	coeffs = get_interpolation_weights(r,k)[0]

	for l in np.arange(0,r,1):
		df[l] = coeffs[l]*np.roll(df.y,k-1)

	return pd.Series(df.iloc[:,2:].sum(axis=1,skipna=False))

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



def get_beta2(f, f_left, f_right):
	beta = np.zeros((2,len(f),3))
	beta[1] = (f-f_left[0])**2
	beta[0] = (f_right[0] - f)**2
	return beta

def get_beta3(f, f_left, f_right):
	beta = np.zeros((3,len(f),3))
	beta[2] = 13/12*(f_left[1] - 2*f_left[0] + f)**2 + \
		1/4*(f_left[1] - 4*f_left[0] + 3*f)**2
	
	beta[1] = 13/12*(f_left[0] - 2*f + f_right[0])**2 + \
		1/4*(f_left[0] - f_right[0])**2

	beta[0] = 13/12*(f- 2*f_right[0] + f_right[1])**2 + \
		1/4*(3*f - 4*f_right[0] + f_right[1])**2

	return beta


def get_beta(f,r,get_transform):
	Q, Q_inv = get_transform(f)
	f, f_right, f_left = apply_transform(Q_inv, f.reshape((len(f),3,1)), r)

	if r == 2:
		beta = get_beta2(f, f_right, f_left)

	elif r == 3:
		beta = get_beta3(f, f_right, f_left)

	return beta

def apply_transform(Q_inv, f, r):
	f_left = np.zeros(np.insert(f.shape,0,r-1))
	f_right = np.zeros(np.insert(f.shape,0,r-1))
	for i in range(r-1):
		f_left[i] = np.matmul(Q_inv,shift(f,i+1))
		f_right[i] = np.matmul(Q_inv,shift(f,-i-1))

	f = np.matmul(Q_inv,f).reshape(len(f),3)
	f_left = f_left.reshape(f_left.shape[:-1])
	f_right = f_right.reshape(f_right.shape[:-1])
	return f, f_left, f_right