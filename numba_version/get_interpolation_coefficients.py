import numpy as np
from numba import vectorize

# @vectorize
def get_interpolation_coefficients(r): # referenced as a[k_s,l] for a given order
	r = r+1
	a = np.empty([r,r])
	for k_s in range(r): # for each stencil
		for l in range(r):
			a[k_s,l] = 1			
			for i in range(r-1):
				a_top = 1/2+i-k_s
				if i < l:
					a_top = a_top - 1 
				a_bottom = i - l + 1
				if a_bottom <= 0:
					a_bottom = a_bottom - 1
				a[k_s,l] = a[k_s,l]*a_top/a_bottom
	return a