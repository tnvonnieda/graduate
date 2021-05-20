from get_interpolation_coefficients import get_interpolation_coefficients
import numpy as np
import math 
from numba import vectorize

# @vectorize
def get_optimal_weights(r): # referenced as b[k_s] for a given order
	a1 = get_interpolation_coefficients(r)
	a2 = get_interpolation_coefficients(2*r)
	r = r+1
	b = np.zeros(r)	
	for m in range(0,int(math.ceil(r/2))):
		b_top_1 = a2[r-1,2*r-2-m]
		b_top_2 = a2[r-1,m]
		b_bottom_1 = a1[m,r-1]		
		b_bottom_2 = a1[r-1-m,0]
		for i in range(m):
			b_top_1 = b_top_1 - b[i]*a1[i,r-m-1+i]
			b_top_2 = b_top_2 - b[r-1-i]*a1[r-1-i,m-i]
		b[m] = b_top_1 / b_bottom_1
		b[r-1-m] = b_top_2 / b_bottom_2
	return b