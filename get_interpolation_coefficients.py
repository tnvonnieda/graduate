import numpy as np
from fractions import Fraction
def get_interpolation_coefficients(r): # referenced as a[k_s,l] for a given order
	r = r+1
	a = np.zeros([r,r], dtype=np.longdouble)
	# # a_rational = np.empty([r,r])
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
			# a_rational[k_s,l] = Fraction.from_float(a[k_s, l]).limit_denominator(1000000)
		
	# a_rational = []
	# for i in range(len(d)):
	# 	d_rational.append(Fraction.from_float(d[i]).limit_denominator(1000000))
	# return d_rational
	
	# a = np.array([[5/16, 15/16, -5/16, 1/16], [-1/16, 9/16, 9/16, -1/16], [1/16, -5/16, 15/16, 5/16], [-5/16, 21/16, -35/16, 35/16]])

	return a