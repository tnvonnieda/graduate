import numpy as np
from numba import vectorize

# @vectorize
def get_derivative_coefficients(r): # referenced as d[i] for a given order where i = [0,2*r - 1]
	if r == 1:
		return [1]
	else:
		relevant_orders = []
		for i in range(r+1):
			if i % 2 != 0:
				relevant_orders.append(i)
		c = np.zeros([len(relevant_orders),len(relevant_orders)]) # taylor series coefficients of order r
		for i in range(len(relevant_orders)):
			for j in range(len(relevant_orders)):
				c[i,j] = c[i,j] + relevant_orders[j]**relevant_orders[i]  
		b = np.zeros(len(relevant_orders))
		b[0] = 1
		d = np.linalg.solve(c,b)
		
		return d
		