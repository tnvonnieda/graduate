import numpy as np
import math

def calculate_convergence_rate(error_array, step_size_array):
	r_c = np.zeros(len(error_array))
	for i in range(1, len(error_array)):
		r_c[i] = math.log(error_array[i]/error_array[i-1])/math.log(step_size_array[i]/step_size_array[i-1])

	return r_c
