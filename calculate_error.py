import numpy as np

# Takes 2 1-dimensional arrays and calculates L1, L2, and L_infinity norms of error
def calculate_error(u_calculated, u_exact):
	error = u_calculated - u_exact
	num_points = len(u_calculated)

	L_1_error = np.sum(abs(error))/num_points
	L_2_error = np.sqrt(np.sum(error**2))/num_points
	L_inf_error = np.max(abs(error))

	return L_1_error, L_2_error, L_inf_error