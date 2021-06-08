from numba import jit
import numpy as np
import sys
from smoothness_13 import sigma_6
from smoothness_15 import sigma_7
from smoothness_17 import sigma_8

def calculate_beta_characteristic(u, r, Q, Q_inverse, shift):
	u, u_minus, u_plus = apply_transform(Q_inverse, u.reshape((len(u),u.shape[1],1)), r, shift)
	if r == 0:
		return calculate_beta_0(u, u_minus, u_plus)
	elif r == 1:
		return calculate_beta_1(u, u_minus, u_plus)
	elif r == 2:
		return calculate_beta_2(u, u_minus, u_plus)
	elif r == 3:
		return calculate_beta_3(u, u_minus, u_plus)
	elif r == 4:
		return calculate_beta_4(u, u_minus, u_plus)
	elif r == 5:
		return calculate_beta_5(u, u_minus, u_plus)
	elif r == 6:
		return calculate_beta_6(u, u_minus, u_plus)
	elif r == 7:
		return calculate_beta_7(u, u_minus, u_plus)
	elif r == 8:
		return calculate_beta_8(u, u_minus, u_plus)

def calculate_beta_no_characteristic(u, r, shift):
	u_minus = np.empty(np.insert(u.shape, 0, r))
	u_plus = np.empty(np.insert(u.shape, 0, r))
	for i in range(r):
		u_minus[i] = shift(u, i+1)
		u_plus[i] = shift(u, -i-1)
	if r == 0:
		return calculate_beta_0(u, u_minus, u_plus)
	elif r == 1:
		return calculate_beta_1(u, u_minus, u_plus)
	elif r == 2:
		return calculate_beta_2(u, u_minus, u_plus)
	elif r == 3:
		return calculate_beta_3(u, u_minus, u_plus)
	elif r == 4:
		return calculate_beta_4(u, u_minus, u_plus)
	elif r == 5:
		return calculate_beta_5(u, u_minus, u_plus)
	elif r == 6:
		return calculate_beta_6(u, u_minus, u_plus)
	elif r == 7:
		return calculate_beta_7(u, u_minus, u_plus)
	elif r == 8:
		return calculate_beta_8(u, u_minus, u_plus)

def apply_transform(Q_inverse, u, r, shift):
	u_minus = np.empty(np.insert(u.shape, 0, r)) 
	u_plus = np.empty(np.insert(u.shape, 0, r))
	
	for i in range(r):
		u_minus[i] = np.matmul(Q_inverse, shift(u, i+1))
		u_plus[i] = np.matmul(Q_inverse, shift(u, -i-1))

	u = np.matmul(Q_inverse, u).reshape(len(u), u.shape[1])
	u_minus = u_minus.reshape(u_minus.shape[:-1]) 
	u_plus = u_plus.reshape(u_plus.shape[:-1])
	
	return u, u_minus, u_plus

def calculate_beta_0(u, u_minus, u_plus):
	# beta = np.empty((1, len(u), len(u[0])))

	beta = np.array([1])
	return beta

def calculate_beta_1(u, u_minus, u_plus):
	beta = np.empty((2, len(u), len(u[0])))
	beta[0] = (u_plus[0]-u)**2
	beta[1] = (u-u_minus[0])**2
	return beta

# @jit(nopython=True)
def calculate_beta_2(u, u_minus, u_plus):
	beta = np.empty((3, len(u), len(u[0]))) 
	beta[0] = 13/12*(u-2*u_plus[0]+u_plus[1])**2 + 1/4*(3*u-4*u_plus[0]+u_plus[1])**2
	beta[1] = 13/12*(u_minus[0]-2*u+u_plus[0])**2 + 1/4*(u_minus[0]-u_plus[0])**2
	beta[2] = 13/12*(u_minus[1]-2*u_minus[0]+u)**2 + 1/4*(u_minus[1]-4*u_minus[0]+3*u)**2
	return beta	

# @jit(nopython=True, parallel=True)
def calculate_beta_3(u, u_minus, u_plus):
	beta = np.empty((4, len(u), len(u[0])), dtype=np.longdouble)

	# beta[0] = u*(2107*u-9402*u_plus[0]+7042*u_plus[1]-1854*u_plus[2]) + \
	# 	u_plus[0]*(11003*u_plus[0]-17246*u_plus[1]+4642*u_plus[2]) + \
	# 	u_plus[1]*(7043*u_plus[1]-3882*u_plus[2]) + \
	# 	547*u_plus[2]**2

	# beta[1] = u_minus[0]*(547*u_minus[0]-2522*u+1922*u_plus[0]-494*u_plus[1]) + \
	# 	u*(3443*u-5966*u_plus[0]+1602*u_plus[1]) + \
	# 	u_plus[0]*(2843*u_plus[0]-1642*u_plus[1]) + \
	# 	267*u_plus[1]**2

	# beta[2] = u_minus[1]*(267*u_minus[1]-1642*u_minus[0]+1602*u-494*u_plus[0]) + \
	# 	u_minus[0]*(2843*u_minus[0]-5966*u+1922*u_plus[0]) + \
	# 	u*(3443*u-2522*u_plus[0]) + \
	# 	547*u_plus[0]**2

	# beta[3] = u_minus[2]*(547*u_minus[2]-3882*u_minus[1]+4642*u_minus[0]-1854*u) + \
	# 	u_minus[1]*(7043*u_minus[1]-17246*u_minus[0]+7042*u) + \
	# 	u_minus[0]*(11003*u_minus[0]-9402*u) + \
	# 	2107*u**2

	beta[3] = u_minus[2]*(547*u_minus[2]-3882*u_minus[1]+4642*u_minus[0]-1854*u) + \
		u_minus[1]*(7043*u_minus[1]-17246*u_minus[0]+7042*u) + \
		u_minus[0]*(11003*u_minus[0]-9402*u)+2107*u**2
	beta[2] = u_minus[1]*(267*u_minus[1]-1642*u_minus[0]+1602*u-494*u_plus[0]) + \
    	u_minus[0]*(2843*u_minus[0]-5966*u+1922*u_plus[0]) + \
    	u*(3443*u-2522*u_plus[0])+547*u_plus[0]**2
	beta[1] = u_minus[0]*(547*u_minus[0]-2522*u+1922*u_plus[0]-494*u_plus[1]) + \
    	u*(3443*u-5966*u_plus[0]+1602*u_plus[1]) + \
    	u_plus[0]*(2843*u_plus[0]-1642*u_plus[1])+267*u_plus[1]**2
	beta[0] = u*(2107*u-9402*u_plus[0]+7042*u_plus[1]-1854*u_plus[2]) + \
    	u_plus[0]*(11003*u_plus[0]-17246*u_plus[1]+4642*u_plus[2]) + \
    	u_plus[1]*(7043*u_plus[1]-3882*u_plus[2])+547*u_plus[2]**2

	return beta		

@jit(nopython=True, parallel=True)
def calculate_beta_4(u, u_minus, u_plus):
	beta = np.empty((5, len(u), len(u[0])), dtype=np.longdouble)

	beta[0] = u*(107918*u-649501*u_plus[0]+758823*u_plus[1]-411487*u_plus[2]+86329*u_plus[3]) + \
		u_plus[0]*(1020563*u_plus[0]-2462076*u_plus[1]+1358458*u_plus[2]-288007*u_plus[3]) + \
		u_plus[1]*(1521393*u_plus[1]-1704396*u_plus[2]+364863*u_plus[3]) + \
		u_plus[2]*(482963*u_plus[2]-208501*u_plus[3]) + \
		22658*u_plus[3]**2

	beta[1] = u_minus[0]*(22658*u_minus[0]-140251*u+165153*u_plus[0]-88297*u_plus[1]+18079*u_plus[2]) + \
		u*(242723*u-611976*u_plus[0]+337018*u_plus[1]-70237*u_plus[2]) + \
		u_plus[0]*(406293*u_plus[0]-464976*u_plus[1]+99213*u_plus[2]) + \
		u_plus[1]*(138563*u_plus[1]-60871*u_plus[2]) + \
		6908*u_plus[2]**2

	beta[2] = u_minus[1]*(6908*u_minus[1]-51001*u_minus[0]+67923*u-38947*u_plus[0]+8209*u_plus[1]) + \
		u_minus[0]*(104963*u_minus[0]-299076*u+179098*u_plus[0]-38947*u_plus[1]) + \
		u*(231153*u-299076*u_plus[0]+67923*u_plus[1]) + \
		u_plus[0]*(104963*u_plus[0]-51001*u_plus[1]) + \
		6908*u_plus[1]**2

	beta[3] = u_minus[2]*(6908*u_minus[2]-60871*u_minus[1]+99213*u_minus[0]-70237*u+18079*u_plus[0]) + \
		u_minus[1]*(138563*u_minus[1]-464976*u_minus[0]+337018*u-88297*u_plus[0]) + \
		u_minus[0]*(406293*u_minus[0]-611976*u+165153*u_plus[0]) + \
		u*(242723*u-140251*u_plus[0]) + \
		22658*u_plus[0]**2

	beta[4] = u_minus[3]*(22658*u_minus[3]-208501*u_minus[2]+364863*u_minus[1]-288007*u_minus[0]+86329*u) + \
		u_minus[2]*(482963*u_minus[2]-1704396*u_minus[1]+1358458*u_minus[0]-411487*u) + \
		u_minus[1]*(1521393*u_minus[1]-2462076*u_minus[0]+758823*u) + \
		u_minus[0]*(1020563*u_minus[0]-649501*u) + \
		107918*u**2
	return beta

@jit(nopython=True, parallel=True)
def calculate_beta_5(u, u_minus, u_plus):
	beta = np.empty((6, len(u), len(u[0])), dtype=np.longdouble)

	beta[0] = u*(6150211*u-47460464*u_plus[0]+76206736*u_plus[1]-63394124*u_plus[2]+27060170*u_plus[3]-4712740*u_plus[4]) + \
		u_plus[0]*(94851237*u_plus[0]-311771244*u_plus[1]+262901672*u_plus[2]-113206788*u_plus[3]+19834350*u_plus[4]) + \
		u_plus[1]*(260445372*u_plus[1]-444003904*u_plus[2]+192596472*u_plus[3]-33918804*u_plus[4]) + \
		u_plus[2]*(190757572*u_plus[2]-166461044*u_plus[3]+29442256*u_plus[4]) + \
		u_plus[3]*(36480687*u_plus[3]-12950184*u_plus[4]) + \
		1152561*u_plus[4]**2

	beta[1] = u_minus[0]*(1152561*u_minus[0]-9117992*u+14742480*u_plus[0]-12183636*u_plus[1]+5134574*u_plus[2]-880548*u_plus[3]) + \
		u*(19365967*u-65224244*u_plus[0]+55053752*u_plus[1]-23510468*u_plus[2]+4067018*u_plus[3]) + \
		u_plus[0]*(56662212*u_plus[0]-97838784*u_plus[1]+42405032*u_plus[2]-7408908*u_plus[3]) + \
		u_plus[1]*(43093692*u_plus[1]-37913324*u_plus[2]+6694608*u_plus[3]) + \
		u_plus[2]*(8449957*u_plus[2]-3015728*u_plus[3]) + \
		271779*u_plus[3]**2

	beta[2] = u_minus[1]*(271779*u_minus[1]-2380800*u_minus[0]+4086352*u-3462252*u_plus[0]+1458762*u_plus[1]-245620*u_plus[2]) + \
		u_minus[0]*(5653317*u_minus[0]-20427884*u+17905032*u_plus[0]-7727988*u_plus[1]+1325006*u_plus[2]) + \
		u*(19510972*u-35817664*u_plus[0]+15929912*u_plus[1]-2792660*u_plus[2]) + \
		u_plus[0]*(17195652*u_plus[0]-15880404*u_plus[1]+2863984*u_plus[2]) + \
		u_plus[1]*(3824847*u_plus[1]-1429976*u_plus[2]) + \
		139633*u_plus[2]**2

	beta[3] = u_minus[2]*(139633*u_minus[2]-1429976*u_minus[1]+2863984*u_minus[0]-2792660*u+1325006*u_plus[0]-245620*u_plus[1]) + \
		u_minus[1]*(3824847*u_minus[1]-15880404*u_minus[0]+15929912*u-7727988*u_plus[0]+1458762*u_plus[1]) + \
		u_minus[0]*(17195652*u_minus[0]-35817664*u+17905032*u_plus[0]-3462252*u_plus[1]) + \
		u*(19510972*u-20427884*u_plus[0]+4086352*u_plus[1]) + \
		u_plus[0]*(5653317*u_plus[0]-2380800*u_plus[1]) + \
		271779*u_plus[1]**2

	beta[4] = u_minus[3]*(271779*u_minus[3]-3015728*u_minus[2]+6694608*u_minus[1]-7408908*u_minus[0]+4067018*u-880548*u_plus[0]) + \
		u_minus[2]*(8449957*u_minus[2]-37913324*u_minus[1]+42405032*u_minus[0]-23510468*u+5134574*u_plus[0]) + \
		u_minus[1]*(43093692*u_minus[1]-97838784*u_minus[0]+55053752*u-12183636*u_plus[0]) + \
		u_minus[0]*(56662212*u_minus[0]-65224244*u+14742480*u_plus[0]) + \
		u*(19365967*u-9117992*u_plus[0]) + \
		1152561*u_plus[0]**2

	beta[5] = u_minus[4]*(1152561*u_minus[4]-12950184*u_minus[3]+29442256*u_minus[2]-33918804*u_minus[1]+19834350*u_minus[0]-4712740*u) + \
		u_minus[3]*(36480687*u_minus[3]-166461044*u_minus[2]+192596472*u_minus[1]-113206788*u_minus[0]+27060170*u) + \
		u_minus[2]*(190757572*u_minus[2]-444003904*u_minus[1]+262901672*u_minus[0]-63394124*u) + \
		u_minus[1]*(260445372*u_minus[1]-311771244*u_minus[0]+76206736*u) + \
		u_minus[0]*(94851237*u_minus[0]-47460464*u) + \
		6150211*u**2

	return beta

# def calculate_beta_6(u, r, Q_inverse):
# 	beta = np.zeros((r+1,3))
# 	for k_s in range(r+1):
# 		for l in range(r+1):
# 			for m in range(l+1):
# 				beta[k_s] += sigma_6[k_s,l,m]*np.dot(Q_inverse, u[2*r-k_s-l])*np.dot(Q_inverse, u[2*r-k_s-m])
# 	return beta

@jit(nopython=True, parallel=True)
def calculate_beta_6(u, u_minus, u_plus):
	beta = np.empty((7, len(u), len(u[0])), dtype=np.longdouble)
	
	beta[0] = sigma_6[0,0,0]*u_plus[5]**2 + \
		sigma_6[0,1,0]*u_plus[4]*u_plus[5] + sigma_6[0,1,1]*u_plus[4]**2 + \
		sigma_6[0,2,0]*u_plus[3]*u_plus[5] + sigma_6[0,2,1]*u_plus[3]*u_plus[4] + sigma_6[0,2,2]*u_plus[3]**2 + \
		sigma_6[0,3,0]*u_plus[2]*u_plus[5] + sigma_6[0,3,1]*u_plus[2]*u_plus[4] + sigma_6[0,3,2]*u_plus[2]*u_plus[3] + sigma_6[0,3,3]*u_plus[2]**2 + \
		sigma_6[0,4,0]*u_plus[1]*u_plus[5] + sigma_6[0,4,1]*u_plus[1]*u_plus[4] + sigma_6[0,4,2]*u_plus[1]*u_plus[3] + sigma_6[0,4,3]*u_plus[1]*u_plus[2] + sigma_6[0,4,4]*u_plus[1]**2 + \
		sigma_6[0,5,0]*u_plus[0]*u_plus[5] + sigma_6[0,5,1]*u_plus[0]*u_plus[4] + sigma_6[0,5,2]*u_plus[0]*u_plus[3] + sigma_6[0,5,3]*u_plus[0]*u_plus[2] + sigma_6[0,5,4]*u_plus[0]*u_plus[1] + sigma_6[0,5,5]*u_plus[0]**2 + \
		sigma_6[0,6,0]*u*u_plus[5]         + sigma_6[0,6,1]*u*u_plus[4]         + sigma_6[0,6,2]*u*u_plus[3]         + sigma_6[0,6,3]*u*u_plus[2]         + sigma_6[0,6,4]*u*u_plus[1]         + sigma_6[0,6,5]*u*u_plus[0]  + sigma_6[0,6,6]*u**2

	beta[1] = sigma_6[1,0,0]*u_plus[4]**2 + \
		sigma_6[1,1,0]*u_plus[3]*u_plus[4]  + sigma_6[1,1,1]*u_plus[3]**2 + \
		sigma_6[1,2,0]*u_plus[2]*u_plus[4]  + sigma_6[1,2,1]*u_plus[2]*u_plus[3]  + sigma_6[1,2,2]*u_plus[2]**2 + \
		sigma_6[1,3,0]*u_plus[1]*u_plus[4]  + sigma_6[1,3,1]*u_plus[1]*u_plus[3]  + sigma_6[1,3,2]*u_plus[1]*u_plus[2]  + sigma_6[1,3,3]*u_plus[1]**2 + \
		sigma_6[1,4,0]*u_plus[0]*u_plus[4]  + sigma_6[1,4,1]*u_plus[0]*u_plus[3]  + sigma_6[1,4,2]*u_plus[0]*u_plus[2]  + sigma_6[1,4,3]*u_plus[0]*u_plus[1]  + sigma_6[1,4,4]*u_plus[0]**2 + \
		sigma_6[1,5,0]*u*u_plus[4]          + sigma_6[1,5,1]*u*u_plus[3]          + sigma_6[1,5,2]*u*u_plus[2]          + sigma_6[1,5,3]*u*u_plus[1]          + sigma_6[1,5,4]*u*u_plus[0]          + sigma_6[1,5,5]*u**2 + \
		sigma_6[1,6,0]*u_minus[0]*u_plus[4] + sigma_6[1,6,1]*u_minus[0]*u_plus[3] + sigma_6[1,6,2]*u_minus[0]*u_plus[2] + sigma_6[1,6,3]*u_minus[0]*u_plus[1] + sigma_6[1,6,4]*u_minus[0]*u_plus[0] + sigma_6[1,6,5]*u_minus[0]*u + sigma_6[1,6,6]*u_minus[0]**2	

	beta[2] = sigma_6[2,0,0]*u_plus[3]**2 + \
		sigma_6[2,1,0]*u_plus[2]*u_plus[3]  + sigma_6[2,1,1]*u_plus[2]**2 + \
		sigma_6[2,2,0]*u_plus[1]*u_plus[3]  + sigma_6[2,2,1]*u_plus[1]*u_plus[2]  + sigma_6[2,2,2]*u_plus[1]**2 + \
		sigma_6[2,3,0]*u_plus[0]*u_plus[3]  + sigma_6[2,3,1]*u_plus[0]*u_plus[2]  + sigma_6[2,3,2]*u_plus[0]*u_plus[1]  + sigma_6[2,3,3]*u_plus[0]**2 + \
		sigma_6[2,4,0]*u*u_plus[3]          + sigma_6[2,4,1]*u*u_plus[2]          + sigma_6[2,4,2]*u*u_plus[1]          + sigma_6[2,4,3]*u*u_plus[0]          + sigma_6[2,4,4]*u**2 + \
		sigma_6[2,5,0]*u_minus[0]*u_plus[3] + sigma_6[2,5,1]*u_minus[0]*u_plus[2] + sigma_6[2,5,2]*u_minus[0]*u_plus[1] + sigma_6[2,5,3]*u_minus[0]*u_plus[0] + sigma_6[2,5,4]*u_minus[0]*u + sigma_6[2,5,5]*u_minus[0]**2 + \
		sigma_6[2,6,0]*u_minus[1]*u_plus[3] + sigma_6[2,6,1]*u_minus[1]*u_plus[2] + sigma_6[2,6,2]*u_minus[1]*u_plus[1] + sigma_6[2,6,3]*u_minus[1]*u_plus[0] + sigma_6[2,6,4]*u_minus[1]*u + sigma_6[2,6,5]*u_minus[1]*u_minus[0] + sigma_6[2,6,6]*u_minus[1]**2

	beta[3] = sigma_6[3,0,0]*u_plus[2]**2 + \
		sigma_6[3,1,0]*u_plus[1]*u_plus[2]  + sigma_6[3,1,1]*u_plus[1]**2 + \
		sigma_6[3,2,0]*u_plus[0]*u_plus[2]  + sigma_6[3,2,1]*u_plus[0]*u_plus[1]  + sigma_6[3,2,2]*u_plus[0]**2 + \
		sigma_6[3,3,0]*u*u_plus[2]          + sigma_6[3,3,1]*u*u_plus[1]          + sigma_6[3,3,2]*u*u_plus[0]          + sigma_6[3,3,3]*u**2 + \
		sigma_6[3,4,0]*u_minus[0]*u_plus[2] + sigma_6[3,4,1]*u_minus[0]*u_plus[1] + sigma_6[3,4,2]*u_minus[0]*u_plus[0] + sigma_6[3,4,3]*u_minus[0]*u + sigma_6[3,4,4]*u_minus[0]**2 + \
		sigma_6[3,5,0]*u_minus[1]*u_plus[2] + sigma_6[3,5,1]*u_minus[1]*u_plus[1] + sigma_6[3,5,2]*u_minus[1]*u_plus[0] + sigma_6[3,5,3]*u_minus[1]*u + sigma_6[3,5,4]*u_minus[1]*u_minus[0] + sigma_6[3,5,5]*u_minus[1]**2 + \
		sigma_6[3,6,0]*u_minus[2]*u_plus[2] + sigma_6[3,6,1]*u_minus[2]*u_plus[1] + sigma_6[3,6,2]*u_minus[2]*u_plus[0] + sigma_6[3,6,3]*u_minus[2]*u + sigma_6[3,6,4]*u_minus[2]*u_minus[0] + sigma_6[3,6,5]*u_minus[2]*u_minus[1] + sigma_6[3,6,6]*u_minus[2]**2

	beta[4] = sigma_6[4,0,0]*u_plus[1]**2 + \
		sigma_6[4,1,0]*u_plus[0]*u_plus[1]  + sigma_6[4,1,1]*u_plus[0]**2 + \
		sigma_6[4,2,0]*u*u_plus[1]          + sigma_6[4,2,1]*u*u_plus[0]          + sigma_6[4,2,2]*u**2 + \
		sigma_6[4,3,0]*u_minus[0]*u_plus[1] + sigma_6[4,3,1]*u_minus[0]*u_plus[0] + sigma_6[4,3,2]*u_minus[0]*u + sigma_6[4,3,3]*u_minus[0]**2 + \
		sigma_6[4,4,0]*u_minus[1]*u_plus[1] + sigma_6[4,4,1]*u_minus[1]*u_plus[0] + sigma_6[4,4,2]*u_minus[1]*u + sigma_6[4,4,3]*u_minus[1]*u_minus[0] + sigma_6[4,4,4]*u_minus[1]**2 + \
		sigma_6[4,5,0]*u_minus[2]*u_plus[1] + sigma_6[4,5,1]*u_minus[2]*u_plus[0] + sigma_6[4,5,2]*u_minus[2]*u + sigma_6[4,5,3]*u_minus[2]*u_minus[0] + sigma_6[4,5,4]*u_minus[2]*u_minus[1] + sigma_6[4,5,5]*u_minus[2]**2 + \
		sigma_6[4,6,0]*u_minus[3]*u_plus[1] + sigma_6[4,6,1]*u_minus[3]*u_plus[0] + sigma_6[4,6,2]*u_minus[3]*u + sigma_6[4,6,3]*u_minus[3]*u_minus[0] + sigma_6[4,6,4]*u_minus[3]*u_minus[1] + sigma_6[4,6,5]*u_minus[3]*u_minus[2] + sigma_6[4,6,6]*u_minus[3]**2

	beta[5] = sigma_6[5,0,0]*u_plus[0]**2 + \
		sigma_6[5,1,0]*u*u_plus[0]          + sigma_6[5,1,1]*u**2 + \
		sigma_6[5,2,0]*u_minus[0]*u_plus[0] + sigma_6[5,2,1]*u_minus[0]*u + sigma_6[5,2,2]*u_minus[0]**2 + \
		sigma_6[5,3,0]*u_minus[1]*u_plus[0] + sigma_6[5,3,1]*u_minus[1]*u + sigma_6[5,3,2]*u_minus[1]*u_minus[0] + sigma_6[5,3,3]*u_minus[1]**2 + \
		sigma_6[5,4,0]*u_minus[2]*u_plus[0] + sigma_6[5,4,1]*u_minus[2]*u + sigma_6[5,4,2]*u_minus[2]*u_minus[0] + sigma_6[5,4,3]*u_minus[2]*u_minus[1] + sigma_6[5,4,4]*u_minus[2]**2 + \
		sigma_6[5,5,0]*u_minus[3]*u_plus[0] + sigma_6[5,5,1]*u_minus[3]*u + sigma_6[5,5,2]*u_minus[3]*u_minus[0] + sigma_6[5,5,3]*u_minus[3]*u_minus[1] + sigma_6[5,5,4]*u_minus[3]*u_minus[2] + sigma_6[5,5,5]*u_minus[3]**2 + \
		sigma_6[5,6,0]*u_minus[4]*u_plus[0] + sigma_6[5,6,1]*u_minus[4]*u + sigma_6[5,6,2]*u_minus[4]*u_minus[0] + sigma_6[5,6,3]*u_minus[4]*u_minus[1] + sigma_6[5,6,4]*u_minus[4]*u_minus[2] + sigma_6[5,6,5]*u_minus[4]*u_minus[3] + sigma_6[5,6,6]*u_minus[4]**2

	beta[6] = sigma_6[6,0,0]*u**2 + \
		sigma_6[6,1,0]*u_minus[0]*u + sigma_6[6,1,1]*u_minus[0]**2 + \
		sigma_6[6,2,0]*u_minus[1]*u + sigma_6[6,2,1]*u_minus[1]*u_minus[0] + sigma_6[6,2,2]*u_minus[1]**2 + \
		sigma_6[6,3,0]*u_minus[2]*u + sigma_6[6,3,1]*u_minus[2]*u_minus[0] + sigma_6[6,3,2]*u_minus[2]*u_minus[1] + sigma_6[6,3,3]*u_minus[2]**2 + \
		sigma_6[6,4,0]*u_minus[3]*u + sigma_6[6,4,1]*u_minus[3]*u_minus[0] + sigma_6[6,4,2]*u_minus[3]*u_minus[1] + sigma_6[6,4,3]*u_minus[3]*u_minus[2] + sigma_6[6,4,4]*u_minus[3]**2 + \
		sigma_6[6,5,0]*u_minus[4]*u + sigma_6[6,5,1]*u_minus[4]*u_minus[0] + sigma_6[6,5,2]*u_minus[4]*u_minus[1] + sigma_6[6,5,3]*u_minus[4]*u_minus[2] + sigma_6[6,5,4]*u_minus[4]*u_minus[3] + sigma_6[6,5,5]*u_minus[4]**2 + \
		sigma_6[6,6,0]*u_minus[5]*u + sigma_6[6,6,1]*u_minus[5]*u_minus[0] + sigma_6[6,6,2]*u_minus[5]*u_minus[1] + sigma_6[6,6,3]*u_minus[5]*u_minus[2] + sigma_6[6,6,4]*u_minus[5]*u_minus[3] + sigma_6[6,6,5]*u_minus[5]*u_minus[4] + sigma_6[6,6,6]*u_minus[5]**2

	return beta

@jit(nopython=True, parallel=True)
def calculate_beta_7(u, u_minus, u_plus):
	beta = np.empty((8, len(u), len(u[0])), dtype=np.longdouble)

	beta[0] = sigma_7[0,0,0]*u_plus[6]**2 + \
		sigma_7[0,1,0]*u_plus[5]*u_plus[6] + sigma_7[0,1,1]*u_plus[5]**2 + \
		sigma_7[0,2,0]*u_plus[4]*u_plus[6] + sigma_7[0,2,1]*u_plus[4]*u_plus[5] + sigma_7[0,2,2]*u_plus[4]**2 + \
		sigma_7[0,3,0]*u_plus[3]*u_plus[6] + sigma_7[0,3,1]*u_plus[3]*u_plus[5] + sigma_7[0,3,2]*u_plus[3]*u_plus[4] + sigma_7[0,3,3]*u_plus[3]**2 + \
		sigma_7[0,4,0]*u_plus[2]*u_plus[6] + sigma_7[0,4,1]*u_plus[2]*u_plus[5] + sigma_7[0,4,2]*u_plus[2]*u_plus[4] + sigma_7[0,4,3]*u_plus[2]*u_plus[3] + sigma_7[0,4,4]*u_plus[2]**2 + \
		sigma_7[0,5,0]*u_plus[1]*u_plus[6] + sigma_7[0,5,1]*u_plus[1]*u_plus[5] + sigma_7[0,5,2]*u_plus[1]*u_plus[4] + sigma_7[0,5,3]*u_plus[1]*u_plus[3] + sigma_7[0,5,4]*u_plus[1]*u_plus[2] + sigma_7[0,5,5]*u_plus[1]**2 + \
		sigma_7[0,6,0]*u_plus[0]*u_plus[6] + sigma_7[0,6,1]*u_plus[0]*u_plus[5] + sigma_7[0,6,2]*u_plus[0]*u_plus[4] + sigma_7[0,6,3]*u_plus[0]*u_plus[3] + sigma_7[0,6,4]*u_plus[0]*u_plus[2] + sigma_7[0,6,5]*u_plus[0]*u_plus[1] + sigma_7[0,6,6]*u_plus[0]**2 + \
		sigma_7[0,7,0]*u*u_plus[6]         + sigma_7[0,7,1]*u*u_plus[5]         + sigma_7[0,7,2]*u*u_plus[4]         + sigma_7[0,7,3]*u*u_plus[3]         + sigma_7[0,7,4]*u*u_plus[2]         + sigma_7[0,7,5]*u*u_plus[1]         + sigma_7[0,7,6]*u*u_plus[0]  + sigma_7[0,7,7]*u**2

	beta[1] = sigma_7[1,0,0]*u_plus[5]**2 + \
		sigma_7[1,1,0]*u_plus[4]*u_plus[5]  + sigma_7[1,1,1]*u_plus[4]**2 + \
		sigma_7[1,2,0]*u_plus[3]*u_plus[5]  + sigma_7[1,2,1]*u_plus[3]*u_plus[4]  + sigma_7[1,2,2]*u_plus[3]**2 + \
		sigma_7[1,3,0]*u_plus[2]*u_plus[5]  + sigma_7[1,3,1]*u_plus[2]*u_plus[4]  + sigma_7[1,3,2]*u_plus[2]*u_plus[3]  + sigma_7[1,3,3]*u_plus[2]**2 + \
		sigma_7[1,4,0]*u_plus[1]*u_plus[5]  + sigma_7[1,4,1]*u_plus[1]*u_plus[4]  + sigma_7[1,4,2]*u_plus[1]*u_plus[3]  + sigma_7[1,4,3]*u_plus[1]*u_plus[2]  + sigma_7[1,4,4]*u_plus[1]**2 + \
		sigma_7[1,5,0]*u_plus[0]*u_plus[5]  + sigma_7[1,5,1]*u_plus[0]*u_plus[4]  + sigma_7[1,5,2]*u_plus[0]*u_plus[3]  + sigma_7[1,5,3]*u_plus[0]*u_plus[2]  + sigma_7[1,5,4]*u_plus[0]*u_plus[1]  + sigma_7[1,5,5]*u_plus[0]**2 + \
		sigma_7[1,6,0]*u*u_plus[5]          + sigma_7[1,6,1]*u*u_plus[4]          + sigma_7[1,6,2]*u*u_plus[3]          + sigma_7[1,6,3]*u*u_plus[2]          + sigma_7[1,6,4]*u*u_plus[1]          + sigma_7[1,6,5]*u*u_plus[0]          + sigma_7[1,6,6]*u**2 + \
		sigma_7[1,7,0]*u_minus[0]*u_plus[5] + sigma_7[1,7,1]*u_minus[0]*u_plus[4] + sigma_7[1,7,2]*u_minus[0]*u_plus[3] + sigma_7[1,7,3]*u_minus[0]*u_plus[2] + sigma_7[1,7,4]*u_minus[0]*u_plus[1] + sigma_7[1,7,5]*u_minus[0]*u_plus[0] + sigma_7[1,7,6]*u_minus[0]*u + sigma_7[1,7,7]*u_minus[0]**2	

	beta[2] = sigma_7[2,0,0]*u_plus[4]**2 + \
		sigma_7[2,1,0]*u_plus[3]*u_plus[4]  + sigma_7[2,1,1]*u_plus[3]**2 + \
		sigma_7[2,2,0]*u_plus[2]*u_plus[4]  + sigma_7[2,2,1]*u_plus[2]*u_plus[3]  + sigma_7[2,2,2]*u_plus[2]**2 + \
		sigma_7[2,3,0]*u_plus[1]*u_plus[4]  + sigma_7[2,3,1]*u_plus[1]*u_plus[3]  + sigma_7[2,3,2]*u_plus[1]*u_plus[2]  + sigma_7[2,3,3]*u_plus[1]**2 + \
		sigma_7[2,4,0]*u_plus[0]*u_plus[4]  + sigma_7[2,4,1]*u_plus[0]*u_plus[3]  + sigma_7[2,4,2]*u_plus[0]*u_plus[2]  + sigma_7[2,4,3]*u_plus[0]*u_plus[1]  + sigma_7[2,4,4]*u_plus[0]**2 + \
		sigma_7[2,5,0]*u*u_plus[4]          + sigma_7[2,5,1]*u*u_plus[3]          + sigma_7[2,5,2]*u*u_plus[2]          + sigma_7[2,5,3]*u*u_plus[1]          + sigma_7[2,5,4]*u*u_plus[0]          + sigma_7[2,5,5]*u**2 + \
		sigma_7[2,6,0]*u_minus[0]*u_plus[4] + sigma_7[2,6,1]*u_minus[0]*u_plus[3] + sigma_7[2,6,2]*u_minus[0]*u_plus[2] + sigma_7[2,6,3]*u_minus[0]*u_plus[1] + sigma_7[2,6,4]*u_minus[0]*u_plus[0] + sigma_7[2,6,5]*u_minus[0]*u + sigma_7[2,6,6]*u_minus[0]**2 + \
		sigma_7[2,7,0]*u_minus[1]*u_plus[4] + sigma_7[2,7,1]*u_minus[1]*u_plus[3] + sigma_7[2,7,2]*u_minus[1]*u_plus[2] + sigma_7[2,7,3]*u_minus[1]*u_plus[1] + sigma_7[2,7,4]*u_minus[1]*u_plus[0] + sigma_7[2,7,5]*u_minus[1]*u + sigma_7[2,7,6]*u_minus[1]*u_minus[0] + sigma_7[2,7,7]*u_minus[1]**2

	beta[3] = sigma_7[3,0,0]*u_plus[3]**2 + \
		sigma_7[3,1,0]*u_plus[2]*u_plus[3]  + sigma_7[3,1,1]*u_plus[2]**2 + \
		sigma_7[3,2,0]*u_plus[1]*u_plus[3]  + sigma_7[3,2,1]*u_plus[1]*u_plus[2]  + sigma_7[3,2,2]*u_plus[1]**2 + \
		sigma_7[3,3,0]*u_plus[0]*u_plus[3]  + sigma_7[3,3,1]*u_plus[0]*u_plus[2]  + sigma_7[3,3,2]*u_plus[0]*u_plus[1]  + sigma_7[3,3,3]*u_plus[0]**2 + \
		sigma_7[3,4,0]*u*u_plus[3]          + sigma_7[3,4,1]*u*u_plus[2]          + sigma_7[3,4,2]*u*u_plus[1]          + sigma_7[3,4,3]*u*u_plus[0]          + sigma_7[3,4,4]*u**2 + \
		sigma_7[3,5,0]*u_minus[0]*u_plus[3] + sigma_7[3,5,1]*u_minus[0]*u_plus[2] + sigma_7[3,5,2]*u_minus[0]*u_plus[1] + sigma_7[3,5,3]*u_minus[0]*u_plus[0] + sigma_7[3,5,4]*u_minus[0]*u + sigma_7[3,5,5]*u_minus[0]**2 + \
		sigma_7[3,6,0]*u_minus[1]*u_plus[3] + sigma_7[3,6,1]*u_minus[1]*u_plus[2] + sigma_7[3,6,2]*u_minus[1]*u_plus[1] + sigma_7[3,6,3]*u_minus[1]*u_plus[0] + sigma_7[3,6,4]*u_minus[1]*u + sigma_7[3,6,5]*u_minus[1]*u_minus[0] + sigma_7[3,6,6]*u_minus[1]**2 + \
		sigma_7[3,7,0]*u_minus[2]*u_plus[3] + sigma_7[3,7,1]*u_minus[2]*u_plus[2] + sigma_7[3,7,2]*u_minus[2]*u_plus[1] + sigma_7[3,7,3]*u_minus[2]*u_plus[0] + sigma_7[3,7,4]*u_minus[2]*u + sigma_7[3,7,5]*u_minus[2]*u_minus[0] + sigma_7[3,7,6]*u_minus[2]*u_minus[1] + sigma_7[3,7,7]*u_minus[2]**2

	beta[4] = sigma_7[4,0,0]*u_plus[2]**2 + \
		sigma_7[4,1,0]*u_plus[1]*u_plus[2]  + sigma_7[4,1,1]*u_plus[1]**2 + \
		sigma_7[4,2,0]*u_plus[0]*u_plus[2]  + sigma_7[4,2,1]*u_plus[0]*u_plus[1]  + sigma_7[4,2,2]*u_plus[0]**2 + \
		sigma_7[4,3,0]*u*u_plus[2]          + sigma_7[4,3,1]*u*u_plus[1]          + sigma_7[4,3,2]*u*u_plus[0]          + sigma_7[4,3,3]*u**2 + \
		sigma_7[4,4,0]*u_minus[0]*u_plus[2] + sigma_7[4,4,1]*u_minus[0]*u_plus[1] + sigma_7[4,4,2]*u_minus[0]*u_plus[0] + sigma_7[4,4,3]*u_minus[0]*u + sigma_7[4,4,4]*u_minus[0]**2 + \
		sigma_7[4,5,0]*u_minus[1]*u_plus[2] + sigma_7[4,5,1]*u_minus[1]*u_plus[1] + sigma_7[4,5,2]*u_minus[1]*u_plus[0] + sigma_7[4,5,3]*u_minus[1]*u + sigma_7[4,5,4]*u_minus[1]*u_minus[0] + sigma_7[4,5,5]*u_minus[1]**2 + \
		sigma_7[4,6,0]*u_minus[2]*u_plus[2] + sigma_7[4,6,1]*u_minus[2]*u_plus[1] + sigma_7[4,6,2]*u_minus[2]*u_plus[0] + sigma_7[4,6,3]*u_minus[2]*u + sigma_7[4,6,4]*u_minus[2]*u_minus[0] + sigma_7[4,6,5]*u_minus[2]*u_minus[1] + sigma_7[4,6,6]*u_minus[2]**2 + \
		sigma_7[4,7,0]*u_minus[3]*u_plus[2] + sigma_7[4,7,1]*u_minus[3]*u_plus[1] + sigma_7[4,7,2]*u_minus[3]*u_plus[0] + sigma_7[4,7,3]*u_minus[3]*u + sigma_7[4,7,4]*u_minus[3]*u_minus[0] + sigma_7[4,7,5]*u_minus[3]*u_minus[1] + sigma_7[4,7,6]*u_minus[3]*u_minus[2] + sigma_7[4,7,7]*u_minus[3]**2

	beta[5] = sigma_7[5,0,0]*u_plus[1]**2 + \
		sigma_7[5,1,0]*u_plus[0]*u_plus[1]  + sigma_7[5,1,1]*u_plus[0]**2 + \
		sigma_7[5,2,0]*u*u_plus[1]          + sigma_7[5,2,1]*u*u_plus[0]          + sigma_7[5,2,2]*u**2 + \
		sigma_7[5,3,0]*u_minus[0]*u_plus[1] + sigma_7[5,3,1]*u_minus[0]*u_plus[0] + sigma_7[5,3,2]*u_minus[0]*u + sigma_7[5,3,3]*u_minus[0]**2 + \
		sigma_7[5,4,0]*u_minus[1]*u_plus[1] + sigma_7[5,4,1]*u_minus[1]*u_plus[0] + sigma_7[5,4,2]*u_minus[1]*u + sigma_7[5,4,3]*u_minus[1]*u_minus[0] + sigma_7[5,4,4]*u_minus[1]**2 + \
		sigma_7[5,5,0]*u_minus[2]*u_plus[1] + sigma_7[5,5,1]*u_minus[2]*u_plus[0] + sigma_7[5,5,2]*u_minus[2]*u + sigma_7[5,5,3]*u_minus[2]*u_minus[0] + sigma_7[5,5,4]*u_minus[2]*u_minus[1] + sigma_7[5,5,5]*u_minus[2]**2 + \
		sigma_7[5,6,0]*u_minus[3]*u_plus[1] + sigma_7[5,6,1]*u_minus[3]*u_plus[0] + sigma_7[5,6,2]*u_minus[3]*u + sigma_7[5,6,3]*u_minus[3]*u_minus[0] + sigma_7[5,6,4]*u_minus[3]*u_minus[1] + sigma_7[5,6,5]*u_minus[3]*u_minus[2] + sigma_7[5,6,6]*u_minus[3]**2 + \
		sigma_7[5,7,0]*u_minus[4]*u_plus[1] + sigma_7[5,7,1]*u_minus[4]*u_plus[0] + sigma_7[5,7,2]*u_minus[4]*u + sigma_7[5,7,3]*u_minus[4]*u_minus[0] + sigma_7[5,7,4]*u_minus[4]*u_minus[1] + sigma_7[5,7,5]*u_minus[4]*u_minus[2] + sigma_7[5,7,6]*u_minus[4]*u_minus[3] + sigma_7[5,7,7]*u_minus[4]**2

	beta[6] = sigma_7[6,0,0]*u_plus[0]**2 + \
		sigma_7[6,1,0]*u*u_plus[0]          + sigma_7[6,1,1]*u**2 + \
		sigma_7[6,2,0]*u_minus[0]*u_plus[0] + sigma_7[6,2,1]*u_minus[0]*u + sigma_7[6,2,2]*u_minus[0]**2 + \
		sigma_7[6,3,0]*u_minus[1]*u_plus[0] + sigma_7[6,3,1]*u_minus[1]*u + sigma_7[6,3,2]*u_minus[1]*u_minus[0] + sigma_7[6,3,3]*u_minus[1]**2 + \
		sigma_7[6,4,0]*u_minus[2]*u_plus[0] + sigma_7[6,4,1]*u_minus[2]*u + sigma_7[6,4,2]*u_minus[2]*u_minus[0] + sigma_7[6,4,3]*u_minus[2]*u_minus[1] + sigma_7[6,4,4]*u_minus[2]**2 + \
		sigma_7[6,5,0]*u_minus[3]*u_plus[0] + sigma_7[6,5,1]*u_minus[3]*u + sigma_7[6,5,2]*u_minus[3]*u_minus[0] + sigma_7[6,5,3]*u_minus[3]*u_minus[1] + sigma_7[6,5,4]*u_minus[3]*u_minus[2] + sigma_7[6,5,5]*u_minus[3]**2 + \
		sigma_7[6,6,0]*u_minus[4]*u_plus[0] + sigma_7[6,6,1]*u_minus[4]*u + sigma_7[6,6,2]*u_minus[4]*u_minus[0] + sigma_7[6,6,3]*u_minus[4]*u_minus[1] + sigma_7[6,6,4]*u_minus[4]*u_minus[2] + sigma_7[6,6,5]*u_minus[4]*u_minus[3] + sigma_7[6,6,6]*u_minus[4]**2 + \
		sigma_7[6,7,0]*u_minus[5]*u_plus[0] + sigma_7[6,7,1]*u_minus[5]*u + sigma_7[6,7,2]*u_minus[5]*u_minus[0] + sigma_7[6,7,3]*u_minus[5]*u_minus[1] + sigma_7[6,7,4]*u_minus[5]*u_minus[2] + sigma_7[6,7,5]*u_minus[5]*u_minus[3] + sigma_7[6,7,6]*u_minus[5]*u_minus[4] + sigma_7[6,7,7]*u_minus[5]**2

	beta[7] = sigma_7[7,0,0]*u**2 + \
		sigma_7[7,1,0]*u_minus[0]*u + sigma_7[7,1,1]*u_minus[0]**2 + \
		sigma_7[7,2,0]*u_minus[1]*u + sigma_7[7,2,1]*u_minus[1]*u_minus[0] + sigma_7[7,2,2]*u_minus[1]**2 + \
		sigma_7[7,3,0]*u_minus[2]*u + sigma_7[7,3,1]*u_minus[2]*u_minus[0] + sigma_7[7,3,2]*u_minus[2]*u_minus[1] + sigma_7[7,3,3]*u_minus[2]**2 + \
		sigma_7[7,4,0]*u_minus[3]*u + sigma_7[7,4,1]*u_minus[3]*u_minus[0] + sigma_7[7,4,2]*u_minus[3]*u_minus[1] + sigma_7[7,4,3]*u_minus[3]*u_minus[2] + sigma_7[7,4,4]*u_minus[3]**2 + \
		sigma_7[7,5,0]*u_minus[4]*u + sigma_7[7,5,1]*u_minus[4]*u_minus[0] + sigma_7[7,5,2]*u_minus[4]*u_minus[1] + sigma_7[7,5,3]*u_minus[4]*u_minus[2] + sigma_7[7,5,4]*u_minus[4]*u_minus[3] + sigma_7[7,5,5]*u_minus[4]**2 + \
		sigma_7[7,6,0]*u_minus[5]*u + sigma_7[7,6,1]*u_minus[5]*u_minus[0] + sigma_7[7,6,2]*u_minus[5]*u_minus[1] + sigma_7[7,6,3]*u_minus[5]*u_minus[2] + sigma_7[7,6,4]*u_minus[5]*u_minus[3] + sigma_7[7,6,5]*u_minus[5]*u_minus[4] + sigma_7[7,6,6]*u_minus[5]**2 + \
		sigma_7[7,7,0]*u_minus[6]*u + sigma_7[7,7,1]*u_minus[6]*u_minus[0] + sigma_7[7,7,2]*u_minus[6]*u_minus[1] + sigma_7[7,7,3]*u_minus[6]*u_minus[2] + sigma_7[7,7,4]*u_minus[6]*u_minus[3] + sigma_7[7,7,5]*u_minus[6]*u_minus[4] + sigma_7[7,7,6]*u_minus[6]*u_minus[5] + sigma_7[7,7,7]*u_minus[6]**2

	return beta

@jit(nopython=True, parallel=True)
def calculate_beta_8(u, u_minus, u_plus):
	beta = np.empty((9, len(u), len(u[0])), dtype=np.longdouble)
	beta[0] = sigma_8[0,0,0]*u_plus[7]**2 + \
		sigma_8[0,1,0]*u_plus[6]*u_plus[7] + sigma_8[0,1,1]*u_plus[6]**2 + \
		sigma_8[0,2,0]*u_plus[5]*u_plus[7] + sigma_8[0,2,1]*u_plus[5]*u_plus[6] + sigma_8[0,2,2]*u_plus[5]**2 + \
		sigma_8[0,3,0]*u_plus[4]*u_plus[7] + sigma_8[0,3,1]*u_plus[4]*u_plus[6] + sigma_8[0,3,2]*u_plus[4]*u_plus[5] + sigma_8[0,3,3]*u_plus[4]**2 + \
		sigma_8[0,4,0]*u_plus[3]*u_plus[7] + sigma_8[0,4,1]*u_plus[3]*u_plus[6] + sigma_8[0,4,2]*u_plus[3]*u_plus[5] + sigma_8[0,4,3]*u_plus[3]*u_plus[4] + sigma_8[0,4,4]*u_plus[3]**2 + \
		sigma_8[0,5,0]*u_plus[2]*u_plus[7] + sigma_8[0,5,1]*u_plus[2]*u_plus[6] + sigma_8[0,5,2]*u_plus[2]*u_plus[5] + sigma_8[0,5,3]*u_plus[2]*u_plus[4] + sigma_8[0,5,4]*u_plus[2]*u_plus[3] + sigma_8[0,5,5]*u_plus[2]**2 + \
		sigma_8[0,6,0]*u_plus[1]*u_plus[7] + sigma_8[0,6,1]*u_plus[1]*u_plus[6] + sigma_8[0,6,2]*u_plus[1]*u_plus[5] + sigma_8[0,6,3]*u_plus[1]*u_plus[4] + sigma_8[0,6,4]*u_plus[1]*u_plus[3] + sigma_8[0,6,5]*u_plus[1]*u_plus[2] + sigma_8[0,6,6]*u_plus[1]**2 + \
		sigma_8[0,7,0]*u_plus[0]*u_plus[7] + sigma_8[0,7,1]*u_plus[0]*u_plus[6] + sigma_8[0,7,2]*u_plus[0]*u_plus[5] + sigma_8[0,7,3]*u_plus[0]*u_plus[4] + sigma_8[0,7,4]*u_plus[0]*u_plus[3] + sigma_8[0,7,5]*u_plus[0]*u_plus[2] + sigma_8[0,7,6]*u_plus[0]*u_plus[1] + sigma_8[0,7,7]*u_plus[0]**2 + \
		sigma_8[0,8,0]*u*u_plus[7]         + sigma_8[0,8,1]*u*u_plus[6]         + sigma_8[0,8,2]*u*u_plus[5]         + sigma_8[0,8,3]*u*u_plus[4]         + sigma_8[0,8,4]*u*u_plus[3]         + sigma_8[0,8,5]*u*u_plus[2]         + sigma_8[0,8,6]*u*u_plus[1]         + sigma_8[0,8,7]*u*u_plus[0]  + sigma_8[0,8,8]*u**2

	beta[1] = sigma_8[1,0,0]*u_plus[6]**2 + \
		sigma_8[1,1,0]*u_plus[5]*u_plus[6]  + sigma_8[1,1,1]*u_plus[5]**2 + \
		sigma_8[1,2,0]*u_plus[4]*u_plus[6]  + sigma_8[1,2,1]*u_plus[4]*u_plus[5]  + sigma_8[1,2,2]*u_plus[4]**2 + \
		sigma_8[1,3,0]*u_plus[3]*u_plus[6]  + sigma_8[1,3,1]*u_plus[3]*u_plus[5]  + sigma_8[1,3,2]*u_plus[3]*u_plus[4]  + sigma_8[1,3,3]*u_plus[3]**2 + \
		sigma_8[1,4,0]*u_plus[2]*u_plus[6]  + sigma_8[1,4,1]*u_plus[2]*u_plus[5]  + sigma_8[1,4,2]*u_plus[2]*u_plus[4]  + sigma_8[1,4,3]*u_plus[2]*u_plus[3]  + sigma_8[1,4,4]*u_plus[2]**2 + \
		sigma_8[1,5,0]*u_plus[1]*u_plus[6]  + sigma_8[1,5,1]*u_plus[1]*u_plus[5]  + sigma_8[1,5,2]*u_plus[1]*u_plus[4]  + sigma_8[1,5,3]*u_plus[1]*u_plus[3]  + sigma_8[1,5,4]*u_plus[1]*u_plus[2]  + sigma_8[1,5,5]*u_plus[1]**2 + \
		sigma_8[1,6,0]*u_plus[0]*u_plus[6]  + sigma_8[1,6,1]*u_plus[0]*u_plus[5]  + sigma_8[1,6,2]*u_plus[0]*u_plus[4]  + sigma_8[1,6,3]*u_plus[0]*u_plus[3]  + sigma_8[1,6,4]*u_plus[0]*u_plus[2]  + sigma_8[1,6,5]*u_plus[0]*u_plus[1]  + sigma_8[1,6,6]*u_plus[0]**2 + \
		sigma_8[1,7,0]*u*u_plus[6]          + sigma_8[1,7,1]*u*u_plus[5]          + sigma_8[1,7,2]*u*u_plus[4]          + sigma_8[1,7,3]*u*u_plus[3]          + sigma_8[1,7,4]*u*u_plus[2]          + sigma_8[1,7,5]*u*u_plus[1]          + sigma_8[1,7,6]*u*u_plus[0]          + sigma_8[1,7,7]*u**2 + \
		sigma_8[1,8,0]*u_minus[0]*u_plus[6] + sigma_8[1,8,1]*u_minus[0]*u_plus[5] + sigma_8[1,8,2]*u_minus[0]*u_plus[4] + sigma_8[1,8,3]*u_minus[0]*u_plus[3] + sigma_8[1,8,4]*u_minus[0]*u_plus[2] + sigma_8[1,8,5]*u_minus[0]*u_plus[1] + sigma_8[1,8,6]*u_minus[0]*u_plus[0] + sigma_8[1,8,7]*u_minus[0]*u + sigma_8[1,8,8]*u_minus[0]**2

	beta[2] = sigma_8[2,0,0]*u_plus[5]**2 + \
		sigma_8[2,1,0]*u_plus[4]*u_plus[5]  + sigma_8[2,1,1]*u_plus[4]**2 + \
		sigma_8[2,2,0]*u_plus[3]*u_plus[5]  + sigma_8[2,2,1]*u_plus[3]*u_plus[4]  + sigma_8[2,2,2]*u_plus[3]**2 + \
		sigma_8[2,3,0]*u_plus[2]*u_plus[5]  + sigma_8[2,3,1]*u_plus[2]*u_plus[4]  + sigma_8[2,3,2]*u_plus[2]*u_plus[3]  + sigma_8[2,3,3]*u_plus[2]**2 + \
		sigma_8[2,4,0]*u_plus[1]*u_plus[5]  + sigma_8[2,4,1]*u_plus[1]*u_plus[4]  + sigma_8[2,4,2]*u_plus[1]*u_plus[3]  + sigma_8[2,4,3]*u_plus[1]*u_plus[2]  + sigma_8[2,4,4]*u_plus[1]**2 + \
		sigma_8[2,5,0]*u_plus[0]*u_plus[5]  + sigma_8[2,5,1]*u_plus[0]*u_plus[4]  + sigma_8[2,5,2]*u_plus[0]*u_plus[3]  + sigma_8[2,5,3]*u_plus[0]*u_plus[2]  + sigma_8[2,5,4]*u_plus[0]*u_plus[1]  + sigma_8[2,5,5]*u_plus[0]**2 + \
		sigma_8[2,6,0]*u*u_plus[5]          + sigma_8[2,6,1]*u*u_plus[4]          + sigma_8[2,6,2]*u*u_plus[3]          + sigma_8[2,6,3]*u*u_plus[2]          + sigma_8[2,6,4]*u*u_plus[1]          + sigma_8[2,6,5]*u*u_plus[0]          + sigma_8[2,6,6]*u**2 + \
		sigma_8[2,7,0]*u_minus[0]*u_plus[5] + sigma_8[2,7,1]*u_minus[0]*u_plus[4] + sigma_8[2,7,2]*u_minus[0]*u_plus[3] + sigma_8[2,7,3]*u_minus[0]*u_plus[2] + sigma_8[2,7,4]*u_minus[0]*u_plus[1] + sigma_8[2,7,5]*u_minus[0]*u_plus[0] + sigma_8[2,7,6]*u_minus[0]*u + sigma_8[2,7,7]*u_minus[0]**2 + \
		sigma_8[2,8,0]*u_minus[1]*u_plus[5] + sigma_8[2,8,1]*u_minus[1]*u_plus[4] + sigma_8[2,8,2]*u_minus[1]*u_plus[3] + sigma_8[2,8,3]*u_minus[1]*u_plus[2] + sigma_8[2,8,4]*u_minus[1]*u_plus[1] + sigma_8[2,8,5]*u_minus[1]*u_plus[0] + sigma_8[2,8,6]*u_minus[1]*u + sigma_8[2,8,7]*u_minus[1]*u_minus[0] + sigma_8[2,8,8]*u_minus[1]**2
	
	beta[3] = sigma_8[3,0,0]*u_plus[4]**2 + \
		sigma_8[3,1,0]*u_plus[3]*u_plus[4]  + sigma_8[3,1,1]*u_plus[3]**2 + \
		sigma_8[3,2,0]*u_plus[2]*u_plus[4]  + sigma_8[3,2,1]*u_plus[2]*u_plus[3]  + sigma_8[3,2,2]*u_plus[2]**2 + \
		sigma_8[3,3,0]*u_plus[1]*u_plus[4]  + sigma_8[3,3,1]*u_plus[1]*u_plus[3]  + sigma_8[3,3,2]*u_plus[1]*u_plus[2]  + sigma_8[3,3,3]*u_plus[1]**2 + \
		sigma_8[3,4,0]*u_plus[0]*u_plus[4]  + sigma_8[3,4,1]*u_plus[0]*u_plus[3]  + sigma_8[3,4,2]*u_plus[0]*u_plus[2]  + sigma_8[3,4,3]*u_plus[0]*u_plus[1]  + sigma_8[3,4,4]*u_plus[0]**2 + \
		sigma_8[3,5,0]*u*u_plus[4]          + sigma_8[3,5,1]*u*u_plus[3]          + sigma_8[3,5,2]*u*u_plus[2]          + sigma_8[3,5,3]*u*u_plus[1]          + sigma_8[3,5,4]*u*u_plus[0]          + sigma_8[3,5,5]*u**2 + \
		sigma_8[3,6,0]*u_minus[0]*u_plus[4] + sigma_8[3,6,1]*u_minus[0]*u_plus[3] + sigma_8[3,6,2]*u_minus[0]*u_plus[2] + sigma_8[3,6,3]*u_minus[0]*u_plus[1] + sigma_8[3,6,4]*u_minus[0]*u_plus[0] + sigma_8[3,6,5]*u_minus[0]*u + sigma_8[3,6,6]*u_minus[0]**2 + \
		sigma_8[3,7,0]*u_minus[1]*u_plus[4] + sigma_8[3,7,1]*u_minus[1]*u_plus[3] + sigma_8[3,7,2]*u_minus[1]*u_plus[2] + sigma_8[3,7,3]*u_minus[1]*u_plus[1] + sigma_8[3,7,4]*u_minus[1]*u_plus[0] + sigma_8[3,7,5]*u_minus[1]*u + sigma_8[3,7,6]*u_minus[1]*u_minus[0] + sigma_8[3,7,7]*u_minus[1]**2 + \
		sigma_8[3,8,0]*u_minus[2]*u_plus[4] + sigma_8[3,8,1]*u_minus[2]*u_plus[3] + sigma_8[3,8,2]*u_minus[2]*u_plus[2] + sigma_8[3,8,3]*u_minus[2]*u_plus[1] + sigma_8[3,8,4]*u_minus[2]*u_plus[0] + sigma_8[3,8,5]*u_minus[2]*u + sigma_8[3,8,6]*u_minus[2]*u_minus[0] + sigma_8[3,8,7]*u_minus[2]*u_minus[1] + sigma_8[3,8,8]*u_minus[2]**2

	beta[4] = sigma_8[4,0,0]*u_plus[3]**2 + \
		sigma_8[4,1,0]*u_plus[2]*u_plus[3]  + sigma_8[4,1,1]*u_plus[2]**2 + \
		sigma_8[4,2,0]*u_plus[1]*u_plus[3]  + sigma_8[4,2,1]*u_plus[1]*u_plus[2]  + sigma_8[4,2,2]*u_plus[1]**2 + \
		sigma_8[4,3,0]*u_plus[0]*u_plus[3]  + sigma_8[4,3,1]*u_plus[0]*u_plus[2]  + sigma_8[4,3,2]*u_plus[0]*u_plus[1]  + sigma_8[4,3,3]*u_plus[0]**2 + \
		sigma_8[4,4,0]*u*u_plus[3]          + sigma_8[4,4,1]*u*u_plus[2]          + sigma_8[4,4,2]*u*u_plus[1]          + sigma_8[4,4,3]*u*u_plus[0]          + sigma_8[4,4,4]*u**2 + \
		sigma_8[4,5,0]*u_minus[0]*u_plus[3] + sigma_8[4,5,1]*u_minus[0]*u_plus[2] + sigma_8[4,5,2]*u_minus[0]*u_plus[1] + sigma_8[4,5,3]*u_minus[0]*u_plus[0] + sigma_8[4,5,4]*u_minus[0]*u + sigma_8[4,5,5]*u_minus[0]**2 + \
		sigma_8[4,6,0]*u_minus[1]*u_plus[3] + sigma_8[4,6,1]*u_minus[1]*u_plus[2] + sigma_8[4,6,2]*u_minus[1]*u_plus[1] + sigma_8[4,6,3]*u_minus[1]*u_plus[0] + sigma_8[4,6,4]*u_minus[1]*u + sigma_8[4,6,5]*u_minus[1]*u_minus[0] + sigma_8[4,6,6]*u_minus[1]**2 + \
		sigma_8[4,7,0]*u_minus[2]*u_plus[3] + sigma_8[4,7,1]*u_minus[2]*u_plus[2] + sigma_8[4,7,2]*u_minus[2]*u_plus[1] + sigma_8[4,7,3]*u_minus[2]*u_plus[0] + sigma_8[4,7,4]*u_minus[2]*u + sigma_8[4,7,5]*u_minus[2]*u_minus[0] + sigma_8[4,7,6]*u_minus[2]*u_minus[1] + sigma_8[4,7,7]*u_minus[2]**2 + \
		sigma_8[4,8,0]*u_minus[3]*u_plus[3] + sigma_8[4,8,1]*u_minus[3]*u_plus[2] + sigma_8[4,8,2]*u_minus[3]*u_plus[1] + sigma_8[4,8,3]*u_minus[3]*u_plus[0] + sigma_8[4,8,4]*u_minus[3]*u + sigma_8[4,8,5]*u_minus[3]*u_minus[0] + sigma_8[4,8,6]*u_minus[3]*u_minus[1] + sigma_8[4,8,7]*u_minus[3]*u_minus[2] + sigma_8[4,8,8]*u_minus[3]**2
	
	beta[5] = sigma_8[5,0,0]*u_plus[2]**2 + \
		sigma_8[5,1,0]*u_plus[1]*u_plus[2]  + sigma_8[5,1,1]*u_plus[1]**2 + \
		sigma_8[5,2,0]*u_plus[0]*u_plus[2]  + sigma_8[5,2,1]*u_plus[0]*u_plus[1]  + sigma_8[5,2,2]*u_plus[0]**2 + \
		sigma_8[5,3,0]*u*u_plus[2]          + sigma_8[5,3,1]*u*u_plus[1]          + sigma_8[5,3,2]*u*u_plus[0]          + sigma_8[5,3,3]*u**2 + \
		sigma_8[5,4,0]*u_minus[0]*u_plus[2] + sigma_8[5,4,1]*u_minus[0]*u_plus[1] + sigma_8[5,4,2]*u_minus[0]*u_plus[0] + sigma_8[5,4,3]*u_minus[0]*u + sigma_8[5,4,4]*u_minus[0]**2 + \
		sigma_8[5,5,0]*u_minus[1]*u_plus[2] + sigma_8[5,5,1]*u_minus[1]*u_plus[1] + sigma_8[5,5,2]*u_minus[1]*u_plus[0] + sigma_8[5,5,3]*u_minus[1]*u + sigma_8[5,5,4]*u_minus[1]*u_minus[0] + sigma_8[5,5,5]*u_minus[1]**2 + \
		sigma_8[5,6,0]*u_minus[2]*u_plus[2] + sigma_8[5,6,1]*u_minus[2]*u_plus[1] + sigma_8[5,6,2]*u_minus[2]*u_plus[0] + sigma_8[5,6,3]*u_minus[2]*u + sigma_8[5,6,4]*u_minus[2]*u_minus[0] + sigma_8[5,6,5]*u_minus[2]*u_minus[1] + sigma_8[5,6,6]*u_minus[2]**2 + \
		sigma_8[5,7,0]*u_minus[3]*u_plus[2] + sigma_8[5,7,1]*u_minus[3]*u_plus[1] + sigma_8[5,7,2]*u_minus[3]*u_plus[0] + sigma_8[5,7,3]*u_minus[3]*u + sigma_8[5,7,4]*u_minus[3]*u_minus[0] + sigma_8[5,7,5]*u_minus[3]*u_minus[1] + sigma_8[5,7,6]*u_minus[3]*u_minus[2] + sigma_8[5,7,7]*u_minus[3]**2 + \
		sigma_8[5,8,0]*u_minus[4]*u_plus[2] + sigma_8[5,8,1]*u_minus[4]*u_plus[1] + sigma_8[5,8,2]*u_minus[4]*u_plus[0] + sigma_8[5,8,3]*u_minus[4]*u + sigma_8[5,8,4]*u_minus[4]*u_minus[0] + sigma_8[5,8,5]*u_minus[4]*u_minus[1] + sigma_8[5,8,6]*u_minus[4]*u_minus[2] + sigma_8[5,8,7]*u_minus[4]*u_minus[3] + sigma_8[5,8,8]*u_minus[4]**2

	beta[6] = sigma_8[6,0,0]*u_plus[1]**2 + \
		sigma_8[6,1,0]*u_plus[0]*u_plus[1]  + sigma_8[6,1,1]*u_plus[0]**2 + \
		sigma_8[6,2,0]*u*u_plus[1]          + sigma_8[6,2,1]*u*u_plus[0]          + sigma_8[6,2,2]*u**2 + \
		sigma_8[6,3,0]*u_minus[0]*u_plus[1] + sigma_8[6,3,1]*u_minus[0]*u_plus[0] + sigma_8[6,3,2]*u_minus[0]*u + sigma_8[6,3,3]*u_minus[0]**2 + \
		sigma_8[6,4,0]*u_minus[1]*u_plus[1] + sigma_8[6,4,1]*u_minus[1]*u_plus[0] + sigma_8[6,4,2]*u_minus[1]*u + sigma_8[6,4,3]*u_minus[1]*u_minus[0] + sigma_8[6,4,4]*u_minus[1]**2 + \
		sigma_8[6,5,0]*u_minus[2]*u_plus[1] + sigma_8[6,5,1]*u_minus[2]*u_plus[0] + sigma_8[6,5,2]*u_minus[2]*u + sigma_8[6,5,3]*u_minus[2]*u_minus[0] + sigma_8[6,5,4]*u_minus[2]*u_minus[1] + sigma_8[6,5,5]*u_minus[2]**2 + \
		sigma_8[6,6,0]*u_minus[3]*u_plus[1] + sigma_8[6,6,1]*u_minus[3]*u_plus[0] + sigma_8[6,6,2]*u_minus[3]*u + sigma_8[6,6,3]*u_minus[3]*u_minus[0] + sigma_8[6,6,4]*u_minus[3]*u_minus[1] + sigma_8[6,6,5]*u_minus[3]*u_minus[2] + sigma_8[6,6,6]*u_minus[3]**2 + \
		sigma_8[6,7,0]*u_minus[4]*u_plus[1] + sigma_8[6,7,1]*u_minus[4]*u_plus[0] + sigma_8[6,7,2]*u_minus[4]*u + sigma_8[6,7,3]*u_minus[4]*u_minus[0] + sigma_8[6,7,4]*u_minus[4]*u_minus[1] + sigma_8[6,7,5]*u_minus[4]*u_minus[2] + sigma_8[6,7,6]*u_minus[4]*u_minus[3] + sigma_8[6,7,7]*u_minus[4]**2 + \
		sigma_8[6,8,0]*u_minus[5]*u_plus[1] + sigma_8[6,8,1]*u_minus[5]*u_plus[0] + sigma_8[6,8,2]*u_minus[5]*u + sigma_8[6,8,3]*u_minus[5]*u_minus[0] + sigma_8[6,8,4]*u_minus[5]*u_minus[1] + sigma_8[6,8,5]*u_minus[5]*u_minus[2] + sigma_8[6,8,6]*u_minus[5]*u_minus[3] + sigma_8[6,8,7]*u_minus[5]*u_minus[4] + sigma_8[6,8,8]*u_minus[5]**2

	beta[7] = sigma_8[7,0,0]*u_plus[0]**2 + \
		sigma_8[7,1,0]*u*u_plus[0]          + sigma_8[7,1,1]*u**2 + \
		sigma_8[7,2,0]*u_minus[0]*u_plus[0] + sigma_8[7,2,1]*u_minus[0]*u + sigma_8[7,2,2]*u_minus[0]**2 + \
		sigma_8[7,3,0]*u_minus[1]*u_plus[0] + sigma_8[7,3,1]*u_minus[1]*u + sigma_8[7,3,2]*u_minus[1]*u_minus[0] + sigma_8[7,3,3]*u_minus[1]**2 + \
		sigma_8[7,4,0]*u_minus[2]*u_plus[0] + sigma_8[7,4,1]*u_minus[2]*u + sigma_8[7,4,2]*u_minus[2]*u_minus[0] + sigma_8[7,4,3]*u_minus[2]*u_minus[1] + sigma_8[7,4,4]*u_minus[2]**2 + \
		sigma_8[7,5,0]*u_minus[3]*u_plus[0] + sigma_8[7,5,1]*u_minus[3]*u + sigma_8[7,5,2]*u_minus[3]*u_minus[0] + sigma_8[7,5,3]*u_minus[3]*u_minus[1] + sigma_8[7,5,4]*u_minus[3]*u_minus[2] + sigma_8[7,5,5]*u_minus[3]**2 + \
		sigma_8[7,6,0]*u_minus[4]*u_plus[0] + sigma_8[7,6,1]*u_minus[4]*u + sigma_8[7,6,2]*u_minus[4]*u_minus[0] + sigma_8[7,6,3]*u_minus[4]*u_minus[1] + sigma_8[7,6,4]*u_minus[4]*u_minus[2] + sigma_8[7,6,5]*u_minus[4]*u_minus[3] + sigma_8[7,6,6]*u_minus[4]**2 + \
		sigma_8[7,7,0]*u_minus[5]*u_plus[0] + sigma_8[7,7,1]*u_minus[5]*u + sigma_8[7,7,2]*u_minus[5]*u_minus[0] + sigma_8[7,7,3]*u_minus[5]*u_minus[1] + sigma_8[7,7,4]*u_minus[5]*u_minus[2] + sigma_8[7,7,5]*u_minus[5]*u_minus[3] + sigma_8[7,7,6]*u_minus[5]*u_minus[4] + sigma_8[7,7,7]*u_minus[5]**2 + \
		sigma_8[7,8,0]*u_minus[6]*u_plus[0] + sigma_8[7,8,1]*u_minus[6]*u + sigma_8[7,8,2]*u_minus[6]*u_minus[0] + sigma_8[7,8,3]*u_minus[6]*u_minus[1] + sigma_8[7,8,4]*u_minus[6]*u_minus[2] + sigma_8[7,8,5]*u_minus[6]*u_minus[3] + sigma_8[7,8,6]*u_minus[6]*u_minus[4] + sigma_8[7,8,7]*u_minus[6]*u_minus[5] + sigma_8[7,8,8]*u_minus[6]**2

	beta[8] = sigma_8[8,0,0]*u**2 + \
		sigma_8[8,1,0]*u_minus[0]*u + sigma_8[8,1,1]*u_minus[0]**2 + \
		sigma_8[8,2,0]*u_minus[1]*u + sigma_8[8,2,1]*u_minus[1]*u_minus[0] + sigma_8[8,2,2]*u_minus[1]**2 + \
		sigma_8[8,3,0]*u_minus[2]*u + sigma_8[8,3,1]*u_minus[2]*u_minus[0] + sigma_8[8,3,2]*u_minus[2]*u_minus[1] + sigma_8[8,3,3]*u_minus[2]**2 + \
		sigma_8[8,4,0]*u_minus[3]*u + sigma_8[8,4,1]*u_minus[3]*u_minus[0] + sigma_8[8,4,2]*u_minus[3]*u_minus[1] + sigma_8[8,4,3]*u_minus[3]*u_minus[2] + sigma_8[8,4,4]*u_minus[3]**2 + \
		sigma_8[8,5,0]*u_minus[4]*u + sigma_8[8,5,1]*u_minus[4]*u_minus[0] + sigma_8[8,5,2]*u_minus[4]*u_minus[1] + sigma_8[8,5,3]*u_minus[4]*u_minus[2] + sigma_8[8,5,4]*u_minus[4]*u_minus[3] + sigma_8[8,5,5]*u_minus[4]**2 + \
		sigma_8[8,6,0]*u_minus[5]*u + sigma_8[8,6,1]*u_minus[5]*u_minus[0] + sigma_8[8,6,2]*u_minus[5]*u_minus[1] + sigma_8[8,6,3]*u_minus[5]*u_minus[2] + sigma_8[8,6,4]*u_minus[5]*u_minus[3] + sigma_8[8,6,5]*u_minus[5]*u_minus[4] + sigma_8[8,6,6]*u_minus[5]**2 + \
		sigma_8[8,7,0]*u_minus[6]*u + sigma_8[8,7,1]*u_minus[6]*u_minus[0] + sigma_8[8,7,2]*u_minus[6]*u_minus[1] + sigma_8[8,7,3]*u_minus[6]*u_minus[2] + sigma_8[8,7,4]*u_minus[6]*u_minus[3] + sigma_8[8,7,5]*u_minus[6]*u_minus[4] + sigma_8[8,7,6]*u_minus[6]*u_minus[5] + sigma_8[8,7,7]*u_minus[6]**2 + \
		sigma_8[8,8,0]*u_minus[7]*u + sigma_8[8,8,1]*u_minus[7]*u_minus[0] + sigma_8[8,8,2]*u_minus[7]*u_minus[1] + sigma_8[8,8,3]*u_minus[7]*u_minus[2] + sigma_8[8,8,4]*u_minus[7]*u_minus[3] + sigma_8[8,8,5]*u_minus[7]*u_minus[4] + sigma_8[8,8,6]*u_minus[7]*u_minus[5] + sigma_8[8,8,7]*u_minus[7]*u_minus[6] + sigma_8[8,8,8]*u_minus[7]**2
	

	# beta[0] = sigma_8[0,0,0]*u_plus[7]**2 + \
	# 	sigma_8[0,1,0]*u_plus[6]*u_plus[7] + sigma_8[0,1,1]*u_plus[6]**2 + \
	# 	sigma_8[0,2,0]*u_plus[5]*u_plus[7] + sigma_8[0,2,1]*u_plus[5]*u_plus[6] + sigma_8[0,2,2]*u_plus[5]**2 + \
	# 	sigma_8[0,3,0]*u_plus[4]*u_plus[7] + sigma_8[0,3,1]*u_plus[4]*u_plus[6] + sigma_8[0,3,2]*u_plus[4]*u_plus[5] + sigma_8[0,3,3]*u_plus[4]**2 + \
	# 	sigma_8[0,4,0]*u_plus[3]*u_plus[7] + sigma_8[0,4,1]*u_plus[3]*u_plus[6] + sigma_8[0,4,2]*u_plus[3]*u_plus[5] + sigma_8[0,4,3]*u_plus[3]*u_plus[4] + sigma_8[0,4,4]*u_plus[3]**2 + \
	# 	sigma_8[0,5,0]*u_plus[2]*u_plus[7] + sigma_8[0,5,1]*u_plus[2]*u_plus[6] + sigma_8[0,5,2]*u_plus[2]*u_plus[5] + sigma_8[0,5,3]*u_plus[2]*u_plus[4] + sigma_8[0,5,4]*u_plus[2]*u_plus[3] + sigma_8[0,5,5]*u_plus[2]**2 + \
	# 	sigma_8[0,6,0]*u_plus[1]*u_plus[7] + sigma_8[0,6,1]*u_plus[1]*u_plus[6] + sigma_8[0,6,2]*u_plus[1]*u_plus[5] + sigma_8[0,6,3]*u_plus[1]*u_plus[4] + sigma_8[0,6,4]*u_plus[1]*u_plus[3] + sigma_8[0,6,5]*u_plus[1]*u_plus[2] + sigma_8[0,6,6]*u_plus[1]**2 + \
	# 	sigma_8[0,7,0]*u_plus[0]*u_plus[7] + sigma_8[0,7,1]*u_plus[0]*u_plus[6] + sigma_8[0,7,2]*u_plus[0]*u_plus[5] + sigma_8[0,7,3]*u_plus[0]*u_plus[4] + sigma_8[0,7,4]*u_plus[0]*u_plus[3] + sigma_8[0,7,5]*u_plus[0]*u_plus[2] + sigma_8[0,7,6]*u_plus[0]*u_plus[1] + sigma_8[0,7,7]*u_plus[0]**2 + \
	# 	sigma_8[0,8,0]*u*u_plus[7]         + sigma_8[0,8,1]*u*u_plus[6]         + sigma_8[0,8,2]*u*u_plus[5]         + sigma_8[0,8,3]*u*u_plus[4]         + sigma_8[0,8,4]*u*u_plus[3]         + sigma_8[0,8,5]*u*u_plus[2]         + sigma_8[0,8,6]*u*u_plus[1]         + sigma_8[0,8,7]*u*u_plus[0] + sigma_8[0,8,8]*u**2

	# beta[1] = sigma_8[1,0,0]*u_plus[6]**2 + \
	# 	sigma_8[1,1,0]*u_plus[5]*u_plus[6]  + sigma_8[1,1,1]*u_plus[5]**2 + \
	# 	sigma_8[1,2,0]*u_plus[4]*u_plus[6]  + sigma_8[1,2,1]*u_plus[4]*u_plus[5]  + sigma_8[1,2,2]*u_plus[4]**2 + \
	# 	sigma_8[1,3,0]*u_plus[3]*u_plus[6]  + sigma_8[1,3,1]*u_plus[3]*u_plus[5]  + sigma_8[1,3,2]*u_plus[3]*u_plus[4]  + sigma_8[1,3,3]*u_plus[3]**2 + \
	# 	sigma_8[1,4,0]*u_plus[2]*u_plus[6]  + sigma_8[1,4,1]*u_plus[2]*u_plus[5]  + sigma_8[1,4,2]*u_plus[2]*u_plus[4]  + sigma_8[1,4,3]*u_plus[2]*u_plus[3]  + sigma_8[1,4,4]*u_plus[2]**2 + \
	# 	sigma_8[1,5,0]*u_plus[1]*u_plus[6]  + sigma_8[1,5,1]*u_plus[1]*u_plus[5]  + sigma_8[1,5,2]*u_plus[1]*u_plus[4]  + sigma_8[1,5,3]*u_plus[1]*u_plus[3]  + sigma_8[1,5,4]*u_plus[1]*u_plus[2]  + sigma_8[1,5,5]*u_plus[1]**2 + \
	# 	sigma_8[1,6,0]*u_plus[0]*u_plus[6]  + sigma_8[1,6,1]*u_plus[0]*u_plus[5]  + sigma_8[1,6,2]*u_plus[0]*u_plus[4]  + sigma_8[1,6,3]*u_plus[0]*u_plus[3]  + sigma_8[1,6,4]*u_plus[0]*u_plus[2]  + sigma_8[1,6,5]*u_plus[0]*u_plus[1]  + sigma_8[1,6,6]*u_plus[0]**2 + \
	# 	sigma_8[1,7,0]*u*u_plus[6]          + sigma_8[1,7,1]*u*u_plus[5]          + sigma_8[1,7,2]*u*u_plus[4]          + sigma_8[1,7,3]*u*u_plus[3]          + sigma_8[1,7,4]*u*u_plus[2]          + sigma_8[1,7,5]*u*u_plus[1]          + sigma_8[1,7,6]*u*u_plus[0]          + sigma_8[1,7,7]*u**2 + \
	# 	sigma_8[1,8,0]*u_minus[0]*u_plus[6] + sigma_8[1,8,1]*u_minus[0]*u_plus[5] + sigma_8[1,8,2]*u_minus[0]*u_plus[4] + sigma_8[1,8,3]*u_minus[0]*u_plus[3] + sigma_8[1,8,4]*u_minus[0]*u_plus[2] + sigma_8[1,8,5]*u_minus[0]*u_plus[1] + sigma_8[1,8,6]*u_minus[0]*u_plus[0] + sigma_8[1,8,7]*u_minus[0]*u + sigma_8[1,8,8]*u_minus[0]**2

	# beta[2] = sigma_8[2,0,0]*u_plus[5]**2 + \
	# 	sigma_8[2,1,0]*u_plus[4]*u_plus[5]  + sigma_8[2,1,1]*u_plus[4]**2 + \
	# 	sigma_8[2,2,0]*u_plus[3]*u_plus[5]  + sigma_8[2,2,1]*u_plus[3]*u_plus[4]  + sigma_8[2,2,2]*u_plus[3]**2 + \
	# 	sigma_8[2,3,0]*u_plus[2]*u_plus[5]  + sigma_8[2,3,1]*u_plus[2]*u_plus[4]  + sigma_8[2,3,2]*u_plus[2]*u_plus[3]  + sigma_8[2,3,3]*u_plus[2]**2 + \
	# 	sigma_8[2,4,0]*u_plus[1]*u_plus[5]  + sigma_8[2,4,1]*u_plus[1]*u_plus[4]  + sigma_8[2,4,2]*u_plus[1]*u_plus[3]  + sigma_8[2,4,3]*u_plus[1]*u_plus[2]  + sigma_8[2,4,4]*u_plus[1]**2 + \
	# 	sigma_8[2,5,0]*u_plus[0]*u_plus[5]  + sigma_8[2,5,1]*u_plus[0]*u_plus[4]  + sigma_8[2,5,2]*u_plus[0]*u_plus[3]  + sigma_8[2,5,3]*u_plus[0]*u_plus[2]  + sigma_8[2,5,4]*u_plus[0]*u_plus[1]  + sigma_8[2,5,5]*u_plus[0]**2 + \
	# 	sigma_8[2,6,0]*u*u_plus[5]          + sigma_8[2,6,1]*u*u_plus[4]          + sigma_8[2,6,2]*u*u_plus[3]          + sigma_8[2,6,3]*u*u_plus[2]          + sigma_8[2,6,4]*u*u_plus[1]          + sigma_8[2,6,5]*u*u_plus[0]          + sigma_8[2,6,6]*u**2 + \
	# 	sigma_8[2,7,0]*u_minus[0]*u_plus[5] + sigma_8[2,7,1]*u_minus[0]*u_plus[4] + sigma_8[2,7,2]*u_minus[0]*u_plus[3] + sigma_8[2,7,3]*u_minus[0]*u_plus[2] + sigma_8[2,7,4]*u_minus[0]*u_plus[1] + sigma_8[2,7,5]*u_minus[0]*u_plus[0] + sigma_8[2,7,6]*u_minus[0]*u + sigma_8[2,7,7]*u_minus[0]**2 + \
	# 	sigma_8[2,8,0]*u_minus[1]*u_plus[5] + sigma_8[2,8,1]*u_minus[1]*u_plus[4] + sigma_8[2,8,2]*u_minus[1]*u_plus[3] + sigma_8[2,8,3]*u_minus[1]*u_plus[2] + sigma_8[2,8,4]*u_minus[1]*u_plus[1] + sigma_8[2,8,5]*u_minus[1]*u_plus[0] + sigma_8[2,8,6]*u_minus[1]*u + sigma_8[2,8,7]*u_minus[1]*u_minus[0] + sigma_8[2,8,8]*u_minus[1]**2

	# beta[3] = sigma_8[3,0,0]*u_plus[4]**2 + \
	# 	sigma_8[3,1,0]*u_plus[3]*u_plus[4]  + sigma_8[3,1,1]*u_plus[3]**2 + \
	# 	sigma_8[3,2,0]*u_plus[2]*u_plus[4]  + sigma_8[3,2,1]*u_plus[2]*u_plus[3]  + sigma_8[3,2,2]*u_plus[2]**2 + \
	# 	sigma_8[3,3,0]*u_plus[1]*u_plus[4]  + sigma_8[3,3,1]*u_plus[1]*u_plus[3]  + sigma_8[3,3,2]*u_plus[1]*u_plus[2]  + sigma_8[3,3,3]*u_plus[1]**2 + \
	# 	sigma_8[3,4,0]*u_plus[0]*u_plus[4]  + sigma_8[3,4,1]*u_plus[0]*u_plus[3]  + sigma_8[3,4,2]*u_plus[0]*u_plus[2]  + sigma_8[3,4,3]*u_plus[0]*u_plus[1]  + sigma_8[3,4,4]*u_plus[0]**2 + \
	# 	sigma_8[3,5,0]*u*u_plus[4]          + sigma_8[3,5,1]*u*u_plus[3]          + sigma_8[3,5,2]*u*u_plus[2]          + sigma_8[3,5,3]*u*u_plus[1]          + sigma_8[3,5,4]*u*u_plus[0]          + sigma_8[3,5,5]*u**2 + \
	# 	sigma_8[3,6,0]*u_minus[0]*u_plus[4] + sigma_8[3,6,1]*u_minus[0]*u_plus[3] + sigma_8[3,6,2]*u_minus[0]*u_plus[2] + sigma_8[3,6,3]*u_minus[0]*u_plus[1] + sigma_8[3,6,4]*u_minus[0]*u_plus[0] + sigma_8[3,6,5]*u_minus[0]*u + sigma_8[3,6,6]*u_minus[0]**2 + \
	# 	sigma_8[3,7,0]*u_minus[1]*u_plus[4] + sigma_8[3,7,1]*u_minus[1]*u_plus[3] + sigma_8[3,7,2]*u_minus[1]*u_plus[2] + sigma_8[3,7,3]*u_minus[1]*u_plus[1] + sigma_8[3,7,4]*u_minus[1]*u_plus[0] + sigma_8[3,7,5]*u_minus[1]*u + sigma_8[3,7,6]*u_minus[1]*u_minus[0] + sigma_8[3,7,7]*u_minus[1]**2 + \
	# 	sigma_8[3,8,0]*u_minus[2]*u_plus[4] + sigma_8[3,8,1]*u_minus[2]*u_plus[3] + sigma_8[3,8,2]*u_minus[2]*u_plus[2] + sigma_8[3,8,3]*u_minus[2]*u_plus[1] + sigma_8[3,8,4]*u_minus[2]*u_plus[0] + sigma_8[3,8,5]*u_minus[2]*u + sigma_8[3,8,6]*u_minus[2]*u_minus[0] + sigma_8[3,8,7]*u_minus[2]*u_minus[1] + sigma_8[3,8,8]*u_minus[2]**2

	# beta[4] = sigma_8[4,0,0]*u_plus[3]**2 + \
	# 	sigma_8[4,1,0]*u_plus[2]*u_plus[3]  + sigma_8[4,1,1]*u_plus[2]**2 + \
	# 	sigma_8[4,2,0]*u_plus[1]*u_plus[3]  + sigma_8[4,2,1]*u_plus[1]*u_plus[2]  + sigma_8[4,2,2]*u_plus[1]**2 + \
	# 	sigma_8[4,3,0]*u_plus[0]*u_plus[3]  + sigma_8[4,3,1]*u_plus[0]*u_plus[2]  + sigma_8[4,3,2]*u_plus[0]*u_plus[1]  + sigma_8[4,3,3]*u_plus[0]**2 + \
	# 	sigma_8[4,4,0]*u*u_plus[3]          + sigma_8[4,4,1]*u*u_plus[2]          + sigma_8[4,4,2]*u*u_plus[1]          + sigma_8[4,4,3]*u*u_plus[0]          + sigma_8[4,4,4]*u**2 + \
	# 	sigma_8[4,5,0]*u_minus[0]*u_plus[3] + sigma_8[4,5,1]*u_minus[0]*u_plus[2] + sigma_8[4,5,2]*u_minus[0]*u_plus[1] + sigma_8[4,5,3]*u_minus[0]*u_plus[0] + sigma_8[4,5,4]*u_minus[0]*u + sigma_8[4,5,5]*u_minus[0]**2 + \
	# 	sigma_8[4,6,0]*u_minus[1]*u_plus[3] + sigma_8[4,6,1]*u_minus[1]*u_plus[2] + sigma_8[4,6,2]*u_minus[1]*u_plus[1] + sigma_8[4,6,3]*u_minus[1]*u_plus[0] + sigma_8[4,6,4]*u_minus[1]*u + sigma_8[4,6,5]*u_minus[1]*u_minus[0] + sigma_8[4,6,6]*u_minus[1]**2 + \
	# 	sigma_8[4,7,0]*u_minus[2]*u_plus[3] + sigma_8[4,7,1]*u_minus[2]*u_plus[2] + sigma_8[4,7,2]*u_minus[2]*u_plus[1] + sigma_8[4,7,3]*u_minus[2]*u_plus[0] + sigma_8[4,7,4]*u_minus[2]*u + sigma_8[4,7,5]*u_minus[2]*u_minus[0] + sigma_8[4,7,6]*u_minus[2]*u_minus[1] + sigma_8[4,7,7]*u_minus[2]**2 + \
	# 	sigma_8[4,8,0]*u_minus[3]*u_plus[3] + sigma_8[4,8,1]*u_minus[3]*u_plus[2] + sigma_8[4,8,2]*u_minus[3]*u_plus[1] + sigma_8[4,8,3]*u_minus[3]*u_plus[0] + sigma_8[4,8,4]*u_minus[3]*u + sigma_8[4,8,5]*u_minus[3]*u_minus[0] + sigma_8[4,8,6]*u_minus[3]*u_minus[1] + sigma_8[4,8,7]*u_minus[3]*u_minus[2] + sigma_8[4,8,8]*u_minus[3]**2

	# beta[5] = sigma_8[5,0,0]*u_plus[2]**2 + \
	# 	sigma_8[5,1,0]*u_plus[1]*u_plus[2]  + sigma_8[5,1,1]*u_plus[1]**2 + \
	# 	sigma_8[5,2,0]*u_plus[0]*u_plus[2]  + sigma_8[5,2,1]*u_plus[0]*u_plus[1]  + sigma_8[5,2,2]*u_plus[0]**2 + \
	# 	sigma_8[5,3,0]*u*u_plus[2]          + sigma_8[5,3,1]*u*u_plus[1]          + sigma_8[5,3,2]*u*u_plus[0]          + sigma_8[5,3,3]*u**2 + \
	# 	sigma_8[5,4,0]*u_minus[0]*u_plus[2] + sigma_8[5,4,1]*u_minus[0]*u_plus[1] + sigma_8[5,4,2]*u_minus[0]*u_plus[0] + sigma_8[5,4,3]*u_minus[0]*u + sigma_8[5,4,4]*u_minus[0]**2 + \
	# 	sigma_8[5,5,0]*u_minus[1]*u_plus[2] + sigma_8[5,5,1]*u_minus[1]*u_plus[1] + sigma_8[5,5,2]*u_minus[1]*u_plus[0] + sigma_8[5,5,3]*u_minus[1]*u + sigma_8[5,5,4]*u_minus[1]*u_minus[0] + sigma_8[5,5,5]*u_minus[1]**2 + \
	# 	sigma_8[5,6,0]*u_minus[2]*u_plus[2] + sigma_8[5,6,1]*u_minus[2]*u_plus[1] + sigma_8[5,6,2]*u_minus[2]*u_plus[0] + sigma_8[5,6,3]*u_minus[2]*u + sigma_8[5,6,4]*u_minus[2]*u_minus[0] + sigma_8[5,6,5]*u_minus[2]*u_minus[1] + sigma_8[5,6,6]*u_minus[2]**2 + \
	# 	sigma_8[5,7,0]*u_minus[3]*u_plus[2] + sigma_8[5,7,1]*u_minus[3]*u_plus[1] + sigma_8[5,7,2]*u_minus[3]*u_plus[0] + sigma_8[5,7,3]*u_minus[3]*u + sigma_8[5,7,4]*u_minus[3]*u_minus[0] + sigma_8[5,7,5]*u_minus[3]*u_minus[1] + sigma_8[5,7,6]*u_minus[3]*u_minus[2] + sigma_8[5,7,7]*u_minus[3]**2 + \
	# 	sigma_8[5,8,0]*u_minus[4]*u_plus[2] + sigma_8[5,8,1]*u_minus[4]*u_plus[1] + sigma_8[5,8,2]*u_minus[4]*u_plus[0] + sigma_8[5,8,3]*u_minus[4]*u + sigma_8[5,8,4]*u_minus[4]*u_minus[0] + sigma_8[5,8,5]*u_minus[4]*u_minus[1] + sigma_8[5,8,6]*u_minus[4]*u_minus[2] + sigma_8[5,8,7]*u_minus[4]*u_minus[3] + sigma_8[5,8,8]*u_minus[4]**2

	# beta[6] = sigma_8[6,0,0]*u_plus[1]**2 + \
	# 	sigma_8[6,1,0]*u_plus[0]*u_plus[1]  + sigma_8[6,1,1]*u_plus[0]**2 + \
	# 	sigma_8[6,2,0]*u*u_plus[1]          + sigma_8[6,2,1]*u*u_plus[0]          + sigma_8[6,2,2]*u**2 + \
	# 	sigma_8[6,3,0]*u_minus[0]*u_plus[1] + sigma_8[6,3,1]*u_minus[0]*u_plus[0] + sigma_8[6,3,2]*u_minus[0]*u + sigma_8[6,3,3]*u_minus[0]**2 + \
	# 	sigma_8[6,4,0]*u_minus[1]*u_plus[1] + sigma_8[6,4,1]*u_minus[1]*u_plus[0] + sigma_8[6,4,2]*u_minus[1]*u + sigma_8[6,4,3]*u_minus[1]*u_minus[0] + sigma_8[6,4,4]*u_minus[1]**2 + \
	# 	sigma_8[6,5,0]*u_minus[2]*u_plus[1] + sigma_8[6,5,1]*u_minus[2]*u_plus[0] + sigma_8[6,5,2]*u_minus[2]*u + sigma_8[6,5,3]*u_minus[2]*u_minus[0] + sigma_8[6,5,4]*u_minus[2]*u_minus[1] + sigma_8[6,5,5]*u_minus[2]**2 + \
	# 	sigma_8[6,6,0]*u_minus[3]*u_plus[1] + sigma_8[6,6,1]*u_minus[3]*u_plus[0] + sigma_8[6,6,2]*u_minus[3]*u + sigma_8[6,6,3]*u_minus[3]*u_minus[0] + sigma_8[6,6,4]*u_minus[3]*u_minus[1] + sigma_8[6,6,5]*u_minus[3]*u_minus[2] + sigma_8[6,6,6]*u_minus[3]**2 + \
	# 	sigma_8[6,7,0]*u_minus[4]*u_plus[1] + sigma_8[6,7,1]*u_minus[4]*u_plus[0] + sigma_8[6,7,2]*u_minus[4]*u + sigma_8[6,7,3]*u_minus[4]*u_minus[0] + sigma_8[6,7,4]*u_minus[4]*u_minus[1] + sigma_8[6,7,5]*u_minus[4]*u_minus[2] + sigma_8[6,7,6]*u_minus[4]*u_minus[3] + sigma_8[6,7,7]*u_minus[4]**2 + \
	# 	sigma_8[6,8,0]*u_minus[5]*u_plus[1] + sigma_8[6,8,1]*u_minus[5]*u_plus[0] + sigma_8[6,8,2]*u_minus[5]*u + sigma_8[6,8,3]*u_minus[5]*u_minus[0] + sigma_8[6,8,4]*u_minus[5]*u_minus[1] + sigma_8[6,8,5]*u_minus[5]*u_minus[2] + sigma_8[6,8,6]*u_minus[5]*u_minus[3] + sigma_8[6,8,7]*u_minus[5]*u_minus[4] + sigma_8[6,8,8]*u_minus[5]**2

	# beta[7] = sigma_8[7,0,0]*u_plus[0]**2 + \
	# 	sigma_8[7,1,0]*u*u_plus[0]          + sigma_8[7,1,1]*u**2 + \
	# 	sigma_8[7,2,0]*u_minus[0]*u_plus[0] + sigma_8[7,2,1]*u_minus[0]*u + sigma_8[7,2,2]*u_minus[0]**2 + \
	# 	sigma_8[7,3,0]*u_minus[1]*u_plus[0] + sigma_8[7,3,1]*u_minus[1]*u + sigma_8[7,3,2]*u_minus[1]*u_minus[0] + sigma_8[7,3,3]*u_minus[1]**2 + \
	# 	sigma_8[7,4,0]*u_minus[2]*u_plus[0] + sigma_8[7,4,1]*u_minus[2]*u + sigma_8[7,4,2]*u_minus[2]*u_minus[0] + sigma_8[7,4,3]*u_minus[2]*u_minus[1] + sigma_8[7,4,4]*u_minus[2]**2 + \
	# 	sigma_8[7,5,0]*u_minus[3]*u_plus[0] + sigma_8[7,5,1]*u_minus[3]*u + sigma_8[7,5,2]*u_minus[3]*u_minus[0] + sigma_8[7,5,3]*u_minus[3]*u_minus[1] + sigma_8[7,5,4]*u_minus[3]*u_minus[2] + sigma_8[7,5,5]*u_minus[3]**2 + \
	# 	sigma_8[7,6,0]*u_minus[4]*u_plus[0] + sigma_8[7,6,1]*u_minus[4]*u + sigma_8[7,6,2]*u_minus[4]*u_minus[0] + sigma_8[7,6,3]*u_minus[4]*u_minus[1] + sigma_8[7,6,4]*u_minus[4]*u_minus[2] + sigma_8[7,6,5]*u_minus[4]*u_minus[3] + sigma_8[7,6,6]*u_minus[4]**2 + \
	# 	sigma_8[7,7,0]*u_minus[5]*u_plus[0] + sigma_8[7,7,1]*u_minus[5]*u + sigma_8[7,7,2]*u_minus[5]*u_minus[0] + sigma_8[7,7,3]*u_minus[5]*u_minus[1] + sigma_8[7,7,4]*u_minus[5]*u_minus[2] + sigma_8[7,7,5]*u_minus[5]*u_minus[3] + sigma_8[7,7,6]*u_minus[5]*u_minus[4] + sigma_8[7,7,7]*u_minus[5]**2 + \
	# 	sigma_8[7,8,0]*u_minus[6]*u_plus[0] + sigma_8[7,8,1]*u_minus[6]*u + sigma_8[7,8,2]*u_minus[6]*u_minus[0] + sigma_8[7,8,3]*u_minus[6]*u_minus[1] + sigma_8[7,8,4]*u_minus[6]*u_minus[2] + sigma_8[7,8,5]*u_minus[6]*u_minus[3] + sigma_8[7,8,6]*u_minus[6]*u_minus[4] + sigma_8[7,8,7]*u_minus[6]*u_minus[5] + sigma_8[7,8,8]*u_minus[6]**2

	# beta[8] = sigma_8[8,0,0]*u**2 + \
	# 	sigma_8[8,1,0]*u_minus[0]*u + sigma_8[8,1,1]*u_minus[0]**2 + \
	# 	sigma_8[8,2,0]*u_minus[1]*u + sigma_8[8,2,1]*u_minus[1]*u_minus[0] + sigma_8[8,2,2]*u_minus[1]**2 + \
	# 	sigma_8[8,3,0]*u_minus[2]*u + sigma_8[8,3,1]*u_minus[2]*u_minus[0] + sigma_8[8,3,2]*u_minus[2]*u_minus[1] + sigma_8[8,3,3]*u_minus[2]**2 + \
	# 	sigma_8[8,4,0]*u_minus[3]*u + sigma_8[8,4,1]*u_minus[3]*u_minus[0] + sigma_8[8,4,2]*u_minus[3]*u_minus[1] + sigma_8[8,4,3]*u_minus[3]*u_minus[2] + sigma_8[8,4,4]*u_minus[3]**2 + \
	# 	sigma_8[8,5,0]*u_minus[4]*u + sigma_8[8,5,1]*u_minus[4]*u_minus[0] + sigma_8[8,5,2]*u_minus[4]*u_minus[1] + sigma_8[8,5,3]*u_minus[4]*u_minus[2] + sigma_8[8,5,4]*u_minus[4]*u_minus[3] + sigma_8[8,5,5]*u_minus[4]**2 + \
	# 	sigma_8[8,6,0]*u_minus[5]*u + sigma_8[8,6,1]*u_minus[5]*u_minus[0] + sigma_8[8,6,2]*u_minus[5]*u_minus[1] + sigma_8[8,6,3]*u_minus[5]*u_minus[2] + sigma_8[8,6,4]*u_minus[5]*u_minus[3] + sigma_8[8,6,5]*u_minus[5]*u_minus[4] + sigma_8[8,6,6]*u_minus[5]**2 + \
	# 	sigma_8[8,7,0]*u_minus[6]*u + sigma_8[8,7,1]*u_minus[6]*u_minus[0] + sigma_8[8,7,2]*u_minus[6]*u_minus[1] + sigma_8[8,7,3]*u_minus[6]*u_minus[2] + sigma_8[8,7,4]*u_minus[6]*u_minus[3] + sigma_8[8,7,5]*u_minus[6]*u_minus[4] + sigma_8[8,7,6]*u_minus[6]*u_minus[5] + sigma_8[8,7,7]*u_minus[6]**2 + \
	# 	sigma_8[8,8,0]*u_minus[7]*u + sigma_8[8,8,1]*u_minus[7]*u_minus[0] + sigma_8[8,8,2]*u_minus[7]*u_minus[1] + sigma_8[8,8,3]*u_minus[7]*u_minus[2] + sigma_8[8,8,4]*u_minus[7]*u_minus[3] + sigma_8[8,8,5]*u_minus[7]*u_minus[4] + sigma_8[8,8,6]*u_minus[7]*u_minus[5] + sigma_8[8,8,7]*u_minus[7]*u_minus[6] + sigma_8[8,8,8]*u_minus[7]**2

	return beta