from numba import njit
import numpy as np
import sys
from smoothness_13 import sigma_6
from smoothness_15 import sigma_7
from smoothness_17 import sigma_8

@njit
def calculate_beta_characteristic(u, r, Q_inverse):
	if r == 0:
		return calculate_beta_0(u, r, Q_inverse)
	elif r == 1:
		return calculate_beta_1(u, r, Q_inverse)
	elif r == 2:
		return calculate_beta_2(u, r, Q_inverse)
	elif r == 3:
		return calculate_beta_3(u, r, Q_inverse)
	elif r == 4:
		return calculate_beta_4(u, r, Q_inverse)
	elif r == 5:
		return calculate_beta_5(u, r, Q_inverse)
	elif r == 6:
		return calculate_beta_6(u, r, Q_inverse)
	elif r == 7:
		return calculate_beta_7(u, r, Q_inverse)
	elif r == 8:
		return calculate_beta_8(u, r, Q_inverse)

'''
For each calculate_beta() function, u_i is referenced as u[r]
'''

@njit
def calculate_beta_0(u, r, Q_inverse):
	beta = np.array([[1.0]])
	return beta

@njit
def calculate_beta_1(u, r, Q_inverse):
	beta = np.zeros((r+1,3))
	beta[0] = (np.dot(Q_inverse,u[2])-np.dot(Q_inverse,u[1]))**2
	beta[1] = (np.dot(Q_inverse,u[1])-np.dot(Q_inverse,u[0]))**2
	return beta

@njit
def calculate_beta_2(u, r, Q_inverse):
	beta = np.zeros((r+1,3))
	beta[0] = 13/12*(np.dot(Q_inverse, u[2]-2*u[3]+u[4]))**2 + 1/4*(np.dot(Q_inverse, 3*u[2]-4*u[3]+u[4]))**2
	beta[1] = 13/12*(np.dot(Q_inverse, u[1]-2*u[2]+u[3]))**2 + 1/4*(np.dot(Q_inverse, u[1]-u[3]))**2
	beta[2] = 13/12*(np.dot(Q_inverse, u[0]-2*u[1]+u[2]))**2 + 1/4*(np.dot(Q_inverse, u[0]-4*u[1]+3*u[2]))**2
	return beta	

@njit
def calculate_beta_3(u, r, Q_inverse):
	beta = np.zeros((r+1,3))

	beta[0] = np.dot(Q_inverse,u[3])*np.dot(Q_inverse,2107*u[3]-9402*u[4]+7042*u[5]-1854*u[6]) + \
		np.dot(Q_inverse,u[4])*np.dot(Q_inverse,11003*u[4]-17246*u[5]+4642*u[6]) + \
		np.dot(Q_inverse,u[5])*np.dot(Q_inverse,7043*u[5]-3882*u[6])+547*np.dot(Q_inverse,u[6])**2
	beta[1] = np.dot(Q_inverse,u[2])*(np.dot(Q_inverse,547*u[2]-2522*u[3]+1922*u[4]-494*u[5])) + \
		np.dot(Q_inverse,u[3])*np.dot(Q_inverse,3443*u[3]-5966*u[4]+1602*u[5]) + \
		np.dot(Q_inverse,u[4])*np.dot(Q_inverse,2843*u[4]-1642*u[5])+267*np.dot(Q_inverse,u[5])**2
	beta[2] = np.dot(Q_inverse,u[1])*np.dot(Q_inverse,267*u[1]-1642*u[2]+1602*u[3]-494*u[4]) + \
		np.dot(Q_inverse,u[2])*np.dot(Q_inverse,2843*u[2]-5966*u[3]+1922*u[4]) + \
		np.dot(Q_inverse,u[3])*np.dot(Q_inverse,3443*u[3]-2522*u[4])+547*np.dot(Q_inverse,u[4])**2
	beta[3] = np.dot(Q_inverse,u[0])*np.dot(Q_inverse,547*u[0]-3882*u[1]+4642*u[2]-1854*u[3]) + \
		np.dot(Q_inverse,u[1])*np.dot(Q_inverse,7043*u[1]-17246*u[2]+7042*u[3]) + \
		np.dot(Q_inverse,u[2])*np.dot(Q_inverse,11003*u[2]-9402*u[3])+2107*np.dot(Q_inverse,u[3])**2
	
	return beta		

@njit
def calculate_beta_4(u, r, Q_inverse):
	beta = np.zeros((r+1,3))

	beta[0] = np.dot(Q_inverse,u[4])*np.dot(Q_inverse,107918*u[4]-649501*u[5]+758823*u[6]-411487*u[7]+86329*u[8]) + \
		np.dot(Q_inverse,u[5])*np.dot(Q_inverse,1020563*u[5]-2462076*u[6]+1358458*u[7]-288007*u[8]) + \
		np.dot(Q_inverse,u[6])*np.dot(Q_inverse,1521393*u[6]-1704396*u[7]+364863*u[8]) + \
		np.dot(Q_inverse,u[7])*np.dot(Q_inverse,482963*u[7]-208501*u[8]) + \
		22658*np.dot(Q_inverse,u[8])**2

	beta[1] = np.dot(Q_inverse,u[3])*np.dot(Q_inverse,22658*u[3]-140251*u[4]+165153*u[5]-88297*u[6]+18079*u[7]) + \
		np.dot(Q_inverse,u[4])*np.dot(Q_inverse,242723*u[4]-611976*u[5]+337018*u[6]-70237*u[7]) + \
		np.dot(Q_inverse,u[5])*np.dot(Q_inverse,406293*u[5]-464976*u[6]+99213*u[7]) + \
		np.dot(Q_inverse,u[6])*np.dot(Q_inverse,138563*u[6]-60871*u[7]) + \
		6908*np.dot(Q_inverse,u[7])**2

	beta[2] = np.dot(Q_inverse,u[2])*np.dot(Q_inverse,6908*u[2]-51001*u[3]+67923*u[4]-38947*u[5]+8209*u[6]) + \
		np.dot(Q_inverse,u[3])*np.dot(Q_inverse,104963*u[3]-299076*u[4]+179098*u[5]-38947*u[6]) + \
		np.dot(Q_inverse,u[4])*np.dot(Q_inverse,231153*u[4]-299076*u[5]+67923*u[6]) + \
		np.dot(Q_inverse,u[5])*np.dot(Q_inverse,104963*u[5]-51001*u[6]) + \
		6908*np.dot(Q_inverse,u[6])**2

	beta[3] = np.dot(Q_inverse,u[1])*np.dot(Q_inverse,6908*u[1]-60871*u[2]+99213*u[3]-70237*u[4]+18079*u[5]) + \
		np.dot(Q_inverse,u[2])*np.dot(Q_inverse,138563*u[2]-464976*u[3]+337018*u[4]-88297*u[5]) + \
		np.dot(Q_inverse,u[3])*np.dot(Q_inverse,406293*u[3]-611976*u[4]+165153*u[5]) + \
		np.dot(Q_inverse,u[4])*np.dot(Q_inverse,242723*u[4]-140251*u[5]) + \
		22658*np.dot(Q_inverse,u[5])**2

	beta[4] = np.dot(Q_inverse,u[0])*np.dot(Q_inverse,22658*u[0]-208501*u[1]+364863*u[2]-288007*u[3]+86329*u[4]) + \
		np.dot(Q_inverse,u[1])*np.dot(Q_inverse,482963*u[1]-1704396*u[2]+1358458*u[3]-411487*u[4]) + \
		np.dot(Q_inverse,u[2])*np.dot(Q_inverse,1521393*u[2]-2462076*u[3]+758823*u[4]) + \
		np.dot(Q_inverse,u[3])*np.dot(Q_inverse,1020563*u[3]-649501*u[4]) + \
		107918*np.dot(Q_inverse,u[4])**2
	return beta

@njit
def calculate_beta_5(u, r, Q_inverse):
	beta = np.zeros((r+1,3))

	beta[0] = np.dot(Q_inverse,u[5])*np.dot(Q_inverse, 6150211*u[5]-47460464*u[6]+76206736*u[7]-63394124*u[8]+27060170*u[9]-4712740*u[10]) + \
		np.dot(Q_inverse,u[6])*np.dot(Q_inverse, 94851237*u[6]-311771244*u[7]+262901672*u[8]-113206788*u[9]+19834350*u[10]) + \
		np.dot(Q_inverse,u[7])*np.dot(Q_inverse, 260445372*u[7]-444003904*u[8]+192596472*u[9]-33918804*u[10]) + \
		np.dot(Q_inverse,u[8])*np.dot(Q_inverse, 190757572*u[8]-166461044*u[9]+29442256*u[10]) + \
		np.dot(Q_inverse,u[9])*np.dot(Q_inverse, 36480687*u[9]-12950184*u[10]) + \
		1152561*np.dot(Q_inverse,u[10])**2

	beta[1] = np.dot(Q_inverse,u[4])*np.dot(Q_inverse, 1152561*u[4]-9117992*u[5]+14742480*u[6]-12183636*u[7]+5134574*u[8]-880548*u[9]) + \
		np.dot(Q_inverse,u[5])*np.dot(Q_inverse, 19365967*u[5]-65224244*u[6]+55053752*u[7]-23510468*u[8]+4067018*u[9]) + \
		np.dot(Q_inverse,u[6])*np.dot(Q_inverse, 56662212*u[6]-97838784*u[7]+42405032*u[8]-7408908*u[9]) + \
		np.dot(Q_inverse,u[7])*np.dot(Q_inverse, 43093692*u[7]-37913324*u[8]+6694608*u[9]) + \
		np.dot(Q_inverse,u[8])*np.dot(Q_inverse, 8449957*u[8]-3015728*u[9]) + \
		271779*np.dot(Q_inverse,u[9])**2

	beta[2] = np.dot(Q_inverse,u[3])*np.dot(Q_inverse, 271779*u[3]-2380800*u[4]+4086352*u[5]-3462252*u[6]+1458762*u[7]-245620*u[8]) + \
		np.dot(Q_inverse,u[4])*np.dot(Q_inverse, 5653317*u[4]-20427884*u[5]+17905032*u[6]-7727988*u[7]+1325006*u[8]) + \
		np.dot(Q_inverse,u[5])*np.dot(Q_inverse, 19510972*u[5]-35817664*u[6]+15929912*u[7]-2792660*u[8]) + \
		np.dot(Q_inverse,u[6])*np.dot(Q_inverse, 17195652*u[6]-15880404*u[7]+2863984*u[8]) + \
		np.dot(Q_inverse,u[7])*np.dot(Q_inverse, 3824847*u[7]-1429976*u[8]) + \
		139633*np.dot(Q_inverse,u[8])**2

	beta[3] = np.dot(Q_inverse,u[2])*np.dot(Q_inverse, 139633*u[2]-1429976*u[3]+2863984*u[4]-2792660*u[5]+1325006*u[6]-245620*u[7]) + \
		np.dot(Q_inverse,u[3])*np.dot(Q_inverse, 3824847*u[3]-15880404*u[4]+15929912*u[5]-7727988*u[6]+1458762*u[7]) + \
		np.dot(Q_inverse,u[4])*np.dot(Q_inverse, 17195652*u[4]-35817664*u[5]+17905032*u[6]-3462252*u[7]) + \
		np.dot(Q_inverse,u[5])*np.dot(Q_inverse, 19510972*u[5]-20427884*u[6]+4086352*u[7]) + \
		np.dot(Q_inverse,u[6])*np.dot(Q_inverse, 5653317*u[6]-2380800*u[7]) + \
		271779*np.dot(Q_inverse,u[7])**2

	beta[4] = np.dot(Q_inverse,u[1])*np.dot(Q_inverse, 271779*u[1]-3015728*u[2]+6694608*u[3]-7408908*u[4]+4067018*u[5]-880548*u[6]) + \
		np.dot(Q_inverse,u[2])*np.dot(Q_inverse, 8449957*u[2]-37913324*u[3]+42405032*u[4]-23510468*u[5]+5134574*u[6]) + \
		np.dot(Q_inverse,u[3])*np.dot(Q_inverse, 43093692*u[3]-97838784*u[4]+55053752*u[5]-12183636*u[6]) + \
		np.dot(Q_inverse,u[4])*np.dot(Q_inverse, 56662212*u[4]-65224244*u[5]+14742480*u[6]) + \
		np.dot(Q_inverse,u[5])*np.dot(Q_inverse, 19365967*u[5]-9117992*u[6]) + \
		1152561*np.dot(Q_inverse,u[6])**2

	beta[5] = np.dot(Q_inverse,u[0])*np.dot(Q_inverse, 1152561*u[0]-12950184*u[1]+29442256*u[2]-33918804*u[3]+19834350*u[4]-4712740*u[5]) + \
		np.dot(Q_inverse,u[1])*np.dot(Q_inverse, 36480687*u[1]-166461044*u[2]+192596472*u[3]-113206788*u[4]+27060170*u[5]) + \
		np.dot(Q_inverse,u[2])*np.dot(Q_inverse, 190757572*u[2]-444003904*u[3]+262901672*u[4]-63394124*u[5]) + \
		np.dot(Q_inverse,u[3])*np.dot(Q_inverse, 260445372*u[3]-311771244*u[4]+76206736*u[5]) + \
		np.dot(Q_inverse,u[4])*np.dot(Q_inverse, 94851237*u[4]-47460464*u[5]) + \
		6150211*np.dot(Q_inverse,u[5])**2

	return beta

# # @jit(nopython=True, parallel=True)
@njit
def calculate_beta_6(u, r, Q_inverse):
	beta = np.zeros((r+1,3))
	for k_s in range(r+1):
		for l in range(r+1):
			for m in range(l+1):
				beta[k_s] += sigma_6[k_s,l,m]*np.dot(Q_inverse, u[2*r-k_s-l])*np.dot(Q_inverse, u[2*r-k_s-m])
	return beta


	# beta[0] = sigma_6[0,0,0]*np.dot(Q_inverse,u[12])**2 + \
	# 	sigma_6[0,1,0]*np.dot(Q_inverse,u[11]*u[12]) + sigma_6[0,1,1]*np.dot(Q_inverse,u[11])**2 + \
	# 	sigma_6[0,2,0]*np.dot(Q_inverse,u[10]*u[12]) + sigma_6[0,2,1]*np.dot(Q_inverse,u[10]*u[11]) + sigma_6[0,2,2]*np.dot(Q_inverse,u[10])**2 + \
	# 	sigma_6[0,3,0]*np.dot(Q_inverse,u[9]*u[12]) + sigma_6[0,3,1]*np.dot(Q_inverse,u[9]*u[11]) + sigma_6[0,3,2]*np.dot(Q_inverse,u[9]*u[10]) + sigma_6[0,3,3]*np.dot(Q_inverse,u[9])**2 + \
	# 	sigma_6[0,4,0]*np.dot(Q_inverse,u[8]*u[12]) + sigma_6[0,4,1]*np.dot(Q_inverse,u[8]*u[11]) + sigma_6[0,4,2]*np.dot(Q_inverse,u[8]*u[10]) + sigma_6[0,4,3]*np.dot(Q_inverse,u[8]*u[9]) + sigma_6[0,4,4]*np.dot(Q_inverse,u[8])**2 + \
	# 	sigma_6[0,5,0]*np.dot(Q_inverse,u[7]*u[12]) + sigma_6[0,5,1]*np.dot(Q_inverse,u[7]*u[11]) + sigma_6[0,5,2]*np.dot(Q_inverse,u[7]*u[10]) + sigma_6[0,5,3]*np.dot(Q_inverse,u[7]*u[9]) + sigma_6[0,5,4]*np.dot(Q_inverse,u[7]*u[8]) + sigma_6[0,5,5]*np.dot(Q_inverse,u[7])**2 + \
	# 	sigma_6[0,6,0]*np.dot(Q_inverse,u[6]*u[12]) + sigma_6[0,6,1]*np.dot(Q_inverse,u[6]*u[11]) + sigma_6[0,6,2]*np.dot(Q_inverse,u[6]*u[10]) + sigma_6[0,6,3]*np.dot(Q_inverse,u[6]*u[9]) + sigma_6[0,6,4]*np.dot(Q_inverse,u[6]*u[8]) + sigma_6[0,6,5]*np.dot(Q_inverse,u[6]*u[7]) + sigma_6[0,6,6]*np.dot(Q_inverse,u[6])**2

	# beta[1] = sigma_6[1,0,0]*np.dot(Q_inverse,u[11])**2 + \
	# 	sigma_6[1,1,0]*np.dot(Q_inverse,u[10]*u[11]) + sigma_6[1,1,1]*np.dot(Q_inverse,u[10])**2 + \
	# 	sigma_6[1,2,0]*np.dot(Q_inverse,u[9]*u[11]) + sigma_6[1,2,1]*np.dot(Q_inverse,u[9]*u[10]) + sigma_6[1,2,2]*np.dot(Q_inverse,u[9])**2 + \
	# 	sigma_6[1,3,0]*np.dot(Q_inverse,u[8]*u[11]) + sigma_6[1,3,1]*np.dot(Q_inverse,u[8]*u[10]) + sigma_6[1,3,2]*np.dot(Q_inverse,u[8]*u[9]) + sigma_6[1,3,3]*np.dot(Q_inverse,u[8])**2 + \
	# 	sigma_6[1,4,0]*np.dot(Q_inverse,u[7]*u[11]) + sigma_6[1,4,1]*np.dot(Q_inverse,u[7]*u[10]) + sigma_6[1,4,2]*np.dot(Q_inverse,u[7]*u[9]) + sigma_6[1,4,3]*np.dot(Q_inverse,u[7]*u[8]) + sigma_6[1,4,4]*np.dot(Q_inverse,u[7])**2 + \
	# 	sigma_6[1,5,0]*np.dot(Q_inverse,u[6]*u[11]) + sigma_6[1,5,1]*np.dot(Q_inverse,u[6]*u[10]) + sigma_6[1,5,2]*np.dot(Q_inverse,u[6]*u[9]) + sigma_6[1,5,3]*np.dot(Q_inverse,u[6]*u[8]) + sigma_6[1,5,4]*np.dot(Q_inverse,u[6]*u[7]) + sigma_6[1,5,5]*np.dot(Q_inverse,u[6])**2 + \
	# 	sigma_6[1,6,0]*np.dot(Q_inverse,u[5]*u[11]) + sigma_6[1,6,1]*np.dot(Q_inverse,u[5]*u[10]) + sigma_6[1,6,2]*np.dot(Q_inverse,u[5]*u[9]) + sigma_6[1,6,3]*np.dot(Q_inverse,u[5]*u[8]) + sigma_6[1,6,4]*np.dot(Q_inverse,u[5]*u[7]) + sigma_6[1,6,5]*np.dot(Q_inverse,u[5]*u[6]) + sigma_6[1,6,6]*np.dot(Q_inverse,u[5])**2

	# beta[2] = sigma_6[2,0,0]*np.dot(Q_inverse,u[10])**2 + \
	# 	sigma_6[2,1,0]*np.dot(Q_inverse,u[9]*u[10]) + sigma_6[2,1,1]*np.dot(Q_inverse,u[9])**2 + \
	# 	sigma_6[2,2,0]*np.dot(Q_inverse,u[8]*u[10]) + sigma_6[2,2,1]*np.dot(Q_inverse,u[8]*u[9]) + sigma_6[2,2,2]*np.dot(Q_inverse,u[8])**2 + \
	# 	sigma_6[2,3,0]*np.dot(Q_inverse,u[7]*u[10]) + sigma_6[2,3,1]*np.dot(Q_inverse,u[7]*u[9]) + sigma_6[2,3,2]*np.dot(Q_inverse,u[7]*u[8]) + sigma_6[2,3,3]*np.dot(Q_inverse,u[7])**2 + \
	# 	sigma_6[2,4,0]*np.dot(Q_inverse,u[6]*u[10]) + sigma_6[2,4,1]*np.dot(Q_inverse,u[6]*u[9]) + sigma_6[2,4,2]*np.dot(Q_inverse,u[6]*u[8]) + sigma_6[2,4,3]*np.dot(Q_inverse,u[6]*u[7]) + sigma_6[2,4,4]*np.dot(Q_inverse,u[6])**2 + \
	# 	sigma_6[2,5,0]*np.dot(Q_inverse,u[5]*u[10]) + sigma_6[2,5,1]*np.dot(Q_inverse,u[5]*u[9]) + sigma_6[2,5,2]*np.dot(Q_inverse,u[5]*u[8]) + sigma_6[2,5,3]*np.dot(Q_inverse,u[5]*u[7]) + sigma_6[2,5,4]*np.dot(Q_inverse,u[5]*u[6]) + sigma_6[2,5,5]*np.dot(Q_inverse,u[5])**2 + \
	# 	sigma_6[2,6,0]*np.dot(Q_inverse,u[4]*u[10]) + sigma_6[2,6,1]*np.dot(Q_inverse,u[4]*u[9]) + sigma_6[2,6,2]*np.dot(Q_inverse,u[4]*u[8]) + sigma_6[2,6,3]*np.dot(Q_inverse,u[4]*u[7]) + sigma_6[2,6,4]*np.dot(Q_inverse,u[4]*u[6]) + sigma_6[2,6,5]*np.dot(Q_inverse,u[4]*u[5]) + sigma_6[2,6,6]*np.dot(Q_inverse,u[4])**2

	# beta[3] = sigma_6[3,0,0]*np.dot(Q_inverse,u[9])**2 + \
	# 	sigma_6[3,1,0]*np.dot(Q_inverse,u[8]*u[9]) + sigma_6[3,1,1]*np.dot(Q_inverse,u[8])**2 + \
	# 	sigma_6[3,2,0]*np.dot(Q_inverse,u[7]*u[9]) + sigma_6[3,2,1]*np.dot(Q_inverse,u[7]*u[8]) + sigma_6[3,2,2]*np.dot(Q_inverse,u[7])**2 + \
	# 	sigma_6[3,3,0]*np.dot(Q_inverse,u[6]*u[9]) + sigma_6[3,3,1]*np.dot(Q_inverse,u[6]*u[8]) + sigma_6[3,3,2]*np.dot(Q_inverse,u[6]*u[7]) + sigma_6[3,3,3]*np.dot(Q_inverse,u[6])**2 + \
	# 	sigma_6[3,4,0]*np.dot(Q_inverse,u[5]*u[9]) + sigma_6[3,4,1]*np.dot(Q_inverse,u[5]*u[8]) + sigma_6[3,4,2]*np.dot(Q_inverse,u[5]*u[7]) + sigma_6[3,4,3]*np.dot(Q_inverse,u[5]*u[6]) + sigma_6[3,4,4]*np.dot(Q_inverse,u[5])**2 + \
	# 	sigma_6[3,5,0]*np.dot(Q_inverse,u[4]*u[9]) + sigma_6[3,5,1]*np.dot(Q_inverse,u[4]*u[8]) + sigma_6[3,5,2]*np.dot(Q_inverse,u[4]*u[7]) + sigma_6[3,5,3]*np.dot(Q_inverse,u[4]*u[6]) + sigma_6[3,5,4]*np.dot(Q_inverse,u[4]*u[5]) + sigma_6[3,5,5]*np.dot(Q_inverse,u[4])**2 + \
	# 	sigma_6[3,6,0]*np.dot(Q_inverse,u[3]*u[9]) + sigma_6[3,6,1]*np.dot(Q_inverse,u[3]*u[8]) + sigma_6[3,6,2]*np.dot(Q_inverse,u[3]*u[7]) + sigma_6[3,6,3]*np.dot(Q_inverse,u[3]*u[6]) + sigma_6[3,6,4]*np.dot(Q_inverse,u[3]*u[5]) + sigma_6[3,6,5]*np.dot(Q_inverse,u[3]*u[4]) + sigma_6[3,6,6]*np.dot(Q_inverse,u[3])**2

	# beta[4] = sigma_6[4,0,0]*np.dot(Q_inverse,u[8])**2 + \
	# 	sigma_6[4,1,0]*np.dot(Q_inverse,u[7]*u[8]) + sigma_6[4,1,1]*np.dot(Q_inverse,u[7])**2 + \
	# 	sigma_6[4,2,0]*np.dot(Q_inverse,u[6]*u[8]) + sigma_6[4,2,1]*np.dot(Q_inverse,u[6]*u[7]) + sigma_6[4,2,2]*np.dot(Q_inverse,u[6])**2 + \
	# 	sigma_6[4,3,0]*np.dot(Q_inverse,u[5]*u[8]) + sigma_6[4,3,1]*np.dot(Q_inverse,u[5]*u[7]) + sigma_6[4,3,2]*np.dot(Q_inverse,u[5]*u[6]) + sigma_6[4,3,3]*np.dot(Q_inverse,u[5])**2 + \
	# 	sigma_6[4,4,0]*np.dot(Q_inverse,u[4]*u[8]) + sigma_6[4,4,1]*np.dot(Q_inverse,u[4]*u[7]) + sigma_6[4,4,2]*np.dot(Q_inverse,u[4]*u[6]) + sigma_6[4,4,3]*np.dot(Q_inverse,u[4]*u[5]) + sigma_6[4,4,4]*np.dot(Q_inverse,u[4])**2 + \
	# 	sigma_6[4,5,0]*np.dot(Q_inverse,u[3]*u[8]) + sigma_6[4,5,1]*np.dot(Q_inverse,u[3]*u[7]) + sigma_6[4,5,2]*np.dot(Q_inverse,u[3]*u[6]) + sigma_6[4,5,3]*np.dot(Q_inverse,u[3]*u[5]) + sigma_6[4,5,4]*np.dot(Q_inverse,u[3]*u[4]) + sigma_6[4,5,5]*np.dot(Q_inverse,u[3])**2 + \
	# 	sigma_6[4,6,0]*np.dot(Q_inverse,u[2]*u[8]) + sigma_6[4,6,1]*np.dot(Q_inverse,u[2]*u[7]) + sigma_6[4,6,2]*np.dot(Q_inverse,u[2]*u[6]) + sigma_6[4,6,3]*np.dot(Q_inverse,u[2]*u[5]) + sigma_6[4,6,4]*np.dot(Q_inverse,u[2]*u[4]) + sigma_6[4,6,5]*np.dot(Q_inverse,u[2]*u[3]) + sigma_6[4,6,6]*np.dot(Q_inverse,u[2])**2

	# beta[5] = sigma_6[5,0,0]*np.dot(Q_inverse,u[7])**2 + \
	# 	sigma_6[5,1,0]*np.dot(Q_inverse,u[6]*u[7]) + sigma_6[5,1,1]*np.dot(Q_inverse,u[6])**2 + \
	# 	sigma_6[5,2,0]*np.dot(Q_inverse,u[5]*u[7]) + sigma_6[5,2,1]*np.dot(Q_inverse,u[5]*u[6]) + sigma_6[5,2,2]*np.dot(Q_inverse,u[5])**2 + \
	# 	sigma_6[5,3,0]*np.dot(Q_inverse,u[4]*u[7]) + sigma_6[5,3,1]*np.dot(Q_inverse,u[4]*u[6]) + sigma_6[5,3,2]*np.dot(Q_inverse,u[4]*u[5]) + sigma_6[5,3,3]*np.dot(Q_inverse,u[4])**2 + \
	# 	sigma_6[5,4,0]*np.dot(Q_inverse,u[3]*u[7]) + sigma_6[5,4,1]*np.dot(Q_inverse,u[3]*u[6]) + sigma_6[5,4,2]*np.dot(Q_inverse,u[3]*u[5]) + sigma_6[5,4,3]*np.dot(Q_inverse,u[3]*u[4]) + sigma_6[5,4,4]*np.dot(Q_inverse,u[3])**2 + \
	# 	sigma_6[5,5,0]*np.dot(Q_inverse,u[2]*u[7]) + sigma_6[5,5,1]*np.dot(Q_inverse,u[2]*u[6]) + sigma_6[5,5,2]*np.dot(Q_inverse,u[2]*u[5]) + sigma_6[5,5,3]*np.dot(Q_inverse,u[2]*u[4]) + sigma_6[5,5,4]*np.dot(Q_inverse,u[2]*u[3]) + sigma_6[5,5,5]*np.dot(Q_inverse,u[2])**2 + \
	# 	sigma_6[5,6,0]*np.dot(Q_inverse,u[1]*u[7]) + sigma_6[5,6,1]*np.dot(Q_inverse,u[1]*u[6]) + sigma_6[5,6,2]*np.dot(Q_inverse,u[1]*u[5]) + sigma_6[5,6,3]*np.dot(Q_inverse,u[1]*u[4]) + sigma_6[5,6,4]*np.dot(Q_inverse,u[1]*u[3]) + sigma_6[5,6,5]*np.dot(Q_inverse,u[1]*u[2]) + sigma_6[5,6,6]*np.dot(Q_inverse,u[1])**2

	# beta[6] = sigma_6[6,0,0]*np.dot(Q_inverse,u[6])**2 + \
	# 	sigma_6[6,1,0]*np.dot(Q_inverse,u[5]*u[6]) + sigma_6[6,1,1]*np.dot(Q_inverse,u[5])**2 + \
	# 	sigma_6[6,2,0]*np.dot(Q_inverse,u[4]*u[6]) + sigma_6[6,2,1]*np.dot(Q_inverse,u[4]*u[5]) + sigma_6[6,2,2]*np.dot(Q_inverse,u[4])**2 + \
	# 	sigma_6[6,3,0]*np.dot(Q_inverse,u[3]*u[6]) + sigma_6[6,3,1]*np.dot(Q_inverse,u[3]*u[5]) + sigma_6[6,3,2]*np.dot(Q_inverse,u[3]*u[4]) + sigma_6[6,3,3]*np.dot(Q_inverse,u[3])**2 + \
	# 	sigma_6[6,4,0]*np.dot(Q_inverse,u[2]*u[6]) + sigma_6[6,4,1]*np.dot(Q_inverse,u[2]*u[5]) + sigma_6[6,4,2]*np.dot(Q_inverse,u[2]*u[4]) + sigma_6[6,4,3]*np.dot(Q_inverse,u[2]*u[3]) + sigma_6[6,4,4]*np.dot(Q_inverse,u[2])**2 + \
	# 	sigma_6[6,5,0]*np.dot(Q_inverse,u[1]*u[6]) + sigma_6[6,5,1]*np.dot(Q_inverse,u[1]*u[5]) + sigma_6[6,5,2]*np.dot(Q_inverse,u[1]*u[4]) + sigma_6[6,5,3]*np.dot(Q_inverse,u[1]*u[3]) + sigma_6[6,5,4]*np.dot(Q_inverse,u[1]*u[2]) + sigma_6[6,5,5]*np.dot(Q_inverse,u[1])**2 + \
	# 	sigma_6[6,6,0]*np.dot(Q_inverse,u[0]*u[6]) + sigma_6[6,6,1]*np.dot(Q_inverse,u[0]*u[5]) + sigma_6[6,6,2]*np.dot(Q_inverse,u[0]*u[4]) + sigma_6[6,6,3]*np.dot(Q_inverse,u[0]*u[3]) + sigma_6[6,6,4]*np.dot(Q_inverse,u[0]*u[2]) + sigma_6[6,6,5]*np.dot(Q_inverse,u[0]*u[1]) + sigma_6[6,6,6]*np.dot(Q_inverse,u[0])**2
	
@njit
def calculate_beta_7(u, r, Q_inverse):
	beta = np.zeros((r+1,3))
	for k_s in range(r+1):
		for l in range(r+1):
			for m in range(l+1):
				beta[k_s] += sigma_7[k_s,l,m]*np.dot(Q_inverse, u[2*r-k_s-l])*np.dot(Q_inverse, u[2*r-k_s-m])
	return beta

# 	beta[0] = sigma_7[0,0,0]*u_plus[6]**2 + \
# 		sigma_7[0,1,0]*u_plus[5]*u_plus[6] + sigma_7[0,1,1]*u_plus[5]**2 + \
# 		sigma_7[0,2,0]*u_plus[4]*u_plus[6] + sigma_7[0,2,1]*u_plus[4]*u_plus[5] + sigma_7[0,2,2]*u_plus[4]**2 + \
# 		sigma_7[0,3,0]*u_plus[3]*u_plus[6] + sigma_7[0,3,1]*u_plus[3]*u_plus[5] + sigma_7[0,3,2]*u_plus[3]*u_plus[4] + sigma_7[0,3,3]*u_plus[3]**2 + \
# 		sigma_7[0,4,0]*u_plus[2]*u_plus[6] + sigma_7[0,4,1]*u_plus[2]*u_plus[5] + sigma_7[0,4,2]*u_plus[2]*u_plus[4] + sigma_7[0,4,3]*u_plus[2]*u_plus[3] + sigma_7[0,4,4]*u_plus[2]**2 + \
# 		sigma_7[0,5,0]*u_plus[1]*u_plus[6] + sigma_7[0,5,1]*u_plus[1]*u_plus[5] + sigma_7[0,5,2]*u_plus[1]*u_plus[4] + sigma_7[0,5,3]*u_plus[1]*u_plus[3] + sigma_7[0,5,4]*u_plus[1]*u_plus[2] + sigma_7[0,5,5]*u_plus[1]**2 + \
# 		sigma_7[0,6,0]*u_plus[0]*u_plus[6] + sigma_7[0,6,1]*u_plus[0]*u_plus[5] + sigma_7[0,6,2]*u_plus[0]*u_plus[4] + sigma_7[0,6,3]*u_plus[0]*u_plus[3] + sigma_7[0,6,4]*u_plus[0]*u_plus[2] + sigma_7[0,6,5]*u_plus[0]*u_plus[1] + sigma_7[0,6,6]*u_plus[0]**2 + \
# 		sigma_7[0,7,0]*u*u_plus[6]         + sigma_7[0,7,1]*u*u_plus[5]         + sigma_7[0,7,2]*u*u_plus[4]         + sigma_7[0,7,3]*u*u_plus[3]         + sigma_7[0,7,4]*u*u_plus[2]         + sigma_7[0,7,5]*u*u_plus[1]         + sigma_7[0,7,6]*u*u_plus[0]  + sigma_7[0,7,7]*u**2

# 	beta[1] = sigma_7[1,0,0]*u_plus[5]**2 + \
# 		sigma_7[1,1,0]*u_plus[4]*u_plus[5]  + sigma_7[1,1,1]*u_plus[4]**2 + \
# 		sigma_7[1,2,0]*u_plus[3]*u_plus[5]  + sigma_7[1,2,1]*u_plus[3]*u_plus[4]  + sigma_7[1,2,2]*u_plus[3]**2 + \
# 		sigma_7[1,3,0]*u_plus[2]*u_plus[5]  + sigma_7[1,3,1]*u_plus[2]*u_plus[4]  + sigma_7[1,3,2]*u_plus[2]*u_plus[3]  + sigma_7[1,3,3]*u_plus[2]**2 + \
# 		sigma_7[1,4,0]*u_plus[1]*u_plus[5]  + sigma_7[1,4,1]*u_plus[1]*u_plus[4]  + sigma_7[1,4,2]*u_plus[1]*u_plus[3]  + sigma_7[1,4,3]*u_plus[1]*u_plus[2]  + sigma_7[1,4,4]*u_plus[1]**2 + \
# 		sigma_7[1,5,0]*u_plus[0]*u_plus[5]  + sigma_7[1,5,1]*u_plus[0]*u_plus[4]  + sigma_7[1,5,2]*u_plus[0]*u_plus[3]  + sigma_7[1,5,3]*u_plus[0]*u_plus[2]  + sigma_7[1,5,4]*u_plus[0]*u_plus[1]  + sigma_7[1,5,5]*u_plus[0]**2 + \
# 		sigma_7[1,6,0]*u*u_plus[5]          + sigma_7[1,6,1]*u*u_plus[4]          + sigma_7[1,6,2]*u*u_plus[3]          + sigma_7[1,6,3]*u*u_plus[2]          + sigma_7[1,6,4]*u*u_plus[1]          + sigma_7[1,6,5]*u*u_plus[0]          + sigma_7[1,6,6]*u**2 + \
# 		sigma_7[1,7,0]*u_minus[0]*u_plus[5] + sigma_7[1,7,1]*u_minus[0]*u_plus[4] + sigma_7[1,7,2]*u_minus[0]*u_plus[3] + sigma_7[1,7,3]*u_minus[0]*u_plus[2] + sigma_7[1,7,4]*u_minus[0]*u_plus[1] + sigma_7[1,7,5]*u_minus[0]*u_plus[0] + sigma_7[1,7,6]*u_minus[0]*u + sigma_7[1,7,7]*u_minus[0]**2	

# 	beta[2] = sigma_7[2,0,0]*u_plus[4]**2 + \
# 		sigma_7[2,1,0]*u_plus[3]*u_plus[4]  + sigma_7[2,1,1]*u_plus[3]**2 + \
# 		sigma_7[2,2,0]*u_plus[2]*u_plus[4]  + sigma_7[2,2,1]*u_plus[2]*u_plus[3]  + sigma_7[2,2,2]*u_plus[2]**2 + \
# 		sigma_7[2,3,0]*u_plus[1]*u_plus[4]  + sigma_7[2,3,1]*u_plus[1]*u_plus[3]  + sigma_7[2,3,2]*u_plus[1]*u_plus[2]  + sigma_7[2,3,3]*u_plus[1]**2 + \
# 		sigma_7[2,4,0]*u_plus[0]*u_plus[4]  + sigma_7[2,4,1]*u_plus[0]*u_plus[3]  + sigma_7[2,4,2]*u_plus[0]*u_plus[2]  + sigma_7[2,4,3]*u_plus[0]*u_plus[1]  + sigma_7[2,4,4]*u_plus[0]**2 + \
# 		sigma_7[2,5,0]*u*u_plus[4]          + sigma_7[2,5,1]*u*u_plus[3]          + sigma_7[2,5,2]*u*u_plus[2]          + sigma_7[2,5,3]*u*u_plus[1]          + sigma_7[2,5,4]*u*u_plus[0]          + sigma_7[2,5,5]*u**2 + \
# 		sigma_7[2,6,0]*u_minus[0]*u_plus[4] + sigma_7[2,6,1]*u_minus[0]*u_plus[3] + sigma_7[2,6,2]*u_minus[0]*u_plus[2] + sigma_7[2,6,3]*u_minus[0]*u_plus[1] + sigma_7[2,6,4]*u_minus[0]*u_plus[0] + sigma_7[2,6,5]*u_minus[0]*u + sigma_7[2,6,6]*u_minus[0]**2 + \
# 		sigma_7[2,7,0]*u_minus[1]*u_plus[4] + sigma_7[2,7,1]*u_minus[1]*u_plus[3] + sigma_7[2,7,2]*u_minus[1]*u_plus[2] + sigma_7[2,7,3]*u_minus[1]*u_plus[1] + sigma_7[2,7,4]*u_minus[1]*u_plus[0] + sigma_7[2,7,5]*u_minus[1]*u + sigma_7[2,7,6]*u_minus[1]*u_minus[0] + sigma_7[2,7,7]*u_minus[1]**2

# 	beta[3] = sigma_7[3,0,0]*u_plus[3]**2 + \
# 		sigma_7[3,1,0]*u_plus[2]*u_plus[3]  + sigma_7[3,1,1]*u_plus[2]**2 + \
# 		sigma_7[3,2,0]*u_plus[1]*u_plus[3]  + sigma_7[3,2,1]*u_plus[1]*u_plus[2]  + sigma_7[3,2,2]*u_plus[1]**2 + \
# 		sigma_7[3,3,0]*u_plus[0]*u_plus[3]  + sigma_7[3,3,1]*u_plus[0]*u_plus[2]  + sigma_7[3,3,2]*u_plus[0]*u_plus[1]  + sigma_7[3,3,3]*u_plus[0]**2 + \
# 		sigma_7[3,4,0]*u*u_plus[3]          + sigma_7[3,4,1]*u*u_plus[2]          + sigma_7[3,4,2]*u*u_plus[1]          + sigma_7[3,4,3]*u*u_plus[0]          + sigma_7[3,4,4]*u**2 + \
# 		sigma_7[3,5,0]*u_minus[0]*u_plus[3] + sigma_7[3,5,1]*u_minus[0]*u_plus[2] + sigma_7[3,5,2]*u_minus[0]*u_plus[1] + sigma_7[3,5,3]*u_minus[0]*u_plus[0] + sigma_7[3,5,4]*u_minus[0]*u + sigma_7[3,5,5]*u_minus[0]**2 + \
# 		sigma_7[3,6,0]*u_minus[1]*u_plus[3] + sigma_7[3,6,1]*u_minus[1]*u_plus[2] + sigma_7[3,6,2]*u_minus[1]*u_plus[1] + sigma_7[3,6,3]*u_minus[1]*u_plus[0] + sigma_7[3,6,4]*u_minus[1]*u + sigma_7[3,6,5]*u_minus[1]*u_minus[0] + sigma_7[3,6,6]*u_minus[1]**2 + \
# 		sigma_7[3,7,0]*u_minus[2]*u_plus[3] + sigma_7[3,7,1]*u_minus[2]*u_plus[2] + sigma_7[3,7,2]*u_minus[2]*u_plus[1] + sigma_7[3,7,3]*u_minus[2]*u_plus[0] + sigma_7[3,7,4]*u_minus[2]*u + sigma_7[3,7,5]*u_minus[2]*u_minus[0] + sigma_7[3,7,6]*u_minus[2]*u_minus[1] + sigma_7[3,7,7]*u_minus[2]**2

# 	beta[4] = sigma_7[4,0,0]*u_plus[2]**2 + \
# 		sigma_7[4,1,0]*u_plus[1]*u_plus[2]  + sigma_7[4,1,1]*u_plus[1]**2 + \
# 		sigma_7[4,2,0]*u_plus[0]*u_plus[2]  + sigma_7[4,2,1]*u_plus[0]*u_plus[1]  + sigma_7[4,2,2]*u_plus[0]**2 + \
# 		sigma_7[4,3,0]*u*u_plus[2]          + sigma_7[4,3,1]*u*u_plus[1]          + sigma_7[4,3,2]*u*u_plus[0]          + sigma_7[4,3,3]*u**2 + \
# 		sigma_7[4,4,0]*u_minus[0]*u_plus[2] + sigma_7[4,4,1]*u_minus[0]*u_plus[1] + sigma_7[4,4,2]*u_minus[0]*u_plus[0] + sigma_7[4,4,3]*u_minus[0]*u + sigma_7[4,4,4]*u_minus[0]**2 + \
# 		sigma_7[4,5,0]*u_minus[1]*u_plus[2] + sigma_7[4,5,1]*u_minus[1]*u_plus[1] + sigma_7[4,5,2]*u_minus[1]*u_plus[0] + sigma_7[4,5,3]*u_minus[1]*u + sigma_7[4,5,4]*u_minus[1]*u_minus[0] + sigma_7[4,5,5]*u_minus[1]**2 + \
# 		sigma_7[4,6,0]*u_minus[2]*u_plus[2] + sigma_7[4,6,1]*u_minus[2]*u_plus[1] + sigma_7[4,6,2]*u_minus[2]*u_plus[0] + sigma_7[4,6,3]*u_minus[2]*u + sigma_7[4,6,4]*u_minus[2]*u_minus[0] + sigma_7[4,6,5]*u_minus[2]*u_minus[1] + sigma_7[4,6,6]*u_minus[2]**2 + \
# 		sigma_7[4,7,0]*u_minus[3]*u_plus[2] + sigma_7[4,7,1]*u_minus[3]*u_plus[1] + sigma_7[4,7,2]*u_minus[3]*u_plus[0] + sigma_7[4,7,3]*u_minus[3]*u + sigma_7[4,7,4]*u_minus[3]*u_minus[0] + sigma_7[4,7,5]*u_minus[3]*u_minus[1] + sigma_7[4,7,6]*u_minus[3]*u_minus[2] + sigma_7[4,7,7]*u_minus[3]**2

# 	beta[5] = sigma_7[5,0,0]*u_plus[1]**2 + \
# 		sigma_7[5,1,0]*u_plus[0]*u_plus[1]  + sigma_7[5,1,1]*u_plus[0]**2 + \
# 		sigma_7[5,2,0]*u*u_plus[1]          + sigma_7[5,2,1]*u*u_plus[0]          + sigma_7[5,2,2]*u**2 + \
# 		sigma_7[5,3,0]*u_minus[0]*u_plus[1] + sigma_7[5,3,1]*u_minus[0]*u_plus[0] + sigma_7[5,3,2]*u_minus[0]*u + sigma_7[5,3,3]*u_minus[0]**2 + \
# 		sigma_7[5,4,0]*u_minus[1]*u_plus[1] + sigma_7[5,4,1]*u_minus[1]*u_plus[0] + sigma_7[5,4,2]*u_minus[1]*u + sigma_7[5,4,3]*u_minus[1]*u_minus[0] + sigma_7[5,4,4]*u_minus[1]**2 + \
# 		sigma_7[5,5,0]*u_minus[2]*u_plus[1] + sigma_7[5,5,1]*u_minus[2]*u_plus[0] + sigma_7[5,5,2]*u_minus[2]*u + sigma_7[5,5,3]*u_minus[2]*u_minus[0] + sigma_7[5,5,4]*u_minus[2]*u_minus[1] + sigma_7[5,5,5]*u_minus[2]**2 + \
# 		sigma_7[5,6,0]*u_minus[3]*u_plus[1] + sigma_7[5,6,1]*u_minus[3]*u_plus[0] + sigma_7[5,6,2]*u_minus[3]*u + sigma_7[5,6,3]*u_minus[3]*u_minus[0] + sigma_7[5,6,4]*u_minus[3]*u_minus[1] + sigma_7[5,6,5]*u_minus[3]*u_minus[2] + sigma_7[5,6,6]*u_minus[3]**2 + \
# 		sigma_7[5,7,0]*u_minus[4]*u_plus[1] + sigma_7[5,7,1]*u_minus[4]*u_plus[0] + sigma_7[5,7,2]*u_minus[4]*u + sigma_7[5,7,3]*u_minus[4]*u_minus[0] + sigma_7[5,7,4]*u_minus[4]*u_minus[1] + sigma_7[5,7,5]*u_minus[4]*u_minus[2] + sigma_7[5,7,6]*u_minus[4]*u_minus[3] + sigma_7[5,7,7]*u_minus[4]**2

# 	beta[6] = sigma_7[6,0,0]*u_plus[0]**2 + \
# 		sigma_7[6,1,0]*u*u_plus[0]          + sigma_7[6,1,1]*u**2 + \
# 		sigma_7[6,2,0]*u_minus[0]*u_plus[0] + sigma_7[6,2,1]*u_minus[0]*u + sigma_7[6,2,2]*u_minus[0]**2 + \
# 		sigma_7[6,3,0]*u_minus[1]*u_plus[0] + sigma_7[6,3,1]*u_minus[1]*u + sigma_7[6,3,2]*u_minus[1]*u_minus[0] + sigma_7[6,3,3]*u_minus[1]**2 + \
# 		sigma_7[6,4,0]*u_minus[2]*u_plus[0] + sigma_7[6,4,1]*u_minus[2]*u + sigma_7[6,4,2]*u_minus[2]*u_minus[0] + sigma_7[6,4,3]*u_minus[2]*u_minus[1] + sigma_7[6,4,4]*u_minus[2]**2 + \
# 		sigma_7[6,5,0]*u_minus[3]*u_plus[0] + sigma_7[6,5,1]*u_minus[3]*u + sigma_7[6,5,2]*u_minus[3]*u_minus[0] + sigma_7[6,5,3]*u_minus[3]*u_minus[1] + sigma_7[6,5,4]*u_minus[3]*u_minus[2] + sigma_7[6,5,5]*u_minus[3]**2 + \
# 		sigma_7[6,6,0]*u_minus[4]*u_plus[0] + sigma_7[6,6,1]*u_minus[4]*u + sigma_7[6,6,2]*u_minus[4]*u_minus[0] + sigma_7[6,6,3]*u_minus[4]*u_minus[1] + sigma_7[6,6,4]*u_minus[4]*u_minus[2] + sigma_7[6,6,5]*u_minus[4]*u_minus[3] + sigma_7[6,6,6]*u_minus[4]**2 + \
# 		sigma_7[6,7,0]*u_minus[5]*u_plus[0] + sigma_7[6,7,1]*u_minus[5]*u + sigma_7[6,7,2]*u_minus[5]*u_minus[0] + sigma_7[6,7,3]*u_minus[5]*u_minus[1] + sigma_7[6,7,4]*u_minus[5]*u_minus[2] + sigma_7[6,7,5]*u_minus[5]*u_minus[3] + sigma_7[6,7,6]*u_minus[5]*u_minus[4] + sigma_7[6,7,7]*u_minus[5]**2

# 	beta[7] = sigma_7[7,0,0]*u**2 + \
# 		sigma_7[7,1,0]*u_minus[0]*u + sigma_7[7,1,1]*u_minus[0]**2 + \
# 		sigma_7[7,2,0]*u_minus[1]*u + sigma_7[7,2,1]*u_minus[1]*u_minus[0] + sigma_7[7,2,2]*u_minus[1]**2 + \
# 		sigma_7[7,3,0]*u_minus[2]*u + sigma_7[7,3,1]*u_minus[2]*u_minus[0] + sigma_7[7,3,2]*u_minus[2]*u_minus[1] + sigma_7[7,3,3]*u_minus[2]**2 + \
# 		sigma_7[7,4,0]*u_minus[3]*u + sigma_7[7,4,1]*u_minus[3]*u_minus[0] + sigma_7[7,4,2]*u_minus[3]*u_minus[1] + sigma_7[7,4,3]*u_minus[3]*u_minus[2] + sigma_7[7,4,4]*u_minus[3]**2 + \
# 		sigma_7[7,5,0]*u_minus[4]*u + sigma_7[7,5,1]*u_minus[4]*u_minus[0] + sigma_7[7,5,2]*u_minus[4]*u_minus[1] + sigma_7[7,5,3]*u_minus[4]*u_minus[2] + sigma_7[7,5,4]*u_minus[4]*u_minus[3] + sigma_7[7,5,5]*u_minus[4]**2 + \
# 		sigma_7[7,6,0]*u_minus[5]*u + sigma_7[7,6,1]*u_minus[5]*u_minus[0] + sigma_7[7,6,2]*u_minus[5]*u_minus[1] + sigma_7[7,6,3]*u_minus[5]*u_minus[2] + sigma_7[7,6,4]*u_minus[5]*u_minus[3] + sigma_7[7,6,5]*u_minus[5]*u_minus[4] + sigma_7[7,6,6]*u_minus[5]**2 + \
# 		sigma_7[7,7,0]*u_minus[6]*u + sigma_7[7,7,1]*u_minus[6]*u_minus[0] + sigma_7[7,7,2]*u_minus[6]*u_minus[1] + sigma_7[7,7,3]*u_minus[6]*u_minus[2] + sigma_7[7,7,4]*u_minus[6]*u_minus[3] + sigma_7[7,7,5]*u_minus[6]*u_minus[4] + sigma_7[7,7,6]*u_minus[6]*u_minus[5] + sigma_7[7,7,7]*u_minus[6]**2

# 	return beta

@njit
def calculate_beta_8(u, r, Q_inverse):
	beta = np.zeros((r+1,3))
	for k_s in range(r+1):
		for l in range(r+1):
			for m in range(l+1):
				beta[k_s] += sigma_8[k_s,l,m]*np.dot(Q_inverse, u[2*r-k_s-l])*np.dot(Q_inverse, u[2*r-k_s-m])
	return beta
# # @jit(nopython=True, parallel=True)

# def calculate_beta_8(u, r, Q_inverse):
# 	beta = np.empty((9, len(u), len(u[0])), dtype=np.longdouble)
# 	beta[0] = sigma_8[0,0,0]*u_plus[7]**2 + \
# 		sigma_8[0,1,0]*u_plus[6]*u_plus[7] + sigma_8[0,1,1]*u_plus[6]**2 + \
# 		sigma_8[0,2,0]*u_plus[5]*u_plus[7] + sigma_8[0,2,1]*u_plus[5]*u_plus[6] + sigma_8[0,2,2]*u_plus[5]**2 + \
# 		sigma_8[0,3,0]*u_plus[4]*u_plus[7] + sigma_8[0,3,1]*u_plus[4]*u_plus[6] + sigma_8[0,3,2]*u_plus[4]*u_plus[5] + sigma_8[0,3,3]*u_plus[4]**2 + \
# 		sigma_8[0,4,0]*u_plus[3]*u_plus[7] + sigma_8[0,4,1]*u_plus[3]*u_plus[6] + sigma_8[0,4,2]*u_plus[3]*u_plus[5] + sigma_8[0,4,3]*u_plus[3]*u_plus[4] + sigma_8[0,4,4]*u_plus[3]**2 + \
# 		sigma_8[0,5,0]*u_plus[2]*u_plus[7] + sigma_8[0,5,1]*u_plus[2]*u_plus[6] + sigma_8[0,5,2]*u_plus[2]*u_plus[5] + sigma_8[0,5,3]*u_plus[2]*u_plus[4] + sigma_8[0,5,4]*u_plus[2]*u_plus[3] + sigma_8[0,5,5]*u_plus[2]**2 + \
# 		sigma_8[0,6,0]*u_plus[1]*u_plus[7] + sigma_8[0,6,1]*u_plus[1]*u_plus[6] + sigma_8[0,6,2]*u_plus[1]*u_plus[5] + sigma_8[0,6,3]*u_plus[1]*u_plus[4] + sigma_8[0,6,4]*u_plus[1]*u_plus[3] + sigma_8[0,6,5]*u_plus[1]*u_plus[2] + sigma_8[0,6,6]*u_plus[1]**2 + \
# 		sigma_8[0,7,0]*u_plus[0]*u_plus[7] + sigma_8[0,7,1]*u_plus[0]*u_plus[6] + sigma_8[0,7,2]*u_plus[0]*u_plus[5] + sigma_8[0,7,3]*u_plus[0]*u_plus[4] + sigma_8[0,7,4]*u_plus[0]*u_plus[3] + sigma_8[0,7,5]*u_plus[0]*u_plus[2] + sigma_8[0,7,6]*u_plus[0]*u_plus[1] + sigma_8[0,7,7]*u_plus[0]**2 + \
# 		sigma_8[0,8,0]*u*u_plus[7]         + sigma_8[0,8,1]*u*u_plus[6]         + sigma_8[0,8,2]*u*u_plus[5]         + sigma_8[0,8,3]*u*u_plus[4]         + sigma_8[0,8,4]*u*u_plus[3]         + sigma_8[0,8,5]*u*u_plus[2]         + sigma_8[0,8,6]*u*u_plus[1]         + sigma_8[0,8,7]*u*u_plus[0]  + sigma_8[0,8,8]*u**2

# 	beta[1] = sigma_8[1,0,0]*u_plus[6]**2 + \
# 		sigma_8[1,1,0]*u_plus[5]*u_plus[6]  + sigma_8[1,1,1]*u_plus[5]**2 + \
# 		sigma_8[1,2,0]*u_plus[4]*u_plus[6]  + sigma_8[1,2,1]*u_plus[4]*u_plus[5]  + sigma_8[1,2,2]*u_plus[4]**2 + \
# 		sigma_8[1,3,0]*u_plus[3]*u_plus[6]  + sigma_8[1,3,1]*u_plus[3]*u_plus[5]  + sigma_8[1,3,2]*u_plus[3]*u_plus[4]  + sigma_8[1,3,3]*u_plus[3]**2 + \
# 		sigma_8[1,4,0]*u_plus[2]*u_plus[6]  + sigma_8[1,4,1]*u_plus[2]*u_plus[5]  + sigma_8[1,4,2]*u_plus[2]*u_plus[4]  + sigma_8[1,4,3]*u_plus[2]*u_plus[3]  + sigma_8[1,4,4]*u_plus[2]**2 + \
# 		sigma_8[1,5,0]*u_plus[1]*u_plus[6]  + sigma_8[1,5,1]*u_plus[1]*u_plus[5]  + sigma_8[1,5,2]*u_plus[1]*u_plus[4]  + sigma_8[1,5,3]*u_plus[1]*u_plus[3]  + sigma_8[1,5,4]*u_plus[1]*u_plus[2]  + sigma_8[1,5,5]*u_plus[1]**2 + \
# 		sigma_8[1,6,0]*u_plus[0]*u_plus[6]  + sigma_8[1,6,1]*u_plus[0]*u_plus[5]  + sigma_8[1,6,2]*u_plus[0]*u_plus[4]  + sigma_8[1,6,3]*u_plus[0]*u_plus[3]  + sigma_8[1,6,4]*u_plus[0]*u_plus[2]  + sigma_8[1,6,5]*u_plus[0]*u_plus[1]  + sigma_8[1,6,6]*u_plus[0]**2 + \
# 		sigma_8[1,7,0]*u*u_plus[6]          + sigma_8[1,7,1]*u*u_plus[5]          + sigma_8[1,7,2]*u*u_plus[4]          + sigma_8[1,7,3]*u*u_plus[3]          + sigma_8[1,7,4]*u*u_plus[2]          + sigma_8[1,7,5]*u*u_plus[1]          + sigma_8[1,7,6]*u*u_plus[0]          + sigma_8[1,7,7]*u**2 + \
# 		sigma_8[1,8,0]*u_minus[0]*u_plus[6] + sigma_8[1,8,1]*u_minus[0]*u_plus[5] + sigma_8[1,8,2]*u_minus[0]*u_plus[4] + sigma_8[1,8,3]*u_minus[0]*u_plus[3] + sigma_8[1,8,4]*u_minus[0]*u_plus[2] + sigma_8[1,8,5]*u_minus[0]*u_plus[1] + sigma_8[1,8,6]*u_minus[0]*u_plus[0] + sigma_8[1,8,7]*u_minus[0]*u + sigma_8[1,8,8]*u_minus[0]**2

# 	beta[2] = sigma_8[2,0,0]*u_plus[5]**2 + \
# 		sigma_8[2,1,0]*u_plus[4]*u_plus[5]  + sigma_8[2,1,1]*u_plus[4]**2 + \
# 		sigma_8[2,2,0]*u_plus[3]*u_plus[5]  + sigma_8[2,2,1]*u_plus[3]*u_plus[4]  + sigma_8[2,2,2]*u_plus[3]**2 + \
# 		sigma_8[2,3,0]*u_plus[2]*u_plus[5]  + sigma_8[2,3,1]*u_plus[2]*u_plus[4]  + sigma_8[2,3,2]*u_plus[2]*u_plus[3]  + sigma_8[2,3,3]*u_plus[2]**2 + \
# 		sigma_8[2,4,0]*u_plus[1]*u_plus[5]  + sigma_8[2,4,1]*u_plus[1]*u_plus[4]  + sigma_8[2,4,2]*u_plus[1]*u_plus[3]  + sigma_8[2,4,3]*u_plus[1]*u_plus[2]  + sigma_8[2,4,4]*u_plus[1]**2 + \
# 		sigma_8[2,5,0]*u_plus[0]*u_plus[5]  + sigma_8[2,5,1]*u_plus[0]*u_plus[4]  + sigma_8[2,5,2]*u_plus[0]*u_plus[3]  + sigma_8[2,5,3]*u_plus[0]*u_plus[2]  + sigma_8[2,5,4]*u_plus[0]*u_plus[1]  + sigma_8[2,5,5]*u_plus[0]**2 + \
# 		sigma_8[2,6,0]*u*u_plus[5]          + sigma_8[2,6,1]*u*u_plus[4]          + sigma_8[2,6,2]*u*u_plus[3]          + sigma_8[2,6,3]*u*u_plus[2]          + sigma_8[2,6,4]*u*u_plus[1]          + sigma_8[2,6,5]*u*u_plus[0]          + sigma_8[2,6,6]*u**2 + \
# 		sigma_8[2,7,0]*u_minus[0]*u_plus[5] + sigma_8[2,7,1]*u_minus[0]*u_plus[4] + sigma_8[2,7,2]*u_minus[0]*u_plus[3] + sigma_8[2,7,3]*u_minus[0]*u_plus[2] + sigma_8[2,7,4]*u_minus[0]*u_plus[1] + sigma_8[2,7,5]*u_minus[0]*u_plus[0] + sigma_8[2,7,6]*u_minus[0]*u + sigma_8[2,7,7]*u_minus[0]**2 + \
# 		sigma_8[2,8,0]*u_minus[1]*u_plus[5] + sigma_8[2,8,1]*u_minus[1]*u_plus[4] + sigma_8[2,8,2]*u_minus[1]*u_plus[3] + sigma_8[2,8,3]*u_minus[1]*u_plus[2] + sigma_8[2,8,4]*u_minus[1]*u_plus[1] + sigma_8[2,8,5]*u_minus[1]*u_plus[0] + sigma_8[2,8,6]*u_minus[1]*u + sigma_8[2,8,7]*u_minus[1]*u_minus[0] + sigma_8[2,8,8]*u_minus[1]**2
	
# 	beta[3] = sigma_8[3,0,0]*u_plus[4]**2 + \
# 		sigma_8[3,1,0]*u_plus[3]*u_plus[4]  + sigma_8[3,1,1]*u_plus[3]**2 + \
# 		sigma_8[3,2,0]*u_plus[2]*u_plus[4]  + sigma_8[3,2,1]*u_plus[2]*u_plus[3]  + sigma_8[3,2,2]*u_plus[2]**2 + \
# 		sigma_8[3,3,0]*u_plus[1]*u_plus[4]  + sigma_8[3,3,1]*u_plus[1]*u_plus[3]  + sigma_8[3,3,2]*u_plus[1]*u_plus[2]  + sigma_8[3,3,3]*u_plus[1]**2 + \
# 		sigma_8[3,4,0]*u_plus[0]*u_plus[4]  + sigma_8[3,4,1]*u_plus[0]*u_plus[3]  + sigma_8[3,4,2]*u_plus[0]*u_plus[2]  + sigma_8[3,4,3]*u_plus[0]*u_plus[1]  + sigma_8[3,4,4]*u_plus[0]**2 + \
# 		sigma_8[3,5,0]*u*u_plus[4]          + sigma_8[3,5,1]*u*u_plus[3]          + sigma_8[3,5,2]*u*u_plus[2]          + sigma_8[3,5,3]*u*u_plus[1]          + sigma_8[3,5,4]*u*u_plus[0]          + sigma_8[3,5,5]*u**2 + \
# 		sigma_8[3,6,0]*u_minus[0]*u_plus[4] + sigma_8[3,6,1]*u_minus[0]*u_plus[3] + sigma_8[3,6,2]*u_minus[0]*u_plus[2] + sigma_8[3,6,3]*u_minus[0]*u_plus[1] + sigma_8[3,6,4]*u_minus[0]*u_plus[0] + sigma_8[3,6,5]*u_minus[0]*u + sigma_8[3,6,6]*u_minus[0]**2 + \
# 		sigma_8[3,7,0]*u_minus[1]*u_plus[4] + sigma_8[3,7,1]*u_minus[1]*u_plus[3] + sigma_8[3,7,2]*u_minus[1]*u_plus[2] + sigma_8[3,7,3]*u_minus[1]*u_plus[1] + sigma_8[3,7,4]*u_minus[1]*u_plus[0] + sigma_8[3,7,5]*u_minus[1]*u + sigma_8[3,7,6]*u_minus[1]*u_minus[0] + sigma_8[3,7,7]*u_minus[1]**2 + \
# 		sigma_8[3,8,0]*u_minus[2]*u_plus[4] + sigma_8[3,8,1]*u_minus[2]*u_plus[3] + sigma_8[3,8,2]*u_minus[2]*u_plus[2] + sigma_8[3,8,3]*u_minus[2]*u_plus[1] + sigma_8[3,8,4]*u_minus[2]*u_plus[0] + sigma_8[3,8,5]*u_minus[2]*u + sigma_8[3,8,6]*u_minus[2]*u_minus[0] + sigma_8[3,8,7]*u_minus[2]*u_minus[1] + sigma_8[3,8,8]*u_minus[2]**2

# 	beta[4] = sigma_8[4,0,0]*u_plus[3]**2 + \
# 		sigma_8[4,1,0]*u_plus[2]*u_plus[3]  + sigma_8[4,1,1]*u_plus[2]**2 + \
# 		sigma_8[4,2,0]*u_plus[1]*u_plus[3]  + sigma_8[4,2,1]*u_plus[1]*u_plus[2]  + sigma_8[4,2,2]*u_plus[1]**2 + \
# 		sigma_8[4,3,0]*u_plus[0]*u_plus[3]  + sigma_8[4,3,1]*u_plus[0]*u_plus[2]  + sigma_8[4,3,2]*u_plus[0]*u_plus[1]  + sigma_8[4,3,3]*u_plus[0]**2 + \
# 		sigma_8[4,4,0]*u*u_plus[3]          + sigma_8[4,4,1]*u*u_plus[2]          + sigma_8[4,4,2]*u*u_plus[1]          + sigma_8[4,4,3]*u*u_plus[0]          + sigma_8[4,4,4]*u**2 + \
# 		sigma_8[4,5,0]*u_minus[0]*u_plus[3] + sigma_8[4,5,1]*u_minus[0]*u_plus[2] + sigma_8[4,5,2]*u_minus[0]*u_plus[1] + sigma_8[4,5,3]*u_minus[0]*u_plus[0] + sigma_8[4,5,4]*u_minus[0]*u + sigma_8[4,5,5]*u_minus[0]**2 + \
# 		sigma_8[4,6,0]*u_minus[1]*u_plus[3] + sigma_8[4,6,1]*u_minus[1]*u_plus[2] + sigma_8[4,6,2]*u_minus[1]*u_plus[1] + sigma_8[4,6,3]*u_minus[1]*u_plus[0] + sigma_8[4,6,4]*u_minus[1]*u + sigma_8[4,6,5]*u_minus[1]*u_minus[0] + sigma_8[4,6,6]*u_minus[1]**2 + \
# 		sigma_8[4,7,0]*u_minus[2]*u_plus[3] + sigma_8[4,7,1]*u_minus[2]*u_plus[2] + sigma_8[4,7,2]*u_minus[2]*u_plus[1] + sigma_8[4,7,3]*u_minus[2]*u_plus[0] + sigma_8[4,7,4]*u_minus[2]*u + sigma_8[4,7,5]*u_minus[2]*u_minus[0] + sigma_8[4,7,6]*u_minus[2]*u_minus[1] + sigma_8[4,7,7]*u_minus[2]**2 + \
# 		sigma_8[4,8,0]*u_minus[3]*u_plus[3] + sigma_8[4,8,1]*u_minus[3]*u_plus[2] + sigma_8[4,8,2]*u_minus[3]*u_plus[1] + sigma_8[4,8,3]*u_minus[3]*u_plus[0] + sigma_8[4,8,4]*u_minus[3]*u + sigma_8[4,8,5]*u_minus[3]*u_minus[0] + sigma_8[4,8,6]*u_minus[3]*u_minus[1] + sigma_8[4,8,7]*u_minus[3]*u_minus[2] + sigma_8[4,8,8]*u_minus[3]**2
	
# 	beta[5] = sigma_8[5,0,0]*u_plus[2]**2 + \
# 		sigma_8[5,1,0]*u_plus[1]*u_plus[2]  + sigma_8[5,1,1]*u_plus[1]**2 + \
# 		sigma_8[5,2,0]*u_plus[0]*u_plus[2]  + sigma_8[5,2,1]*u_plus[0]*u_plus[1]  + sigma_8[5,2,2]*u_plus[0]**2 + \
# 		sigma_8[5,3,0]*u*u_plus[2]          + sigma_8[5,3,1]*u*u_plus[1]          + sigma_8[5,3,2]*u*u_plus[0]          + sigma_8[5,3,3]*u**2 + \
# 		sigma_8[5,4,0]*u_minus[0]*u_plus[2] + sigma_8[5,4,1]*u_minus[0]*u_plus[1] + sigma_8[5,4,2]*u_minus[0]*u_plus[0] + sigma_8[5,4,3]*u_minus[0]*u + sigma_8[5,4,4]*u_minus[0]**2 + \
# 		sigma_8[5,5,0]*u_minus[1]*u_plus[2] + sigma_8[5,5,1]*u_minus[1]*u_plus[1] + sigma_8[5,5,2]*u_minus[1]*u_plus[0] + sigma_8[5,5,3]*u_minus[1]*u + sigma_8[5,5,4]*u_minus[1]*u_minus[0] + sigma_8[5,5,5]*u_minus[1]**2 + \
# 		sigma_8[5,6,0]*u_minus[2]*u_plus[2] + sigma_8[5,6,1]*u_minus[2]*u_plus[1] + sigma_8[5,6,2]*u_minus[2]*u_plus[0] + sigma_8[5,6,3]*u_minus[2]*u + sigma_8[5,6,4]*u_minus[2]*u_minus[0] + sigma_8[5,6,5]*u_minus[2]*u_minus[1] + sigma_8[5,6,6]*u_minus[2]**2 + \
# 		sigma_8[5,7,0]*u_minus[3]*u_plus[2] + sigma_8[5,7,1]*u_minus[3]*u_plus[1] + sigma_8[5,7,2]*u_minus[3]*u_plus[0] + sigma_8[5,7,3]*u_minus[3]*u + sigma_8[5,7,4]*u_minus[3]*u_minus[0] + sigma_8[5,7,5]*u_minus[3]*u_minus[1] + sigma_8[5,7,6]*u_minus[3]*u_minus[2] + sigma_8[5,7,7]*u_minus[3]**2 + \
# 		sigma_8[5,8,0]*u_minus[4]*u_plus[2] + sigma_8[5,8,1]*u_minus[4]*u_plus[1] + sigma_8[5,8,2]*u_minus[4]*u_plus[0] + sigma_8[5,8,3]*u_minus[4]*u + sigma_8[5,8,4]*u_minus[4]*u_minus[0] + sigma_8[5,8,5]*u_minus[4]*u_minus[1] + sigma_8[5,8,6]*u_minus[4]*u_minus[2] + sigma_8[5,8,7]*u_minus[4]*u_minus[3] + sigma_8[5,8,8]*u_minus[4]**2

# 	beta[6] = sigma_8[6,0,0]*u_plus[1]**2 + \
# 		sigma_8[6,1,0]*u_plus[0]*u_plus[1]  + sigma_8[6,1,1]*u_plus[0]**2 + \
# 		sigma_8[6,2,0]*u*u_plus[1]          + sigma_8[6,2,1]*u*u_plus[0]          + sigma_8[6,2,2]*u**2 + \
# 		sigma_8[6,3,0]*u_minus[0]*u_plus[1] + sigma_8[6,3,1]*u_minus[0]*u_plus[0] + sigma_8[6,3,2]*u_minus[0]*u + sigma_8[6,3,3]*u_minus[0]**2 + \
# 		sigma_8[6,4,0]*u_minus[1]*u_plus[1] + sigma_8[6,4,1]*u_minus[1]*u_plus[0] + sigma_8[6,4,2]*u_minus[1]*u + sigma_8[6,4,3]*u_minus[1]*u_minus[0] + sigma_8[6,4,4]*u_minus[1]**2 + \
# 		sigma_8[6,5,0]*u_minus[2]*u_plus[1] + sigma_8[6,5,1]*u_minus[2]*u_plus[0] + sigma_8[6,5,2]*u_minus[2]*u + sigma_8[6,5,3]*u_minus[2]*u_minus[0] + sigma_8[6,5,4]*u_minus[2]*u_minus[1] + sigma_8[6,5,5]*u_minus[2]**2 + \
# 		sigma_8[6,6,0]*u_minus[3]*u_plus[1] + sigma_8[6,6,1]*u_minus[3]*u_plus[0] + sigma_8[6,6,2]*u_minus[3]*u + sigma_8[6,6,3]*u_minus[3]*u_minus[0] + sigma_8[6,6,4]*u_minus[3]*u_minus[1] + sigma_8[6,6,5]*u_minus[3]*u_minus[2] + sigma_8[6,6,6]*u_minus[3]**2 + \
# 		sigma_8[6,7,0]*u_minus[4]*u_plus[1] + sigma_8[6,7,1]*u_minus[4]*u_plus[0] + sigma_8[6,7,2]*u_minus[4]*u + sigma_8[6,7,3]*u_minus[4]*u_minus[0] + sigma_8[6,7,4]*u_minus[4]*u_minus[1] + sigma_8[6,7,5]*u_minus[4]*u_minus[2] + sigma_8[6,7,6]*u_minus[4]*u_minus[3] + sigma_8[6,7,7]*u_minus[4]**2 + \
# 		sigma_8[6,8,0]*u_minus[5]*u_plus[1] + sigma_8[6,8,1]*u_minus[5]*u_plus[0] + sigma_8[6,8,2]*u_minus[5]*u + sigma_8[6,8,3]*u_minus[5]*u_minus[0] + sigma_8[6,8,4]*u_minus[5]*u_minus[1] + sigma_8[6,8,5]*u_minus[5]*u_minus[2] + sigma_8[6,8,6]*u_minus[5]*u_minus[3] + sigma_8[6,8,7]*u_minus[5]*u_minus[4] + sigma_8[6,8,8]*u_minus[5]**2

# 	beta[7] = sigma_8[7,0,0]*u_plus[0]**2 + \
# 		sigma_8[7,1,0]*u*u_plus[0]          + sigma_8[7,1,1]*u**2 + \
# 		sigma_8[7,2,0]*u_minus[0]*u_plus[0] + sigma_8[7,2,1]*u_minus[0]*u + sigma_8[7,2,2]*u_minus[0]**2 + \
# 		sigma_8[7,3,0]*u_minus[1]*u_plus[0] + sigma_8[7,3,1]*u_minus[1]*u + sigma_8[7,3,2]*u_minus[1]*u_minus[0] + sigma_8[7,3,3]*u_minus[1]**2 + \
# 		sigma_8[7,4,0]*u_minus[2]*u_plus[0] + sigma_8[7,4,1]*u_minus[2]*u + sigma_8[7,4,2]*u_minus[2]*u_minus[0] + sigma_8[7,4,3]*u_minus[2]*u_minus[1] + sigma_8[7,4,4]*u_minus[2]**2 + \
# 		sigma_8[7,5,0]*u_minus[3]*u_plus[0] + sigma_8[7,5,1]*u_minus[3]*u + sigma_8[7,5,2]*u_minus[3]*u_minus[0] + sigma_8[7,5,3]*u_minus[3]*u_minus[1] + sigma_8[7,5,4]*u_minus[3]*u_minus[2] + sigma_8[7,5,5]*u_minus[3]**2 + \
# 		sigma_8[7,6,0]*u_minus[4]*u_plus[0] + sigma_8[7,6,1]*u_minus[4]*u + sigma_8[7,6,2]*u_minus[4]*u_minus[0] + sigma_8[7,6,3]*u_minus[4]*u_minus[1] + sigma_8[7,6,4]*u_minus[4]*u_minus[2] + sigma_8[7,6,5]*u_minus[4]*u_minus[3] + sigma_8[7,6,6]*u_minus[4]**2 + \
# 		sigma_8[7,7,0]*u_minus[5]*u_plus[0] + sigma_8[7,7,1]*u_minus[5]*u + sigma_8[7,7,2]*u_minus[5]*u_minus[0] + sigma_8[7,7,3]*u_minus[5]*u_minus[1] + sigma_8[7,7,4]*u_minus[5]*u_minus[2] + sigma_8[7,7,5]*u_minus[5]*u_minus[3] + sigma_8[7,7,6]*u_minus[5]*u_minus[4] + sigma_8[7,7,7]*u_minus[5]**2 + \
# 		sigma_8[7,8,0]*u_minus[6]*u_plus[0] + sigma_8[7,8,1]*u_minus[6]*u + sigma_8[7,8,2]*u_minus[6]*u_minus[0] + sigma_8[7,8,3]*u_minus[6]*u_minus[1] + sigma_8[7,8,4]*u_minus[6]*u_minus[2] + sigma_8[7,8,5]*u_minus[6]*u_minus[3] + sigma_8[7,8,6]*u_minus[6]*u_minus[4] + sigma_8[7,8,7]*u_minus[6]*u_minus[5] + sigma_8[7,8,8]*u_minus[6]**2

# 	beta[8] = sigma_8[8,0,0]*u**2 + \
# 		sigma_8[8,1,0]*u_minus[0]*u + sigma_8[8,1,1]*u_minus[0]**2 + \
# 		sigma_8[8,2,0]*u_minus[1]*u + sigma_8[8,2,1]*u_minus[1]*u_minus[0] + sigma_8[8,2,2]*u_minus[1]**2 + \
# 		sigma_8[8,3,0]*u_minus[2]*u + sigma_8[8,3,1]*u_minus[2]*u_minus[0] + sigma_8[8,3,2]*u_minus[2]*u_minus[1] + sigma_8[8,3,3]*u_minus[2]**2 + \
# 		sigma_8[8,4,0]*u_minus[3]*u + sigma_8[8,4,1]*u_minus[3]*u_minus[0] + sigma_8[8,4,2]*u_minus[3]*u_minus[1] + sigma_8[8,4,3]*u_minus[3]*u_minus[2] + sigma_8[8,4,4]*u_minus[3]**2 + \
# 		sigma_8[8,5,0]*u_minus[4]*u + sigma_8[8,5,1]*u_minus[4]*u_minus[0] + sigma_8[8,5,2]*u_minus[4]*u_minus[1] + sigma_8[8,5,3]*u_minus[4]*u_minus[2] + sigma_8[8,5,4]*u_minus[4]*u_minus[3] + sigma_8[8,5,5]*u_minus[4]**2 + \
# 		sigma_8[8,6,0]*u_minus[5]*u + sigma_8[8,6,1]*u_minus[5]*u_minus[0] + sigma_8[8,6,2]*u_minus[5]*u_minus[1] + sigma_8[8,6,3]*u_minus[5]*u_minus[2] + sigma_8[8,6,4]*u_minus[5]*u_minus[3] + sigma_8[8,6,5]*u_minus[5]*u_minus[4] + sigma_8[8,6,6]*u_minus[5]**2 + \
# 		sigma_8[8,7,0]*u_minus[6]*u + sigma_8[8,7,1]*u_minus[6]*u_minus[0] + sigma_8[8,7,2]*u_minus[6]*u_minus[1] + sigma_8[8,7,3]*u_minus[6]*u_minus[2] + sigma_8[8,7,4]*u_minus[6]*u_minus[3] + sigma_8[8,7,5]*u_minus[6]*u_minus[4] + sigma_8[8,7,6]*u_minus[6]*u_minus[5] + sigma_8[8,7,7]*u_minus[6]**2 + \
# 		sigma_8[8,8,0]*u_minus[7]*u + sigma_8[8,8,1]*u_minus[7]*u_minus[0] + sigma_8[8,8,2]*u_minus[7]*u_minus[1] + sigma_8[8,8,3]*u_minus[7]*u_minus[2] + sigma_8[8,8,4]*u_minus[7]*u_minus[3] + sigma_8[8,8,5]*u_minus[7]*u_minus[4] + sigma_8[8,8,6]*u_minus[7]*u_minus[5] + sigma_8[8,8,7]*u_minus[7]*u_minus[6] + sigma_8[8,8,8]*u_minus[7]**2
	

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

	# return beta