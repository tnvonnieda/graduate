import numpy as np

sigma_6 = np.zeros((7,7,7), dtype=np.longdouble)
# Coefficients sigma_6{k_s,l,m}, used in equation (16) of, and tabulated in, Gerolymos 2009

# ----- k_s = 0 ----- #

sigma_6[0,0,0] = 62911297 / 2993760

sigma_6[0,1,0] = - 5556669277 / 19958400
sigma_6[0,1,1] = 2047941883 / 2217600

sigma_6[0,2,0] = 15476926351 / 19958400
sigma_6[0,2,1] = - 3809437823 / 739200
sigma_6[0,2,2] = 199730921 / 27720

sigma_6[0,3,0] = - 17425032203 / 14968800
sigma_6[0,3,1] = 38683385051 / 4989600
sigma_6[0,3,2] = - 21693002767 / 997920
sigma_6[0,3,3] = 49256859919 / 2993760

sigma_6[0,4,0] = 4964771899 / 4989600
sigma_6[0,4,1] = - 14734178999 / 2217600
sigma_6[0,4,2] = 8290771913 / 443520
sigma_6[0,4,3] = - 28364892607 / 997920
sigma_6[0,4,4] = 1369404749 / 110880

sigma_6[0,5,0] = - 9181961959 / 19958400 
sigma_6[0,5,1] = 3417057367 / 1108800 
sigma_6[0,5,2] = - 19308505679 / 2217600
sigma_6[0,5,3] = 66440049371 / 4989600
sigma_6[0,5,4] = - 8623431623 / 739200
sigma_6[0,5,5] = 6182612731 / 2217600

sigma_6[0,6,0] = 5391528799 / 59875200
sigma_6[0,6,1] = - 2416885043 / 3991680
sigma_6[0,6,2] = 8579309749 / 4989600
sigma_6[0,6,3] = - 39645439643 / 14968800
sigma_6[0,6,4] = 46808583631 / 19958400
sigma_6[0,6,5] = - 22763092357 / 19958400
sigma_6[0,6,6] = 897207163 / 7484400

# ----- k_s = 1 --- #

sigma_6[1,0,0] = 64361771 / 14968800

sigma_6[1,1,0] = - 377474689 / 6652800
sigma_6[1,1,1] = 1250007643 / 6652800

sigma_6[1,2,0] = 3126718481 / 19958400
sigma_6[1,2,1] = - 6932480657 / 6652800
sigma_6[1,2,2] = 53678683 / 36960

sigma_6[1,3,0] = - 3465607493 / 14968800
sigma_6[1,3,1] = 857838469 / 554400
sigma_6[1,3,2] = - 4330640057 / 997920
sigma_6[1,3,3] = 9780057169 / 2993760

sigma_6[1,4,0] = 320782183 / 1663200
sigma_6[1,4,1] = -8619440987 / 6652800
sigma_6[1,4,2] = 4868089189 / 1330560
sigma_6[1,4,3] = -616410313 / 110880
sigma_6[1,4,4] = 796358777 / 332640

sigma_6[1,5,0] = -341910757 / 3991680
sigma_6[1,5,1] = 1924032511 / 3326400
sigma_6[1,5,2] = -405382961 / 246400
sigma_6[1,5,3] = 12601009501 / 4989600
sigma_6[1,5,4] = -14684933057 / 6652800
sigma_6[1,5,5] = 127942497 / 246400

sigma_6[1,6,0] = 945155329 / 59875200
sigma_6[1,6,1] = -712745603 / 6652800
sigma_6[1,6,2] = 1531307249 / 4989600
sigma_6[1,6,3] = -7124638253 / 14968800
sigma_6[1,6,4] = 2811067067 / 6652800
sigma_6[1,6,5] = -4074544787 / 19958400
sigma_6[1,6,6] = 62911297 / 2993760

# ----- k_s = 2 ----- #

sigma_6[2,0,0] = 2627203 / 1871100

sigma_6[2,1,0] = -359321429 / 19958400
sigma_6[2,1,1] = 130013563 / 2217600

sigma_6[2,2,0] = 105706999 / 2217600
sigma_6[2,2,1] = -2096571887 / 6652800
sigma_6[2,2,2] = 143270957 / 332640

sigma_6[2,3,0] = - 995600723 / 14968800
sigma_6[2,3,1] = 2224538011 / 4989600
sigma_6[2,3,2] = -412424029 / 332640
sigma_6[2,3,3] = 2726585359 / 2993760

sigma_6[2,4,0] = 256556849 / 4989600
sigma_6[2,4,1] = -773749439 / 2217600
sigma_6[2,4,2] = 1312114459 / 1330560
sigma_6[2,4,3] = -1476618887 / 997920
sigma_6[2,4,4] = 34187317 / 55440

sigma_6[2,5,0] = - 15401629 / 739200
sigma_6[2,5,1] = 475321093 / 3326400
sigma_6[2,5,2] = -2725575317 / 6652800
sigma_6[2,5,3] = 1042531337 / 1663200
sigma_6[2,5,4] = -3573798407 / 6652800
sigma_6[2,5,5] = 806338417 / 6652800

sigma_6[2,6,0] = 8279479 / 2395008
sigma_6[2,6,1] = -95508139 / 3991680
sigma_6[2,6,2] = 115524053 / 1663200
sigma_6[2,6,3] = - 1618284323 / 14968800
sigma_6[2,6,4] = 1894705391 / 19958400
sigma_6[2,6,5] = - 295455983 / 6652800
sigma_6[2,6,6] = 64361771 / 14968800

# ----- k_s = 3 ----- #

sigma_6[3,0,0] = 2627203 / 1871100

sigma_6[3,1,0] = - 323333323 / 19958400
sigma_6[3,1,1] = 108444169 / 2217600

sigma_6[3,2,0] = 761142961 / 19958400
sigma_6[3,2,1] = - 176498513 / 739200
sigma_6[3,2,2] = 16790707 / 55440

sigma_6[3,3,0] = - 701563133 / 14968800
sigma_6[3,3,1] = 1506944981 / 4989600
sigma_6[3,3,2] = - 790531177 / 997920
sigma_6[3,3,3] = 1607739169 / 2993760

sigma_6[3,4,0] = 158544319 / 4989600
sigma_6[3,4,1] = - 464678369 / 2217600
sigma_6[3,4,2] = 250523543 / 443520
sigma_6[3,4,3] = - 790531177 / 997920
sigma_6[3,4,4] = 16790707 / 55440

sigma_6[3,5,0] = - 225623953 / 19958400
sigma_6[3,5,1] = 84263749 / 1108800
sigma_6[3,5,2] = - 464678369 / 2217600
sigma_6[3,5,3] = 1506944981 / 4989600
sigma_6[3,5,4] = - 176498513 / 739200
sigma_6[3,5,5] = 108444169 / 2217600

sigma_6[3,6,0] = 99022657 / 59875200
sigma_6[3,6,1] = - 225623953 / 19958400
sigma_6[3,6,2] = 158544319 / 4989600
sigma_6[3,6,3] = - 701563133 / 14968800
sigma_6[3,6,4] = 761142961 / 19958400
sigma_6[3,6,5] = - 323333323 / 19958400
sigma_6[3,6,6] = 2627203 / 1871100

# ----- k_s = 4 ----- #

sigma_6[4,0,0] = 64361771 / 14968800

sigma_6[4,1,0] = - 295455983 / 6652800
sigma_6[4,1,1] = 806338417 / 6652800

sigma_6[4,2,0] = 1894705391 / 19958400
sigma_6[4,2,1] = - 3573798407 / 6652800
sigma_6[4,2,2] = 34187317 / 55440

sigma_6[4,3,0] = - 1618284323 / 14968800
sigma_6[4,3,1] = 1042531337 / 1663200
sigma_6[4,3,2] = - 1476618887 / 997920
sigma_6[4,3,3] = 2726585359 / 2993760

sigma_6[4,4,0] = 115524053 / 1663200
sigma_6[4,4,1] = - 2725575317 / 6652800
sigma_6[4,4,2] = 1312114459 / 1330560
sigma_6[4,4,3] = - 412424029 / 332640
sigma_6[4,4,4] = 143270957 / 332640

sigma_6[4,5,0] = - 95508139 / 3991680
sigma_6[4,5,1] = 475321093 / 3326400
sigma_6[4,5,2] = - 773749439 / 2217600
sigma_6[4,5,3] = 2224538011 / 4989600
sigma_6[4,5,4] = - 2096571887 / 6652800
sigma_6[4,5,5] = 130013563 / 2217600
 
sigma_6[4,6,0] = 8279479 / 2395008
sigma_6[4,6,1] = - 15401629 / 739200
sigma_6[4,6,2] = 256556849 / 4989600
sigma_6[4,6,3] = - 995600723 / 14968800
sigma_6[4,6,4] = 105706999 / 2217600
sigma_6[4,6,5] = - 359321429 / 19958400
sigma_6[4,6,6] = 2627203 / 1871100

# ----- k_s = 5 ----- #

sigma_6[5,0,0] = 62911297 / 2993760

sigma_6[5,1,0] = - 4074544787 / 19958400
sigma_6[5,1,1] = 127942497 / 246400

sigma_6[5,2,0] = 2811067067 / 6652800
sigma_6[5,2,1] = - 14684933057 / 6652800
sigma_6[5,2,2] = 796358777 / 332640

sigma_6[5,3,0] = - 7124638253 / 14968800
sigma_6[5,3,1] = 12601009501 / 4989600
sigma_6[5,3,2] = - 616410313 / 110880
sigma_6[5,3,3] = 9780057169 / 2993760

sigma_6[5,4,0] = 1531307249 / 4989600
sigma_6[5,4,1] = - 405382961 / 246400
sigma_6[5,4,2] = 4868089189 / 1330560
sigma_6[5,4,3] = - 4330640057 / 997920
sigma_6[5,4,4] = 53678683 / 36960

sigma_6[5,5,0] = - 712745603 / 6652800
sigma_6[5,5,1] = 1924032511 / 3326400
sigma_6[5,5,2] = - 8619440987 / 6652800
sigma_6[5,5,3] = 857838469 / 554400
sigma_6[5,5,4] = - 6932480657 / 6652800
sigma_6[5,5,5] = 1250007643 / 6652800

sigma_6[5,6,0] = 945155329 / 59875200
sigma_6[5,6,1] = - 341910757 / 3991680
sigma_6[5,6,2] = 320782183 / 1663200
sigma_6[5,6,3] = - 3465607493 / 14968800
sigma_6[5,6,4] = 3126718481 / 19958400
sigma_6[5,6,5] = - 377474689 / 6652800
sigma_6[5,6,6] = 64361771 / 14968800

# ----- k_s = 6 ----- #

sigma_6[6,0,0] = 897207163 / 7484400

sigma_6[6,1,0] = - 22763092357 / 19958400
sigma_6[6,1,1] = 6182612731 / 2217600 

sigma_6[6,2,0] = 46808583631 / 19958400
sigma_6[6,2,1] = - 8623431623 / 739200
sigma_6[6,2,2] = 1369404749 / 110880

sigma_6[6,3,0] = - 39645439643 / 14968800
sigma_6[6,3,1] = 66440049371 / 4989600
sigma_6[6,3,2] = - 28364892607 / 997920
sigma_6[6,3,3] = 49256859919 / 2993760

sigma_6[6,4,0] = 8579309749 / 4989600
sigma_6[6,4,1] = - 19308505679 / 2217600
sigma_6[6,4,2] = 8290771913 / 443520
sigma_6[6,4,3] = - 21693002767 / 997920
sigma_6[6,4,4] = 199730921 / 27720

sigma_6[6,5,0] = - 2416885043 / 3991680
sigma_6[6,5,1] = 3417057367 / 1108800
sigma_6[6,5,2] = - 14734178999 / 2217600
sigma_6[6,5,3] = 38683385051 / 4989600
sigma_6[6,5,4] = - 3809437823 / 739200
sigma_6[6,5,5] = 2047941883 / 2217600

sigma_6[6,6,0] = 5391528799 / 59875200
sigma_6[6,6,1] = - 9181961959 / 19958400
sigma_6[6,6,2] = 4964771899 / 4989600
sigma_6[6,6,3] = - 17425032203 / 14968800
sigma_6[6,6,4] = 15476926351 / 19958400
sigma_6[6,6,5] = - 5556669277 / 19958400
sigma_6[6,6,6] = 62911297 / 2993760

