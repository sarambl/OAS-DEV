# In this file the model levels of different variables in different models
import numpy as np #For scientific computig
import sys

def level(var, lev):
		
	if var == 'N_AER' or var == 'CN':
		level = lev > 850. #Extracts pressure values greater than 850hPa. 
		#level = np.argwhere(lev > 850.) #Extracts pressure values greater than 850hPa. 
		#level = len(lev) - 1  #Bottom model level, NorESM is top down
	elif var == 'N_tot':
		level = lev > 850. #Extracts pressure values greater than 850hPa. 
		#level = np.argwhere(lev > 850.) #Extracts pressure values greater than 850hPa. 
	elif var == 'CCN0.20' or var == 'CCN1.00' or var == 'CCN4' or var == 'CCN6' or var == 'CCN02' or var == 'CCN10':
		level = lev > 300. #Extracts pressure values greater than 300hPa. 
		#level = np.argwhere(lev > 300.) #Extracts pressure values greater than 300hPa. 
		#level = 0  #Bottom model level, TM5 grid is bottom up
	elif var == 'AWNC' or var == 'AREL' or var == 'cdnc' or var == 're_liq' or var == 'CDNC_2D' or var == 'REFF_2D' or var == 'CDNC' or var == 'REFFL' or var == 'CLOUD' or var == 'aclcac':
		level = lev > 300. #Extracts pressure values greater than 300hPa. 
		#level = np.argwhere(lev > 300.) #Extracts pressure values greater than 300hPa. 
		#level = np.arange(len(lev)) # All model levels
		#print 'level', level
	else:
		level = 'dummy'

	print('levels', lev, '\n', level)
	return level


def fixed_levels(model,lev_nrs):
	
	if model == 'EC-Earth':
		if np.size(lev_nrs) == 34: #TM5 variables
			hyam = get_hyam_TM5()
			hybm = get_hybm_TM5()

			lev = hyam[:] + hybm[:]*101325.0 #The pressure at the interfaces
			
			fix_lev = lev / 100.

		elif np.size(lev_nrs) == 62:
			fix_lev = np.array([51.5086, 56.6316, 61.9984, 67.5973, 73.415, 79.4434, 85.7016, 92.2162, 99.0182, 106.1445, 113.6382, 121.5502, 129.9403, 138.8558,
				148.326, 158.3816, 169.0545, 180.3786, 192.3889, 205.1222, 218.6172, 232.914, 248.0547, 264.0833, 281.0456, 298.9895, 317.9651, 338.0245,
				359.2221, 381.6144, 405.2606, 430.2069, 456.4813, 483.8505, 512.0662, 540.8577, 569.9401, 599.031, 627.9668, 656.6129, 684.8491, 712.5573,
				739.5739, 765.7697, 791.0376, 815.2774, 838.3507, 860.1516, 880.608, 899.6602, 917.2205, 933.2247, 947.6584, 960.5245, 971.8169, 981.5301,
				989.7322, 996.8732, 1002.8013, 1007.4431, 1010.8487, 1013.25])
		else:
			sys.exit('Problem in get_model_lev with %d number of levels' %np.size(lev_nrs))	
			
	elif model == 'ECHAM':
		fix_lev = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31])


	return fix_lev

def fixed_ilevels(model,lev_nrs, hyai=[], hybi=[]):
	
	if model == 'EC-Earth':
		if np.size(lev_nrs) == 34: #TM5 variables
			hyai = get_hyai_TM5()
			hybi = get_hybi_TM5()

			lev = hyai[:] + hybi[:]*101325.0 #The pressure at the interfaces
			
			fix_lev = lev / 100.


		elif np.size(lev_nrs) == 62:
			lev = hyai[:] + hybi[:]*101325.0
			fix_lev = lev / 100.
			print('WARNING! UNSURE ABOUT ILEV FOR EC-EARTH IFS VARS')
			#fix_lev = np.array([51.5086, 56.6316, 61.9984, 67.5973, 73.415, 79.4434, 85.7016, 92.2162, 99.0182, 106.1445, 113.6382, 121.5502, 129.9403, 138.8558,
			#	148.326, 158.3816, 169.0545, 180.3786, 192.3889, 205.1222, 218.6172, 232.914, 248.0547, 264.0833, 281.0456, 298.9895, 317.9651, 338.0245,
			#	359.2221, 381.6144, 405.2606, 430.2069, 456.4813, 483.8505, 512.0662, 540.8577, 569.9401, 599.031, 627.9668, 656.6129, 684.8491, 712.5573,
			#	739.5739, 765.7697, 791.0376, 815.2774, 838.3507, 860.1516, 880.608, 899.6602, 917.2205, 933.2247, 947.6584, 960.5245, 971.8169, 981.5301,
			#	989.7322, 996.8732, 1002.8013, 1007.4431, 1010.8487, 1013.25])
		else:
			sys.exit('Problem in get_model_lev with %d number of levels' %np.size(lev_nrs))	
			
	elif model == 'ECHAM':
		print('Not ready for ECHAM')
		#fix_lev = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 
		#20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31])


	return fix_lev



def get_hyai_TM5():
	hyai = np.array([0.000000, 21.413612, 76.167656, 204.637451, 450.685791, 857.945801, 1463.163940, 2292.155518, 3358.425781, 4663.776367, 6199.839355, 7341.469727, 8564.624023, 9873.560547, 11262.484375, 12713.897461, 14192.009766, 15638.053711, 16990.623047, 18191.029297, 19184.544922, 19919.796875, 20348.916016, 20319.011719, 19348.775391, 17385.595703, 14665.645508, 11543.166992, 8356.252930, 5422.802734, 3010.146973, 1297.656128, 336.772369, 6.575628, 0.000000])
	hyai = hyai[::-1]	
	#print 'hyai', hyai

	return(hyai)

def get_hybi_TM5():
	hybi = np.array([0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000055, 0.000279, 0.001000, 0.002765, 0.006322, 0.012508, 0.022189, 0.036227, 0.055474, 0.080777, 0.112979, 0.176091, 0.259554, 0.362203, 0.475016, 0.589317, 0.698224, 0.795385, 0.875518, 0.935157, 0.973466, 0.994204, 1.000000])
	
	hybi = hybi[::-1]	

	return(hybi)


def get_hyam_TM5():

	hyai = get_hyai_TM5()
	hyai = hyai[::-1]

	hyam = (hyai[0:-1] + hyai[1::]) / 2.	
	
	hyam = hyam[::-1]	

	return(hyam)


def get_hybm_TM5():
	
	hybi = get_hybi_TM5()
	hybi = hybi[::-1]

	hybm = (hybi[0:-1] + hybi[1::]) / 2.

	hybm = hybm[::-1]	

	return(hybm)

