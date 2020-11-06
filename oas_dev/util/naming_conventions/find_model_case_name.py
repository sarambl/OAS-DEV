import sys
import useful_scit.util.log as log


# This file contains the casename for the different models
casenames_SECTIONAL={'Sectional eq.20': 'PD_SECT_CHC6_nudgeERA_ctrl',
					 'Sectional eq.18':'PD_SECT_CHC6_nudgeERA_eq18',
					 'Sectional eq.20, rednuc': 'PD_SECT_CHC6_nudgeERA_rednuc',
					 'Sectional eq.20, 1.5xBVOC':'PD_SECT_CHC6_nudgeERA_eq20_15BVOC',
					 'Original eq.18':'PD_noSECT_nudgeERA_eq18',
					 'Original eq.20':'PD_noSECT_nudgeERA_eq20',
					 'Original eq.20, 1.5xBVOC':'PD_noSECT_nudgeERA_eq20_15BVOC',
					 'Original eq.20, rednuc': 'PD_noSECT_nudgeERA_eq20_rednuc',
					 'Original eq.20,b':'PD_noSECT_nudgeERA_eq20_branch',
					 'Sect ac eq.20': 'PD_SECT_CHC6_nudgeERA_test_autocoag_addgrowth',
					 'Sect ac eq.20, rednuc': 'PD_SECT_CHC6_nudgeERA_test_autocoag_addgrowth_rednuc',
					 'Sect ac eq.20, 1.5xBVOC': 'PD_SECT_CHC6_nudgeERA_test_autocoag_addgrowth_15BVOC',
					 'Sect ac eq.18':'PD_SECT_CHC6_nudgeERA_test_autocoag_addgrowth_eq18',
					 'Sect ac eq.20, inc coag': 'PD_SECT_CHC6_nudgeERA_inc_coagNPF',
					 'Sect ac eq.20, inc coag 10%': 'PD_SECT_CHC6_nudgeERA_inc_coagNPF_10pct',
					 'Sect ac eq.20, inc coag 200%': 'PD_SECT_CHC6_nudgeERA_inc_coagNPF_200pct',
					 'Sect ac eq.20, corr NPF diam': 'PD_SECT_CHC6_nudgeERA_NPF_coag_corr_diam',
					 'Sect ac eq.20, corr NPF diam, fxdt': 'PD_SECT_CHC6_nudgeERA_NPF_coag_corr_diam_fxdt',
					 'Sect ac eq.20, corr NPF diam, fxdt, 1.5xBVOC': 'PD_SECT_CHC6_nudgeERA_NPF_coag_corr_diam_fxdt_15BVOC',
					 'Sect ac eq.20, corr NPF diam, fxdt, vdiam': 'PD_SECT_CHC6_nudgeERA_NPF_coag_corr_diam_fxdt_maxbinf',
					 'Sect ac eq.18, corr NPF diam, fxdt, vdiam': 'PD_SECT_CHC6_nudgeERA_NPF_coag_corr_diam_fxdt_maxbinf_eq18',
					 'Sect ac eq.20, corr NPF diam, fxdt, vdiam, rednuc': 'PD_SECT_CHC6_nudgeERA_NPF_coag_corr_diam_fxdt_maxbinf_rednuc',
					 'Sect ac eq.20, corr NPF diam, fxdt, vdiam, 1.5xBVOC': 'PD_SECT_CHC6_nudgeERA_NPF_coag_corr_diam_fxdt_maxbinf_15BVOC',
					 'Sect ac eq.18, corr NPF diam, fxdt': 'PD_SECT_CHC6_nudgeERA_NPF_coag_corr_diam_fxdt_eq.18',
					 'Sect ac eq.18, corr NPF diam, fxdt, deccoag': 'PD_SECT_CHC6_nudgeERA_NPF_coag_corr_diam_fxdt_inc_coag',
					 'Sect ac eq.20, corr NPF diam,1.5xBVOC': 'PD_SECT_CHC6_nudgeERA_NPF_coag_corr_diam_inc_coag_15BVOC',
					 'Sect ac eq.20, corr NPF diam, rednuc': 'PD_SECT_CHC6_nudgeERA_NPF_coag_corr_diam_rednuc',
					 'Sect ac eq.18, corr NPF diam': 'PD_SECT_CHC6_nudgeERA_NPF_coag_corr_diam_eq18',
					 'Sect ac eq.20, corr NPF diam, inc coag 200%': 'PD_SECT_CHC6_nudgeERA_NPF_coag_corr_diam_inc_coag',
					 'Sect ac eq.20,b':'PD_SECT_CHC6_nudgeERA_test_autocoag_addgrowth_branch'
					 }
casenames_dic={}
casenames_dic['NorESM']=casenames_SECTIONAL
casenames_dic['EC-Earth']={}
casenames_dic['ECHAM'] = {}



def find_name(Model, Case):

	if Model == 'NorESM':
		
		if Case == 'CTRL':
			Case_name = 'CTRL_NCS'
		elif Case == 'Yield higher':
			Case_name = 'Yield_higher_NCS'
		elif Case == 'Yield lower':
			Case_name = 'Yield_lower_NCS'
		elif Case == 'no LVSOA':
			Case_name = 'no_LVSOA_NCS'
		elif Case == 'no isoprene':
			Case_name = 'no_isoprene_NCS'
		elif Case == 'no monoterpene':
			Case_name = 'no_monoterpene_NCS'
		elif Case in casenames_dic[Model]:
			Case_name = casenames_dic[Model][Case]
		else:
			Case_name = Case
			log.ger.debug('No alternative name found for case %s' %Case)
			#print('No alternative name found for case %s' %Case)

	elif Model == 'EC-Earth':

		if Case == 'CTRL':
			Case_name = 'CTRL'
		elif Case == 'Yield higher':
			Case_name = 'ALLp50'
		elif Case == 'Yield lower':
			Case_name = 'ALLm50'
		elif Case == 'no LVSOA':
			Case_name = 'noLVSOA'
		elif Case == 'no isoprene':
			Case_name = 'noIsop'
		elif Case == 'no monoterpene':
			Case_name = 'noTerp'
		else:
			sys.exit('Error: No case name for %s and %s' %(Model, Case))
	
	elif Model == 'ECHAM':

		if Case == 'CTRL':
			Case_name = 'ctrl'
		elif Case == 'Yield higher':
			Case_name = 'incrYield'
		elif Case == 'Yield lower':
			Case_name = 'decrYield'
		elif Case == 'no LVSOA':
			Case_name = 'noELVOC'
		elif Case == 'no isoprene':
			Case_name = 'noIsoprene'
		elif Case == 'no monoterpene':
			Case_name = 'noMonoterpene'
		else:
			sys.exit('Error: No case name for %s and %s' %(Model, Case))
		
	else:
		sys.exit('Error: Model name %s does not exist' %(Model))

	return Case_name	
