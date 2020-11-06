# This script gives a ECHAM variable names and units for variables to be used in figures
import sys

# %%
ECHAM2NorESM_dic = {'AOD_550': 'CDOD550',
                    'CCN02': 'CCN4',
                    'CCN10': 'CCN6',
                    'CN': 'N_AER',
                    'swcf': 'SWCF',
                    'lwcf': 'LWCF',
                    'NCFT': 'NCFT',
                    'NCFT_Ghan': 'NCFT_Ghan',
                    'SWCF_Ghan': 'SWCF_Ghan',
                    'LWCF_Ghan': 'LWCF_Ghan',
                    'NDAF_Ghan': 'NDAF_Ghan',
                    'isopemis': 'SFisoprene',
                    'mterpemis': 'SFmonoterp',
                    'xlvi': 'TGCLDCWP',
                    'CDNC': 'AWNC',
                    'REFFL': 'AREL',
                    'REFF_2D': 'REFF_2D',
                    'D_BURDEN_OC': 'cb_SOA_dry',
                    'temp2': 'TREFHT',
                    'aclcov': 'CLDTOT',
                    'aclcac': 'CLOUD',
                    'SOAPROD': 'condTend_SOA_total',
                    'D_BURDEN_ISOP': 'cb_isoprene',
                    'cb_monoterp': 'cb_monoterp',
                    'aps': 'PS',
                    'st': 'T'}
EC_Earth2NorESM_dic = {'od550aer': 'CDOD550',
                       'mass_frac_SOA': 'cb_frac_SOA',
                       'loadisop': 'cb_isoprene',
                       'loadterp': 'cb_monoterp',
                       'CCN0.20': 'CCN4',
                       'CCN1.00': 'CCN6',
                       'N_tot': 'N_AER',
                       'SWCF': 'SWCF',
                       'LWCF': 'LWCF',
                       'NCFT': 'NCFT',
                       'NCFT_Ghan': 'NCFT_Ghan',
                       'SWCF_Ghan': 'SWCF_Ghan',
                       'LWCF_Ghan': 'LWCF_Ghan',
                       'NDAF_Ghan': 'NDAF_Ghan',
                       'emiisop': 'SFisoprene',
                       'emiterp': 'SFmonoterp',
                       'LWP': 'TGCLDLWP',
                       'IWP': 'TGCLDIWP',
                       'CWP': 'TGCLDCWP',
                       'cdnc': 'AWNC',
                       'CDNC_2D': 'CDNC_2D',
                       're_liq': 'AREL',
                       'REFF_2D': 'REFF_2D',
                       'loadsoa': 'cb_SOA_dry',
                       'T2m': 'TREFHT',
                       'condTend_SOA_total': 'condTend_SOA_total',
                       'ps': 'PS'}

units_ECHAM = {'N_AER': 'cm^{-3}'}


def NorESM2model(var, model_name):
    if (model_name == 'ECHAM'):
        out_var = ''
        for key in ECHAM2NorESM_dic:
            if ECHAM2NorESM_dic[key] == var:
                out_var = key
        if (len(out_var) == 0):
            print('Warning: No ECHAM var in dictionary for %s' % var)
            out_var = var
    elif (model_name == 'EC-Earth'):
        out_var = ''
        for key in EC_Earth2NorESM_dic:
            if EC_Earth2NorESM_dic[key] == var:
                out_var = key
        if (len(out_var) == 0):
            print('Warning: No EC-Earth var in dictionary for %s' % var)
            out_var = var

    return out_var


def model2NorESM(var, model_name):
    if (model_name == 'ECHAM'):

        if var not in ECHAM2NorESM_dic:
            print('Warning: No ECHAM var in dictionary for %s' % var)
            NorESM_var = var
        else:
            NorESM_var = ECHAM2NorESM_dic[var]
    elif (model_name == 'EC-Earth'):
        if var not in EC_Earth2NorESM_dic:
            print('Warning: No EC-Earth var in dictionary for %s' % var)
            NorESM_var = var
        else:
            NorESM_var = EC_Earth2NorESM_dic[var]
    elif (model_name == 'NorESM'):
        NorESM_var = var
    else:
        print('Warning: did not find model name in translate code')

    return NorESM_var


varComp_EC_Earth = {'ifs': ['SWCF'], 'tm5': []}