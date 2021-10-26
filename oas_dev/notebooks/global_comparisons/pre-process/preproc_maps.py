# %%
from oas_dev.util.imports.get_fld_fixed import get_field_fixed
from oas_dev.util.plot.plot_maps import plot_map_diff, fix_axis4map_plot, plot_map_abs_abs_diff, plot_map, subplots_map, plot_map_diff_2case
from useful_scit.imps import (np, xr, plt, pd)
from oas_dev.util.imports import get_averaged_fields
from IPython.display import clear_output
from useful_scit.imps import *

log.ger.setLevel(log.log.INFO)

# load and autoreload
from IPython import get_ipython

# noinspection PyBroadException
try:
    _ipython = get_ipython()
    _magic = _ipython.magic
    _magic('load_ext autoreload')
    _magic('autoreload 2')
except:
    pass
# %%
model = 'NorESM'

startyear = '2008-01'
endyear = '2014-12'
p_level=1013.
pmin = 850.  # minimum pressure level
avg_over_lev = True  # True#True#False#True
pressure_adjust = True  # Can only be false if avg_over_lev false. Plots particular hybrid sigma lev
if avg_over_lev:
    pressure_adjust = True
p_levels = [1013.,900., 800., 700., 600.]  # used if not avg
# %%
cases_sec = ['SECTv21_ctrl_koagD']#'SECTv21_ctrl',,'SECTv21_ctrl_def']
cases_orig = ['noSECTv21_default_dd', 'noSECTv21_ox_ricc_dd']
cases = cases_sec + cases_orig
# %%
varl = ['ACTNL_incld', 'ACTREL_incld',
        'TGCLDCWP',
        'TGCLDIWP',
        'TGCLDLWP',
        'NCFT_Ghan',
        'HYGRO01',
        'SOA_NAcondTend',
        'SO4_NAcondTend',
        'cb_SOA_NA',
        'cb_SO4_NA',
        'HYGRO01',
        'cb_SOA_LV',
        'cb_H2SO4',
        'SO2',
        'DMS',
        'isoprene',
        'monoterp',
        'N_AER',
        'NCONC01',
        'NMR01',
        'GR',
        'COAGNUCL',
        'NUCLRATE',
        'FORMRATE',
        'H2SO4',
        'SOA_LV',
        'SOA_SV',
        'SOA_NA',
        'SO4_NA',
        'SOA_A1',
        'NCFT_Ghan',
        'SFisoprene',
        'SFmonoterp',
        'SOA_NA_totLossR',
        'SOA_NA_lifetime',
        'SO4_NA_totLossR',
        'SO4_NA_lifetime',
        'cb_SOA_NA_OCW',
        'cb_SO4_NA_OCW',
        'SO4_NA_OCWDDF',
        'SO4_NA_OCWSFWET',
        'SOA_NA_OCWDDF',
        'SOA_NA_OCWSFWET',
        'cb_SOA_A1',
        'cb_SO4_A1',
        'cb_SOA_NA',
        'cb_SO4_NA',
        'cb_NA',

        'SWCF_Ghan',
        'LWCF_Ghan',
        'AWNC_incld',
        'AREL_incld',
        'CLDHGH',
        'CLDLOW',
        'CLDMED',
        'CLDTOT',
        'CDNUMC',
        'DIR_Ghan',
        'CDOD550',
        'SWDIR_Ghan',

        ]
varl_sec = [
    'nrSOA_SEC_tot',
    'nrSO4_SEC_tot',
    'nrSEC_tot',
    'cb_SOA_SEC01',
    'cb_SOA_SEC02',
    'cb_SOA_SEC03',
    'leaveSecSOA',
    'leaveSecH2SO4',
]
# %%
for case in cases:
    get_field_fixed(case,varl, startyear, endyear, #raw_data_path=constants.get_input_datapath(),
                pressure_adjust=True, model = 'NorESM', history_fld='.h0.', comp='atm', chunks=None)

maps_dic = get_averaged_fields.get_maps_cases(cases,varl,startyear, endyear,
                                              avg_over_lev=avg_over_lev,
                                              pmin=pmin,
                                              pressure_adjust=pressure_adjust, p_level=p_level)
maps_dic = get_averaged_fields.get_maps_cases(cases_sec,varl_sec,startyear, endyear,
                                              avg_over_lev=avg_over_lev,
                                              pmin=pmin,
                                              pressure_adjust=pressure_adjust, p_level=p_level)


for period in ['JJA','DJF']:
    maps_dic = get_averaged_fields.get_maps_cases(cases,varl,startyear, endyear,
                                              avg_over_lev=avg_over_lev,
                                              pmin=pmin,
                                              pressure_adjust=pressure_adjust,
                                              p_level=p_level,
                                              time_mask=period)
    maps_dic = get_averaged_fields.get_maps_cases(cases_sec,varl_sec,startyear, endyear,
                                              avg_over_lev=avg_over_lev,
                                              pmin=pmin,
                                              pressure_adjust=pressure_adjust, p_level=p_level,
                                              time_mask=period)
