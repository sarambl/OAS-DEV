# %%
from sectional_v2.util.plot.plot_maps import plot_map_diff, fix_axis4map_plot, plot_map_abs_abs_diff, plot_map, subplots_map, plot_map_diff_2case
from useful_scit.imps import (np, xr, plt, pd)
from sectional_v2.util.imports import get_averaged_fields
from IPython.display import clear_output

# load and autoreload
from IPython import get_ipython
from useful_scit.imps import *

log.ger.setLevel(log.log.INFO)

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
pressure_adjust = True  # Can only be false if avg_over_lev false. Plots particular hybrid sigma lev
p_levels = [1013.,900., 800., 700., 600.]  # used if not avg
# %%
cases_sec = ['SECTv21_ctrl_koagD']# 'SECTv21_ctrl'
cases_orig = ['noSECTv21_default_dd', 'noSECTv21_ox_ricc_dd']
cases = cases_sec + cases_orig
# %%
areas = ['Global','notLand','notLand','Amazonas and surroundings', 'Polar N','Polar S','Pacific']
# %%
varl = [
        'HYGRO01',
        'SO2','DMS','isoprene','monoterp',
        'N_AER','NCONC01', 'NMR01','GR','NUCLRATE','FORMRATE',
        'H2SO4','SOA_LV','SOA_SV','SOA_NA','SO4_NA','SOA_A1',

        ]
varl_sec = ['nrSOA_SEC_tot', 'nrSO4_SEC_tot','nrSEC_tot']
# %%
for area in areas:
    prof_dic = get_averaged_fields.get_profiles(cases,varl,startyear, endyear,area=area,
                                                pressure_adjust=pressure_adjust)

for area in areas:
    prof_dic = get_averaged_fields.get_profiles(cases_sec,varl_sec,startyear, endyear,area=area,
                                                pressure_adjust=pressure_adjust)
