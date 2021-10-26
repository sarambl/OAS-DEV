# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %matplotlib inline

from dask.distributed import Client
import xarray as xr

# %% [markdown]
# ## Imports:

# %%
from oas_dev.util.Nd.sizedist_class_v2.SizedistributionBins import SizedistributionStationBins
from oas_dev.util.Nd.sizedist_class_v2.SizedistributionStation import SizedistributionStation
from oas_dev.util.collocate.collocateLONLAToutput import CollocateLONLATout
from oas_dev.constants import sized_varListNorESM, list_sized_vars_noresm, list_sized_vars_nonsec
#from useful_scit.util import log
import useful_scit.util.log as log
import time
log.ger.setLevel(log.log.INFO)

# %% [markdown]
# ## Settings models:
#
# Should run for one year at a time, i.e. set first 
# ```python 
# from_t = '2008-01-01'
# to_t = '2009-01-01'
# ```
# then,
# ```python 
# from_t = '2009-01-01'
# to_t = '2010-01-01'
# ```

# %%
nr_of_bins = 5
maxDiameter = 39.6  #    23.6 #e-9
minDiameter = 5.0  # e-9
history_field='.h1.'
variables = sized_varListNorESM['NCONC'] + sized_varListNorESM['NMR'] + sized_varListNorESM['SIGMA']
cases_sec = []#'SECTv11_ctrl']
cases_orig =[]#'noSECTv11_ctrl'] #/noSECTv11_ctrl
cases_sec = ['SECTv11_noresm2_ctrl']#'SECTv11_ctrl_fbvoc']#'SECTv11_ctrl']#,'SECTv11_ctrl_fbvoc']#'SECTv11_ctrl']
cases_orig =['noSECTv11_noresm2_ricc']#'noSECTv11_ctrl_fbvoc'] #/no SECTv11_ctrl

from_t = '2008-01-01'
to_t = '2009-01-01'
cases_sec = ['SECTv11_noresm2_ctrl', 'SECTv11_noresm2_SP']#'SECTv11_noresm2_adj', 'SECTv11_noresm2_eq18', 'SECTv11_noresm2_ctrl']#'SECTv11_noresm2_NFHIST']#'SECTv11_ctrl_fbvoc']#'SECTv11_ctrl']#,'SECTv11_ctrl_fbvoc']#'SECTv11_ctrl']
cases_orig =['noSECTv11_noresm2_ricc', 'noSECTv11_noresm2_ctrl','noSECTv11_ctrl']#'noSECTv11_noresm2_NFHIST']#'noSECTv11_ctrl_fbvoc'] #/no SECTv11_ctrl
from_t = '2007-06-01'
to_t = '2007-06-06'
cases_sec = ['SECTv11_noresm2_adj_sct_1dt']#'SECTv11_noresm2_2000','SECTv11_noresm2_eq20','SECTv11_noresm2_nr','SECTv11_noresm2_ctrl', 'SECTv11_noresm2_ctrl_s', 'SECTv11_ctrl_fbvoc','SECTv11_noresm2_adj','SECTv11_noresm2_adj_s','SECTv11_noresm2_eq18'][::-1]#'SECTv11_noresm2_NFHIST']#'SECTv11_ctrl_fbvoc']#'SECTv11_ctrl']#,'SECTv11_ctrl_fbvoc']#'SECTv11_ctrl']
cases_orig =['noSECTv11_noresm2_ricc_oxdiur']#'noSECTv11_noresm2_ricc_oxdiur_radup','noSECTv11_noresm2_ricc_oxdiur','noSECTv11_noresm2_ricc', 'noSECTv11_noresm2_ctrl', 'noSECTv11_ctrl_fbvoc','noSECTv11_ctrl']#'noSECTv11_noresm2_NFHIST']#'noSECTv11_ctrl_fbvoc'] #/no SECTv11_ctrl
from_t = '2007-04-01'
to_t = '2007-05-01'
t1 =time.time()
cases_sec = ['SECTv21_ctrl', 'SECTv21_ctrl_koagD']#,'SECTv21_ctrl_def',]#'SECTv11_noresm2_ctrl', 'SECTv11_ctrl_fbvoc','SECTv11_noresm2_adj','SECTv11_noresm2_eq18']#'SECTv11_noresm2_NFHIST']#'SECTv11_ctrl_fbvoc']#'SECTv11_ctrl']#,'SECTv11_ctrl_fbvoc']#'SECTv11_ctrl']
cases_orig =['noSECTv21_default_dd', 'noSECTv21_ox_ricc_dd']#'noSECTv11_noresm2_ricc', 'noSECTv11_noresm2_ctrl', 'noSECTv11_ctrl_fbvoc','noSECTv11_ctrl']#'noSECTv11_noresm2_NFHIST']#'noSECTv11_ctrl_fbvoc'] #/no SECTv11_ctrl
from_t = '2009-01-01'
to_t = '2010-01-01'

#cases_sec = ['SECTv11_noresm2_ctrl','SECTv11_noresm2_adj_s','SECTv11_noresm2_adj']#[::-1] # 'SECTv11_noresm2_NFHIST']#'SECTv11_ctrl_fbvoc']#'SECTv11_ctrl']#,'SECTv11_ctrl_fbvoc']#'SECTv11_ctrl']
#cases_orig =['noSECTv11_noresm2_ricc', 'noSECTv11_noresm2_ctrl']#, 'noSECTv11_ctrl_fbvoc']#['noSECTv11_noresm2_ricc','noSECTv11_ctrl'] # 'noSECTv11_noresm2_NFHIST']#'noSECTv11_ctrl_fbvoc'] #/no SECTv11_ctrl
#from_t = '2007-01-01'
#to_t = '2007-01-05'


# %% [markdown]
# ## Collocate NCONC\*, NMR\* and SIGMA\*

# %%
for case_name in cases_sec:
    varlist = list_sized_vars_noresm
    c = CollocateLONLATout(case_name, from_t, to_t,
                           True,
                           'hour',
                           history_field=history_field)
    if c.check_if_load_raw_necessary(varlist ):
        a = c.make_station_data_all()
for case_name in cases_orig:
    varlist = list_sized_vars_nonsec
    c = CollocateLONLATout(case_name, from_t, to_t,
                           False,
                           'hour',
                           history_field=history_field)
    if c.check_if_load_raw_necessary(varlist ):
        a = c.make_station_data_all()

# %% [markdown]
# ## Calculate sis

# %%

# Make station N50 etc.
for case_name in cases_sec:
    s = SizedistributionStation(case_name, from_t, to_t, [minDiameter, maxDiameter], True, 'hour',
                 nr_bins=nr_of_bins, history_field=history_field)
    #s.compute_Nd_vars()
    s.compute_sizedist_tot()
for case_name in cases_orig:
    t1 =time.time()

    s = SizedistributionStation(case_name, from_t, to_t, [minDiameter, maxDiameter], False, 'hour',
                                    nr_bins=nr_of_bins, history_field=history_field)
    #s.compute_Nd_vars()
    a = s.compute_sizedist_mod_tot()
    t2 =time.time()


# %%
