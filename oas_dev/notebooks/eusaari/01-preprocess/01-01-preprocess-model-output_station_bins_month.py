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
from oas_dev.preprocess.launch_monthly_station_collocation import launch_monthly_station_output
from oas_dev.util.Nd.sizedist_class_v2.SizedistributionBins import SizedistributionStationBins
from oas_dev.util.collocate.collocateLONLAToutput import CollocateLONLATout
from oas_dev.constants import list_sized_vars_nonsec, list_sized_vars_noresm
import useful_scit.util.log as log
log.ger.setLevel(log.log.INFO)
import time

# %% [markdown]
# ## Settings:

# %%
nr_of_bins = 5
maxDiameter = 39.6  #    23.6 #e-9
minDiameter = 5.0  # e-9
history_field='.h1.'

# %%
from_t = '2009-01-01'
to_t = '2010-01-01'

# %% [markdown]
# ## Cases:

# %%
cases_sec = ['SECTv21_ctrl_koagD']#'SECTv21_ctrl_def','SECTv11_noresm2_ctrl', 'SECTv11_ctrl_fbvoc','SECTv11_noresm2_adj','SECTv11_noresm2_eq18']#'SECTv11_noresm2_NFHIST']#'SECTv11_ctrl_fbvoc']#'SECTv11_ctrl']#,'SECTv11_ctrl_fbvoc']#'SECTv11_ctrl']
cases_orig = ['noSECTv21_default_dd', 'noSECTv21_ox_ricc_dd']#, 'noSECTv21_ox_ricc']#'noSECTv11_noresm2_ricc', 'noSECTv11_noresm2_ctrl', 'noSECTv11_ctrl_fbvoc','noSECTv11_ctrl']#'noSECTv11_noresm2_NFHIST']#'noSECTv11_ctrl_fbvoc'] #/no SECTv11_ctrl

# %%
 
log.ger.info(f'TIMES:****: {from_t} {to_t}')

# %% [markdown]
# ## launches subprocesses that compute monthly

# %%
for case_name in cases_sec:
    launch_monthly_station_output(case_name, True, from_time=from_t, to_time=to_t)
for case_name in cases_orig:
    launch_monthly_station_output(case_name, False, from_time=from_t, to_time=to_t)

    

# %% [markdown]
# ## Merge monthly

# %%
for case_name in cases_sec:
    varlist = list_sized_vars_noresm
    c = CollocateLONLATout(case_name, from_t, to_t,
                           True,
                           'hour',
                           history_field=history_field)
    if c.check_if_load_raw_necessary(varlist ):
        time1 = time.time()
        a = c.make_station_data_merge_monthly(varlist)
        print(a)

        time2 = time.time()
        print('DONE : took {:.3f} s'.format( (time2-time1)))
    else:
        print('UUUPS')
for case_name in cases_orig:
    varlist = list_sized_vars_nonsec# list_sized_vars_noresm
    c = CollocateLONLATout(case_name, from_t, to_t,
                           False,
                           'hour',
                           history_field=history_field)
    if c.check_if_load_raw_necessary(varlist ):
        time1 = time.time()
        a = c.make_station_data_merge_monthly(varlist)
        print(a)

        time2 = time.time()
        print('DONE : took {:.3f} s'.format( (time2-time1)))
    else:
        print('UUUPS')

# %%

# %% [markdown]
#
# ## Compute binned dataset

# %% [markdown]
# ### Make station N50 etc.

# %%
for case_name in cases_sec:
    s = SizedistributionStationBins(case_name, from_t, to_t, [minDiameter, maxDiameter], True, 'hour',
                 nr_bins=nr_of_bins, history_field=history_field)
    s.compute_Nd_vars()

for case_name in cases_orig:
    s = SizedistributionStationBins(case_name, from_t, to_t, [minDiameter, maxDiameter], False, 'hour',
                                    nr_bins=nr_of_bins, history_field=history_field)
    s.compute_Nd_vars()
