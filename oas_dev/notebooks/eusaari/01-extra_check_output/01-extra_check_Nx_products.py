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
from useful_scit.imps import *

# %%
from sectional_v2.util.Nd.sizedist_class_v2.SizedistributionBins import SizedistributionStationBins
from sectional_v2.util.Nd.sizedist_class_v2.SizedistributionStation import SizedistributionStation
from sectional_v2.util.collocate.collocateLONLAToutput import CollocateLONLATout
from sectional_v2.constants import sized_varListNorESM, list_sized_vars_noresm, list_sized_vars_nonsec
#from useful_scit.util import log
import useful_scit.util.log as log
import time
log.ger.setLevel(log.log.INFO)

# %%

cases_orig =['noSECTv11_noresm2_ricc_oxdiur']#,'noSECTv11_noresm2_ricc_oxdiur','noSECTv11_noresm2_ricc']#, 'noSECTv11_noresm2_ctrl', 'noSECTv11_ctrl_fbvoc']#,'noSECTv11_ctrl']#'noSECTv11_noresm2_NFHIST']#'noSECTv11_ctrl_fbvoc'] #/no SECTv11_ctrl
cases_sec = ['SECTv21_ctrl_koagD']#'SECTv11_noresm2_ctrl', 'SECTv11_ctrl_fbvoc','SECTv11_noresm2_adj','SECTv11_noresm2_eq18']#'SECTv11_noresm2_NFHIST']#'SECTv11_ctrl_fbvoc']#'SECTv11_ctrl']#,'SECTv11_ctrl_fbvoc']#'SECTv11_ctrl']
cases_orig =['noSECTv21_ox_ricc_dd','noSECTv21_default_dd']#, 'noSECTv21_ox_ricc']#'noSECTv11_noresm2_ricc', 'noSECTv11_noresm2_ctrl', 'noSECTv11_ctrl_fbvoc','noSECTv11_ctrl']#'noSECTv11_noresm2_NFHIST']#'noSECTv11_ctrl_fbvoc'] #/no SECTv11_ctrl


# %% [markdown]
# # 2008-2009

# %%
from_t = '2008-01-01'
to_t = '2009-01-01'

# %%
varl=['N250',
      'N30-50',
      'N50',
      'N100']
n_rows = 2
n_cols=2

# %%
#arl=['COAGNUCL','GR','GRSOA','GRH2SO4','NUCLRATE','FORMRATE','SO2', 'SOA_NA','SO4_NA', 'cb_SOA_NA','cb_SO4_NA','H2SO4','SOA_LV','SOA_SV', 'NCONC01','NMR01','SIGMA01']
ds_dic={}
for case_name in cases_sec:
    varlist =varl
    c = CollocateLONLATout(case_name, from_t, to_t,
                           True,
                           'hour')
    a =c.get_station_ds(varlist)
    ds_dic[case_name]=a
    #if c.check_if_load_raw_necessary(varlist ):
    #    a = c.make_station_data_all()
for case_name in cases_orig:
    varlist =varl
    c = CollocateLONLATout(case_name, from_t, to_t,
                           False,
                           'hour')
    a =c.get_station_ds(varlist)
    ds_dic[case_name]=a

    


# %%
for case in cases_orig+cases_sec:
    fig, axs = plt.subplots(n_rows,n_cols, figsize=[20,20])
    for var, ax in zip(varl, axs.flatten()):
        #_,ax = plt.subplots()
        _da = ds_dic[case].isel(lev=-1)[var]
        for station in _da.station.values:
            _da.sel(station=station).plot(ax=ax)
        ax.set_title(case)
    plt.show()

# %% [markdown]
# # 2009-2010

# %%
from_t = '2009-01-01'
to_t = '2010-01-01'

# %%
varl=['N250',
      'N30-50',
      'N50',
      'N100']
n_rows = 2
n_cols=2

# %%
#arl=['COAGNUCL','GR','GRSOA','GRH2SO4','NUCLRATE','FORMRATE','SO2', 'SOA_NA','SO4_NA', 'cb_SOA_NA','cb_SO4_NA','H2SO4','SOA_LV','SOA_SV', 'NCONC01','NMR01','SIGMA01']
ds_dic={}
for case_name in cases_sec:
    varlist =varl
    c = CollocateLONLATout(case_name, from_t, to_t,
                           True,
                           'hour')
    a =c.get_station_ds(varlist)
    ds_dic[case_name]=a
    #if c.check_if_load_raw_necessary(varlist ):
    #    a = c.make_station_data_all()
for case_name in cases_orig:
    varlist =varl
    c = CollocateLONLATout(case_name, from_t, to_t,
                           False,
                           'hour')
    a =c.get_station_ds(varlist)
    ds_dic[case_name]=a

    


# %%
for case in cases_orig+cases_sec:
    fig, axs = plt.subplots(n_rows,n_cols, figsize=[20,20])
    for var, ax in zip(varl, axs.flatten()):
        #_,ax = plt.subplots()
        _da = ds_dic[case].isel(lev=-1)[var]
        for station in _da.station.values:
            _da.sel(station=station).plot(ax=ax)
        ax.set_title(case)
    plt.show()

# %%

# %%

# %%
