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
import os
import xarray as xr
# %% [markdown]
# ## Imports:

# %%
from oas_dev.util.Nd.sizedist_class_v2 import SizedistributionStation
from oas_dev.util.eusaar_data.eusaar_noresm import compute_all_subsets_percs_flag, get_all_distc_noresm

# %% [markdown]
# ## Cases:

# %%
# Case names:
cases_sec = ['SECTv21_ctrl_koagD']# 'SECTv21_ctrl']#,'SECTv11_redSOA_LVnuc','SECTv11_incBVOC']#'PD_SECT_CHC7_diur_ricc']#, 'PD_SECT_CHC7_diurnal']# , 'PD_SECT_CHC7_diur_ricc_incC']
cases_orig = []#'noSECTv11_ctrl']#'noSECTv11_ctrl']#,'PD_noSECT_nudgeERA_eq20']#'Original eq.20']  # , 'Original eq.18','Original eq.20, 1.5xBVOC','Original eq.20, rednuc']
#cases_sec = ['SECTv11_noresm2_ctrl']#'SECTv11_ctrl_fbvoc']#'SECTv11_ctrl']#,'SECTv11_ctrl_fbvoc']#'SECTv11_ctrl']
cases_orig =[]#'noSECTv21_default_dd','noSECTv21_ox_ricc_dd']#'noSECTv11_ctrl_fbvoc'] #/no SECTv11_ctrl
cases = cases_sec + cases_orig



# %% [markdown]
# ## Settings

# %%

from_t = '2008-01-01'
to_t = '2010-01-01'

nr_of_bins = 5
maxDiameter = 39.6  # 23.6 #e-9
minDiameter = 5.0  # e-9
time_resolution = 'hour'
history_field='.h1.'

# %% [markdown]
# ## Load datasets:

# %%
path_tmp = '/persistent01/tmp/'
def get_fn_tmp(case, from_t, to_t):
    return path_tmp + '%s_%s-%s_concat_dNdlogDs.nc'%(case,from_t,to_t)
cases_loaded=[]
# %%

if from_t=='2008-01-01' and to_t=='2010-01-01':
    ds_sec={}
    dic_mod_all = {}
    for case_name in cases_sec:
        fn = get_fn_tmp(case_name, from_t, to_t)
        if os.path.isfile(fn):
            continue
        else:
            cases_loaded.append(case_name)
        ls_ds = []
        for f_t, t_t in zip(['2008-01-01','2009-01-01'], ['2009-01-01','2010-01-01']):
            s = SizedistributionStation.SizedistributionStation(case_name, f_t, t_t,
                                                                [minDiameter, maxDiameter], True, time_resolution,
                                                                history_field=history_field)
            ls_ds.append(s.get_collocated_dataset(parallel=True))
        ds_conc = xr.concat(ls_ds, 'time')
        ds_conc:xr.Dataset
        # remove duplicates in time:
        ds_conc = ds_conc.sel(time=~ds_conc.indexes['time'].duplicated())
        ds_sec[case_name] = ds_conc #xr.concat(ls,'time')
        dic_mod_all[case_name] = ds_sec[case_name]# = s.return_Nd_ds()

    ds_orig={}
    for case_name in cases_orig:
        fn = get_fn_tmp(case_name, from_t, to_t)
        if os.path.isfile(fn):
            continue
        else:
            cases_loaded.append(case_name)
        ls_ds = []
        for f_t, t_t in zip(['2008-01-01','2009-01-01'], ['2009-01-01','2010-01-01']):
            s = SizedistributionStation.SizedistributionStation(case_name, f_t, t_t,
                                                                [minDiameter, maxDiameter], False, time_resolution,
                                                                history_field=history_field)
            ls_ds.append(s.get_collocated_dataset(parallel=True))
        ds_conc = xr.concat(ls_ds, 'time')
        # remove duplicates in time:
        ds_conc = ds_conc.sel(time=~ds_conc.indexes['time'].duplicated())
        ds_orig[case_name] = ds_conc #xr.concat(ls,'time')
        dic_mod_all[case_name] = ds_orig[case_name]# = s.return_Nd_ds()

else:
    dic_mod_all = {}
    dic_sec = {}
    dic_sized ={}
    for case in cases_sec:
        fn = get_fn_tmp(case, from_t, to_t)
        if os.path.isfile(fn):
            continue
        else:
            cases_loaded.append(case)
        s = SizedistributionStation.SizedistributionStation(case, from_t, to_t,
                                                        [minDiameter, maxDiameter], True, time_resolution,
                                                        history_field=history_field)
        ds_conc=s.get_collocated_dataset()
        dic_mod_all[case]=ds_conc
        dic_sec[case] = ds_conc#redo=True)
        dic_sized[case] = s

    dic_orig = {}
    for case in cases_orig:
        fn = get_fn_tmp(case, from_t, to_t)
        if os.path.isfile(fn):
            print(f'found file; {fn}')
            continue

        else:
            cases_loaded.append(case)
        s = SizedistributionStation.SizedistributionStation(case, from_t, to_t,
                                                        [minDiameter, maxDiameter], False, time_resolution,
                                                        history_field=history_field)
        ds_conc=s.get_collocated_dataset()
        dic_mod_all[case]=ds_conc
        dic_orig[case] = ds_conc#redo=True)
        dic_sized[case] = s



# %%
from useful_scit.util.make_folders import  make_folders
from dask.diagnostics import ProgressBar


make_folders(path_tmp)
for case in cases_loaded:
    fn = get_fn_tmp(case, from_t, to_t)
    print(fn)
    _ds = dic_mod_all[case].isel(lev=slice(-5,None))
    #fn = path_tmp + '%s_concat_dNdlogDs.nc'%case
    delayed_obj = _ds.to_netcdf(fn, compute=False)
    with ProgressBar():
        results = delayed_obj.compute()
    dic_mod_all[case].close() 

# %%
from useful_scit.util.make_folders import  make_folders
make_folders(path_tmp)
for case in cases:
    fn = get_fn_tmp(case, from_t, to_t)

    dic_mod_all[case] = xr.open_dataset(fn).isel(lev=slice(-5,None))
    print(dic_mod_all[case]['lev'])

# %% [markdown]
# ## Calculate DISTC dataset for model

# %%
dic_finish = {}
for case_name in dic_mod_all.keys():
    _ds = dic_mod_all[case_name]
    ds = get_all_distc_noresm(case_name, from_t, to_t, ds=_ds, recompute=True)
    dic_finish[case_name] = ds


# %%
