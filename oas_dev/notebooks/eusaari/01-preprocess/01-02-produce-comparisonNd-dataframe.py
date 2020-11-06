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
from useful_scit.imps import (xr, plt, np, pd)
from useful_scit.util.make_folders import make_folders
# %load_ext autoreload
# %autoreload 2
from sectional_v2.util.Nd.sizedist_class_v2.SizedistributionBins import SizedistributionStationBins
from sectional_v2.util.collocate.collocateLONLAToutput import CollocateLONLATout
from sectional_v2.constants import sized_varListNorESM
#from useful_scit.util import log
import useful_scit.util.log as log
log.ger.setLevel(log.log.INFO)



# %% [markdown]
# ## Savepath:

# %%
from sectional_v2.constants import get_outdata_path
path_out = get_outdata_path('eusaar')
version ='_noresmv21_dd'#_noresm2'#_fbvoc'
file_out = path_out + 'Nd_cat_sources_timeseries%s.csv'%version

# %%
file_out

# %% [markdown]
# ### Model data:

# %%
nr_of_bins = 5
maxDiameter = 39.6  # 23.6 #e-9
minDiameter = 5.0  # e-9
history_field='.h1.'
#cases_sec = ['SECTv11_ctrl', 'SECTv11_ctrl_fbvoc']
cases_sec = ['SECTv11_ctrl','SECTv11_noresm2_ctrl', 'SECTv11_noresm2_adj']#_fbvoc']
#cases_orig =['noSECTv11_ctrl', 'noSECTv11_ctrl_fbvoc'] #/noSECTv11_ctrl
cases_orig =[ 'noSECTv11_ctrl','noSECTv11_noresm2_ricc']#_fbvoc'] #/noSECTv11_ctrl
#cases_sec = ['SECTv11_noresm2_NFHIST']#'SECTv11_ctrl_fbvoc']#'SECTv11_ctrl']#,'SECTv11_ctrl_fbvoc']#'SECTv11_ctrl']
#cases_orig =['noSECTv11_noresm2_NFHIST']#'noSECTv11_ctrl_fbvoc'] #/no SECTv11_ctrl
cases_sec = ['SECTv21_ctrl','SECTv21_ctrl_koagD']
cases_orig =['noSECTv21_default_dd', 'noSECTv21_ox_ricc_dd']
from_t = '2008-01-01'
to_t = '2010-01-01'



# %%
# Make station N50 etc.
if from_t=='2008-01-01' and to_t=='2010-01-01':
    ds_sec={}
    dic_mod_all = {}
    for case_name in cases_sec:
        ls = []
        for f_t, t_t in zip(['2008-01-01','2009-01-01'], ['2009-01-01','2010-01-01']):
            s = SizedistributionStationBins(case_name, f_t, t_t, [minDiameter, maxDiameter], True, 'hour',
                 nr_bins=nr_of_bins, history_field=history_field)
            ls.append(s.return_Nd_ds())
        ds_conc = xr.concat(ls, 'time')
        # remove duplicates in time:
        ds_conc = ds_conc.sel(time=~ds_conc.indexes['time'].duplicated())
        ds_sec[case_name] = ds_conc 
        dic_mod_all[case_name] = ds_sec[case_name]# = s.return_Nd_ds()

    ds_orig={}
    for case_name in cases_orig:
        ls = []
        for f_t, t_t in zip(['2008-01-01','2009-01-01'], ['2009-01-01','2010-01-01']):

            s = SizedistributionStationBins(case_name, f_t, t_t, [minDiameter, maxDiameter], False, 'hour',
                                    nr_bins=nr_of_bins, history_field=history_field)
            ls.append(s.return_Nd_ds())
        ds_conc = xr.concat(ls, 'time')
        # remove duplicates in time:
        ds_conc = ds_conc.sel(time=~ds_conc.indexes['time'].duplicated())
        ds_orig[case_name] = ds_conc 

        #ds_orig[case_name] = xr.concat(ls,'time')
        dic_mod_all[case_name] = ds_orig[case_name]# = s.return_Nd_ds()
    

    
    
else:
    ds_sec={}
    dic_mod_all = {}
    for case_name in cases_sec:
        s = SizedistributionStationBins(case_name, from_t, to_t, [minDiameter, maxDiameter], True, 'hour',
                 nr_bins=nr_of_bins, history_field=history_field)
        ds_sec[case_name] = s.return_Nd_ds()
        dic_mod_all[case_name] = ds_sec[case_name]# = s.return_Nd_ds()

    ds_orig={}
    for case_name in cases_orig:
        s = SizedistributionStationBins(case_name, from_t, to_t, [minDiameter, maxDiameter], False, 'hour',
                                    nr_bins=nr_of_bins, history_field=history_field)
        ds_orig[case_name] = s.return_Nd_ds()
        dic_mod_all[case_name] = ds_orig[case_name]# = s.return_Nd_ds()
    



# %% [markdown]
# # Drop excess coordinates in dataset:

# %%
def drop_excess_coords(ds):
    coords = list(ds.coords)
    lon_lat_coords = [coo for coo in coords if ('LON' in coo or 'LAT' in coo) or ('lon_' in coo or 'lat_' in coo)]
    return ds.drop(lon_lat_coords)


# %%
for case_name in dic_mod_all:
    dic_mod_all[case_name] = drop_excess_coords(dic_mod_all[case_name]).isel(lev=-1).drop(['lev']).sel(time=slice(from_t, to_t))

# %% [markdown]
# ## Import eusaar

# %%
from sectional_v2.constants import path_eusaar_data# path_eusaar_data
import numpy as np
from sectional_v2.util import eusaar_data
from sectional_v2.util.eusaar_data.histc_vars import load_var_as_dtframe
from sectional_v2.util.eusaar_data import  distc_var, histc_vars, histsc_hists # import load_var_as_dtframe
import matplotlib.pyplot as plt
from useful_scit.plot import get_cmap_dic

# %% [markdown]
# ### Rename N30 to N30-50

# %%
st_ds = histc_vars.get_histc_vars_xr()
st_ds = st_ds.rename_vars({'N30':'N30-50'})

# %% [markdown]
# ### Make vars with source name:

# %%
for case_name in dic_mod_all:
    dic_mod_all[case_name]['N50-100']=dic_mod_all[case_name]['N50']-dic_mod_all[case_name]['N100']
    dic_mod_all[case_name]['N30-100']=dic_mod_all[case_name]['N30-50']+dic_mod_all[case_name]['N50-100']
st_ds['N50-100']=st_ds['N50']-st_ds['N100']
st_ds['N30-100']=st_ds['N30-50']+st_ds['N50-100']


# %% [markdown]
# ## Add flags where observation data good:

# %%
st_ds['time']

# %%
from sectional_v2.util.eusaar_data.flags import load_flags_allstations
flags = load_flags_allstations()

# %%
for case_name in dic_mod_all:
    dic_mod_all[case_name] = dic_mod_all[case_name].sel(time=slice(flags['time'].min(),flags['time'].max())) 
    dic_mod_all[case_name]['flag_gd'] = flags['gd']
    
st_ds['flag_gd'] = flags['gd']

# %% [markdown]
# ## Combine to one dataset:

# %%
dic_all_df = {}
first=True
# Merge model cases
for case_name in dic_mod_all:
    dic_all_df[case_name] = dic_mod_all[case_name].to_dataframe()
    dic_all_df[case_name]['source']=case_name    
    if first:
        df_mod_time = dic_all_df[case_name].reset_index()
        first=False
    else:
        df_mod_time = pd.merge(df_mod_time, dic_all_df[case_name].reset_index(), how='outer')
# set source in observations
df_obs_time =  st_ds.to_dataframe()
df_obs_time['source']='eusaar'

# %%
df_merged = pd.merge(df_obs_time.reset_index(), df_mod_time, how='outer')
df_merged

# %% [markdown]
# ## Construnc N30-100 and N50-100
# Mark that N50=N50-500 and N100=N100-500, so 
# $$N_{50-100}=N_{50}-N_{100}$$

# %%
df_merged['N30-100'] = df_merged['N30-50']+ df_merged['N50']-df_merged['N100']
df_merged['N50-100'] =df_merged['N50']-df_merged['N100']


# %%
from useful_scit.util.make_folders import make_folders
make_folders(file_out)
df_merged.to_csv(file_out)

# %%
df_merged

# %%
df_merged['source'].unique()

# %%
df_merged#[df_merged['source']=='SECTv11_ctrl_fbvoc']#.unique()

# %%

# %%

# %%
