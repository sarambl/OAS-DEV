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
from sectional_v2.util.Nd.sizedist_class_v2.SizedistributionBins import SizedistributionStationBins
from sectional_v2.util.Nd.sizedist_class_v2.SizedistributionStation import SizedistributionStation
from sectional_v2.util.collocate.collocateLONLAToutput import CollocateLONLATout
from sectional_v2.constants import sized_varListNorESM, list_sized_vars_noresm, list_sized_vars_nonsec
#from useful_scit.util import log
import useful_scit.util.log as log
import time
log.ger.setLevel(log.log.INFO)

# %%
nr_of_bins = 5
maxDiameter = 39.6  #    23.6 #e-9
minDiameter = 5.0  # e-9
history_field='.h1.'
variables = sized_varListNorESM['NCONC'] + sized_varListNorESM['NMR'] + sized_varListNorESM['SIGMA']
cases_sec = []#'SECTv11_ctrl']
cases_orig =[]#'noSECTv11_ctrl'] #/noSECTv11_ctrl
cases_sec = ['SECTv21_ctrl', 'SECTv21_ctrl_koagD']#'SECTv11_ctrl_fbvoc']#'SECTv11_ctrl']#,'SECTv11_ctrl_fbvoc']#'SECTv11_ctrl']
cases_orig =['noSECTv21_default_dd','noSECTv21_ox_ricc_dd']#'noSECTv11_ctrl_fbvoc'] #/no SECTv11_ctrl

from_t = '2008-01-01'
to_t = '2009-01-01'


# %% [markdown]
# ## Collocate NCONC\*, NMR\* and SIGMA\*

# %%
varl_s=['dNdlogD_mod','dNdlogD_mode01', 'dNdlogD_sec', 'NMR01','NCONC01']#,'SO4_NAcondTend', 'SOA_NAcondTend']
varl_o=['dNdlogD_mod','dNdlogD_mode01', 'NMR01','NCONC01']#,'SO4_NAcondTend', 'SOA_NAcondTend']
dic_sd={}
for case_name in cases_sec:
    varlist =varl_s
    c = CollocateLONLATout(case_name, from_t, to_t,
                           True,
                           'hour',
                           history_field=history_field)
    a =c.get_station_ds(varlist)
    dic_sd[case_name]=a.isel(lev=-1)
    print(case_name)
    #if c.check_if_load_raw_necessary(varlist ):
    #    a = c.make_station_data_all()
for case_name in cases_orig:
    print(case_name)
    varlist =varl_o
    c = CollocateLONLATout(case_name, from_t, to_t,
                           False,
                           'hour',
                           history_field=history_field)
    a =c.get_station_ds(varlist)
    dic_sd[case_name]=a.isel(lev=-1)

    


# %% [markdown]
# ## EBAS data:

# %%
from sectional_v2.util.EBAS_data.sizedistrib import create_EBAS_sizedist, raw_data_EBAS

# %%
fn = raw_data_EBAS / 'FI0050R.20080101000000.20181205100800.dmps.particle_number_size_distribution.aerosol.1y.1h.FI03L_DMPS_HYY_01.FI03L_TRY_TDMPS.lev2.nc'

#fn = raw_data_EBAS / 'FI0050R.20100101000000.20181205100800.dmps.particle_number_size_distribution.aerosol.1h.1h.FI03L_DMPS_HYY_01.FI03L_TRY_TDMPS.lev2.nc'
# %%
eb = create_EBAS_sizedist(fn)
# %%
eb.plot_time_from_to(from_time='2008-05-01', to_time='2008-05-10')


# %% [markdown]
# $ \frac{dN}{dlogD} = \frac{dN}{dlog_{10}D}\cdot \frac{dlog_{10}D}{dlogD}$
#
# $10^{x} = e^{log(10)x}$

# %% [markdown]
# $10^{log_{10}D} = D$
#
# $e^{log(10)\cdot log_{10}D} = D$

# %% [markdown]
# $ln(D) = log(10)\cdot log_{10}D$

# %% [markdown]
# $ \frac{dN}{dlogD} = \frac{dN}{dlog_{10}D}\cdot \frac{dlog_{10}D}{dlogD}= \frac{dN}{dlog_{10}D} \cdot \frac{1}{log(10)}$
#

# %% [markdown]
# $ \frac{dN}{dlog_{10}D} = \frac{dN}{dlogD} \cdot log(10)$
#

# %%
import numpy as np

# %%
np.log(10)

# %%
_from_time='2008-05-01' 
_to_time='2008-05-10'


# %%
for case in cases_sec:
    print(case)
    _ds = dic_sd[case].sel(time=slice(_from_time,_to_time)) # ['dNdlogD']
    _ds['dNdlogD'] = _ds['dNdlogD_mod'] + _ds['dNdlogD_sec']
    _ds['dNdlog10D'] =_ds['dNdlogD']*np.log(10)
    dic_sd[case] = _ds
for case in cases_orig:
    print(case)
    
    #_ds = dic_sd[case] # ['dNdlogD']
    _ds = dic_sd[case].sel(time=slice(_from_time,_to_time)) # ['dNdlogD']
    _ds['dNdlogD'] = _ds['dNdlogD_mod']# + _ds['dNdlogD_sec']
    _ds['dNdlog10D'] =_ds['dNdlogD']*np.log(10)
    dic_sd[case] = _ds
    

# %% [markdown]
# ## Hyytiala

# %%
station='SMR'
figsize = [10,10]

# %%
fn = raw_data_EBAS / 'FI0050R.20080101000000.20181205100800.dmps.particle_number_size_distribution.aerosol.1y.1h.FI03L_DMPS_HYY_01.FI03L_TRY_TDMPS.lev2.nc'

#fn = raw_data_EBAS / 'FI0050R.20100101000000.20181205100800.dmps.particle_number_size_distribution.aerosol.1h.1h.FI03L_DMPS_HYY_01.FI03L_TRY_TDMPS.lev2.nc'
# %%
eb = create_EBAS_sizedist(fn)
# %%
import matplotlib.pyplot as plt

# %%
from sectional_v2.data_info import get_nice_name_case


# %%
def plt_all_cases_time_sized(eb,_from_time, _to_time,station,  axs=None, figsize=[10,10]):
    if axs is None:
        fig, axs = plt.subplots(5, figsize=figsize, sharex=True, sharey=True)
    eb.plot_time_from_to(from_time=_from_time, to_time=_to_time,cmap='Reds', ax=axs[0])
    axs = axs.flatten()
    axs[0].set_title(f'{station}: EBAS data')
    for case,ax in zip(cases_sec+cases_orig, axs[1:]):
        _ds = dic_sd[case]
        eb.plot_time_from_to(from_time=_from_time, to_time=_to_time, dataset=_ds.sel(station=station), cmap='Reds', ax=ax)
        ax.set_title(f'{station}: {get_nice_name_case(case)}')
      
plt_all_cases_time_sized(eb, _from_time, _to_time, 'SMR')
plt.tight_layout()

# %% [markdown]
# ## Melpiz

# %%
from sectional_v2.util.EBAS_data.sizedistrib import create_EBAS_sizedist, raw_data_EBAS

# %%
fn = raw_data_EBAS / 'DE0044R.20080101000000.20181205100800.dmps.particle_number_size_distribution.aerosol.1y.1h.DE08L_DMPS_IFT_MELPITZ02_until_20080818_.DE08L_IFT_DRY_TDMPS_until_20080818_DE08L_IFT_C.lev2.nc'

#fn = raw_data_EBAS / 'FI0050R.20100101000000.20181205100800.dmps.particle_number_size_distribution.aerosol.1h.1h.FI03L_DMPS_HYY_01.FI03L_TRY_TDMPS.lev2.nc'
# %%

# %%
eb = create_EBAS_sizedist(fn, place='MPZ', perc=False)
# %%
plt_all_cases_time_sized(eb, _from_time, _to_time, 'MPZ')
plt.tight_layout()

# %%
eb.plot_time_from_to(from_time=_from_time, to_time=_to_time,cmap='Reds')


# %%
eb.plot_time_from_to(from_time=_from_time, to_time=_to_time, dataset=_ds.sel(station='SMR'), cmap='Reds')


# %%
_ds = dic_sd['noSECTv21_default_dd']#['dNdlogD']
_ds['dNdlogD'] = _ds['dNdlogD_mod'] #+ _ds['dNdlogD_sec']
_ds['dNdlog10D'] =_ds['dNdlogD']*np.log(10)

# %%
eb.plot_time_from_to(from_time=_from_time, to_time=_to_time, dataset=_ds.sel(station='SMR'), cmap='Reds')

# %%
_ds = dic_sd['noSECTv21_ox_ricc_dd']#['dNdlogD']
_ds['dNdlogD'] = _ds['dNdlogD_mod'] #+ _ds['dNdlogD_sec']
_ds['dNdlog10D'] =_ds['dNdlogD']*np.log(10)

# %%
eb.plot_time_from_to(from_time=_from_time, to_time=_to_time, dataset=_ds.sel(station='SMR'), cmap='Reds')

# %%
from sectional_v2.util.EBAS_data.sizedistrib import create_EBAS_sizedist, raw_data_EBAS

# %%
fn = raw_data_EBAS / 'DE0044R.20080101000000.20181205100800.dmps.particle_number_size_distribution.aerosol.1y.1h.DE08L_DMPS_IFT_MELPITZ02_until_20080818_.DE08L_IFT_DRY_TDMPS_until_20080818_DE08L_IFT_C.lev2.nc'

#fn = raw_data_EBAS / 'FI0050R.20100101000000.20181205100800.dmps.particle_number_size_distribution.aerosol.1h.1h.FI03L_DMPS_HYY_01.FI03L_TRY_TDMPS.lev2.nc'
# %%
import xarray as xr

# %%
eb = create_EBAS_sizedist(fn, place='MPZ', perc=False)
# %%
_from_time='2008-05-01' 
_to_time='2008-05-10'
eb.plot_time_from_to(from_time=_from_time, to_time=_to_time, vmax=4e4, cmap='Reds')


# %%
from matplotlib.colors import LogNorm
norm = LogNorm(vmin=100, vmax=4e4)

# %%
from_time= '2008-06-01'
to_time = '2008-09-01'

# %%
eb.dataset['dNdlog10D'].sel(time=slice(_from_time, _to_time)).groupby('time.hour').mean().plot(yscale='log', ylim=[3,1e3], norm=norm)#, cmap='Reds')

# %%
_ds = dic_sd['SECTv21_ctrl']#['dNdlogD']
_ds['dNdlog10D'].sel(time=slice(_from_time, _to_time)).sel(station='MPZ').groupby('time.hour').mean().plot(y='diameter',yscale='log', ylim=[3,1e3], norm=norm)

# %%
_ds = dic_sd['SECTv21_ctrl_koagD']#['dNdlogD']
_ds['dNdlog10D'].sel(time=slice(_from_time, _to_time)).sel(station='MPZ').groupby('time.hour').mean().plot(y='diameter',yscale='log', ylim=[3,1e3], norm=norm)

# %%
_ds = dic_sd['noSECTv21_default_dd']#['dNdlogD']
_ds['dNdlog10D'].sel(time=slice(from_time, to_time)).sel(station='MPZ').groupby('time.hour').mean().plot(y='diameter',yscale='log', ylim=[3,1e3], norm=norm)

# %%
_ds = dic_sd['noSECTv21_ox_ricc_dd']#['dNdlogD']
_ds['dNdlog10D'].sel(time=slice(from_time, to_time)).sel(station='MPZ').groupby('time.hour').mean().plot(y='diameter',yscale='log', ylim=[3,1e3], norm=norm)

# %%
for stat in _ds['station'].values:
    for case in cases_sec +cases_orig:
        _ds = dic_sd[case]#['dNdlogD']
        _ds['dNdlog10D'].sel(station=stat).mean(['time']).plot(x='diameter',xscale='log', xlim=[3,1e3],label=case, yscale='log',ylim=[1,4e4])#, norm=norm)
    plt.legend()
    plt.show()

# %%
_ds = dic_sd['SECTv21_ctrl']#['dNdlogD']
_ds['dNdlog10D'].groupby('time.hour').mean().sel(station='MPZ').plot(y='diameter',yscale='log', vmax=8000, ylim=[3,1e3])

# %%
_ds = dic_sd['SECTv21_ctrl']#['dNdlogD']
eb.plot_time_from_to(from_time=_from_time, to_time=_to_time, dataset=_ds.sel(station='MPZ'), vmax=4e4, cmap='Reds')


# %%
_ds = dic_sd['noSECTv21_default_dd']#['dNdlogD']
eb.plot_time_from_to(from_time=_from_time, to_time=_to_time, dataset=_ds.sel(station='MPZ'), vmax=4e4, cmap='Reds')

# %%
_ds = dic_sd['noSECTv21_ox_ricc_dd']#['dNdlogD']
eb.plot_time_from_to(from_time=_from_time, to_time=_to_time, dataset=_ds.sel(station='MPZ'), vmax=4e4)

# %%
import numpy as np
from useful_scit.imps import (np, plt, pd)

# %%
stations = dic_sd[cases_sec[0]]['station'].values

# %%
import matplotlib.colors as mcolors
st = '2008-04-09'
et = '2008-04-16'
for i in np.arange(len(stations)):
    fig, axs = plt.subplots(3, figsize=[10,10])
    for case_name in cases_sec:
        ax = axs[0]
        plt_ds = dic_sd[case_name]
                  #['dNdlogD_mod']+dic_sd[case_name]['dNdlogD_sec'])
        plt_ds = plt_ds.sel(time=slice(st,et)).isel(station=i, lev=-1)
        plt_da = plt_ds['dNdlogD_mod'] #+ plt_ds['dNdlogD_sec']
        plt_da.plot(x='time', yscale='log', ylim=[3,500], label=case_name, norm=mcolors.LogNorm(vmin=10,vmax=8e3), cmap='Reds', ax=ax)
        ax.set_title(case_name)
    for case_name, ax in zip(cases_orig, axs[1:]):
        plt_ds = dic_sd[case_name]
                  #['dNdlogD_mod']+dic_sd[case_name]['dNdlogD_sec'])
        plt_ds = plt_ds.sel(time=slice(st,et)).isel(station=i, lev=-1)
        plt_da = plt_ds['dNdlogD_mod']# + plt_ds['dNdlogD_sec']
        plt_da.plot(x='time', yscale='log', ylim=[3,500], label=case_name, norm=mcolors.LogNorm(vmin=10,vmax=8e3), cmap='Reds', ax=ax)
        ax.set_title(case_name)
    #    plt_da = (dic_sd[case_name]['dNdlogD_mod'])
    #    plt_da = plt_da.sel(time=slice(st,et)).isel(station=i, lev=-1)
    #    
    #    plt_da.plot(x='time', yscale='log', ylim=[3,500], label=case_name, norm=mcolors.LogNorm(vmin=10,vmax=8e3), cmap='Reds', ax=ax)
    #    ax.set_title(case_name)

        
    plt.show()


# %%
import matplotlib.colors as mcolors
st = '2008-04-09'
et = '2008-04-16'
for i in np.arange(len(stations)):
    fig, axs = plt.subplots(3, figsize=[10,10], sharex=True)
    for case_name in cases_sec:
        ax = axs[0]
        plt_ds = dic_sd[case_name]
                  #['dNdlogD_mod']+dic_sd[case_name]['dNdlogD_sec'])
        plt_ds = plt_ds.sel(time=slice(st,et)).isel(station=i, lev=-1)
        plt_da = plt_ds['dNdlogD_mod'] #+ plt_ds['dNdlogD_sec']
        plt_da.plot(x='time', yscale='log', ylim=[3,500], label=case_name, norm=mcolors.LogNorm(vmin=10,vmax=8e3), cmap='Reds', ax=ax)
        ax.set_title(f'{case_name}, {stations[i]}')
        
    for case_name, ax in zip(cases_orig, axs[1:]):
        plt_ds = dic_sd[case_name]
                  #['dNdlogD_mod']+dic_sd[case_name]['dNdlogD_sec'])
        plt_ds = plt_ds.sel(time=slice(st,et)).isel(station=i, lev=-1)
        plt_da = plt_ds['dNdlogD_mod']# + plt_ds['dNdlogD_sec']
        plt_da.plot(x='time', yscale='log', ylim=[3,500], label=case_name, norm=mcolors.LogNorm(vmin=10,vmax=8e3), cmap='Reds', ax=ax)
        ax.set_title(f'{case_name}, {stations[i]}')
    #    plt_da = (dic_sd[case_name]['dNdlogD_mod'])
    #    plt_da = plt_da.sel(time=slice(st,et)).isel(station=i, lev=-1)
    #    
    #    plt_da.plot(x='time', yscale='log', ylim=[3,500], label=case_name, norm=mcolors.LogNorm(vmin=10,vmax=8e3), cmap='Reds', ax=ax)
    #    ax.set_title(case_name)

        
    plt.show()


# %%
drop_coords = [c for c in plt_da.coords if c not in ['time', 'diameter']]


# %%
plt_da.drop(drop_coords).groupby('time.hour').mean().plot(x='hour',yscale='log', ylim=[3,500], label=case_name, norm=mcolors.LogNorm(vmin=10,vmax=8e3), cmap='Reds')

# %%
from sectional_v2.util.plot.colors import get_case_col#(case)

# %%
import matplotlib.colors as mcolors
st = '2008-03-01'
et = '2008-06-01'
cmap='viridis'
var = 'NCONC01'
fig, axs = plt.subplots(1, figsize=[8,5], sharex=True)

for i in np.arange(len(stations)):
    for case_name in cases_sec + cases_orig:
        col = get_case_col(case_name)
        ax = axs#[0]
        plt_ds = dic_sd[case_name]
                  #['dNdlogD_mod']+dic_sd[case_name]['dNdlogD_sec'])
        plt_ds = plt_ds.sel(time=slice(st,et)).isel(station=i, lev=-1)
        plt_da = plt_ds[var]# + plt_ds['dNdlogD_sec']
        plt_da = plt_da.groupby('time.hour').median()#.median('station')
        plt_da.plot( yscale='log', c =col, label='_nolegend_', alpha=0.2)#, linewidth=0.2)#x='hour')#, yscale='log', ylim=[3,500], , norm=mcolors.LogNorm(vmin=10,vmax=8e3), cmap=cmap, ax=ax)
        #ax.set_title(f'{case_name}, {stations[i]}')
for case_name in cases_sec + cases_orig:
    col = get_case_col(case_name)
    ax = axs#[0]
    plt_ds = dic_sd[case_name]
                  #['dNdlogD_mod']+dic_sd[case_name]['dNdlogD_sec'])
    plt_ds = plt_ds.sel(time=slice(st,et)).isel( lev=-1)
    plt_da = plt_ds[var]# + plt_ds['dNdlogD_sec']
    plt_da = plt_da.groupby('time.hour').median().median('station')
    plt_da.plot( yscale='log', c =col, label=case_name)#, alpha=0.2)#x='hour')#, yscale='log', ylim=[3,500], , norm=mcolors.LogNorm(vmin=10,vmax=8e3), cmap=cmap, ax=ax)
        #ax.set_title(f'{case_name}, {stations[i]}')

plt.ylim([100,4000])
plt.legend()
        


# %%
    for case_name in cases_orig:#, axs[1:]):
        plt_ds = dic_sd[case_name]
                  #['dNdlogD_mod']+dic_sd[case_name]['dNdlogD_sec'])
        plt_ds = plt_ds.sel(time=slice(st,et)).isel( lev=-1)
        plt_da = plt_ds[var]# + plt_ds['dNdlogD_sec']
        plt_da = plt_da.groupby('time.hour').median().median('station')
        
        plt_da.plot(label=case_name, yscale='log')#x='hour', yscale='log', ylim=[3,500], label=case_name, norm=mcolors.LogNorm(vmin=10,vmax=8e3), cmap=cmap, ax=ax)
        ax.set_title(f'{case_name}, {stations[i]}')
    #    plt_da = (dic_sd[case_name]['dNdlogD_mod'])
    #    plt_da = plt_da.sel(time=slice(st,et)).isel(station=i, lev=-1)
    #    
    #    plt_da.plot(x='time', yscale='log', ylim=[3,500], label=case_name, norm=mcolors.LogNorm(vmin=10,vmax=8e3), cmap='Reds', ax=ax)
    #    ax.set_title(case_name)

    plt.legend()
    plt.show()


# %%
import matplotlib.colors as mcolors
st = '2008-03-01'
et = '2008-06-01'
cmap='viridis'
var = 'NCONC01'
for i in np.arange(len(stations)):
    fig, axs = plt.subplots(1, figsize=[10,4], sharex=True)
    axs=[axs]
    for case_name in cases_sec:
        ax = axs[0]
        plt_ds = dic_sd[case_name]
                  #['dNdlogD_mod']+dic_sd[case_name]['dNdlogD_sec'])
        plt_ds = plt_ds.sel(time=slice(st,et)).isel(station=i, lev=-1)
        plt_da = plt_ds[var]# + plt_ds['dNdlogD_sec']
        plt_da = plt_da.groupby('time.hour').median()
        plt_da.plot(label=case_name, yscale='log')#x='hour')#, yscale='log', ylim=[3,500], , norm=mcolors.LogNorm(vmin=10,vmax=8e3), cmap=cmap, ax=ax)
        ax.set_title(f'{case_name}, {stations[i]}')
        
    for case_name in cases_orig:#, axs[1:]):
        plt_ds = dic_sd[case_name]
                  #['dNdlogD_mod']+dic_sd[case_name]['dNdlogD_sec'])
        plt_ds = plt_ds.sel(time=slice(st,et)).isel(station=i, lev=-1)
        plt_da = plt_ds[var]# + plt_ds['dNdlogD_sec']
        plt_da = plt_da.groupby('time.hour').median()
        
        plt_da.plot(label=case_name, yscale='log')#x='hour', yscale='log', ylim=[3,500], label=case_name, norm=mcolors.LogNorm(vmin=10,vmax=8e3), cmap=cmap, ax=ax)
        ax.set_title(f'{case_name}, {stations[i]}')
    #    plt_da = (dic_sd[case_name]['dNdlogD_mod'])
    #    plt_da = plt_da.sel(time=slice(st,et)).isel(station=i, lev=-1)
    #    
    #    plt_da.plot(x='time', yscale='log', ylim=[3,500], label=case_name, norm=mcolors.LogNorm(vmin=10,vmax=8e3), cmap='Reds', ax=ax)
    #    ax.set_title(case_name)

    plt.legend()
    plt.show()


# %%
plt_ds['dNdlogD_mode01']

# %%
import matplotlib.colors as mcolors
st = '2008-03-01'
et = '2008-06-01'
cmap='Reds'
norm = mcolors.LogNorm(vmin=50,vmax=2e3)
for i in np.arange(len(stations)):
    fig, axs = plt.subplots(3, figsize=[10,10], sharex=True)
    for case_name in cases_sec:
        ax = axs[0]
        plt_ds = dic_sd[case_name]
                  #['dNdlogD_mod']+dic_sd[case_name]['dNdlogD_sec'])
        plt_ds = plt_ds.sel(time=slice(st,et)).isel(station=i, lev=-1)
        plt_da = plt_ds['dNdlogD_mode01']# + plt_ds['dNdlogD_sec']
        plt_da = plt_da.groupby('time.hour').median()
        plt_da.plot(x='hour', yscale='log', ylim=[3,500], label=case_name, norm=norm
                    , cmap=cmap, ax=ax)
        ax.set_title(f'{case_name}, {stations[i]}')
        
    for case_name, ax in zip(cases_orig, axs[1:]):
        plt_ds = dic_sd[case_name]
                  #['dNdlogD_mod']+dic_sd[case_name]['dNdlogD_sec'])
        plt_ds = plt_ds.sel(time=slice(st,et)).isel(station=i, lev=-1)
        plt_da = plt_ds['dNdlogD_mode01']# + plt_ds['dNdlogD_sec']
        plt_da = plt_da.groupby('time.hour').median()
        
        plt_da.plot(x='hour', yscale='log', ylim=[3,500], label=case_name, norm=norm, cmap=cmap, ax=ax)
        ax.set_title(f'{case_name}, {stations[i]}')
    #    plt_da = (dic_sd[case_name]['dNdlogD_mod'])
    #    plt_da = plt_da.sel(time=slice(st,et)).isel(station=i, lev=-1)
    #    
    #    plt_da.plot(x='time', yscale='log', ylim=[3,500], label=case_name, norm=mcolors.LogNorm(vmin=10,vmax=8e3), cmap='Reds', ax=ax)
    #    ax.set_title(case_name)

        
    plt.show()


# %%
import matplotlib.colors as mcolors
st = '2008-03-01'
et = '2008-06-01'
for i in np.arange(len(stations)):
    fig, axs = plt.subplots(3, figsize=[10,10], sharex=True)
    for case_name in cases_sec:
        ax = axs[0]
        plt_ds = dic_sd[case_name]
                  #['dNdlogD_mod']+dic_sd[case_name]['dNdlogD_sec'])
        plt_ds = plt_ds.sel(time=slice(st,et)).isel(station=i, lev=-1)
        plt_da = plt_ds['dNdlogD_mod'] + plt_ds['dNdlogD_sec']
        plt_da = plt_da.groupby('time.hour').median()
        plt_da.plot(x='hour', yscale='log', ylim=[3,500], label=case_name, norm=mcolors.LogNorm(vmin=10,vmax=8e3), cmap='Reds', ax=ax)
        ax.set_title(f'{case_name}, {stations[i]}')
        
    for case_name, ax in zip(cases_orig, axs[1:]):
        plt_ds = dic_sd[case_name]
                  #['dNdlogD_mod']+dic_sd[case_name]['dNdlogD_sec'])
        plt_ds = plt_ds.sel(time=slice(st,et)).isel(station=i, lev=-1)
        plt_da = plt_ds['dNdlogD_mod']# + plt_ds['dNdlogD_sec']
        plt_da = plt_da.groupby('time.hour').median()
        
        plt_da.plot(x='hour', yscale='log', ylim=[3,500], label=case_name, norm=mcolors.LogNorm(vmin=10,vmax=8e3), cmap='Reds', ax=ax)
        ax.set_title(f'{case_name}, {stations[i]}')
    #    plt_da = (dic_sd[case_name]['dNdlogD_mod'])
    #    plt_da = plt_da.sel(time=slice(st,et)).isel(station=i, lev=-1)
    #    
    #    plt_da.plot(x='time', yscale='log', ylim=[3,500], label=case_name, norm=mcolors.LogNorm(vmin=10,vmax=8e3), cmap='Reds', ax=ax)
    #    ax.set_title(case_name)

        
    plt.show()


# %%
import matplotlib.colors as mcolors
st = '2008-04-09'
et = '2008-04-16'
norm = mcolors.LogNorm(vmin=20,vmax=2e3)
for i in np.arange(len(stations)):
    fig, axs = plt.subplots(3, figsize=[10,10], sharex=True)
    for case_name in cases_sec:
        ax = axs[0]
        plt_ds = dic_sd[case_name]
                  #['dNdlogD_mod']+dic_sd[case_name]['dNdlogD_sec'])
        plt_ds = plt_ds.sel(time=slice(st,et)).isel(station=i, lev=-15)
        plt_da = plt_ds['dNdlogD_mode01'] + plt_ds['dNdlogD_sec']
        plt_da = plt_da.groupby('time.hour').median()
        plt_da.plot(x='hour', yscale='log', ylim=[3,500], label=case_name, norm=norm, cmap='Reds', ax=ax)
        ax.set_title(f'{case_name}, {stations[i]}')
        
    for case_name, ax in zip(cases_orig, axs[1:]):
        plt_ds = dic_sd[case_name]
                  #['dNdlogD_mod']+dic_sd[case_name]['dNdlogD_sec'])
        plt_ds = plt_ds.sel(time=slice(st,et)).isel(station=i, lev=-15)
        plt_da = plt_ds['dNdlogD_mode01']# + plt_ds['dNdlogD_sec']
        
        plt_da = plt_da.groupby('time.hour').median()
        
        plt_da.plot(x='hour', yscale='log', ylim=[3,500], label=case_name, norm=norm, cmap='Reds', ax=ax)
        ax.set_title(f'{case_name}, {stations[i]}')
    #    plt_da = (dic_sd[case_name]['dNdlogD_mod'])
    #    plt_da = plt_da.sel(time=slice(st,et)).isel(station=i, lev=-1)
    #    
    #    plt_da.plot(x='time', yscale='log', ylim=[3,500], label=case_name, norm=mcolors.LogNorm(vmin=10,vmax=8e3), cmap='Reds', ax=ax)
    #    ax.set_title(case_name)

        
    plt.show()


# %%
