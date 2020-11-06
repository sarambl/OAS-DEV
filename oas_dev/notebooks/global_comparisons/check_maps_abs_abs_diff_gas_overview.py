# -*- coding: utf-8 -*-
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
from sectional_v2.util.plot.plot_maps import plot_map_diff, fix_axis4map_plot, plot_map_abs_abs_diff, plot_map
from useful_scit.imps import (np, xr, plt, pd) 
from sectional_v2.util.imports import get_averaged_fields
from IPython.display import clear_output

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
# %% [markdown]
# ## Ideas:
# - Root mean square diffence??
# - Scatter plots of all values, e.g x-- sectional y-- non sectional color by lat/lev? Or lev lat difference. 

# %% [markdown]
# # Map plots number concentration:

# %%
model = 'NorESM'

startyear = '2008-01'
endyear = '2009-12'
pmin = 850.  # minimum pressure level
avg_over_lev = True  # True#True#False#True
pressure_adjust = True  # Can only be false if avg_over_lev false. Plots particular hybrid sigma lev
if avg_over_lev:
    pressure_adjust = True
p_levels = [1013.,900., 800., 700., 600.]  # used if not avg

# %% [markdown]
# ## Cases

# %%
cases_sec = ['SECTv21_ctrl_koagD']
cases_orig =['noSECTv21_default'] 
cases_orig =['noSECTv21_ox_ricc_dd']

cases = cases_orig + cases_sec

cases2 = ['noSECTv21_default_dd']+cases_sec #['noSECTv11_ctrl_fbvoc', 'noSECTv11_noresm2_ctrl']



# %%
def load_and_plot(var, cases,startyear, endyear,
                                   avg_over_lev=avg_over_lev,
                                   pmin=pmin,
                                   pressure_adjust=pressure_adjust, p_level=None, relative=False, kwargs_diff=None):
    maps_dic = get_averaged_fields.get_maps_cases(cases,[var],startyear, endyear,
                                   avg_over_lev=avg_over_lev,
                                   pmin=pmin,
                                   pressure_adjust=pressure_adjust, p_level=p_level)
    return plot_map_abs_abs_diff(var, cases, maps_dic, relative=relative, figsize=[18, 3], cbar_equal=True,
                          kwargs_abs={},
                          kwargs_diff=kwargs_diff, axs=None, cmap_abs='Reds', cmap_diff='RdBu_r')
    


# %% [markdown]
# ## Mean to 850hPa weighted by pressure difference:

# %%
axs = load_and_plot('N_AER', cases, startyear, endyear, avg_over_lev, pmin=pmin, pressure_adjust=pressure_adjust)

# %%
axs = load_and_plot('N_AER', cases, startyear, endyear, avg_over_lev, pmin=pmin,relative=True, pressure_adjust=pressure_adjust)

# %% [markdown]
# ### Particle number from nucleation:

# %% [markdown]
# # In sectional scheme:
#

# %%
from sectional_v2.util.plot.plot_maps import subplots_map
varl= ['nrSOA_SEC_tot', 'nrSO4_SEC_tot','nrSEC_tot' ]
fig, axs = subplots_map(1, len(varl), figsize=[15,3])
for var, ax in zip(varl,axs):
    maps_dic = get_averaged_fields.get_maps_cases(cases_sec,[var],startyear, endyear,
                                   avg_over_lev=avg_over_lev,
                                   pmin=pmin,
                                   pressure_adjust=pressure_adjust)

    plot_map(var, cases_sec[0], maps_dic,  figsize=[6, 3],
                          kwargs_abs={}, ax=ax, cmap_abs='Reds')
    
plt.show()

# %%
from sectional_v2.util.plot.plot_maps import subplots_map
varl= ['cb_SOA_SEC01', 'cb_SOA_SEC02','cb_SOA_SEC03' ]
fig, axs = subplots_map(1, len(varl), figsize=[15,3])
for var, ax in zip(varl,axs):
    maps_dic = get_averaged_fields.get_maps_cases(cases_sec,[var],startyear, endyear,
                                   avg_over_lev=avg_over_lev,
                                   pmin=pmin,
                                   pressure_adjust=pressure_adjust)

    plot_map(var, cases_sec[0], maps_dic,  figsize=[6, 3],
                          kwargs_abs={}, ax=ax, cmap_abs='Reds')
    
plt.show()

# %%

# %%

# %%
axs = load_and_plot('isoprene', cases, startyear, endyear, avg_over_lev, pmin=pmin, pressure_adjust=pressure_adjust, relative=True)

# %%
axs = load_and_plot('SO2', cases, startyear, endyear, avg_over_lev, pmin=pmin, pressure_adjust=pressure_adjust, relative=True, 
                    kwargs_diff=dict(vmin=-40, vmax=-10, cmap='Blues_r'))

# %%
axs = load_and_plot('SO2', cases, startyear, endyear, avg_over_lev, pmin=pmin, pressure_adjust=pressure_adjust, relative=False)

# %%
axs = load_and_plot('SFSO2', cases, startyear, endyear, avg_over_lev, pmin=pmin, pressure_adjust=pressure_adjust, relative=False)

# %%
axs = load_and_plot('DF_SO2', cases, startyear, endyear, avg_over_lev, pmin=pmin, pressure_adjust=pressure_adjust, relative=False)

# %%
axs = load_and_plot('GS_SO2', cases, startyear, endyear, avg_over_lev, pmin=pmin, pressure_adjust=pressure_adjust, relative=False)

# %%
axs = load_and_plot('WD_SO2', cases, startyear, endyear, avg_over_lev, pmin=pmin, pressure_adjust=pressure_adjust, relative=False)

# %%
axs = load_and_plot('GS_H2SO4', cases, startyear, endyear, avg_over_lev, pmin=pmin, pressure_adjust=pressure_adjust, relative=False)

# %%
axs = load_and_plot('DMS', cases, startyear, endyear, avg_over_lev, pmin=pmin, pressure_adjust=pressure_adjust, relative=False)

# %%
axs = load_and_plot('isoprene', cases, startyear, endyear, avg_over_lev, pmin=pmin, pressure_adjust=pressure_adjust)

# %%
axs = load_and_plot('SFisoprene', cases, startyear, endyear, False, pmin=pmin, pressure_adjust=pressure_adjust)

# %%
axs = load_and_plot('SFisoprene', cases, startyear, endyear, False, pmin=pmin, pressure_adjust=pressure_adjust, relative=True)

# %%
axs = load_and_plot('SFmonoterp', cases, startyear, endyear, False, pmin=pmin, pressure_adjust=pressure_adjust)

# %%
axs = load_and_plot('SFmonoterp', cases, startyear, endyear, False, pmin=pmin, pressure_adjust=pressure_adjust, relative=True)

# %%
axs = load_and_plot('monoterp', cases, startyear, endyear, avg_over_lev, pmin=pmin, pressure_adjust=pressure_adjust)

# %%
axs = load_and_plot('monoterp', cases, startyear, endyear, avg_over_lev, pmin=pmin, pressure_adjust=pressure_adjust, relative=True)

# %%
axs = load_and_plot('SOA_LV', cases, startyear, endyear, avg_over_lev, pmin=pmin, pressure_adjust=pressure_adjust)

# %%
axs = load_and_plot('SOA_LV', cases, startyear, endyear, avg_over_lev, pmin=pmin, pressure_adjust=pressure_adjust, relative=True)

# %%
axs = load_and_plot('SOA_NA', cases, startyear, endyear, avg_over_lev, pmin=pmin, pressure_adjust=pressure_adjust)

# %%
axs = load_and_plot('SOA_NA', cases, startyear, endyear, avg_over_lev, pmin=pmin, pressure_adjust=pressure_adjust, relative=True)

# %%
axs = load_and_plot('SOA_A1', cases, startyear, endyear, avg_over_lev, pmin=pmin, pressure_adjust=pressure_adjust)

# %%
axs = load_and_plot('SOA_A1', cases, startyear, endyear, avg_over_lev, pmin=pmin, pressure_adjust=pressure_adjust, relative=True)

# %%
axs = load_and_plot('cb_monoterp', cases, startyear, endyear, avg_over_lev, pmin=pmin, pressure_adjust=pressure_adjust)

# %%
axs = load_and_plot('cb_monoterp', cases, startyear, endyear, avg_over_lev, pmin=pmin, pressure_adjust=pressure_adjust, relative=True)

# %%
axs = load_and_plot('cb_SOA_LV', cases, startyear, endyear, avg_over_lev, pmin=pmin, pressure_adjust=pressure_adjust)

# %%
axs = load_and_plot('cb_SOA_LV', cases, startyear, endyear, avg_over_lev, pmin=pmin, pressure_adjust=pressure_adjust, relative=True)

# %%
axs = load_and_plot('cb_SOA_NA', cases, startyear, endyear, avg_over_lev, pmin=pmin, pressure_adjust=pressure_adjust)

# %%
axs = load_and_plot('cb_SOA_NA', cases, startyear, endyear, avg_over_lev, pmin=pmin, pressure_adjust=pressure_adjust, relative=True)

# %%
axs = load_and_plot('cb_SOA_A1', cases, startyear, endyear, avg_over_lev, pmin=pmin, pressure_adjust=pressure_adjust)

# %%
axs = load_and_plot('cb_SOA_A1', cases, startyear, endyear, avg_over_lev, pmin=pmin, pressure_adjust=pressure_adjust, relative=True)

# %% [markdown]
# ### Total number concentration:

# %%
maps_dic = get_averaged_fields.get_maps_cases(cases,['N_AER'],startyear, endyear,
                                   avg_over_lev=avg_over_lev,
                                   pmin=pmin,
                                   pressure_adjust=pressure_adjust)#, p_level=p_level)

# %%
axs = load_and_plot('N_AER', cases, startyear, endyear, avg_over_lev, pmin=pmin, pressure_adjust=pressure_adjust)

# %%
axs = load_and_plot('N_AER', cases, startyear, endyear, avg_over_lev, pmin=pmin,relative=True, pressure_adjust=pressure_adjust)

# %% [markdown]
# ### Particle number from nucleation:

# %% [markdown]
# # In sectional scheme:
#

# %%
from sectional_v2.util.plot.plot_maps import subplots_map
varl= ['nrSOA_SEC_tot', 'nrSO4_SEC_tot','nrSEC_tot' ]
fig, axs = subplots_map(1, len(varl), figsize=[15,3])
for var, ax in zip(varl,axs):
    maps_dic = get_averaged_fields.get_maps_cases(cases_sec,[var],startyear, endyear,
                                   avg_over_lev=avg_over_lev,
                                   pmin=pmin,
                                   pressure_adjust=pressure_adjust)

    plot_map(var, cases_sec[0], maps_dic,  figsize=[6, 3],
                          kwargs_abs={}, ax=ax, cmap_abs='Reds')
    
plt.show()

# %%
from sectional_v2.util.plot.plot_maps import subplots_map
varl= ['cb_SOA_SEC01', 'cb_SOA_SEC02','cb_SOA_SEC03' ]
fig, axs = subplots_map(1, len(varl), figsize=[15,3])
for var, ax in zip(varl,axs):
    maps_dic = get_averaged_fields.get_maps_cases(cases_sec,[var],startyear, endyear,
                                   avg_over_lev=avg_over_lev,
                                   pmin=pmin,
                                   pressure_adjust=pressure_adjust)

    plot_map(var, cases_sec[0], maps_dic,  figsize=[6, 3],
                          kwargs_abs={}, ax=ax, cmap_abs='Reds')
    
plt.show()
