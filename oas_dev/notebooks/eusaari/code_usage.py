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
# %load_ext autoreload
# %autoreload 2
import oas_dev.util.eusaar_data.flags
from oas_dev.constants import path_eusaar_data# path_eusaar_data
import numpy as np
from oas_dev.util import eusaar_data
from oas_dev.util.eusaar_data.histc_vars import load_var_as_dtframe
from oas_dev.util.eusaar_data import  distc_var, histc_vars, histsc_hists # import load_var_as_dtframe
import matplotlib.pyplot as plt
from useful_scit.plot import get_cmap_dic

# %% [markdown]
# ## distc_var

# %%
a = distc_var.get_distc_xarray_all()

# %%
sts = a['station'].values[:4]
year='BOTH'
subs='TOT'
cma = get_cmap_dic(sts)
da = a.sel(year=year,subset=subs)
for st in sts:
    _da50 = da.sel(station=st, percentile='50th percentile')#
    _da50.plot(yscale='log',xscale='log', label='50th perc, %s'%st, color = cma[st])
    _da95 = da.sel(station=st, percentile='95th percentile')
    _da5 = da.sel(station=st, percentile='5th percentile')
    plt.fill_between(da.diameter, _da5,_da95, alpha=.2 , color = cma[st])
plt.ylim(1,3e4)
plt.xlim(5,1e3)
plt.legend()
plt.title('Test sizedistributions')

# %% [markdown]
# ### Conversion dN/dlog10D to dN/dlnD:
# $$ ln(x) = ln(10)\cdot log10(x)$$
# So
# \begin{align}
# \frac{dN}{dlog10D} =& \frac{dN}{dlnD}\frac{dlnD}{dlog10D} \\
# =& \frac{dN}{dlnD}\frac{d(ln(10)\cdot log10(D)}{dlog10D} \\
# =& \frac{dN}{dlnD}ln(10) \\
# \end{align}
# and 
# $$\frac{dN}{dlnD} = 1/ln(10) \frac{dN}{dlog10D}$$

# %% [markdown]
# ## histc_var

# %%
b = histc_vars.get_histc_vars_xr()
_da = b['N30']#.plot()
for st in sts:
    _da.sel(station=st).plot(label=st, yscale='log')
plt.ylim([1e0,1e4])
plt.legend()

# %%
oas_dev.util.eusaar_data.flags.load_gd()

# %% [markdown]
# ## histsc_hists

# %%
c = histsc_hists.open_hists2xarray()

# %%
c

# %%
_da = c['N30-50 Summer']#.plot()
for st in sts:
    _da.sel(station=st).plot(label=st, xscale='log')
plt.legend()

# %%
mn =  a['N50'].resample({'time':'1M'}).mean()
for station in mn.station:
    mn.sel(station=station).plot(label=station.values)
plt.legend()

# %%
varl = eusaar_data.standard_varlist_histc
fig, axs = plt.subplots(len(varl), 
                        figsize=[10,20])
for var, ax in zip(varl, axs):
    mn =  a[var].groupby('time.month').mean()
    for station in mn.station:
        mn.sel(station=station).plot(label=station.values, ax=ax)
    ax.set_title(var)
ax.legend()

