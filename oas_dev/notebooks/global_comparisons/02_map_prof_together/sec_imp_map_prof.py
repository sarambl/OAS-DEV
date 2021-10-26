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
from oas_dev.constants import get_plotpath
from oas_dev.util.practical_functions import make_folders
from IPython.display import clear_output



plot_path = get_plotpath('comparison')
print(plot_path)
make_folders(plot_path)
fn_base = plot_path + '/prof_map_'
print(fn_base)

# %%
from oas_dev.util.plot.plot_maps import plot_map_diff, fix_axis4map_plot, plot_map_abs_abs_diff, plot_map,plot_map_diff_only
from useful_scit.imps import (np, xr, plt, pd) 
from oas_dev.util.imports import get_averaged_fields
from oas_dev.util.plot.plot_profiles import plot_profile, set_legend_area_profs, set_scalar_formatter

# load and autoreload
from useful_scit.plot import get_cmap_dic
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
from matplotlib import gridspec


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

# %%
cases_sec = ['SECTv21_ctrl']
cases_orig =['noSECTv21_default'] 
cases_orig =['noSECTv21_ox_ricc']

cases = cases_orig + cases_sec

# %% [markdown]
# # Combined plots

# %%
import cartopy.crs as ccrs

# %%
from oas_dev.util.plot.colors import get_area_col
from matplotlib.lines import Line2D
import matplotlib.ticker as mtick
from matplotlib.ticker import ScalarFormatter
from oas_dev.data_info import get_nice_name_case


# %%
linests = ['solid','dashed','dotted']

# %% [markdown]
# ## Area defs 1:

# %%
areas = ['Global','landOnly','notLand', 'Polar N','Polar S']


# %%
import seaborn as sns


# %% [markdown]
# ### Code
#

# %%
def plt_prof_map_together(var, areas, cases, asp_rat=1, width=6):
    fig = plt.figure(figsize = [width,asp_rat*width])
    gs = gridspec.GridSpec(2, 2,height_ratios=[1,1.], width_ratios=[5,1] )#width_ratios=[2, 1]) 
    ax1 = plt.subplot(gs[1,0])
    ax2 = plt.subplot(gs[1,1])
    ax3 = plt.subplot(gs[0,:], projection=ccrs.Robinson())
    ax2.axis('off')
    
    cmapd = get_cmap_dic(areas)
    
    linestd=dict()
    linestd_nn=dict()
    for case, ls in zip(cases, linests):
        linestd[case]=ls
        linestd_nn[get_nice_name_case(case)]=ls
    ax =ax1# plt.subplots(1, figsize=[6,8])
    
    for area in areas:
        prof_dic = get_averaged_fields.get_profiles(cases,[var],startyear, endyear,area=area,
                                                  pressure_adjust=pressure_adjust)
    
        for case in cases:
            kwargs = dict(color=get_area_col(area), linestyle=linestd[case])
            plot_profile(prof_dic[case][var], 
                         ax=ax, 
                         kwargs=kwargs, 
                         xscale='log', 
                         label=case+', '+ area,
                         ylim=[1000,200])#, 
    ax.grid(False, which='both')
    sns.despine(ax=ax)
    ax.set_yscale('log')
    
    
    set_scalar_formatter(ax)
    cases_nn = [get_nice_name_case(case) for case in cases]
    set_legend_area_profs(ax2, areas,cases_nn, linestd_nn)
    # maps:

    maps_dic = get_averaged_fields.get_maps_cases(cases,[var],startyear, endyear,
                                       avg_over_lev=avg_over_lev,
                                       pmin=pmin,
                                       pressure_adjust=pressure_adjust)
    plot_map_diff_only(var, cases, maps_dic, relative=True, cbar_equal=True,
                              kwargs_diff={}, axs=ax3, cmap_diff='RdBu_r')#, cbar_loc='under')
    
    
    #plt.tight_layout()
    return fig, [ax1,ax3]


# %% [markdown]
# ### SOA_NA/SO4_NA

# %%
var = 'SOA_NA'
fig, axs = plt_prof_map_together(var, areas, cases, asp_rat=1)
axs[0].set_xlim([1e-13,1e-11])
fn_figure = '%s%s_%s-%s.'%(fn_base,var,startyear, endyear)
plt.savefig(fn_figure + 'png')
plt.savefig(fn_figure + 'pdf', dpi=300)

plt.show()


# %%
var = 'SO4_NA'
fig, axs = plt_prof_map_together(var, areas, cases, asp_rat=1)
fn_figure = '%s%s_%s-%s.'%(fn_base,var,startyear, endyear)
plt.savefig(fn_figure + 'png')
plt.savefig(fn_figure + 'pdf', dpi=300)

plt.show()


# %% [markdown]
# ### NCONC01

# %%
var = 'NCONC01'
fig, axs = plt_prof_map_together(var, areas, cases, asp_rat=1)
axs[0].set_xlim([10,2e3])
fn_figure = '%s%s_%s-%s.'%(fn_base,var,startyear, endyear)
plt.savefig(fn_figure + 'png')
plt.savefig(fn_figure + 'pdf', dpi=300)

plt.show()


# %% [markdown]
# ### N_AER

# %%
var = 'N_AER'
fig, axs = plt_prof_map_together(var, areas, cases, asp_rat=1)
axs[0].set_xlim([10,2e3])
fn_figure = '%s%s_%s-%s.'%(fn_base,var,startyear, endyear)
plt.savefig(fn_figure + 'png')
plt.savefig(fn_figure + 'pdf', dpi=300)

plt.show()


# %% [markdown]
# ### SOA_SV:

# %%
var = 'SOA_SV'
fig, axs = plt_prof_map_together(var, areas, cases)
plt.show()


# %% [markdown]
# ## Area defs 2:

# %%
areas = ['Global','landOnly','notLand', 'Polar N','Polar S','Amazonas and surroundings']


# %% [markdown]
# ### SOA_NA/SO4_NA

# %%
var = 'SOA_NA'
fig, axs = plt_prof_map_together(var, areas, cases, asp_rat=1)
axs[0].set_xlim([1e-13,1e-11])
fn_figure = '%s%s_%s-%s.'%(fn_base,var,startyear, endyear)
plt.savefig(fn_figure + 'png')
plt.savefig(fn_figure + 'pdf', dpi=300)

plt.show()


# %%
var = 'SO4_NA'
fig, axs = plt_prof_map_together(var, areas, cases, asp_rat=1)
fn_figure = '%s%s_%s-%s.'%(fn_base,var,startyear, endyear)
plt.savefig(fn_figure + 'png')
plt.savefig(fn_figure + 'pdf', dpi=300)

plt.show()


# %% [markdown]
# ### NCONC01

# %%
var = 'NCONC01'
fig, axs = plt_prof_map_together(var, areas, cases, asp_rat=1)
axs[0].set_xlim([10,2e3])
fn_figure = '%s%s_%s-%s.'%(fn_base,var,startyear, endyear)
plt.savefig(fn_figure + 'png')
plt.savefig(fn_figure + 'pdf', dpi=300)

plt.show()


# %% [markdown]
# ### N_AER

# %%
var = 'N_AER'
fig, axs = plt_prof_map_together(var, areas, cases, asp_rat=1)
axs[0].set_xlim([10,2e3])
fn_figure = '%s%s_%s-%s.'%(fn_base,var,startyear, endyear)
plt.savefig(fn_figure + 'png')
plt.savefig(fn_figure + 'pdf', dpi=300)

plt.show()


# %% [markdown]
# ### SOA_SV:

# %%
var = 'SOA_SV'
fig, axs = plt_prof_map_together(var, areas, cases)
plt.show()


# %%
