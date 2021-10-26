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
cases_orig =['noSECTv21_default_dd'] 
#cases_orig =['noSECTv21_ox_ricc']

cases = cases_orig + cases_sec

# %% [markdown]
# ## Combined plots

# %%
import cartopy.crs as ccrs
from oas_dev.util.plot.plot_levlat import plot_levlat_diff, get_cbar_eq_kwargs, make_cbar_kwargs


# %% [markdown]
# ## N_AER

# %% [markdown]
# ## H2SO4

# %%
from oas_dev.util.plot.colors import get_area_col
from matplotlib.lines import Line2D
import matplotlib.ticker as mtick
from matplotlib.ticker import ScalarFormatter
from oas_dev.data_info import get_nice_name_case


# %%
linests = ['solid','dashed','dotted']

# %%
areas = ['Global','notLand','Amazonas and surroundings', 'Polar N','Polar S']


# %%
import seaborn as sns


# %%
        
def plot_levlat_map_together(var, areas, cases, axs, var_map=None,
                            ylim = [1e3, 100], relative=True,
                             yscale='log',
                             cba_kwargs=None,
                             cbar_orientation='horizontal'
                            ):
    if axs is None:
        fig = plt.figure(figsize = [width,asp_rat*width])
        gs = gridspec.GridSpec(2, 2,height_ratios=[1,1.], width_ratios=[5,1])#width_ratios=[2, 1]) 
        ax1 = plt.subplot(gs[1,0])
        ax2 = plt.subplot(gs[1,1])
        ax3 = plt.subplot(gs[0,:], projection=ccrs.Robinson())
        axs=[ax1,ax2,ax3]
        ax2.axis('off')
        cases_nn = [get_nice_name_case(case) for case in cases]
        
        set_legend_area_profs(ax2, areas,cases_nn, linestd_nn)
    ax1 = axs[0]
    ax3 = axs[1]
    
    
    #cmapd = get_cmap_dic(areas)
    
    #linestd=dict()
    #linestd_nn=dict()
    #for case, ls in zip(cases, linests):
    #    linestd[case]=ls
    #    linestd_nn[get_nice_name_case(case)]=ls
    ax =ax1# plt.subplots(1, figsize=[6,8])
    
    #for area in areas:
    cases_dic = get_averaged_fields.get_levlat_cases(cases, [var], startyear, endyear,
                                                 pressure_adjust=pressure_adjust)

    

    plot_levlat_diff(var, cases[0], cases[1], cases_dic,
                         cbar_orientation=cbar_orientation,
                         relative=relative,
                         ylim=ylim,
                         yscale=yscale,
                         ax=ax1,
                         #norm=norm, 
                     )
    
    ax.grid(False, which='both')
    sns.despine(ax=ax)
    ax.set_yscale('log')
    
    
    set_scalar_formatter(ax)
    
    # maps:
    if var_map is not None:
        var=var_map
    maps_dic = get_averaged_fields.get_maps_cases(cases,[var],startyear, endyear,
                                       avg_over_lev=avg_over_lev,
                                       pmin=pmin,
                                       pressure_adjust=pressure_adjust)
    plot_map_diff_only(var, cases, maps_dic, relative=True, cbar_equal=True,
                              kwargs_diff={}, axs=ax3, cmap_diff='RdBu_r', cbar_loc='under')
    
    
    #plt.tight_layout()
    return 

def plt_prof_map_together_ls(var1,var2, areas, cases, asp_rat=1, width=5.5, varl_map=None):
    nvars = 2
    fig = plt.figure(figsize = [width*nvars,asp_rat*width*nvars])
    gs = gridspec.GridSpec(2, nvars+1,height_ratios=[1,1.], width_ratios=[5,5,1])#width_ratios=[2, 1]) 
    axs_prof = []
    axs_maps = []
    ax1 = plt.subplot(gs[1,0])
    ax2 = None#plt.subplot(gs[1,1+i*2])
    ax3 = plt.subplot(gs[0,0], projection=ccrs.Robinson())
    print(var1,areas, cases, [ax1,ax3])
    if varl_map is None:
        var1m =None
        var2m =None
    else:
        var1m=varl_map[0]
        var2m=varl_map[1]
        
    plot_levlat_map_together(var1, areas, cases, [ax1,ax3], var_map=var1m)
    axs_maps.append(ax3)
    axs_prof.append(ax1)
    ax1 = plt.subplot(gs[1,1])
    ax2 = plt.subplot(gs[1,2])
    ax3 = plt.subplot(gs[0,1], projection=ccrs.Robinson())
    plot_levlat_map_together(var2, areas, cases, [ax1,ax3], var_map=var2m)
    axs_maps.append(ax3)
    axs_prof.append(ax1)
    
    
    ax2.axis('off')
    linestd=dict()
    linestd_nn=dict()
    for case, ls in zip(cases, linests):
        linestd[case]=ls
        linestd_nn[get_nice_name_case(case)]=ls
    ax =ax1# plt.subplots(1, figsize=[6,8])
    cases_nn = [get_nice_name_case(case) for case in cases]
    
    #set_legend_area_profs(ax2, areas,cases_nn, linestd_nn)
    
    
    ax1.yaxis.set_ticklabels([])
    ax1.set_ylabel('')
    return fig, axs_maps, axs_prof


# %% [markdown]
# ## NA-mode

# %%
cases


# %%
varl = ['N_AER','NCONC01']
fig, axs_maps, axs_prof = plt_prof_map_together_ls(*varl, areas, cases, asp_rat=.6, width=5.5)
plt.tight_layout()
vars_n = '_'.join(varl)
fn_figure = '%s%s_%s-%s.'%(fn_base,vars_n,startyear, endyear)
print(fn_figure)
#axs_prof[0].set_xlim([1e-13,5e-11])
#axs_prof[1].set_xlim([1e-13,5e-11])
#plt.savefig(fn_figure + 'png')
#plt.savefig(fn_figure + 'pdf', dpi=300)

plt.show()


# %%
varl = ['N_AER','NCONC01']
fig, axs_maps, axs_prof = plt_prof_map_together_ls(*varl, areas, cases, asp_rat=.6, width=5.5)
plt.tight_layout()
vars_n = '_'.join(varl)
fn_figure = '%s%s_%s-%s.'%(fn_base,vars_n,startyear, endyear)
print(fn_figure)
#axs_prof[0].set_xlim([1e-13,5e-11])
#axs_prof[1].set_xlim([1e-13,5e-11])
#plt.savefig(fn_figure + 'png')
#plt.savefig(fn_figure + 'pdf', dpi=300)

plt.show()


# %%
varl = ['SOA_NA','SO4_NA']
fig, axs_maps, axs_prof = plt_prof_map_together_ls(*varl, areas, cases, asp_rat=.6, width=5.5)
plt.tight_layout()
vars_n = '_'.join(varl)
fn_figure = '%s%s_%s-%s.'%(fn_base,vars_n,startyear, endyear)
print(fn_figure)
axs_prof[0].set_xlim([1e-13,5e-11])
axs_prof[1].set_xlim([1e-13,5e-11])
#plt.savefig(fn_figure + 'png')
#plt.savefig(fn_figure + 'pdf', dpi=300)

plt.show()


# %% [markdown]
# ## AREL,AWNC

# %%

varl = ['AREL_incld','AWNC_incld']
fig, axs_maps, axs_prof = plt_prof_map_together_ls(*varl, areas, cases, asp_rat=.6, width=5.5)
plt.tight_layout()
vars_n = '_'.join(varl)
fn_figure = '%s%s_%s-%s.'%(fn_base,vars_n,startyear, endyear)
print(fn_figure)
axs_prof[0].set_xscale('linear')#([1e-13,5e-11])
axs_prof[1].set_xscale('linear')#([1e-13,5e-11])
#axs_prof[1].set_xlim([1e-13,5e-11])
plt.savefig(fn_figure + 'png')
plt.savefig(fn_figure + 'pdf', dpi=300)

plt.show()


# %% [markdown]
# ## AREL,AWNC,ACTREL,ACTNL

# %%

varl = ['AREL_incld','AWNC_incld']
varl_maps=['ACTREL_incld','ACTNL_incld']
fig, axs_maps, axs_prof = plt_prof_map_together_ls(*varl, areas, cases, asp_rat=.6, 
                                                   width=5.5, varl_map=varl_maps)
plt.tight_layout()
vars_n = '_'.join(varl)
fn_figure = '%s%s_%s-%s.'%(fn_base,vars_n,startyear, endyear)
print(fn_figure)
axs_prof[0].set_xscale('linear')#([1e-13,5e-11])
axs_prof[1].set_xscale('linear')#([1e-13,5e-11])
#axs_prof[1].set_xlim([1e-13,5e-11])
plt.savefig(fn_figure + 'png')
plt.savefig(fn_figure + 'pdf', dpi=300)

plt.show()


# %% [markdown]
# # Other areas

# %%
areas=['Global','landOnly','notLand', 'Polar N','Polar S']#,'Boreal forest']

# %%
from oas_dev.util.plot.plot_maps import make_box

# %% [markdown]
# ## NA-mode

# %%
varl = ['SOA_NA','SO4_NA']
fig, axs_maps, axs_prof = plt_prof_map_together_ls(*varl, areas, cases, asp_rat=.6, width=5.5)
plt.tight_layout()
vars_n = '_'.join(varl)
fn_figure = '%s%s_%s-%s.'%(fn_base,vars_n,startyear, endyear)
print(fn_figure)
axs_prof[0].set_xlim([1e-13,5e-11])
axs_prof[1].set_xlim([1e-13,5e-11])
#plt.savefig(fn_figure + 'png')
#plt.savefig(fn_figure + 'pdf', dpi=300)

plt.show()


# %% [markdown]
# ## AREL,AWNC,ACTREL...

# %%

varl = ['AREL_incld','AWNC_incld']
varl_maps=['ACTREL_incld','ACTNL_incld']
fig, axs_maps, axs_prof = plt_prof_map_together_ls(*varl, areas, cases, asp_rat=.6, 
                                                   width=5.5, varl_map=varl_maps)
plt.tight_layout()
vars_n = '_'.join(varl)
fn_figure = '%s%s_%s-%s.'%(fn_base,vars_n,startyear, endyear)
print(fn_figure)
axs_prof[0].set_xscale('linear')#([1e-13,5e-11])
axs_prof[1].set_xscale('linear')#([1e-13,5e-11])
#axs_prof[1].set_xlim([1e-13,5e-11])
#plt.savefig(fn_figure + 'png')
#plt.savefig(fn_figure + 'pdf', dpi=300)

plt.show()


# %% [markdown]
# ## FREQL,FREQI

# %%

varl = ['FREQL','FREQI']
varl_maps=['ACTREL_incld','ACTNL_incld']
fig, axs_maps, axs_prof = plt_prof_map_together_ls(*varl, areas, cases, asp_rat=.6, 
                                                   width=5.5, varl_map=varl_maps)
plt.tight_layout()
vars_n = '_'.join(varl)
fn_figure = '%s%s_%s-%s.'%(fn_base,vars_n,startyear, endyear)
print(fn_figure)
axs_prof[0].set_xscale('linear')#([1e-13,5e-11])
axs_prof[1].set_xscale('linear')#([1e-13,5e-11])
#axs_prof[1].set_xlim([1e-13,5e-11])
#plt.savefig(fn_figure + 'png')
#plt.savefig(fn_figure + 'pdf', dpi=300)

plt.show()


# %% [markdown]
# ## N_AER, NCONC01

# %%

varl = ['N_AER','NCONC01']
varl_maps=None#['ACTREL','ACTNL']
fig, axs_maps, axs_prof = plt_prof_map_together_ls(*varl, areas, cases, asp_rat=.6, 
                                                   width=5.5, varl_map=varl_maps)
plt.tight_layout()
vars_n = '_'.join(varl)
fn_figure = '%s%s_%s-%s.'%(fn_base,vars_n,startyear, endyear)
print(fn_figure)
axs_prof[0].set_xscale('linear')#([1e-13,5e-11])
axs_prof[1].set_xscale('linear')#([1e-13,5e-11])
#axs_prof[1].set_xlim([1e-13,5e-11])
#plt.savefig(fn_figure + 'png')
#plt.savefig(fn_figure + 'pdf', dpi=300)

plt.show()


# %%
