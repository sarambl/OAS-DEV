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
from sectional_v2.util.plot.plot_maps import subplots_map, plot_map_diff_only
from useful_scit.imps import (np, plt)
from sectional_v2.util.imports import get_averaged_fields
import cartopy.crs as ccrs
# load and autoreload
from IPython import get_ipython

from sectional_v2.util.slice_average.significance import load_and_plot_sign

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
endyear = '2011-12'
p_level=1013.
pmin = 850.  # minimum pressure level
avg_over_lev = True  # True#True#False#True
pressure_adjust = True  # Can only be false if avg_over_lev false. Plots particular hybrid sigma lev
if avg_over_lev:
    pressure_adjust = True
p_levels = [1013.,900., 800., 700., 600.]  # used if not avg

# %%
from sectional_v2.constants import get_plotpath
from sectional_v2.util.practical_functions import make_folders
version='v21dd_both'
plot_path = get_plotpath('maps')
filen_base = plot_path+'/_%s'%version
#print(plot_path)
make_folders(plot_path)

# %% [markdown]
# ## Cases

# %%
#cases_sec = ['SECTv21_ctrl']
#cases_orig =['noSECTv21_default'] 
#cases_orig =['noSECTv21_ox_ricc']
to_case = 'SECTv21_ctrl_koagD'
from_cases = ['noSECTv21_default_dd','noSECTv21_ox_ricc_dd']
cases =[to_case]+from_cases


# %%

#plot_path = get_plotpath('maps')
#filen_base = plot_path+'/_%s'%version
#print(plot_path)
#make_folders(plot_path)


# %%

# %%
def load_and_plot_diff_mm(varl,to_case,from_cases,startyear, endyear,
                       avg_over_lev=avg_over_lev,
                       pmin=pmin,nr_cols=2,
                       pressure_adjust=pressure_adjust, 
                       p_level=None, 
                       relative=False,
                      width=6., height=2.3):
    cases = [to_case] + from_cases
    maps_dic = get_averaged_fields.get_maps_cases(cases,varl,startyear, endyear,
                                   avg_over_lev=avg_over_lev,
                                   pmin=pmin,
                                   pressure_adjust=pressure_adjust, p_level=p_level)
    nr_rows = int(np.ceil(len(varl)/nr_cols))
    nr_cols = len(from_cases)
    fig, axs = subplots_map(nr_rows, nr_cols, figsize=[width*nr_cols,height*nr_rows])
    
    for i, var in enumerate(varl):
        if len(varl) == 1: saxs = axs 
        else: saxs = axs[i,:]
        plot_map_diff_only(var, [to_case,*from_cases], maps_dic, relative=relative, cbar_equal=True,
                       cbar_loc='side', tight_layout=False, inverse_diff=True, axs=saxs)


    
    #for from_case,i in zip(from_cases,range(nr_cols)):
    #    sax = axs[:,i]
    #    plot_diff(maps_dic, varl, [from_case,to_case],nr_cols=nr_cols, relative=relative, width=width, axs=sax)
        
    subp_insert_abc(axs, pos_y=0.1)
    return




# %% [markdown]
# ## Mean to 850hPa weighted by pressure difference:

# %% [markdown]
# ### CCN:

# %%
from useful_scit.plot.fig_manip import subp_insert_abc

# %%

#varl=['ACTNL_incld', 'ACTREL_incld','TGCLDCWP']#,'TGCLDCWP']


# %%
varl_rel = ['ACTNL_incld', 'ACTREL_incld','TGCLDCWP']

varl_rel = ['CDNUMC', 'ACTREL_incld','TGCLDCWP']
varl_abs=['NCFT_Ghan']#,'TGCLDCWP']
varl = varl_rel+varl_abs

width=4.7
asp_rat = 0.48
relative=True

cases = [to_case] + from_cases
maps_dic = get_averaged_fields.get_maps_cases(cases,varl,startyear, endyear,
                                   avg_over_lev=avg_over_lev,
                                   pmin=pmin,
                                   pressure_adjust=pressure_adjust, p_level=p_level)

nr_cols = len(from_cases)
nr_rows = int(np.ceil(len(varl)))
fig, axs = subplots_map(nr_rows, nr_cols, figsize=[width*nr_cols,asp_rat*width*nr_rows])
for i, var in enumerate(varl):
    saxs = axs[i,:] 
    plot_map_diff_only(var, [to_case,*from_cases], maps_dic, relative=(var in varl_rel), cbar_equal=True,
                       kwargs_diff={}, axs=saxs, cmap_diff='RdBu_r',
                       cbar_loc='side', tight_layout=False, inverse_diff=True)
    load_and_plot_sign(to_case, from_cases, saxs, var, startyear, endyear, pressure_adjust=pressure_adjust,
                       avg_over_lev=avg_over_lev,
                       ci=.95,
                       groupby=None,
                       dims=('lev',),
                       area='Global',
                       avg_dim='time',
                       hatches=None, hatch_lw = 1, transform=ccrs.PlateCarree(),
                       reverse=False)
    plt.suptitle(endyear)
#for from_case,i in zip(from_cases,range(nr_cols)):
#    sax = axs[:,i]
#    for var, ax in zip(varl, sax.flatten()):
#        plot_map_diff_2case(var, from_case,to_case, maps_dic, relative=(var in varl_rel), 
#                               ax=ax, cmap_diff='RdBu_r')

subp_insert_abc(axs, pos_y=0.1)

fn = filen_base + '_'.join(varl)+f'{relative}.'
print(fn)
plt.tight_layout()
#plt.savefig(fn + 'png')
#plt.savefig(fn + 'pdf')
plt.show()
# %%
varl_rel = []#'ACTNL_incld', 'ACTREL_incld','TGCLDCWP']
varl_abs=['SWCF_Ghan', 'LWCF_Ghan', 'NCFT_Ghan']#'NCFT_Ghan']#,'TGCLDCWP']
varl = varl_rel+varl_abs
    
#varl=['ACTNL_incld', 'ACTREL_incld','TGCLDCWP']#,'TGCLDCWP']

                        
width=4.7
asp_rat = 0.48
relative=True

cases = [to_case] + from_cases
maps_dic = get_averaged_fields.get_maps_cases(cases,varl,startyear, endyear,
                                   avg_over_lev=avg_over_lev,
                                   pmin=pmin,
                                   pressure_adjust=pressure_adjust, p_level=p_level)
nr_cols = len(from_cases)
nr_rows = int(np.ceil(len(varl)))
fig, axs = subplots_map(nr_rows, nr_cols, figsize=[width*nr_cols,asp_rat*width*nr_rows])
for i, var in enumerate(varl):
    saxs = axs[i,:] 
    plot_map_diff_only(var, [to_case,*from_cases], maps_dic, relative=(var in varl_rel), cbar_equal=True,
                       kwargs_diff={}, axs=saxs, cmap_diff='RdBu_r',
                       cbar_loc='side', tight_layout=False, inverse_diff=True)

#for from_case,i in zip(from_cases,range(nr_cols)):
#    sax = axs[:,i]
#    for var, ax in zip(varl, sax.flatten()):
#        plot_map_diff_2case(var, from_case,to_case, maps_dic, relative=(var in varl_rel), 
#                               ax=ax, cmap_diff='RdBu_r')

subp_insert_abc(axs, pos_y=0.1)

#plot_diff(maps_dic, varl, cases[::-1],nr_cols=1, relative=relative)
#load_and_plot_diff_mm(varl,to_case,from_cases, startyear, endyear, avg_over_lev,  pmin=pmin, relative=relative, pressure_adjust=pressure_adjust,nr_cols=1, width=5.5)
fn = filen_base + '_'.join(varl)+f'{relative}.'
print(fn)
plt.tight_layout()
#plt.savefig(fn + 'png')
#plt.savefig(fn + 'pdf')

# %%
varl_rel = ['AWNC_incld','CDNUMC', 'AREL_incld','TGCLDCWP']
varl_abs=['NCFT_Ghan']#,'TGCLDCWP']
varl = varl_rel+varl_abs
    
#varl=['ACTNL_incld', 'ACTREL_incld','TGCLDCWP']#,'TGCLDCWP']

                        
width=4.4
asp_rat = 0.48
relative=True

cases = [to_case] + from_cases
maps_dic = get_averaged_fields.get_maps_cases(cases,varl,startyear, endyear,
                                   avg_over_lev=avg_over_lev,
                                   pmin=pmin,
                                   pressure_adjust=pressure_adjust, p_level=p_level)
nr_cols = len(from_cases)
nr_rows = int(np.ceil(len(varl)))
fig, axs = subplots_map(nr_rows, nr_cols, figsize=[width*nr_cols,asp_rat*width*nr_rows])


for i, var in enumerate(varl):
    saxs = axs[i,:] 
    plot_map_diff_only(var, [to_case,*from_cases], maps_dic, relative=(var in varl_rel), cbar_equal=True,
                       kwargs_diff={}, axs=saxs, cmap_diff='RdBu_r',
                       cbar_loc='side', tight_layout=False, inverse_diff=True)

#for from_case,i in zip([to_case, from_cases[-1]],range(nr_cols)):
#    sax = axs[:,i]
#    for var, ax in zip(varl, sax.flatten()):
#        plot_map_diff_2case(var, from_case,from_cases[0], maps_dic, relative=(var in varl_rel), 
#                               ax=ax, cmap_diff='RdBu_r')


subp_insert_abc(axs, pos_y=0.1)

#plot_diff(maps_dic, varl, cases[::-1],nr_cols=1, relative=relative)
#load_and_plot_diff_mm(varl,to_case,from_cases, startyear, endyear, avg_over_lev,  pmin=pmin, relative=relative, pressure_adjust=pressure_adjust,nr_cols=1, width=5.5)
fn = filen_base + '_'.join(varl)+f'{relative}.'
plt.tight_layout()
#plt.savefig(fn + 'png')
#plt.savefig(fn + 'pdf')

# %%
varl_rel = ['CLDHGH', 'CLDLOW','CLDMED','CLDTOT']
varl_abs=[]#'NCFT_Ghan']#,'TGCLDCWP']
varl = varl_rel+varl_abs
    
#varl=['ACTNL_incld', 'ACTREL_incld','TGCLDCWP']#,'TGCLDCWP']

                        
width=4.4
asp_rat = 0.48
relative=True

cases = [to_case] + from_cases
maps_dic = get_averaged_fields.get_maps_cases(cases,varl,startyear, endyear,
                                   avg_over_lev=avg_over_lev,
                                   pmin=pmin,
                                   pressure_adjust=pressure_adjust, p_level=p_level)
nr_cols = len(from_cases)
nr_rows = int(np.ceil(len(varl)))
fig, axs = subplots_map(nr_rows, nr_cols, figsize=[width*nr_cols,asp_rat*width*nr_rows])


for i, var in enumerate(varl):
    saxs = axs[i,:] 
    plot_map_diff_only(var, [to_case,*from_cases], maps_dic, relative=(var in varl_rel), cbar_equal=True,
                       kwargs_diff={}, axs=saxs, cmap_diff='RdBu_r',
                       cbar_loc='side', tight_layout=False, inverse_diff=True)

#for from_case,i in zip([to_case, from_cases[-1]],range(nr_cols)):
#    sax = axs[:,i]
#    for var, ax in zip(varl, sax.flatten()):
#        plot_map_diff_2case(var, from_case,from_cases[0], maps_dic, relative=(var in varl_rel), 
#                               ax=ax, cmap_diff='RdBu_r')


subp_insert_abc(axs, pos_y=0.1)

#plot_diff(maps_dic, varl, cases[::-1],nr_cols=1, relative=relative)
#load_and_plot_diff_mm(varl,to_case,from_cases, startyear, endyear, avg_over_lev,  pmin=pmin, relative=relative, pressure_adjust=pressure_adjust,nr_cols=1, width=5.5)
fn = filen_base + '_'.join(varl)+f'{relative}.'
plt.tight_layout()
#plt.savefig(fn + 'png')
#plt.savefig(fn + 'pdf')

# %%
varl_rel = ['CLDHGH', 'CLDLOW','CLDMED','CLDTOT','TGCLDIWP','TGCLDLWP']
varl_abs=[]#'NCFT_Ghan']#,'TGCLDCWP']
varl = varl_rel+varl_abs
    
#varl=['ACTNL_incld', 'ACTREL_incld','TGCLDCWP']#,'TGCLDCWP']

                        
width=4.4
asp_rat = 0.48
relative=True

cases = [to_case] + from_cases
maps_dic = get_averaged_fields.get_maps_cases(cases,varl,startyear, endyear,
                                   avg_over_lev=avg_over_lev,
                                   pmin=pmin,
                                   pressure_adjust=pressure_adjust, p_level=p_level)
nr_cols = len(from_cases)
nr_rows = int(np.ceil(len(varl)))
fig, axs = subplots_map(nr_rows, nr_cols, figsize=[width*nr_cols,asp_rat*width*nr_rows])


for i, var in enumerate(varl):
    saxs = axs[i,:] 
    plot_map_diff_only(var, [to_case,*from_cases], maps_dic, relative=(var in varl_rel), cbar_equal=True,
                       kwargs_diff={}, axs=saxs, cmap_diff='RdBu_r',
                       cbar_loc='side', tight_layout=False, inverse_diff=True)

#for from_case,i in zip([to_case, from_cases[-1]],range(nr_cols)):
#    sax = axs[:,i]
#    for var, ax in zip(varl, sax.flatten()):
#        plot_map_diff_2case(var, from_case,from_cases[0], maps_dic, relative=(var in varl_rel), 
#                               ax=ax, cmap_diff='RdBu_r')


subp_insert_abc(axs, pos_y=0.1)

#plot_diff(maps_dic, varl, cases[::-1],nr_cols=1, relative=relative)
#load_and_plot_diff_mm(varl,to_case,from_cases, startyear, endyear, avg_over_lev,  pmin=pmin, relative=relative, pressure_adjust=pressure_adjust,nr_cols=1, width=5.5)
fn = filen_base + '_'.join(varl)+f'{relative}.'
plt.tight_layout()
#plt.savefig(fn + 'png')
#plt.savefig(fn + 'pdf')

# %%

    
varl=['DIR_Ghan']#,'LWDIR_Ghan']#'LWDIR_Ghan']#, 'SO4_NAcondTend']#, 'leaveSecH2SO4','leaveSecSOA']#,'TGCLDCWP']

relative=False
#plot_diff(maps_dic, varl, cases[::-1],nr_cols=1, relative=relative)
load_and_plot_diff_mm(varl,to_case,from_cases, startyear, 
                      endyear, 
                      avg_over_lev,  
                      pmin=pmin, 
                      relative=relative, 
                      pressure_adjust=pressure_adjust,
                      nr_cols=1, 
                      width=4.1,
                     height=2.1)
fn = filen_base + '_'.join(varl)+f'{relative}.'

plt.tight_layout()
plt.savefig(fn + 'png')
plt.savefig(fn + 'pdf', dpi=300)
print(fn)

# %%

    
varl=['CDOD550']#,'LWDIR_Ghan']#'LWDIR_Ghan']#, 'SO4_NAcondTend']#, 'leaveSecH2SO4','leaveSecSOA']#,'TGCLDCWP']

relative=True
#plot_diff(maps_dic, varl, cases[::-1],nr_cols=1, relative=relative)
load_and_plot_diff_mm(varl,to_case,from_cases, startyear, endyear, avg_over_lev,  pmin=pmin, 
                      relative=relative, pressure_adjust=pressure_adjust,nr_cols=1, width=5.5)
fn = filen_base + '_'.join(varl)+f'{relative}.'

plt.tight_layout()
#plt.savefig(fn + 'png')
#plt.savefig(fn + 'pdf')
print(fn)

# %%

    
varl=['NCONC01','NMR01']#,'LWDIR_Ghan']#'LWDIR_Ghan']#, 'SO4_NAcondTend']#, 'leaveSecH2SO4','leaveSecSOA']#,'TGCLDCWP']

relative=True
#plot_diff(maps_dic, varl, cases[::-1],nr_cols=1, relative=relative)
load_and_plot_diff_mm(varl,to_case,from_cases, startyear, endyear, avg_over_lev,  pmin=pmin, 
                      relative=relative, pressure_adjust=pressure_adjust,nr_cols=1, width=5.5)
fn = filen_base + '_'.join(varl)+f'{relative}.'

plt.tight_layout()
#plt.savefig(fn + 'png')
#plt.savefig(fn + 'pdf')
print(fn)

# %%

# %%

    
varl=['SWDIR_Ghan']#'LWDIR_Ghan']#, 'SO4_NAcondTend']#, 'leaveSecH2SO4','leaveSecSOA']#,'TGCLDCWP']

relative=False
#plot_diff(maps_dic, varl, cases[::-1],nr_cols=1, relative=relative)
load_and_plot_diff_mm(varl,to_case,from_cases, startyear, endyear, avg_over_lev,  pmin=pmin, relative=relative, pressure_adjust=pressure_adjust,nr_cols=1, width=5.5)
fn = filen_base + '_'.join(varl)+f'{relative}.'

plt.tight_layout()
#plt.savefig(fn + 'png')
#plt.savefig(fn + 'pdf')

# %%
