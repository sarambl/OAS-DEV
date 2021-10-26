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
from oas_dev.util.plot.plot_maps import plot_map_diff, fix_axis4map_plot, plot_map_abs_abs_diff, plot_map
from useful_scit.imps import (np, xr, plt, pd) 
from oas_dev.util.imports import get_averaged_fields
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
# %%
from oas_dev.util.slice_average.avg_pkg import average_model_var
from oas_dev.data_info import get_nice_name_case

# %% [markdown]
# ## Ideas:
# - Root mean square diffence??
# - Scatter plots of all values, e.g x-- sectional y-- non sectional color by lat/lev? Or lev lat difference. 

# %% [markdown]
# # Map plots number concentration:

# %%
model = 'NorESM'

startyear = '2008-01'
endyear = '2014-12'
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

cases_all = cases_sec + cases_orig + ['noSECTv21_default_dd']


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
so4_spess_fac = dict(SO4_A1=3.06,
                     SO4_A2=3.59, 
                     SO4_AC=3.06, 
                     SO4_NA=3.06,
                     SO4_PR=3.06,
                     SO4_A1_OCW=3.06,
                     SO4_A2_OCW=3.59,
                     SO4_AC_OCW=3.06,
                     SO4_NA_OCW=3.06,
                     SO4_PR_OCW=3.06
                    )

so4_spess = list(so4_spess_fac.keys())

# %%
soa_spess = [
    'SOA_NA',
    'OM_AI', 
    'OM_AC',
    'OM_NI',
    'SOA_NA_OCW', 
    'OM_AI_OCW', 
    'OM_AC_OCW',
    'OM_NI_OCW'
]
soa_spess_fac = {s:1 for s in soa_spess}

# %%
import itertools


# %%
core_vl = soa_spess + so4_spess
var_ext = ["DDF","SFWET"]
varl = [f'cb_{v}' for v in core_vl]
varl = varl + [f'{v}{ext}' for (v,ext) in itertools.product(core_vl, var_ext)]


# %%
varl

# %%
    
#var_ext = [f"{v}DDF",f"{v}SFWET",f"{v}SFSIC",f"{v}SFSBC",f"{v}SFSIS",f"{v}SFSBS"
#           , f"{v}_mixnuc1"]
v='SO4_NA'
#varl=[]
for v in ['SOA_NA','SO4_NA']:#, 'SOA_NA_OCW','SO4_NA_OCW']:
    varl = [f'{v}coagTend',f'{v}clcoagTend',f'{v}condTend']+ varl
    # f"{v}SFSIC",f"{v}SFSBC",f"{v}SFSIS",f"{v}SFSBS", f"{v}_mixnuc1",
"""
    for v in [ 'SOA_NA_OCW','SO4_NA_OCW']:
    varl=varl+ [f'cb_{v}']#'LWDIR_Ghan']#, 'SO4_NAcondTend']#, 'leaveSecH2SO4','leaveSecSOA']#,'TGCLDCWP']
    
    varl = [f"{v}DDF",f"{v}SFWET"]+ varl
   
"""
maps_dic = get_averaged_fields.get_maps_cases(cases_all,varl,startyear, endyear,
                                   avg_over_lev=avg_over_lev,
                                   pmin=pmin,
                                   pressure_adjust=pressure_adjust)#, p_level=p_level)


# %%
def calc_tot_LR(ds,v):
    return (-ds[f'{v}DDF'] + ds[f'{v}SFWET'] + ds[f'{v}coagTend'] + ds[f'{v}clcoagTend'])
def LR_dd_wd(ds,v):
    return (-ds[f'{v}DDF'] + ds[f'{v}SFWET'])# + ds[f'{v}coagTend'] + ds[f'{v}clcoagTend'])


# %%
def comp_lifetime(ds, which, fac_dic ):
    
    lossrate_OCW_DD = 0
    lossrate_OCW_WD = 0
    lossrate_nOCW_DD = 0
    lossrate_nOCW_WD = 0
    cb_OCW = 0
    cb_nOCW = 0
    
    for v in fac_dic.keys():
        f = fac_dic[v]
        if '_OCW' in v:
            cb_OCW = f*ds[f'cb_{v}'] + cb_OCW
            lossrate_OCW_DD = f*(-ds[f'{v}DDF']) + lossrate_OCW_DD
            lossrate_OCW_WD = f*(ds[f'{v}SFWET']) + lossrate_OCW_WD
        else:
            cb_nOCW = f*ds[f'cb_{v}'] + cb_nOCW
            lossrate_nOCW_DD = f*(-ds[f'{v}DDF']) + lossrate_nOCW_DD
            lossrate_nOCW_WD = f*(ds[f'{v}SFWET']) + lossrate_nOCW_WD
    ds[f'cb_{which}'] = cb_nOCW
    ds[f'cb_{which}_OCW'] = cb_OCW
    ds[f'cb_{which}_tot'] = cb_nOCW + cb_OCW
    ds[f'{which}_OCW_DD'] = lossrate_OCW_DD
    ds[f'{which}_OCW_WD'] = lossrate_OCW_WD
    ds[f'{which}_OCW_D'] = lossrate_OCW_WD + lossrate_OCW_DD
    
    ds[f'{which}_DD'] = lossrate_nOCW_DD
    ds[f'{which}_WD'] = lossrate_nOCW_WD
    ds[f'{which}_D'] = lossrate_nOCW_WD + lossrate_nOCW_DD
    ds[f'{which}_tot_WD'] = lossrate_nOCW_WD + lossrate_OCW_WD
    ds[f'{which}_tot_DD'] = lossrate_nOCW_DD + lossrate_OCW_DD
    ds[f'{which}_tot_D'] = lossrate_nOCW_DD + lossrate_OCW_DD + lossrate_nOCW_WD + lossrate_OCW_WD
    
    return ds
    
    
    
    

# %%
for case in cases_all:
    comp_lifetime(maps_dic[case], 'OA', soa_spess_fac )
    comp_lifetime(maps_dic[case], 'SO4', so4_spess_fac )


# %%
def comp_lossr(v, ext, _ds):
    cb = average_model_var(_ds, f'cb_{v}', area='Global', dim=None, minp=850., time_mask=None)
    lr = average_model_var(_ds, f'{v}{ext}', area='Global', dim=None, minp=850., time_mask=None)
    
    out = cb[f'cb_{v}']/lr[f'{v}{ext}']/(60*60*24)
    if out<0:
        out=abs(out)
    out.attrs['units']='days'
    return out



# %%
from oas_dev.data_info import get_nice_name_case

# %%
exts_dic = {
    '_D':'$\tau_{tot}$',
    '_DD':'$\tau_{DDF}$',
    '_WD':'$\tau_{WET}$',
    #'coagTend':'$\tau_{coag}$',
    #'clcoagTend':'$\tau_{clcoag}$'
}

dic_all ={}
for var in ['SO4','SO4_OCW','SO4_tot','OA','OA_OCW','OA_tot',]:
    dic_all[var]={}
    for case in cases_all:
        nncase = get_nice_name_case(case)
        dic_all[var][nncase]={}
        for ext in exts_dic.keys():
            val = comp_lossr(var,ext,maps_dic[case])       
            dic_all[var][nncase][exts_dic[ext]] = val.values


# %%
pd.DataFrame.from_dict(dic_all['SO4'])

# %%
pd.DataFrame.from_dict(dic_all['SO4_tot'])

# %%
pd.DataFrame.from_dict(dic_all['OA_tot'])

# %%
pd.DataFrame.from_dict(dic_all['OA'])

# %%
pd.DataFrame.from_dict(dic_all['OA_OCW'])

# %%
maps_dic[case]

# %%
lss_exts = ['DDF','SFWET','coagTend','clcoagTend']
v = 'SOA_NA'
for v in ['SOA_NA','SO4_NA']:
    for case in cases_all:
        ds = maps_dic[case]
    
        ds[f'{v}_lr_tot'] = -(-ds[f'{v}DDF'] + ds[f'{v}SFWET'] + ds[f'{v}coagTend'] + ds[f'{v}clcoagTend'])
        ds[f'{v}_OCW_lr_tot'] = -(-ds[f'{v}_OCWDDF'] + ds[f'{v}_OCWSFWET'])# + ds[f'{v}coagTend'] + ds[f'{v}clcoagTend'])
        ds[f'{v}_lr_tot_inc'] =ds[f'{v}_OCW_lr_tot'] + ds[f'{v}_OCW_lr_tot']
        ds[f'tau_new_{v}'] = ds[f'cb_{v}']/ds[f'{v}_lr_tot']
        for ex in lss_exts:
            ds[f'tau_{ex}_{v}'] = (ds[f'cb_{v}']/ds[f'{v}{ex}'])/60/60/24
            ds[f'tau_{ex}_{v}'].attrs['units'] = 'days'
        ds[f'tau_prod_{v}'] = ds[f'cb_{v}']/ds[f'{v}condTend']/(60*60*24)
        ds[f'tau_prod_{v}'].attrs['units'] = 'days'
        ds[f'cb_{v}_tot'] = ds[f'cb_{v}']+ ds[f'cb_{v}_OCW']

# %%
from oas_dev.util.slice_average.avg_pkg import average_model_var
from oas_dev.data_info import get_nice_name_case


# %%
def comp_lossr(v, ext, _ds):
    cb = average_model_var(_ds, f'cb_{v}', area='Global', dim=None, minp=850., time_mask=None)
    lr = average_model_var(_ds, f'{v}{ext}', area='Global', dim=None, minp=850., time_mask=None)
    
    out = cb[f'cb_{v}']/lr[f'{v}{ext}']/(60*60*24)
    if out<0:
        out=abs(out)
    out.attrs['units']='days'
    return out



# %% [markdown]
# ## NA-mode lifetime

# %%
exts_dic = {
    '_lr_tot':'$\tau_{tot}$',
    'DDF':'$\tau_{DDF}$',
    'SFWET':'$\tau_{WET}$',
    'coagTend':'$\tau_{coag}$',
    'clcoagTend':'$\tau_{clcoag}$'}

dic_all ={}
for var in ['SOA_NA','SO4_NA']:
    dic_all[var]={}
    for case in cases_all:
        nncase = get_nice_name_case(case)
        dic_all[var][nncase]={}
        for ext in exts_dic.keys():
            val = comp_lossr(var,ext,maps_dic[case])       
            dic_all[var][nncase][exts_dic[ext]] = val.values


# %%
pd.DataFrame.from_dict(dic_all['SOA_NA'])

# %%
pd.DataFrame.from_dict(dic_all['SO4_NA'])

# %%
