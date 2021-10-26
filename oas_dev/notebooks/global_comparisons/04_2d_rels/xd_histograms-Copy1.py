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
import matplotlib

from oas_dev.data_info import get_nice_name_case
from oas_dev.util.imports.import_fields_xr_v2 import import_constants
from oas_dev.util.imports.get_fld_fixed import get_field_fixed
from useful_scit.imps import (np, xr, plt, pd)

# load and autoreload
from IPython import get_ipython

# noinspection PyBroadException
from oas_dev.util.naming_conventions import var_info
from oas_dev.util.naming_conventions.var_info import get_fancy_var_name

try:
    _ipython = get_ipython()
    _magic = _ipython.magic
    _magic('load_ext autoreload')
    _magic('autoreload 2')
except:
    pass
# %%
model = 'NorESM' 

# %%

from oas_dev.constants import get_plotpath
from oas_dev.util.practical_functions import make_folders

plot_path = get_plotpath('comparison') + '/scatter/'
print(plot_path)
make_folders(plot_path)

# %%
from_time = '2008-01'
to_time = '2010-12'
pmin = 850.  # minimum pressure level
avg_over_lev = True  # True#True#False#True
pressure_adjust = True  # Can only be false if avg_over_lev false. Plots particular hybrid sigma lev
#if avg_over_lev:
#    pressure_adjust = True
p_levels = [1013.,900., 800., 700., 600.]  # used if not avg

lev_lim =0.
# %%
cases_sec = ['SECTv21_ctrl_koagD']#, 'PD_SECT_CHC7_diurnal']  # Sect ac eq.20, corr NPF diam, fxdt, vdiam, 1.5xBVOC']
cases_orig = ['noSECTv21_ox_ricc_dd']  # , 'Original eq.18','Original eq.20, 1.5xBVOC','Original eq.20, rednuc']
#cases_orig = ['noSECTv21_default_dd']  # , 'Original eq.18','Original eq.20, 1.5xBVOC','Original eq.20, rednuc']
cases = cases_orig + cases_sec

# %%
var_subl = ['cb_SOA_NA','cb_NA','cb_SO4_NA', 'cb_H2SO4','cb_SOA_LV','SOA_NAcoagTend','SO4_NAcoagTend']#,'SOA_NA','SO4_NA']

# %%
var1 = var_subl[0]
var2 = var_subl[1]
cases_dic ={}
for case in cases:
    dummy = get_field_fixed(case,
                            var_subl,
                            from_time, to_time,
                            pressure_adjust=pressure_adjust)
    print(dummy)
    ds_constants = import_constants(case)
    
    dummy = xr.merge([dummy, ds_constants])
    cases_dic[case] = dummy.copy()

# %%
# select values close to surface:
for case in cases:
    _ds = cases_dic[case]
    _ds = _ds.sel(lev=slice(lev_lim,None))#sel(lev=slice(20,None))
    cases_dic[case] = _ds

# %%
for var in ['H2SO4','SOA_LV']:
    for case in cases:
        _ds = cases_dic[case]
        _ds.load()
        if var in _ds.data_vars:
            if _ds[var].units=='mol/mol':
                _ds[var] = _ds[var]*1e12
                _ds[var].attrs['units']='ppt'

# %%
for case in cases:
    _ds = cases_dic[case]
    _ds['NAcoagTend']=_ds['SOA_NAcoagTend']+_ds['SO4_NAcoagTend']

# %%
for case in cases:
    _ds = cases_dic[case]
    coagn = 'COAGNUCL'
    if coagn not in _ds.data_vars:
        continue
    if _ds[coagn].units=='/s':
        _ds[coagn] = _ds[coagn]*60*60
        _ds[coagn].attrs['units']='hour$^{-1}$'
        print('hey')

        
var = var1
dummy
case_sec = cases_sec[0]
case_orig = cases_orig[0]
ds_diff = (cases_dic[case_sec]- cases_dic[case_orig])#.isel(lev=slice(20,None))
for var in var_subl:
    ds_diff[var+'_'+case_sec] = cases_dic[case_sec][var]#.isel(lev=slice(20,None))
    ds_diff[var+'_'+case_orig] = cases_dic[case_orig][var]#.isel(lev=slice(20,None))
ds_diff.load()
for var in var_subl:
    for case in cases:
        ds_diff[f'log{var}_{case}'] = np.log10(ds_diff[f'{var}_{case}'])#+'_'+ case_orig])

# %%

# %%
import importlib as il
from useful_scit.util import conditional_stats
il.reload(conditional_stats)
def _plt_2dhist(ds_diff, xvar, yvar, nr_bins=40, yscale='symlog', xscale='log',
                xlim = [1e-6,1e-2],ylim=[5.,1e3], ax=None):
    """
    xvar = f'NUCLRATE_{case_orig}'
    yvar='NCONC01'
    xlim = [1e-6,1e-2]
    ylim=[1,1e3]
    nr_bins = 40
    yscale='symlog'
    xscale='log'
    """
    print(xlim)
    print(ylim)
    varList = [xvar, yvar]#f'NUCLRATE_{case_orig}',f'logNUCLRATE_{case_orig}',f'logSOA_LV_{case_orig}',f'logH2SO4_{case_orig}',f'logNCONC01_{case_orig}',f'logN_AER_{case_orig}',f'N_AER_{case_orig}',f'H2SO4_{case_orig}','NCONC01']
    dims = tuple(ds_diff[varList].dims)
    _ds_s = ds_diff[varList].stack(ind=dims)#('lat','lon','lev','time'))


    ybins = mk_bins(ylim[0], vmax = ylim[1], nr_bins=nr_bins, scale=yscale)
    xbins = mk_bins(xlim[0],vmax=xlim[1], nr_bins=nr_bins, scale=xscale)
    data=_ds_s.to_dataframe()
    lim=0
    #data = -data[(data['NCONC01']<lim)]# | (data['NCONC01']>=lim)]
    x=data[xvar]#f'NUCLRATE_{case_orig}']
    y=data[yvar]#'NCONC01']
    if ax is None:
        fig, ax = plt.subplots(1)
    h =ax.hist2d(x,y,bins=[xbins,ybins], cmap='Reds')#,extent=[-3,3,-300,20],yscale='symlog')
    plt.colorbar(h[3], ax=ax, format = OOMFormatter(4, mathText=False))
    #cb = fig.colorbar(c, ax=ax)
    if yscale =='symlog':
        ax.set_yscale('symlog', linthreshy=ylim[0], linscaley=ylim[0]/10,subsy=[2,3,4,5,6,7,8,9])
        yt = ax.get_yticks()
        ml = np.abs(yt[yt!=0]).min()
        ytl = yt
        ytl[(yt==ml)|(yt==-ml)]=None
        ax.set_yticks(ticks=yt)#[y for y in yt if y!=0])#,
        ax.set_yticklabels(ytl)#[-1e2,-1e1,-1e0,1e0,1e1,1e2])


    elif yscale =='log':
        print('set log scale')
        ax.set_yscale('log')#, linthreshy=ylim[0], linscaley=ylim[0]/10,subsy=[2,3,4,5,6,7,8,9])
        
    #ax.set_yticks([y for y in yt if y!=0])#[-1e2,-1e1,-1e0,1e0,1e1,1e2])
    ax.set_xscale('log')
    return ax
    #plt.show()

def mk_bins(v, vmax = 1e3, nr_bins=20, scale='symlog'):
    if scale=='symlog':
        ybins = np.geomspace(v, vmax, int(nr_bins)/2)
        ybins = ybins - ybins[0]
        ybins = [*-ybins[::-1], *ybins[1:]]
    elif scale=='log':
        ybins = np.geomspace(v, vmax, nr_bins)
    elif scale=='neglog':
        ybins = -np.geomspace(v, vmax, nr_bins)[::-1]

    return ybins


class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
            self.format = r'$\mathdefault{%s}$' % self.format


# %%
from useful_scit.plot.fig_manip import subp_insert_abc


# %%
def _plt_tmp(_ds,axs,var_xl, var_diff, xlims,ylim=[5.,1e3], yscale='symlog', case_base = case_orig):
    for var,ax in zip(var_xl, axs.flatten()):
        print(var)
        xlim = xlims[var]
        h = _plt_2dhist(_ds,f'{var}_{case_base}', var_diff,
                nr_bins=40,
                xlim=xlim, ax=ax,
                        ylim=ylim,
                       yscale=yscale)
        uni = var_info.get_fancy_unit_xr(_ds[var],
                                     var)
        ax.set_xlabel(f'{get_fancy_var_name(var)} [{uni}]')
        ax.plot(xlim,[0,0], linewidth=.5, c='k')

    uni = var_info.get_fancy_unit_xr(_ds[var_diff],
                           var_diff)
    fvar_diff = get_fancy_var_name(var_diff)
    ylab = f'$\Delta${fvar_diff} [{uni}]'
    for ax in axs[:,0]:
        ax.set_ylabel(ylab)
    
    subp_insert_abc(axs)

    suptit =f'{get_nice_name_case(case_sec)}-{get_nice_name_case(case_base)} vs.  '
    suptit =f'{fvar_diff}$(m_1)-${fvar_diff}$(m_2)$ vs. $X(m_2)$ \n for $m_1=${get_nice_name_case(case_sec)}, $m_2=${get_nice_name_case(case_base)}'
    fig.subplots_adjust(hspace=.5, wspace=0.1)#,top=0.8, )
    stit = fig.suptitle(suptit,  fontsize=12, y=.98)
    
    return stit


# %%
def _plt_tmp_mv1v2(_ds,axs,var_xl, xlims, yscale='log', case_base = case_orig, case_oth = case_sec):
    for var,ax in zip(var_xl, axs.flatten()):
        print(var)
        xlim = xlims[var]
        h = _plt_2dhist(_ds,f'{var}_{case_base}', f'{var}_{case_oth}',
                        nr_bins=40,
                        xlim=xlim, 
                        ax=ax,
                        ylim=xlim,
                       yscale=yscale)
        uni = var_info.get_fancy_unit_xr(_ds[var],
                                     var)
        ax.set_xlabel(f'{get_fancy_var_name(var)},{get_nice_name_case(case_base)} [{uni}]')
        ax.set_ylabel(f'{get_fancy_var_name(var)},{get_nice_name_case(case_oth)} [{uni}]')
        #ax.plot(xlim,[0,0], linewidth=.5, c='k')
        ax.set_ylim(xlim)
        ax.set_ylim(xlim)
    
    subp_insert_abc(axs)

    suptit =f'{get_nice_name_case(case_sec)}-{get_nice_name_case(case_base)} vs.  '
    suptit =f'{fvar_diff}$(m_1)-${fvar_diff}$(m_2)$ vs. $X(m_2)$ \n for $m_1=${get_nice_name_case(case_sec)}, $m_2=${get_nice_name_case(case_base)}'
    fig.subplots_adjust(hspace=.5, wspace=0.1)#,top=0.8, )
    stit = fig.suptitle(suptit,  fontsize=12, y=.98)
    
    return stit


# %%
#case_base=case_orig

fig, axs = plt.subplots(3,2, figsize=[6.4,7.1], sharey=False)#, constrained_layout=True)
var_xl = ['NUCLRATE','GR','H2SO4','SOA_LV','NCONC01','COAGNUCL']
lev_min=100.
_ds = ds_diff.sel(lev=slice(lev_min, None))
var_diff = 'NCONC01_'+case_base
xlims = {
    'NUCLRATE' : [1.e-6,10],
    'H2SO4' : [1.e-3,1],
    'SOA_LV' : [1.e-5,1],
    'GR' : [1.e-3,1],
    'N_AER':[1e0,1e4],
    'NCONC01':[10,5e3],
    'COAGNUCL':[1e-4,1],
}

stit = _plt_tmp_mv1v2(_ds,axs,var_xl, xlims)#,ylim=[5.,1e3], yscale='log')
                      #_plt_tmp(_ds,axs,var_xl, var_diff, xlims, ylim=[10,5e3], yscale='log', case_base=case_base)
uni = var_info.get_fancy_unit_xr(_ds[var_diff], var_diff)
fvar_diff = get_fancy_var_name('NCONC01')

    
suptit =f'{fvar_diff}$(m)$ vs. $X(m)$ \n for $m=${get_nice_name_case(case_base)}'
fig.subplots_adjust(hspace=.5, wspace=0.1)#,top=0.8, )
stit = fig.suptitle(suptit,  fontsize=12, y=.98)
fn = plot_path + f'2dhist_abs{case_base}.'
#fig.savefig(fn+'pdf',bbox_extra_artists=(stit,), bbox_inches='tight')
#fig.savefig(fn+'png',bbox_extra_artists=(stit,), bbox_inches='tight')
fig.tight_layout()
plt.show()
print(fn)

# %% [markdown]
# # ABSOLUTE RELATIONSHIPS:

# %% [markdown]
# ### All below 100 hPa

# %%
case_sec

# %% [markdown]
# ### Case: OsloAero_imp

# %%
case_base=case_orig

fig, axs = plt.subplots(3,2, figsize=[6.4,7.1], sharey=True)#, constrained_layout=True)
var_xl = ['NUCLRATE','GR','H2SO4','SOA_LV','NCONC01','COAGNUCL']
lev_min=100.
_ds = ds_diff.sel(lev=slice(lev_min, None))
var_diff = 'NCONC01_'+case_base
xlims = {
    'NUCLRATE' : [1.e-6,10],
    'H2SO4' : [1.e-3,1],
    'SOA_LV' : [1.e-5,1],
    'GR' : [1.e-3,1],
    'N_AER':[1e0,1e4],
    'NCONC01':[10,5e3],
    'COAGNUCL':[1e-4,1],
}

stit = _plt_tmp(_ds,axs,var_xl, var_diff, xlims, ylim=[10,5e3], yscale='log', case_base=case_base)
uni = var_info.get_fancy_unit_xr(_ds[var_diff], var_diff)
fvar_diff = get_fancy_var_name('NCONC01')
for ax in axs[:,0]:
    print('hey')
    ax.set_ylabel(f'{fvar_diff} [{uni}]')

    
suptit =f'{fvar_diff}$(m)$ vs. $X(m)$ \n for $m=${get_nice_name_case(case_base)}'
fig.subplots_adjust(hspace=.5, wspace=0.1)#,top=0.8, )
stit = fig.suptitle(suptit,  fontsize=12, y=.98)

fn = plot_path + f'2dhist_abs{case_base}.'
fig.savefig(fn+'pdf',bbox_extra_artists=(stit,), bbox_inches='tight')
fig.savefig(fn+'png',bbox_extra_artists=(stit,), bbox_inches='tight')
plt.show()
print(fn)

# %% [markdown]
# ### Case: OsloAeroSec

# %%
case_base=case_sec

fig, axs = plt.subplots(3,2, figsize=[6.4,7.1], sharey=True)#, constrained_layout=True)
var_xl = ['NUCLRATE','GR','H2SO4','SOA_LV','NCONC01','COAGNUCL']
lev_min=100.
_ds = ds_diff.sel(lev=slice(lev_min, None))
var_diff = 'NCONC01_'+case_base
xlims = {
    'NUCLRATE' : [1.e-6,10],
    'H2SO4' : [1.e-3,1],
    'SOA_LV' : [1.e-5,1],
    'GR' : [1.e-3,1],
    'N_AER':[1e0,1e4],
    'NCONC01':[10,5e3],
    'COAGNUCL':[1e-4,1],
}

stit = _plt_tmp(_ds,axs,var_xl, var_diff, xlims, ylim=[10,5e3], yscale='log', case_base=case_base)
uni = var_info.get_fancy_unit_xr(_ds[var_diff], var_diff)
fvar_diff = get_fancy_var_name('NCONC01')
for ax in axs[:,0]:
    print('hey')
    ax.set_ylabel(f'{fvar_diff} [{uni}]')

    
suptit =f'{fvar_diff}$(m)$ vs. $X(m)$ \n for $m=${get_nice_name_case(case_base)}'
fig.subplots_adjust(hspace=.5, wspace=0.1)#,top=0.8, )
stit = fig.suptitle(suptit,  fontsize=12, y=.98)

fn = plot_path + f'2dhist_abs{case_base}.'
#fig.savefig(fn+'pdf',bbox_extra_artists=(stit,), bbox_inches='tight')
#fig.savefig(fn+'png',bbox_extra_artists=(stit,), bbox_inches='tight')
plt.show()
print(fn)

# %% [markdown]
# ## Surface

# %%
fig, axs = plt.subplots(3,2, figsize=[6.4,7.1], sharey=True)#, constrained_layout=True)
var_xl = ['NUCLRATE','GR','H2SO4','SOA_LV','NCONC01','COAGNUCL']
lev_min=100.
#_ds = ds_diff.sel(lev=slice(lev_min, None))
_ds = ds_diff.isel(lev=-1)#lev=slice(lev_min, None))

var_diff = 'NCONC01_'+case_orig
xlims = {
    'NUCLRATE' : [1.e-6,10],
    'H2SO4' : [1.e-3,1],
    'SOA_LV' : [1.e-5,1],
    'GR' : [1.e-3,1],
    'N_AER':[1e0,1e4],
    'NCONC01':[10,5e3],
    'COAGNUCL':[1e-4,1],
}

stit = _plt_tmp(_ds,axs,var_xl, var_diff, xlims, ylim=[10,5e3], yscale='log')
uni = var_info.get_fancy_unit_xr(_ds[var_diff], var_diff)
fvar_diff = get_fancy_var_name('NCONC01')
for ax in axs[:,0]:
    print('hey')
    ax.set_ylabel(f'{fvar_diff} [{uni}]')

    
suptit =f'{fvar_diff}$(m)$ vs. $X(m)$ \n for $m=${get_nice_name_case(case_orig)}'
fig.subplots_adjust(hspace=.5, wspace=0.1)#,top=0.8, )
stit = fig.suptitle(suptit,  fontsize=12, y=.98)

fn = plot_path + f'2dhist_abs{case_orig}.'
#fig.savefig(fn+'pdf',bbox_extra_artists=(stit,), bbox_inches='tight')
#fig.savefig(fn+'png',bbox_extra_artists=(stit,), bbox_inches='tight')
plt.show()
print(fn)

# %% [markdown]
# ### Case: OsloAeroSec

# %%
case_base=case_sec

fig, axs = plt.subplots(3,2, figsize=[6.4,7.1], sharey=True)#, constrained_layout=True)
var_xl = ['NUCLRATE','GR','H2SO4','SOA_LV','NCONC01','COAGNUCL']
lev_min=850.
_ds = ds_diff.isel(lev=-1)#slice(lev_min, None))
var_diff = 'NCONC01_'+case_base
xlims = {
    'NUCLRATE' : [1.e-6,10],
    'H2SO4' : [1.e-3,1],
    'SOA_LV' : [1.e-5,1],
    'GR' : [1.e-3,1],
    'N_AER':[1e0,1e4],
    'NCONC01':[10,5e3],
    'COAGNUCL':[1e-4,1],
}

stit = _plt_tmp(_ds,axs,var_xl, var_diff, xlims, ylim=[10,5e3], yscale='log', case_base=case_base)
uni = var_info.get_fancy_unit_xr(_ds[var_diff], var_diff)
fvar_diff = get_fancy_var_name('NCONC01')
for ax in axs[:,0]:
    print('hey')
    ax.set_ylabel(f'{fvar_diff} [{uni}]')

    
suptit =f'{fvar_diff}$(m)$ vs. $X(m)$ \n for $m=${get_nice_name_case(case_base)}'
fig.subplots_adjust(hspace=.5, wspace=0.1)#,top=0.8, )
stit = fig.suptitle(suptit,  fontsize=12, y=.98)

fn = plot_path + f'2dhist_abs{case_base}.'
#fig.savefig(fn+'pdf',bbox_extra_artists=(stit,), bbox_inches='tight')
#fig.savefig(fn+'png',bbox_extra_artists=(stit,), bbox_inches='tight')
plt.show()
print(fn)

# %% [markdown]
# ### below 850.

# %% [markdown]
# ### Case: OsloAero_imp

# %%
case_base=case_orig

fig, axs = plt.subplots(3,2, figsize=[6.4,7.1], sharey=True)#, constrained_layout=True)
var_xl = ['NUCLRATE','GR','H2SO4','SOA_LV','NCONC01','COAGNUCL']
lev_min=850.
_ds = ds_diff.sel(lev=slice(lev_min, None))
var_diff = 'NCONC01_'+case_base
xlims = {
    'NUCLRATE' : [1.e-6,10],
    'H2SO4' : [1.e-3,1],
    'SOA_LV' : [1.e-5,1],
    'GR' : [1.e-3,1],
    'N_AER':[1e0,1e4],
    'NCONC01':[10,5e3],
    'COAGNUCL':[1e-4,1],
}

stit = _plt_tmp(_ds,axs,var_xl, var_diff, xlims, ylim=[10,5e3], yscale='log', case_base=case_base)
uni = var_info.get_fancy_unit_xr(_ds[var_diff], var_diff)
fvar_diff = get_fancy_var_name('NCONC01')
for ax in axs[:,0]:
    print('hey')
    ax.set_ylabel(f'{fvar_diff} [{uni}]')

    
suptit =f'{fvar_diff}$(m)$ vs. $X(m)$ \n for $m=${get_nice_name_case(case_base)}'
fig.subplots_adjust(hspace=.5, wspace=0.1)#,top=0.8, )
stit = fig.suptitle(suptit,  fontsize=12, y=.98)

fn = plot_path + f'2dhist_abs{case_base}.'
#fig.savefig(fn+'pdf',bbox_extra_artists=(stit,), bbox_inches='tight')
#fig.savefig(fn+'png',bbox_extra_artists=(stit,), bbox_inches='tight')
plt.show()
print(fn)

# %% [markdown]
# ### Case: OsloAeroSec

# %%
case_base=case_sec

fig, axs = plt.subplots(3,2, figsize=[6.4,7.1], sharey=True)#, constrained_layout=True)
var_xl = ['NUCLRATE','GR','H2SO4','SOA_LV','NCONC01','COAGNUCL']
lev_min=850.
_ds = ds_diff.sel(lev=slice(lev_min, None))
var_diff = 'NCONC01_'+case_base
xlims = {
    'NUCLRATE' : [1.e-6,10],
    'H2SO4' : [1.e-3,1],
    'SOA_LV' : [1.e-5,1],
    'GR' : [1.e-3,1],
    'N_AER':[1e0,1e4],
    'NCONC01':[10,5e3],
    'COAGNUCL':[1e-4,1],
}

stit = _plt_tmp(_ds,axs,var_xl, var_diff, xlims, ylim=[10,5e3], yscale='log', case_base=case_base)
uni = var_info.get_fancy_unit_xr(_ds[var_diff], var_diff)
fvar_diff = get_fancy_var_name('NCONC01')
for ax in axs[:,0]:
    print('hey')
    ax.set_ylabel(f'{fvar_diff} [{uni}]')

    
suptit =f'{fvar_diff}$(m)$ vs. $X(m)$ \n for $m=${get_nice_name_case(case_base)}'
fig.subplots_adjust(hspace=.5, wspace=0.1)#,top=0.8, )
stit = fig.suptitle(suptit,  fontsize=12, y=.98)

fn = plot_path + f'2dhist_abs{case_base}.'
#fig.savefig(fn+'pdf',bbox_extra_artists=(stit,), bbox_inches='tight')
#fig.savefig(fn+'png',bbox_extra_artists=(stit,), bbox_inches='tight')
plt.show()
print(fn)

# %% [markdown]
# ### above 850.

# %% [markdown]
# ### Case: OsloAero_imp

# %%
case_base=case_orig

fig, axs = plt.subplots(3,2, figsize=[6.4,7.1], sharey=True)#, constrained_layout=True)
var_xl = ['NUCLRATE','GR','H2SO4','SOA_LV','NCONC01','COAGNUCL']
lev_min=850.
_ds = ds_diff.sel(lev=slice(None,lev_min))
var_diff = 'NCONC01_'+case_base
xlims = {
    'NUCLRATE' : [1.e-6,10],
    'H2SO4' : [1.e-3,1],
    'SOA_LV' : [1.e-5,1],
    'GR' : [1.e-3,1],
    'N_AER':[1e0,1e4],
    'NCONC01':[10,5e3],
    'COAGNUCL':[1e-4,1],
}

stit = _plt_tmp(_ds,axs,var_xl, var_diff, xlims, ylim=[10,5e3], yscale='log', case_base=case_base)
uni = var_info.get_fancy_unit_xr(_ds[var_diff], var_diff)
fvar_diff = get_fancy_var_name('NCONC01')
for ax in axs[:,0]:
    print('hey')
    ax.set_ylabel(f'{fvar_diff} [{uni}]')

    
suptit =f'{fvar_diff}$(m)$ vs. $X(m)$ \n for $m=${get_nice_name_case(case_base)}'
fig.subplots_adjust(hspace=.5, wspace=0.1)#,top=0.8, )
stit = fig.suptitle(suptit,  fontsize=12, y=.98)

fn = plot_path + f'2dhist_abs{case_base}.'
#fig.savefig(fn+'pdf',bbox_extra_artists=(stit,), bbox_inches='tight')
#fig.savefig(fn+'png',bbox_extra_artists=(stit,), bbox_inches='tight')
plt.show()
print(fn)

# %% [markdown]
# ### Case: OsloAeroSec

# %%
case_base=case_sec

fig, axs = plt.subplots(3,2, figsize=[6.4,7.1], sharey=True)#, constrained_layout=True)
var_xl = ['NUCLRATE','GR','H2SO4','SOA_LV','NCONC01','COAGNUCL']
lev_min=850.
#_ds = ds_diff.sel(lev=slice(lev_min, None))
_ds = ds_diff.sel(lev=slice(None,lev_min))

var_diff = 'NCONC01_'+case_base
xlims = {
    'NUCLRATE' : [1.e-6,10],
    'H2SO4' : [1.e-3,1],
    'SOA_LV' : [1.e-5,1],
    'GR' : [1.e-3,1],
    'N_AER':[1e0,1e4],
    'NCONC01':[10,5e3],
    'COAGNUCL':[1e-4,1],
}

stit = _plt_tmp(_ds,axs,var_xl, var_diff, xlims, ylim=[10,5e3], yscale='log', case_base=case_base)
uni = var_info.get_fancy_unit_xr(_ds[var_diff], var_diff)
fvar_diff = get_fancy_var_name('NCONC01')
for ax in axs[:,0]:
    print('hey')
    ax.set_ylabel(f'{fvar_diff} [{uni}]')

    
suptit =f'{fvar_diff}$(m)$ vs. $X(m)$ \n for $m=${get_nice_name_case(case_base)}'
fig.subplots_adjust(hspace=.5, wspace=0.1)#,top=0.8, )
stit = fig.suptitle(suptit,  fontsize=12, y=.98)

fn = plot_path + f'2dhist_abs{case_base}.'
#fig.savefig(fn+'pdf',bbox_extra_artists=(stit,), bbox_inches='tight')
#fig.savefig(fn+'png',bbox_extra_artists=(stit,), bbox_inches='tight')
plt.show()
print(fn)

# %% [markdown]
# ### Mask by COAGNUCL

# %%
gr = ds_diff['GR_' + case_orig]
coag = ds_diff['COAGNUCL_' + case_orig]
ma = (coag) > coag.quantile(.75)

# %% [markdown]
# ### Case: OsloAero_imp

# %%
case_base=case_orig

fig, axs = plt.subplots(3,2, figsize=[6.4,7.1], sharey=True)#, constrained_layout=True)
var_xl = ['NUCLRATE','GR','H2SO4','SOA_LV','NCONC01','COAGNUCL']
lev_min=850.
_ds = ds_diff.where(ma)#lev=slice(None,lev_min))
var_diff = 'NCONC01_'+case_base
xlims = {
    'NUCLRATE' : [1.e-6,10],
    'H2SO4' : [1.e-3,1],
    'SOA_LV' : [1.e-5,1],
    'GR' : [1.e-3,1],
    'N_AER':[1e0,1e4],
    'NCONC01':[10,5e3],
    'COAGNUCL':[1e-4,1],
}

stit = _plt_tmp(_ds,axs,var_xl, var_diff, xlims, ylim=[10,5e3], yscale='log', case_base=case_base)
uni = var_info.get_fancy_unit_xr(_ds[var_diff], var_diff)
fvar_diff = get_fancy_var_name('NCONC01')
for ax in axs[:,0]:
    print('hey')
    ax.set_ylabel(f'{fvar_diff} [{uni}]')

    
suptit =f'{fvar_diff}$(m)$ vs. $X(m)$ \n for $m=${get_nice_name_case(case_base)}'
fig.subplots_adjust(hspace=.5, wspace=0.1)#,top=0.8, )
stit = fig.suptitle(suptit,  fontsize=12, y=.98)

fn = plot_path + f'2dhist_abs{case_base}.'
#fig.savefig(fn+'pdf',bbox_extra_artists=(stit,), bbox_inches='tight')
#fig.savefig(fn+'png',bbox_extra_artists=(stit,), bbox_inches='tight')
plt.show()
print(fn)

# %% [markdown]
# ### Case: OsloAeroSec

# %%
case_base=case_sec

fig, axs = plt.subplots(3,2, figsize=[6.4,7.1], sharey=True)#, constrained_layout=True)
var_xl = ['NUCLRATE','GR','H2SO4','SOA_LV','NCONC01','COAGNUCL']
lev_min=850.
#_ds = ds_diff.sel(lev=slice(lev_min, None))
_ds = ds_diff.where(ma)#lev=slice(None,lev_min))

var_diff = 'NCONC01_'+case_base
xlims = {
    'NUCLRATE' : [1.e-6,10],
    'H2SO4' : [1.e-3,1],
    'SOA_LV' : [1.e-5,1],
    'GR' : [1.e-3,1],
    'N_AER':[1e0,1e4],
    'NCONC01':[10,5e3],
    'COAGNUCL':[1e-4,1],
}

stit = _plt_tmp(_ds,axs,var_xl, var_diff, xlims, ylim=[10,5e3], yscale='log', case_base=case_base)
uni = var_info.get_fancy_unit_xr(_ds[var_diff], var_diff)
fvar_diff = get_fancy_var_name('NCONC01')
for ax in axs[:,0]:
    print('hey')
    ax.set_ylabel(f'{fvar_diff} [{uni}]')

    
suptit =f'{fvar_diff}$(m)$ vs. $X(m)$ \n for $m=${get_nice_name_case(case_base)}'
fig.subplots_adjust(hspace=.5, wspace=0.1)#,top=0.8, )
stit = fig.suptitle(suptit,  fontsize=12, y=.98)

fn = plot_path + f'2dhist_abs{case_base}.'
#fig.savefig(fn+'pdf',bbox_extra_artists=(stit,), bbox_inches='tight')
#fig.savefig(fn+'png',bbox_extra_artists=(stit,), bbox_inches='tight')
plt.show()
print(fn)

# %% [markdown]
# ### lowest 50 %

# %%
gr = ds_diff['GR_' + case_orig]
coag = ds_diff['COAGNUCL_' + case_orig]
ma = (coag) < coag.quantile(.75)

# %% [markdown]
# ### Case: OsloAero_imp

# %%
case_base=case_orig

fig, axs = plt.subplots(3,2, figsize=[6.4,7.1], sharey=True)#, constrained_layout=True)
var_xl = ['NUCLRATE','GR','H2SO4','SOA_LV','NCONC01','COAGNUCL']
lev_min=850.
_ds = ds_diff.where(ma)#lev=slice(None,lev_min))
var_diff = 'NCONC01_'+case_base
xlims = {
    'NUCLRATE' : [1.e-6,10],
    'H2SO4' : [1.e-3,1],
    'SOA_LV' : [1.e-5,1],
    'GR' : [1.e-3,1],
    'N_AER':[1e0,1e4],
    'NCONC01':[10,5e3],
    'COAGNUCL':[1e-4,1],
}

stit = _plt_tmp(_ds,axs,var_xl, var_diff, xlims, ylim=[10,5e3], yscale='log', case_base=case_base)
uni = var_info.get_fancy_unit_xr(_ds[var_diff], var_diff)
fvar_diff = get_fancy_var_name('NCONC01')
for ax in axs[:,0]:
    print('hey')
    ax.set_ylabel(f'{fvar_diff} [{uni}]')

    
suptit =f'{fvar_diff}$(m)$ vs. $X(m)$ \n for $m=${get_nice_name_case(case_base)}'
fig.subplots_adjust(hspace=.5, wspace=0.1)#,top=0.8, )
stit = fig.suptitle(suptit,  fontsize=12, y=.98)

fn = plot_path + f'2dhist_abs{case_base}.'
#fig.savefig(fn+'pdf',bbox_extra_artists=(stit,), bbox_inches='tight')
#fig.savefig(fn+'png',bbox_extra_artists=(stit,), bbox_inches='tight')
plt.show()
print(fn)

# %% [markdown]
# ### Case: OsloAeroSec

# %%
case_base=case_sec

fig, axs = plt.subplots(3,2, figsize=[6.4,7.1], sharey=True)#, constrained_layout=True)
var_xl = ['NUCLRATE','GR','H2SO4','SOA_LV','NCONC01','COAGNUCL']
lev_min=850.
#_ds = ds_diff.sel(lev=slice(lev_min, None))
_ds = ds_diff.where(ma)#lev=slice(None,lev_min))

var_diff = 'NCONC01_'+case_base
xlims = {
    'NUCLRATE' : [1.e-6,10],
    'H2SO4' : [1.e-3,1],
    'SOA_LV' : [1.e-5,1],
    'GR' : [1.e-3,1],
    'N_AER':[1e0,1e4],
    'NCONC01':[10,5e3],
    'COAGNUCL':[1e-4,1],
}

stit = _plt_tmp(_ds,axs,var_xl, var_diff, xlims, ylim=[10,5e3], yscale='log', case_base=case_base)
uni = var_info.get_fancy_unit_xr(_ds[var_diff], var_diff)
fvar_diff = get_fancy_var_name('NCONC01')
for ax in axs[:,0]:
    print('hey')
    ax.set_ylabel(f'{fvar_diff} [{uni}]')

    
suptit =f'{fvar_diff}$(m)$ vs. $X(m)$ \n for $m=${get_nice_name_case(case_base)}'
fig.subplots_adjust(hspace=.5, wspace=0.1)#,top=0.8, )
stit = fig.suptitle(suptit,  fontsize=12, y=.98)

fn = plot_path + f'2dhist_abs{case_base}.'
#fig.savefig(fn+'pdf',bbox_extra_artists=(stit,), bbox_inches='tight')
#fig.savefig(fn+'png',bbox_extra_artists=(stit,), bbox_inches='tight')
plt.show()
print(fn)

# %% [markdown]
# # Difference:
#

# %% [markdown]
# ## Main plot: below 100 hPa:

# %%
fig, axs = plt.subplots(3,2, figsize=[6.4,7.1], sharey=True)#, constrained_layout=True)
var_xl = ['NUCLRATE','GR','H2SO4','SOA_LV','NCONC01','COAGNUCL']
lev_min=100.
_ds = ds_diff.sel(lev=slice(lev_min, None))
var_diff = 'NCONC01'
xlims = {
    'NUCLRATE' : [1.e-6,10],
    'H2SO4' : [1.e-3,1],
    'SOA_LV' : [1.e-5,1],
    'GR' : [1.e-3,1],
    'N_AER':[1e0,1e4],
    'NCONC01':[5e-1,1e3],
    'COAGNUCL':[1e-5,.1],

}

stit = _plt_tmp(_ds,axs,var_xl, var_diff, xlims)
fn = plot_path + f'2dhist_{case_orig}_{case_sec}.'
fig.savefig(fn+'pdf',bbox_extra_artists=(stit,), bbox_inches='tight')
fig.savefig(fn+'png',bbox_extra_artists=(stit,), bbox_inches='tight')
plt.show()
print(fn)

# %% [markdown]
# ## Surface layer

# %%
fig, axs = plt.subplots(3,2, figsize=[6.4,7.1], sharey=True)#, constrained_layout=True)
var_xl = ['NUCLRATE','GR','H2SO4','SOA_LV','NCONC01','COAGNUCL']
lev_min=100.
_ds = ds_diff.isel(lev=-1)#slice(lev_min, None))
var_diff = 'NCONC01'
xlims = {
    'NUCLRATE' : [1.e-6,10],
    'H2SO4' : [1.e-3,1],
    'SOA_LV' : [1.e-5,1],
    'GR' : [1.e-3,1],
    'N_AER':[1e0,1e4],
    'NCONC01':[5e-1,1e3],
    'COAGNUCL':[1e-5,.1],

}

_plt_tmp(_ds,axs,var_xl, var_diff, xlims)
fn = plot_path + f'2dhist_{case_orig}_{case_sec}.'
#fig.savefig(fn+'pdf',bbox_extra_artists=(stit,), bbox_inches='tight')
#fig.savefig(fn+'png',bbox_extra_artists=(stit,), bbox_inches='tight')
plt.show()
print(fn)

# %% [markdown]
# ## Up to 850 hPa

# %%
fig, axs = plt.subplots(3,2, figsize=[6.4,7.1], sharey=True)#, constrained_layout=True)
var_xl = ['NUCLRATE','GR','H2SO4','SOA_LV','NCONC01','COAGNUCL']
lev_min=850.
_ds = ds_diff.sel(lev=slice(lev_min, None))
var_diff = 'NCONC01'
xlims = {
    'NUCLRATE' : [1.e-6,10],
    'H2SO4' : [1.e-3,1],
    'SOA_LV' : [1.e-5,1],
    'GR' : [1.e-3,1],
    'N_AER':[1e0,1e4],
    'NCONC01':[5e-1,1e3],
    'COAGNUCL':[1e-5,.1],
}

_plt_tmp(_ds,axs,var_xl, var_diff, xlims)
fn = plot_path + f'2dhist_{case_orig}_{case_sec}.'
#fig.savefig(fn+'pdf',bbox_extra_artists=(stit,), bbox_inches='tight')
#fig.savefig(fn+'png',bbox_extra_artists=(stit,), bbox_inches='tight')
plt.show()
print(fn)

# %% [markdown]
# ## Above 850

# %%
fig, axs = plt.subplots(3,2, figsize=[6.4,7.1], sharey=True)#, constrained_layout=True)
var_xl = ['NUCLRATE','GR','H2SO4','SOA_LV','NCONC01','COAGNUCL']
lev_min=850.
_ds = ds_diff.sel(lev=slice(None,lev_min))#, None))
var_diff = 'NCONC01'
xlims = {
    'NUCLRATE' : [1.e-6,10],
    'H2SO4' : [1.e-3,1],
    'SOA_LV' : [1.e-5,1],
    'GR' : [1.e-3,1],
    'N_AER':[1e0,1e4],
    'NCONC01':[5e-1,1e3],
    'COAGNUCL':[1e-5,.1],
    
}

_plt_tmp(_ds,axs,var_xl, var_diff, xlims)
fn = plot_path + f'2dhist_{case_orig}_{case_sec}.'
#fig.savefig(fn+'pdf',bbox_extra_artists=(stit,), bbox_inches='tight')
#fig.savefig(fn+'png',bbox_extra_artists=(stit,), bbox_inches='tight')
plt.show()
print(fn)

# %% [markdown]
# ## Mask by N_AER?

# %%
v= 'logN_AER_noSECTv21_ox_ricc_dd'
mask_ = ds_diff[v] <ds_diff[v].quantile(.5)

# %%
fig, axs = plt.subplots(3,2, figsize=[6.4,7.1], sharey=True)#, constrained_layout=True)
var_xl = ['NUCLRATE','GR','H2SO4','SOA_LV','NCONC01','COAGNUCL']
lev_min=850.
_ds = ds_diff.where(mask_)#sel(lev=slice(lev_min, None))
var_diff = 'NCONC01'
xlims = {
    'NUCLRATE' : [1.e-6,10],
    'H2SO4' : [1.e-3,1],
    'SOA_LV' : [1.e-5,1],
    'GR' : [1.e-3,1],
    'N_AER':[1e0,1e4],
    'NCONC01':[5e-1,1e3],
    'COAGNUCL':[1e-8,1e-3],
}

_plt_tmp(_ds,axs,var_xl, var_diff, xlims)
fn = plot_path + f'2dhist_{case_orig}_{case_sec}.'
#fig.savefig(fn+'pdf',bbox_extra_artists=(stit,), bbox_inches='tight')
#fig.savefig(fn+'png',bbox_extra_artists=(stit,), bbox_inches='tight')
plt.show()
print(fn)

# %% [markdown]
# ## Oposite

# %%
fig, axs = plt.subplots(3,2, figsize=[6.4,7.1], sharey=True)#, constrained_layout=True)
var_xl = ['NUCLRATE','GR','H2SO4','SOA_LV','NCONC01','COAGNUCL']
lev_min=850.
_ds = ds_diff.where(~mask_)#sel(lev=slice(lev_min, None))
var_diff = 'NCONC01'
xlims = {
    'NUCLRATE' : [1.e-6,10],
    'H2SO4' : [1.e-3,1],
    'SOA_LV' : [1.e-5,1],
    'GR' : [1.e-3,1],
    'N_AER':[1e0,1e4],
    'NCONC01':[5e-1,1e3],
    'COAGNUCL':[1e-8,1e-3],
}

_plt_tmp(_ds,axs,var_xl, var_diff, xlims)
fn = plot_path + f'2dhist_{case_orig}_{case_sec}.'
#fig.savefig(fn+'pdf',bbox_extra_artists=(stit,), bbox_inches='tight')
#fig.savefig(fn+'png',bbox_extra_artists=(stit,), bbox_inches='tight')
plt.show()
print(fn)

# %% [markdown]
# ## Mask by CoagS?

# %%
ds_diff

# %%
v= 'COAGNUCL_noSECTv21_ox_ricc_dd'
mask_ = ds_diff[v] <ds_diff[v].quantile(.5)
#v= 'NCONC01_noSECTv21_ox_ricc_dd'
#mask_ = mask_ & (ds_diff[v] >10)#ds_diff[v].quantile(.1))


# %%
fig, axs = plt.subplots(3,2, figsize=[6.4,7.1], sharey=True)#, constrained_layout=True)
var_xl = ['NUCLRATE','GR','H2SO4','SOA_LV','NCONC01','COAGNUCL']
lev_min=850.
_ds = ds_diff.where(mask_)#sel(lev=slice(lev_min, None))
var_diff = 'NCONC01'
xlims = {
    'NUCLRATE' : [1.e-6,10],
    'H2SO4' : [1.e-3,1],
    'SOA_LV' : [1.e-5,1],
    'GR' : [1.e-3,1],
    'N_AER':[1e0,1e4],
    'NCONC01':[5e-1,1e3],
    'COAGNUCL':[1e-8,1e-3],
}

_plt_tmp(_ds,axs,var_xl, var_diff, xlims)
fn = plot_path + f'2dhist_{case_orig}_{case_sec}.'
#fig.savefig(fn+'pdf',bbox_extra_artists=(stit,), bbox_inches='tight')
#fig.savefig(fn+'png',bbox_extra_artists=(stit,), bbox_inches='tight')
plt.show()
print(fn)

# %% [markdown]
# ## Oposite

# %%
fig, axs = plt.subplots(3,2, figsize=[6.4,7.1], sharey=True)#, constrained_layout=True)
var_xl = ['NUCLRATE','GR','H2SO4','SOA_LV','NCONC01','COAGNUCL']
lev_min=850.
_ds = ds_diff.where(~mask_)#sel(lev=slice(lev_min, None))
var_diff = 'NCONC01'
xlims = {
    'NUCLRATE' : [1.e-6,10],
    'H2SO4' : [1.e-3,1],
    'SOA_LV' : [1.e-5,1],
    'GR' : [1.e-3,1],
    'N_AER':[1e0,1e4],
    'NCONC01':[5e-1,1e3],
    'COAGNUCL':[1e-8,1e-3],
}

_plt_tmp(_ds,axs,var_xl, var_diff, xlims)
fn = plot_path + f'2dhist_{case_orig}_{case_sec}.'
#fig.savefig(fn+'pdf',bbox_extra_artists=(stit,), bbox_inches='tight')
#fig.savefig(fn+'png',bbox_extra_artists=(stit,), bbox_inches='tight')
plt.show()
print(fn)

# %% [markdown]
# ## Ratio gr/coag
#

# %%
gr = ds_diff['GR_' + case_orig]
coag = ds_diff['COAGNUCL_' + case_orig]
ma = (gr/coag) > 38

# %%
(gr/coag).quantile([.05, .25,.5,.75,.84,.95])

# %%
fig, axs = plt.subplots(3,2, figsize=[6.4,7.1], sharey=True)#, constrained_layout=True)
var_xl = ['NUCLRATE','GR','H2SO4','SOA_LV','NCONC01','COAGNUCL']
lev_min=850.
_ds = ds_diff.where(ma)#sel(lev=slice(lev_min, None))
var_diff = 'NCONC01'
xlims = {
    'NUCLRATE' : [1.e-6,10],
    'H2SO4' : [1.e-3,1],
    'SOA_LV' : [1.e-5,1],
    'GR' : [1.e-3,1],
    'N_AER':[1e0,1e4],
    'NCONC01':[5e-1,1e3],
    'COAGNUCL':[1e-5,.1],
    
}

_plt_tmp(_ds,axs,var_xl, var_diff, xlims)
fn = plot_path + f'2dhist_{case_orig}_{case_sec}.'
#fig.savefig(fn+'pdf',bbox_extra_artists=(stit,), bbox_inches='tight')
#fig.savefig(fn+'png',bbox_extra_artists=(stit,), bbox_inches='tight')
plt.show()
print(fn)

# %%
fig, axs = plt.subplots(3,2, figsize=[6.4,7.1], sharey=True)#, constrained_layout=True)
var_xl = ['NUCLRATE','GR','H2SO4','SOA_LV','NCONC01','COAGNUCL']
lev_min=850.
_ds = ds_diff.where(~ma)#sel(lev=slice(lev_min, None))
var_diff = 'NCONC01'
xlims = {
    'NUCLRATE' : [1.e-6,10],
    'H2SO4' : [1.e-3,1],
    'SOA_LV' : [1.e-5,1],
    'GR' : [1.e-3,1],
    'N_AER':[1e0,1e4],
    'NCONC01':[5e-1,1e3],
    'COAGNUCL':[1e-5,.1],
    
}

_plt_tmp(_ds,axs,var_xl, var_diff, xlims)
fn = plot_path + f'2dhist_{case_orig}_{case_sec}.'
#fig.savefig(fn+'pdf',bbox_extra_artists=(stit,), bbox_inches='tight')
#fig.savefig(fn+'png',bbox_extra_artists=(stit,), bbox_inches='tight')
plt.show()
print(fn)

# %% [markdown]
# ## Mask by quantile NCONC01

# %%
ds_diff

# %%
v= 'logNCONC01_noSECTv21_ox_ricc_dd'
mask_ = ds_diff[v] <ds_diff[v].quantile(.25)
#v= 'NCONC01_noSECTv21_ox_ricc_dd'
#mask_ = mask_ & (ds_diff[v] >10)#ds_diff[v].quantile(.1))


# %%
fig, axs = plt.subplots(3,2, figsize=[6.4,7.1], sharey=True)#, constrained_layout=True)
var_xl = ['NUCLRATE','GR','H2SO4','SOA_LV','NCONC01','COAGNUCL']
lev_min=850.
_ds = ds_diff.where(mask_)#sel(lev=slice(lev_min, None))
var_diff = 'NCONC01'
xlims = {
    'NUCLRATE' : [1.e-6,10],
    'H2SO4' : [1.e-3,1],
    'SOA_LV' : [1.e-5,1],
    'GR' : [1.e-3,1],
    'N_AER':[1e0,1e4],
    'NCONC01':[5e-1,1e3],
    'COAGNUCL':[1e-8,1e-3],
}

_plt_tmp(_ds,axs,var_xl, var_diff, xlims)
fn = plot_path + f'2dhist_{case_orig}_{case_sec}.'
#fig.savefig(fn+'pdf',bbox_extra_artists=(stit,), bbox_inches='tight')
#fig.savefig(fn+'png',bbox_extra_artists=(stit,), bbox_inches='tight')
plt.show()
print(fn)

# %%
v= 'logNCONC01_noSECTv21_ox_ricc_dd'
mask_ = (ds_diff[v] >ds_diff[v].quantile(.25))&(ds_diff[v] <ds_diff[v].quantile(.50))
#v= 'NCONC01_noSECTv21_ox_ricc_dd'
#mask_ = mask_ & (ds_diff[v] >10)#ds_diff[v].quantile(.1))


# %%
fig, axs = plt.subplots(3,2, figsize=[6.4,7.1], sharey=True)#, constrained_layout=True)
var_xl = ['NUCLRATE','GR','H2SO4','SOA_LV','NCONC01','COAGNUCL']
lev_min=850.
_ds = ds_diff.where(mask_)#sel(lev=slice(lev_min, None))
var_diff = 'NCONC01'
xlims = {
    'NUCLRATE' : [1.e-6,10],
    'H2SO4' : [1.e-3,1],
    'SOA_LV' : [1.e-5,1],
    'GR' : [1.e-3,1],
    'N_AER':[1e0,1e4],
    'NCONC01':[5e-1,1e3],
    'COAGNUCL':[1e-8,1e-3],
}

_plt_tmp(_ds,axs,var_xl, var_diff, xlims)
fn = plot_path + f'2dhist_{case_orig}_{case_sec}.'
#fig.savefig(fn+'pdf',bbox_extra_artists=(stit,), bbox_inches='tight')
#fig.savefig(fn+'png',bbox_extra_artists=(stit,), bbox_inches='tight')
plt.show()
print(fn)

# %%
v= 'logNCONC01_noSECTv21_ox_ricc_dd'
mask_ = (ds_diff[v] >ds_diff[v].quantile(.50))&(ds_diff[v] <ds_diff[v].quantile(.75))
#v= 'NCONC01_noSECTv21_ox_ricc_dd'
#mask_ = mask_ & (ds_diff[v] >10)#ds_diff[v].quantile(.1))


# %%
fig, axs = plt.subplots(3,2, figsize=[6.4,7.1], sharey=True)#, constrained_layout=True)
var_xl = ['NUCLRATE','GR','H2SO4','SOA_LV','NCONC01','COAGNUCL']
lev_min=850.
_ds = ds_diff.where(mask_)#sel(lev=slice(lev_min, None))
var_diff = 'NCONC01'
xlims = {
    'NUCLRATE' : [1.e-6,10],
    'H2SO4' : [1.e-3,1],
    'SOA_LV' : [1.e-5,1],
    'GR' : [1.e-3,1],
    'N_AER':[1e0,1e4],
    'NCONC01':[5e-1,1e3],
    'COAGNUCL':[1e-8,1e-3],
}

_plt_tmp(_ds,axs,var_xl, var_diff, xlims)
fn = plot_path + f'2dhist_{case_orig}_{case_sec}.'
#fig.savefig(fn+'pdf',bbox_extra_artists=(stit,), bbox_inches='tight')
#fig.savefig(fn+'png',bbox_extra_artists=(stit,), bbox_inches='tight')
plt.show()
print(fn)

# %%
v= 'logNCONC01_noSECTv21_ox_ricc_dd'
mask_ = (ds_diff[v] >ds_diff[v].quantile(.75))#&(ds_diff[v] <ds_diff[v].quantile(.75))
#v= 'NCONC01_noSECTv21_ox_ricc_dd'
#mask_ = mask_ & (ds_diff[v] >10)#ds_diff[v].quantile(.1))


# %%
fig, axs = plt.subplots(3,2, figsize=[6.4,7.1], sharey=True)#, constrained_layout=True)
var_xl = ['NUCLRATE','GR','H2SO4','SOA_LV','NCONC01','COAGNUCL']
lev_min=850.
_ds = ds_diff.where(mask_)#sel(lev=slice(lev_min, None))
var_diff = 'NCONC01'
xlims = {
    'NUCLRATE' : [1.e-6,10],
    'H2SO4' : [1.e-3,1],
    'SOA_LV' : [1.e-5,1],
    'GR' : [1.e-3,1],
    'N_AER':[1e0,1e4],
    'NCONC01':[5e-1,1e3],
    'COAGNUCL':[1e-8,1e-3],
}

_plt_tmp(_ds,axs,var_xl, var_diff, xlims)
fn = plot_path + f'2dhist_{case_orig}_{case_sec}.'
#fig.savefig(fn+'pdf',bbox_extra_artists=(stit,), bbox_inches='tight')
#fig.savefig(fn+'png',bbox_extra_artists=(stit,), bbox_inches='tight')
plt.show()
print(fn)

# %%
fig, axs = plt.subplots(3,2, figsize=[6.4,7.1], sharey=True)#, constrained_layout=True)
var_xl = ['NUCLRATE','GR','H2SO4','SOA_LV','N_AER','COAGNUCL']
lev_min=800.
_ds = ds_diff.sel(lev=slice(lev_min, None))
var_diff = 'NCONC01'
xlims = {
    'NUCLRATE' : [1.e-6,10],
    'H2SO4' : [1.e-3,1],
    'SOA_LV' : [1.e-5,1],
    'GR' : [1.e-3,1],
    'N_AER':[1e0,1e4],
    'NCONC01':[5e-1,1e3],
    'COAGNUCL':[1e-8,1e-3],
}
for var,ax in zip(var_xl, axs.flatten()):
    print(var)
    xlim = xlims[var]
    h = _plt_2dhist(_ds,f'{var}_{case_orig}', var_diff,
                nr_bins=40,
                xlim=xlim, ax=ax)
    uni = var_info.get_fancy_unit_xr(_ds[var],
                                     var)
    ax.set_xlabel(f'{get_fancy_var_name(var)} [{uni}]')
    ax.plot(xlim,[0,0], linewidth=.5, c='k')

uni = var_info.get_fancy_unit_xr(_ds[var_diff],
                           var_diff)
fvar_diff = get_fancy_var_name(var_diff)
ylab = f'$\Delta${fvar_diff} [{uni}]'
for ax in axs[:,0]:
    ax.set_ylabel(ylab)
    
subp_insert_abc(axs)

suptit =f'{get_nice_name_case(case_sec)}-{get_nice_name_case(case_orig)} vs.  '
suptit =f'{fvar_diff}$(m_1)-${fvar_diff}$(m_2)$ vs. $X(m_2)$ \n for $m_1=${get_nice_name_case(case_sec)}, $m_2=${get_nice_name_case(case_orig)}'
fig.subplots_adjust(hspace=.5, wspace=0.1)#,top=0.8, )
stit = fig.suptitle(suptit,  fontsize=12, y=.98)
fn = plot_path + f'2dhist_{case_orig}_{case_sec}.'
#fig.savefig(fn+'pdf',bbox_extra_artists=(stit,), bbox_inches='tight')
#fig.savefig(fn+'png',bbox_extra_artists=(stit,), bbox_inches='tight')
plt.show()
print(fn)

# %%
fig, axs = plt.subplots(3,2, figsize=[6.4,7.1], sharey=True)#, constrained_layout=True)
var_xl = ['NUCLRATE','GR','H2SO4','SOA_LV','NCONC01','COAGNUCL']
lev_min=800.
_ds = ds_diff.sel(lev=slice( None,lev_min))
var_diff = 'NCONC01'
xlims = {
    'NUCLRATE' : [1.e-6,10],
    'H2SO4' : [1.e-3,1],
    'SOA_LV' : [1.e-5,1],
    'GR' : [1.e-3,1],
    'N_AER':[1e0,1e4],
    'NCONC01':[5e-1,1e3],
    'COAGNUCL':[1e-8,1e-3],
}
for var,ax in zip(var_xl, axs.flatten()):
    print(var)
    xlim = xlims[var]
    h = _plt_2dhist(_ds,f'{var}_{case_orig}', var_diff,
                nr_bins=40,
                xlim=xlim, ax=ax)
    uni = var_info.get_fancy_unit_xr(_ds[var],
                                     var)
    ax.set_xlabel(f'{get_fancy_var_name(var)} [{uni}]')
    ax.plot(xlim,[0,0], linewidth=.5, c='k')

uni = var_info.get_fancy_unit_xr(_ds[var_diff],
                           var_diff)
fvar_diff = get_fancy_var_name(var_diff)
ylab = f'$\Delta${fvar_diff} [{uni}]'
for ax in axs[:,0]:
    ax.set_ylabel(ylab)
    
subp_insert_abc(axs)

suptit =f'{get_nice_name_case(case_sec)}-{get_nice_name_case(case_orig)} vs.  '
suptit =f'{fvar_diff}$(m_1)-${fvar_diff}$(m_2)$ vs. $X(m_2)$ \n for $m_1=${get_nice_name_case(case_sec)}, $m_2=${get_nice_name_case(case_orig)}'
fig.subplots_adjust(hspace=.5, wspace=0.1)#,top=0.8, )
stit = fig.suptitle(suptit,  fontsize=12, y=.98)
fn = plot_path + f'2dhist_{case_orig}_{case_sec}.'
#fig.savefig(fn+'pdf',bbox_extra_artists=(stit,), bbox_inches='tight')
#fig.savefig(fn+'png',bbox_extra_artists=(stit,), bbox_inches='tight')
plt.show()
print(fn)

# %%
fig, axs = plt.subplots(3,2, figsize=[6.4,7.1], sharey=True)#, constrained_layout=True)
var_xl = ['NUCLRATE','GR','H2SO4','SOA_LV','NCONC01','COAGNUCL']
lev_min=600.
_ds = ds_diff.sel(lev=slice( None,lev_min))
var_diff = 'NCONC01'
xlims = {
    'NUCLRATE' : [1.e-6,10],
    'H2SO4' : [1.e-3,1],
    'SOA_LV' : [1.e-5,1],
    'GR' : [1.e-3,1],
    'N_AER':[1e0,1e4],
    'NCONC01':[5e-1,1e3],
    'COAGNUCL':[1e-8,1e-3],
}
for var,ax in zip(var_xl, axs.flatten()):
    print(var)
    xlim = xlims[var]
    h = _plt_2dhist(_ds,f'{var}_{case_orig}', var_diff,
                nr_bins=40,
                xlim=xlim, ax=ax)
    uni = var_info.get_fancy_unit_xr(_ds[var],
                                     var)
    ax.set_xlabel(f'{get_fancy_var_name(var)} [{uni}]')
    ax.plot(xlim,[0,0], linewidth=.5, c='k')

uni = var_info.get_fancy_unit_xr(_ds[var_diff],
                           var_diff)
fvar_diff = get_fancy_var_name(var_diff)
ylab = f'$\Delta${fvar_diff} [{uni}]'
for ax in axs[:,0]:
    ax.set_ylabel(ylab)
    
subp_insert_abc(axs)

suptit =f'{get_nice_name_case(case_sec)}-{get_nice_name_case(case_orig)} vs.  '
suptit =f'{fvar_diff}$(m_1)-${fvar_diff}$(m_2)$ vs. $X(m_2)$ \n for $m_1=${get_nice_name_case(case_sec)}, $m_2=${get_nice_name_case(case_orig)}'
fig.subplots_adjust(hspace=.5, wspace=0.1)#,top=0.8, )
stit = fig.suptitle(suptit,  fontsize=12, y=.98)
fn = plot_path + f'2dhist_{case_orig}_{case_sec}.'
#fig.savefig(fn+'pdf',bbox_extra_artists=(stit,), bbox_inches='tight')
#fig.savefig(fn+'png',bbox_extra_artists=(stit,), bbox_inches='tight')
plt.show()
print(fn)

# %%
import seaborn as sns

# %%
fig, axs = plt.subplots(3,2, figsize=[6.4,7.1], sharey=True)#, constrained_layout=True)
var_xl = ['NUCLRATE','GR','H2SO4','SOA_LV','N_AER','COAGNUCL']
lev_min=900
_ds = ds_diff#.sel(lev=slice( lev_min,None))
var_diff = 'SOA_LV'
xlims = {
    'NUCLRATE' : [1.e-6,10],
    'H2SO4' : [1.e-3,1],
    'SOA_LV' : [1.e-5,1],
    'GR' : [1.e-3,1],
    'N_AER':[1e0,1e4],
    'NCONC01':[1e0,1e4],
    'COAGNUCL':[1e-8,1e-4],
}
for var,ax in zip(var_xl, axs.flatten()):
    print(var)
    xlim = xlims[var]
    _plt_2dhist(_ds,f'{var}_{case_orig}', var_diff,
                nr_bins=40,
                xlim=xlim, ax=ax, ylim=xlims[var_diff])
    uni = var_info.get_fancy_unit_xr(_ds[var],
                                     var)
    ax.set_xlabel(f'{get_fancy_var_name(var)} [{uni}]')

uni = var_info.get_fancy_unit_xr(_ds[var_diff],
                           var_diff)
fvar_diff = get_fancy_var_name(var_diff)
ylab = f'$\Delta${fvar_diff} [{uni}]'
for ax in axs[:,0]:
    ax.set_ylabel(ylab)
suptit =f'{get_nice_name_case(case_sec)}-{get_nice_name_case(case_orig)} vs.  '
suptit =f'{fvar_diff}$(m_1)-${fvar_diff}$(m_2)$ vs. $X(m_2)$ \n for $m_1=${get_nice_name_case(case_sec)}, $m_2=${get_nice_name_case(case_orig)}'
fig.subplots_adjust(hspace=.5, wspace=0.1)#,top=0.8, )
stit = fig.suptitle(suptit,  fontsize=12, y=.98)
fn = plot_path + f'2dhist_{case_orig}_{case_sec}.'
#fig.savefig(fn+'pdf',bbox_extra_artists=(stit,), bbox_inches='tight')
#fig.savefig(fn+'png',bbox_extra_artists=(stit,), bbox_inches='tight')
plt.show()

# %%
fig, axs = plt.subplots(3,2, figsize=[6.4,7.1], sharey=True)#, constrained_layout=True)
var_xl = ['NUCLRATE','GR','H2SO4','SOA_LV','N_AER','COAGNUCL']
lev_min=100.
_ds = ds_diff.sel(lev=slice(lev_min, None))
var_diff = 'NCONC01'
xlims = {
    'NUCLRATE' : [1.e-6,10],
    'H2SO4' : [1.e-3,1],
    'SOA_LV' : [1.e-5,1],
    'GR' : [1.e-3,1],
    'N_AER':[1e0,1e4],
    'NCONC01':[1e0,1e4],
    'COAGNUCL':[1e-8,1e-4],
}
for var,ax in zip(var_xl, axs.flatten()):
    print(var)
    xlim = xlims[var]
    _plt_2dhist(_ds,f'{var}_{case_orig}', var_diff,
                nr_bins=40,
                xlim=xlim, ax=ax)
    uni = var_info.get_fancy_unit_xr(_ds[var],
                                     var)
    ax.set_xlabel(f'{get_fancy_var_name(var)} [{uni}]')

uni = var_info.get_fancy_unit_xr(_ds[var_diff],
                           var_diff)
fvar_diff = get_fancy_var_name(var_diff)
ylab = f'$\Delta${fvar_diff} [{uni}]'
for ax in axs[:,0]:
    ax.set_ylabel(ylab)
suptit =f'{get_nice_name_case(case_sec)}-{get_nice_name_case(case_orig)} vs.  '
suptit =f'{fvar_diff}$(m_1)-${fvar_diff}$(m_2)$ vs. $X(m_2)$ \n for $m_1=${get_nice_name_case(case_sec)}, $m_2=${get_nice_name_case(case_orig)}'
fig.subplots_adjust(hspace=.5, wspace=0.1)#,top=0.8, )
stit = fig.suptitle(suptit,  fontsize=12, y=.98)
fn = plot_path + f'2dhist_{case_orig}_{case_sec}.'
fig.savefig(fn+'pdf',bbox_extra_artists=(stit,), bbox_inches='tight')
fig.savefig(fn+'png',bbox_extra_artists=(stit,), bbox_inches='tight')
plt.show()

# %%
_ds

# %%
import seaborn as sns


# %%
import importlib as il
from useful_scit.util import conditional_stats
il.reload(conditional_stats)
varList = ['H2SO4','SOA_LV','NCONC01']
varList = ['N_AER','GR','NCONC01']
sec_sub = cases_dic[case_sec][varList].stack(ind=('lat','lon','lev','time'))#
#sec_sub = sec_sub
nosec_sub = cases_dic[case_orig][varList].stack(ind=('lat','lon','lev','time'))
fig, axs = plt.subplots(3,2, figsize=[10,10])
axs=axs.flatten()
df, pl_xr_ns,  kwargs, xedges, yedges = conditional_stats.plot_cond_statistic(nosec_sub,varList,axs[0],quant=0.1,scale='log', stat='mean',  plt_title='Mean Non sectional', cscale='log')
df, pl_xr_s,  kwargs, xedges, yedges = conditional_stats.plot_cond_statistic(sec_sub,  varList,axs[1],quant=0.1,scale='log', stat='mean',  plt_title='Mean Sectional',     cscale='log', xedges=xedges, yedges=yedges)
conditional_stats.plot_cond_statistic(nosec_sub,varList,axs[2],quant=0.1,scale='log', stat='std',   plt_title='Std Non sectional',  cscale='log', xedges=xedges, yedges=yedges)
conditional_stats.plot_cond_statistic(sec_sub,  varList,axs[3],quant=0.1,scale='log', stat='std',   plt_title='Std Sectional',      cscale='log', xedges=xedges, yedges=yedges)
df, pl_xr_ns_c,  kwargs, xedges, yedges =conditional_stats.plot_cond_statistic(nosec_sub,varList,axs[4],quant=0.1,scale='log', stat='count', plt_title='Count Non sectional',cscale='log', xedges=xedges, yedges=yedges)
df, pl_xr_s_c,  kwargs, xedges, yedges =conditional_stats.plot_cond_statistic(sec_sub,  varList,axs[5],quant=0.1,scale='log', stat='count', plt_title='Count Sectional',    cscale='log', xedges=xedges, yedges=yedges)

plt.tight_layout()
plt.show()
#pl_xr_ns

# %%
(pl_xr_s -pl_xr_ns).plot(x=varList[0],y=varList[1],xlim=kwargs['xlim'], robust=True,
                         ylim=kwargs['ylim'], yscale='log',xscale='log')

plt.show()

# %%
(pl_xr_s_c -pl_xr_ns_c).plot(x=varList[0],y=varList[1],xlim=kwargs['xlim'], robust=True,
                         ylim=kwargs['ylim'], yscale='log',xscale='log')

plt.show()

# %%
import importlib as il
from useful_scit.util import conditional_stats
il.reload(conditional_stats)
varList = ['H2SO4','SOA_LV','NCONC01']
varList = ['GR','COAGNUCL','NCONC01']
sec_sub = cases_dic[case_sec][varList].stack(ind=('lat','lon','lev','time'))#
#sec_sub = sec_sub
nosec_sub = cases_dic[case_orig][varList].stack(ind=('lat','lon','lev','time'))
fig, axs = plt.subplots(3,2, figsize=[10,10])
axs=axs.flatten()
df, pl_xr_ns,  kwargs, xedges, yedges = conditional_stats.plot_cond_statistic(nosec_sub,varList,axs[0],quant=0.1,scale='log', stat='mean',  plt_title='Mean Non sectional', cscale='log')
df, pl_xr_s,  kwargs, xedges, yedges = conditional_stats.plot_cond_statistic(sec_sub,  varList,axs[1],quant=0.1,scale='log', stat='mean',  plt_title='Mean Sectional',     cscale='log', xedges=xedges, yedges=yedges)
conditional_stats.plot_cond_statistic(nosec_sub,varList,axs[2],quant=0.1,scale='log', stat='std',   plt_title='Std Non sectional',  cscale='log', xedges=xedges, yedges=yedges)
conditional_stats.plot_cond_statistic(sec_sub,  varList,axs[3],quant=0.1,scale='log', stat='std',   plt_title='Std Sectional',      cscale='log', xedges=xedges, yedges=yedges)
df, pl_xr_ns_c,  kwargs, xedges, yedges =conditional_stats.plot_cond_statistic(nosec_sub,varList,axs[4],quant=0.1,scale='log', stat='count', plt_title='Count Non sectional',cscale='log', xedges=xedges, yedges=yedges)
df, pl_xr_s_c,  kwargs, xedges, yedges =conditional_stats.plot_cond_statistic(sec_sub,  varList,axs[5],quant=0.1,scale='log', stat='count', plt_title='Count Sectional',    cscale='log', xedges=xedges, yedges=yedges)

plt.tight_layout()
plt.show()
#pl_xr_ns

# %%
(pl_xr_s -pl_xr_ns).plot(x=varList[0],y=varList[1],xlim=kwargs['xlim'], robust=True,
                         ylim=kwargs['ylim'], yscale='log',xscale='log')

plt.show()

# %%
(pl_xr_s_c -pl_xr_ns_c).plot(x=varList[0],y=varList[1],xlim=kwargs['xlim'], robust=True,
                         ylim=kwargs['ylim'], yscale='log',xscale='log')

plt.show()

# %%
# %%
(pl_xr_s_c -pl_xr_ns_c).plot(x=varList[0],y=varList[1],xlim=kwargs['xlim'], robust=True,
                         ylim=kwargs['ylim'], yscale='log',xscale='log')

plt.show()

# %%
(pl_xr_s -pl_xr_ns).plot(x=varList[0],y=varList[1],xlim=kwargs['xlim'], robust=True,
                        ylim=kwargs['ylim'], yscale='log',xscale='log')

plt.show()

# %%
# %%
# %%
import importlib as il
from useful_scit.util import conditional_stats
il.reload(conditional_stats)
varList = ['N_AER','GR','NCONC01']
sec_sub = cases_dic[case_sec][varList].stack(ind=('lat','lon','lev','time'))#
#sec_sub = sec_sub
nosec_sub = cases_dic[case_orig][varList].stack(ind=('lat','lon','lev','time'))
fig, axs = plt.subplots(3,2, figsize=[10,10])
axs=axs.flatten()
df, pl_xr_s,  kwargs, xedges, yedges = conditional_stats.plot_cond_statistic(sec_sub,  varList,axs[0],quant=0.1,scale='log', stat='mean',  plt_title='Mean Sectional',     cscale='log')
df, pl_xr_ns,  kwargs, xedges, yedges = conditional_stats.plot_cond_statistic(nosec_sub,varList,axs[1],quant=0.1,scale='log', stat='mean',  plt_title='Mean Non sectional', cscale='log', xedges=xedges, yedges=yedges)
conditional_stats.plot_cond_statistic(sec_sub,  varList,axs[2],quant=0.1,scale='log', stat='std',   plt_title='Std Sectional',      cscale='log')
conditional_stats.plot_cond_statistic(nosec_sub,varList,axs[3],quant=0.1,scale='log', stat='std',   plt_title='Std Non sectional',  cscale='log')
conditional_stats.plot_cond_statistic(sec_sub,  varList,axs[4],quant=0.1,scale='log', stat='count', plt_title='Count Sectional',    cscale='log')
conditional_stats.plot_cond_statistic(nosec_sub,varList,axs[5],quant=0.1,scale='log', stat='count', plt_title='Count Non sectional',cscale='log')
plt.tight_layout()
plt.show()
#pl_xr_ns
# %%

# %%
(pl_xr_s -pl_xr_ns).plot(x=varList[0],y=varList[1],xlim=kwargs['xlim'], robust=True,
                         ylim=kwargs['ylim'], yscale='log',xscale='log')

plt.show()

# %%
# %%
# %%
import importlib as il
from useful_scit.util import conditional_stats
il.reload(conditional_stats)
varList = ['N_AER','NCONC01','H2SO4']
sec_sub = cases_dic[case_sec][varList].stack(ind=('lat','lon','lev','time'))#
#sec_sub = sec_sub
nosec_sub = cases_dic[case_orig][varList].stack(ind=('lat','lon','lev','time'))
fig, axs = plt.subplots(3,2, figsize=[10,10])
axs=axs.flatten()
df, pl_xr_s,  kwargs, xedges, yedges = conditional_stats.plot_cond_statistic(sec_sub,  varList,axs[0],quant=0.1,scale='log', stat='mean',  plt_title='Mean Sectional',     cscale='log')
df, pl_xr_ns,  kwargs, xedges, yedges = conditional_stats.plot_cond_statistic(nosec_sub,varList,axs[1],quant=0.1,scale='log', stat='mean',  plt_title='Mean Non sectional', cscale='log', xedges=xedges, yedges=yedges)
conditional_stats.plot_cond_statistic(sec_sub,  varList,axs[2],quant=0.1,scale='log', stat='std',   plt_title='Std Sectional',      cscale='log', xedges=xedges, yedges=yedges)
conditional_stats.plot_cond_statistic(nosec_sub,varList,axs[3],quant=0.1,scale='log', stat='std',   plt_title='Std Non sectional',  cscale='log', xedges=xedges, yedges=yedges)
df, pl_xr_s_c,  kwargs, xedges, yedges =conditional_stats.plot_cond_statistic(sec_sub,  varList,axs[4],quant=0.1,scale='log', stat='count', plt_title='Count Sectional',    cscale='log', xedges=xedges, yedges=yedges)
df, pl_xr_ns_c,  kwargs, xedges, yedges =conditional_stats.plot_cond_statistic(nosec_sub,varList,axs[5],quant=0.1,scale='log', stat='count', plt_title='Count Non sectional',cscale='log', xedges=xedges, yedges=yedges)
plt.tight_layout()
plt.show()
#pl_xr_ns

# %%
# %%
(pl_xr_s -pl_xr_ns).plot(x=varList[0],y=varList[1],xlim=kwargs['xlim'], robust=True,
                         ylim=kwargs['ylim'], yscale='log',xscale='log')

plt.show()
# %%

# %%
for case in cases:
    _ds = cases_dic[case]
    _ds['N_nt01'] = _ds['N_AER']-_ds['NCONC01']
    _ds['N_nt01'] = _ds['N_nt01'].where(_ds['N_nt01']>0)

# %%
import importlib as il
from useful_scit.util import conditional_stats
il.reload(conditional_stats)
varList = ['H2SO4','NCONC01','SOA_LV']
sec_sub = cases_dic[case_sec][varList].stack(ind=('lat','lon','lev','time'))#
#sec_sub = sec_sub
nosec_sub = cases_dic[case_orig][varList].stack(ind=('lat','lon','lev','time'))
fig, axs = plt.subplots(3,2, figsize=[10,10])
axs=axs.flatten()
df, pl_xr_s,  kwargs, xedges, yedges = conditional_stats.plot_cond_statistic(sec_sub,  varList,axs[0],quant=0.1,scale='log', stat='mean',  plt_title='Mean Sectional',     cscale='log')
df, pl_xr_ns,  kwargs, xedges, yedges = conditional_stats.plot_cond_statistic(nosec_sub,varList,axs[1],quant=0.1,scale='log', stat='mean',  plt_title='Mean Non sectional', cscale='log', xedges=xedges, yedges=yedges)
conditional_stats.plot_cond_statistic(sec_sub,  varList,axs[2],quant=0.1,scale='log', stat='std',   plt_title='Std Sectional',      cscale='log')
conditional_stats.plot_cond_statistic(nosec_sub,varList,axs[3],quant=0.1,scale='log', stat='std',   plt_title='Std Non sectional',  cscale='log')
df, pl_xr_s_c,  kwargs, xedges, yedges = conditional_stats.plot_cond_statistic(sec_sub,  varList,axs[4],quant=0.1,scale='log', stat='count', plt_title='Count Sectional',    cscale='log')
df, pl_xr_ns_c,  kwargs, xedges, yedges = conditional_stats.plot_cond_statistic(nosec_sub,varList,axs[5],quant=0.1,scale='log', stat='count', plt_title='Count Non sectional',cscale='log', xedges=xedges, yedges=yedges)
plt.tight_layout()
plt.show()
#pl_xr_ns
# %%

# %%
# %%
from useful_scit.util.zarray import corr
(pl_xr_s -pl_xr_ns).plot(x=varList[0],y=varList[1],xlim=kwargs['xlim'], robust=True,
                         ylim=kwargs['ylim'], yscale='log',xscale='log')

plt.show()

# %%
# %%
import importlib as il
from useful_scit.util import conditional_stats
il.reload(conditional_stats)
varList = ['N_nt01','NUCLRATE','NCONC01']
sec_sub = cases_dic[case_sec][varList].stack(ind=('lat','lon','lev','time'))#
#sec_sub = sec_sub
nosec_sub = cases_dic[case_orig][varList].stack(ind=('lat','lon','lev','time'))
fig, axs = plt.subplots(3,2, figsize=[10,10])
axs=axs.flatten()
df, pl_xr_s,  kwargs, xedges, yedges = conditional_stats.plot_cond_statistic(sec_sub,  varList,axs[0],quant=0.1,scale='log', stat='mean',  plt_title='Mean Sectional',     cscale='log')
df, pl_xr_ns,  kwargs, xedges, yedges = conditional_stats.plot_cond_statistic(nosec_sub,varList,axs[1],quant=0.1,scale='log', stat='mean',  plt_title='Mean Non sectional', cscale='log', xedges=xedges, yedges=yedges)
conditional_stats.plot_cond_statistic(sec_sub,  varList,axs[2],quant=0.1,scale='log', stat='std',   plt_title='Std Sectional',      cscale='log')
conditional_stats.plot_cond_statistic(nosec_sub,varList,axs[3],quant=0.1,scale='log', stat='std',   plt_title='Std Non sectional',  cscale='log')
conditional_stats.plot_cond_statistic(sec_sub,  varList,axs[4],quant=0.1,scale='log', stat='count', plt_title='Count Sectional',    cscale='log')
conditional_stats.plot_cond_statistic(nosec_sub,varList,axs[5],quant=0.1,scale='log', stat='count', plt_title='Count Non sectional',cscale='log')
plt.tight_layout()
plt.show()
#pl_xr_ns
# %%

# %%
from useful_scit.util.zarray import corr
(pl_xr_s -pl_xr_ns).plot(x=varList[0],y=varList[1],xlim=kwargs['xlim'], robust=True,
                         ylim=kwargs['ylim'], yscale='log',xscale='log')

plt.show()

# %%
# %%
import importlib as il
from useful_scit.util import conditional_stats
il.reload(conditional_stats)
varList = ['GR','NCONC01','NUCLRATE']
sec_sub = cases_dic[case_sec][varList].stack(ind=('lat','lon','lev','time'))#
#sec_sub = sec_sub
nosec_sub = cases_dic[case_orig][varList].stack(ind=('lat','lon','lev','time'))
fig, axs = plt.subplots(3,2, figsize=[10,10])
axs=axs.flatten()
df, pl_xr_s,  kwargs, xedges, yedges = conditional_stats.plot_cond_statistic(sec_sub,  varList,axs[0],quant=0.1,scale='log', stat='mean',  plt_title='Mean Sectional',     cscale='log')
df, pl_xr_ns,  kwargs, xedges, yedges = conditional_stats.plot_cond_statistic(nosec_sub,varList,axs[1],quant=0.1,scale='log', stat='mean',  plt_title='Mean Non sectional', cscale='log', xedges=xedges, yedges=yedges)
conditional_stats.plot_cond_statistic(sec_sub,  varList,axs[2],quant=0.1,scale='log', stat='std',   plt_title='Std Sectional',      cscale='log')
conditional_stats.plot_cond_statistic(nosec_sub,varList,axs[3],quant=0.1,scale='log', stat='std',   plt_title='Std Non sectional',  cscale='log')
conditional_stats.plot_cond_statistic(sec_sub,  varList,axs[4],quant=0.1,scale='log', stat='count', plt_title='Count Sectional',    cscale='log')
conditional_stats.plot_cond_statistic(nosec_sub,varList,axs[5],quant=0.1,scale='log', stat='count', plt_title='Count Non sectional',cscale='log')
plt.tight_layout()
plt.show()
#pl_xr_ns
# %%
from useful_scit.util.zarray import corr
(pl_xr_s -pl_xr_ns).plot(x=varList[0],y=varList[1],xlim=kwargs['xlim'], robust=True,
                         ylim=kwargs['ylim'], yscale='log',xscale='log')

plt.show()
# %%

# %%
plt.figure()
pl_xr.plot(ylim=[1e-15,2e-2], yscale='log',xscale='log')
plt.show()
# %%
import seaborn as sns
sec_sub
_df = sec_sub[['SOA_LV','H2SO4']].to_dataframe()
_df =_df[~_df['SOA_LV'].isnull()]
_df = _df[_df['SOA_LV']>0]
# %%
sns.jointplot(x='SOA_LV',y='H2SO4',data=np.log10(_df), kind='kde')
plt.show()
# %%
#q5 = ds_diff['logH2SO4'+'_'+ case_orig].quantile(.01)
#q95 = ds_diff['logH2SO4'+'_'+ case_orig].quantile(.99)
# %%
ds_diff_m = ds_diff#.where(ds_diff['logH2SO4'+'_'+ case_orig]>q5).where( ds_diff['logH2SO4'+'_'+ case_orig]<q95)

# %%

ds_diff_m = ds_diff_m.where(ds_diff_m['lev']>lev_lim)

# %%

var = var1
ds_diff_m.mean('time').plot.scatter(y=(var), hue=(f'log{var2}_{case_orig}'), x=(f'log{var1}_{case_orig}'), alpha=0.3, cmap='viridis', robust=True)
#plt.plot([1e0,1e4],[1e0,1e4])
#plt.xscale('log')
#plt.yscale('log')
plt.show()
# %%
var = var1
ds_diff_m.mean('time').plot.scatter(y=(var), hue=(f'log{var_subl[2]}_{case_orig}'), x=(f'log{var1}_{case_orig}'), alpha=0.3, cmap='viridis', robust=True)
#plt.plot([1e0,1e4],[1e0,1e4])
#plt.xscale('log')
#plt.yscale('log')
plt.show()

# %%

var = 'N_AER'
ds_diff_m.mean('time').plot.scatter(x=(var+'_'+case_sec), y=(var +'_'+case_orig), hue=('logH2SO4'+'_'+case_orig), alpha=0.4)
plt.plot([1e0,1e4],[1e0,1e4])
plt.xscale('log')
plt.yscale('log')
plt.show()
# %%

var = 'N_AER'
ds_diff_m.mean('time').plot.scatter(x=(var+'_'+case_sec), y=(var +'_'+case_orig), hue=(f'log{var_subl[2]}_{case_orig}'), alpha=0.4)
plt.plot([1e0,1e4],[1e0,1e4])
plt.xscale('log')
plt.yscale('log')
plt.show()

# %%
var = 'N_AER'
ds_diff_m.mean('time').plot.scatter(x=(f'{var}_{case_sec}'), y=(f'{var}_{case_orig}'), hue=(var), alpha=0.4)
plt.plot([1e0,1e4],[1e0,1e4])
plt.xscale('log')
plt.yscale('log')
plt.show()

# %%
var = 'N_AER'
var_hue = 'H2SO4'
ds_diff_m.mean('time').plot.scatter(x=(var+'_'+case_sec), y=(var +'_'+case_orig), hue=(f'log{var_hue}_{case_sec}'), alpha=0.4)
plt.plot([1e0,1e4],[1e0,1e4])
plt.xscale('log')
plt.yscale('log')
plt.show()
# %%

# %%
# %%
var = 'N_AER'
i_lo = 80
for i in np.arange(0,len(ds_diff_m['lat']),10):

    ds_diff_m.isel(lat=slice(i,i+10), lon=slice(i_lo,i_lo+10)).plot.scatter(x=(var), hue=(f'log{var}_{case_orig}'), y=(f'log{var_hue}_{case_sec}'), alpha=0.4, vmin=1)
    lo1=ds_diff_m['lon'][i_lo]
    lo2=ds_diff_m['lon'][i_lo+10]
    la1 = ds_diff_m['lat'][i]
    if i+10>=96:
        la2=ds_diff_m['lat'][-1]
    else:
        la2 =ds_diff_m['lat'][i+10]
    plt.title('Lat:%f-%f , lon: %f-%f'%(la1,la2, lo1,lo2))
    plt.show()

# %%
from useful_scit.util.zarray import corr,cov
var = 'NCONC01'
var2='H2SO4'
co = cov(ds_diff_m[var]-ds_diff_m[var].mean(), ds_diff_m['log'+var+'_'+case_sec], dim='time')
for i in range(7):
    co.isel(lev=-i).plot(robust=True)
    plt.title('corr %s and %s'%(var,'logH2SO4'+'_'+case_sec))
    plt.show()
# %%
from useful_scit.util.zarray import corr
co = corr(ds_diff_m[var], ds_diff_m[var+'_'+case_sec], dim='time')
for i in range(7):
    co.isel(lev=-i).plot()
    plt.title('corr %s and %s'%(var,var+'_'+case_sec))
    plt.show()

# %%
plt.plot([1,2])
plt.show()

# %%
_vars = [var1, var2]
for case in cases:
    _vars = _vars + [f'{var}_{case}' for var in var_subl]
    _vars = _vars + [f'log{var}_{case}' for var in var_subl]
# %%
df = ds_diff_m[_vars].isel(lev=slice(-10,None)).to_dataframe()

# %%
from useful_scit.imps import (sns)
df[f'log{var}_noSEC_gr'] = np.floor(df[f'log{var}_{case_orig}'])
df_ri = df.reset_index()
# %%

#df_ri['NN_AER']=df_ri['N_AER']

# %%
df_ri#['N_AER']
# %%
res = df_ri.groupby(by=f'log{var1}_noSEC_gr')[var1].std()
res.name = 'STD'

import pandas as pd
merg = pd.merge(df_ri,res,right_on=f'log{var1}_noSEC_gr',left_on=f'log{var1}_noSEC_gr')
merg[f'{var1}_norm'] = merg[var1]/merg['STD']
_a = merg[f'log{var1}_noSEC_gr']
#_df=merg[(_a<=1) | (_a>=5)]
_df = merg

# %%
merg[f'{var1}_norm'].describe()

# %%
_df[f'log{var1}_noSEC_gr'].plot.hist(bins=20)
plt.show()
# %%
# %%
#g = sns.relplot(y='N_AER',x='logH2SO4_SECTv11_ctrl', col='logN_AER_noSEC_gr', data=df, col_wrap=3,alpha=.3 )
g = sns.FacetGrid( col=f'log{var1}_noSEC_gr', data=_df, col_wrap=4)
def _f(x,y,color):
    plt.hexbin(x,y, gridsize=40, extent=[-16,-11.5,-4,4], cmap='Reds')
g.map(_f,f'log{var2}_'+case_sec,f'{var1}_norm')

plt.show()
# %%

naer =_df['N_AER']
_df[f'log{var1}']=np.nan
_df[f'log{var1}'][naer>0] = np.log10(_df[naer>0][var1])
_df[f'log{var1}'][naer<0] = -np.log10(-_df[naer<0][var1])

# %%
extent = {
    '0':[-15,-11.5, -5,5],
    '1':[-15,-11.5, -30,30],
    '2':[-15,-11.5, -300,300],
    '3':[-15,-11.5, -1000,1000],
}
fig, axs = plt.subplots(4, figsize=[4,10], sharex=True)
for ii,ax in zip(range(4), axs.flatten()):
    _a = _df[f'log{var1}_noSEC_gr']
    _df_s = _df[_a==float(ii)]
    x= _df_s[f'log{var2}_'+case_orig]
    y= _df_s[var1]
    c = ax.hexbin(x,y, gridsize=40, cmap='Reds', extent=extent[str(ii)])
    cb = fig.colorbar(c, ax=ax)

    ax.set_title('N$_a$ in [10$^{%s}$-10$^{%s}$)'%(ii, (ii+1)))
    ax.set_ylabel('N$_a$: SECTv11_ctrl-noSECTv11_ctrl')
    ax.plot([-15,-11], [0,0])
ax.set_xlabel('log([H2SO4]) from noSECTv11_ctrl')
plt.tight_layout()
plt.show()
