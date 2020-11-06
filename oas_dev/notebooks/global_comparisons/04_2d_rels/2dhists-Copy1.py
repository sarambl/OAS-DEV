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

from sectional_v2.data_info import get_nice_name_case
from sectional_v2.util.imports.import_fields_xr_v2 import import_constants
from sectional_v2.util.imports.get_fld_fixed import get_field_fixed
from useful_scit.imps import (np, xr, plt, pd)

# load and autoreload
from IPython import get_ipython

# noinspection PyBroadException
from sectional_v2.util.naming_conventions import var_info
from sectional_v2.util.naming_conventions.var_info import get_fancy_var_name

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

from sectional_v2.constants import get_plotpath
from sectional_v2.util.practical_functions import make_folders

plot_path = get_plotpath('comparison') + '/scatter/'
print(plot_path)
make_folders(plot_path)

# %%
from_time = '2008-01'
to_time = '2008-12'
pmin = 850.  # minimum pressure level
avg_over_lev = True  # True#True#False#True
pressure_adjust = False  # Can only be false if avg_over_lev false. Plots particular hybrid sigma lev
#if avg_over_lev:
#    pressure_adjust = True
p_levels = [1013.,900., 800., 700., 600.]  # used if not avg

lev_lim =0.
# %%

cases_sec = ['SECTv21_ctrl']#, 'PD_SECT_CHC7_diurnal']  # Sect ac eq.20, corr NPF diam, fxdt, vdiam, 1.5xBVOC']
cases_orig = ['noSECTv21_ox_ricc_dd']  # , 'Original eq.18','Original eq.20, 1.5xBVOC','Original eq.20, rednuc']
#cases_orig = ['noSECTv21_default_dd']  # , 'Original eq.18','Original eq.20, 1.5xBVOC','Original eq.20, rednuc']
cases = cases_orig + cases_sec

# %%
# %%
var_subl = ['NCONC01', 'NMR01','AWNC_incld','AREL_incld']#,'SOA_NA','SO4_NA']

# %%
var_subl = ['NCONC01', 'H2SO4','SOA_LV','N_AER','COAGNUCL','NUCLRATE','GR','FORMRATE']#,'SOA_NA','SO4_NA']

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

# %% [markdown]
# for var in ['H2SO4','SOA_LV']:
#     for case in cases:
#         _ds = cases_dic[case]
#         _ds.load()
#         if _ds[var].units=='mol/mol':
#             _ds[var] = _ds[var]*1e12
#             _ds[var].attrs['units']='ppt'
# var = var1

# %%
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
ds_diff


# %%
ds_diff['NCONC01'].mean('time').isel(lev=-1).plot()
plt.show()
# %%
import seaborn as sns

# %%
v=5
dims = tuple(ds_diff['NCONC01'].dims)
_df = ds_diff['NCONC01'].stack(ind=dims).to_dataframe()
_df = _df[(_df['NCONC01']>-100)&(_df['NCONC01']<100)]
xbins = mk_bins(v, vmax = 1e3, nr_bins=20, scale='symlog')

sns.distplot( _df['NCONC01'], hist_kws={"histtype": "step", "linewidth": 3,
                            "alpha": 1, "color": "g"}, bins=xbins)

plt.xscale('symlog', linthreshx=v*2, symscale=v)


# %%
import matplotlib.colors as colors


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
    if yscale=='symlog':
        ax.set_yscale('symlog', linthreshy=ylim[0], linscaley=ylim[0]/10,subsy=[2,3,4,5,6,7,8,9])
        yt = ax.get_yticks()
        ml = np.abs(yt[yt!=0]).min()
        ytl = yt
        ytl[(yt==ml)|(yt==-ml)]=None
        ax.set_yticks(ticks=yt)#[y for y in yt if y!=0])#,
        ax.set_yticklabels(ytl)#[-1e2,-1e1,-1e0,1e0,1e1,1e2])
    if xscale=='symlog':
        ax.set_xscale('symlog', linthreshy=xlim[0], linscaley=xlim[0]/10,subsy=[2,3,4,5,6,7,8,9])
        xt = ax.get_xticks()
        ml = np.abs(xt[xt!=0]).min()
        xtl = xt
        xtl[(xt==ml)|(xt==-ml)]=None
        ax.set_xticks(ticks=xt)#[y for y in yt if y!=0])#,
        ax.set_xticklabels(xtl)#[-1e2,-1e1,-1e0,1e0,1e1,1e2])

    elif xscale !='linear':
        ax.set_xscale('log')
    return ax
    #plt.show()

def mk_bins(v, vmax = 1e3, nr_bins=20, scale='symlog'):
    print(scale)
    if scale=='symlog':
        ybins = np.geomspace(v, vmax, int(nr_bins)/2)
        ybins = ybins - ybins[0]
        ybins = [*-ybins[::-1], *ybins[1:]]
    elif scale=='log':
        ybins = np.geomspace(v, vmax, nr_bins)
    elif scale=='neglog':
        ybins = -np.geomspace(v, vmax, nr_bins)[::-1]
    elif scale=='linear':
        ybins=np.linspace(v, vmax, nr_bins)

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

# %% [markdown]
# ## Main plot:

# %%
fig, axs = plt.subplots(1,2, figsize=[7,4])#, sharey=True)#, constrained_layout=True)
var_xl = ['NCONC01','NMR01']
lev_min=600.
_ds = ds_diff.sel(lev=slice(lev_min, None))
var_diff = 'AWNC_incld'
xlims = {
    'NCONC01':[-1e3,1e3],
    'NMR01':[-70,70],
    'AWNC_incld':[-100,100],
    'AREL_incld':[-100,100],
    
#    'NUCLRATE' : [1.e-6,10],
#    'H2SO4' : [1.e-3,1],
#    'SOA_LV' : [1.e-5,1],
#    'GR' : [1.e-3,1],
#    'N_AER':[1e0,1e4],
#    'NCONC01':[5e-1,1e3],
#    'COAGNUCL':[1e-8,1e-3],
}
xlims = {
    'NCONC01':[5,1e3],
    'NMR01':[.1,70],
    'AWNC_incld':[1,50],
    'AREL_incld':[1,100],
    
#    'NUCLRATE' : [1.e-6,10],
#    'H2SO4' : [1.e-3,1],
#    'SOA_LV' : [1.e-5,1],
#    'GR' : [1.e-3,1],
#    'N_AER':[1e0,1e4],
#    'NCONC01':[5e-1,1e3],
#    'COAGNUCL':[1e-8,1e-3],
}

for var,ax in zip(var_xl, axs.flatten()):
    print(var)
    xlim = xlims[var]
    h = _plt_2dhist(_ds,var, var_diff,ylim = xlims[var_diff],
                nr_bins=40, xscale='symlog', yscale='symlog',
                xlim=xlim, ax=ax)
    uni = var_info.get_fancy_unit_xr(_ds[var],
                                     var)
    ax.set_xlabel(f'{get_fancy_var_name(var)} [{uni}]')
    ax.plot(xlim,[0,0], linewidth=.5, c='k')

uni = var_info.get_fancy_unit_xr(_ds[var_diff],
                           var_diff)
fvar_diff = get_fancy_var_name(var_diff)
ylab = f'$\Delta${fvar_diff} [{uni}]'
for ax in axs:#[:,0]:
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
ds_diff['FREQL'].max(['time','lon']).plot()

# %%
fig, axs = plt.subplots(1,2, figsize=[7,4])#, sharey=True)#, constrained_layout=True)
var_xl = ['NCONC01','NMR01']
lev_min=700
_ds = ds_diff.sel(lev=slice(lev_min, None)).where(ds_diff['NCONC01']<-1)
var_diff = 'AWNC_incld'
xlims = {
    'NCONC01':[-10,10],
    'NMR01':[-3,3],
    'AWNC_incld':[-3,3],
    'AREL_incld':[-100,100],
    
#    'NUCLRATE' : [1.e-6,10],
#    'H2SO4' : [1.e-3,1],
#    'SOA_LV' : [1.e-5,1],
#    'GR' : [1.e-3,1],
#    'N_AER':[1e0,1e4],
#    'NCONC01':[5e-1,1e3],
#    'COAGNUCL':[1e-8,1e-3],
}
#xlims = {
#    'NCONC01':[10,1e2],
#    'NMR01':[1,20],
#    'AWNC_incld':[1,20],
#    'AREL_incld':[1,100],
    
#    'NUCLRATE' : [1.e-6,10],
#    'H2SO4' : [1.e-3,1],
#    'SOA_LV' : [1.e-5,1],
#    'GR' : [1.e-3,1],
#    'N_AER':[1e0,1e4],
#    'NCONC01':[5e-1,1e3],
#    'COAGNUCL':[1e-8,1e-3],
#}

for var,ax in zip(var_xl, axs.flatten()):
    print(var)
    xlim = xlims[var]
    #var = f'{var}_{case_orig}'
    h = _plt_2dhist(_ds,var, var_diff,ylim = xlims[var_diff],
                nr_bins=40, xscale='linear', yscale='linear',
                xlim=xlim, ax=ax)
    uni = var_info.get_fancy_unit_xr(_ds[var],
                                     var)
    ax.set_xlabel(f'{get_fancy_var_name(var)} [{uni}]')
    ax.plot(xlim,[0,0], linewidth=.5, c='k')

uni = var_info.get_fancy_unit_xr(_ds[var_diff],
                           var_diff)
fvar_diff = get_fancy_var_name(var_diff)
ylab = f'$\Delta${fvar_diff} [{uni}]'
for ax in axs:#[:,0]:
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
cond = (ds_diff['NCONC01']<1e10)
x = ds_diff['NMR01'].sel(lev=slice(lev_min, None)).where(cond).to_dataframe()
y = ds_diff['AWNC_incld'].sel(lev=slice(lev_min, None)).where(cond).to_dataframe()

# %%
lev= x.reset_index(['lev'])['lev']


# %%
plt.scatter(x,y, marker='x', alpha=.2, s=.5, c=lev)
plt.ylim([-500,500])
plt.yscale('symlog', linthresh=10)
plt.colorbar()

# %%
cond = (ds_diff['NCONC01']<1e10)
x = ds_diff['NCONC01'].sel(lev=slice(lev_min, None)).where(cond).to_dataframe()
y = ds_diff['AWNC_incld'].sel(lev=slice(lev_min, None)).where(cond).to_dataframe()

# %%
lev= x.reset_index(['lev'])['lev']


# %%
from useful_scit.util.zarray import corr,cov

# %%
lev_min=200
x = ds_diff[f'NCONC01_{case_orig}']#.sel(lev=slice(lev_min, None))#.where(cond).to_dataframe()
y = ds_diff[f'AWNC_incld_{case_orig}']#.sel(lev=slice(lev_min, None))#.where(cond).to_dataframe()
xa = corr(x,y, dim =['time','lon'])

xa.plot(ylim=[1e3,200])

# %%
lev_min=200
x = ds_diff[f'NMR01_{case_orig}']#.sel(lev=slice(lev_min, None))#.where(cond).to_dataframe()
y = ds_diff[f'AWNC_incld_{case_orig}']#.sel(lev=slice(lev_min, None))#.where(cond).to_dataframe()
xa = corr(x,y, dim =['time','lon'])

xa.plot(ylim=[1e3,200])

# %%
lev_min=200
x = ds_diff['NCONC01']#.sel(lev=slice(lev_min, None))#.where(cond).to_dataframe()
y = ds_diff['AWNC_incld']#.sel(lev=slice(lev_min, None))#.where(cond).to_dataframe()
xa = corr(x,y, dim =['time','lon'])

xa.plot(ylim=[1e3,200])

# %%
lev_min=200
x = ds_diff['NMR01']#.sel(lev=slice(lev_min, None))#.where(cond).to_dataframe()
y = ds_diff['AWNC_incld']#.sel(lev=slice(lev_min, None))#.where(cond).to_dataframe()
xa = corr(x,y, dim =['time','lon'])

# %%
xa.plot(ylim=[1e3,200])

# %%
plt.scatter(x,y, marker='x', alpha=.2, s=.5, c=lev)
plt.ylim([-200,200])
plt.xlim([-1000,1000])
#plt.yscale('symlog', linthresh=10)
#plt.xscale('symlog', linthresh=10)
plt.colorbar()

# %%
fig, axs = plt.subplots(1,2, figsize=[7,4])#, sharey=True)#, constrained_layout=True)
var_xl = ['NCONC01','NMR01']
lev_min=600.
_ds = ds_diff.sel(lev=slice(lev_min, None))
var_diff = 'AREL_incld'
xlims = {
    'NCONC01':[-1e3,1e3],
    'NMR01':[-70,70],
    'AWNC_incld':[-100,100],
    'AREL_incld':[-100,100],
    
#    'NUCLRATE' : [1.e-6,10],
#    'H2SO4' : [1.e-3,1],
#    'SOA_LV' : [1.e-5,1],
#    'GR' : [1.e-3,1],
#    'N_AER':[1e0,1e4],
#    'NCONC01':[5e-1,1e3],
#    'COAGNUCL':[1e-8,1e-3],
}
xlims = {
    'NCONC01':[5,1e3],
    'NMR01':[.1,70],
    'AWNC_incld':[1,50],
    'AREL_incld':[.1,20],
    
#    'NUCLRATE' : [1.e-6,10],
#    'H2SO4' : [1.e-3,1],
#    'SOA_LV' : [1.e-5,1],
#    'GR' : [1.e-3,1],
#    'N_AER':[1e0,1e4],
#    'NCONC01':[5e-1,1e3],
#    'COAGNUCL':[1e-8,1e-3],
}

for var,ax in zip(var_xl, axs.flatten()):
    print(var)
    xlim = xlims[var]
    h = _plt_2dhist(_ds,var, var_diff,ylim = xlims[var_diff],
                nr_bins=40, xscale='symlog', yscale='symlog',
                xlim=xlim, ax=ax)
    uni = var_info.get_fancy_unit_xr(_ds[var],
                                     var)
    ax.set_xlabel(f'{get_fancy_var_name(var)} [{uni}]')
    ax.plot(xlim,[0,0], linewidth=.5, c='k')

uni = var_info.get_fancy_unit_xr(_ds[var_diff],
                           var_diff)
fvar_diff = get_fancy_var_name(var_diff)
ylab = f'$\Delta${fvar_diff} [{uni}]'
for ax in axs:#[:,0]:
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

# %% [markdown]
# # Abs values
#

# %%
ds_diff


# %%
fig, axs = plt.subplots(3,2, figsize=[6.4,7.1], sharey=True)#, constrained_layout=True)
var_xl = ['NUCLRATE','GR','H2SO4','SOA_LV','NCONC01','COAGNUCL']
lev_min=800
_ds = ds_diff#.sel(lev=slice(lev_min, None))
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
    h = _plt_2dhist(_ds,f'{var}_{case_orig}', f'{var}_{var_diff}',
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

# %% [markdown]
# ## masked by height

# %%
fig, axs = plt.subplots(3,2, figsize=[6.4,7.1], sharey=True)#, constrained_layout=True)
var_xl = ['NUCLRATE','GR','H2SO4','SOA_LV','NCONC01','COAGNUCL']
lev_min=800
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
lev_min=800
_ds = ds_diff.sel(lev=slice(None,lev_min))
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

# %% [markdown]
# ## Mask by COAGS?

# %%
fig, axs = plt.subplots(3,2, figsize=[6.4,7.1], sharey=True)#, constrained_layout=True)
var_xl = ['NUCLRATE','GR','H2SO4','SOA_LV','NCONC01','COAGNUCL']
lev_min=950
_coag_ctrl = ds_diff[f'COAGNUCL_{case_orig}']
_ds = ds_diff.where(_coag_ctrl>1e-6)#sel(lev=slice(lev_min, None))
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
lev_min=950
_coag_ctrl = ds_diff[f'COAGNUCL_{case_orig}']
_gr_ctrl = ds_diff[f'GR_{case_orig}']
_ds = ds_diff.where((_coag_ctrl<1e-6) )#sel(lev=slice(lev_min, None))
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
                nr_bins=40,#yscale='linear',ylim=[-50,50],
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

# %% jupyter={"outputs_hidden": true}
ds_diff

# %%
fig, axs = plt.subplots(3,2, figsize=[6.4,7.1], sharey=True)#, constrained_layout=True)
var_xl = ['FORMRATE','GR','H2SO4','SOA_LV','NCONC01','COAGNUCL']
lev_min=950
_coag_ctrl = ds_diff[f'COAGNUCL_{case_orig}']
_gr_ctrl = ds_diff[f'GR_{case_orig}']
_ds = ds_diff#.where((_coag_ctrl<1e-6))#sel(lev=slice(lev_min, None))
var_diff = 'NCONC01'
xlims = {
    'NUCLRATE' : [1.e-6,10],
    'H2SO4' : [1.e-3,1],
    'SOA_LV' : [1.e-5,1],
    'GR' : [1.e-3,1],
    'N_AER':[1e0,1e4],
    'NCONC01':[5e-1,1e3],
    'COAGNUCL':[1e-8,1e-3],
    'FORMRATE':[1.e-8,1],
}
for var,ax in zip(var_xl, axs.flatten()):
    print(var)
    xlim = xlims[var]
    h = _plt_2dhist(_ds,f'{var}_{case_orig}', var_diff,
                nr_bins=40,#yscale='linear',ylim=[-50,50],
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
var_xl = ['FORMRATE','GR','H2SO4','SOA_LV','NCONC01','COAGNUCL']
lev_min=950
_coag_ctrl = ds_diff[f'COAGNUCL_{case_orig}']
_gr_ctrl = ds_diff[f'GR_{case_orig}']
_ds = ds_diff.where((_coag_ctrl<1e-6))#sel(lev=slice(lev_min, None))
var_diff = 'NCONC01'
xlims = {
    'NUCLRATE' : [1.e-6,10],
    'H2SO4' : [1.e-3,1],
    'SOA_LV' : [1.e-5,1],
    'GR' : [1.e-3,1],
    'N_AER':[1e0,1e4],
    'NCONC01':[5e-1,1e3],
    'COAGNUCL':[1e-8,1e-3],
    'FORMRATE':[1.e-8,1],
}
for var,ax in zip(var_xl, axs.flatten()):
    print(var)
    xlim = xlims[var]
    h = _plt_2dhist(_ds,f'{var}_{case_orig}', var_diff,
                nr_bins=40,#yscale='linear',ylim=[-50,50],
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
var_xl = ['FORMRATE','H2SO4','SOA_LV','N_AER','COAGNUCL','soa/h2SO4']#_{case_orig}']
lev_min=950
_coag_ctrl = ds_diff[f'H2SO4_{case_orig}']
_gr_ctrl = ds_diff[f'SOA_LV_{case_orig}']
_ds = ds_diff.where((_coag_ctrl<_gr_ctrl))#sel(lev=slice(lev_min, None))
_ds[f'soa/h2SO4_{case_orig}'] = _ds[f'SOA_LV_{case_orig}']/_ds[f'H2SO4_{case_orig}']
_ds[f'soa/h2SO4'] = _ds[f'SOA_LV']/_ds[f'H2SO4']
var_diff = 'NCONC01'
xlims = {
    'NUCLRATE' : [1.e-6,10],
    'H2SO4' : [1.e-3,1],
    'SOA_LV' : [1.e-5,1],
    'GR' : [1.e-3,1],
    'N_AER':[1e0,1e4],
    'NCONC01':[5e-1,1e3],
    'COAGNUCL':[1e-8,1e-3],
    'FORMRATE':[1.e-6,10],
    'soa/h2SO4': [1e-4,1e4]
}
for var,ax in zip(var_xl, axs.flatten()):
    print(var)
    xlim = xlims[var]
    h = _plt_2dhist(_ds,f'{var}_{case_orig}', var_diff,
                nr_bins=40,#yscale='linear',ylim=[-50,50],
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
var_xl = ['FORMRATE','GR','H2SO4','SOA_LV','NCONC01','COAGNUCL']
lev_min=950
_coag_ctrl = ds_diff[f'H2SO4_{case_orig}']
_gr_ctrl = ds_diff[f'SOA_LV_{case_orig}']
_ds = ds_diff.where((_coag_ctrl>2*_gr_ctrl))#sel(lev=slice(lev_min, None))
var_diff = 'NCONC01'
xlims = {
    'NUCLRATE' : [1.e-6,10],
    'H2SO4' : [1.e-3,1],
    'SOA_LV' : [1.e-5,1],
    'GR' : [1.e-3,1],
    'N_AER':[1e0,1e4],
    'NCONC01':[5e-1,1e3],
    'COAGNUCL':[1e-8,1e-3],
    'FORMRATE':[1.e-6,10],
}
for var,ax in zip(var_xl, axs.flatten()):
    print(var)
    xlim = xlims[var]
    h = _plt_2dhist(_ds,f'{var}_{case_orig}', var_diff,
                nr_bins=40,#yscale='linear',ylim=[-50,50],
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

# %% [markdown]
# # swich base case:

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
    'COAGNUCL':[1e-8,1e-3],
}
for var,ax in zip(var_xl, axs.flatten()):
    print(var)
    xlim = xlims[var]
    h = _plt_2dhist(_ds,f'{var}_{case_sec}', var_diff,
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
suptit =f'{fvar_diff}$(m_1)-${fvar_diff}$(m_2)$ vs. $X(m_1)$ \n for $m_1=${get_nice_name_case(case_sec)}, $m_2=${get_nice_name_case(case_orig)}'
fig.subplots_adjust(hspace=.5, wspace=0.1)#,top=0.8, )
stit = fig.suptitle(suptit,  fontsize=12, y=.98)
fn = plot_path + f'2dhist_flip_{case_orig}_{case_sec}.'
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

# %%

# %%
sns.distplot( _ds['NCONC01'].to_dataframe(), hist_kws={"histtype": "step", "linewidth": 3,
                            "alpha": 1, "color": "g"})

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
lev_min=900
_ds = ds_diff#.sel(lev=slice( lev_min,None))
var_diff = 'H2SO4'
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
#fig.savefig(fn+'pdf',bbox_extra_artists=(stit,), bbox_inches='tight')
#fig.savefig(fn+'png',bbox_extra_artists=(stit,), bbox_inches='tight')
plt.show()
