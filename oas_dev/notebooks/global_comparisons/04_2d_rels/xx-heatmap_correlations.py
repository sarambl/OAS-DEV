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

lev_lim =100.
# %%

cases_sec = ['SECTv21_ctrl_koagD']#, 'PD_SECT_CHC7_diurnal']  # Sect ac eq.20, corr NPF diam, fxdt, vdiam, 1.5xBVOC']
cases_orig = ['noSECTv21_ox_ricc_dd']  # , 'Original eq.18','Original eq.20, 1.5xBVOC','Original eq.20, rednuc']
#cases_orig = ['noSECTv21_default_dd']  # , 'Original eq.18','Original eq.20, 1.5xBVOC','Original eq.20, rednuc']
cases = cases_orig + cases_sec

# %%
var_subl = ['NCONC01', 'H2SO4','SOA_LV','N_AER','COAGNUCL','NUCLRATE','GR','PBLH']#,'SOA_NA','SO4_NA']

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
        if _ds[var].units=='mol/mol':
            _ds[var] = _ds[var]*1e12
            _ds[var].attrs['units']='ppt'
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
    ds_diff[f'log{var}_diff'] = ds_diff[f'log{var}_{case_sec}'] -ds_diff[f'log{var}_{case_orig}']

# %%
ds_diff['NCONC01'].mean('time').isel(lev=-1).plot()
plt.show()
# %%
ds_diff.to_netcdf('test.nc')

# %%
ds_diff= xr.open_dataset('test.nc')

# %%
vars_inc = ['NCONC01','logNCONC01_noSECTv21_ox_ricc_dd','logGR_noSECTv21_ox_ricc_dd','logNUCLRATE_noSECTv21_ox_ricc_dd',
            'logCOAGNUCL_noSECTv21_ox_ricc_dd','logH2SO4_noSECTv21_ox_ricc_dd',
           'logSOA_LV_noSECTv21_ox_ricc_dd']
vars_an =vars_inc[1:]
#['NCONC01_noSECTv21_ox_ricc_dd','logGR_noSECTv21_ox_ricc_dd','logNUCLRATE_noSECTv21_ox_ricc_dd',
#            'logCOAGNUCL_noSECTv21_ox_ricc_dd','logH2SO4_noSECTv21_ox_ricc_dd',
#           'logSOA_LV_noSECTv21_ox_ricc_dd']
_da = ds_diff[vars_inc]#.to_array()#,sample_dims=['test'])

# %%
_das = _da.stack(z = ('lat','lon','lev','time'))

# %%
_df = _das.to_dataframe()

# %%
_sss = _df['logNCONC01_noSECTv21_ox_ricc_dd']==np.inf

_sss.unique()


# %%
def clean_nan(df):
    for col in df.columns:
        df=df[df[col].notnull()]
        df = df[df[col] != np.inf]
        df = df[df[col] != -np.inf]
    return df
_df = clean_nan(_df)

# %%
_df['NCONC_cat'] = pd.qcut(_df['NCONC01'], q=4)

# %%
from sklearn.preprocessing import StandardScaler
x = _df.loc[:, vars_an].values
x = StandardScaler().fit_transform(x) # normalizing the features

# %%
x.shape

# %%
x.shapecase_orig


# %%
np.mean(x),np.std(x)

# %%
feat_cols = [va.split('_')[0] for va in vars_an]
#]]#'feature'+str(i) for i in range(x.shape[1])]


# %%
feat_cols

# %%
normalised_breast = pd.DataFrame(x,columns=feat_cols)


# %%
normalised_breast#.std()


# %%
import seaborn as sns
for col in normalised_breast.columns:
    sns.distplot(normalised_breast[col])
    plt.title(col)
    plt.show()

# %%
mean_vec = np.mean(x, axis=0)
cov_mat = (x - mean_vec).T.dot((x - mean_vec)) / (x.shape[0]-1)

# %%
cov_mat

# %%
corrMatrix = _df.corr()
sns.heatmap(corrMatrix, annot=True, cmap='RdBu_r', vmin=-.9, vmax=.9)
plt.show()

# %% [markdown]
# ## Above q75

# %%
va = _df.logNCONC01_noSECTv21_ox_ricc_dd
ma = va>va.quantile(.75) 
corrMatrix = _df[ma].corr()
sns.heatmap(corrMatrix, annot=True, cmap='RdBu_r', vmin=-.9, vmax=.9)
plt.show()

# %% [markdown]
# ##  q25

# %%
va = _df.logNCONC01_noSECTv21_ox_ricc_dd
ma = va<va.quantile(.25) 
corrMatrix = _df[ma].corr()
sns.heatmap(corrMatrix, annot=True, cmap='RdBu_r', vmin=-.9, vmax=.9)
plt.show()

# %%
corrMatrix.loc['NCONC01']

# %%
va = 10**_df.logNCONC01_noSECTv21_ox_ricc_dd
qs = va.quantile([0,.10,.20,.30,.40,.50,.60,.70,.80,.85,.90,.95,1.00])
quant_corr = pd.DataFrame(index=vars_inc)
for i in range(len(qs)-1):
    ma = (qs.iloc[i]<=va)&(va<qs.iloc[i+1])
    c_m = _df[ma].corr()
    quant_corr[qs.iloc[i+1]] = c_m.loc['NCONC01']
    #quant_corr.append_row
    
#ma = va<va.quantile(.25) 


# %%
sns.heatmap(quant_corr.transpose(), annot=True, cmap='RdBu_r', vmin=-.9, vmax=.9)
plt.show()

# %%
fig, ax = plt.subplots(1,figsize=[10,10])
quant_corr.transpose().plot(ax = ax, marker='*')
ax.set_ylabel('corr Na, X ')

# %%
from sklearn.decomposition import PCA
pca_breast = PCA(n_components=3)
principalComponents_breast = pca_breast.fit_transform(x)

# %%
from sklearn.decomposition import PCA
pca_breast = PCA(n_components=0.9)#n_components=2)
principalComponents_breast = pca_breast.fit(x)

# %%
pca_breast.n_components_


# %%
pca_breast.explained_variance_ratio_


# %%
pca_breast.components_[0]

# %%
pca_breast.components_[1]

# %% jupyter={"outputs_hidden": true}
for i,v in enumerate(vars_an):
    print(v,i)
    for j,v2 in enumerate(vars_an):
        print([pca_breast.components_[0][i],pca_breast.components_[0][j]])
        print([pca_breast.components_[1][i],pca_breast.components_[1][j]])
        print([pca_breast.components_[2][i],pca_breast.components_[2][j]])
        
        plt.plot(pca_breast.components_[0][i],pca_breast.components_[0][j], label='pca0', marker='*' )
        plt.plot(pca_breast.components_[1][i],pca_breast.components_[1][j], label='pca1', marker='*' )
        plt.plot(pca_breast.components_[2][i],pca_breast.components_[2][j], label='pca2', marker='*' )
        plt.ylabel(v2)
        plt.xlabel(v)
        plt.show()

# %%
principal_breast_Df = pd.DataFrame(data = principalComponents_breast
             , columns = ['principal component 1', 'principal component 2','principal component 3'])

# %%
principal_breast_Df.tail()

# %%
print('Explained variation per principal component: {}'.format(pca_breast.explained_variance_ratio_))


# %%
pca_breast.components_

# %%
plt.figure()
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Principal Component - 1',fontsize=20)
plt.ylabel('Principal Component - 2',fontsize=20)
plt.title("Principal Component Analysis of Breast Cancer Dataset",fontsize=20)
targets = _df['NCONC_cat'].unique()
colors = ['r','b','y','g']
#colors = ['r', 'g']
#plt.scatter(principal_breast_Df['principal component 1']
#            , principal_breast_Df['principal component 2'], c=_df['NCONC01'])
for target, color in zip(targets,colors):
    indicesToKeep = _df['NCONC_cat'] == target
    plt.scatter(principal_breast_Df.loc[indicesToKeep.values, 'principal component 1']
               , principal_breast_Df.loc[indicesToKeep.values, 'principal component 2'], c = color, alpha=0.3, s = 2)
plt.legend(targets,prop={'size': 15})

# %%

# %%
plt.figure()
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Principal Component - 1',fontsize=20)
plt.ylabel('Principal Component - 2',fontsize=20)
plt.title("Principal Component Analysis of Breast Cancer Dataset",fontsize=20)
targets = _df['NCONC_cat'].unique()
colors = ['r','b','y','g']
#colors = ['r', 'g']
#plt.scatter(principal_breast_Df['principal component 1']
#            , principal_breast_Df['principal component 2'], c=_df['NCONC01'])
for target, color in zip(targets,colors):
    indicesToKeep = _df['NCONC_cat'] == target
    plt.scatter(principal_breast_Df.loc[indicesToKeep.values, 'principal component 1']
               , principal_breast_Df.loc[indicesToKeep.values, 'principal component 2'], c = color)#, s = 50)
    plt.title(target)
    plt.show()
plt.legend(targets,prop={'size': 15})

# %%
_df['NCONC01'].describe()#.mi()

# %%

# %%
_das.values

# %%
np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]]).shape

# %%
from sklearn.datasets import load_breast_cancer
breast = load_breast_cancer()
breast_data = breast.data
breast_labels = breast.target
labels = np.reshape(breast_labels,(569,1))
final_breast_data = np.concatenate([breast_data,labels],axis=1)


# %%
import numpy as np
from sklearn.decomposition import PCA
X = _das.values.transpose()# np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

pca = PCA(n_components=2)
pca.fit(X)
PCA(n_components=2)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)


# %%
pca.components_

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
    ax.set_yscale('symlog', linthreshy=ylim[0], linscaley=ylim[0]/10,subsy=[2,3,4,5,6,7,8,9])
    yt = ax.get_yticks()
    ml = np.abs(yt[yt!=0]).min()
    ytl = yt
    ytl[(yt==ml)|(yt==-ml)]=None
    ax.set_yticks(ticks=yt)#[y for y in yt if y!=0])#,
    ax.set_yticklabels(ytl)#[-1e2,-1e1,-1e0,1e0,1e1,1e2])

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
def _plt_tmp(_ds,axs,var_xl, var_diff, xlims):
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
    
    return 


# %%
X


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

_plt_tmp(_ds,axs,var_xl, var_diff, xlims)
fn = plot_path + f'2dhist_{case_orig}_{case_sec}.'
fig.savefig(fn+'pdf',bbox_extra_artists=(stit,), bbox_inches='tight')
fig.savefig(fn+'png',bbox_extra_artists=(stit,), bbox_inches='tight')
plt.show()
print(fn)
