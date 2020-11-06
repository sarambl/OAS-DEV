import matplotlib.colors as colors
# load and autoreload
from IPython import get_ipython
from useful_scit.imps import (np, plt)

from sectional_v2.data_info import get_nice_name_case
from sectional_v2.util.imports import get_averaged_fields
from sectional_v2.util.imports.get_fld_fixed import get_field_fixed
from sectional_v2.util.plot.plot_levlat import plot_levlat_diff, get_cbar_eq_kwargs
from sectional_v2.constants import get_plotpath
from sectional_v2.util.practical_functions import make_folders

# noinspection PyBroadException
try:
    _ipython = get_ipython()
    _magic = _ipython.magic
    _magic('load_ext autoreload')
    _magic('autoreload 2')
except:
    pass
# %%
model = 'NorESM'

startyear = '2008-01'
endyear = '2009-12'
p_level = 1013.
pmin = 850.  # minimum pressure level
avg_over_lev = True  # True#True#False#True
pressure_adjust = True  # Can only be false if avg_over_lev false. Plots particular hybrid sigma lev
p_levels = [1013., 900., 800., 700., 600.]  # used if not avg

# %% [markdown]
# ## Cases

# %%
cases_sec = ['SECTv21_ctrl']
cases_orig = ['noSECTv21_default_dd', 'noSECTv21_ox_ricc_dd']
# cases_orig =['noSECTv21_ox_ricc']

cases = cases_orig + cases_sec

# %%

version = 'v21'
plot_path = get_plotpath('levlat')
filen_base = plot_path + '/_%s' % version
# print(plot_path)
make_folders(plot_path)


# %%
def load_and_plot_diffs(varl, case_ctrl, case_other, start_time, end_time,
                        pressure_coords=True,
                        relative=False,
                        cbar_orient='vertical',
                        asp_ratio=2, subfig_size=3,
                        ncol=None,
                        ylim=None,
                        yscale='log',
                        norm=None
                        ):
    if ylim is None:
        ylim = [1e3, 100]
    cases_dict = get_averaged_fields.get_levlat_cases(cases, varl, start_time, end_time,
                                                      pressure_adjust=pressure_coords)
    _nv = len(varl)
    if ncol is None:
        if _nv > 3:
            ncol = 2
        else:
            ncol = 1
    # noinspection PyUnresolvedReferences
    nrow = int(np.ceil(_nv / ncol))
    figsize = [subfig_size * ncol * asp_ratio, subfig_size * nrow]
    # noinspection PyTypeChecker
    fig, axs = plt.subplots(nrow, ncol, figsize=figsize, sharex=True, sharey=True)
    for ax, var in zip(axs.flat, varl):
        plot_levlat_diff(var, case_ctrl, case_other, cases_dict,
                         cbar_orientation=cbar_orient,
                         relative=relative,
                         ylim=ylim,
                         yscale=yscale,
                         ax=ax,
                         norm=norm)

    return axs


def load_and_plot_diffs_more_cases(varl, cases, case_oth, startyear, endyear,
                                   pressure_adjust=pressure_adjust,
                                   relative=False,
                                   cbar_orientation='vertical',
                                   asp_ratio=2, subfig_size=3,
                                   ylim=None,
                                   yscale='log',
                                   cbar_eq=True,
                                   norm=None):
    if ylim is None:
        ylim = [1e3, 100]
    imp_cases = list(set(cases).union({case_oth}))
    print(imp_cases)
    cases_dic = get_averaged_fields.get_levlat_cases(imp_cases, varl, startyear, endyear,
                                                     pressure_adjust=pressure_adjust)
    ncol = len(cases)
    nrow = len(varl)
    figsize = [subfig_size * ncol * asp_ratio, subfig_size * nrow]
    # noinspection PyTypeChecker
    fig, axs = plt.subplots(nrow, ncol, figsize=figsize, sharex=True, sharey=True)
    for j, var in enumerate(varl):
        saxs = axs[j, :]
        levlat_more_cases_var(var, case_oth, cases, cases_dic, cbar_eq, cbar_orientation, saxs, norm, relative, ylim,
                              yscale)
    return axs


def levlat_more_cases_var(var, case_oth, cases, cases_dic, cbar_eq=True, cbar_orientation='vertical', axs=None,
                          norm=None, relative=False, ylim=None, yscale='log'):
    if ylim is None:
        ylim = [1e3, 100]
    if cbar_eq:
        cba_kwargs = get_cbar_eq_kwargs(cases, case_oth, relative, cases_dic, var)
        if norm is not None:
            del cba_kwargs['vmin']
            del cba_kwargs['vmax']
            del cba_kwargs['robust']

    else:
        cba_kwargs = None
    for i, case_ctrl in enumerate(cases):
        ax = axs[i]
        # if cbar_eq and i<len(cases)-1:
        #    cba_kwargs['add_colorbar']=False
        # else:
        #    cba_kwargs['add_colorbar']=True

        plot_levlat_diff(var, case_ctrl, case_oth, cases_dic,
                         cbar_orientation=cbar_orientation,
                         relative=relative,
                         ylim=ylim,
                         yscale=yscale,
                         ax=ax,
                         norm=norm, **cba_kwargs)


# %%
varlist = ['NCONC01', 'NMR01', 'AWNC_incld', 'AREL_incld']
cbar_orientation = 'vertical'
cases_ctrl = cases_orig
case_oth = cases_sec[0]
ncol = len(cases_ctrl)
nrow = len(varlist)
subfig_size = 2.6
asp_ratio = 1.6
figsize = [subfig_size * ncol * asp_ratio, subfig_size * nrow]
# noinspection PyTypeChecker
fig, axs = plt.subplots(nrow, ncol, figsize=figsize, sharex=True, sharey=True)

norm_dic = dict(
    NCONC01=colors.SymLogNorm(vmin=-1e3, vmax=1e3, linthresh=10),
    NMR01=colors.SymLogNorm(vmin=-10, vmax=10, linthresh=.1),
    AWNC_incld=colors.SymLogNorm(vmin=-20, vmax=20, linthresh=.1),
    AREL_incld=colors.SymLogNorm(vmin=-5, vmax=5, linthresh=.1)
)
cases_dic = get_averaged_fields.get_levlat_cases(cases, varlist, startyear, endyear,
                                                 pressure_adjust=pressure_adjust)

for j, var in enumerate(varlist):
    saxs = axs[j, :]
    levlat_more_cases_var(var, case_oth, cases_ctrl, cases_dic, cbar_eq=True,
                          cbar_orientation='vertical',
                          axs=saxs,
                          norm=norm_dic[var],
                          relative=False,
                          ylim=[1e3, 200],
                          yscale='log')
for ax in axs.flatten():
    ax.set_ylabel('')
    ax.set_xlabel('')
for ax in axs[:, 0]:
    ax.set_ylabel('Pressure [hPa]')
for ax in axs[-1, :]:
    ax.set_xlabel('Latitude [$^\circ$N]')
fig.tight_layout()
fn = filen_base + f'N_clouds_{case_oth}' + '_'.join(cases_ctrl) + f'{startyear}-{endyear}'
plt.savefig(fn + '.pdf')
plt.savefig(fn + '.png')
plt.show()
# %%

# correlation:
# %%
var_subl = ['NCONC01','AWNC_incld','AREL_incld','NMR01','HYGRO01']
cases_dic = {}
for case in cases:

    dummy = get_field_fixed(case,
                        var_subl,
                        startyear, endyear,
                        pressure_adjust=pressure_adjust)
    cases_dic[case]=dummy.copy()
#ds_constants = import_constants(case,
#                                path=raw_data_path,
#                                model=model)
#dummy = xr.merge([dummy, ds_constants])
# %%
from useful_scit.util.zarray import corr
from sectional_v2.util.plot.plot_levlat import plot_levlat
# %%
cases_ctrl=cases_orig
case_oth = cases_sec[0]


def corr_plt():
    # %%
    var_c = 'AWNC_incld'
    varl_to = ['NCONC01','NMR01']
    cbar_orientation = 'vertical'
    cases_ctrl = cases_orig
    case_oth = cases_sec[0]
    ncol = len(cases_ctrl)
    nrow = len(varlist)
    subfig_size = 2.6
    asp_ratio = 1.6
    figsize = [subfig_size * ncol * asp_ratio, subfig_size * nrow]
    # noinspection PyTypeChecker
    fig, axs = plt.subplots(nrow, ncol, figsize=figsize, sharex=True, sharey=True)

    norm_dic = dict(
        NCONC01=colors.SymLogNorm(vmin=-1e3, vmax=1e3, linthresh=10),
        NMR01=colors.SymLogNorm(vmin=-10, vmax=10, linthresh=.1),
        AWNC_incld=colors.SymLogNorm(vmin=-20, vmax=20, linthresh=.1),
        AREL_incld=colors.SymLogNorm(vmin=-5, vmax=5, linthresh=.1)
    )
    for j, var in enumerate(varl_to):
        saxs = axs[j, :]
        for i, case in enumerate(cases_ctrl):
            ax = saxs[i]
            _vars = [var, var_c]
            _ds = cases_dic[case_oth][_vars]- cases_dic[case][_vars]
            _da_corr = corr(_ds[var],_ds[var_c], dim=['time','lon'])
            nn_ctrl = get_nice_name_case(case)
            nn_oth = get_nice_name_case(case_oth)
            title = f'corr($\Delta V_1, \Delta V_2$), 1={nn_oth}, 2={nn_ctrl}'
            plot_levlat(ax, 'RdBu_r', _da_corr, '', [1e3,200],
                        cbar_orientation='vertical',
                        #ax=ax,
                        #norm=norm_dic[var],
                        #relative=False,
                        #ylim=[1e3, 200],
                        yscale='log')


    for ax in axs.flatten():
        ax.set_ylabel('')
        ax.set_xlabel('')
    for ax in axs[:, 0]:
        ax.set_ylabel('Pressure [hPa]')
    for ax in axs[-1, :]:
        ax.set_xlabel('Latitude [$^\circ$N]')
    fig.tight_layout()
    fn = filen_base + f'N_clouds_{case_oth}' + '_'.join(cases_ctrl) + f'{startyear}-{endyear}'
    #plt.savefig(fn + '.pdf')
    #plt.savefig(fn + '.png')
    plt.show()
    # %%