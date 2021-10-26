# %%
from oas_dev.util.plot.plot_maps import fix_axis4map_plot, subplots_map, plt_map
from oas_dev.util.slice_average.avg_pkg import yearly_mean_dic
from oas_dev.util.imports import get_averaged_fields

# load and autoreload
from useful_scit.imps import *

from oas_dev.util.slice_average.significance import get_significance_map_paired_monthly

log.ger.setLevel(log.log.INFO)
# %%
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
endyear = '2010-12'
p_level=1013.
pmin = 850.  # minimum pressure level
avg_over_lev = True  # True#True#False#True
pressure_adjust = True  # Can only be false if avg_over_lev false. Plots particular hybrid sigma lev
if avg_over_lev:
    pressure_adjust = True
p_levels = [1013.,900., 800., 700., 600.]  # used if not avg
# %%
cases_sec = ['SECTv21_ctrl_koagD']#'SECTv21_ctrl',,'SECTv21_ctrl_def']
cases_orig = ['noSECTv21_default_dd', 'noSECTv21_ox_ricc_dd']
cases = cases_sec + cases_orig
# %%
areas = ['Global','notLand','landOnly', 'Polar N','Polar S','Pacific','Amazonas and surroundings']
# %%
varl = ['ACTNL_incld', 'ACTREL_incld','TGCLDCWP', 'NCFT_Ghan', 'HYGRO01',
        'SOA_NAcondTend', 'SO4_NAcondTend',
        'cb_SOA_NA', 'cb_SO4_NA',
        'HYGRO01','cb_SOA_LV','cb_H2SO4',
        'SO2','DMS','isoprene','monoterp',
        'N_AER','NCONC01', 'NMR01','GR','NUCLRATE','FORMRATE',
        'H2SO4','SOA_LV','SOA_SV','SOA_NA','SO4_NA','SOA_A1',
        'NCFT_Ghan',
        'SFisoprene',
        'SFmonoterp',
        'SOA_NA_totLossR',
        'SOA_NA_lifetime',
        'SO4_NA_totLossR',
        'SO4_NA_lifetime',
        'cb_SOA_NA_OCW',
        'cb_SO4_NA_OCW',
        'SO4_NA_OCWDDF',
        'SO4_NA_OCWSFWET',
        'SOA_NA_OCWDDF',
        'SOA_NA_OCWSFWET',
        'cb_SOA_NA',
        'cb_SO4_NA',
        'cb_NA'


        ]
varl_sec = ['nrSOA_SEC_tot', 'nrSO4_SEC_tot','nrSEC_tot']

# %%
for area in areas:
    prof_dic = get_averaged_fields.get_area_avg_dic(cases,
                                                    varl,
                                                    area,
                                                    startyear,
                                                    endyear,
                                                    avg_over_lev=avg_over_lev,
                                                    pmin=pmin,
                                                    pressure_adjust=pressure_adjust
                                                    )
# %%
for area in areas:
    prof_dic = get_averaged_fields.get_area_avg_dic(cases_sec,
                                                    varl_sec,
                                                    area,
                                                    startyear,
                                                    endyear,
                                                    avg_over_lev=avg_over_lev,
                                                    pmin=pmin,
                                                    pressure_adjust=pressure_adjust
                                                    )


# %%

fn='/home/ubuntu/mnts/nird/projects//Output_data_OAS-DEV/Fields_pressure_coordinates/NorESM/SECTv21_ctrl_koagD/N_AER_NorESM_SECTv21_ctrl_koagD_2008-01-2009-12.nc'

# %%
ds = xr.open_dataset(fn)
# %%


# %%
def lev_time():
    # %%
    var='NCONC01'
    case1= cases[0]
    case2 = cases[2]
    avg_over_lev=True
    groupby='time.season'
    dims=('lon',)#'time',)
    area='Global'
    # %%
    dic_means_yr = yearly_mean_dic([var], [case1, case2], avg_over_lev=avg_over_lev, groupby=groupby, dims=dims, area=area)
    da1 = dic_means_yr[case1][var]
    da2 = dic_means_yr[case2][var]
    # %%

    # %%
    # %%
    diff = (da1-da2)/da2
    diff
    # %%
    fig, axs = plt.subplots(2,2,)
    for ax, seas in zip(axs.flatten(), ['JJA','SON','DJF','MAM']):
        diff.sel(season=seas).plot(robust=True, yscale='log',ylim=[1e3,100], ax=ax)
    plt.show()
    # %%
    da1.mean('time').plot(robust=True)
    plt.show()

    # %%
# %%
def NMSD(var,
         case1,
         case2,
         startyear,
         endyear,
         pmin,
         pressure_adjust,
         groupby='time.year',
         dims=None,
         area='Global'):
    # %%
    #var='ACTNL_incld'
    #case1= cases[0]
    #case2 = cases[1]
    #avg_over_lev=True
    #groupby=None
    #dims=('lev',)
    #area='Global'
    # %%
    dic_means_yr = yearly_mean_dic([var],
                                   [case1, case2],
                                   startyear,
                                   endyear,
                                   pmin,
                                   pressure_adjust,
                                   groupby=groupby,
                                   dims=dims,
                                   area=area
                                   )


    da1 = dic_means_yr[case1][var]
    da2 = dic_means_yr[case2][var]
    # %%

    _diffsq = (da1-da2)**2
    _nrmse = np.sqrt(_diffsq.mean('time'))/da2.std('time')
    # %%
    _nrmse.plot(robust=True)
    plt.show()
    # %%
    diff = (da1-da2)/da2
    # %%
    diff.mean('lon').plot(x='time',robust=True)
    plt.show()
    # %%
    da1.mean('time').plot(robust=True)
    plt.show()

    # %%


# %%

# %%
def plot_stuff():
    # %%
    var='AWNC_incld'
    case1= cases[0]
    case2 = cases[2]
    avg_over_lev=True
    groupby=None
    dims=('lon',)
    area='Global'
    ci=.95
    avg_dim='time'
    endyear = '2010-12'
    startyear = '2008-01'
    # %%
    T, t, sig_map,data4comp = get_significance_map_paired_monthly(var,
                                                                  case1,
                                                                  case2,
                                                                  startyear,
                                                                  endyear,
                                                                  pmin= pmin,
                                                                  pressure_adjust=pressure_adjust,
                                                                  avg_over_lev=avg_over_lev,
                                                                  ci = ci,
                                                                  groupby=groupby,
                                                                  dims=dims,
                                                                  area=area,
                                                                  avg_dim = avg_dim
                                                                  )
    # %%
    da1 = data4comp[case1]
    da2 = data4comp[case2]
    mean_D = data4comp[f'{case1}-{case2}'].mean(avg_dim)
    # %%
    fig, axs = plt.subplots(3, figsize=[4,7])


    t.where((t>T)|(t<-T)).plot( ax=axs[0], robust=True)
    (mean_D/da2.mean(avg_dim)*100).where((t>T)|(t<-T)).plot( ax=axs[1], vmax=10)
    ((mean_D/da2.mean(avg_dim)*100)).plot( ax=axs[2], robust=True,vmax=10)
    plt.suptitle(endyear)
    for ax in axs.flatten():
        ax.set_ylim([1e3,100])
        ax.set_yscale('log')
    plt.show()
    # %%
    fig, axs = subplots_map(3, figsize=[4,7])
    for ax in axs.flatten():
        fix_axis4map_plot(ax)

    plt_map(t.where((t>T)|(t<-T)), ax=axs[0], robust=True)
    plt_map((mean_D/da2.mean(avg_dim)*100).where((t>T)|(t<-T)), ax=axs[1], vmax=10)
    plt_map((mean_D/da2.mean(avg_dim)*100), ax=axs[2], robust=True,vmax=10)
    plt.suptitle(endyear)
    plt.show()
    # %%
    return








    # %%
    #dummy_dic[cases_sec[0]]
