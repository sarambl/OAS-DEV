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

# %% [markdown]
# # Seasonal sizedistribution plots:

# %%
 

# %%
# %load_ext autoreload
# %autoreload 2
from useful_scit.imps import (np, plt)

import sectional_v2.constants as constants
import useful_scit.util.log as log

# %%
from sectional_v2.util.plot.combination_plots import plot_sizedist_time_cases, \
    plot_seasonal_surface_loc_sizedistributions

avg_over_lev = True  # True#True#False#True
pressure_adjust = True  # Can only be false if avg_over_lev false. Plots particular hybrid sigma lev
if avg_over_lev:
    pressure_adjust = True

cases_sec = ['PD_SECT_CHC7_diur_ricc_incC','PD_SECT_CHC7_diur_ricc', 'PD_SECT_CHC7_diurnal']# , 'PD_SECT_CHC7_diur_ricc_incC']
cases_orig = ['PD_noSECT_nudgeERA_eq_ricc','PD_noSECT_nudgeERA_eq20']#'Original eq.20']  # , 'Original eq.18','Original eq.20, 1.5xBVOC','Original eq.20, rednuc']

cases = cases_sec + cases_orig

models = ['NorESM']
Path_local = constants.get_input_datapath(models[0])

history_field ='.h1.'
Path = {'NorESM': Path_local}
from_t = '2008-01-01'
to_t = '2009-01-01'

nr_of_bins = 5
maxDiameter = 39.6  # 23.6 #e-9
minDiameter = 5.0  # e-9

time_resolution = 'month'

# %%
log.ger.setLevel(log.log.WARNING)



# %%

plot_seasonal_surface_loc_sizedistributions(cases_sec, cases_orig, from_t, to_t)

# %%

from matplotlib import colors
## Plot sizedist time:

ss_start_t = '2008-06-01'
ss_end_t = '2008-06-10'
# %%


# %%
# test:
ss_start_t = '2008-01-01'
ss_end_t = '2009-01-01'

case = cases_sec[0]

location='Beijing'
figsize = [14,15]
plot_sizedist_time_cases(cases, ss_start_t, ss_end_t)
# %%





cba_kwargs = {'label': r'dN/dlogD [#/cm$^3$]', 'aspect': 8}
(dummy1.size_dtset['dNdlogD_sec'] + dummy1.size_dtset[dNdlogD]). \
    plot(x='time',
         y='logD', yscale='log',
         norm=colors.LogNorm(vmin=vmin, vmax=vmax),
         ylim=[3, 1e3], ax=axs[0],
         cbar_kwargs=cba_kwargs)
if isinstance(dummy1, NorESM_SizedistDataset_spec_lat_lon_output):
    axs[0].set_title(label1 + ', loc=' + dummy1.location)
else:
    axs[0].set_title(label1)

(dummy2.size_dtset['dNdlogD_sec'] + dummy2.size_dtset[dNdlogD]). \
    plot(x='time', y='logD', yscale='log',
         norm=colors.LogNorm(vmin=vmin, vmax=vmax),
         ylim=[3, 1e3], ax=axs[1],
         cbar_kwargs=cba_kwargs)
if isinstance(dummy2, NorESM_SizedistDataset_spec_lat_lon_output):
    axs[1].set_title(label2 + ', loc=' + dummy1.location)
else:
    axs[1].set_title(label2)
# axs[1].set_title(label2 + ', loc='+ dummy2.location)
if not nodiff:
    (-(dummy1.size_dtset['dNdlogD_sec'] + dummy1.size_dtset[dNdlogD]) + (
            dummy2.size_dtset['dNdlogD_sec'] + dummy2.size_dtset[dNdlogD])). \
        plot(x='time', y='logD', yscale='log', ax=axs[2],
             norm=colors.SymLogNorm(linthresh=10, linscale=1, vmin=-vmax, vmax=vmax),
             cmap='RdBu_r', ylim=[3, 1e3],
             cbar_kwargs=cba_kwargs)
    axs[2].set_title(label2 + ' - ' + label1)  # + ', loc='+ dummy1.location)
for i in np.arange(nr_subp):
    axs[i].set_ylabel('Diameter [nm]')
    axs[i].get_xaxis().get_label().set_visible(False)  # .get_xlabel().set_visible(False)
    # axs[i].get_xlabel().set_visible(False)

plotpath = 'plots/time_sizedist/'
plotpath = plotpath + '/lat%.1f_lon%.1f_pres%.1f/' % (latitude_coord, longitude_coord, pressure_coord)
plotpath = plotpath + dummy1.case_name + '_' + dummy2.case_name
plotpath = plotpath + 'time_%s-%s' % (from_time, to_time)
if nodiff:
    plotpath = plotpath + 'nodiff'
plotpath = plotpath + '.png'
practical_functions.make_folders(plotpath)
print(plotpath)
plt.tight_layout()
plt.savefig(plotpath, dpi=300)
plt.show()


# %%
s = s_list[1]
s.case_name

# %%
a_list = []
varl_sec = ['nrSOA_SEC_tot', 'nrSO4_SEC_tot', 'SOA_SEC_tot', 'SO4_SEC_tot', 'NCONC01', 'NNAT_1', 'N_AER', 'NUCLRATE_pbl','NUCLRATE','FORMRATE']
varl_orig = [ 'NCONC01', 'NNAT_1', 'N_AER', 'NUCLRATE_pbl','NUCLRATE','FORMRATE']
for s in s_list:
    if s.isSectional:
        a_list.append( s.get_input_data(varl_sec))
    else:
        a_list.append( s.get_input_data(varl_orig))
        

# %%
for a, s in zip(a_list, s_list):
    if not s.isSectional: continue
    a['nrSOA_SEC_tot'] =0.*a['nrSOA_SEC01']
    #.isel(time=1, lev=-1).values
    for var in ['nrSOA_SEC0%s'%ii for ii in range(1,6)]:
        a['nrSOA_SEC_tot']  = a['nrSOA_SEC_tot'] + a[var]
    (a['nrSO4_SEC_tot']+ a['nrSOA_SEC_tot']).mean(['time','lev','lon']).plot(label=s.case_name_nice)#robust=True)
plt.legend()

# %%
for a, s in zip(a_list, s_list):
    #.isel(time=1, lev=-1).values
    (a['NCONC01']).mean(['time','lev','lon']).plot(label=s.case_name_nice)#robust=True)
plt.legend()

# %%
#for a, s in zip(a_list, s_list):
    #.isel(time=1, lev=-1).values
a1 = a_list[1]
a2 = a_list[2]
((a1['NCONC01']-a2['NCONC01'])/a2['NCONC01']).mean(['time','lev','lon']).plot(label='sec')#robust=True)
a1 = a_list[3]
a2 = a_list[4]
((a1['NCONC01']-a2['NCONC01'])/a2['NCONC01']).mean(['time','lev','lon']).plot(label='orig')#robust=True)

plt.legend()

# %%
for a, s in zip(a_list, s_list):
    #.isel(time=1, lev=-1).values
    (a['N_AER']).mean(['time','lev','lon']).plot(label=s.case_name_nice)#robust=True)
plt.legend()

# %%
a['NNAT_1'].mean(['time','lev']).plot(robust=True)

# %%
a['nrSO4_SEC05'].mean(['time','lev']).plot(robust=True)

# %%
a['nrSOA_SEC_tot'] =0.*a['nrSOA_SEC01']
#.isel(time=1, lev=-1).values
for var in ['nrSOA_SEC0%s'%ii for ii in range(1,6)]:
    print(var)
    a['nrSOA_SEC_tot']  = a['nrSOA_SEC_tot'] + a[var]

# %%
(a['nrSO4_SEC_tot']+ a['nrSOA_SEC_tot']).mean('time').mean('lev').plot(robust=True)

# %%
(a['nrSO4_SEC05']+a['nrSOA_SEC05']).mean('time').mean('lev').plot(robust=True)

# %%
from sectional_v2.util.Nd.sizedist_class_v2 import get_bin_diameter
get_bin_diameter(5)

# %%


# %%
ma = get_bin_diameter(5)[1][-1]
mi = get_bin_diameter(5)[1][-2]
500/np.log(ma/mi)


# %% [markdown]
# \begin{align}
# \frac{dN}{dD} =\frac{dN}{dlogD}\cdot \frac{dlogD}{dD}
# \end{align}
# \begin{align}
# \frac{dN}{dlogD}= \frac{dN}{dD} \cdot \big(\frac{dlogD}{dD}\big)^{-1}
# \end{align}

# %% [markdown]
# \begin{align}
# \frac{dlogD}{dD} = \frac{1}{D}
# \end{align}

# %%
me = (ma+mi)/2.
500/(ma-mi)*me

# %%
(a['NCONC01']).mean('time').mean('lev').plot(robust=True)

# %%
a['N_AER'].mean(['time','lev']).sel(lat=slice(-10,1))

# %%
(a['nrSO4_SEC_tot']+ a['nrSOA_SEC_tot']).mean('time').sel(lon=-63., lat=-3., method='nearest').plot()#robust=True)

# %%
(a['N_AER']).mean('time').sel(lon=-60., lat=-3., method='nearest').plot()#robust=True)

# %%
a['N_AER'].mean(['time','lev'], keep_attrs=True).plot(robust=True)

# %%
(a['SO4_SEC_tot']).mean(['time','lev'], keep_attrs=True).plot(robust=True)

# %%

# %%
a['dNdlogD'] = a['dNdlogD_mod'] + a['dNdlogD_sec']

# %%
from matplotlib.colors import LogNorm
a.isel(lev=-1).isel(location=2)['dNdlogD'].plot(x='time',xscale='log', yscale='log', norm=LogNorm(vmin=1))


# %%
a.isel(lev=-1, location=2).mean('time')['dNdlogD'].plot(xscale='log', yscale='log', ylim=[4,1e4], xlim=[4,1e3])

# %%
b = s.get_collocated_dataset(variables=['dNdlogD_sec01', 'dNdlogD_sec02'])

# %%
for var in ['dNdlogD_sec01', 'dNdlogD_sec02']:
    b[var].isel(lev=-1, location=2).mean('time').plot(xscale='log', yscale='log', ylim=[100,1e4], xlim=[4,1e3])
plt.show()

# %%
