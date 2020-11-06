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

# %% jupyter={"outputs_hidden": true}
from useful_scit.imps import (np, plt)
 
import sectional_v2.constants as constants
import useful_scit.util.log as log
from sectional_v2.util.plot.combination_plots import plot_sizedist_time_cases, \
    plot_seasonal_surface_loc_sizedistributions


# %% pycharm={"name": "#%%\n"} jupyter={"outputs_hidden": false}

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
from sectional_v2.util.Nd.sizedist_class_v2 import Sizedistribution
 
def plot_seasonal_surface_loc_sizedistributions(cases_sec, cases_orig, from_t, to_t,
                                                variables=['dNdlogD'],
                                                minDiameter=5., maxDiameter=39.6, resolution='month',
                                                history_field='.h0.',
                                                figsize= [12,6], locations=constants.collocate_locations.keys()):
    cl = Sizedistribution
    s_list = []
    for case in cases_sec:
        s1 = cl(case, from_t, to_t,
            [minDiameter, maxDiameter], True, resolution,
            history_field=history_field)
        s_list.append(s1)
    for case in cases_orig:
        s1 = cl(case, from_t, to_t,
            [minDiameter, maxDiameter], False, resolution,
            history_field=history_field)
        s_list.append(s1)
    cmap_dic = get_cmap_dic(cases_sec+cases_orig)

    for loc in locations:
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        axs = axs.flatten()
        for s in s_list:
            ls = variables
            #if s.isSectional:
            #    ls = ls + ['dNdlogD_sec']
            s.plot_location(variables=ls,
                        c=cmap_dic[s.case_name],
                        axs=axs,
                        loc=loc,
                        ylim=[10,1e4])
        fig.tight_layout()
        savepath = constants.paths_plotsave['sizedist'] +'/season/'
        _cas = '_'.join(cases_sec) + '_'+'_'.join(cases_orig)
        savepath = savepath +loc+ _cas + 'mean_%s-%s_%s.png'%(from_t, to_t, resolution)
        make_folders(savepath)
        plt.savefig(savepath, dpi=200)
        plt.show()
