
from useful_scit.imps import (np, plt)

## Settings:
import oas_dev.constants as constants
from oas_dev.util.Nd.sizedist_class_v2 import Sizedistribution
import useful_scit.util.log as log
from oas_dev.util.Nd.sizedist_class_v2.SizedistributionSurface import SizedistributionSurface
log.ger.setLevel(log.log.DEBUG)

pmin = 850.  # minimum pressure level
avg_over_lev = True  # True#True#False#True
pressure_adjust = True  # Can only be false if avg_over_lev false. Plots particular hybrid sigma lev
if avg_over_lev:
    pressure_adjust = True

cases_sec = []#'SECTv11_ctrl2']#,'SECTv11_redSOA_LVnuc','SECTv11_incBVOC']#'PD_SECT_CHC7_diur_ricc']#, 'PD_SECT_CHC7_diurnal']# , 'PD_SECT_CHC7_diur_ricc_incC']
#cases_sec = ['SECTv11_redSOA_LVnuc','SECTv11_incBVOC']#'PD_SECT_CHC7_diur_ricc']#, 'PD_SECT_CHC7_diurnal']# , 'PD_SECT_CHC7_diur_ricc_incC']
cases_orig = ['noSECTv11_ctrl']#'noSECTv11_ctrl']#,'PD_noSECT_nudgeERA_eq20']#'Original eq.20']  # , 'Original eq.18','Original eq.20, 1.5xBVOC','Original eq.20, rednuc']
cases = cases_sec + cases_orig

models = ['NorESM']
# Path_local = constants.get_input_datapath(models[0])


from_t = '2009-01-01'
to_t = '2010-01-01'
calculate_sizedists_time = True
loc_dataset = True  # if location output.
pressure_coor = 1013.  # 600.

diameters = [0, 20., 60., 80., 100., 200., 500., 1000.]
nr_of_bins = 5
maxDiameter = 39.6  # 23.6 #e-9
minDiameter = 5.0  # e-9

time_resolution = 'month'
# %%


from oas_dev.util.Nd.sizedist_class_v2.SizedistributionBins import SizedistributionBins, SizedistributionSurfaceBins
def produce_binned_sizedist(cases_sec, cases_orig, from_t, to_t, minDiameter, maxDiameter,
                            history_field='.h0.', time_resolution='month'):
    for case in cases_sec:
        s = SizedistributionBins(case, from_t, to_t,
                                 [minDiameter, maxDiameter], True, 'month',
                                 history_field=history_field)
        s.compute_Nd_vars()#redo=True)
        s = SizedistributionSurfaceBins(case, from_t, to_t,
                                        [minDiameter, maxDiameter], True, 'month',
                                        history_field=history_field)
        s.compute_Nd_vars()
    for case in cases_orig:
        s = SizedistributionBins(case, from_t, to_t,
                                 [minDiameter, maxDiameter], False, time_resolution,
                                 history_field=history_field)
        s.compute_Nd_vars()#redo=True)


produce_binned_sizedist(cases_sec, cases_orig, from_t, to_t, minDiameter, maxDiameter, time_resolution=time_resolution)

# %%

surface = True
def produce_sizedist_surf(cases_sec, cases_orig, from_t, to_t, minDiameter, maxDiameter,
                     history_field='.h0.', time_resolution='month'):
    for case in cases_sec:
        s = SizedistributionSurface(case, from_t, to_t,
                                    [minDiameter, maxDiameter], True, time_resolution,
                                    history_field=history_field)
        s.get_collocated_dataset()#redo=True)

    for case in cases_orig:
        s = SizedistributionSurface(case, from_t, to_t,
                                [minDiameter, maxDiameter], False, time_resolution,
                                history_field=history_field)
        s.get_collocated_dataset()#redo=True)

produce_sizedist_surf(cases_sec, cases_orig, from_t, to_t, minDiameter, maxDiameter, history_field='.h0.', time_resolution=time_resolution)

# %%
## Sizedataset:
def produce_sizedist(cases_sec, cases_orig, from_t, to_t, minDiameter, maxDiameter,
                     history_field='.h0.', time_resolution='month'):
    for case in cases_sec:
        s = Sizedistribution(case, from_t, to_t,
                             [minDiameter, maxDiameter], True, 'month',
                             history_field=history_field)
        s.compute_sizedist_tot()#redo=True)

    for case in cases_orig:
        s = Sizedistribution(case, from_t, to_t,
                                [minDiameter, maxDiameter], False, time_resolution,
                                history_field=history_field)
        a = s.compute_sizedist_mod_tot()#redo=True)

produce_sizedist(cases_sec, cases_orig, from_t, to_t, minDiameter, maxDiameter, time_resolution=time_resolution)


# %%
