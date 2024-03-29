#class MyTestCase(unittest.TestCase):
#    def test_something(self):
#        self.assertEqual(True, False)


#if __name__ == '__main__':
#    unittest.main()
from useful_scit.imps import (plt)
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

cases_sec = ['PD_SECT_CHC7_diur_ricc', 'PD_SECT_CHC7_diurnal']  # Sect ac eq.20, corr NPF diam, fxdt, vdiam, 1.5xBVOC']
# cases_orig = ['PD_noS']
cases_orig = ['PD_noSECT_nudgeERA_eq20']#'Original eq.20']  # , 'Original eq.18','Original eq.20, 1.5xBVOC','Original eq.20, rednuc']
cases = cases_sec + cases_orig

models = ['NorESM']
Path_local = constants.get_input_datapath(models[0])

# history_field = '.h1.'

Path = {'NorESM': Path_local}
from_t = '2008-01-01'
to_t = '2009-01-01'
calculate_sizedists_time = True
loc_dataset = True  # if location output.
# locations = constants.locations
pressure_coor = 1013.  # 600.
# Sectional settings:
places = {'Hyytiala': [62, 24], 'Beijing': [39.9, 116.4]}

diameters = [0, 20., 60., 80., 100., 200., 500., 1000.]
nr_of_bins = 5
maxDiameter = 39.6  # 23.6 #e-9
minDiameter = 5.0  # e-9

time_resolution = 'month'
# 'PD_SECT_CHC7_diurnal'
# %%

surface = True
if surface:
    cl = SizedistributionSurface
else:
    cl = Sizedistribution
for case in cases_sec:
    s = cl(case, from_t, to_t,
           [minDiameter, maxDiameter], True, 'month',
           history_field='.h0.')
    s.compute_sizedist_mod_tot()
    s.compute_sizedist_sec_tot()
for case in cases_orig:
    s = cl(case, from_t, to_t,
           [minDiameter, maxDiameter], False, 'month'
           , history_field='.h0.'
                         )
    s.compute_sizedist_mod_tot()

#s.compute_sizedist_mod_tot()
s.get_sizedist_var()
# a,b = s.compute_sizedist_var('dNdlogD_mod01')
# a['dNdlogD_mod01'].isel(lev=20).mean(['time','lat','lon']).plot(xscale='log', yscale='log')
plt.show()
#s.get_sizedistrib_dataset()
s.raw_data_path

# %%

#if surface:
#    cl = SizedistributionSurface
#else:
#    cl = Sizedistribution
case = cases_sec[0]
s = SizedistributionSurface(case, from_t, to_t,[minDiameter, maxDiameter], True, 'month')
a = s.get_sizedist_var()
# %%

#cm =
# %%
from oas_dev.util.collocate.collocate import CollocateModel
cm = CollocateModel(case, from_t, to_t, isSectional=True, time_res='month')
a =cm.load_sizedist_dataset([minDiameter, maxDiameter])

