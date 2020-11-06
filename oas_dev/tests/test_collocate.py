#class MyTestCase(unittest.TestCase):
#    def test_something(self):
#        self.assertEqual(True, False)


#if __name__ == '__main__':
#    unittest.main()
# %%

from useful_scit.imps import (plt)


import sectional_v2.constants as constants
from sectional_v2.util.Nd.sizedist_class_v2 import Sizedistribution
import useful_scit.util.log as log
from sectional_v2.util.Nd.sizedist_class_v2.SizedistributionSurface import SizedistributionSurface
log.ger.setLevel(log.log.DEBUG)

pmin = 850.  # minimum pressure level
avg_over_lev = True  # True#True#False#True
pressure_adjust = True  # Can only be false if avg_over_lev false. Plots particular hybrid sigma lev
if avg_over_lev:
    pressure_adjust = True

cases_sec = ['PD_SECT_CHC7_diur_ricc', 'PD_SECT_CHC7_diurnal']# , 'PD_SECT_CHC7_diur_ricc_incC']
cases_orig = ['PD_noSECT_nudgeERA_eq_ricc','PD_noSECT_nudgeERA_eq20']#'Original eq.20']  # , 'Original eq.18','Original eq.20, 1.5xBVOC','Original eq.20, rednuc']

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
for case in [cases_sec[0]]:
    s = cl(case, from_t, to_t,
           [minDiameter, maxDiameter], True, 'month',
           history_field='.h0.')
    #s.compute_sizedist_mod_tot()
    #s.compute_sizedist_sec_tot()

    a = s.get_collocated_dataset(redo=True)
    print('*************************************************ASD')
    print(a)

# %%
cl = SizedistributionSurface
s_list = []
for case in cases_sec:
    s1 = cl(case, from_t, to_t,
           [minDiameter, maxDiameter], True, 'month',
           history_field='.h0.')
    s_list.append(s1)
for case in cases_orig:
    s1 = cl(case, from_t, to_t,
           [minDiameter, maxDiameter], False, 'month',
           history_field='.h0.')
    s_list.append(s1)
    #s1.compute_sizedist_mod_tot()
    #s.compute_sizedist_sec_tot()
#a1 = s1.get_collocated_dataset()

case = cases_sec[1]

s2 = cl(case, from_t, to_t,
           [minDiameter, maxDiameter], True, 'month',
           history_field='.h0.')
#a2 = s1.get_collocated_dataset()
from useful_scit.plot import get_cmap_dic
cmap_dic = get_cmap_dic(cases_sec+cases_orig)



for loc in constants.collocate_locations.keys():
    print(loc,'***********************************************')
    fig, axs = plt.subplots(2, 2)
    axs = axs.flatten()

    for s in s_list:
        ls = ['dNdlogD_mode01']
        if s.isSectional:
            ls = ls + ['dNdlogD_sec']
        a = s.get_collocated_dataset(variables=ls)
        print(a)
        b = a['dNdlogD_mode01']# + a['dNdlogD_sec']
        c = b.mean('lev').groupby('time.season').mean('time')

        for seas, ax in zip(c['season'].values, axs):
            _da = c.sel(season=seas, location=loc)

            _da.plot(xscale='log',yscale='log', ax=ax,
                                label = s.case_name,
                                ylim = [1e1,1e4],
                                xlim = [5,1e3],
                     c=cmap_dic[s.case_name]
                                )
            ax.set_title(seas+', '+loc)
        if not s.isSectional: continue
        b = a['dNdlogD_sec']
        c = b.mean('lev').groupby('time.season').mean('time')

        for seas, ax in zip(c['season'].values, axs):
            _da = c.sel(season=seas, location=loc)

            _da.plot(xscale='log',yscale='log', ax=ax,
                                label='__nolegend__',
                                ylim=[1e1,1e4],
                                xlim = [5,1e3],

                     c=cmap_dic[s.case_name]
                                )
            ax.set_title(seas+', '+loc)

    #for s
    #b.sel(location='Melpitz').mean(['lev', 'time']).plot(xscale='log',yscale='log',# norm=colors.LogNorm(vmin=1),
    #                                            label=s.case_name,
    #                                                     ylim=[1e0,1e4],
    #                                                      xlim = [5,1e3]
    #                                                     )
    #plt.title(s.case_name)
    plt.legend()
    plt.show()


# %%

for case in cases_orig:
    s = cl(case, from_t, to_t,
           [minDiameter, maxDiameter], False, 'month'
           , history_field='.h0.'
                         )
    s.compute_sizedist_mod_tot()
    s.get_collocated_dataset()

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
s = SizedistributionSurface(case, from_t, to_t, [minDiameter, maxDiameter], True, 'month')
a = s.get_sizedist_var()
# %%

#cm =
# %%
#from sectional_v2.util.Nd.collocate import CollocateModel
#cm = CollocateModel(case, from_t, to_t, isSectional=True, time_res='month')
#a =cm.load_sizedist_dataset([minDiameter, maxDiameter])

