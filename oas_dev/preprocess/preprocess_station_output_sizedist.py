from sectional_v2.util.Nd.sizedist_class_v2.SizedistributionStation import SizedistributionStation
from sectional_v2.util.collocate.collocateLONLAToutput import CollocateLONLATout
from sectional_v2.constants import sized_varListNorESM, list_sized_vars_noresm, list_sized_vars_nonsec
#from useful_scit.util import log
import useful_scit.util.log as log
import time
log.ger.setLevel(log.log.INFO)

nr_of_bins = 5
maxDiameter = 39.6  #    23.6 #e-9
minDiameter = 5.0  # e-9
history_field='.h1.'
cases_sec = ['SECTv11_ctrl']
cases_orig =['noSECTv11_noresm2_NFHIST']#'noSECTv11_ctrl'] #/noSECTv11_ctrl
from_t = '2007-01-01'
to_t = '2008-01-01'
t1 =time.time()
# %%
for case_name in cases_sec:
    varlist = list_sized_vars_noresm
    c = CollocateLONLATout(case_name, from_t, to_t,
                           True,
                           'hour',
                           history_field=history_field)
    if c.check_if_load_raw_necessary(varlist ):
        a = c.make_station_data_all()
for case_name in cases_orig:
    varlist = list_sized_vars_nonsec
    c = CollocateLONLATout(case_name, from_t, to_t,
                           False,
                           'hour',
                           history_field=history_field)
    if c.check_if_load_raw_necessary(varlist ):
        a = c.make_station_data_all()


# %%

# Make station N50 etc.
t1 =time.time()

for case_name in cases_sec:
    s = SizedistributionStation(case_name, from_t, to_t, [minDiameter, maxDiameter], True, 'hour',
                 nr_bins=nr_of_bins, history_field=history_field)
    #s.compute_Nd_vars()
    s.compute_sizedist_tot()

for case_name in cases_orig:
    print(case_name)

    s = SizedistributionStation(case_name, from_t, to_t, [minDiameter, maxDiameter], False, 'hour',
                                    nr_bins=nr_of_bins, history_field=history_field)
    #s.compute_Nd_vars()
    #a = s.compute_sizedist_mod_tot()
    s.compute_sizedist_tot()
t2 =time.time()
print('Time for sizedist: %f' %(t2-t1))
print('done')
