from oas_dev.util.Nd.sizedist_class_v2.SizedistributionBins import SizedistributionStationBins
from oas_dev.util.collocate.collocateLONLAToutput import CollocateLONLATout
from oas_dev.constants import list_sized_vars_nonsec, list_sized_vars_noresm
import useful_scit.util.log as log
log.ger.setLevel(log.log.INFO)

nr_of_bins = 5
maxDiameter = 39.6  #    23.6 #e-9
minDiameter = 5.0  # e-9
history_field='.h1.'
cases_sec = ['SECTv11_noresm2_NFHIST']#,'SECTv11_ctrl_fbvoc']#'SECTv11_ctrl']
cases_orig =[]#'noSECTv11_ctrl_fbvoc'] #/no SECTv11_ctrl
from_t = '2008-01-01'
to_t = '2008-01-03'

# %%
for case_name in cases_sec:
    varlist = list_sized_vars_noresm
    c = CollocateLONLATout(case_name, from_t, to_t,
                       True,
                       'hour',
                       history_field=history_field)
    if c.check_if_load_raw_necessary(varlist ):
        print('HEY1 in preprocess station data 25')
        a = c.make_station_data_all()
    else:
        print('UUUPS')
for case_name in cases_orig:
    varlist = list_sized_vars_nonsec
    c = CollocateLONLATout(case_name, from_t, to_t,
                           False,
                           'hour',
                           history_field=history_field)

    if c.check_if_load_raw_necessary(varlist ):
        a = c.make_station_data_all()
    else:
        print('IIIIP')





# %%

# Make station N50 etc.
for case_name in cases_sec:
    s = SizedistributionStationBins(case_name, from_t, to_t, [minDiameter, maxDiameter], True, 'hour',
                 nr_bins=nr_of_bins, history_field=history_field)
    s.compute_Nd_vars()

for case_name in cases_orig:
    s = SizedistributionStationBins(case_name, from_t, to_t, [minDiameter, maxDiameter], False, 'hour',
                                    nr_bins=nr_of_bins, history_field=history_field)
    s.compute_Nd_vars()

# %%
print('***************FINISHED********************')