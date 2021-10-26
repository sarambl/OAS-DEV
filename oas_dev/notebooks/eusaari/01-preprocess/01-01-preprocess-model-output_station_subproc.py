

# %%
import sys

from oas_dev.util.collocate.collocateLONLAToutput import CollocateLONLATout
from oas_dev.constants import list_sized_vars_nonsec, list_sized_vars_noresm
import useful_scit.util.log as log
log.ger.setLevel(log.log.INFO)
import time

# %% [markdown]
# ### Settings

# %%

nr_of_bins = 5
maxDiameter = 39.6
minDiameter = 5.0
history_field = '.h1.'
from_t = sys.argv[1]
to_t = sys.argv[2]
case = sys.argv[3]
sectional = sys.argv[4].strip()
if sectional == 'True':
    sectional = True
elif sectional == 'False':
    sectional = False
else:
    sys.exit('Last arguemnt must be True or False')
if sectional:
    cases_sec = [case]
    cases_orig = []
else:
    cases_orig = [case]
    cases_sec = []
# %% [markdown]
# ## Compute collocated datasets from latlon specified output

# %% jupyter={"outputs_hidden": true}
for case_name in cases_sec:
    varlist = list_sized_vars_noresm
    c = CollocateLONLATout(case_name, from_t, to_t,
                       True,
                       'hour',
                       history_field=history_field)
    if c.check_if_load_raw_necessary(varlist):
        time1 = time.time()
        a = c.make_station_data_all()
        time2 = time.time()
        print('****************DONE: took {:.3f} s'.format((time2-time1)))
    else:
        print(f'Already computed for {case_name} ')

for case_name in cases_orig:
    varlist = list_sized_vars_nonsec
    c = CollocateLONLATout(case_name, from_t, to_t,
                           False,
                           'hour',
                           history_field=history_field)
    print(varlist)
    if c.check_if_load_raw_necessary(varlist):
        a = c.make_station_data_all()
    else:
        print(f'Already computed for {case_name} ')
