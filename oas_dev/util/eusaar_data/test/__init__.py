from oas_dev.util.eusaar_data import time_h
from oas_dev.constants import path_eusaar_data
import netCDF4
import numpy as np
import pandas as pd

# %%
def load_time():
    p = path_eusaar_data + '/GEN/'
    timef = 'timevec_DOY.dat'
    ti = np.loadtxt(p+timef)
    return ti

def test_time():
    ti = load_time()
    units = 'days since 2008-01-01'
    time = netCDF4.num2date(ti, units, 'standard')
    # microseconds by precision error?
    time =pd.to_datetime(time).map(lambda x: x.replace(microsecond=0))#.asfreq('h')#[0]
    assert np.all(time_h.values == time.values)
    print(np.all(time_h.values == time.values))
    return

# %%
test_time()

# %%