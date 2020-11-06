from sectional_v2.constants import get_outdata_path, collocate_locations
import xarray as xr
from pathlib import Path
import shutil

# %%
from sectional_v2.util.practical_functions import make_folders

dir_collocated = get_outdata_path('collocated')

locs_t = collocate_locations.transpose()

# %%


# %%
    # %%
def fix_coord_station(ds):
    # %%
    both_coords = get_ds_old_stationc2new()
    both_coords
# %%
    ds['nstation']= both_coords['nstation']
    ds_n = ds.rename({'station':'station_tab', 'nstation':'station'})
    ds_n = ds_n.swap_dims({'station_tab':'station'})
    return ds_n#ds_n['station'].values
# %%
def get_ds_old_stationc2new():
    alt_c = 'Alternative_code'
    both_coords = locs_t.reset_index()[[alt_c, 'index']].to_xarray()  # .rename({alt_c:'station'})
    both_coords = both_coords.swap_dims({'index': alt_c}).rename({'index': 'nstation', alt_c: 'station'})
    return both_coords
#pd.DataFrame.to_xarray()
# %%
def backup_filename(path):
    npath = str(path)
    _ss = npath.split('/')
    npath = '/'.join(_ss[:-1]) + '/backup/' + _ss[-1]
    return npath


# %%
paths = list(Path(dir_collocated).rglob('*.nc'))
list(paths)
len(paths)
# %%
for path in paths:
    if '/backup/' in str(path):
        continue

    print(path.name)
    ds = xr.open_dataset(path).load()
    if 'station_tab' in ds:
        ds.close()
        continue
    ds.close()
    npath = backup_filename(path)
    make_folders(npath)
    shutil.copy2(path, npath)
    ds.to_netcdf(npath)
    ds_n = fix_coord_station(ds)
    ds_n.to_netcdf(path)