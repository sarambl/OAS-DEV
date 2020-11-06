import Ngl
# import Nio
# import os
import xarray as xr
import numpy as np
# import matplotlib as mpl
from sectional_v2 import constants
# from useful_scit.util import log
import useful_scit.util.log as log

from sectional_v2.util.filenames import get_filename_pressure_coordinate_field
from sectional_v2.util.practical_functions import extract_path_from_filepath, make_folders

default_save_pressure_coordinates = constants.get_outdata_path('pressure_coords')  # 'Data/Fields_pressure_coordinates'


def hybsig2pres(ds, var, save_field=False):
    pnew = ds['lev'].values  # [:]  # [1013, 850.]
    if 'time' in ds['hyam'].dims:
        hyam = ds["hyam"].isel(time=0).values  # [:]
    else:
        hyam = ds["hyam"].values  # [:]
    if 'time' in ds['hybm'].dims:
        hybm = ds["hybm"].isel(time=0).values  # [:]
    else:
        hybm = ds["hybm"].values
    vals_in = ds[var].values
    da_in = ds[var]
    psrf = (ds["PS"][:, :, :])
    if 'time' in ds["P0"].dims:
        P0mb = 0.01 * ds["P0"].isel(time=0).values
    else:
        P0mb = 0.01 * ds["P0"].values

    #  Do the interpolation.
    intyp = 1  # 1=linear, 2=log, 3=log-log
    kxtrp = False  # True=extrapolate (when the output pressure level is outside of the range of psrf)

    out = Ngl.vinth2p(vals_in, hyam, hybm, pnew, psrf, intyp, P0mb, 1, kxtrp)
    out[out == 1e30] = np.NaN

    da_out = xr.DataArray(out, coords=ds[var].coords, dims=ds[var].dims)
    da_out.attrs = da_in.attrs.copy()
    da_out.coords['lev'].attrs['long_name'] = 'Pressure'
    da_out.coords['lev'].attrs['units'] = 'hPa'
    da_out.attrs['pressure_coords'] = 'True'
    if save_field:
        log.ger.info('Saved:' + var)

        ds_save = ds.copy().drop_vars(list(ds.data_vars))
        ds_save[var] = da_out
        # ds_save['lev']
        save_pressure_coordinate_field(ds_save, var)
    return da_out


def convert_all_vars_to_pressure(ds: xr.Dataset):
    _vars = ds.data_vars
    hybsig2pres_vars(ds, _vars)
    return ds


def hybsig2pres_vars(ds, _vars=None, save_field=True):
    log.ger.debug(f'hybsig2pres:{_vars}')

    if _vars is None:
        _vars = list(set(ds.data_vars) - set(constants.not_pressure_coords))

    for var in _vars:
        da = ds[var]
        if not ('pressure_coords' in da.attrs):
            compute = True
        elif da.attrs['pressure_coords'] == 'False' or \
                da.attrs['pressure_coords'] is False:
            compute = True
        else:
            compute = False
        if 'lev' not in da.dims or var in constants.not_pressure_coords:
            log.ger.info('Saved:' + var)
            ds[var].attrs['pressure_coords'] = 'True'
            save_pressure_coordinate_field(ds, var)
            compute = False
        if compute:
            # da_out = hybsig2pres()
            ds[var] = hybsig2pres(ds, var, save_field=save_field)
    return ds


def save_pressure_coordinate_field(ds, var,
                                   model='NorESM'):  # path_savePressCoord = default_save_pressure_coordinates):

    # test for pressure coordinates
    check_list = set(ds[var].attrs.keys()).intersection({'Pres_addj', 'pressure_coords'})
    check = False
    for key in check_list:
        if ds[var].attrs[key] is True or ds[var].attrs[key] == 'True':
            check = True
        elif var in constants.not_pressure_coords:
            check = True

    if not check:  # (not ds[var].attrs['Pres_addj']) and (not ds[var].attrs['pressure_coords'])\
        log.ger.info('Not pressure adjusted! Will not save')
    else:
        from_t = ds.attrs['from_time']  # ds['time'][argmin].values
        to_t = ds.attrs['to_time']  # ds['time'][argmax].values
        case = ds.attrs['case_name']

        filename = get_filename_pressure_coordinate_field(var, model, case, from_t, to_t)
        dummy = ds[var].copy()
        make_folders(extract_path_from_filepath(filename))
        log.ger.debug('Saving %s pressure coordinate field to file %s' % (var, filename))  # 'Time not in datetime')
        dummy.attrs['pressure_coords'] = 'True'
        try:
            log.ger.info(filename)
            dummy.to_netcdf(filename, mode='w')  # ,encoding={'time':{'units':'days since 2000-01-01 00:00:00'}})
        except PermissionError:
            log.ger.info(f'couldnt access file {filename}')
        del dummy
        return
