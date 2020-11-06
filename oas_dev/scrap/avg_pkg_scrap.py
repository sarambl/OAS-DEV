import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

from sectional_v2.util import var_overview_sql, practical_functions
from sectional_v2.util.filenames import filename_map_avg
from sectional_v2.util.slice_average import area_mod
from sectional_v2.util.slice_average.avg_pkg import get_fields4weighted_avg, average_model_var, \
    compute_weighted_averages, is_weighted_avg_var, path_to_global_avg, get_lat_wgts_matrix, get_pres_wgts_matrix
from sectional_v2.util.slice_average.avg_pkg.maps import load_average2map


def get_average_map(xr_ds: xr.Dataset, varList: list, model_name: str, pmin: float, case_name: str, avg_over_lev: bool,
                    p_level: float, pressure_adj: bool
                    , look_for_file: bool = True, save_avg=True) -> xr.Dataset:
    """
    :param xr_ds:
    :param varList:
    :param model_name:
    :param pmin:
    :param case_name:
    :param avg_over_lev:
    :param p_level:
    :param pressure_adj:
    :param look_for_file:
    :return:
    """
    # df = pd.DataFrame(columns=[model_name], index = varList)
    startyear = xr_ds.attrs['startyear']
    endyear = xr_ds.attrs['endyear']
    xr_out = xr_ds.copy()
    for var in varList:
        print(var)
        found_area_mean = False
        if look_for_file:
            dummy, found_area_mean = load_average2map(model_name, case_name, var, startyear, endyear, avg_over_lev,
                                                      pmin, p_level, pressure_adj)
        if not found_area_mean:
            sub_varL = get_fields4weighted_avg(var)

            dummy = xr_ds.copy()
            for svar in sub_varL:
                if avg_over_lev or 'lev' not in xr_ds.dims:
                    dummy = average_model_var(dummy, svar, area='Global', dim=list(set(xr_ds.dims)-{'lat', 'lon'}), minp=pmin) #\
                else:
                    #print(dummy)
                    dummy = average_model_var(dummy, svar, area='Global',
                                            dim=list(set(xr_ds.dims)-{'lat', 'lon', 'lev'}))
                    if 'lev' in dummy[svar].dims:
                        dummy[svar] = dummy[svar].sel(lev=p_level, method='nearest')
            dummy = compute_weighted_averages(dummy, var, model_name)
            dummy[var].attrs['Calc_weight_mean']=str(is_weighted_avg_var(var)) + ' map'
            var_info_df = var_overview_sql.open_and_fetch_var_case(model_name, case_name, var)
            if len(var_info_df) > 0:
                had_lev_coord = bool(var_info_df['lev_is_dim'].values)
            else:
                had_lev_coord = True
            if save_avg:
                filen = filename_map_avg(model_name, case_name, var, startyear, endyear, avg_over_lev, pmin, p_level,
                                         pressure_adj, lev_was_coord=had_lev_coord)
                practical_functions.make_folders(filen)
                practical_functions.save_dataset_to_netcdf(dummy, filen)
        xr_out[var] = dummy[var]
    return xr_out


def average_xr_ds(xr_ds, var, model_name, avg_to, pressure_adjust, area='Global', minp=850., p_level = 1013,avg_over_lev=True):
    if avg_to=='latlon':
        if avg_over_lev:
            return average_timelatlon_lev(xr_ds,var,model_name,area, minp)
        else:
            return  average_timelatlon_at_lev(xr_ds, var, model_name, p_level)
    elif avg_to =='levlat':
        return xr_ds[var].mean(dim=['lon','time'])
    elif avg_to =='profile':
        #TODO: add
        return None
    elif avg_to =='one_value':
        #TODO: Add
        return None
    else:
        print('Did not recognize avg type')
        return


def avg_timelatlonlev_2_dataframe(nested_datasets, models, cases, N_vars, area, pmin, avg_over_lev, p_level,
                                  pressure_adjust, startyear, endyear, load_from_file_only=False):
    nested_pd = {}
    for case in cases:
        first = True
        for model in models:
            if load_from_file_only:
                pd_dummy = load_avgerage_timelatlonlev_dtset(N_vars, model, area, pmin, case, avg_over_lev, p_level,
                                                             pressure_adjust, startyear, endyear)
            else:
                pd_dummy = avgerage_timelatlonlev_dtset(nested_datasets[model][case],
                                                                                    N_vars, model, area, pmin, case,
                                                                                    avg_over_lev,
                                                                                    p_level, pressure_adjust)
            if first:
                nested_pd[case] = pd_dummy
                first = False
            else:
                nested_pd[case] = pd.concat([nested_pd[case], pd_dummy], axis=1,
                                            join_axes=[nested_pd[case].index])  # , sort=False)#, sorted=False)
    return nested_pd


def load_area_mean(model, case, var, area, startyear, endyear, avg_over_lev, pmin, p_level, pressure_adj):
    path_to_data = path_to_global_avg
    var_info_df = var_overview_sql.open_and_fetch_var_case(model, case, var)
    if len(var_info_df) > 0:
        had_lev_coord = bool(var_info_df['lev_is_dim'].values)
    else:
        had_lev_coord = True
    # df_var_info, df_filen = practical_functions.open_model_info_csv(model)#:var_mod_info_filen = '%s_variable_info.csv' % model_name
    # if var in df_var_info:
    #    had_lev_coord = df_var_info.loc['lev is dimension',var]=='True'
    filen = filename_global_avg(model, case, var, startyear, endyear, avg_over_lev, area, pmin, p_level, pressure_adj,
                                lev_was_coord=had_lev_coord)
    if os.path.isfile(filen):
        print('Loading file %s' % filen)
        xr_ds = xr.open_dataset(filen).copy()
        return xr_ds, True
    else:
        return [], False


def average_timelatlon_lev(xr_ds, var, model_name, area, minp):
    """
    Assumes pressure coordinates. Calculates average from surface to minp pressure level
    :param xr_ds:
    :param var:
    :param model_name:
    :param area:
    :param minp:
    :return:
    """
    # Get weight matrices:
    wg_matrix = get_lat_wgts_matrix(var, xr_ds)
    press_diff_matrix = get_pres_wgts_matrix(var, xr_ds)
    dummy, pressure = xr.broadcast(xr_ds[var], xr_ds['lev'])
    # Let weights be product of weighted by lat and by pressure different.
    wgts_matrix = wg_matrix*press_diff_matrix
    mask1, area_masked = area_mod.get_4d_area_mask_xa(area, xr_ds, var)  # get 3D mask for area
    mask2 = pressure <= minp
    mask = np.logical_or(mask1,mask2) # mask whenever one value says True.
    da = xr_ds[var] # pick out DataArray from DataSet
    da_masked = da.where(np.logical_not(mask)) # Mask values not in accordance to mask (sets to nans)
    wgts_masked = wgts_matrix.where(np.logical_not(mask)) # Set weights to nan where not right area
    mean = da_masked*wgts_masked/wgts_masked.sum()

    if 'startyear' in xr_ds[var].attrs: mean.attrs['startyear'] = xr_ds[var].attrs['startyear']
    else: mean.attrs['startyear'] = xr_ds['time.year'].min().values
    if 'endyear' in xr_ds[var].attrs: mean.attrs['endyear'] = xr_ds[var].attrs['endyear']
    else: mean.attrs['startyear'] = xr_ds['time.year'].max().values
    mean.attrs = xr_ds[var].attrs
    return mean


def average_timelatlon_at_lev(xr_ds, var, model_name, area, level_to_plot):
    # dummy=nested_datasets[model][case].copy()
    time = xr_ds['time'].values
    lon = xr_ds['lon'].values.copy()
    lat = xr_ds['lat'].values.copy()
    lon = xr_ds['lon'].values.copy()  # +180
    startyear = xr_ds['time.year'].min().values
    endyear = xr_ds['time.year'].max().values
    if 'lev' in xr_ds:
        lev = xr_ds['lev'].values
        lev_ind = np.argmin(np.abs(lev - level_to_plot))
    else:
        lev = [1000.]
    area_masked = False
    mask_area = None
    mask, area_masked = area_mod.get_xd_area_mask(area, xr_ds, lev, var, time)
    # area_pkg_sara.plot_my_area(xr_ds, np.logical_not(mask), var, area)

    if 'lat_wg' in xr_ds:
        wgts_ = xr_ds['lat_wg'].values
    else:
        wgts_ = area_mod.get_wghts(xr_ds['lat'].values)
    if 'lev' in xr_ds[var]:
        wgts_matrix = np.empty([len(time), len(lev), len(lat), len(lon)])
        for lo in np.arange(len(lon)):
            for le in np.arange(len(lev)):
                for ti in np.arange(len(time)):
                    wgts_matrix[ti, le, :, lo] = wgts_
    else:
        wgts_matrix = np.empty([len(time), len(lat), len(lon)])
        for lo in np.arange(len(lon)):
            for ti in np.arange(len(time)):
                wgts_matrix[ti, :, lo] = wgts_

    xr_ds_da = xr_ds[var].copy()

    i = 0
    for di in list(xr_ds_da.dims):
        if (di == 'lat'):
            lat_ind = i
        else:
            i = i + 1

    a1 = np.ma.array(xr_ds_da.values, mask=mask)
    m1 = np.ma.array(wgts_matrix, mask=mask)
    lat_mean = np.ma.average(a1, weights=m1, axis=lat_ind)  # .sum(dim='lat')
    coords = []
    dims = []
    for dim in list(xr_ds_da.dims):
        if (dim != 'lat'):
            dims.append(dim)
            coords.append(xr_ds_da.coords[dim])
    dummy = xr_ds.copy()
    dummy[var] = xr.DataArray(lat_mean, coords=coords, dims=dims, attrs=xr_ds_da.attrs, name=var)
    if 'lev' in xr_ds[var]:
        mean = dummy.mean(dim=('time', 'lon'), skipna=True, keep_attrs=True).isel(lev=lev_ind)
    else:
        mean = dummy.mean(dim=('time', 'lon'), skipna=True, keep_attrs=True)
    del dummy
    mean.attrs['startyear'] = startyear
    mean.attrs['endyear'] = endyear
    return mean


def avgerage_timelatlonlev_dtset(xr_ds, varList, model_name, area, pmin, case_name, avg_over_lev, p_level, pressure_adj
                                 , look_for_file=True):
    df = pd.DataFrame(columns=[model_name], index=varList)
    startyear = xr_ds.attrs['startyear']
    endyear = xr_ds.attrs['endyear']
    for var in varList:
        print(var)
        if look_for_file:
            dummy, found_area_mean = load_area_mean(model_name, case_name, var, area, startyear, endyear, avg_over_lev,
                                                    pmin, p_level, pressure_adj)
        else:
            found_area_mean = True  # not true actually
            dummy = average_timelatlon_lev(xr_ds, var, model_name, area, pmin)
        if not found_area_mean:
            dummy = average_timelatlon_lev(xr_ds, var, model_name, area, pmin)
            filen = filename_global_avg(model_name, case_name, var, startyear, endyear, avg_over_lev, area, pmin,
                                        p_level, pressure_adj)
            practical_functions.make_folders(filen)
            practical_functions.save_dataset_to_netcdf(dummy, filen)
        # print(dummy)
        df.loc[var, model_name] = float(
            dummy[var].values)  # average_timelatlon_lev(xr_ds, var, model_name, area, minp).values)

    # xr_ds.to_netcdf(savepath+'%s_%s.nc' %(model_name, case_name))
    return df


def load_avgerage_timelatlonlev_dtset(varList, model_name, area, pmin, case_name, avg_over_lev, p_level, pressure_adj,
                                      startyear, endyear):
    """

    :param varList:
    :param model_name:
    :param area:
    :param pmin:
    :param case_name:
    :param avg_over_lev:
    :param p_level:
    :param pressure_adj:
    :param startyear:
    :param endyear:
    :return:
    """
    df = pd.DataFrame(columns=[model_name], index=varList)
    for var in varList:
        dummy, found_area_mean = load_area_mean(model_name, case_name, var, area, startyear, endyear, avg_over_lev,
                                                pmin, p_level, pressure_adj)
        if not found_area_mean:
            print('load_average_timelatlon_dtset: Could not find mean dataset')
            sys.exit()
        df.loc[var, model_name] = float(
            dummy[var].values)  # average_timelatlon_lev(xr_ds, var, model_name, area, minp).values)

    return df


def filename_global_avg(model, case, var,
                        startyear, endyear,
                        avg_over_lev, area, pmin, p_lev, pres_adj,
                        lev_was_coord=True, path_to_data=path_to_global_avg):
    filen = path_to_data + '/' + area + '/%s/%s/%s_%.0f-%.0f' % (model, case, var, startyear, endyear)
    if not lev_was_coord:
        filen = filen + '_lev_not_dim'
    elif avg_over_lev:
        filen = filen + '_avg2lev%.0f' % pmin
    else:
        filen = filen + '_atlev%.0f' % p_lev
    if not pres_adj:
        filen = filen + 'not_pres_adj.nc'
    else:
        filen = filen + '.nc'
    return filen.replace(' ','_')


def average2map2lev(xr_ds, var, model_name, minp):
    xr_out = average_lev(xr_ds, var, model_name, 'Global', minp)
    xr_out = xr_out.mean(dim='time')
    xr_out[var].attrs= xr_ds[var].attrs

    return xr_out


def average_lev(xr_ds, var, model_name, area, minp):
    # print(xr_ds)
    if 'lev' not in xr_ds[var].coords:
        return xr_ds
    time = xr_ds['time'].values
    lon = xr_ds['lon'].values.copy()  # +180
    lat = xr_ds['lat'].values.copy()
    lev = xr_ds['lev'].values
    ilev = xr_ds['ilev'].values

    mask1, area_masked = area_mod.get_xd_area_mask(area, xr_ds, lev, var, time)

    # avg over pressure:
    pres_diff_1d = np.zeros_like(lev) * np.nan
    if lev[0] > lev[1]:  # lev oriented upwards:
        pres_diff_1d[:] = ilev[0:-1] - ilev[1::]
    else:
        pres_diff_1d[:] = ilev[1::] - ilev[0:-1]

    pres_diff = np.repeat(
        np.repeat(np.tile(pres_diff_1d, (len(time), 1))[:, :, np.newaxis], len(lat), axis=2)[:, :, :, np.newaxis],
        len(lon), axis=3)
    xr_ds_da = xr_ds[var].copy()

    mask2 = np.repeat((lev <= minp)[np.newaxis, :], len(time), axis=0)
    mask2 = np.repeat(mask2[:, :, np.newaxis], len(lat), axis=2)
    mask2 = np.repeat(mask2[:, :, :, np.newaxis], len(lon), axis=3)
    # combine masks:
    mask =  np.logical_or(mask1,mask2)
    # make masked array:
    a1 = np.ma.array(xr_ds_da.values, mask=mask)
    # make masked weights:
    # avg over weights:
    if 'time' in xr_ds.dims:
        ind = 1
    else:
        ind = 0

    lev_mean = np.ma.average(a1, weights=pres_diff, axis=ind)  # .sum(dim='lat')

    coords = []
    dims = []
    for dim in list(xr_ds_da.dims):
        if (dim != 'lev'):
            dims.append(dim)
            coords.append(xr_ds_da.coords[dim])
    xr_ds_out = xr_ds.copy()
    xr_ds_out[var] = xr.DataArray(lev_mean, coords=coords, dims=dims, attrs=xr_ds_da.attrs, name=var)
    return xr_ds_out