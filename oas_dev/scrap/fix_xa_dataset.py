import xarray as xr
from netCDF4 import num2date
#from util import
#import analysis_tools.var_overview_sql
from sectional_v2.util import var_overview_sql
#import analysis_tools.area_pkg_sara
import sectional_v2.util.slice_average.area_mod as area_pkg_sara
import numpy as np
import os
#from analysis_tools import import_fields_xr, practical_functions, get_model_lev_parfix as get_model_lev, \
#    translate_var_names
from sectional_v2.util.imports import get_model_lev as get_model_lev
from sectional_v2.scrap import import_fields_xr
from sectional_v2.util.naming_conventions import translate_var_names
from sectional_v2.util import practical_functions
#from sectional_v2.util.imports import import_fields_xr, get_model_lev_parfix as get_model_lev, translate_var_names
# from analysis_tools.practical_functions import add_variable_info_to_model_info_csv, add_index_model_info_csv
from sectional_v2 import constants
#from constants import path_outdata_pressure_coord
default_save_pressure_coordinates = constants.get_outdata_path('pressure_coords')#'Data/Fields_pressure_coordinates'
default_conversion_data = constants.get_outdata_path('pressure_coords_converstion_fields')#'Data/Pressure_coordinates_conversion_fields'
default_pressure_density_path = constants.get_outdata_path('pressure_density_path')#Data/Pressure_density'

practical_functions.make_folders(default_conversion_data)
practical_functions.make_folders(default_save_pressure_coordinates)

vars_N_EC_Earth = ['N_NUS', 'N_AIS', 'N_ACS', 'N_COS', 'N_AII', 'N_ACI', 'N_COI']
vars_radi_EC_Earth = ['RDRY_NUS', 'RDRY_AIS', 'RDRY_ACS', 'RDRY_COS', 'RWET_AII', 'RWET_ACI', 'RWET_COI']

vars_N_ECHAM = ['NUM_NS', 'NUM_KS', 'NUM_AS', 'NUM_CS', 'NUM_KI', 'NUM_AI', 'NUM_CI']
vars_radi_ECHAM = ['RWET_NS', 'RWET_KS', 'RWET_AS', 'RWET_CS', 'RWET_KI', 'RWET_AI', 'RWET_CI']
NCONC_noresm = []
NMR_noresm = []
for i in np.arange(14):
    NCONC_noresm.append('NCONC%02.0f' % (i + 1))
    NMR_noresm.append('NMR%02.0f' % (i + 1))

sizedist_vars = vars_N_EC_Earth + vars_radi_EC_Earth + vars_N_ECHAM + vars_radi_ECHAM + NCONC_noresm + NMR_noresm


def xr_fix(dtset, model_name='NorESM', sizedistribution=False):
    """

    :param dtset:
    :param model_name:
    :param sizedistribution:
    :return:
    """
    print('xr_fix: Doing various fixes for %s' % model_name)

    # Rename stuff:
    if (model_name != 'NorESM'):
        for key in dtset.variables:
            #print(key)
            if (not sizedistribution or key not in sizedist_vars):
                var_name_noresm = translate_var_names.model2NorESM(key, model_name)

                if 'orig_name' not in dtset[key].attrs:
                    dtset[key].attrs['orig_name'] = key
                if (len(var_name_noresm) > 0):
                    print('Translate %s to %s ' % (key, var_name_noresm))
                    dtset = dtset.rename({key: var_name_noresm})

    ############################
    # NorESM:
    ############################

    elif (model_name == 'NorESM'):
        # print('So far not much to do')
        time = dtset['time'].values  # do not cast to numpy array yet

        if isinstance(time[0], float):
            time_unit = dtset['time'].attrs['units']
            time_convert = num2date(time[:] - 15, time_unit, dtset.time.attrs['calendar'])
            dtset.coords['time'] = time_convert
        for nconc in NCONC_noresm:
            if nconc in dtset:
                if (dtset[nconc].attrs['units'] == '#/m3'):
                    print('xr_fix: converting %s from m-3 to cm-3' % nconc)
                    dtset[nconc].values = dtset[nconc].values * 1e-6
                    dtset[nconc].attrs['units'] = 'cm-3'
        for nmr in NMR_noresm:
            if nmr in dtset:
                if (dtset[nmr].attrs['units'] == 'm'):
                    print('xr_fix: converting %s from m to nm' % nmr)
                    dtset[nmr].values = dtset[nmr].values * 1e9
                    dtset[nmr].attrs['units'] = 'nm'
        if 'NNAT_0' in dtset.data_vars:
            dtset['SIGMA00'] = dtset['NNAT_0'] * 0 + 1.6  # Kirkevag et al 2018
            dtset['SIGMA00'].attrs['units'] = '-'  # Kirkevag et al 2018
            dtset['NMR00'] = dtset['NNAT_0'] * 0 + 62.6  ##nm Kirkevag et al 2018
            dtset['NMR00'].attrs['units'] = 'nm'  ##nm Kirkevag et al 2018
            dtset['NCONC00'] = dtset['NNAT_0']
        for cvar in ['AWNC']:
            if cvar in dtset:
                if dtset[cvar].units == 'm-3':
                    dtset[cvar].values = 1.e-6 * dtset[cvar].values
                    dtset[cvar].attrs['units'] = '#/cm^3'
        for cvar in ['ACTNI', 'ACTNL']:
            if cvar in dtset:
                if dtset[cvar].units == 'Micron':
                    dtset[cvar].values = 1.e-6 * dtset[cvar].values
                    dtset[cvar].attrs['units'] = '#/cm^3'

        cont = True
        i = 1
        while cont:
            varSEC = 'nrSO4_SEC%02.0f' % i
            if varSEC in dtset.data_vars:
                if dtset[varSEC].attrs['units'] != 'cm-3':
                    dtset[varSEC] = dtset[varSEC] * 1e-6  # m-3 --> cm-3
                    dtset[varSEC].attrs['units'] = 'cm-3'
            else:
                cont = False
            i += 1
        i = 1

        cont = True
        while cont:
            varSEC = 'nrSOA_SEC%02.0f' % i
            if varSEC in dtset.data_vars:
                if dtset[varSEC].attrs['units'] != 'cm-3':
                    dtset[varSEC] = dtset[varSEC] * 1e-6  # m-3 --> cm-3
                    dtset[varSEC].attrs['units'] = 'cm-3'
            else:
                cont = False
            i += 1

        first = True
        all_sec_in = True
        for sec_var in ['nrSO4_SEC_tot', 'nrSOA_SEC_tot', 'nrSEC_tot'] + ['nrSEC%02.0f' % ii for ii in range(1, 6)]:
            if sec_var in dtset:
                if dtset[sec_var].attrs['units'] == 'unit':
                    dtset[sec_var].values = dtset[sec_var].values * 1.e-6
                    dtset[sec_var].attrs['units'] = 'cm-3'
        for ii in np.arange(1, 6):
            sec_nr = 'nrSOA_SEC%02.0f' % ii
            if sec_nr in dtset:
                if dtset[sec_nr].attrs['units'] == 'unit':
                    dtset[sec_nr].values = dtset[sec_nr].values * 1e-6
                    dtset[sec_nr].attrs['units'] = 'cm-3'
                if first:
                    sum_sec = dtset[sec_nr].copy()
                    first = False
                else:

                    sum_sec = sum_sec + dtset[sec_nr]
            else:
                all_sec_in = False

            sec_nr = 'nrSO4_SEC%02.0f' % ii
            if sec_nr in dtset:
                if dtset[sec_nr].attrs['units'] == 'unit':
                    dtset[sec_nr].values = dtset[sec_nr].values * 1e-6
                    dtset[sec_nr].attrs['units'] = 'cm-3'
                if first:
                    sum_sec = dtset[sec_nr].copy()
                    first = False
                else:
                    sum_sec += dtset[sec_nr]
            else:
                all_sec_in = False
        if all_sec_in:
            dtset['SUM_SEC'] = sum_sec

    # get weights:
    wgts_ = area_pkg_sara.get_wghts(dtset['lat'].values)
    dtset['lat_wg'] = xr.DataArray(wgts_, coords=[dtset.coords['lat']], dims=['lat'], name='lat_wg')

    if (np.min(dtset['lon'].values) >= 0):
        print('xr_fix: shifting lon to -180-->180')
        dtset.coords['lon'] = (dtset['lon'] + 180) % 360 - 180
        dtset = dtset.sortby('lon')

    index = ['lev is dimension', 'orig_name', 'units']
    for var in dtset.data_vars:
        keys = []
        var_entery = []
        if 'orig_name' in dtset[var].attrs:
            keys.append('original_var_name')
            var_entery.append(dtset[var].attrs['orig_name'])
        if 'units' in dtset[var].attrs:
            keys.append('units')
            var_entery.append(dtset[var].attrs['units'])
        keys.append('lev_is_dim')
        var_entery.append(int('lev' in dtset[var].coords))
        var_overview_sql.open_and_create_var_entery(model_name,
                                                                   dtset.attrs['case_name'],
                                                                   var, var_entery, keys)

    dtset.attrs['startyear'] = int(dtset['time.year'].min())
    dtset.attrs['endyear'] = int(dtset['time.year'].max())

    return dtset


def xr_hybsigma2pressure(dtset, model_name, varList, path_savePressCoord=default_save_pressure_coordinates,
                         return_pressurevars=False, conv_vars=xr.Dataset(), save_conv_m=True,
                         savePressure_coord_file=True):
    """

    :param dtset:
    :param model_name:
    :param varList:
    :param path_savePressCoord:
    :param return_pressurevars:
    :param conv_vars:
    :param save_conv_m:
    :param savePressure_coord_file:
    :return:
    """
    # Get coordinates
    time = dtset['time'].values
    lat = dtset['lat'].values
    lon = dtset['lon'].values
    lev = dtset['lev'].values
    practical_functions.make_folders(path_savePressCoord)
    # Check if pressure coordinates needed for variables or if they can be found in file:
    case = dtset.attrs['case_name']
    print('xr_hybsigma2pressure: hybrid-sigma to pressure coordinates for %s, case: %s' % (model_name, case))

    # If not already pressure adjusted and not all vars already saved in
    # pressure coordinates:
    # check if conv variables already given as input:
    if not (len(conv_vars.data_vars) > 0):
        del conv_vars
        conv_vars = get_transformation_matrix(dtset, model_name,
                                              path_pressure_conversion_fields_path=default_conversion_data,
                                              save_conv_m=save_conv_m)

    index_lev = conv_vars['index_lev'].values
    count_lev = conv_vars['count_lev'].values

    if (dtset.attrs['Pres_addj']):
        print('Already calculated pressure coordinates?! Check!')
        if return_pressurevars:
            return dtset, conv_vars
        else:
            return dtset

    # Get pressure coordinate vars either from file or calculated:
    for var in list(set(varList) & set(dtset.variables)):
        dtarr, file_exists = practical_functions.open_pressure_coordinate_field(dtset, var, model_name,
                                                                                path_savePressCoord)
        # if no file exists, calculate:
        # file_exists=False
        if file_exists:
            #  read from file.
            print('xr_hybsigma2pressure: Reading pressure coordinate %s from file' % var)
            dtset[var].values = dtarr[var].values
            dtset[var].attrs['Pres_addj'] = True
            del dtarr
        else:
            print('Adjusting %s' % var)
            Field = dtset[var].values
            Field_pres_add = np.zeros(Field.shape)
            for la in np.arange(len(lat)):
                for lo in np.arange(len(lon)):
                    for ti in np.arange(len(time)):
                        for le in np.arange(len(lev)):
                            Field_pres_add[ti, int(index_lev[ti, le, la, lo]), la, lo] += Field[ti, le, la, lo] / \
                                                                                          count_lev[ti, int(index_lev[
                                                                                                                ti, le, la, lo]), la, lo]
                            if (count_lev[ti, int(index_lev[ti, le, la, lo]), la, lo] == 0): print(
                                'UPS: SOMETHING WRONG')
            # if no values added, let be nan
            Field_pres_add[count_lev == 0] = np.nan
            dtset[var].values = Field_pres_add
            dtset[var].attrs['Pres_addj'] = True
            if savePressure_coord_file:
                practical_functions.save_pressure_coordinate_field(dtset, var, model_name, path_savePressCoord)

    # Set pressure coordinates to true.
    dtset.attrs['Pres_addj'] = True

    del index_lev
    del count_lev
    # del pressure
    if return_pressurevars:
        return dtset, conv_vars
    else:
        return dtset


######################### GET TRANSFORMATION MATRICES###############################

def get_transformation_matrix(dtset, model_name, path_pressure_conversion_fields_path=default_conversion_data,
                              save_conv_m=True):
    time = dtset['time'].values
    lev = dtset['lev'].values
    lat = dtset['lat'].values
    lon = dtset['lon'].values
    PS = dtset['PS'].values

    startyear = dtset['time.year'].min().values
    endyear = dtset['time.year'].max().values
    not_entire_year = False
    if len(dtset['time']) < 12:
        not_entire_year = True
        start_month = dtset['time.month'].min().values
        end_month = dtset['time.month'].max().values
    case = dtset.attrs['case_name']

    # pressure  = np.empty([len(time), len(lev), len(lat),len(lon)])#hyam*1.e5+hybm*PS
    index_lev = np.empty([len(time), len(lev), len(lat), len(lon)])  # hyam*1.e5+hybm*PS
    count_lev = np.empty([len(time), len(lev), len(lat), len(lon)])  # hyam*1.e5+hybm*PS
    # filename to save or open transformation_matrices.
    if not_entire_year:
        filename = path_pressure_conversion_fields_path + \
                   '/%s/%s_%s_%s_%s-%s_%s-%s.nc' % \
                   (model_name, 'pres_coords_index_lev_count_lev', model_name, case, startyear, start_month, endyear,
                    end_month)
    else:
        filename = path_pressure_conversion_fields_path + \
                   '/%s/%s_%s_%s_%s_%s.nc' % \
                   (model_name, 'pres_coords_index_lev_count_lev', model_name, case, startyear, endyear)
    if os.path.isfile(filename):
        print('xr_hybsigma2pressure: Reading transformation matrix from file')

        conv_vars = xr.open_dataset(filename)  # , autoclose=True)
        # index_lev=dummy_press['index_lev'].values
        # count_lev=dummy_press['count_lev'].values
        # del dummy_press
        conv_vars.attrs['Pres_addj'] = True
        return conv_vars
    else:
        print('xr_hybsigma2pressure: Calculating pressure coordinate transformation')
        hyam, hybm = get_hybsig_coeffs(dtset, model_name)
        #####################
        timen = len(time)
        levn = len(lev)
        latn = len(lat)
        lonn = len(lon)
        hyam_matrix = np.repeat(hyam, latn * lonn).reshape([timen, levn, latn, lonn])
        hybm_matrix = np.repeat(hybm, latn * lonn).reshape([timen, levn, latn, lonn])
        PS_matrix = np.repeat(PS[:, np.newaxis, :, :], levn, axis=1)

        # pressure=hyai_matrix*1.e5+ hybi_matrix*PS_matrix
        if (model_name == 'NorESM'):
            pressure = hyam_matrix * 1.e5 + hybm_matrix * PS_matrix
        elif (model_name == 'ECHAM'):
            pressure = hyam_matrix + hybm_matrix * PS_matrix  #
        elif (model_name == 'EC-Earth'):
            pressure = hyam_matrix + hybm_matrix * PS_matrix  #

        #####################

        for la in np.arange(len(lat)):
            for lo in np.arange(len(lon)):
                for le in np.arange(len(lev)):

                    for ti in np.arange(len(time)):
                        diff = np.abs(lev - pressure[ti, le, la, lo] / 100.)
                        # print(pressure[ti,le,la,lo])
                        # print(diff)
                        index_lev[ti, le, la, lo] = int(diff.argmin())
                for ti in np.arange(len(time)):
                    dummy = index_lev[ti, :, la, lo]
                    for le in np.arange(len(lev)):
                        count_lev[ti, le, la, lo] = len(dummy[dummy == le])
    # save count_lev and index_lev in nc-file:
    # write pressure adjusted field to file:

    conv_vars = xr.Dataset(data_vars={'index_lev': (['time', 'lev', 'lat', 'lon'], index_lev),
                                      'count_lev': (['time', 'lev', 'lat', 'lon'], count_lev)},
                           coords={'time': time, 'lev': lev, 'lat': lat, 'lon': lon})
    conv_vars.attrs['Pres_addj'] = 'True'
    conv_vars.attrs['case_name'] = case
    conv_vars.time.encoding['units'] = 'days since 2000-01-01'
    conv_vars.time.encoding['calendar'] = 'standard'
    if ((len(path_pressure_conversion_fields_path) > 0 and not (os.path.isfile(filename))) and save_conv_m):
        practical_functions.make_folders(practical_functions.extract_path_from_filepath(filename))
        conv_vars.to_netcdf(filename)

    return conv_vars


######################### GET HYBRID SIGMA COEFFICIENTS###############################

def get_hybsig_coeffs(dtset, model_name):
    time = dtset['time'].values
    lev = dtset['lev'].values
    if (model_name == 'NorESM'):
        hyam = dtset['hyam'].values
        hybm = dtset['hybm'].values

    elif (model_name == 'ECHAM'):
        a_dummy = dtset['hyam'].values
        b_dummy = dtset['hybm'].values
        hyam = np.empty([len(time), len(a_dummy)])
        hybm = np.empty([len(time), len(a_dummy)])
        for ti in np.arange(len(time)):
            hyam[ti, :] = a_dummy
            hybm[ti, :] = b_dummy
    elif (model_name == 'EC-Earth'):
        if (len(lev) == 62):
            print('get_hybsig_coeff: Cant get hybm/byam for ifs yet')
        else:
            a_dummy = get_model_lev.get_hyam_TM5()
            b_dummy = get_model_lev.get_hybm_TM5()
            hyam = np.empty([len(time), len(a_dummy)])
            hybm = np.empty([len(time), len(a_dummy)])
            for ti in np.arange(len(time)):
                hyam[ti, :] = a_dummy
                hybm[ti, :] = b_dummy
    else:
        print('get_hybsig_coeff: Did not recognize model %s. Will crash very soon :/ :o :C ' % model_name)
    return hyam, hybm


def get_hybsig_int_coeffs(dtset, model_name):
    time = dtset['time'].values
    lev = dtset['lev'].values
    if (model_name == 'NorESM'):
        hyai = dtset['hyai'].values
        hybi = dtset['hybi'].values

    elif (model_name == 'ECHAM'):
        a_dummy = dtset['hyai'].values
        b_dummy = dtset['hybi'].values
        hyai = np.empty([len(time), len(a_dummy)])
        hybi = np.empty([len(time), len(a_dummy)])
        for ti in np.arange(len(time)):
            hyai[ti, :] = a_dummy
            hybi[ti, :] = b_dummy
    elif (model_name == 'EC-Earth'):
        if (len(lev) == 62):
            print('get_hybsig_coeff: Cant get hybm/byam for ifs yet')
        else:
            a_dummy = get_model_lev.get_hyai_TM5()
            b_dummy = get_model_lev.get_hybi_TM5()
            hyai = np.empty([len(time), len(a_dummy)])
            hybi = np.empty([len(time), len(a_dummy)])
            for ti in np.arange(len(time)):
                hyai[ti, :] = a_dummy
                hybi[ti, :] = b_dummy
    else:
        print('get_hybsig_coeff: Did not recognize model %s. Will crash very soon :/ :o :C ' % model_name)
    return hyai, hybi


def calculate_or_read_pressure(dtset, model_name, pressure_coord=False):
    print('calculate_or_read_pressure: Pressure for %s' % model_name)
    # dtset=calculate_or_read_pressure(dtset,model_name,Path_savePressCoord=Path_savePressCoord)
    # if 'T' in d
    # Get coordinates
    PS = dtset['PS'].values
    time = dtset['time'].values
    lat = dtset['lat'].values
    lon = dtset['lon'].values
    lev = dtset['lev'].values
    if pressure_coord:
        dtset['pressure'] = 1e2 * np.repeat(
            np.repeat(np.repeat(lev[np.newaxis, :], len(time), axis=0)[:, :, np.newaxis],
                      len(lat), axis=2)[:, :, :, np.newaxis], len(lon), axis=3)  # from hPa to Pa
        dtset['pressure'].attrs['Pres_addj'] = True
        dtset['pressure'].attrs['units'] = 'Pa'
        dtset['pressure'].attrs['Pres_addj'] = True
        return dtset
    # initialize arrays to hold pressure, index for grid box (which level it is closest to),
    # and count_lev: number of gridboxes added together in a box.
    # Hyam/hybm dependent on model:
    hyam, hybm = get_hybsig_coeffs(dtset, model_name)

    # Check if pressure coordinates needed for variables or if they can be found in file:
    startyear = dtset['time.year'].min().values
    endyear = dtset['time.year'].max().values
    case = dtset.attrs['case_name']

    # check if conv variables already given as input:
    dtarr, file_exists = practical_functions.open_pressure_coordinate_field(dtset, 'pressure', model_name,
                                                                            default_pressure_density_path)

    if not (file_exists):
        print('calculate_or_read_pressure: Calculating pressure field')
        #####################
        timen = len(time)
        levn = len(lev)
        latn = len(lat)
        lonn = len(lon)
        # print(levn)
        # print(hyai.shape)
        hyam_matrix = np.repeat(hyam, latn * lonn).reshape([timen, levn, latn, lonn])
        hybm_matrix = np.repeat(hybm, latn * lonn).reshape([timen, levn, latn, lonn])
        PSi_matrix = np.repeat(PS[:, np.newaxis, :, :], levn, axis=1)

        # pressure=hyai_matrix*1.e5+ hybi_matrix*PS_matrix
        if (model_name == 'NorESM'):
            pressure = hyam_matrix * 1.e5 + hybm_matrix * PSi_matrix
        elif (model_name == 'ECHAM'):
            pressure = hyam_matrix + hybm_matrix * PSi_matrix  # - (hyai[:,le-1]+ hybi[:,le-1]*PS[:,la,lo])
        elif (model_name == 'EC-Earth'):
            pressure = hyam_matrix + hybm_matrix * PSi_matrix  # - (hyai[:,le-1]+ hybi[:,le-1]*PS[:,la,lo])
        dtset['pressure'] = xr.DataArray(pressure,
                                         coords=[dtset.coords['time'], dtset.coords['lev'], dtset.coords['lat'],
                                                 dtset.coords['lon']])
        dtset['pressure'].attrs['Pres_addj'] = False
        dtset['pressure'].attrs['units'] = 'Pa'
        practical_functions.save_pressure_coordinate_field(dtset, 'pressure', model_name, default_pressure_density_path)
    else:
        dtset['pressure'] = dtarr['pressure']
        dtset['pressure'].attrs['Pres_addj'] = False

    return dtset


def calculate_or_read_density(dtset, model_name, path_to_data='', Path_savePressCoord='', press_coords=True):
    print('calculate_or_read_density: Calculate density  for %s' % model_name)

    if 'T' not in dtset.data_vars:
        casename = dtset.attrs['case_name']
        startyear = dtset['time.year'].min().values
        endyear = dtset['time.year'].max().values
        # print('----------------------------------------------------------')
        xr_ds = import_fields_xr.xr_import(casename, ['T'], path_to_data,
                                           model_name=model_name, comp='atm', from_year=startyear,
                                           to_year=endyear, EC_earth_comp='tm5')
        # print('----------------------------------------------------------')
        xr_ds = xr_fix(xr_ds, model_name)
        if press_coords:
            xr_ds = xr_hybsigma2pressure(xr_ds, model_name, ['T'], path_savePressCoord=Path_savePressCoord)

        TEMP = xr_ds['T']
        del xr_ds
    else:
        xr_ds = dtset.copy()
        if press_coords:
            xr_ds = xr_hybsigma2pressure(xr_ds, model_name, ['T'], path_savePressCoord=Path_savePressCoord)
        TEMP = xr_ds['T']
    if 'pressure' not in dtset.data_vars:
        dtset = calculate_or_read_pressure(dtset, model_name)  # , Path_savePressCoord=Path_savePressCoord)

    pressure = dtset['pressure']
    R_dry = 287.058
    density = pressure / (R_dry * TEMP)
    dtset['density'] = xr.DataArray(density, coords=[dtset.coords['time'], dtset.coords['lev'], dtset.coords['lat'],
                                                     dtset.coords['lon']])
    dtset['density'].attrs['Pres_addj'] = press_coords
    dtset['density'].attrs['units'] = 'kg/m3'

    return dtset


def perMassAir2perVolume(dtset, model_name, var, path_to_data='', Path_savePressCoord='', press_coords=True):
    print('perMassAir2perVolume: converting %s to per m3')

    old_unit = dtset[var].attrs['units']
    dtset = calculate_or_read_density(dtset, model_name, path_to_data=path_to_data,
                                      Path_savePressCoord=Path_savePressCoord, press_coords=press_coords)
    print(dtset)
    a = dtset[var] * dtset['density'].values
    dtset[var].values = a.values

    dtset[var].attrs['units'] = dtset[var].attrs['units'] + '*kg/m3'
    return dtset
