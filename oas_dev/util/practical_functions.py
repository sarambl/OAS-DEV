import os
import sys

import numpy as np
import pandas as pd
import xarray as xr
# import analysis_tools.naming_conventions.var_info
from sectional_v2.util.filenames import get_filename_pressure_coordinate_field

from sectional_v2.util.naming_conventions import var_info


def make_folders(path):
    """
    Takes path and creates to folders
    :param path: Path you want to create (if not already existant)
    :return: nothing
    """
    path = extract_path_from_filepath(path)
    split_path = path.split('/')
    if (path[0] == '/'):

        path_inc = '/'
    else:
        path_inc = ''
    for ii in np.arange(len(split_path)):
        # if ii==0: path_inc=path_inc+split_path[ii]
        path_inc = path_inc + split_path[ii]
        if not os.path.exists(path_inc):
            os.makedirs(path_inc)
        path_inc = path_inc + '/'

    return
def append2dic(self, ds_append, ds_add):
    for key in ds_add.attrs.keys():
        if key not in ds_append.attrs:
            ds_append.attrs[key] = ds_add.attrs[key]
    return ds_append

def extract_path_from_filepath(file_path):
    """
    ex: 'folder/to/file.txt' returns 'folder/to/'
    :param file_path:
    :return:
    """

    st_ind=file_path.rfind('/')
    foldern = file_path[0:st_ind]+'/'
    return foldern


def save_dataset_to_netcdf(dtset, filepath):
    """

    :param dtset:
    :param filepath:
    :return:
    """
    dummy = dtset.copy()

    # dummy.time.encoding['calendar']='standard'
    go_through_list= list(dummy.coords)
    if isinstance(dummy, xr.Dataset):
        go_through_list = go_through_list + list(dummy.data_vars)
    for key in go_through_list:
        if 'Pres_addj' in dummy[key].attrs:
            if dummy[key].attrs['Pres_addj']:
                dummy[key].attrs['Pres_addj'] = 'True'
            else:
                dummy[key].attrs['Pres_addj'] = 'False'

    if ('Pres_addj' in dummy.attrs):
        if dummy.attrs['Pres_addj']:
            dummy.attrs['Pres_addj'] = 'True'
        else:
            dummy.attrs['Pres_addj'] = 'False'
    if 'time' in dummy.coords:
        if 'units' in dummy['time'].attrs:
            del dummy['time'].attrs['units']
        if 'calendar' in dummy['time'].attrs:
            del dummy['time'].attrs['calendar']
    print('Saving dataset to: '+ filepath)
    make_folders(filepath)
    dummy.load()
    dummy.to_netcdf(filepath, mode='w')  # ,encoding={'time':{'units':'days since 2000-01-01 00:00:00'}})
    del dummy
    return





#def get_varn_eusaar_comp(varn):


def save_pressure_coordinate_field(dtset, var, model, path_savePressCoord):

    if (not dtset[var].attrs['Pres_addj']):
        print('Not pressure adjusted! Will not save')
    else:
        argmax=dtset['time'].argmax().values
        argmin=dtset['time'].argmin().values
        if 'startyear' in dtset.attrs: startyear = dtset.attrs['startyear']
        else: startyear = dtset['time.year'].min().values
        if 'endyear' in dtset.attrs: endyear = dtset.attrs['endyear']
        else: endyear = dtset['time.year'].max().values
        startmonth = dtset['time.month'][argmin].values
        endmonth = dtset['time.month'][argmax].values
        case = dtset.attrs['case_name']
        filename, filename_m = get_filename_pressure_coordinate_field(case, dtset, endmonth, endyear, model, path_savePressCoord,
                                                          startmonth, startyear, var)
        dummy = dtset[var].copy()
        if ('calendar' in dummy['time'].attrs):
            del dummy['time'].attrs['calendar']
        if ('units' in dummy['time'].attrs):
            del dummy['time'].attrs['units']
        dummy.time.encoding['units'] = 'days since 2000-01-01'
        dummy.time.encoding['calendar'] = 'standard'
        for key in dummy.coords:
            # print(dummy.coords[key])
            if 'Pres_addj' in dummy[key].attrs:
                dummy[key].attrs['Pres_addj']= boolean_2_string(dummy[key].attrs['Pres_addj'])
        print( boolean_2_string(dummy.attrs['Pres_addj']))
        if ('Pres_addj' in dummy.attrs):
            dummy.attrs['Pres_addj'] = boolean_2_string(dummy.attrs['Pres_addj'])

        make_folders(extract_path_from_filepath(filename))
        print('Saving %s pressure coordinate field to file %s' %(var,filename))
        if len(dtset['time'])<12:
            dummy.to_netcdf(filename_m, mode='w')  # ,encoding={'time':{'units':'days since 2000-01-01 00:00:00'}})
        else:
            dummy.to_netcdf(filename, mode='w')  # ,encoding={'time':{'units':'days since 2000-01-01 00:00:00'}})
        del dummy
    # check_dummy=xr.open_dataarray(filename)

    # print(check_dummy['time'].values)
def boolean_2_string(b):
    if type(b) is bool:
        if b:
            return 'True'
        else:
            return 'False'
    else:
        return b

#def get_filename_pressure_coordinate_field(case, dtset, endmonth, endyear, model, path_savePressCoord, startmonth,
#                                           startyear, var):
#    filename_m = path_savePressCoord + '/%s/%s_%s_%s_%s-%s_%s-%s.nc' % (model, var, model, case, startyear, startmonth, endyear, endmonth)
#    filename_m = filename_m.replace(" ", "_")
#    filename = path_savePressCoord + '/%s/%s_%s_%s_%s_%s.nc' % (model, var, model, case, startyear, endyear)
#    filename = filename.replace(" ", "_")
#    return filename, filename_m


def open_pressure_coordinate_field(dtset, var, model, path_savePressCoord):
    startyear = dtset['time.year'].min().values
    endyear = dtset['time.year'].max().values
    argmax=dtset['time'].argmax().values
    argmin=dtset['time'].argmin().values
    startmonth = dtset['time.month'][argmin].values
    endmonth = dtset['time.month'][argmax].values
    case = dtset.attrs['case_name']

    filename, filename_m = get_filename_pressure_coordinate_field(case, dtset, endmonth, endyear, model, path_savePressCoord,
                                                                  startmonth, startyear, var)
    print('Reading pressure coordinate from file:')
    print(filename, filename_m)
    #filename = path_savePressCoord + '/%s/%s_%s_%s_%s_%s.nc' % (model, var, model, case, startyear, endyear)
    #filename.replace(" ", "_")
    print('Checking for %s or %s' %(filename,filename_m))
    # if (not dummy.attrs['Pres_addj']):
    if not os.path.isfile(filename) and not os.path.isfile(filename_m):
        print('file doesnt exist. Returns unadjusted file')
        return dtset, False
    else:
        if os.path.isfile(filename):
            dummy = xr.open_dataset(filename)#, autoclose=True)dd
            if startmonth!=1 or endmonth!= 12:
                #print(dummy.time.values)
                dummy = dummy.sel(time = slice(dtset['time'].min(), dtset['time'].max()))#(time=slice(test['time'].min(), test['time'].max()))
        else:
            dummy = xr.open_dataset(filename_m)#, autoclose=True)
        if len(dtset.time)!= len(dummy.time):
            dummy = dummy.sel(time=slice(dtset.time.min(), dtset.time.max()))
        if (var == 'pressure'):
            dtset[var] = dummy[var]
        dtset[var].values = dummy[var].values  # .copy()
        dtset[var].attrs = dummy[var].attrs
        dtset[var].attrs['Pres_addj'] = True
        del dummy
        return dtset, True


def add_variable_info_to_model_info_csv(model_name, df_var_info, var, index_key, value):
    var_mod_info_filen = '%s_variable_info.csv' % model_name
    if index_key not in df_var_info.index:
        df_var_info, var_mod_info_filen = add_index_model_info_csv([index_key], model_name)
    if var not in df_var_info:
        df_var_info[var]=np.zeros(len(df_var_info))*np.nan
    df_var_info.loc[index_key, var] = value
    df_var_info.to_csv(var_mod_info_filen+'_save.csv')
    df_var_info.to_csv(var_mod_info_filen)
    return


def open_model_info_csv(model_name):
    var_mod_info_filen = '%s_variable_info.csv' % model_name
    if os.path.isfile(var_mod_info_filen):
        df_var_info = pd.read_csv(var_mod_info_filen, index_col=0)

    else:
        df_var_info = pd.DataFrame()
    return df_var_info, var_mod_info_filen


def add_index_model_info_csv(index, model_name):
    df_var_info, var_mod_info_filen= open_model_info_csv(model_name)
    df_dummy = pd.DataFrame(index=index)
    df_var_info = df_var_info.merge(df_dummy, how='outer', right_index=True, left_index=True)
    return df_var_info, var_mod_info_filen


def get_foldername_Nd(caseName, from_year, model_name, pressure_adjust, to_year, from_diam, to_diam):
    filename = get_filename_Nd(caseName, from_year, model_name, pressure_adjust, to_year, from_diam, to_diam)
    st_ind=filename.rfind('/')
    foldern = filename[0:st_ind]
    return foldern


def get_filename_Nd(caseName, from_year, model_name, pressure_adjust, to_year, from_diam, to_diam):
    if pressure_adjust:
        filen = dataset_path_Nd + '/' + model_name +  '/%s_%s_%s_%s_dmin%d_maxd%d_press_adj.nc' % (
            model_name, caseName, from_year, to_year, from_diam, to_diam)
    else:
        filen = dataset_path_Nd + '/' + model_name +  '/%s_%s_%s_%s_dmin%d_maxd%d.nc' % (
            model_name, caseName, from_year, to_year, from_diam, to_diam)
    return filen


def get_filename_Nd_from_varName(varName, caseName, from_year, model_name, pressure_adjust, to_year):
    """
    Get filename from varName for N_d variable
    :param varName:
    :param caseName:
    :param from_year:
    :param model_name:
    :param pressure_adjust:
    :param to_year:
    :return:
    """
    n_split = varName.split('_')
    if len(n_split)==3:
        from_diam = int(n_split[0][1:])
        to_diam = int(n_split[-1])
    else:
        from_diam=0
        to_diam = int(n_split[-1])
    filen = get_filename_Nd(caseName, from_year, model_name, pressure_adjust, to_year, from_diam, to_diam)
    return filen


dataset_path='Data/Avg_sizedist_datasets'
dataset_path_Nd = 'Data/Datasets_Nd'


