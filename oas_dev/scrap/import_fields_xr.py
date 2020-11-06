#############################################################################
# Opens nc files into xarray datasets for 3 models. Specially designed for 
# the structure of the files. 
#############################################################################
# importing modules and functions

import numpy as np  # For scientific computig
# np.set_printoptions(threshold=np.inf)
from os import listdir
import datetime
#import analysis_tools.translate_var_names
from sectional_v2.util.imports.import_fields_xr_v2 import get_vars_for_computed_vars, create_computed_fields
from sectional_v2.util.naming_conventions import translate_var_names, find_model_case_name
#import analysis_tools.var_overview_sql
#from analysis_tools import find_model_case_name, fix_xa_dataset
import sys
import xarray as xr

# Some variables should always be included if present:
always_include = ['area', 'landfrac', 'date', 'hyam', 'hybm', 'PS', 'gw', 'LANDFRAC', 'LOGR', 'Press_surf', 'ps', 'aps',
                  'hyai', 'hybi', 'ilev']
# Not currently in use:
EC_Earth_ifs_additionals = ['ClearSky_TOA_SW_net', 'Total_TOA_SW_net', 'ClearSky_surf_SW_net', 'Total_surf_SW_net',
                            'ClearSky_TOA_LW_net', 'Total_TOA_LW_net', 'ClearSky_surf_LW_net', 'Total_surf_LW_net',
                            'ClearSky_TOA_SW_net_woAer', 'Total_TOA_SW_net_woAer', 'ClearSky_surf_SW_net_woAer',
                            'Total_surf_SW_net_woAer', 'ClearSky_TOA_LW_net_woAer', 'Total_TOA_LW_net_woAer',
                            'ClearSky_surf_LW_net_woAer', 'Total_surf_LW_net_woAer']

EC_Earth_ifs = ['hyai', 'hybi', 'hyam', 'hybm', 'time', 'surf_SW_down', 'surf_LW_down', 'surf_SW_net', 'surf_LW_net',
                'TOA_SW_net', 'TOA_LW_net', 'TOA_SW_net_ClSky', 'TOA_LW_net_ClSky', 'surf_SW_net_ClSky',
                'surf_LW_net_ClSky', 'SST', 'LWP', 'IWP', 'Press_surf', 'MSLP', 'T2m', 'Temp3D', 'Geopot', 'Press3D',
                'cdnc', 're_liq', 'SWCF', 'LWCF']

EC_Earth_tm5 = ['CCN0.20', 'CCN1.00', 'od550aer', 'loadsoa', 'emiterp', 'emiisop', 'p_svoc2D', 'p_elvoc2D', 'loadisop',
                'loadterp', 'mass_frac_SOA', 'N_tot']

EC_Earth_tm5_additionals = ['prod_elvoc', 'prod_svoc', 'p_el_ohterp', 'p_el_o3terp', 'p_el_ohisop', 'p_el_o3isop',
                            'p_sv_ohterp', 'p_sv_o3terp', 'p_sv_ohisop', 'p_sv_o3isop', 'M_SOANUS', 'M_SOAAIS',
                            'M_SOAACS', 'M_SOACOS', 'M_SOAAII', 'N_NUS', 'N_AIS', 'N_ACS', 'N_COS', 'N_AII', 'N_ACI',
                            'N_COI', 'GAS_O3', 'GAS_TERP', 'GAS_OH', 'GAS_ELVOC', 'GAS_SVOC', 'GAS_ISOP', 'loadoa',
                            'loadbc', 'loadso4', 'loaddust', 'loadss', 'loadno3', 'od550so4', 'od550bc', 'od550oa',
                            'od550soa', 'od550ss', 'od550dust', 'od550no3', 'od550aerh2o', 'od550lt1aer',
                            'od550lt1dust', 'od550lt1ss', 'loadsoa']


def xr_import(caseName, varNames, path, model_name='NorESM', from_year=0, to_year=99999, mustInclude='.h0.', comp='atm',
              size_distrib=False, EC_earth_comp='tm5'):
    ''' This function imports data from nc files between startyear and endyear.
	the whole years as well as winter and summer. 
	Input:
	startyear - first year to average over
	endyear - last year to average over
	Case - Name of case run
	ctrlcase -  Name of control simulation
	var - name of variable
	lnd_var - list of land variables (i.e not atmopshere variables)
	vadList3D - list of varibles in 3D format (i.e has vertical levels)
	Path - Path to folder where cases are savedi
	comp -- atm/lnd for NorESM, ifs/tm5 for EC-earth. Not used for echam. 
	Returns:
	Output
	'''
    # Decides if data is imported from cam or clm
    if comp == 'lnd':
        model = 'clm2'
    else:
        model = 'cam'

    print('Model: ' + model_name)
    print('Variable list:')
    print(varNames)
    # print(path_mod)
    caseName_orig = caseName
    caseName = find_model_case_name.find_name(model_name, caseName)

    if model_name =='NorESM':
        varNames_mod = get_vars_for_computed_vars(varNames, model_name)
        drop_list, pathfile_list = filelist_NorESM(caseName, comp, from_year, model, model_name, mustInclude, path, to_year,
                                               varNames_mod)
        dict_of_vars = xr.open_mfdataset(pathfile_list, decode_times=False, drop_variables=drop_list, combine='by_coords')#, autoclose=True)
        dict_of_vars = create_computed_fields(dict_of_vars, varNames, model_name)
        varNames = varNames_mod
    elif (model_name == 'EC-Earth'):
        varNames_mod = get_vars_for_computed_vars(varNames, model_name)
        drop_list, pathfile_list_add, pathfile_list_norm, decode_times = filelist_ECEarth(EC_earth_comp, caseName, from_year, path,
                                                                            size_distrib, to_year, varNames_mod)
        if (len(pathfile_list_norm) > 0):
            dict_of_vars_norm = open_mfdataset_merge_only(pathfile_list_norm, decode_times=decode_times, drop_variables=drop_list, combine='by_coords')
            #dict_of_vars_norm = xr.open_mfdataset(pathfile_list_norm, decode_times=False, drop_variables=drop_list,
            #                                      compat='no_conflicts',
            #                                      concat_dim='time')
            if (len(pathfile_list_add) > 0):
                dict_of_vars_add = xr.open_mfdataset(pathfile_list_add, decode_times=decode_times, drop_variables=drop_list,combine='by_coords')
                dict_of_vars = xr.merge([dict_of_vars_add, dict_of_vars_norm])
            else:
                dict_of_vars = dict_of_vars_norm
        else:
            dict_of_vars = xr.open_mfdataset(pathfile_list_add, decode_times=False, drop_variables=drop_list,combine='by_coords')
        dict_of_vars = create_computed_fields(dict_of_vars, varNames, model_name)

    elif (model_name == 'ECHAM'):
        print('ECHAM:')
        varNames_mod = get_vars_for_computed_vars(varNames, model_name)
        drop_list, filename = filename_ECHAM(caseName, path, varNames_mod)

        dummy = xr.open_dataset(filename, decode_times=True, drop_variables=drop_list)#, autoclose=True)
        # dict_of_vars=dummy.sel('%f.0-01-01' %(from_year):'%f.0-01-01' %(to_year))
        dict_of_vars = dummy.sel(time=slice('%.0f-01-01' % (from_year), '%.0f-01-01' % (to_year + 1)))
        dict_of_vars = create_computed_fields(dict_of_vars, varNames, model_name)


    for key in dict_of_vars.variables:
        dict_of_vars[key].attrs['Pres_addj'] = False
    dict_of_vars.attrs['Pres_addj'] = False
    dict_of_vars.attrs['case_name'] = caseName_orig
    dict_of_vars.attrs['startyear'] = from_year
    dict_of_vars.attrs['endyear'] = to_year


    return dict_of_vars


def importBVOC_ECHAM(from_year, to_year, path):
    dtset = xr.open_dataset(path+'bvoc_emissions_annual.nc', decode_times=False)
    dtset['time'].values = np.array([datetime.datetime(year=2000, month=1, day=1)])
    return dtset

def open_mfdataset_merge_only(paths, **kwargs):
    #if isinstance(paths, basestring()):
    #    paths = sorted(glob(paths))
    return xr.merge([xr.open_dataset(path, **kwargs) for path in paths])

def filename_ECHAM(caseName, path, varNames):
    filename = path + 'bacchus.echam.d4.4.' + caseName + '.nc'
    print(filename)

    path_mod = path  # path to files
    varNames = [translate_var_names.NorESM2model(var, 'ECHAM') for var in varNames]
    #varNames = [analysis_tools.var_names_2_noresm_var_names.get_var_name_ECHAM(var) for var in varNames]
    dummy = xr.open_dataset(filename)
    drop_list = list((set(dummy.data_vars.keys()) - set(varNames)) - set(always_include))
    dummy.close()
    return drop_list, filename


def filelist_ECEarth(EC_earth_comp, caseName, from_year, path, size_distrib, to_year, varNames, month=-1, year=-1):
    '''
    Gets lists of files to be opened EC-Earth
    :param EC_earth_comp:
    :param caseName:
    :param from_year:
    :param path:
    :param size_distrib:
    :param to_year:
    :param varNames:
    :param month: If only one month, insert here
    :return:
    '''
    print(caseName)
    ifs = False
    tm5 = False
    for var in varNames:
        if var in translate_var_names.varComp_EC_Earth['ifs']: ifs=True
        if var not in translate_var_names.varComp_EC_Earth['ifs']: tm5 =True
    if ifs and tm5: sys.exit('Variables from both ifs and tm5 cannot be loaded into same dic')
    if ifs: EC_earth_comp = 'ifs'
    print('EC EARTCH COMP *******')
    print(EC_earth_comp)
    mustNotInclude = 'CLCV'
    mustInclude = EC_earth_comp
    #mustInclude = '_' #test removing constraint.
    path_mod = path + caseName + '/'  # path to files
    print(path_mod)
    if size_distrib:  # include sizedist:
        filelist_d = [f for f in listdir(path_mod) if ((mustInclude in f) and (
                'SizeDist.nc' in f or '_ps.nc' in f))]  # and (mustNotInclude not in f))]	#list of filenames with correct req in path folder
    else:  # exclude sizedist
        filelist_d = [f for f in listdir(path_mod) if ((mustInclude in f) and (
                'SizeDist' not in f) and ('CLCV' not in f) and ('3D' not in f))]  # list of filenames with correct req in path folder
    filelist_d = [f for f in filelist_d if f[(len(caseName) + 1):(len(caseName) + 5)].isdigit()]
    #print(filelist_d)
    # picks out years from files:
    if ((caseName == 'noIsop') and EC_earth_comp == 'tm5'):
        skift = 0
        print('noIsop/ALLm50')
    else:
        skift = 1
    filelist_y = [f[(len(caseName) + skift):(len(caseName) + 4 + skift)] for f in filelist_d if
                  f[(len(caseName) + 1):(len(caseName) + 5)].isdigit()]
    filelist_ym = [f[(len(caseName) + skift):(len(caseName) + 6 + skift)] for f in filelist_d if
                   f[(len(caseName) + 1):(len(caseName) + 5)].isdigit()]
    sortargs = np.array(np.array([int(f) for f in filelist_ym])).argsort()
    filelist_d = [filelist_d[i] for i in sortargs]
    filelist_y = [filelist_y[i] for i in sortargs]
    filelist_ym = [filelist_ym[i] for i in sortargs]
    if month>0 and year>=0:
        filelist_d=[filelist_d[i] for i in np.arange(len(filelist_d)) if filelist_ym[i]=='%04d%02d' %(year,month)]
        filelist_y=[filelist_y[i] for i in np.arange(len(filelist_y)) if filelist_ym[i]=='%04d%02d' %(year,month)]
        filelist_ym=[filelist_ym[i] for i in np.arange(len(filelist_ym)) if filelist_ym[i]=='%04d%02d' %(year,month)]
    # Reads from two sets of files:
    if size_distrib:
        filelist_mod_norm = [filelist_d[i] for i in np.arange(len(filelist_d)) if
                             ((int(filelist_y[i]) >= from_year and int(filelist_y[i]) <= to_year))]
        filelist_mod_add = []
    else:
        filelist_mod_norm = [filelist_d[i] for i in np.arange(len(filelist_d)) if (
                (int(filelist_y[i]) >= from_year and int(filelist_y[i]) <= to_year) and (
                'onal.nc' not in filelist_d[i]) and 'M7-rates.nc' not in filelist_d[i])]
        filelist_mod_add = [filelist_d[i] for i in np.arange(len(filelist_d)) if (
                (int(filelist_y[i]) >= from_year and int(filelist_y[i]) <= to_year) and (
                'onal.nc' in filelist_d[i]))]
    #print(filelist_mod_norm)
    varNames = [translate_var_names.NorESM2model(var, 'EC-Earth') for var in varNames]
    print(varNames)
    drop_list_norm = []
    drop_list_add = []
    # Open dataset that includes all variations for 1 month:
    # flist_1month=[(path_mod+ f) for f in (filelist_mod_norm+filelist_mod_add) if ((str(from_year)+'01') in f)]
    if month>=0 and year>=0:
        firstmonth='%04d%02d' %(year,month)
    else:
        firstmonth =str(from_year)+'01'
    flist_1month = [(path_mod + f) for f in (filelist_mod_norm + filelist_mod_add) if ( firstmonth in f)]
    dummy = xr.open_mfdataset(flist_1month, decode_times=False,combine='by_coords')
    drop_list = list((set(dummy.data_vars.keys())) - set(varNames) - set(always_include))
    dummy.close()
    pathfile_list_norm = []
    pathfile_list_add = []
    print('Importing following files:****************************************')
    for f in filelist_mod_norm:
        pathfile_list_norm.append(path_mod + f)
        print(f)
    for f in filelist_mod_add:
        pathfile_list_add.append(path_mod + f)
        print(f)
    return drop_list, pathfile_list_add, pathfile_list_norm, EC_earth_comp=='ifs'

def filelist_NorESM(caseName, comp, from_year, model, model_name, mustInclude, path, to_year, varNames, month=-1, year=-1):
    print('Case: ' + caseName)
    path_mod = path + caseName + '/' + comp + '/hist/'  # path to files
    print(path_mod)
    filelist_d = [f for f in listdir(path_mod) if
                  ((mustInclude in f) and f[0] != '.')]  # list of filenames with correct req in path folder
    start_year=len(caseName) + len(model) + 5
    start_month=start_year+4+1
    filelist_y = [f[start_year:start_year+4] for f in filelist_d]

    # pick out months from filenames:
    filelist_m = [f[start_month:start_month+2] for f in filelist_d]
    # make list of year and month from which to sort:
    filelist_ym = [filelist_y[i] + filelist_m[i] for i in np.arange(len(filelist_y))]  #

    # sort files:
    sortargs = np.array(np.array([int(f) for f in filelist_ym])).argsort()
    filelist_d = [filelist_d[i] for i in sortargs]
    filelist_y = [filelist_y[i] for i in sortargs]
    # modify: only take years that should be in:
    filelist_mod = [filelist_d[i] for i in np.arange(len(filelist_d)) if
                    (int(filelist_y[i]) >= from_year and int(filelist_y[i]) <= to_year)]
    if (year>=0 and month>0): # Extract one month only
        filelist_mod = [f for f in filelist_mod if f[start_year:start_year+4+1+2] == '%04d-%02d' %(year, month)]

    print('Importing following files:')
    for f in filelist_mod:
        print(f)

    drop_list = []

    f = filelist_mod[0]
    dummy = xr.open_dataset(path_mod + f)
    # print(dummy.keys())
    drop_list = list((set(dummy.data_vars.keys()) - set(varNames)) - set(always_include))
    dummy.close()
    pathfile_list = []
    for f in filelist_mod:
        pathfile_list.append(path_mod + f)

    return drop_list, pathfile_list

def xr_import_one_month(caseName, varNames, path, year, month, model_name='NorESM', mustInclude='.h0.', comp='atm',
                        size_distrib=False, EC_earth_comp='tm5'):
    '''
	Imports one month of data
	:param caseName: case
	:param varNames: variables to be loaded
	:param path: path to data
	:param year: year to bie loaded
	:param month: month to be loaded
	:param model_name: 'NorESM', 'ECHAM' or 'EC-Earth'
	:param mustInclude:
	:param comp:
	:param size_distrib:
	:param EC_earth_comp:
	:return:
	'''
    # Decides if data is imported from cam or clm
    if comp == 'lnd':
        model = 'clm2'
    else:
        model = 'cam'

    print('Model: ' + model_name)
    print('Variable list:')
    print(varNames)
    # print(path_mod)
    caseName_orig = caseName
    caseName = find_model_case_name.find_name(model_name, caseName)

    if model_name =='NorESM':
        drop_list, pathfile_list = filelist_NorESM(caseName, comp, year, model, model_name, mustInclude, path, year,
                                               varNames,year=year, month=month)
        dict_of_vars = xr.open_mfdataset(pathfile_list, decode_times=False, drop_variables=drop_list, autoclose=True,combine='by_coords')


    elif (model_name == 'EC-Earth'):
        drop_list, pathfile_list_add, pathfile_list_norm, dummy= filelist_ECEarth(EC_earth_comp, caseName, year, path,
                                                                            size_distrib, year, varNames, year=year, month=month)
        if (len(pathfile_list_norm) > 0):
            dict_of_vars_norm = xr.open_mfdataset(pathfile_list_norm, decode_times=False, drop_variables=drop_list,
                                                  autoclose=True,combine='by_coords')
            if (len(pathfile_list_add) > 0):
                dict_of_vars_add = xr.open_mfdataset(pathfile_list_add, decode_times=False, drop_variables=drop_list,
                                                     autoclose=True,combine='by_coords')
                dict_of_vars = xr.merge([dict_of_vars_add, dict_of_vars_norm])
            else:
                dict_of_vars = dict_of_vars_norm
        else:
            dict_of_vars = xr.open_mfdataset(pathfile_list_add, decode_times=False, drop_variables=drop_list,
                                             autoclose=True,combine='by_coords')



    elif (model_name == 'ECHAM'):
        print('ECHAM:')
        print(caseName)
        drop_list, filename = filename_ECHAM(caseName, path, varNames)

        dummy = xr.open_dataset(filename, decode_times=True, drop_variables=drop_list, autoclose=True)
        # dict_of_vars=dummy.sel('%f.0-01-01' %(from_year):'%f.0-01-01' %(to_year))
        from_ym='%04d-%02d-01'% (year, month)
        to_ym = '%04d-%02d-01'% (year, month+1)
        if month==12:
            to_ym = '%04d-%02d-01'% (year+1, 1)

        dict_of_vars = dummy.sel(time=slice(from_ym, to_ym))


    for key in dict_of_vars.variables:
        dict_of_vars[key].attrs['Pres_addj'] = False
    dict_of_vars.attrs['Pres_addj'] = False
    dict_of_vars.attrs['case_name'] = caseName_orig

    return dict_of_vars



