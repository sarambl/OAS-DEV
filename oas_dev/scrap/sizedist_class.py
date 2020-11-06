# from useful_scit.imps import *
import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt
from os import listdir
# from os.path import isfile, join
from datetime import datetime
from datetime import timedelta
import matplotlib.colors as colors
# from netCDF4 import num2date, date2num
from sectional_v2.util.slice_average import area_mod
from sectional_v2.util.naming_conventions import find_model_case_name
from sectional_v2.util import practical_functions
from sectional_v2.scrap import fix_xa_dataset
from sectional_v2 import constants
default_savepath = constants.get_outdata_path('sizedistrib_files')
#calender_spec = 'days since 2007-1-1 00:00:00'
calender_type = 'gregorian'

varListNorESM = {'NCONC': ['NCONC01', 'NCONC02', 'NCONC04', 'NCONC05', 'NCONC06', 'NCONC07', 'NCONC08',
                           'NCONC09', 'NCONC10', 'NCONC12', 'NCONC14'],
                 'SIGMA': ['SIGMA01', 'SIGMA02', 'SIGMA04', 'SIGMA05', 'SIGMA06', 'SIGMA07', 'SIGMA08',
                           'SIGMA09', 'SIGMA10', 'SIGMA12', 'SIGMA14'],
                 'NMR': ['NMR01', 'NMR02', 'NMR04', 'NMR05', 'NMR06', 'NMR07', 'NMR08',
                         'NMR09', 'NMR10', 'NMR12', 'NMR14']}

SOA_SEC = ['nrSOA_SEC01', 'nrSOA_SEC02', 'nrSOA_SEC03', 'nrSOA_SEC04', 'nrSOA_SEC05']

SO4_SEC = ['nrSO4_SEC01', 'nrSO4_SEC02', 'nrSO4_SEC03', 'nrSO4_SEC04', 'nrSO4_SEC05']

list_noresm = varListNorESM['NCONC'] + varListNorESM['SIGMA'] + varListNorESM['NMR'] + SOA_SEC + SO4_SEC
list_noresm_nonsec = varListNorESM['NCONC'] + varListNorESM['SIGMA'] + varListNorESM['NMR']
places2coords = {'Hyytiala':[62,24], 'Beijing':[39.9,116.4]}


#######################################################
###  Creates and returns the right dataset 
#######################################################
def createRightDataset(case_name, model_name, from_time, to_time, data_path, locations=[], loc_dataset=False, print_stat=False):
    """

    :param case_name: case_name as written
    :param model_name: model name
    :param from_time: string: from ti
    :param to_time:
    :param data_path: path to raw data
    :param locations: if data output for specific locations, list these here
    :param loc_dataset: if location output
    :return:
    """
    if model_name == 'NorESM':
        if loc_dataset:
            if len(locations) > 0:
                return NorESM_SizedistDataset_spec_lat_lon_output(case_name, model_name, from_time, to_time, data_path, print_stat=print_stat)
            else:
                return NorESM_SizedistDataset_spec_lat_lon_output(case_name, model_name, from_time, to_time, data_path, print_stat=print_stat)
        else:
            return NorESM_SizedistDataset(case_name, model_name, from_time, to_time, data_path, print_stat=print_stat)

    else:
        print('Model name not recognized')


#######################################################
###  CLASS TO HOLD SIZEDISTRIBUTION DATASET
#######################################################
class SizedistDataset:
    # dic_of_area_averages=[]
    # dic_of_points=[]
    def __init__(self, case_name, model_name, from_time, to_time, data_path, print_stat=False):
        self.model_name = model_name
        self.case_plotting_name = model_name
        self.case_name = find_model_case_name.find_name(model_name, case_name)
        self.data_path = data_path
        self.from_time = from_time
        self.to_time = to_time
        self.savepath = '%s/%s/%s' % (default_savepath, model_name, case_name)
        self.print = print_stat
        if print_stat:
            print(self.data_path)
    def print_attrs(self):
        for attr, value in self.__dict__.items():
            print('Attribute name:' + str(attr or ""))
            print('Attr value: ' + str(value or ""))


class NorESM_SizedistDataset(SizedistDataset):
    # def __init__(self, case_name, model_name, from_time,to_time, data_path):

    def import_data_seq(self, var_names, comp, history_field):

        return

    def import_data(self, var_names, comp, history_field, remove_extra_vars=True,
                    monthly=False, save_to_obj=True,
                    from_time='', to_time='', concatinated_file=False, years=[2008,2008]):
        """
        Imports the data from apropriate files
        :param var_names:
        :param comp:
        :param history_field: '.h1.'/'.h2.' etc
        :param remove_extra_vars: Boolean
        :param monthly: True if loading monthly files.
        :param save_to_obj: If true, saves the imported data to the object
        :param from_time: str, if set selects data from a specified file
        :param to_time:
        :return:
        """
        if len(from_time) == 0:
            from_time = self.from_time
        if len(to_time) == 0:
            to_time = self.to_time

        path_mod = self.data_path + self.case_name + '/' + comp + '/hist/'
        filelist_d = [f for f in listdir(path_mod) if ((history_field in f) and f[0] != '.') and ('concat' not in f)]
        if monthly:
            filelist_time = [
                f[(len(self.case_name) + len(self.model_name) + 2):(len(self.case_name) + len(self.model_name) + 9)] for
                f in filelist_d]
            print(filelist_time)
            filelist_date = [datetime.strptime(f, '%Y-%m') for f in filelist_time]
        else:
            filelist_time = [
                f[(len(self.case_name) + len(self.model_name) + 2):(len(self.case_name) + len(self.model_name) + 12)]
                for f in filelist_d]
            filelist_date = [datetime.strptime(f, '%Y-%m-%d') for f in filelist_time]
        from_dt = datetime.strptime(from_time, '%Y-%m-%d')
        to_dt = datetime.strptime(to_time, '%Y-%m-%d')
        tf = np.array([to_dt >= filelist_date[i] >= from_dt for i in np.arange(len(filelist_d))])
        if self.print and not concatinated_file:
            print(np.array(filelist_d)[tf])
        from_time_n = filelist_time[np.array(filelist_date)[tf].argmin()]
        to_time_n = filelist_time[np.array(filelist_date)[tf].argmax()]
        import_list = np.array(filelist_d)[tf]
        pathfile_list = [path_mod + imp for imp in import_list]
        if concatinated_file:
            print('CONCATINATED FILE')
            filename = '%s.cam.%s.concat_%4d-%4d.nc'%(self.case_name, history_field.replace('.',''), years[0], years[1])
            filelist_conc = [f for f in listdir(path_mod) if ((history_field in f) and f[0] != '.') and ('concat' in f) and ('tmp' not in f)]
            pathfile_list = [path_mod + f for f in filelist_conc]
            print(pathfile_list)
        dataset_vars = self.import_filelist(pathfile_list, var_names, remove_extra_vars=remove_extra_vars)
        if save_to_obj:
            self.dataset = dataset_vars

            self.time = dataset_vars['time']
            if 'lat' in dataset_vars:
                self.lat = dataset_vars['lat']
            if 'lon' in dataset_vars:
                self.lon = dataset_vars['lon']
            self.lev = dataset_vars['lev']

            logD = np.logspace(0, 4, 50)  # , name='logD'
            self.size_dtset = xr.Dataset(coords={'time': self.time, 'logD': logD})  # , dims=['time', 'logD'])


        return dataset_vars

    def import_filelist(self, pathfile_list, var_names, remove_extra_vars=True):
        """
        Imports filelist.
        :param pathfile_list: List of files.
        :param var_names: list of variable names
        :param remove_extra_vars: boolean
        :return: returns dataset with specified variables
        """
        dataset_vars = xr.open_mfdataset(pathfile_list,combine='by_coords')#,
                                         #decode_times=False)  # ,chunks={'lat': 10, 'lon': 10} )#, autoclose=True)
        if remove_extra_vars:
            dataset_vars = dataset_vars.drop(list(set(dataset_vars.data_vars) - set(list_noresm)))
        #time = dataset_vars.time.values
        # print(time)
        #print(dataset_vars['time'].attrs)
        dataset_vars = dataset_vars.sortby('time')

        return dataset_vars

    #######################################################
    ###  CHANGE UNITS
    ### fixes units
    #######################################################
    def convert_to_pressure_coordinates(self, varList):
        dataset = self.dataset
        dtset = fix_xa_dataset.xr_hybsigma2pressure(dataset, self.model_name, varList)


    def change_units(self, dataset=None):
        """

        :param dataset:
        :return:
        """
        if dataset is None:
            dataset=self.dataset
        for i in np.arange(14):
            varN = 'NCONC%02.0f' % (i + 1)
            # varSIG = 'SIGMA%02.0f'%(i+1)
            varNMR = 'NMR%02.0f' % (i + 1)
            if varN in dataset.data_vars:
                #print(dataset.d)
                if dataset[varN].attrs['units'] != 'cm-3':
                    dataset[varN] = dataset[varN] * 1e-6  # m-3 --> cm-3
                    dataset[varN].attrs['units'] = 'cm-3'
            if varNMR in dataset.data_vars:
                if dataset[varNMR].attrs['units'] != 'nm':
                    dataset[varNMR] = dataset[varNMR] * 2 * 1e9  # m --> nm, radius --> diameter
                    dataset[varNMR].attrs['units'] = 'nm'
        cont = True
        i = 1
        while cont:
            varSEC = 'nrSO4_SEC%02.0f' % i
            if varSEC in dataset.data_vars:
                if dataset[varSEC].attrs['units'] != 'cm-3':
                    dataset[varSEC] = dataset[varSEC] * 1e-6  # m-3 --> cm-3
                    dataset[varSEC].attrs['units'] = 'cm-3'
            varSEC = 'nrSOA_SEC%02.0f' % i
            if varSEC in dataset.data_vars:
                if dataset[varSEC].attrs['units'] != 'cm-3':
                    dataset[varSEC] = dataset[varSEC] * 1e-6  # m-3 --> cm-3
                    dataset[varSEC].attrs['units'] = 'cm-3'
            else:
                cont = False
            i += 1

        # if dataset==None:
        #    self.dataset = dataset

        return dataset

    #######################################################
    ###  pick out location and create a sizedistribution
    #######################################################

    def pick_out_loc(self, latitude, longitude, pres=1000., dataset=xr.Dataset()):
        """
        Pick out specific location and save to file
        :param latitude:
        :param longitude:
        :param pres:
        :param dataset:
        :return:
        """
        filen = self.get_filename_location_sizedist(latitude, longitude, pres)
        if os.path.isfile(filen):
            local_dtset = xr.open_dataset(filen)
        else:
            if len(dataset) == 0:
                dataset = self.dataset
            local_dtset = dataset.sel(lat=latitude, lon=longitude, lev=pres, method='nearest')  # .load()
            make_folders(self.savepath)
            del local_dtset['time'].attrs['units']
            del local_dtset['time'].attrs['calendar']

            # local_dtset.to_netcdf(filen)

        return local_dtset

    def pick_out_loc_and_create_sizedist_day_by_day(self, latitude, longitude, pres=1000., history_field='.h2.'):
        """
        Pick out location and create sizedistribution day by day.
        :param latitude:
        :param longitude:
        :param pres:
        :param history_field:
        :return:
        """
        dt_from = datetime.strptime(self.from_time, '%Y-%m-%d')
        dt_to = datetime.strptime(self.to_time, '%Y-%m-%d')
        day = timedelta(days=1)
        first = True
        while dt_from <= dt_to:
            from_str = datetime.strftime(dt_from, '%Y-%m-%d-%H')
            to_str = datetime.strftime(dt_from + day, '%Y-%m-%d-%H')
            filen = self.savepath + '/%s_lat%.0f_lon%.0f_pres%.0f_from%s_to%s.nc' \
                    % (self.case_name, latitude, longitude, pres, from_str, from_str)
            if os.path.isfile(filen):
                loc_ds = xr.open_dataset(filen)
            else:
                dtset = self.import_data([], 'atm', history_field, remove_extra_vars=False,
                                         save_to_obj=False,
                                         from_time=datetime.strftime(dt_from, '%Y-%m-%d'),
                                         to_time=datetime.strftime(dt_from, '%Y-%m-%d'))
                lat_ind = np.argmin(abs(latitude - dtset['lat']))
                lon_ind = np.argmin(abs(longitude - dtset['lon']))
                lev_ind = np.argmin(abs(pres - dtset['lev']))
                loc_ds = dtset.isel(lat=lat_ind, lon=lon_ind, lev=lev_ind)
                # loc_ds = dtset.sel(lat=latitude,lon=longitude,lev=pres, method='nearest')
                del loc_ds['time'].attrs['units']
                del loc_ds['time'].attrs['calendar']
                practical_functions.make_folders(filen)
                make_folders(filen)
                loc_ds.to_netcdf(filen)

            if first:
                local_dataset_concat = loc_ds
                first = False
            else:
                local_dataset_concat = xr.concat([local_dataset_concat, loc_ds], dim='time')
            dt_from = dt_from + day
        self.create_sizedist_mode(local_dataset_concat)
        self.create_sizedist_sec(local_dataset_concat)
        self.local_dtset = local_dataset_concat
        return local_dataset_concat

    def get_sizedist_at_loc(self, latitude, longitude, pres=1000., history_field='.h2.', remove_extra_vars=False):
        """
        Import only if necessary,
        :param latitude:
        :param longitude:
        :param pres:
        :return:
        """
        filen = self.get_filename_location_sizedist(latitude, longitude, pres)
        if os.path.isfile(filen):
            local_dtset = xr.open_dataset(filen)
        else:
            self.import_data([],'atm', history_field, remove_extra_vars=remove_extra_vars)
            local_dtset = self.pick_out_loc_and_create_sizedist(latitude, longitude, pres = pres)
        return local_dtset

    def pick_out_loc_and_create_sizedist(self, latitude, longitude, pres=1000., avg_time=False ):
        """
        Pick out lat, lon and pressure and create sizedistribution for each.
        :param latitude:
        :param longitude:
        :param pres:
        :return:
        """
        filen = self.get_filename_location_sizedist(latitude, longitude, pres)
        if os.path.isfile(filen):
            local_dtset = xr.open_dataset(filen)
        else:
            local_dtset = self.dataset.sel(lat=latitude, lon=longitude, lev=pres, method='nearest').copy()  # .load()
            del local_dtset['time'].attrs['units']
            del local_dtset['time'].attrs['calendar']

            local_dtset = self.change_units(dataset=local_dtset)
            practical_functions.make_folders(filen)
            local_dtset.to_netcdf(filen)
        self.create_sizedist_mode(local_dtset)
        self.create_sizedist_sec(local_dtset)
        self.local_dtset = local_dtset

        return local_dtset

    def get_filename_location_sizedist(self, latitude, longitude, pres):
        filen = self.savepath + '/local_timeseries/%s_lat%.0f_lon%.0f_pres%.0f_from%s_to%s.nc' % (
            self.case_name, latitude, longitude, pres, self.from_time.replace('-', ''), self.to_time.replace('-', ''))
        return filen

    def average_and_create_sizedist(self, latitude, longitude, pres=1000., area='Global'):
        """
        Average and greate sizedist
        :param latitude:
        :param longitude:
        :param pres:
        :param area:
        :return:
        """
        filen = self.savepath + '/%s_%s_pres%.0f_from%s_to%s.nc' % (
        self.case_name, area, pres, self.from_time.replace('-', ''), self.to_time.replace('-', ''))
        if os.path.isfile(filen):
            local_dtset = xr.open_dataset(filen)
        else:

            local_dtset = self.dataset.sel(lat=latitude, lon=longitude, lev=pres, method='nearest').copy()  # .load()
            make_folders(self.savepath)
            del local_dtset['time'].attrs['units']
            del local_dtset['time'].attrs['calendar']

            make_folders(filen)
            local_dtset.to_netcdf(filen)
        self.create_sizedist_mode(local_dtset)
        self.create_sizedist_sec(local_dtset)
        self.local_dtset = local_dtset

        return local_dtset
    def avg_area_and_create_sizedist(self, area='Global', pres=1000.):
        """
        Avg area and create sizedist
        :param area:
        :param pres:
        :return:
        """
        filen_mean = self.savepath + 'avg_sizedist/vars_%s_%s_pres%.0f_from%s_to%s.nc' % (
        self.case_name, area, pres, self.from_time.replace('-', ''), self.to_time.replace('-', ''))
        filen_sizedist = self.savepath + 'avg_sizedist/sizedist_%s_%s_pres%.0f_from%s_to%s.nc' % (
        self.case_name, area, pres, self.from_time.replace('-', ''), self.to_time.replace('-', ''))
        if 'nrSOA_SEC01' in self.dataset:
            varList = list_noresm
        else:
            varList = list_noresm_nonsec
        if os.path.isfile(filen_mean):
            mean_ds = xr.open_dataset(filen_mean)
        else:
            make_folders(filen_mean)
            mean_ds = self.dataset[varList].copy()
            for var in varList:
                mean_ds[var] = self.average_timelatlon_at_lev(var, area, level_to_plot=pres)
                # print(mean_ds[var])
            mean_ds['time'] = mean_ds['time'].isel(time=0)
            del mean_ds['time']
            mean_ds = self.change_units(dataset=mean_ds)
            mean_ds.to_netcdf(filen_mean)
        if os.path.isfile(filen_sizedist):
            sizedist_ds = xr.open_dataset(filen_sizedist)
        else:
            self.create_sizedist_mode(mean_ds)
            sizedist_ds = self.create_sizedist_sec(mean_ds, one_time=True)
            if 'time' in sizedist_ds:
                del sizedist_ds['time']  # .attrs['units']
            sizedist_ds.to_netcdf(filen_sizedist)
        return mean_ds, sizedist_ds

    #######################################################
    ###  Create sizedistribution dataset available at
    ###  self.size_dtset
    #######################################################

    def create_sizedist_mode(self, local_dtset, along_dim='time'):  # include coords?
        """
        Create the sizedistribution for a sizedistribution dataset
        :param local_dtset:
        :return:
        """
        size_dtset = self.size_dtset
        logD = self.size_dtset['logD']
        for i in np.arange(len(varListNorESM['NCONC'])):
            if self.print:
                print(varListNorESM['NCONC'][i])
            # print(varListNorESM['NCONC'][i])
            varN = varListNorESM['NCONC'][i]
            varSIG = varListNorESM['SIGMA'][i]
            varNMR = varListNorESM['NMR'][i]
            NCONC = local_dtset[varN]  # [::]*10**(-6) #m-3 --> cm-3
            SIGMA = local_dtset[varSIG]  # [::]#*10**6
            NMR = local_dtset[varNMR] #*2  # [::]*2*10**9  # m --> nm + radius --> diameter
            size_dtset['dNdlogD_mode%s' % (varN[-2::])] = NCONC / (np.log(SIGMA) * np.sqrt(2 * np.pi)) * np.exp(
                -(np.log(logD) - np.log(NMR)) ** 2 / (2 * np.log(SIGMA) ** 2))
            size_dtset['dNdlogD_mode%s' % (varN[-2::])].attrs['units'] = 'cm-3'
            size_dtset['dSdlogD_mode%s' % (varN[-2::])] = 1e-9 * 4. * np.pi * logD ** 2 * (
                        NCONC / (np.log(SIGMA) * np.sqrt(2 * np.pi)) * np.exp(
                    -(np.log(logD) - np.log(NMR)) ** 2 / (2 * np.log(SIGMA) ** 2)))
            size_dtset['dSdlogD_mode%s' % (varN[-2::])].attrs['units'] = 'um2/cm3'
            size_dtset['dVdlogD_mode%s' % (varN[-2::])] = 1e-9 * 1 / 6. * np.pi * logD ** 3 * (
                        NCONC / (np.log(SIGMA) * np.sqrt(2 * np.pi)) * np.exp(
                    -(np.log(logD) - np.log(NMR)) ** 2 / (2 * np.log(SIGMA) ** 2)))
            size_dtset['dVdlogD_mode%s' % (varN[-2::])].attrs['units'] = 'um3/cm3'
            if i == 0:
                size_dtset['dNdlogD'] = 0. * size_dtset[
                    'dNdlogD_mode%s' % (varN[-2::])].copy()  # size_dtset['dNdlogD%s'%(varN[-2::])].copy()
                size_dtset['dSdlogD'] = 0. * size_dtset['dSdlogD_mode%s' % (varN[-2::])].copy()
                size_dtset['dVdlogD'] = 0. * size_dtset['dVdlogD_mode%s' % (varN[-2::])].copy()

            size_dtset['dNdlogD'] = size_dtset['dNdlogD_mode%s' % (varN[-2::])] + size_dtset['dNdlogD']
            size_dtset['dSdlogD'] = size_dtset['dSdlogD_mode%s' % (varN[-2::])] + size_dtset['dSdlogD']
            size_dtset['dVdlogD'] = size_dtset['dVdlogD_mode%s' % (varN[-2::])] + size_dtset['dVdlogD']
        self.size_dtset = size_dtset  # .persist()
        return size_dtset

    ##################################################################
    #### Create sectional dataset
    ##################################################################
    def create_sizedist_sec(self, local_dtset, one_time=False, along_dim='time'):  # include coords?
        """
        create sizedistribution with sectional.
        :param local_dtset:
        :param one_time:
        :return:
        """
        size_dtset = self.size_dtset
        logD = self.size_dtset['logD']
        SECnr = self.nr_of_bins
        binDiam_l = self.binDiameter_l
        #binDiam_h = self.binDiameter_h
        binDiam = self.binDiameter
        if not one_time:
            dNlogD_sec = np.zeros([len(local_dtset[along_dim]), logD.shape[0]])
            dSlogD_sec = np.zeros([len(local_dtset[along_dim]), logD.shape[0]])
        else:
            dNlogD_sec = np.zeros([logD.shape[0]])
            dSlogD_sec = np.zeros([logD.shape[0]])
        # dNdlogD=
        if ('nrSOA_SEC01' in local_dtset):
            for i in np.arange(SECnr):
                # print('nrSOA_SEC%02.0f'%(i+1))
                varSOA = 'nrSOA_SEC%02.0f' % (i + 1)
                varSO4 = 'nrSO4_SEC%02.0f' % (i + 1)
                athird = 1. / 3.
                SOA = local_dtset[varSOA].values  # *1e-6
                SO4 = local_dtset[varSO4].values  # *1e-6
                # dNdlogD=(SOA+SO4)/np.log(binDiam_h[i]-binDiam_l[i])#*binDiam[i]*np.log(10)
                if (i != SECnr - 1):
                    dNdlogD = (SOA + SO4)/(np.log(binDiam_l[i+1]/binDiam_l[i]))# * binDiam[i] / (binDiam_l[i + 1] - binDiam_l[
                    #dNdlogD = (SOA + SO4) * binDiam[i] / (binDiam_l[i + 1] - binDiam_l[
                    #    i])  # ((np.log(binDiam_l[i+1]/binDiam_l[i])))#"*binDiam[i]*np.log(10)
                    # dNdlogD=(SOA+SO4)*((np.log(binDiam_l[i+1]/binDiam_l[i])))#"*binDiam[i]*np.log(10)
                    dSdlogD = (SOA + SO4) * 1e-9 * 4. * np.pi * logD[i].values ** 2
                else:
                    dSdlogD = (SOA + SO4) * 1e-9 * 4. * np.pi * logD[i].values ** 2
                    dNdlogD = (SOA + SO4) /np.log(self.maxDiameter-binDiam_l[i])#* binDiam[i] / (self.maxDiameter - binDiam_l[
                    #dNdlogD = (SOA + SO4) * binDiam[i] / (self.maxDiameter - binDiam_l[
                    #    i])  # ((np.log(self.maxDiameter/binDiam_l[i])))#"*binDiam[i]*np.log(10)
                    # dNdlogD=(SOA+SO4)/((np.log(self.maxDiameter/binDiam_l[i])))#"*binDiam[i]*np.log(10)
                if (i != SECnr - 1):
                    # inds=[j for j in np.arange(len(logD)) if (logD[j]*1e-9>=binDiam_l[i] and logD[j]*1e-9<binDiam_l[i+1])]
                    inds = [j for j in np.arange(len(logD)) if (logD[j] >= binDiam_l[i] and logD[j] < binDiam_l[i + 1])]
                else:
                    # inds=[j for j in np.arange(len(logD)) if (logD[j]*1e-9>=binDiam_l[i] and logD[j]*1e-9<self.maxDiameter)]
                    inds = [j for j in np.arange(len(logD)) if (logD[j] >= binDiam_l[i] and logD[j] < self.maxDiameter)]
                if not one_time:
                    for j in np.arange(len(local_dtset[along_dim])):
                        dNlogD_sec[j, inds] += dNdlogD[j]
                        dSlogD_sec[j, inds] += dSdlogD[j]
                else:
                    dNlogD_sec[inds] += dNdlogD
                    dSlogD_sec[inds] += dSdlogD
        if one_time:
            self.size_dtset['dNdlogD_sec'] = xr.DataArray(dNlogD_sec, coords=([logD]), dims=('logD'))
            self.size_dtset['dSdlogD_sec'] = xr.DataArray(dSlogD_sec, coords=([logD]), dims=('logD'))
        else:
            self.size_dtset['dNdlogD_sec'] = xr.DataArray(dNlogD_sec, coords=([local_dtset[along_dim], logD]), dims=(along_dim, 'logD'))
            self.size_dtset['dSdlogD_sec'] = xr.DataArray(dSlogD_sec, coords=([local_dtset[along_dim], logD]), dims=(along_dim, 'logD'))
        return self.size_dtset

    #######################################################
    ###  Set parameters for sectional schemes
    #######################################################

    def set_sectional_parameters(self, nr_of_bins, minDiameter=3.0,
                                 maxDiameter=23.6):  # minDiameter=3.0e-9,maxDiameter=23.6e-9):
        """
        Set sectional parameters.
        :param nr_of_bins:
        :param minDiameter:
        :param maxDiameter:
        :return:
        """
        self.nr_of_bins = nr_of_bins
        self.minDiameter = minDiameter
        self.maxDiameter = maxDiameter
        binDiam, binDiam_l =get_bin_diams(nr_of_bins, minDiameter=minDiameter,
                                 maxDiameter=maxDiameter)  # minDiameter=3.0e-9,maxDiameter=23.6e-9):

        #d_rat = (maxDiameter / minDiameter) ** (1 / nr_of_bins)
        #binDiam = np.zeros(nr_of_bins)
        #binDiam_l = np.zeros(nr_of_bins)
        #binDiam_h = np.zeros(nr_of_bins)
        #binDiam[0] = minDiameter
        #athird = 1. / 3.
        ##binDiam_l[0] = (2 / (1 + d_rat)) * binDiam[0]
        #binDiam_l[0] = np.sqrt(1/d_rat*binDiam[0]**2)#(2 / (1 + d_rat)) * binDiam[0]
        #binDiam_h[0] = d_rat * binDiam[0] * (2 / (1 + d_rat))
        ## dNlogD_sec=np.zeros([timenr, logR.shape[0]])
        #for i in np.arange(1, nr_of_bins):
        #    binDiam[i] = binDiam[i - 1] * d_rat
        #    binDiam_l[i] = np.sqrt(binDiam[i]*binDiam[i-1])
        #    binDiam_h[i] = np.sqrt(binDiam[i]*binDiam[i]*d_rat)
        #    #binDiam_l[i] = (2 / (1 + d_rat)) * binDiam[i]
        #    #binDiam_h[i] = (2 / (1 + d_rat)) * d_rat * binDiam[i]
        self.binDiameter = binDiam
        self.binDiameter_l = binDiam_l
        #self.binDiameter_h = binDiam_h
        self.soa_sec_var_list = []
        self.so4_sec_var_list = []
        for i in np.arange(nr_of_bins):
            self.soa_sec_var_list.append('nrSOA_SEC%02.0f' % (i + 1))
            self.so4_sec_var_list.append('nrSO4_SEC%02.0f' % (i + 1))

        return

    ########################################################################################################
    ### AVERAGING:
    ########################################################################################################
    def average_timelatlon_at_lev(self, var, area, level_to_plot=1000.):
        xr_ds = self.dataset
        time = xr_ds['time'].values
        lev = xr_ds['lev'].values
        lon = xr_ds['lon'].values.copy()
        lat = xr_ds['lat'].values.copy()
        lon = xr_ds['lon'].values.copy()  # +180
        startyear = xr_ds['time.year'].min().values
        endyear = xr_ds['time.year'].max().values
        lev_ind = np.argmin(np.abs(lev - level_to_plot))
        mask, area_masked = area_mod.get_4d_area_mask(area, xr_ds, lev, var, time)

        wgts_ = xr_ds['gw'].values
        if len(wgts_.shape) == 2:
            wgts_ = wgts_[0, :]

        wgts_matrix = np.empty([len(time), len(lev), len(lat), len(lon)])
        for lo in np.arange(len(lon)):
            for le in np.arange(len(lev)):
                for ti in np.arange(len(time)):
                    wgts_matrix[ti, le, :, lo] = wgts_

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
        dummy = xr_ds[var].copy()
        dummy[var] = xr.DataArray(lat_mean, coords=coords, dims=dims, attrs=xr_ds_da.attrs, name=var)
        # print(dummy)
        mean = dummy[var].mean(dim=('time', 'lon'), skipna=True, keep_attrs=True).isel(lev=lev_ind)
        del dummy
        mean.attrs['startyear'] = startyear
        mean.attrs['endyear'] = endyear
        return mean
    """
    if 'lat_wg' in dtset:
        wgts_ = dtset['lat_wg']/dtset['lat_wg'].sum() # Get latitude weights
    else:
        wgts_ = xr.DataArray(analysis_tools.area_pkg_sara.get_wghts(dtset['lat'].values), #define xarray with weights and dimension lat
                             dims={'lat':dtset['lat']})
        wgts_= wgts_/wgts_.sum()
    dummy, wg_matrix = xr.broadcast(dtset[var],wgts_) # broadcast one 1D weights to all dimensions in dtset[var]
    lev= dtset['lev'].values; time = dtset['time']
    mask, area_masked = get_4d_area_mask_xa(area, dtset, lev, var, time) # get 3D mask for area

    da = dtset[var] # pick out DataArray from DataSet
    da_masked = da.where(np.logical_not(mask)) # Mask values not in accordance to mask (sets to nans)
    wgts_masked = wg_matrix.where(np.logical_not(mask)) # Set weights to nan where not right area
    da_lat = (da_masked*wgts_masked/wgts_masked.sum('lat')).sum('lat') # mean over latitude dimension
    da_mean = da_lat.mean(dim=list((set(da_lat.dims)-set({'lev'})))) # unweighted mean over rest of dimensions.
    print(list((set(da_lat.dims)-set({'lev'})))) #
    da_mean.attrs=dtset[var].attrs #
    dtset[var]=da_mean
    dtset[var].attrs = da_mean.attrs
    print(dtset.attrs)
    save_profile_data(dtset, var, model, area, save_path=save_path)
    """
    def plot_time_distribution(plot_type='number'):

        return
        #######################################################

    ###  IMPORT FILES
    ### imports files and picks out time slice
    #######################################################
    def import_local_dataset(self, var_names, comp, history_field, latitude, longitude, pres=1000.):
        filen = self.get_filename_location_sizedist(latitude, longitude, pres)

        if os.path.isfile(filen):
            local_dataset = xr.open_dataset(filen)
        else:

            path_mod = self.data_path + self.case_name + '/' + comp + '/hist/'
            filelist_d = [f for f in listdir(path_mod) if ((history_field in f) and f[0] != '.')]
            filelist_time = [
                f[(len(self.case_name) + len(self.model_name) + 2):(len(self.case_name) + len(self.model_name) + 12)]
                for f in filelist_d]
            # print(filelist_time) #0001-01-05
            filelist_date = [datetime.strptime(f, '%Y-%m-%d') for f in filelist_time]
            # print(filelist_date)
            from_dt = datetime.strptime(self.from_time, '%Y-%m-%d')
            to_dt = datetime.strptime(self.to_time, '%Y-%m-%d')
            tf = np.array([to_dt >= filelist_date[i] >= from_dt for i in np.arange(len(filelist_d))])
            import_list = np.array(filelist_d)[tf]
            pathfile_list = [path_mod + imp for imp in import_list]
            first = True
            for file in pathfile_list:

                print('loading file: ' + file)
                dtset_tmp = self.import_filelist([file], var_names)
                local_dtset_tmp = self.pick_out_loc(latitude, longitude, pres=1000., dataset=dtset_tmp)
                if first:
                    local_dataset = local_dtset_tmp.load()
                    first = False
                else:
                    local_dataset = xr.concat([local_dataset, local_dtset_tmp.load()], dim='time')
                # local_dataset=local_dataset.load()

            local_dataset = local_dataset.sortby('time')
            make_folders(filen)
            local_dataset.to_netcdf(filen)
        # self.local_dataset=local_dataset
        self.local_dtset = local_dataset

        return local_dataset


class NorESM_SizedistDataset_spec_lat_lon_output(NorESM_SizedistDataset):

    def import_data(self, var_names, comp, history_field, remove_extra_vars=True,
                    locations=['LON_116e_LAT_40n', 'LON_24e_LAT_62n', 'LON_63w_LAT_3s'],
                    concatinated_file =False, years=[2008,2008]):
        super().import_data(var_names, comp, history_field, remove_extra_vars=remove_extra_vars, concatinated_file=concatinated_file, years=years)
        dataset_orig = self.dataset
        locations=[var[8:] for var in dataset_orig.data_vars if var[0:7]=='NCONC01']
        print(locations)

        new_dtset = xr.Dataset(
            coords={'LOCATIONS': locations, 'time': dataset_orig['time'], 'ilev': dataset_orig['ilev'],
                    'lev': dataset_orig['lev']})
        variables_in_dtset = list(self.dataset.data_vars)
        locations = new_dtset['LOCATIONS'].values
        vars_already_reg = []
        for var in variables_in_dtset:
            loc_in_var = False
            for loc in locations:
                if loc in var:
                    loc_in_var = True
                    loc_name = loc
                    break
            if loc_in_var:
                # print(var)
                shape = dataset_orig[var].shape
                # print(shape)
                if len(shape) == 4:
                    coords = {'time': dataset_orig['time'], 'lev': dataset_orig['lev'], 'LOCATIONS': locations}
                    dims = ('time', 'lev', 'LOCATIONS')
                    shape = [shape[0], shape[1], len(locations)]
                else:
                    coords = {'time': dataset_orig['time'], 'LOCATIONS': locations}
                    dims = ('time', 'LOCATIONS')
                    shape = [shape[0], len(locations)]
                var_alone = var[:-(len(loc_name) + 1)]
                if var_alone not in vars_already_reg:
                    vars_already_reg.append(var_alone)
                    new_dtset[var_alone] = xr.DataArray(np.zeros(shape), coords=coords, dims=dims)
                    new_dtset[var_alone].attrs = dataset_orig[var].attrs
                    i = 0
                    for loc in locations:
                        if len(shape) == 3:
                            new_dtset[var_alone][:, :, i] = dataset_orig[var_alone + '_' + loc][:, :, 0, 0]
                        else:
                            new_dtset[var_alone][:, i] = dataset_orig[var_alone + '_' + loc][:, 0, 0]
                        i += 1
            else:
                new_dtset[var] = dataset_orig[var]
        self.dataset = new_dtset
        self.locations = locations
        return
    def get_sizedist_at_loc(self, location, pres=1000., history_field='.h2.', remove_extra_vars=False,
                            concatinated_file=False, years=[2008,2008], avg_time=False):
        """
        Import only if necessary,
        :param latitude:
        :param longitude:
        :param pres:
        :return:
        """
        filen = self.get_filename_location_sizedist(location, pres, avg_over_time=avg_time)
        make_folders(filen)
        if os.path.isfile(filen):
            local_dtset = xr.open_dataset(filen)
            if 'time' in local_dtset:
                self.time = local_dtset['time']
            self.location = location
            logD = np.logspace(0, 4, 50)  # , name='logD'
            along_dim ='time'
            if avg_time: along_dim='lev'
            self.size_dtset = xr.Dataset(coords={along_dim: local_dtset[along_dim], 'logD': logD})  # , dims=['time', 'logD'])
            size_dtset =  super().create_sizedist_mode(local_dtset)
            size_dtset = super().create_sizedist_sec(local_dtset, along_dim=along_dim)
        else:
            self.import_data([],'atm', history_field, remove_extra_vars=remove_extra_vars, locations=[location],
                             concatinated_file=concatinated_file, years=years)
            local_dtset, size_dtset = self.pick_out_loc_and_create_sizedist(location, pres = pres, avg_time=avg_time)
        return local_dtset, size_dtset

    def pick_out_loc_and_create_sizedist(self, location, pres=1000., avg_time=False):
        if self.print:
            print(location)
        self.location = location
        filen = self.get_filename_location_sizedist(location, pres, avg_over_time = avg_time)
        make_folders(filen)
        if os.path.isfile(filen):
            local_dtset = xr.open_dataset(filen)
        else:
            ii = 0
            for loc in self.dataset.coords['LOCATIONS']:
                if loc == location: ind_loc = ii
                ii += 1
            if avg_time:
                local_dtset = self.dataset.sel(method='nearest').isel(LOCATIONS=ind_loc).copy()
                local_dtset = local_dtset.mean('time', keep_attrs=True)
            else:
                local_dtset = self.dataset.sel(lev=pres, method='nearest').isel(LOCATIONS=ind_loc).copy()
                # print('************************************************************************')
                if 'units' in local_dtset['time'].attrs:
                    del local_dtset['time'].attrs['units']
                if 'calendar' in local_dtset['time'].attrs:
                    del local_dtset['time'].attrs['calendar']
            make_folders(self.savepath)

            # print(local_dtset['time'])
            self.change_units(dataset=local_dtset)
            # print(local_dtset)
            local_dtset.to_netcdf(filen)
        along_dim = 'time'
        if avg_time: along_dim='lev'
        size_dtset = super().create_sizedist_mode(local_dtset, along_dim=along_dim)
        size_dtset = super().create_sizedist_sec(local_dtset, along_dim=along_dim)
        self.local_dtset = local_dtset

        return local_dtset, size_dtset

    def get_filename_location_sizedist(self, location, pres, avg_over_time=False):
        if avg_over_time:
            st = 'avg_time'
        else: st=''
        filen = self.savepath + '/%s_loc_%s_pres%.0f_from%s_to%s_%s.nc' % (
            self.case_name, location, pres, self.from_time.replace('-', ''), self.to_time.replace('-', ''), st)
        return filen


############################
## Make folder from path
############################
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


def extract_path_from_filepath(file_path):
    """
    ex: 'folder/to/file.txt' returns 'folder/to/'
    :param file_path:
    :return:
    """

    st_ind = file_path.rfind('/')
    foldern = file_path[0:st_ind] + '/'
    return foldern

def plot_2_cases_size_time_diff_xr(dummy1, dummy2, label1, label2, latitude_coord,
                                longitude_coord, pressure_coord, vmin=1e1, vmax=1e4, only_mode01=False,
                                nodiff=False, from_time='', to_time=''):
    MEDIUM_SIZE = 18
    if len(from_time) > 0 and len(to_time) > 0:
        dummy1.size_dtset = dummy1.size_dtset.sel(time=slice(from_time, to_time))
        dummy2.size_dtset = dummy2.size_dtset.sel(time=slice(from_time, to_time))
    else:
        from_time = dummy1.from_time
        to_time = dummy1.to_time
    if only_mode01:
        dNdlogD = 'dNdlogD_mode01'
    else:
        dNdlogD = 'dNdlogD'
    # for dummy in [dummy1, dummy2]:
    # ntime=num2date(dummy.size_dtset['time'].values,
    #               units='days since 2008-01-01')#dummy.size_dtset['time'].attrs['units'])#,
    # calendar='gregorian')
    # calendar=dummy.size_dtset['time'].attrs['calendar'])
    # dummy['time']=ntime
    if nodiff:
        nr_subp = 2
        fig, axs = plt.subplots(2, figsize=[15, 6], sharex=True)
    else:
        nr_subp = 3
        fig, axs = plt.subplots(3, figsize=[15, 8], sharex=True)
    cba_kwargs = {'label': r'dN/dlogD [#/cm$^3$]', 'aspect': 8}
    (dummy1.size_dtset['dNdlogD_sec'] + dummy1.size_dtset[dNdlogD]). \
        plot(x='time',
             y='logD', yscale='log',
             norm=colors.LogNorm(vmin=vmin, vmax=vmax),
             ylim=[3, 1e3], ax=axs[0],
             cbar_kwargs=cba_kwargs)
    if isinstance(dummy1, NorESM_SizedistDataset_spec_lat_lon_output):
        axs[0].set_title(label1 + ', loc=' + dummy1.location)
    else:
        axs[0].set_title(label1)

    (dummy2.size_dtset['dNdlogD_sec'] + dummy2.size_dtset[dNdlogD]). \
        plot(x='time', y='logD', yscale='log',
             norm=colors.LogNorm(vmin=vmin, vmax=vmax),
             ylim=[3, 1e3], ax=axs[1],
             cbar_kwargs=cba_kwargs)
    if isinstance(dummy2, NorESM_SizedistDataset_spec_lat_lon_output):
        axs[1].set_title(label2 + ', loc=' + dummy1.location)
    else:
        axs[1].set_title(label2)
    # axs[1].set_title(label2 + ', loc='+ dummy2.location)
    if not nodiff:
        (-(dummy1.size_dtset['dNdlogD_sec'] + dummy1.size_dtset[dNdlogD]) + (
                    dummy2.size_dtset['dNdlogD_sec'] + dummy2.size_dtset[dNdlogD])). \
            plot(x='time', y='logD', yscale='log', ax=axs[2],
                 norm=colors.SymLogNorm(linthresh=10, linscale=1, vmin=-vmax, vmax=vmax),
                 cmap='RdBu_r', ylim=[3, 1e3],
                 cbar_kwargs=cba_kwargs)
        axs[2].set_title(label2 + ' - ' + label1)  # + ', loc='+ dummy1.location)
    for i in np.arange(nr_subp):
        axs[i].set_ylabel('Diameter [nm]')
        axs[i].get_xaxis().get_label().set_visible(False)  # .get_xlabel().set_visible(False)
        # axs[i].get_xlabel().set_visible(False)

    plotpath = 'plots/time_sizedist/'
    plotpath = plotpath + '/lat%.1f_lon%.1f_pres%.1f/' % (latitude_coord, longitude_coord, pressure_coord)
    plotpath = plotpath + dummy1.case_name + '_' + dummy2.case_name
    plotpath = plotpath + 'time_%s-%s' % (from_time, to_time)
    if nodiff:
        plotpath = plotpath + 'nodiff'
    plotpath = plotpath + '.png'
    practical_functions.make_folders(plotpath)
    print(plotpath)
    plt.tight_layout()
    plt.savefig(plotpath, dpi=300)
    plt.show()

def plot_2_cases_size_time_diff_givefig(dummy1, dummy2, label1, label2, latitude_coord,
                                longitude_coord, pressure_coord,axs, vmin=1e1, vmax=1e4, only_mode01=False,
                                nodiff=False, from_time='', to_time=''):
    MEDIUM_SIZE = 18
    if len(from_time) > 0 and len(to_time) > 0:
        dummy1.size_dtset = dummy1.size_dtset.sel(time=slice(from_time, to_time))
        dummy2.size_dtset = dummy2.size_dtset.sel(time=slice(from_time, to_time))
    else:
        from_time = dummy1.from_time
        to_time = dummy1.to_time
    if only_mode01:
        dNdlogD = 'dNdlogD_mode01'
    else:
        dNdlogD = 'dNdlogD'
    # for dummy in [dummy1, dummy2]:
    # ntime=num2date(dummy.size_dtset['time'].values,
    #               units='days since 2008-01-01')#dummy.size_dtset['time'].attrs['units'])#,
    # calendar='gregorian')
    # calendar=dummy.size_dtset['time'].attrs['calendar'])
    # dummy['time']=ntime
    if nodiff:
        nr_subp = 2
    #    fig, axs = plt.subplots(2, figsize=[15, 6], sharex=True)
    else:
        nr_subp = 3
    #    fig, axs = plt.subplots(3, figsize=[15, 8], sharex=True)
    cba_kwargs = {'label': r'dN/dlogD [#/cm$^3$]', 'aspect': 8}
    (dummy1.size_dtset['dNdlogD_sec'] + dummy1.size_dtset[dNdlogD]). \
        plot(x='time',
             y='logD', yscale='log',
             norm=colors.LogNorm(vmin=vmin, vmax=vmax),
             ylim=[3, 1e3], ax=axs[0],
             cbar_kwargs=cba_kwargs)
    if isinstance(dummy1, NorESM_SizedistDataset_spec_lat_lon_output):
        axs[0].set_title(label1 + ', loc=' + dummy1.location)
    else:
        axs[0].set_title(label1)

    (dummy2.size_dtset['dNdlogD_sec'] + dummy2.size_dtset[dNdlogD]). \
        plot(x='time', y='logD', yscale='log',
             norm=colors.LogNorm(vmin=vmin, vmax=vmax),
             ylim=[3, 1e3], ax=axs[1],
             cbar_kwargs=cba_kwargs)
    if isinstance(dummy2, NorESM_SizedistDataset_spec_lat_lon_output):
        axs[1].set_title(label2 + ', loc=' + dummy1.location)
    else:
        axs[1].set_title(label2)
    # axs[1].set_title(label2 + ', loc='+ dummy2.location)
    if not nodiff:
        (-(dummy1.size_dtset['dNdlogD_sec'] + dummy1.size_dtset[dNdlogD]) + (
                    dummy2.size_dtset['dNdlogD_sec'] + dummy2.size_dtset[dNdlogD])). \
            plot(x='time', y='logD', yscale='log', ax=axs[2],
                 norm=colors.SymLogNorm(linthresh=10, linscale=1, vmin=-vmax, vmax=vmax),
                 cmap='RdBu_r', ylim=[3, 1e3],
                 cbar_kwargs=cba_kwargs)
        axs[2].set_title(label2 + ' - ' + label1)  # + ', loc='+ dummy1.location)
    for i in np.arange(nr_subp):
        axs[i].set_ylabel('Diameter [nm]')
        axs[i].get_xaxis().get_label().set_visible(False)  # .get_xlabel().set_visible(False)
        # axs[i].get_xlabel().set_visible(False)

    #plotpath = 'plots/time_sizedist/'
    #plotpath = plotpath + '/lat%.1f_lon%.1f_pres%.1f/' % (latitude_coord, longitude_coord, pressure_coord)
    #plotpath = plotpath + dummy1.case_name + '_' + dummy2.case_name
    #plotpath = plotpath + 'time_%s-%s' % (from_time, to_time)
    #if nodiff:
    #    plotpath = plotpath + 'nodiff'
    #plotpath = plotpath + '.png'
    #practical_functions.make_folders(plotpath)
    #print(plotpath)
    plt.tight_layout()
    #plt.savefig(plotpath, dpi=300)
    #plt.show()

def plot_2_cases_size_time_diff(dummy1, dummy2, label1, label2, latitude_coord,
                                longitude_coord, pressure_coord, vmin=1e1, vmax=1e4, only_mode01=False,
                                nodiff=False, from_time='', to_time=''):
    MEDIUM_SIZE = 18
    if len(from_time) > 0 and len(to_time) > 0:
        dummy1.size_dtset = dummy1.size_dtset.sel(time=slice(from_time, to_time))
        dummy2.size_dtset = dummy2.size_dtset.sel(time=slice(from_time, to_time))
    else:
        from_time = dummy1.from_time
        to_time = dummy1.to_time
    if only_mode01:
        dNdlogD = 'dNdlogD_mode01'
    else:
        dNdlogD = 'dNdlogD'
    # for dummy in [dummy1, dummy2]:
    # ntime=num2date(dummy.size_dtset['time'].values,
    #               units='days since 2008-01-01')#dummy.size_dtset['time'].attrs['units'])#,
    # calendar='gregorian')
    # calendar=dummy.size_dtset['time'].attrs['calendar'])
    # dummy['time']=ntime
    if nodiff:
        nr_subp = 2
        fig, axs = plt.subplots(2, figsize=[15, 6], sharex=True)
    else:
        nr_subp = 3
        fig, axs = plt.subplots(3, figsize=[15, 8], sharex=True)
    cba_kwargs = {'label': r'dN/dlogD [#/cm$^3$]', 'aspect': 8}
    (dummy1.size_dtset['dNdlogD_sec'] + dummy1.size_dtset[dNdlogD]). \
        plot(x='time',
             y='logD', yscale='log',
             norm=colors.LogNorm(vmin=vmin, vmax=vmax),
             ylim=[3, 1e3], ax=axs[0],
             cbar_kwargs=cba_kwargs)
    if isinstance(dummy1, NorESM_SizedistDataset_spec_lat_lon_output):
        axs[0].set_title(label1 + ', loc=' + dummy1.location)
    else:
        axs[0].set_title(label1)

    (dummy2.size_dtset['dNdlogD_sec'] + dummy2.size_dtset[dNdlogD]). \
        plot(x='time', y='logD', yscale='log',
             norm=colors.LogNorm(vmin=vmin, vmax=vmax),
             ylim=[3, 1e3], ax=axs[1],
             cbar_kwargs=cba_kwargs)
    if isinstance(dummy2, NorESM_SizedistDataset_spec_lat_lon_output):
        axs[1].set_title(label2 + ', loc=' + dummy1.location)
    else:
        axs[1].set_title(label2)
    # axs[1].set_title(label2 + ', loc='+ dummy2.location)
    if not nodiff:
        (-(dummy1.size_dtset['dNdlogD_sec'] + dummy1.size_dtset[dNdlogD]) + (
                    dummy2.size_dtset['dNdlogD_sec'] + dummy2.size_dtset[dNdlogD])). \
            plot(x='time', y='logD', yscale='log', ax=axs[2],
                 norm=colors.SymLogNorm(linthresh=10, linscale=1, vmin=-vmax, vmax=vmax),
                 cmap='RdBu_r', ylim=[3, 1e3],
                 cbar_kwargs=cba_kwargs)
        axs[2].set_title(label2 + ' - ' + label1)  # + ', loc='+ dummy1.location)
    for i in np.arange(nr_subp):
        axs[i].set_ylabel('Diameter [nm]')
        axs[i].get_xaxis().get_label().set_visible(False)  # .get_xlabel().set_visible(False)
        # axs[i].get_xlabel().set_visible(False)

    plotpath = 'plots/time_sizedist/'
    plotpath = plotpath + '/lat%.1f_lon%.1f_pres%.1f/' % (latitude_coord, longitude_coord, pressure_coord)
    plotpath = plotpath + dummy1.case_name + '_' + dummy2.case_name
    plotpath = plotpath + 'time_%s-%s' % (from_time, to_time)
    if nodiff:
        plotpath = plotpath + 'nodiff'
    plotpath = plotpath + '.png'
    practical_functions.make_folders(plotpath)
    print(plotpath)
    plt.tight_layout()
    plt.savefig(plotpath, dpi=300)
    plt.show()

def plot_2_cases_dN_dlogD_locs2(dummy1, dummy2, label1, label2, locs, yscale='log', ylims=[10, 4e4], pres=1000.,
                                only_mode1=False):
    #print(locs)
    fig, axs = plt.subplots(1, len(locs), figsize=[20, 5])
    i = 0
    for location in locs:
        dummy1.pick_out_loc_and_create_sizedist(location=location, pres=pres)
        dummy2.pick_out_loc_and_create_sizedist(location=location, pres=pres)
        if only_mode1:
            (dummy1.size_dtset['dNdlogD_mode01'] + dummy1.size_dtset['dNdlogD_sec']).mean(dim='time').plot(
                yscale=yscale, xscale='log', ylim=ylims, xlim=[3, 1e3], label=label1, ax=axs[i])
            (dummy2.size_dtset['dNdlogD_mode01'] + dummy2.size_dtset['dNdlogD_sec']).mean(dim='time').plot(
                yscale=yscale, xscale='log', ylim=ylims, xlim=[3, 1e3], linestyle='--', label=label2, ax=axs[i])
        else:
            (dummy1.size_dtset['dNdlogD'] + dummy1.size_dtset['dNdlogD_sec']).mean(dim='time').plot(yscale=yscale,
                                                                                                    xscale='log',
                                                                                                    ylim=ylims,
                                                                                                    xlim=[3, 1e3],
                                                                                                    label=label1,
                                                                                                    ax=axs[i])
            (dummy2.size_dtset['dNdlogD'] + dummy2.size_dtset['dNdlogD_sec']).mean(dim='time').plot(yscale=yscale,
                                                                                                    xscale='log',
                                                                                                    ylim=ylims,
                                                                                                    xlim=[3, 1e3],
                                                                                                    linestyle='--',
                                                                                                    label=label2,
                                                                                                    ax=axs[i])
        axs[i].set_title(dummy1.location)
        plt.ylabel('dN/dlogD')
        plt.xlabel('diameter [nm]')
        plt.legend()
        i += 1


def plot_sizedist(cases, nested_dt, pres, from_time, to_time, xlim=[4, 1e3], ylim=[10, 1e4], scale='log',
                  area='Global'):
    fig, ax = plt.subplots(1, figsize=[5, 4])
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors_dic = {'Original': colors[0], 'Sect ac': colors[1], 'Sectional': colors[2]}
    for case_name in cases:
        if 'Original' in case_name:
            col = colors_dic['Original']
        elif 'Sect ac' in case_name:
            col = colors_dic['Sect ac']
        elif 'Sectional' in case_name:
            col = colors_dic['Sectional']
        else:
            col = colors[0]
        (nested_dt[case_name]['dNdlogD'] + nested_dt[case_name]['dNdlogD_sec']).plot(
            xscale='log', yscale=scale, ax=ax, label=case_name, alpha=0.8, xlim=xlim, ylim=ylim,
            c=col, linewidth=2)

    # for case_name in cases:
    #    (nested_dt[case_name]['dNdlogD']+nested_dt[case_name]['dNdlogD_sec']).plot(
    #        xscale='log', yscale=scale, ax=ax,  label=case_name, alpha=0.8, xlim=xlim, ylim=ylim)
    plt.legend()
    plt.xlabel('Diameter [nm]', fontsize=12)
    plt.ylabel(r'dN/dlogD [#/cm$^3$]')
    ax.grid(b=True, which="both", color='k', alpha=0.1, ls="-", axis='both')

    plotpath = 'plots/sizedist/' + area
    plotpath = plotpath + '/'
    for case in cases:
        plotpath = plotpath + case.replace(' ', '-')
    plotpath = plotpath + '%.0f' % pres
    plotpath = plotpath + 'time_%s-%s' % (from_time, to_time)
    plotpath = plotpath + scale
    plotpath = plotpath + '.png'
    practical_functions.make_folders(plotpath)
    print(plotpath)
    plt.savefig(plotpath, dpi=300)
    plt.show()


def plot_sizedist_pairs(cases, nested_dt, pres, from_time, to_time, xlim=[4, 1e3], ylim=[10, 1e4], scale='log',
                        area='Global'):
    fig, ax = plt.subplots(1, figsize=[5, 4])
    i = 0
    first = True
    linestyle = '-'
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors_dic = {'Original': colors[0], 'Sect ac': colors[1], 'Sectional': colors[2]}
    for case_name in cases:
        if 'Original' in case_name:
            col = colors_dic['Original']
        elif 'Sect ac' in case_name:
            col = colors_dic['Sect ac']
        elif 'Sectional' in case_name:
            col = colors_dic['Sectional']
        else:
            col = colors[i]
        (nested_dt[case_name]['dNdlogD'] + nested_dt[case_name]['dNdlogD_sec']).plot(
            xscale='log', yscale=scale, ax=ax, label=case_name, alpha=0.8, xlim=xlim, ylim=ylim,
            c=col, linestyle=linestyle, linewidth=2)
        if first:
            linestyle = '--'
            first = False
        else:
            i += 1
            linestyle = '-'
            first = True
    plt.legend()
    plt.xlabel('Diameter [nm]', fontsize=12)
    plt.ylabel(r'dN/dlogD [#/cm$^3$]')
    ax.grid(b=True, which="both", color='k', alpha=0.1, ls="-", axis='both')

    plotpath = 'plots/sizedist/' + area
    plotpath = plotpath + '/'
    for case in cases:
        plotpath = plotpath + case.replace(' ', '-')
    plotpath = plotpath + '%.0f' % pres
    plotpath = plotpath + 'time_%s-%s' % (from_time, to_time)
    plotpath = plotpath + scale
    plotpath = plotpath + '.png'
    practical_functions.make_folders(plotpath)
    print(plotpath)
    plt.savefig(plotpath, dpi=300)
    plt.show()


def plot_sizedist_diff(cases, case_ctr, nested_dt, pres, from_time, to_time,
                       xlim=[4, 1e3], ylim=[-1e4, 1e4], scale='symlog',
                       area='Global'):
    fig, ax = plt.subplots(1, figsize=[5, 4])
    for case_name in cases:
        plt_ds = nested_dt[case_name]['dNdlogD'] + nested_dt[case_name]['dNdlogD_sec'] - \
                 (nested_dt[case_ctr]['dNdlogD'] + nested_dt[case_ctr]['dNdlogD_sec'])
        plt_ds.plot(xscale='log', yscale=scale,
                    ax=ax, label=(case_name + ' - ' + case_ctr),
                    alpha=0.8, xlim=xlim, ylim=ylim)
    plt.yscale(scale, linthresh=100)
    plt.legend()
    plt.xlabel('Diameter [nm]', fontsize=12)
    plt.ylabel(r'dN/dlogD [#/cm$^3$]')
    ax.grid(b=True, which="both", color='k', alpha=0.1, ls="-", axis='both')

    plotpath = 'plots/sizedist/' + area
    plotpath = plotpath + '/diff'
    for case in cases:
        plotpath = plotpath + case.replace(' ', '-')
    plotpath = plotpath + '%.0f' % pres
    plotpath = plotpath + 'time_%s-%s' % (from_time, to_time)
    plotpath = plotpath + scale
    plotpath = plotpath + '.png'
    practical_functions.make_folders(plotpath)
    print(plotpath)
    plt.savefig(plotpath, dpi=300)
    plt.show()


def plot_sizedist_pairs_diff(cases, nested_dt, pres, from_time, to_time, xlim=[4, 1e3], ylim=[10, 1e4], scale='log',
                             area='Global', relative=False):
    fig, ax = plt.subplots(1, figsize=[5, 4])
    i = 0
    first = True
    linestyle = '-'
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors_dic = {'Original': colors[0], 'Sect ac': colors[1], 'Sectional': colors[2]}
    for case_name in cases:
        if first:
            ctrl_ds = nested_dt[case_name]['dNdlogD'] + nested_dt[case_name]['dNdlogD_sec']
            ctrl_case_name = case_name
            first = False
        else:
            if 'Original' in case_name:
                col = colors_dic['Original']
            elif 'Sect ac' in case_name:
                col = colors_dic['Sect ac']
            elif 'Sectional' in case_name:
                col = colors_dic['Sectional']
            else:
                col = colors[i]
            if relative:
                plt_this = 100. * (
                            nested_dt[case_name]['dNdlogD'] + nested_dt[case_name]['dNdlogD_sec'] - ctrl_ds) / ctrl_ds
                label = '%% %s -- %s' % (case_name, ctrl_case_name)
            else:
                plt_this = (nested_dt[case_name]['dNdlogD'] + nested_dt[case_name]['dNdlogD_sec'] - ctrl_ds)  # /ctrl_ds
                label = '%s - %s' % (case_name, ctrl_case_name)

            plt_this.plot(
                xscale='log', yscale=scale, ax=ax, label=label, alpha=0.8, xlim=xlim, ylim=ylim,
                c=col, linestyle=linestyle, linewidth=2)
            if scale == 'symlog':
                plt.yscale(scale, linthreshy=2)
            first = True
    plt.legend()
    plt.xlabel('Diameter [nm]', fontsize=12)
    if relative:
        plt.ylabel(r'%% change in dN/dlogD [#/cm$^3$]')
    else:
        plt.ylabel(r'dN/dlogD [#/cm$^3$]')
    ax.grid(b=True, which="both", color='k', alpha=0.1, ls="-", axis='both')

    plotpath = 'plots/sizedist/' + area + '_paired_diff'
    plotpath = plotpath + '/'
    for case in cases:
        plotpath = plotpath + case.replace(' ', '-')
    plotpath = plotpath + '%.0f' % pres
    plotpath = plotpath + 'time_%s-%s' % (from_time, to_time)
    plotpath = plotpath + scale
    plotpath = plotpath + '.png'
    practical_functions.make_folders(plotpath)
    print(plotpath)
    plt.savefig(plotpath, dpi=300)
    plt.show()



def get_bin_diams(nr_of_bins, minDiameter=5.0,
                                 maxDiameter=23.6):  # minDiameter=3.0e-9,maxDiameter=23.6e-9):
    """
    Set sectional parameters.
    :param nr_of_bins:
    :param minDiameter:
    :param maxDiameter:
    :return:
    """
    d_rat = (maxDiameter / minDiameter) ** (1 / nr_of_bins)
    binDiam = np.zeros(nr_of_bins)
    binDiam_l = np.zeros(nr_of_bins)
    binDiam_h = np.zeros(nr_of_bins)
    binDiam[0] = minDiameter
    athird = 1. / 3.
    binDiam_l[0] = (2 / (1 + d_rat)) * binDiam[0]
    binDiam_h[0] = d_rat * binDiam[0] * (2 / (1 + d_rat))
    # dNlogD_sec=np.zeros([timenr, logR.shape[0]])
    for i in np.arange(1, nr_of_bins):
        binDiam[i] = binDiam[i - 1] * d_rat
        binDiam_l[i] = (2 / (1 + d_rat)) * binDiam[i]
        binDiam_h[i] = (2 / (1 + d_rat)) * d_rat * binDiam[i]

    return binDiam, binDiam_l
