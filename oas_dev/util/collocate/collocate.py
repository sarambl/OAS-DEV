import os

from oas_dev.util.imports.get_pressure_coord_fields import get_pressure_coord_fields

from oas_dev.util.imports.fix_xa_dataset_v2 import xr_fix

from oas_dev.util.imports.import_fields_xr_v2 import xr_import_NorESM
from useful_scit.util import log

from oas_dev import constants
import numpy as np
import xarray as xr
#from oas_dev.util.Nd.sizedist_class_v2.SizedistributionSurface import SizedistributionSurface
from oas_dev.util.naming_conventions import find_model_case_name
import xarray as xr
from oas_dev.util import naming_conventions
from oas_dev.util.practical_functions import make_folders


class Collocate:
    pass


class CollocateModel(Collocate):
    """
    collocate a model to a list of locations
    """
    col_dataset = None
    input_dataset = None

    def __init__(self, case_name, from_time, to_time,
                 isSectional,
                 time_res,
                 space_res='full',
                 model_name='NorESM',
                 history_field='.h0.',
                 raw_data_path=constants.get_input_datapath(),
                 locations=constants.collocate_locations,
                 read_from_file=True,
                 chunks=None,
                 use_pressure_coords=False,
                 dataset=None,
                 savepath_root = constants.get_outdata_path('collocated')
                 ):
        """
        :param case_name:
        :param from_time:
        :param to_time:
        :param raw_data_path:
        :param isSectional:
        :param time_res: 'month', 'year', 'hour'
        :param space_res: 'full', 'locations'
        :param model_name:
        """
        self.chunks = chunks
        self.read_from_file = read_from_file
        self.model_name = model_name
        # self.case_plotting_name = model_name
        self.dataset = None
        self.use_pressure_coords = use_pressure_coords
        self.case_name_nice = find_model_case_name.find_name(model_name, case_name)
        self.case_name = case_name
        self.raw_data_path = raw_data_path
        self.from_time = from_time
        self.to_time = to_time
        self.time_resolution = time_res
        self.space_resolution = space_res
        self.history_field = history_field
        self.locations = locations
        self.isSectional = isSectional
        self.locations = constants.collocate_locations
        self.dataset = dataset
        self.savepath_root = savepath_root

        self.attrs_ds = dict(raw_data_path=self.raw_data_path,
                             model=self.model_name, model_name=self.model_name,
                             case_name=self.case_name, case=self.case_name,
                             case_name_nice=self.case_name_nice,
                             isSectional=str(self.isSectional),
                             from_time=self.from_time,
                             to_time=self.to_time
                             )
    """
    def load_sizedist_dataset(self, dlim_sec, nr_bins=5):

        :param dlim_sec:
        :param nr_bins:
        :return:
        s = SizedistributionSurface(self.case_name, self.from_time, self.to_time,
                                    dlim_sec, self.isSectional, self.time_resolution,
                                    space_res=self.space_resolution, nr_bins=nr_bins,
                                    model_name=self.model_name, history_field=self.history_field,
                                    locations=self.locations,
                                    chunks=self.chunks, use_pressure_coords=False)
        ds = s.get_sizedist_var()
        self.input_dataset = ds
        return ds
    """
    def load_raw_ds(self, varList, chunks={}):
        if self.use_pressure_coords:
            ds = get_pressure_coord_fields(self.case_name, varList, self.from_time, self.to_time, self.history_field, model=self.model_name)
            self.input_dataset = ds
            return ds

        ds = xr_import_NorESM(self.case_name, varList, self.from_time, self.to_time,
                     model=self.model_name, history_fld=self.history_field, comp='atm', chunks=chunks)
        #ds = xr_fix(ds)

        self.input_dataset = ds
        return ds

    def set_input_datset(self, ds):
        self.input_dataset = ds

    def get_collocated_dataset(self, var_names, CHUNKS=None, parallel=True):
        """

        :param var_names:
        :param CHUNKS:
        :return:
        """
        #if not self.isSectional:
        #    var_names = [D_NDLOG_D_MOD]
        #else:
        #    var_names = [D_ND LOG_D_MOD, D_NDLOG_D_SEC]
        if CHUNKS is None:
            CHUNKS = self.chunks
        fn_list = []
        print(var_names)
        if type(var_names) is not list:
            var_names = [var_names]
        for var_name in var_names:

            fn = self.savepath_coll_ds(var_name)
            print(fn)
            fn_list.append(fn)
        log.ger.info('Opening: ['+','.join(fn_list)+']')
        print(fn_list)
        if len(fn_list)>1:
            ds = xr.open_mfdataset(fn_list, combine='by_coords', chunks=CHUNKS, drop_variables='orig_names',
                                   parallel=parallel)
        else:
            ds = xr.open_dataset(fn_list[0],  chunks=CHUNKS)
        #make_tot = True
        #for var in TOTAL_VARS:
        #    if var not in ds.data_vars:
        #       make_tot=False

        #if make_tot:
        #    ds['dNdlogD'] = ds[D_NDLOG_D_SEC] + ds[D_NDLOG_D_MOD]
        #    ds['dNdlogD'].attrs = ds[D_NDLOG_D_MOD].attrs
        #    ds['dNdlogD'].attrs['long_name'] = 'dNdlogD'
        #elif D_NDLOG_D_MOD in ds.data_vars and not self.isSectional:
        #    ds['dNdlogD'] = ds[D_NDLOG_D_MOD].copy()
        #    ds['dNdlogD'].attrs = ds[D_NDLOG_D_MOD].attrs
        #    ds['dNdlogD'].attrs['long_name'] = 'dNdlogD'

        return ds


    def savepath_coll_ds(self, var_name):
        sp = self.savepath_root
        st = sp + '%s'
        st = '%s/%s/%s/%s_%s' % (sp, self.model_name, self.case_name,
                                 var_name, self.case_name)
        st = st + '_%s_%s' % (self.from_time, self.to_time)
        st = st + '_%s_%s' % (self.time_resolution, self.space_resolution)
        fn = st + '.nc'
        make_folders(fn)
        return fn

    def collocate_dataset_vars(self,var_names, redo=False):
        for var in var_names:
            self.collocate_dataset(var, redo=redo)



    def collocate_dataset(self, var_name, redo=False):
        """

        :return:
        """
        fn = self.savepath_coll_ds(var_name)
        if redo and os.path.isfile(fn):
            print(fn)
            #print('REMOVE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            os.remove(fn)

        elif os.path.isfile(fn): return
        ds =collocate_dataset(var_name, self.input_dataset, locations=self.locations)
        ds.to_netcdf(fn)
        ds.close()
        return




def collocate_dataset(var_name, ds, locations=constants.collocate_locations):
    """
    Collocate by method 'nearest' to locations
    :param var_name:
    :param ds:
    :param locations:
    :return:
    """
    # %%
    # maxDiameter = 39.6  # 23.6 #e-9
    # minDiameter = 5.0  # e-9
    # s = SizedistributionSurface('PD_SECT_CHC7_diur_ricc',  '2008-01-01',
    #                            '2009-01-01', [minDiameter, maxDiameter],
    #                            True, 'month')
    # ds = s.get_sizedist_var()
    # var_name = 'dNdlogD'
    da = ds[var_name]
    # da
    ds_tmp = xr.Dataset()
    # locations = constants.collocate_locations
    for loc in locations:
        lat = locations[loc]['lat']
        lon = locations[loc]['lon']
        ds_tmp[loc] = da.sel(lat=lat, lon=lon, method='nearest', drop=True)
        # print(lat, lon)
    da_out = ds_tmp.to_array(dim='location', name=var_name)
    for at in da.attrs:
        if at not in da_out.attrs:
            da_out.attrs[at] = da.attrs[at]
    del ds_tmp

    return da_out
