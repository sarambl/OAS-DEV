import os
from useful_scit.util.make_folders import make_folders
import pandas as pd
#####################################################################
# FILL IN FILEPATHS:
#####################################################################
# fill in path to project location (not including /OAS-DEV)
project_base_path = '/home/ubuntu/mnts/nird/projects/'
# name:
project_name = 'OAS-DEV'
# Fill in path to raw input data from NorESM:
raw_data_path_NorESM = project_base_path + 'model_output/archive/'
# Fill in path to raw input data from EUSAAR
path_eusaar_data = project_base_path + '/EUSAAR_data'

# Output processed data to:
path_outdata = project_base_path + '/Output_data_' + project_name + '/'

#####################################################################
# END FILL IN PART (no need to edit under this line)
#####################################################################

pathdic_raw_data = {'NorESM': raw_data_path_NorESM}#[file_source]}

def get_input_datapath(model = 'NorESM', file_source=None):
    return pathdic_raw_data[model]


## Plots path:
path_plots = project_base_path + '/Plots_' + project_name + '/'
paths_plotsave= dict(maps=path_plots + 'maps/',
                    comparison=path_plots + 'global_comparison/',
                    lineprofiles = path_plots + 'lineprofiles/',
                    sizedist = path_plots+'sizedistribution/',
                    sizedist_time = path_plots+'sizedist_time/',
                    levlat = path_plots+'levlat/',
                    eusaar = path_plots + 'eusaar/'
                    )
def get_plotpath(key):
    if key in paths_plotsave:
        return paths_plotsave[key]
    else:
        return path_plots+'/'+key + '/'


path_EBAS_data=project_base_path + '/EBAS_data'
# eusaar reformatted data:
path_eusaar_outdata = path_eusaar_data


def get_outdata_base():
    return path_outdata


outpaths={}
outpaths['pressure_coords']= path_outdata + '/Fields_pressure_coordinates'
outpaths['original_coords']= path_outdata + '/computed_fields_ng'
outpaths['computed_fields_ng']=path_outdata + '/computed_fields_ng' #native grid computed fields
outpaths['pressure_coords_converstion_fields'] = path_outdata +'/Pressure_coordinates_conversion_fields'
outpaths['pressure_density_path'] =path_outdata + '/Pressure_density'
outpaths['masks'] = path_outdata + '/means/masks/'
outpaths['area_means'] = path_outdata + '/means/area_means/'
outpaths['map_means'] = path_outdata+ '/means/map_means/'
outpaths['levlat_means'] = path_outdata+ '/means/levlat_means/'
outpaths['profile_means'] = path_outdata + '/means/profile_means/'
outpaths['sizedistrib_files'] = path_outdata + '/sizedistrib_files'
outpaths['collocated'] = path_outdata + '/collocated_ds/'
outpaths['eusaar'] = path_outdata + '/eusaar/'


def get_outdata_path(key):
    if key in outpaths:
        return outpaths[key]
    else:
        print('WARNING: key not found in outpaths, constants.py')
        return  path_outdata +'/' + key


make_folders(path_outdata)

# data info
path_data_info = project_base_path + 'OAS-DEV/oas_dev/data_info/'


# output locations:
locations = ['LON_116e_LAT_40n', 'LON_24e_LAT_62n', 'LON_63w_LAT_3s', 'LON_13e_LAT_51n']
path_locations_file = path_data_info+'locations.csv'

if os.path.isfile(path_locations_file):
    collocate_locations = pd.read_csv(path_locations_file, index_col=0)
else:
    _dic = dict(Hyytiala={'lat': 61.51, 'lon': 24.17},
                Melpitz={'lat': 51.32, 'lon': 12.56},
                Amazonas={'lat': -3., 'lon': -63.},
                Beijing={'lat': 40, 'lon': 116})
    collocate_locations = pd.DataFrame.from_dict(_dic)
    collocate_locations.to_csv(path_locations_file)




## Sizedistribution:
sized_varListNorESM = {'NCONC': ['NCONC01', 'NCONC02', 'NCONC04', 'NCONC05', 'NCONC06', 'NCONC07', 'NCONC08',
                           'NCONC09', 'NCONC10', 'NCONC12', 'NCONC14'],
                 'SIGMA': ['SIGMA01', 'SIGMA02', 'SIGMA04', 'SIGMA05', 'SIGMA06', 'SIGMA07', 'SIGMA08',
                           'SIGMA09', 'SIGMA10', 'SIGMA12', 'SIGMA14'],
                 'NMR': ['NMR01', 'NMR02', 'NMR04', 'NMR05', 'NMR06', 'NMR07', 'NMR08',
                         'NMR09', 'NMR10', 'NMR12', 'NMR14']}

sized_varlist_SOA_SEC = ['nrSOA_SEC01', 'nrSOA_SEC02', 'nrSOA_SEC03', 'nrSOA_SEC04', 'nrSOA_SEC05']

sized_varlist_SO4_SEC = ['nrSO4_SEC01', 'nrSO4_SEC02', 'nrSO4_SEC03', 'nrSO4_SEC04', 'nrSO4_SEC05']

list_sized_vars_noresm = sized_varListNorESM['NCONC'] + \
                         sized_varListNorESM['SIGMA'] + \
                         sized_varListNorESM['NMR'] + \
                         sized_varlist_SOA_SEC + \
                         sized_varlist_SO4_SEC
list_sized_vars_nonsec = sized_varListNorESM['NCONC'] + sized_varListNorESM['SIGMA'] + sized_varListNorESM['NMR']



# Imports
import_always_include = ['P0', 'area', 'landfrac', 'hyam', 'hybm', 'PS', 'gw', 'LOGR',
                  'hyai', 'hybi', 'ilev'] # , 'date',  'LANDFRAC','Press_surf',

import_constants = ['P0', 'GRIDAREA', 'landfrac', 'hyam', 'hybm', 'gw', 'LOGR',
                    'hyai', 'hybi', 'ilev', 'LANDFRAC']
not_pressure_coords = ['P0','hyam', 'hybm', 'PS', 'gw', 'LOGR', 'aps',
                  'hyai', 'hybi', 'ilev', 'date']

vars_time_not_dim = ['P0', 'area',]



default_units = dict(
    numberconc={
        'units':'#/cm3',
        'factor':1.e-6,
        'exceptions':['N_AER']
    },
    NMR={
        'units':'nm',
        'factor':1.e9,
        'exceptions':[]
    },
    mixingratio= {
        'unit':'$\mu$g/kg',
        'factor':1e9,
        'exceptions':[]
    }

)