import os

import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib import pyplot as plt, pylab as plt
from scipy.stats import lognorm

#import analysis_tools.plot_settings
import sectional_v2.util.naming_conventions.var_info
from sectional_v2.util.plot import plot_settings


#from analysis_tools import var_overview_sql, import_fields_xr, fix_xa_dataset, practical_functions, \
#    practical_functions as practical_functions
from sectional_v2.scrap import import_fields_xr, fix_xa_dataset
from sectional_v2.util import practical_functions,var_overview_sql

from sectional_v2.util.plot.plot_settings import set_plot_vars, set_equal_axis
from sectional_v2.util.practical_functions import get_filename_Nd, get_foldername_Nd, make_folders, save_dataset_to_netcdf
from sectional_v2.util.naming_conventions.var_info import get_varname_Nd, get_fancylabel_Nd
#from analysis_tools.sizedistrib import varlist, vars_N_EC_Earth, N2radi_EC_Earth, N2sigma_EC_Earth, vars_N_ECHAM, \
#    N2radi_ECHAM, N2sigma_ECHAM, get_filename_avg_sizedist_dtset

from sectional_v2.scrap.sizedistrib import varlist, vars_N_EC_Earth, N2radi_EC_Earth, N2sigma_EC_Earth, vars_N_ECHAM, \
    N2radi_ECHAM, N2sigma_ECHAM, get_filename_avg_sizedist_dtset

def import_and_calculate_Nd(caseName, path, toNd, fromNd=0, path_savePressCoord='', model_name='NorESM', from_year=0,
                            to_year=99999, comp='atm', size_distrib=True, EC_earth_comp='tm5', area='Global',
                            pressure_adjust=True):
    """
    Imports and creates . If variables already computed with specifications, reads from file in
    fixed dataset_path.
    :param caseName: the name of the case
    :param path: path to inputdata
    :param path_savePressCoord: Pressure coordinate data if exists.
    :param model_name: 'NorESM', 'EC-Earth', 'ECHAM'
    :param from_year: start year to be fetched
    :param to_year: End year to be fethced
    :param comp: NorESM component. Should not be changed
    :param size_distrib: If size_distribution True, imports size distribution data.
    :param EC_earth_comp:
    :param area: Which area to be averaged
    :param pressure_adjust: Use pressure coordinates.
    :return: Dataset with the averaged data
    """
    print('Importing and averaging')
    filen = get_filename_Nd(caseName, from_year, model_name, pressure_adjust, to_year, fromNd, toNd)
    # Check if already calculated:
    print('Checking for file %s' %filen)
    if  os.path.isfile(filen):
        xr_ds=xr.open_dataset(filen)
        if 'Pres_addj' in xr_ds.attrs:
            if (xr_ds.attrs['Pres_addj']=='True'):
                xr_ds.attrs['Pres_addj']=True
            else: xr_ds.attrs['Pres_addj']=False


        keys = ['units', 'lev_is_dim', 'path_computed_data', 'is_computed_var']
        var_entery = ['#/cm3', 1, filen, 1]
        varNameN = get_varname_Nd(fromNd, toNd)
        var_overview_sql.open_and_create_var_entery(model_name,
                                                    caseName,
                                                    varNameN, var_entery, keys)
        return xr_ds

    first=True
    for yr in np.arange(from_year, to_year+1):
        print(yr)
        for mnth in np.arange(1,13): #
            print(mnth)
            xr_ds_month = import_fields_xr.xr_import_one_month(caseName, varlist[model_name], path, model_name=model_name,
                                                               comp=comp,
                                                               EC_earth_comp=EC_earth_comp, size_distrib=True, month=mnth, year=yr)
            # do fix (make variable names the same, fix units etc)
            xr_ds_month = fix_xa_dataset.xr_fix(xr_ds_month, model_name, sizedistribution=True)
            if pressure_adjust:
                xr_ds_month, conv_vars = fix_xa_dataset.xr_hybsigma2pressure(xr_ds_month, model_name, varlist[model_name],
                                                                             return_pressurevars=True)


            if (model_name == 'ECHAM'):
                for var in varlist[model_name]:
                    if 'NUM' in var:
                        xr_ds_month.attrs['caseName']=caseName

                        xr_ds_month = fix_xa_dataset.perMassAir2perVolume(xr_ds_month, model_name, var, path_to_data = path, Path_savePressCoord=path_savePressCoord, press_coords=pressure_adjust)
                        # convert from /m3 --> /cm3
                        xr_ds_month[var].values=xr_ds_month[var].values*1e-6
                        xr_ds_month[var].attrs['units']='#/cm3'
            if (model_name == 'NorESM'):
                xr_ds_month['NMR00']=xr_ds_month['NCONC00']*0 + 62.6 #nm Kirkevag et al 2018
                xr_ds_month['SIGMA00']=xr_ds_month['NCONC00']*0 + 62.6 #nm Kirkevag et al 2018


            ############### MAKE new Nd dataset:
            # Initialize empty dataset:
            dtset_Nd = xr_ds_month.drop(list(xr_ds_month.data_vars))
            varNameN = get_varname_Nd(fromNd, toNd)

            #############################################
            ##    NorESM
            #############################################
            if model_name=='NorESM':
                calc_Nd_interval_NorESM(dtset_Nd, fromNd, toNd, varNameN, xr_ds_month)
            #############################################
            ##    EC-Earth & EC-Earth
            #############################################
            if (model_name in ['EC-Earth', 'ECHAM']):
                #dtset_Nd[varNameN]=xr_ds_month[]
                if model_name=='EC-Earth':
                    dtset_Nd[varNameN]=xr_ds_month[vars_N_EC_Earth[0]]*0. #keep dimensions, zero value
                    dtset_Nd[varNameN].values = np.zeros_like(dtset_Nd[varNameN].values)
                    radi_list = N2radi_EC_Earth
                    sigmaList=N2sigma_EC_Earth
                elif model_name =='ECHAM':
                    dtset_Nd[varNameN]=xr_ds_month[vars_N_ECHAM[0]]*0. #keep dimensions, zero value
                    dtset_Nd[varNameN].values = np.zeros_like(dtset_Nd[varNameN].values)
                    radi_list = N2radi_ECHAM
                    sigmaList=N2sigma_ECHAM
                for varN in radi_list:
                    #varSIG = sigmaList[varN]#'SIGMA%02.0f'%(i+1)
                    varNMR = radi_list[varN]#'NMR%02.0f'%(i+1)
                    NCONC = xr_ds_month[varN].values#*10**(-6) #m-3 --> cm-3
                    SIGMA = sigmaList[varN]#plt_ds[varSIG].values#case[varSIG][lev]#*10**6
                    NMR = xr_ds_month[varNMR].values*2#*1e9 #case[varNMR][lev]*2  #  radius --> diameter
                    if fromNd>0:
                        dummy = NCONC*(lognorm.cdf(toNd, np.log(SIGMA), scale=NMR)) - NCONC*(lognorm.cdf(fromNd, np.log(SIGMA), scale=NMR))
                    else:
                        dummy = NCONC*(lognorm.cdf(toNd, np.log(SIGMA), scale=NMR))
                    # if NMR=0 --> nan values. We set these to zero:
                    dummy[NMR==0]=0.
                    dummy[NCONC==0]=0.
                    dummy[np.isnan(NCONC)]=np.nan
                    dtset_Nd[varNameN] += dummy#NCONC*(lognorm.cdf(toNd, np.log(SIGMA), scale=NMR))




            if first:
                save_ds=dtset_Nd.copy()
                save_ds[varNameN].attrs['fancy_name'] = get_fancylabel_Nd(fromNd, toNd)
                first=False
            else:
                save_ds=xr.concat([save_ds, dtset_Nd], "time") #savexr_ds[var].copy()

    if pressure_adjust:
        save_ds[varNameN].attrs['Pres_addj'] = True
    else:
        save_ds[varNameN].attrs['Pres_addj'] = False
    save_ds.attrs['startyear'] = from_year
    save_ds.attrs['endyear'] = to_year
    save_ds.attrs['case_name'] = caseName
    save_ds.attrs['Pres_addj'] = pressure_adjust
    # save dataset:
    foldern= get_foldername_Nd(caseName, from_year, model_name, pressure_adjust, to_year, fromNd, toNd)
    try: make_folders(foldern)
    except OSError: print ('Error: Creating directory. ' +foldern)
    save_dataset_to_netcdf(save_ds,filen)
    keys = ['units', 'lev_is_dim', 'path_computed_data', 'is_computed_var']
    var_entery = ['#/cm3', 1, filen, 1]
    var_overview_sql.open_and_create_var_entery(model_name,
                                                                        caseName,
                                                                        varNameN, var_entery, keys)

    #analysis_tools.var_overview_sql.open_and_create_ar(var,caseName, model_name, 'one value', area, ['path_to_data'], [filen],
    #                                                                 pressure_coords= int(pressure_adjust), to_lev=pmin,
    #                                                                 avg_over_lev=avg_over_lev, at_lev= p_level)




    return save_ds


def calc_Nd_interval_NorESM(dtset_Nd, fromNd, toNd, varNameN, xr_ds_month):
    varN = 'NCONC%02.0f' % (1)
    dtset_Nd[varNameN] = xr_ds_month[varN] * 0.  # keep dimensions, zero value
    i = 0
    for i in np.arange(14):
        varN = 'NCONC%02.0f' % (i + 1)
        varSIG = 'SIGMA%02.0f' % (i + 1)
        varNMR = 'NMR%02.0f' % (i + 1)
        NCONC = xr_ds_month[varN].values  # *10**(-6) #m-3 --> cm-3
        SIGMA = xr_ds_month[varSIG].values  # case[varSIG][lev]#*10**6
        NMR = xr_ds_month[varNMR].values * 2  # *1e9 #case[varNMR][lev]*2  #  radius --> diameter

        if ((i + 1) not in [3, 11, 13]):  # 3,11 and 13 no longer in use.
            # nconc_ab_nlim[case][model]+=logR*NCONC*lognorm.pdf(logR, np.log(SIGMA),scale=NMR)
            if fromNd > 0:
                dummy = NCONC * (lognorm.cdf(toNd, np.log(SIGMA), scale=NMR)) - NCONC * (
                    lognorm.cdf(fromNd, np.log(SIGMA), scale=NMR))
            else:
                dummy = NCONC * (lognorm.cdf(toNd, np.log(SIGMA), scale=NMR))
            # if NMR=0 --> nan values. We set these to zero:
            dummy[NMR == 0] = 0.
            dummy[NCONC == 0] = 0.
            dummy[np.isnan(NCONC)] = np.nan
            dtset_Nd[varNameN] += dummy
            # NCONC*(lognorm.cdf(toNd, np.log(SIGMA),scale=NMR))- NCONC*(lognorm.cdf(fromNd, np.log(SIGMA),scale=NMR))


def import_and_calculate_Nd_from_avg_sizedist(caseName, path, toNd, fromNd=0, path_savePressCoord='', model_name='NorESM', from_year=0,
                            to_year=99999, comp='atm', size_distrib=True, EC_earth_comp='tm5', area='Global',
                            pressure_adjust=True, avg_lev=True, minlev=850.):
    """
    Imports and creates . If variables already computed with specifications, reads from file in
    fixed dataset_path.
    :param caseName: the name of the case
    :param path: path to inputdata
    :param path_savePressCoord: Pressure coordinate data if exists.
    :param model_name: 'NorESM', 'EC-Earth', 'ECHAM'
    :param from_year: start year to be fetched
    :param to_year: End year to be fethced
    :param comp: NorESM component. Should not be changed
    :param size_distrib: If size_distribution True, imports size distribution data.
    :param EC_earth_comp:
    :param area: Which area to be averaged
    :param pressure_adjust: Use pressure coordinates.
    :return: Dataset with the averaged data
    """

    filen = get_filename_avg_sizedist_dtset(area, avg_lev, caseName, from_year, 0, minlev, model_name,
                                            pressure_adjust, to_year)
    # Check if already calculated:
    print('Checking for file %s' %filen)
    if os.path.isfile(filen):
        xr_ds=xr.open_dataset(filen)
        if 'Pres_addj' in xr_ds.attrs:
            if (xr_ds.attrs['Pres_addj']=='True'):
                xr_ds.attrs['Pres_addj']=True
            else: xr_ds.attrs['Pres_addj']=False
    #    return xr_ds


    dtset_Nd = xr_ds.drop(list(xr_ds.data_vars))
    varNameN = get_varname_Nd(fromNd, toNd)

    #############################################
    ##    NorESM
    #############################################

    if model_name=='NorESM':
        varN='NCONC%02.0f'%(1)
        dtset_Nd[varNameN]=xr_ds[varN]*0. # keep dimensions, zero value
        # i=0
        for i in np.arange(14):
            varN='NCONC%02.0f'%(i+1)
            varSIG = 'SIGMA%02.0f'%(i+1)
            varNMR = 'NMR%02.0f'%(i+1)
            NCONC = xr_ds[varN].values#*10**(-6) #m-3 --> cm-3
            SIGMA = xr_ds[varSIG].values#case[varSIG][lev]#*10**6
            NMR = xr_ds[varNMR].values*2#*1e9 #case[varNMR][lev]*2  #  radius --> diameter

            if ((i+1) not in [3,11,13]): #3,11 and 13 no longer in use.
                #nconc_ab_nlim[case][model]+=logR*NCONC*lognorm.pdf(logR, np.log(SIGMA),scale=NMR)
                if fromNd>0:
                    dummy = NCONC*(lognorm.cdf(toNd, np.log(SIGMA),scale=NMR))- NCONC*(lognorm.cdf(fromNd, np.log(SIGMA),scale=NMR))
                else:
                    dummy = NCONC*(lognorm.cdf(toNd, np.log(SIGMA),scale=NMR))
                # if NMR=0 --> nan values. We set these to zero:
                if dummy==np.nan:
                    dummy=0.

                dtset_Nd[varNameN] += dummy #NCONC*(lognorm.cdf(toNd, np.log(SIGMA),scale=NMR))- NCONC*(lognorm.cdf(fromNd, np.log(SIGMA),scale=NMR))
    #############################################
    ##    EC-Earth & EC-Earth
    #############################################
    """
    if (model_name in ['EC-Earth', 'ECHAM']):
        #dtset_Nd[varNameN]=xr_ds_month[]
        if model_name=='EC-Earth':
            dtset_Nd[varNameN]= xr_ds_month[vars_N_EC_Earth[0]]*0. #keep dimensions, zero value
            dtset_Nd[varNameN].values = np.zeros_like(dtset_Nd[varNameN].values)
            radi_list = N2radi_EC_Earth
            sigmaList=N2sigma_EC_Earth
        elif model_name =='ECHAM':
            dtset_Nd[varNameN]=xr_ds_month[vars_N_ECHAM[0]]*0. #keep dimensions, zero value
            dtset_Nd[varNameN].values = np.zeros_like(dtset_Nd[varNameN].values)
            radi_list = N2radi_ECHAM
            sigmaList=N2sigma_ECHAM
        for varN in radi_list:
            #varSIG = sigmaList[varN]#'SIGMA%02.0f'%(i+1)
            varNMR = radi_list[varN]#'NMR%02.0f'%(i+1)
            NCONC = xr_ds_month[varN].values#*10**(-6) #m-3 --> cm-3
            SIGMA = sigmaList[varN]#plt_ds[varSIG].values#case[varSIG][lev]#*10**6
            NMR = xr_ds_month[varNMR].values*2#*1e9 #case[varNMR][lev]*2  #  radius --> diameter
            if fromNd>0:
                dummy = NCONC*(lognorm.cdf(toNd, np.log(SIGMA), scale=NMR)) - NCONC*(lognorm.cdf(fromNd, np.log(SIGMA), scale=NMR))
            else:
                dummy = NCONC*(lognorm.cdf(toNd, np.log(SIGMA), scale=NMR))
            # if NMR=0 --> nan values. We set these to zero:
            dummy[NMR==0]=0.
            dummy[NCONC==0]=0.
            dummy[np.isnan(NCONC)]=np.nan
            dtset_Nd[varNameN] += dummy#NCONC*(lognorm.cdf(toNd, np.log(SIGMA), scale=NMR))
    """
    save_ds=dtset_Nd.copy()
    save_ds[varNameN].attrs['fancy_name'] = get_fancylabel_Nd(fromNd, toNd)

    if pressure_adjust:
        save_ds[varNameN].attrs['Pres_addj'] = True
    else:
        save_ds[varNameN].attrs['Pres_addj'] = False
    save_ds.attrs['startyear'] = from_year
    save_ds.attrs['endyear'] = to_year
    save_ds.attrs['case_name'] = caseName
    save_ds.attrs['Pres_addj'] = pressure_adjust
    # save dataset:


    return save_ds


def get_figname_stacked_Nd(plotfilename, model, area, diameters, endyear, plot_mode, pressure_adjust, startyear):
    #plotfilename = 'N_stack'
    plotfilename =plotfilename + '_'+ model
    for diam in diameters:
        plotfilename = plotfilename + '_%d' % diam
    plotfilename = plotfilename + '%d_%d_' % (startyear, endyear)
    if pressure_adjust:
        plotfilename = plotfilename + '_pres_addj'
    plotfilename = plotfilename + '_' + area
    if plot_mode == 'presentation': plotfilename = plotfilename + '_presm'
    return plotfilename


def plot_profile_Nd(nested_profiles, area, pressure_adjust, diameters, plotpath='plots/profiles_Nd', plotType='number', lin_x=False, plot_mode='normal', ylim=[1000, 100]):
    """
    :param nested_profiles:
    :param area:
    :param pressure_adjust:
    :param diameters:
    :param plotpath:
    :param plotType:
    :param lin_x:
    :param plot_mode:
    :param ylim:
    :return:
    """
    [SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE, EVEN_BIGGER, colors_models, linestyle_models, linewidth, figsize] = set_plot_vars(plot_mode)
    #sns.set()
    plotpath = plotpath+'/%s/' %area
    try:
        make_folders(plotpath)
    except OSError:
        print ('Error: Creating directory. ' )


    # Make plot array:
    models=list(nested_profiles.keys())
    cases=list(nested_profiles[models[0]].keys())
    xmax_sens=0
    xmax_ctrl=0
    for model in models:

        fig, axarr = plt.subplots(2,3, figsize=figsize)
        ii=0
        my_ylim=0 # to set ylim equal in all cases
        for case in cases:
            plt_ds=nested_profiles[model][case]
            if case == 'CTRL':
                ctrl_save = plt_ds.copy()

            subp_ind=tuple(np.array([int(np.floor((ii)/3)),int(ii%3)]).astype(int))

            ii = ii+1

            lev=plt_ds['lev']
            variable_list=[]
            # Make list of the variables

            for i_var in np.arange(1,len(diameters)):
                if i_var==1:
                    lab=get_fancylabel_Nd(0,diameters[i_var])
                    variable_name= get_varname_Nd(0, diameters[i_var])+'_prof'
                    if case != 'CTRL':
                        plt_arr=plt_ds[variable_name]-ctrl_save[variable_name]
                    else:
                        plt_arr = plt_ds[variable_name]
                    axarr[subp_ind].plot(plt_arr, lev, label=lab, linewidth=linewidth)
                    #base = plt_arr

                else:
                    lab = get_fancylabel_Nd(diameters[i_var-1], diameters[i_var])
                    variable_name= get_varname_Nd(diameters[i_var-1], diameters[i_var])+'_prof'
                    if case != 'CTRL':
                        plt_arr =  plt_ds[variable_name]-ctrl_save[variable_name]
                    else:
                        plt_arr = plt_ds[variable_name]
                    axarr[subp_ind].plot(plt_arr, lev, label=lab, linewidth=linewidth)
                    #base = plt_arr
            if not lin_x:
                if case=='CTRL':
                    axarr[subp_ind].set_xscale('log')
                else:
                    axarr[subp_ind].set_xscale('symlog')

            axarr[subp_ind].set_ylim(ylim)
            axarr[subp_ind].set_yscale('log')
            if pressure_adjust:
                axarr[subp_ind].set_ylabel('Pressure [hPa]', fontsize=MEDIUM_SIZE)
            else:
                axarr[subp_ind].set_ylabel('Hybrid sigma [hPa]', fontsize=MEDIUM_SIZE)
            if case != 'CTRL':
                axarr[subp_ind].set_xlabel(r'$\Delta$ Number concentration [cm$^{-3}$]', fontsize=MEDIUM_SIZE)
            else:
                axarr[subp_ind].set_xlabel(r'Number concentration [cm$^{-3}$]', fontsize=MEDIUM_SIZE)
            if case == cases[-1]:
                axarr[subp_ind].legend(loc=2, fontsize=SMALL_SIZE)



            if (case=='CTRL'): title_txt=case
            else: title_txt=case+ ' - CTRL'
            axarr[subp_ind].set_title(title_txt, fontsize=MEDIUM_SIZE)
            if (case !='CTRL'):
                xlim_d=axarr[subp_ind].get_xlim()
                dummy=max(abs(xlim_d[0]),abs(xlim_d[1]))
                xmax_sens=max(dummy,xmax_sens)


        ii=0
        for case in cases:
            subp_ind=tuple(np.array([int(np.floor((ii)/3)),int(ii%3)]).astype(int))
            axarr[subp_ind].tick_params(axis='y',which='both',bottom='on')
            axarr[subp_ind].grid(True,which="both",ls="-",axis='both')
            axarr[subp_ind].grid(True,which='minor', linestyle=':',axis='y')#, linewidth='0.5', color='black')

            ii+=1
            if (case !='CTRL'):
                axarr[subp_ind].set_xlim([-xmax_sens,xmax_sens])
            else:
                #xlim_d = axarr[subp_ind].get_xlim()#[-xmax_sens,xmax_sens])
                axarr[subp_ind].set_xlim([1e-3,5e4])#xmax_sens])






        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=3.0)

        #figure name:
        #print(plt_ds.attrs)
        startyear = nested_profiles[model][case].attrs['startyear']
        endyear = nested_profiles[model][case].attrs['endyear']

        plotfilename = plotpath+'/'+ get_figname_stacked_Nd('N_d_prof', model, area, diameters, endyear, plot_mode, pressure_adjust, startyear)

        figname = plotfilename+'.png'


        plt.savefig(figname)
        plt.show()
    return


def plot_barplot(nested_dataframes, area, pressure_adjust, diameters, to_pressure, startyear, endyear,
                 plotpath='plots/bars_Nd',relative=False, plot_mode='normal' ):

    [SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE, EVEN_BIGGER, colors_models, linestyle_models, linewidth, figsize] = set_plot_vars(plot_mode)
    sns.set()
    plotpath = plotpath+'/%s/' %area
    try:
        make_folders(plotpath)
    except OSError:
        print ('Error: Creating directory. ' )
    fig, axs = plt.subplots(1,4, figsize=[20,6])

    models = list(nested_dataframes.keys())
    pr_cases = []
    i_mod = 0
    first = True
    ctrl_dtf = pd.DataFrame(index=list(nested_dataframes[models[0]].index))
    #print(ctrl_dtf)
    for model in models:
        ctrl_dtf[model] = nested_dataframes[model]['CTRL']
    ctrl_dtf.iloc[::-1].transpose().plot.bar(ax=axs[0], title='CTRL')
    #axs[0].set_ylabel(r'#/cm^3')
    mod_i = 1
    for model in models:
        plt_ds=pd.DataFrame(index=list(nested_dataframes[models[0]].index))
        for ii in np.arange(1, len(nested_dataframes[model].columns)):
            case = nested_dataframes[model].columns[ii]
            if relative:
                plt_ds[case] = 100*(nested_dataframes[model][case]-ctrl_dtf[model])/ctrl_dtf[model]
                axs[mod_i].set_ylabel(r'% change')
            else:
                plt_ds[case] = nested_dataframes[model][case]-ctrl_dtf[model]
                axs[mod_i].set_ylabel(r'#/cm^3')

        plt_ds.iloc[::-1].transpose().plot.bar(ax=axs[mod_i], legend=False, title=model)
        mod_i+=1
    ymax=0.
    for mod_i in np.arange(1,len(models)+1):
        ymax=max(ymax,np.max(np.abs(axs[mod_i].get_ylim())))
    for mod_i in np.arange(1,len(models)+1):
        axs[mod_i].set_ylim([-ymax,ymax])


    plotfilename = plotpath+'/'+ get_figname_bar_Nd('N_d_', model, area, diameters, endyear, plot_mode, pressure_adjust, startyear, to_pressure, relative)
    plt.tight_layout()
    plt.savefig(plotfilename)
    plt.show()
    return


def get_figname_bar_Nd(plotfilename, model, area, diameters, endyear, plot_mode, pressure_adjust, startyear, to_pressure, relative):
    #plotfilename = 'N_stack'
    plotfilename =plotfilename + '_'+ model
    for diam in diameters:
        plotfilename = plotfilename + '_%d' % diam
    plotfilename = plotfilename + '%d_%d_' % (startyear, endyear)
    if pressure_adjust:
        plotfilename = plotfilename + '_pres_addj'

    plotfilename = plotfilename + '_' + area
    plotfilename = plotfilename+'_toPres%.0f'%to_pressure
    if relative:
        plotfilename = plotfilename+ '_relative'
    if plot_mode == 'presentation': plotfilename = plotfilename + '_presm'
    return plotfilename


def get_N_nice_name_Nd(N_name):
    splt=N_name.split('_')
    if len(splt)==2:
        N_fancy_name='N$_{d<%s}$'%splt[1]
    else:
        N_fancy_name= 'N$_{%s<d<%s}$'%(splt[0][1::],splt[2])
    return N_fancy_name


def plot_profile_stacked(nested_profiles, area, pressure_adjust, diameters, plotpath='plots/profiles_Nd', plotType='number', lin_x=True, plot_mode='normal', ylim=[1000, 100], diff=False):
    """

    :param nested_profiles:
    :param area:
    :param pressure_adjust:
    :param diameters:
    :param plotpath:
    :param plotType:
    :param lin_x:
    :param plot_mode:
    :param ylim:
    :return:
    """
    [SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE, EVEN_BIGGER, colors_models, linestyle_models, linewidth, figsize] = set_plot_vars(plot_mode)
    #if (plot_mode=='presentation'):
    #    set_presentation_mode()
    #nsns.set()
    if plotpath=='plots/profiles_Nd':
        plotpath = plotpath+'/%s/' % area
    try:
        make_folders(plotpath)
    except OSError:
        print ('Error: Creating directory. ' )


    # Make plot array:
    models=list(nested_profiles.keys())
    cases=list(nested_profiles[models[0]].keys())
    xmax_sens=0
    xmax_ctrl=0
    for model in models:

        fig, axarr = plt.subplots(2,3, figsize=figsize)
        ii=0
        my_ylim=0 # to set ylim equal in all cases
        for case in cases:
            plt_ds=nested_profiles[model][case]
            if case == 'CTRL':
                ctrl_save = plt_ds.copy()

            subp_ind=tuple(np.array([int(np.floor((ii)/3)),int(ii%3)]).astype(int))

            ii = ii+1

            lev=plt_ds['lev']
            variable_list=[]
            # Make list of the variables

            for i_var in np.arange(1,len(diameters)):
                if i_var==1:
                    lab=get_fancylabel_Nd(0,diameters[i_var])
                    variable_name= get_varname_Nd(0, diameters[i_var])+'_prof'
                    if case != 'CTRL' and diff:
                        plt_arr=plt_ds[variable_name]-ctrl_save[variable_name]
                    else:
                        plt_arr = plt_ds[variable_name]
                    axarr[subp_ind].fill_betweenx(lev, plt_arr, label=lab, alpha=0.7)
                    base = plt_arr

                else:
                    lab = get_fancylabel_Nd(diameters[i_var-1], diameters[i_var])
                    variable_name= get_varname_Nd(diameters[i_var-1], diameters[i_var])+'_prof'
                    if case != 'CTRL' and diff:
                        plt_arr = base + plt_ds[variable_name]-ctrl_save[variable_name]
                    else:
                        plt_arr = base + plt_ds[variable_name]
                    axarr[subp_ind].fill_betweenx(lev, plt_arr, base, label=lab, alpha=0.7)
                    base = plt_arr
            if not lin_x:
                if case=='CTRL':
                    axarr[subp_ind].set_xscale('log')
                else:
                    axarr[subp_ind].set_xscale('symlog')

            axarr[subp_ind].set_ylim(ylim)
            axarr[subp_ind].set_yscale('log')
            if pressure_adjust:
                axarr[subp_ind].set_ylabel('Pressure [hPa]', fontsize=MEDIUM_SIZE)
            else:
                axarr[subp_ind].set_ylabel('Hybrid sigma [hPa]', fontsize=MEDIUM_SIZE)
            if case != 'CTRL' and diff:
                axarr[subp_ind].set_xlabel(r'$\Delta$ Number concentration [cm$^{-3}$]', fontsize=MEDIUM_SIZE)
            else:
                axarr[subp_ind].set_xlabel(r'Number concentration [cm$^{-3}$]', fontsize=MEDIUM_SIZE)
            if case == cases[-1]:
                axarr[subp_ind].legend(loc=2 ,fontsize=SMALL_SIZE)



            if (case=='CTRL' or not(diff)): title_txt=case
            else: title_txt=case+ ' - CTRL'
            axarr[subp_ind].set_title(title_txt, fontsize=MEDIUM_SIZE)
            if (case !='CTRL' or not(diff)):
                xlim_d=axarr[subp_ind].get_xlim()
                dummy=max(abs(xlim_d[0]),abs(xlim_d[1]))
                xmax_sens=max(dummy,xmax_sens)



        ii=0
        for case in cases:
            subp_ind=tuple(np.array([int(np.floor((ii)/3)),int(ii%3)]).astype(int))
            axarr[subp_ind].tick_params(axis='y',which='both',bottom='on')
            axarr[subp_ind].grid(True,which="both",ls="-",axis='both')
            axarr[subp_ind].grid(True,which='minor', linestyle=':',axis='y')#, linewidth='0.5', color='black')
            axarr[subp_ind].tick_params(labelsize=SMALL_SIZE)
            ii+=1
            if (case !='CTRL' and diff):
                axarr[subp_ind].set_xlim([-xmax_sens,xmax_sens])
            elif case!='CTRL' and not diff:
                axarr[subp_ind].set_xlim([0, xmax_sens])




        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=3.0)

        #figure name:
        #print(plt_ds.attrs)
        startyear = nested_profiles[model][case].attrs['startyear']
        endyear = nested_profiles[model][case].attrs['endyear']

        plotfilename = plotpath +'/' + get_figname_stacked_Nd('N_stack', model, area, diameters, endyear, plot_mode, pressure_adjust, startyear)

        figname = plotfilename+'.png'


        plt.savefig(figname)
        plt.show()
    return


def plot_Nd_bars_single_model(nested_pd_model, model, area, N_vars, relative=False,without_be20=True,plt_path='plots/bars_non_stacked/',
                              ctrl_case='CTRL', figsize=[10,6]):
    #print(nested_pd_model)
    plt_path = plt_path+area+'/'
    practical_functions.make_folders(plt_path)
    filen_base=plt_path+'bars_Nd_ctr_diff'
    cmap = sns.color_palette('muted', len(N_vars))#"cubehelix", n_colors=len(N_vars)) #cubehelix, husl
    #cmap = sns.diverging_palette(220, 20, l=60, n=len(N_vars), center="dark")
    filen=filen_base#plt_path+'bars_Nd_ctr_diff'
    if without_be20:
        filen=filen+'no_sub20'
    #print(model) #fig, axs = plt.subplots(1,2, figsize=[8,5],gridspec_kw = {'width_ratios':[1, 3]})
    if without_be20 and ('N$_{d<20}$' in nested_pd_model[model].index):
        pl_pd= nested_pd_model[model].drop(['N$_{d<20}$'])#, axis=1)#, axis=0)#.transpose()
    else:
        pl_pd= nested_pd_model[model]#.drop(['N$_{d<20}$'])#, axis=1)#, axis=0)#.transpose()
    pl_pd = pl_pd#.iloc[::-1]
    fig, axs = plt.subplots(1,2, figsize=figsize,gridspec_kw = {'width_ratios':[1, len(pl_pd.columns)-1]})
    ax=axs#xs[int(np.floor(ii/3)),ii%3]
    #print(pl_pd)
    pl_pd[ctrl_case].transpose().plot.bar(ax=axs[0], fontsize=14, title=ctrl_case, width=1)#, color=cmap)
    if relative:
        plt_diff = pl_pd.drop(ctrl_case, axis=1).sub( pl_pd[ctrl_case], axis=0).div(np.abs(pl_pd[ctrl_case]), axis=0)*100.
    else:
        plt_diff = pl_pd.drop(ctrl_case, axis=1).sub( pl_pd[ctrl_case], axis=0)#.div(pl_pd['CTRL'])
    if relative:
        plt_diff.transpose().plot.bar(ax=axs[1], fontsize=14, title='Relative difference', width=0.95, legend=False)#, color=cmap)
    else:
        plt_diff.transpose().plot.bar(ax=axs[1], fontsize=14, title='Difference', width=0.9, legend=False)#, color=cmap)
    if relative:
        ax[0].set_ylabel('#/cm$^3$', fontsize=14)
        ax[1].set_ylabel('%', fontsize=14)
    else:
        ax[0].set_ylabel('#/cm$^3$', fontsize=14)
    if relative:
        filen=filen+'_'+model+'rel.png'
    else:
        filen=filen+'_'+model+'.png'
    print(filen)
    plt.tight_layout()
    plt.savefig(filen, dpi=300)
    plt.show()
    return


def plot_Nd_bars_all_models(nested_pd_model, models, area, N_vars, relative=True, sharey_for_diffrel=False,
                            sharey_for_ctrl=False, without_be20=True, plt_path='plots/bars_non_stacked/'):
    plt_path=plt_path +area+'/'
    practical_functions.make_folders(plt_path)
    filen_base=plt_path+'bars_Nd_ctr_diff'
    cmap = sns.color_palette('Paired')#"cubehelix", n_colors=len(N_vars)) #cubehelix, husl
    cmap = sns.diverging_palette(220, 20, l=60, n=len(N_vars), center="dark")

    fig, axs = plt.subplots(len(models),2, figsize=[12,11],gridspec_kw = {'width_ratios':[1, 5]})

    ii=0
    filen=filen_base #plt_path+'bars_Nd_ctr_diff'
    if without_be20:
        filen=filen+'no_sub20'
    for model in models:
        print(model) #fig, axs = plt.subplots(1,2, figsize=[8,5],gridspec_kw = {'width_ratios':[1, 3]})
        ax=axs#xs[int(np.floor(ii/3)),ii%3]
        if without_be20 and ('N$_{d<20}$' in nested_pd_model[model].index):
            pl_pd= nested_pd_model[model].drop(['N$_{d<20}$'])#, axis=1)#, axis=0)#.transpose()
        else:
            pl_pd = nested_pd_model[model]#.drop(['N$_{d<20}$'])#, axis=1)#, axis=0)#.transpose()
        pl_pd.index.name = None

        ax=axs[ii,0]
        plot_Nd_bars_in_ax(ax, axs, ii, model, pl_pd, relative)
        if relative:
            axs[ii,0].set_ylabel('#/cm$^3$', fontsize=14)
            axs[ii,1].set_ylabel('%', fontsize=14)
        else:
            axs[ii,0].set_ylabel('#/cm$^3$', fontsize=14)
        filen = filen+'_'+model
        if ii<len(models)-1:
            axs[ii,1].get_xaxis().set_ticklabels([])
            axs[ii,0].get_xaxis().set_ticklabels([])
        ii+=1
    if sharey_for_diffrel:
        set_equal_axis(axs[:, 1], which='y')
    if sharey_for_ctrl:
        set_equal_axis(axs[:, 0], which='y')
    for ii in np.arange(len(models)):
        plot_settings.insert_abc(axs[ii,0],14,ii*2)
        plot_settings.insert_abc(axs[ii,1],14,ii*2+1)
    if relative:
        filen = filen+'_rel.png'
    else:
        filen = filen+'.png'
    print(filen)
    plt.tight_layout(pad=2.)
    plt.savefig(filen, dpi=300)
    plt.show()


def plot_Nd_bars_in_ax(ax, axs, ii, model, pl_pd, relative):
    pl_pd['CTRL'].transpose().plot.bar(ax=ax, fontsize=14, title='%s: CTRL' % model, width=1)  # , color=cmap)
    if relative:
        plt_diff = pl_pd.drop('CTRL', axis=1).sub(pl_pd['CTRL'], axis=0).div(np.abs(pl_pd['CTRL']), axis=0) * 100.
    else:
        plt_diff = pl_pd.drop('CTRL', axis=1).sub(pl_pd['CTRL'], axis=0)  # .div(pl_pd['CTRL'])
    ax = axs[ii, 1]
    if relative:
        plt_diff.transpose().plot.bar(ax=ax, fontsize=14, title='%s: relative difference' % model, width=0.85,
                                      legend=False)  # , color=cmap)
    else:
        plt_diff.transpose().plot.bar(ax=ax, fontsize=14, title='%s: difference' % model, width=0.9,
                                      legend=False)  # , color=cmap)


def extract_nested_pddf_by_model(models, cases, nested_pd):#
    nested_pd_model ={}
    for model in models:
        first = True
        for case in cases:
            pd_dummy = nested_pd[case][model].to_frame(name=case)
            if first:
                nested_pd_model[model] = pd_dummy
                first = False
            else:
                nested_pd_model[model]= pd.concat([nested_pd_model[model], pd_dummy],axis=1, join_axes=[nested_pd_model[model].index])#, sort=False)#, sorted=False)

    for case in cases:
        #nested_pd[case]= pd.read_csv(filename)
        dummy = nested_pd[case].reset_index()
        for i in np.arange(len(dummy)):
            dummy.loc[i, 'index'] = get_N_nice_name_Nd(dummy.loc[i, 'index'])
        dummy = dummy.rename(columns = {'index' : 'Variable'})
        nested_pd[case]= dummy.set_index('Variable')
    for model in models:
        dummy = nested_pd_model[model].reset_index()
        for i in np.arange(len(dummy)):
            dummy.loc[i, 'index']=get_N_nice_name_Nd(dummy.loc[i, 'index'])
        dummy = dummy.rename(columns = {'index' : 'Variable'})
        nested_pd_model[model]= dummy.set_index('Variable')
    return nested_pd_model


def get_Nd_fields(models, cases, pathRawData, startyear, endyear, comp ='atm', area='Global', pressure_adjust=True,
                  diameters=None,
                  val_type='between'):
    print(pathRawData)  # [model])

    if diameters is None:
        diameters = [20., 60., 80., 100., 200., 500., 1000.]
    nested_datasets = {}
    N_vars = []
    for model in models:
        case_dic = {}
        for case in cases:
            first = True
            N_vars = []
            for i in np.arange(len(diameters) - 1):
                # if vals_between_diam:
                if val_type == 'between':
                    print(diameters[i])
                    print(pathRawData)#[model])
                    print(pathRawData)
                    xr_ds = import_and_calculate_Nd(case, pathRawData[model], diameters[i + 1],
                                                           fromNd=diameters[i], model_name=model,
                                                           from_year=startyear, to_year=endyear, comp=comp,
                                                           size_distrib=True, area=area,
                                                           pressure_adjust=pressure_adjust)
                    var_name = (
                        sectional_v2.util.naming_conventions.var_info.get_varname_Nd(diameters[i], diameters[i + 1]))
                elif val_type == 'below':
                    xr_ds = import_and_calculate_Nd(case, pathRawData[model], diameters[i + 1], model_name=model,
                                                           from_year=startyear, to_year=endyear, comp=comp,
                                                           size_distrib=True, area=area,
                                                           pressure_adjust=pressure_adjust)
                    var_name = (sectional_v2.util.naming_conventions.var_info.get_varname_Nd(0, diameters[i + 1]))
                elif val_type == 'above':
                    xr_ds = import_and_calculate_Nd(case, pathRawData[model], 10000., fromNd = diameters[i + 1],
                                                           model_name = model,
                                                           from_year = startyear, to_year = endyear, comp=comp,
                                                           size_distrib=True, area=area,
                                                           pressure_adjust=pressure_adjust)
                    var_name = (sectional_v2.util.naming_conventions.var_info.get_varname_Nd(diameters[i + 1], 10000.))
                else:
                    print('val_type %s not recognized' % val_type)

                keys = ['units', 'lev_is_dim', 'is_computed_var']
                var_entery = ['#/cm3', 1, 1]
                try:
                    var_overview_sql.open_and_create_var_entery(model, case, var_name, var_entery,
                                                                                    keys)
                except:
                    print('couldnt')

                if first:
                    full_ds = xr_ds.copy()
                    print(full_ds.attrs)
                    first = False
                else:
                    full_ds = xr.merge([xr_ds, full_ds])
                    full_ds.attrs = xr_ds.attrs
                N_vars.append(var_name)  # practical_functions.get_varname_Nd(diameters[i], diameters[i + 1]))
            case_dic[case] = full_ds.copy()
        nested_datasets[model] = case_dic.copy()
    del case_dic
    del xr_ds
    return nested_datasets, N_vars