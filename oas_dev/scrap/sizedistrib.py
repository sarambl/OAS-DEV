from typing import Dict, List, Any, Union
import sys
import xarray as xr
import numpy as np
#from analysis_tools import import_fields_xr, fix_xa_dataset, plot_settings
from sectional_v2.scrap import import_fields_xr, fix_xa_dataset
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#from analysis_tools.avg_pkg import average_timelatlon_lev, average_timelatlon_at_lev, get_average_area
#from sectional_v2.util.slice_average.avg_pkg import average_timelatlon_lev, average_timelatlon_at_lev, get_average_area
from sectional_v2.util.slice_average.avg_pkg.one_value import get_average_area

from sectional_v2.util.plot.plot_settings import set_plot_vars, MinorSymLogLocator, insert_abc
from sectional_v2.util.practical_functions import make_folders, save_dataset_to_netcdf, dataset_path



varListNorESM = ['NNAT_0', 'NCONC01','NCONC02','NCONC03','NCONC04','NCONC05','NCONC06','NCONC07','NCONC08','NCONC09','NCONC10','NCONC11','NCONC12','NCONC13','NCONC14',
           'SIGMA01','SIGMA02','SIGMA03','SIGMA04','SIGMA05','SIGMA06','SIGMA07','SIGMA08','SIGMA09','SIGMA10','SIGMA11','SIGMA12','SIGMA13','SIGMA14',
           'NMR01','NMR02','NMR03','NMR04','NMR05','NMR06','NMR07','NMR08','NMR09','NMR10','NMR11','NMR12','NMR13','NMR14']
varListNorESM_sec = varListNorESM.copy()
for i in range(1,6):
    varListNorESM_sec.append('nrSO4_SEC%02.0f'%i)
    varListNorESM_sec.append('nrSOA_SEC%02.0f'%i)

varListECEarth=['N_NUS', 'N_AIS', 'N_ACS', 'N_COS','N_AII','N_ACI','N_COI','RWET_AII', 'RWET_ACI','RWET_COI','RDRY_NUS', 'RDRY_AIS','RDRY_ACS', 'RDRY_COS']
varListECHAM=['RWET_NS','RWET_KS', 'RWET_AS','RWET_CS','RWET_KI','RWET_AI','RWET_CI','NUM_NS','NUM_KS','NUM_AS','NUM_CS','NUM_KI','NUM_AI','NUM_CI']

sigma_lognormal_ECEarth = [ 1.59, 1.59, 1.59, 2.00, 1.59, 1.59, 2.00 ]

varlist: Dict[str, List[Union[str, Any]]]={'NorESM':varListNorESM, 'EC-Earth': varListECEarth,'ECHAM':varListECHAM, 'NorESM_sec':varListNorESM_sec}


def import_and_average_sizedistrib_vars(caseName, path, path_savePressCoord = '', model_name='NorESM', from_year=0, to_year=99999,
                                        mustInclude='.h0.', comp='atm', size_distrib=True, EC_earth_comp='tm5', area='Global', level=1000.,
                                        avg_lev=False, minlev=850., pressure_adjust=True, sectional=False,
                                        look_for_file=True):
    """
    Imports and averages sizedistribution data. If variables already computed with specifications, reads from file in
    fixed dataset_path.
    :param caseName: the name of the case
    :param path: path to inputdata
    :param path_savePressCoord: Pressure coordinate data if exists.
    :param model_name: 'NorESM', 'EC-Earth', 'ECHAM'
    :param from_year: start year to be fetched
    :param to_year: End year to be fethced
    :param mustInclude: mostly redundant and only used for NorESM
    :param comp: NorESM component. Should not be changed
    :param size_distrib: If size_distribution True, imports size distribution data.
    :param EC_earth_comp:
    :param area: Which area to be averaged
    :param level: Which level to be fetched.
    :param avg_lev: If True: averages over pressure weigheted by pressure difference.
    :param minlev: To what level the pressure average should be calculated
    :param pressure_adjust: Use pressure coordinates.
    :return: Dataset with the averaged data
    """
    ######################################################################################
    # Imports and averages
    # If average of variables for sizedistribution is already saved, reads from file, if not, reads from
    ######################################################################################
    print('Importing and averaging')
    filen = get_filename_avg_sizedist_dtset(area, avg_lev, caseName, from_year, level, minlev, model_name,
                                            pressure_adjust, to_year, sectional=sectional)
    # Check if already calculated:
    print('Checking for file %s' %filen)
    if os.path.isfile(filen) and look_for_file:
        xr_ds=xr.open_dataset(filen)
        if 'Pres_addj' in xr_ds.attrs:
            if (xr_ds.attrs['Pres_addj']=='True'):
                xr_ds.attrs['Pres_addj']=True
            else: xr_ds.attrs['Pres_addj']=False
        return xr_ds


    first = True
    model_name_ = model_name
    if sectional:
        model_name_ = 'NorESM_sec'
    print(varlist[model_name_])
    for var in varlist[model_name_]:
        # import
        print(var)
        # import fields imports without doing anything to it:
        xr_ds = import_fields_xr.xr_import(caseName, [var], path,
                                           model_name=model_name, comp=comp, from_year=from_year,
                                           to_year=to_year, EC_earth_comp=EC_earth_comp, size_distrib=True)
        # do fix (make variable names the same, fix units etc)
        xr_ds = fix_xa_dataset.xr_fix(xr_ds, model_name, sizedistribution=True)
        if var=='NNAT_0': var='NCONC00'
        #print(xr_ds)
        # keep original names (reverse renaming in xr_fix):


        # convert to pressure coords
        if pressure_adjust:
            if first:
                xr_ds, conv_vars = fix_xa_dataset.xr_hybsigma2pressure(xr_ds, model_name, [var],
                                                                       return_pressurevars=True)
            else:
                xr_ds = fix_xa_dataset.xr_hybsigma2pressure(xr_ds, model_name, [var],
                                                            conv_vars=conv_vars)
        if (model_name=='ECHAM' and 'NUM' in var):
            xr_ds.attrs['caseName']=caseName

            xr_ds = fix_xa_dataset.perMassAir2perVolume(xr_ds, model_name, var, path_to_data=path, Path_savePressCoord=path_savePressCoord, press_coords=pressure_adjust)
            # convert from /m3 --> /cm3
            xr_ds[var].values=xr_ds[var].values*1e-6
            xr_ds[var].attrs['units']='#/cm3'

        # average:
        xr_ds  = get_average_area(xr_ds, [var], caseName, area, model_name, minlev, avg_lev, level, pressure_adjust,
                                  look_for_file=look_for_file)
        #xr_ds  = get_average_area(xr_ds, [var], model_name, minlev, caseName, area, avg_lev, level, pressure_adjust)
        #if avg_lev:

            #xr_ds=fix_xa_dataset.xr_calc_pressure_diff(xr_ds, model_name, path_savePressCoord=path_savePressCoord)

        #    xr_ds = average_timelatlon_lev(xr_ds, var, model_name, area, minlev)

        #else:

        #    xr_ds= average_timelatlon_at_lev(xr_ds, var, model_name, area, level)
        print(xr_ds)
        if var == 'NCONC00':
            xr_ds['SIGMA00']= xr_ds['NCONC00'] * 0 + 1.6  # Kirkevag et al 2018
            xr_ds['SIGMA00'].attrs['units']= '-'#xr_ds['NCONC00'] * 0 + 1.6  # Kirkevag et al 2018
            xr_ds['NMR00'] = xr_ds['NCONC00'] * 0 + 62.6  ##nm Kirkevag et al 2018
            xr_ds['NMR00'].attrs['units'] = 'nm'#xr_ds['NCONC00'] * 0 + 62.6  ##nm Kirkevag et al 2018
        if first:
            save_ds=xr_ds.copy()
            first = False
        else:
            save_ds[var] = xr_ds[var].copy()
        del xr_ds



    # save dataset:
    try: make_folders(filen)

    except OSError: print ('Error: Creating directory. ' + dataset_path + '/' + model_name + '/' + area)
    save_dataset_to_netcdf(save_ds,filen)

    return save_ds


def get_filename_avg_sizedist_dtset(area, avg_lev, caseName, from_year, level, minlev, model_name, pressure_adjust,
                                    to_year, sectional = False):

    if avg_lev:
        print('Weighted avg by pressure difference up to %.0f' % minlev)
        # sets filename to be saved:
        if pressure_adjust:
            filen = dataset_path + '/' + model_name + '/' + area + '/%s_avg2lev%.0f_%s_%s_%s_press_adj' % (
            model_name, minlev, caseName, from_year, to_year)
        else:
            filen = dataset_path + '/' + model_name + '/' + area + '/%s_avg2lev%.0f_%s_%s_%s' % (
            model_name, minlev, caseName, from_year, to_year)
    else:
        if pressure_adjust:  # if pressure coordinates:
            # sets filename to be saved:
            filen = dataset_path + '/' + model_name + '/' + area + '/%s_lev%.0f_%s_%s_%s_press_adj' % (
            model_name, level, caseName, from_year, to_year)
        else:  # if not pressure coordinates
            # sets filename to be saved:
            filen = dataset_path + '/' + model_name + '/' + area + '/%s_lev%.0f_%s_%s_%s' % (
            model_name, level, caseName, from_year, to_year)
    if sectional:
        filen = filen + '_sectional.nc'
    else:
        filen = filen +'.nc'
    return filen


#Returns average for specific level

#########################################################################
## AVERAGE TIME,LAT,LON, LEV weighted with pressure difference. 
#########################################################################


################################################################################################
## PLOT SCRIPTS:
################################################################################################
### General info:
#Sigma for EC-Earth/ECHAM:
sigma_lognormal = [ 1.59, 1.59, 1.59, 2.00, 1.59, 1.59, 2.00 ]
vars_N_EC_Earth=['N_NUS', 'N_AIS', 'N_ACS', 'N_COS','N_AII','N_ACI','N_COI']
vars_radi_EC_Earth=['RDRY_NUS', 'RDRY_AIS','RDRY_ACS', 'RDRY_COS','RWET_AII', 'RWET_ACI','RWET_COI']
# ECHAM:
#vars_N_ECHAM=['NUM_NS','NUM_KS','NUM_AS','NUM_CS','NUM_KI','NUM_AI','NUM_CI']
vars_radi_ECHAM=['RWET_NS','RWET_KS', 'RWET_AS','RWET_CS','RWET_KI','RWET_AI','RWET_CI']
vars_N_ECHAM=['NUM_NS','NUM_KS','NUM_AS','NUM_CS','NUM_KI','NUM_AI','NUM_CI']
#varListECHAM=['RWET_NS', 'RWET_AS','RWET_KS','RWET_CS','RWET_KI','RWET_AI','RWET_CI','NUM_NS','NUM_KS','NUM_AS','NUM_CS','NUM_KI','NUM_AI','NUM_CI']
mode_names={}
mode_names_dic={}
mode_names['EC-Earth']=[s[-3::] for s in vars_N_EC_Earth ]
mode_names['ECHAM']=[s[-2::] for s in vars_N_ECHAM ]
mode_names['NorESM']=['SO4SOA_NA','BC_A', 'OMBC_AI', 'SO4_PR','DST_A2','DST_A3','SS_A1','SS_A2','SS_A3','BC_N','OMBC_NI']
mode_names_dic['NorESM']={'NCONC00':'BC_AX'}
for name, nr in zip(mode_names['NorESM'], [1,2,4,5,6,7,8,9,10,12,14]):
    varN='NCONC%02.0f'%nr
    mode_names_dic['NorESM'][varN] = name
mode_names_dic['EC-Earth']={}
for nvar in vars_N_EC_Earth:
    mode_names_dic['EC-Earth'][nvar] = nvar[-3::]
mode_names_dic['ECHAM']={}
for nvar in vars_N_ECHAM:
    mode_names_dic['ECHAM'][nvar] = nvar[-2::]

#, 'NCONC01'SO4SOA_NA','BC_A', 'OMBC_AI', 'SO4_PR','DST_A2','DST_A3','SS_A1','SS_A2','SS_A3','BC_N','OMBC_NI']


N2radi_EC_Earth={}
N2sigma_EC_Earth={}
N2radi_ECHAM={}
N2sigma_ECHAM={}

for jj in np.arange(len(vars_N_EC_Earth)):
    N2radi_EC_Earth[vars_N_EC_Earth[jj]]=vars_radi_EC_Earth[jj]
    N2sigma_EC_Earth[vars_N_EC_Earth[jj]]=sigma_lognormal[jj]
    N2radi_ECHAM[vars_N_ECHAM[jj]]=vars_radi_ECHAM[jj]
    N2sigma_ECHAM[vars_N_ECHAM[jj]]=sigma_lognormal[jj]

    #print(vars_N_EC_Earth[jj],vars_radi_EC_Earth[jj])
    #print(vars_N_EC_Earth[jj], sigma_lognormal[jj])
# for soluble nucleation (NUS), soluble Aitken (AIS), soluble accumulation (ACS), soluble coarse (COS), insoluble Aitken (AII), insoluble accumulation (ACI), insoluble coarse mode (COI).

# PLOT SETTINGS: #######################
def plot_sizedist_cases1fig(nested_means,plotpath, area,avg_over_lev, pmin, p_level,pressure_adjust, plotType='number',
                  ylim_min = {'number':1e2, 'surface': 1e-2,'volume' : 1e-1},
                  linthresh = {'number':3e1, 'surface': 1e-3,'volume' : 1e-2}, ctrl_name='CTRL',
                  xlim_dic = {'number':[3,1e3], 'surface':[3,1e4], 'volume': [3,1e4]},
                  y_log=True, lin_x=False, dNdD=False, plot_mode='normal' , figsize=[8,7], sectional=False):
    # nested_means should be nested dir with nested[model][case] containing xarray objects
    # plotType='number'/'surface'/'volume'
    # y_log=True decides if y-axis should be log
    # lin_x=False decides if linear x axis
    # dNdD=False if true prints dN/dD instead of dN/dlogD
    [SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE, EVEN_BIGGER, colors_models, linestyle_models, linewidth, figsize_d] = set_plot_vars(plot_mode)
    #if (plot_mode=='presentation'):
    #    set_presentation_mode()
    #sns.set(palette='bright')
    if dNdD:
        y_label_dic = {'number':'dN/dD$_p$  #/cm$^3$', 'surface':'dS/dD$_p$ um$^2$/cm$^3$', 'volume': 'dV/dD$_p$, um$^3$/cm$^3$'}
    else:
        y_label_dic = {'number':'dN/dlogD$_p$  #/cm$^3$', 'surface':'dS/dlogD$_p$ um$^2$/cm$^3$', 'volume': 'dV/dlogD$_p$, um$^3$/cm$^3$'}
    loc_leg_dic = {'number':1,'surface': 2, 'volume': 2}

    try:
        if not os.path.exists(plotpath):
            os.makedirs(plotpath)
            os.makedirs(plotpath+'/'+area)
        elif not os.path.exists(plotpath+'/'+area):
            os.makedirs(plotpath+'/'+area)
    except OSError:
        print ('Error: Creating directory. ' )
    # Make plot array:
    if lin_x:logR=np.linspace(10**(-3),10**3,300)
    else: logR=np.logspace(0,4,300)#*10**(-6)  # should be nmr

    ds_plot=xr.Dataset(coords={'Radius':logR})

    #for var in varList:
    fig, ax = plt.subplots(1, figsize=figsize)
    #fig.set_size_inches(13,10)#,foreward=True)
    ii=0
    ctrl_dic={} # keep ctr to substract from others
    my_ylim=0   # keep ylimits to use for all plots
    models=list(nested_means.keys())
    cases=list(nested_means[models[0]].keys())
    for case in cases:
        ii=ii+1

        for model in models:
            #print(model)
            #if (case=='CTRL'):

            plt_ds=nested_means[model][case]
            # Names of variables:
            nlogR, slogR, vlogR = make_sizedist_dtset(plt_ds, dNdD, model, logR)
            if sectional:
                nlogR_sec_SO4, nlogR_sec_SOA = create_sizedist_sec(plt_ds,logR)
                nlogR = nlogR +nlogR_sec_SO4 + nlogR_sec_SOA
            if len(models)>1: label = model + ': '+case
            else: label= case
            if plotType=='number':
                ax.plot(logR, nlogR, label=case, linestyle=linestyle_models[model], linewidth=linewidth)
            elif plotType=='surface':
                ax.plot(logR, slogR, label=case, linestyle=linestyle_models[model], linewidth=linewidth)
            elif plotType=='volume':
                ax.plot(logR, vlogR, label=case, linestyle=linestyle_models[model], linewidth=linewidth)
            ax.set_ylabel(y_label_dic[plotType], fontsize=MEDIUM_SIZE)
            if y_log: ax.set_yscale('symlog', linthreshy=linthresh[plotType])

            if not lin_x: ax.set_xscale('log')

            ax.set_xlabel('Diameter [nm]', fontsize=MEDIUM_SIZE)
            ax.tick_params(labelsize=SMALL_SIZE)

    ii=0
    ylim = ax.get_ylim()
    ax.set_ylim([ylim_min[plotType], ylim[1]])
    ax.set_title(area)
    ax.get_yaxis().set_minor_locator(MinorSymLogLocator(linthresh[plotType]))
    ax.grid(b=True,which="both",color='k', ls="-",axis='both', alpha=0.2)
    if not y_log:
        ax.ticklabel_format(axis='y',style='sci', scilimits=(0,4))
        ax.yaxis.offsetText.set_fontsize(MEDIUM_SIZE)

    ax.set_xlim(xlim_dic[plotType])
    ax.legend()
    plt.tight_layout(pad=0.8, w_pad=0.5, h_pad=3.0)
    startyear=nested_means[model][case].attrs['startyear']
    endyear=nested_means[model][case].attrs['endyear']
    if plotType=='number': plotname='sizedistrib'
    elif plotType=='surface': plotname='surfdist'
    elif plotType=='volume': plotname='volumedist'

    return ax

def plot_sizedist(nested_means,plotpath, area,avg_over_lev, pmin, p_level,pressure_adjust, plotType='number',
                  ylim_min = {'number':1e2, 'surface': 1e-2,'volume' : 1e-1},
                  linthresh = {'number':3e1, 'surface': 1e-3,'volume' : 1e-2}, ctrl_name='CTRL',
                  xlim_dic = {'number':[3,1e3], 'surface':[3,1e4], 'volume': [3,1e4]},
                  y_log=True, lin_x=False, dNdD=False, plot_mode='normal' , figsize=[18,10], sectional=False):
    # nested_means should be nested dir with nested[model][case] containing xarray objects
    # plotType='number'/'surface'/'volume'
    # y_log=True decides if y-axis should be log
    # lin_x=False decides if linear x axis
    # dNdD=False if true prints dN/dD instead of dN/dlogD
    [SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE, EVEN_BIGGER, colors_models, linestyle_models, linewidth, figsize_d] = set_plot_vars(plot_mode)
    #if (plot_mode=='presentation'):
    #    set_presentation_mode()
    #sns.set(palette='bright')
    if dNdD:
        y_label_dic = {'number':'dN/dD$_p$  #/cm$^3$', 'surface':'dS/dD$_p$ um$^2$/cm$^3$', 'volume': 'dV/dD$_p$, um$^3$/cm$^3$'}
    else:
        y_label_dic = {'number':'dN/dlogD$_p$  #/cm$^3$', 'surface':'dS/dlogD$_p$ um$^2$/cm$^3$', 'volume': 'dV/dlogD$_p$, um$^3$/cm$^3$'}
    loc_leg_dic = {'number':1,'surface': 2, 'volume': 2}

    try:
        if not os.path.exists(plotpath):
            os.makedirs(plotpath)
            os.makedirs(plotpath+'/'+area)
        elif not os.path.exists(plotpath+'/'+area):
            os.makedirs(plotpath+'/'+area)
    except OSError:
        print ('Error: Creating directory. ' )
    # Make plot array:
    if lin_x:logR=np.linspace(10**(-3),10**3,300)
    else: logR=np.logspace(0,4,300)#*10**(-6)  # should be nmr

    ds_plot=xr.Dataset(coords={'Radius':logR})

    #for var in varList:
    fig, axarr = plt.subplots(2,3, figsize=figsize)
    #fig.set_size_inches(13,10)#,foreward=True)
    ii=0
    ctrl_dic={} # keep ctr to substract from others
    my_ylim=0   # keep ylimits to use for all plots
    models=list(nested_means.keys())
    cases=list(nested_means[models[0]].keys())
    for case in cases:
        subp_ind=tuple(np.array([int(np.floor((ii)/3)),int(ii%3)]).astype(int))
        ii=ii+1

        for model in models:
            #print(model)
            #if (case=='CTRL'):

            plt_ds=nested_means[model][case]
            # Names of variables:
            nlogR, slogR, vlogR = make_sizedist_dtset(plt_ds, dNdD, model, logR)
            if sectional:
                nlogR_sec_SO4, nlogR_sec_SOA = create_sizedist_sec(plt_ds,logR)
                nlogR = nlogR +nlogR_sec_SO4 + nlogR_sec_SOA
            if (case== ctrl_name):#'CTRL'):

                if plotType=='number': ctrl_dic[model]=nlogR.copy()
                elif plotType=='surface': ctrl_dic[model]=slogR.copy()
                elif plotType=='volume': ctrl_dic[model]=vlogR.copy()
            else:
                nlogR=nlogR-ctrl_dic[model]
                slogR=slogR-ctrl_dic[model]
                vlogR=vlogR-ctrl_dic[model]


            if plotType=='number':
                axarr[subp_ind].plot(logR, nlogR, label=model, linestyle=linestyle_models[model], color=colors_models[model], linewidth=linewidth)
            elif plotType=='surface':
                axarr[subp_ind].plot(logR, slogR, label=model, linestyle=linestyle_models[model], color=colors_models[model], linewidth=linewidth)
            elif plotType=='volume':
                axarr[subp_ind].plot(logR, vlogR, label=model, linestyle=linestyle_models[model], color=colors_models[model], linewidth=linewidth)
            if subp_ind[1]==0:#,0] or subp_ind[1,0]
                axarr[subp_ind].set_ylabel(y_label_dic[plotType], fontsize=MEDIUM_SIZE)
            if y_log: axarr[subp_ind].set_yscale('symlog', linthreshy=linthresh[plotType])
            if case== ctrl_name:#'CTRL':
                axarr[subp_ind].legend(fontsize=SMALL_SIZE, loc=loc_leg_dic[plotType])

            if not lin_x: axarr[subp_ind].set_xscale('log')

            if (case==ctrl_name): title_txt=case
            else: title_txt=case+ ' - %s' %ctrl_name#CTRL'
            axarr[subp_ind].set_title(title_txt, fontsize=BIGGER_SIZE)
            axarr[subp_ind].set_xlabel('Diameter [nm]', fontsize=MEDIUM_SIZE)
            axarr[subp_ind].tick_params(labelsize=SMALL_SIZE)
            if (case !=ctrl_name):#'CTRL'):
                ylim = axarr[subp_ind].get_ylim()
                dummy = max(abs(ylim[0]),abs(ylim[1]))
                my_ylim = max(dummy,my_ylim)


    ii=0
    for case in cases:
        subp_ind = tuple(np.array([int(np.floor((ii)/3)),int(ii%3)]).astype(int))
        axarr[subp_ind].get_yaxis().set_minor_locator(MinorSymLogLocator(linthresh[plotType]))
        axarr[subp_ind].grid(b=True,which="both",color='k', ls="-",axis='both', alpha=0.2)
        if not y_log:
            axarr[subp_ind].ticklabel_format(axis='y',style='sci', scilimits=(0,4))
            axarr[subp_ind].yaxis.offsetText.set_fontsize(MEDIUM_SIZE)

        # set same ylim for all plots:
        axarr[subp_ind].set_xlim(xlim_dic[plotType])
        if (case !=ctrl_name):#'CTRL'):
            axarr[subp_ind].set_ylim([-my_ylim,my_ylim])
            axarr[subp_ind].get_yaxis().set_minor_locator(MinorSymLogLocator(linthresh[plotType]))
            axarr[subp_ind].grid(b=True,which="both",color='k', ls="-",axis='both', alpha=0.2)
        insert_abc(axarr[subp_ind], MEDIUM_SIZE + 1, ii)
        ii+=1


    plt.tight_layout(pad=0.8, w_pad=0.5, h_pad=3.0)
    startyear=nested_means[model][case].attrs['startyear']
    endyear=nested_means[model][case].attrs['endyear']
    if plotType=='number': plotname='sizedistrib'
    elif plotType=='surface': plotname='surfdist'
    elif plotType=='volume': plotname='volumedist'

    figname=plotpath+'/'+area+'/%s_%s_%.0f_%.0f' %(plotname,area,startyear,endyear)
    if avg_over_lev: figname=figname + '_2lev%.0f'%(pmin)
    else: figname=figname + '_atlev%.0f'%(p_level)
    if pressure_adjust: figname=figname + '_pressure_adj'
    if not y_log: figname=figname + '_linearY'
    if lin_x: figname=figname + '_linearX'
    if dNdD: figname=figname + '_dNdD'
    if (plot_mode=='presentation'): figname=figname+'presm'
    figname=figname+'.png'

    print(figname)
    plt.savefig(figname, dpi=200)
    plt.show()
    return


def make_sizedist_dtset(plt_ds, dNdD, model, logR, sectional=False):
    nlogR = np.zeros(logR.shape)
    slogR = np.zeros(logR.shape)
    vlogR = np.zeros(logR.shape)
    #############################################
    ##    NorESM
    #############################################
    if (model == 'NorESM'):
        for i in np.arange(15):
            varN = 'NCONC%02.0f' % (i)
            varSIG = 'SIGMA%02.0f' % (i)
            varNMR = 'NMR%02.0f' % (i)
            NCONC = plt_ds[varN].values  # *10**(-6) #m-3 --> cm-3
            SIGMA = plt_ds[varSIG].values  # case[varSIG][lev]#*10**6
            NMR = plt_ds[varNMR].values * 2  # *1e9 #case[varNMR][lev]*2  #  radius --> diameter
            if i==0:
                SIGMA =np.mean(SIGMA)
                NMR = np.mean(NMR)
            if ((i) not in [3, 11, 13]):  # 3,11 and 13 no longer in use.
                # slogR += NCONC/(np.log(SIGMA)*np.sqrt(2*np.pi))*np.exp(-(np.log(logR)-np.log(NMR))**2/(2*np.log(SIGMA)**2))*0 #OBS: no meaning
                nlogR += NCONC / (np.log(SIGMA) * np.sqrt(2 * np.pi)) * np.exp(
                    -(np.log(logR) - np.log(NMR)) ** 2 / (2 * np.log(SIGMA) ** 2))
                slogR += 1e-9 * 4. * np.pi * logR ** 2 * (NCONC / (np.log(SIGMA) * np.sqrt(2 * np.pi)) * np.exp(
                    -(np.log(logR) - np.log(NMR)) ** 2 / (2 * np.log(SIGMA) ** 2)))
                vlogR += 1e-9 * 1 / 6. * np.pi * logR ** 3 * (NCONC / (np.log(SIGMA) * np.sqrt(2 * np.pi)) * np.exp(
                    -(np.log(logR) - np.log(NMR)) ** 2 / (2 * np.log(SIGMA) ** 2)))
    #############################################
    ##    EC-Earth & EC-Earth
    #############################################
    if (model in ['EC-Earth', 'ECHAM']):
        nrModes = len(N2radi_EC_Earth.keys())
        nlogR = np.zeros(logR.shape)
        slogR = np.zeros(logR.shape)

        if model == 'EC-Earth':
            radi_list = N2radi_EC_Earth
            sigmaList = N2sigma_EC_Earth
        elif model == 'ECHAM':
            radi_list = N2radi_ECHAM
            sigmaList = N2sigma_ECHAM
        for varN in radi_list:
            varSIG = sigmaList[varN]  # 'SIGMA%02.0f'%(i+1)
            varNMR = radi_list[varN]  # 'NMR%02.0f'%(i+1)
            NCONC = plt_ds[varN].values  # *10**(-6) #m-3 --> cm-3
            SIGMA = sigmaList[varN]  # plt_ds[varSIG].values#case[varSIG][lev]#*10**6
            NMR = plt_ds[varNMR].values * 2  # *1e9 #case[varNMR][lev]*2  #  radius --> diameter

            nlogR += NCONC / (np.log(SIGMA) * np.sqrt(2 * np.pi)) * np.exp(
                -(np.log(logR) - np.log(NMR)) ** 2 / (2 * np.log(SIGMA) ** 2))
            slogR += 1e-9 * 4. * np.pi * logR ** 2 * (NCONC / (np.log(SIGMA) * np.sqrt(2 * np.pi)) * np.exp(
                -(np.log(logR) - np.log(NMR)) ** 2 / (2 * np.log(SIGMA) ** 2)))
            vlogR += 1e-9 * 1 / 6. * np.pi * logR ** 3 * (NCONC / (np.log(SIGMA) * np.sqrt(2 * np.pi)) * np.exp(
                -(np.log(logR) - np.log(NMR)) ** 2 / (2 * np.log(SIGMA) ** 2)))
    if dNdD:
        nlogR = nlogR / logR
        slogR = slogR / logR
        vlogR = vlogR / logR
    return nlogR, slogR, vlogR


    ##################################################################
    #### Create sectional dataset
    ##################################################################
def create_sizedist_sec(dtset, logD, one_time=True, nr_of_bins=5, minDiameter=5.0,
                        maxDiameter=23.6):
    """
    create sizedistribution with sectional.
    :param dtset:
    :param one_time:
    :return:
    """

    SECnr = nr_of_bins
    binDiam, binDiam_l = get_bin_diams(nr_of_bins,minDiameter=minDiameter, maxDiameter=maxDiameter)
    if not one_time:
        #dNlogD_sec = np.zeros([len(dtset.time), logD.shape[0]])
        dNlogD_sec_SO4 = np.zeros([len(dtset.time), logD.shape[0]])
        dNlogD_sec_SOA = np.zeros([len(dtset.time), logD.shape[0]])
        #dSlogD_sec = np.zeros([len(dtset.time), logD.shape[0]])
        dSlogD_sec_SO4 = np.zeros([len(dtset.time), logD.shape[0]])
        dSlogD_sec_SOA = np.zeros([len(dtset.time), logD.shape[0]])
    else:
        #dNlogD_sec = np.zeros([logD.shape[0]])
        dNlogD_sec_SO4 = np.zeros([logD.shape[0]])
        dNlogD_sec_SOA = np.zeros([logD.shape[0]])
        #dSlogD_sec = np.zeros([logD.shape[0]])
        dSlogD_sec_SO4 = np.zeros([logD.shape[0]])
        dSlogD_sec_SOA = np.zeros([logD.shape[0]])
    # dNdlogD=
    if ('nrSOA_SEC01' in dtset):
        for i in np.arange(SECnr):
            varSOA = 'nrSOA_SEC%02.0f' % (i + 1)
            varSO4 = 'nrSO4_SEC%02.0f' % (i + 1)
            athird = 1. / 3.
            SOA = dtset[varSOA].values  # *1e-6
            SO4 = dtset[varSO4].values  # *1e-6
            if (i != SECnr - 1):
                #dNdlogD = (SOA + SO4) * binDiam[i] / (binDiam_l[i + 1] - binDiam_l[i])
                dNdlogD_SO4 = (SO4) * binDiam[i] / (binDiam_l[i + 1] - binDiam_l[i])
                dNdlogD_SOA= (SOA) * binDiam[i] / (binDiam_l[i + 1] - binDiam_l[i])
                #dSdlogD = (SOA + SO4) * 1e-9 * 4. * np.pi * logD[i]** 2
                dSdlogD_SO4 = (SO4) * 1e-9 * 4. * np.pi * logD[i]** 2
                dSdlogD_SOA = (SOA) * 1e-9 * 4. * np.pi * logD[i]** 2
            else:
                #dSdlogD = (SOA + SO4) * 1e-9 * 4. * np.pi * logD[i]** 2
                dSdlogD_SO4 = (SO4) * 1e-9 * 4. * np.pi * logD[i]** 2
                dSdlogD_SOA = (SOA) * 1e-9 * 4. * np.pi * logD[i]** 2
                #dNdlogD = (SOA + SO4) * binDiam[i] / (maxDiameter - binDiam_l[i])
                dNdlogD_SO4 = (SO4) * binDiam[i] / (maxDiameter - binDiam_l[i])
                dNdlogD_SOA = (SOA ) * binDiam[i] / (maxDiameter - binDiam_l[i])
            if (i != SECnr - 1):
                inds = [j for j in np.arange(len(logD)) if (logD[j] >= binDiam_l[i] and logD[j] < binDiam_l[i + 1])]
            else:
                inds = [j for j in np.arange(len(logD)) if (logD[j] >= binDiam_l[i] and logD[j] < maxDiameter)]
            if not one_time:
                for j in np.arange(len(dtset['time'])):
                    #dNlogD_sec[j, inds] += dNdlogD[j]
                    dNlogD_sec_SO4[j, inds] += dNdlogD_SO4[j]
                    dNlogD_sec_SOA[j, inds] += dNdlogD_SOA[j]
                    #dSlogD_sec[j, inds] += dSdlogD[j]
                    dSlogD_sec_SO4[j, inds] += dSdlogD_SO4[j]
                    dSlogD_sec_SOA[j, inds] += dSdlogD_SOA[j]
            else:
                #dNlogD_sec[inds] += dNdlogD
                dNlogD_sec_SO4[inds] += dNdlogD_SO4
                dNlogD_sec_SOA[inds] += dNdlogD_SOA
                #dSlogD_sec[inds] += dSdlogD
                dSlogD_sec_SO4[inds] += dSdlogD_SO4
                dSlogD_sec_SOA[inds] += dSdlogD_SOA
    return dNlogD_sec_SO4, dNlogD_sec_SOA

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









def plot_sizedist_number_size(nested_means,cases, plotpath, area,avg_over_lev, pmin, p_level,pressure_adjust,
                              plotType=['number', 'surface'], ctrl_name = 'CTRL',
                              y_log=True, lin_x=False, dNdD=False, plot_mode='normal' , xlim=[1,3e3]):
    # nested_means should be nested dir with nested[model][case] containing xarray objects
    # plotType='number'/'surface'/'volume'
    # y_log=True decides if y-axis should be log
    # lin_x=False decides if linear x axis
    # dNdD=False if true prints dN/dD instead of dN/dlogD
    [SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE, EVEN_BIGGER, colors_models, linestyle_models, linewidth, figsize] = set_plot_vars(plot_mode)
    #if (plot_mode=='presentation'):
    #    set_presentation_mode()
    #sns.set(palette='bright')

    try:
        if not os.path.exists(plotpath):
            os.makedirs(plotpath)
            os.makedirs(plotpath+'/'+area)
        elif not os.path.exists(plotpath+'/'+area):
            os.makedirs(plotpath+'/'+area)
    except OSError:
        print ('Error: Creating directory. ' )
    # Make plot array:
    if lin_x:logR=np.linspace(10**(-3),10**3,300)
    else: logR=np.logspace(0,4,300)#*10**(-6)  # should be nmr

    ds_plot=xr.Dataset(coords={'Radius':logR})

    #for var in varList:
    fig, axarr = plt.subplots(len(plotType ),len(cases)-1, figsize=[figsize[0],int(figsize[1]/2.*len(plotType))])
    fig.set_size_inches(13, 10/2*len(plotType)) # foreward = True
    ii=0
    ctrl_dic={} # keep ctr to substract from others
    models=list(nested_means.keys())
    for model in models:
        ctrl_dic[model]={}
    my_ylim=np.zeros(len(plotType))   # keep ylimits to use for all plots
    for case in cases:
        subp_ind=tuple(np.array([int(np.floor((ii)/3)),int(ii%3)]).astype(int))


        for model in models:
            #print(model)
            #if (case=='CTRL'):
            plt_ds=nested_means[model][case]
            # Names of variables:
            nlogR=np.zeros(logR.shape)
            slogR=np.zeros(logR.shape)
            vlogR=np.zeros(logR.shape)

            #############################################
            ##    NorESM
            #############################################

            if (model=='NorESM'):
                for i in np.arange(15):
                    varN='NCONC%02.0f'%(i)
                    varSIG = 'SIGMA%02.0f'%(i)
                    varNMR = 'NMR%02.0f'%(i)
                    NCONC = plt_ds[varN].values#*10**(-6) #m-3 --> cm-3
                    SIGMA = plt_ds[varSIG].values#case[varSIG][lev]#*10**6
                    NMR = plt_ds[varNMR].values*2#*1e9 #case[varNMR][lev]*2  #  radius --> diameter
                    if ((i) not in [3,11,13]): #3,11 and 13 no longer in use.
                        #slogR += NCONC/(np.log(SIGMA)*np.sqrt(2*np.pi))*np.exp(-(np.log(logR)-np.log(NMR))**2/(2*np.log(SIGMA)**2))*0 #OBS: no meaning
                        nlogR += NCONC/(np.log(SIGMA)*np.sqrt(2*np.pi))*np.exp(-(np.log(logR)-np.log(NMR))**2/(2*np.log(SIGMA)**2))
                        slogR += 1e-9*4.*np.pi*logR**2*(NCONC/(np.log(SIGMA)*np.sqrt(2*np.pi))*np.exp(-(np.log(logR)-np.log(NMR))**2/(2*np.log(SIGMA)**2)))
                        vlogR += 1e-9*1/6.*np.pi*logR**3*(NCONC/(np.log(SIGMA)*np.sqrt(2*np.pi))*np.exp(-(np.log(logR)-np.log(NMR))**2/(2*np.log(SIGMA)**2)))

            #############################################
            ##    EC-Earth & EC-Earth
            #############################################
            if (model in ['EC-Earth', 'ECHAM']):
                nrModes=len(N2radi_EC_Earth.keys())
                nlogR=np.zeros(logR.shape)
                slogR=np.zeros(logR.shape)

                if model=='EC-Earth':
                    radi_list = N2radi_EC_Earth
                    sigmaList=N2sigma_EC_Earth
                elif model =='ECHAM':
                    radi_list = N2radi_ECHAM
                    sigmaList=N2sigma_ECHAM
                for varN in radi_list:
                    varSIG = sigmaList[varN]#'SIGMA%02.0f'%(i+1)
                    varNMR = radi_list[varN]#'NMR%02.0f'%(i+1)
                    NCONC = plt_ds[varN].values#*10**(-6) #m-3 --> cm-3
                    SIGMA =  sigmaList[varN]#plt_ds[varSIG].values#case[varSIG][lev]#*10**6
                    NMR = plt_ds[varNMR].values*2#*1e9 #case[varNMR][lev]*2  #  radius --> diameter

                    nlogR += NCONC/(np.log(SIGMA)*np.sqrt(2*np.pi))*np.exp(-(np.log(logR)-np.log(NMR))**2/(2*np.log(SIGMA)**2))
                    slogR += 1e-9*4.*np.pi*logR**2*(NCONC/(np.log(SIGMA)*np.sqrt(2*np.pi))*np.exp(-(np.log(logR)-np.log(NMR))**2/(2*np.log(SIGMA)**2)))
                    vlogR += 1e-9*1/6.*np.pi*logR**3*(NCONC/(np.log(SIGMA)*np.sqrt(2*np.pi))*np.exp(-(np.log(logR)-np.log(NMR))**2/(2*np.log(SIGMA)**2)))

            if dNdD:
                nlogR=nlogR/logR
                slogR=slogR/logR
                vlogR=vlogR/logR
            if (case== ctrl_name):#'CTRL'):
                print(model)
                ctrl_dic[model]['number']=nlogR.copy()
                ctrl_dic[model]['surface']=slogR.copy()
                ctrl_dic[model]['volume']=vlogR.copy()
            else:
                nlogR=nlogR-ctrl_dic[model]['number']
                slogR=slogR-ctrl_dic[model]['surface']
                vlogR=vlogR-ctrl_dic[model]['volume']
            # PLOTTING:
            jj=0
            if case==ctrl_name: continue
            for plott in plotType:
                if len(plotType)==1: subp_ind=ii-1
                else: subp_ind=tuple([jj,ii-1])#np.array([int(np.floor((ii)/3)),int(ii%3)]).astype(int))

                if plott=='number':
                    axarr[subp_ind].plot(logR, nlogR, label=model, linestyle=linestyle_models[model], color=colors_models[model], linewidth=linewidth)

                    if dNdD:axarr[subp_ind].set_ylabel('dN/dDp, #/cm$^3$', fontsize=MEDIUM_SIZE)
                    else: axarr[subp_ind].set_ylabel('dN/dlogDp, #/cm$^3$', fontsize=MEDIUM_SIZE)
                    if y_log: axarr[subp_ind].set_yscale('symlog', linthreshy=30)
                    #if case=='CTRL':
                    axarr[subp_ind].legend(fontsize=SMALL_SIZE, loc=1)

                elif plott=='surface':
                    axarr[subp_ind].plot(logR, slogR, label=model, linestyle=linestyle_models[model], color=colors_models[model], linewidth=linewidth)
                    if dNdD: axarr[subp_ind].set_ylabel(r'dS/dDp, um$^2$/cm$^3$', fontsize=MEDIUM_SIZE)
                    else: axarr[subp_ind].set_ylabel(r'dS/dlogDp, um$^2$/cm$^3$', fontsize=MEDIUM_SIZE)
                    if y_log: axarr[subp_ind].set_yscale('symlog', linthreshy=.003)
                    #if case=='CTRL':
                    axarr[subp_ind].legend(fontsize=SMALL_SIZE, loc=2)

                elif plott=='volume':
                    axarr[subp_ind].plot(logR, vlogR, label=model, linestyle=linestyle_models[model], color=colors_models[model], linewidth=linewidth)
                    if dNdD: axarr[subp_ind].set_ylabel(r'dV/dDp, um$^3$/cm$^3$', fontsize=MEDIUM_SIZE)
                    else: axarr[subp_ind].set_ylabel(r'dV/dlogDp, um$^3$/cm$^3$', fontsize=MEDIUM_SIZE)
                    if y_log: axarr[subp_ind].set_yscale('symlog', linthreshy=.01)
                    axarr[subp_ind].legend(fontsize=SMALL_SIZE, loc=2)


                if not lin_x: axarr[subp_ind].set_xscale('log')

                if (case==ctrl_name): title_txt=case
                else: title_txt=case+ ' - %s'%ctrl_name#CTRL'
                axarr[subp_ind].set_title(title_txt, fontsize=BIGGER_SIZE)
                axarr[subp_ind].set_xlabel('Diameter [nm]', fontsize=MEDIUM_SIZE)
                axarr[subp_ind].tick_params(labelsize=SMALL_SIZE)
                if (case != ctrl_name):#'CTRL'):
                    ylim = axarr[subp_ind].get_ylim()
                    dummy = max(abs(ylim[0]),abs(ylim[1]))
                    my_ylim[jj] = max(dummy,my_ylim[jj])
                jj+=1
        ii=ii+1


    ii=0
    for ii in np.arange(len(cases)-1):
        for jj in np.arange(len(plotType)):
            if len(plotType)==1: subp_ind=ii
            else: subp_ind=tuple([jj,ii])#np.array([int(np.floor((ii)/3)),int(ii%3)]).astype(int))
            #subp_ind = tuple([jj,ii])#tuple(np.array([int(np.floor((ii)/3)),int(ii%3)]).astype(int))
            axarr[subp_ind].yaxis.set_minor_locator(MinorSymLogLocator(1e1))

            axarr[subp_ind].grid(b=True,which="both",color='k', ls="-",axis='both')
            axarr[subp_ind].set_xlim(xlim)
            #ii+=1
            if not y_log:
                axarr[subp_ind].ticklabel_format(axis='y',style='sci', scilimits=(0,4))
                axarr[subp_ind].yaxis.offsetText.set_fontsize(MEDIUM_SIZE)

            # set same ylim for all plots:
            if (case !=ctrl_name):#'CTRL'):
                axarr[subp_ind].set_ylim([-my_ylim[jj],my_ylim[jj]])


    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=3.0)
    startyear=nested_means[model][case].attrs['startyear']
    endyear=nested_means[model][case].attrs['endyear']
    plotname=''
    for plott in plotType:
        plotname= plotname+plott
    for case in cases:
        plotname=plotname+case
    figname=plotpath+'/'+area+'/%s_%s_%.0f_%.0f' %(plotname,area,startyear,endyear)
    if avg_over_lev: figname=figname + '_2lev%.0f'%(pmin)
    else: figname=figname + '_atlev%.0f'%(p_level)
    if pressure_adjust: figname=figname + '_pressure_adj'
    if not y_log: figname=figname + '_linearY'
    if lin_x: figname=figname + '_linearX'
    if dNdD: figname=figname + '_dNdD'
    if (plot_mode=='presentation'): figname=figname+'presm'
    figname=figname+'.png'

    print(figname)
    plt.savefig(figname, dpi=300)
    plt.show()
    return



def gen_tick_positions(scale_start=100, scale_max=10000):

    start, finish = np.floor(np.log10((scale_start, scale_max)))
    finish += 1
    majors = [10 ** x for x in np.arange(start, finish)]
    minors = []
    for idx, major in enumerate(majors[:-1]):
        minor_list = np.arange(majors[idx], majors[idx+1], major)
        minors.extend(minor_list[1:])
    return minors, majors

#######################################	
## Stacked sizedistribution:
#######################################
def plot_sizedist_stacked_one_case(nested_means, case,plotpath, area,avg_over_lev, pmin, p_level,pressure_adjust, plotType='number',
                          y_log=True, lin_x=False, dNdD=False , plot_mode='normal', ctrl_name='CTRL',
                          ylim_min = {'number':1e2, 'surface': 1e-2,'volume' : 1e-1}, xlim = [3,1e4],
                          linthresh = {'number':3e1, 'surface': 1e-3,'volume' : 1e-3}, sectional = False, cmap='Paired',
                                   figsize=[8,5], ylim=[1,1e4]):
    [SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE, EVEN_BIGGER, colors_models, linestyle_models, linewidth, figsize_d] = set_plot_vars(plot_mode)
    #if (plot_mode=='presentation'):
    #    set_presentation_mode()
    #sns.set()
    #try:
    #    if not os.path.exists(plotpath):
    #        os.makedirs(plotpath)
    #        os.makedirs(plotpath+'/'+area)
    #    elif not os.path.exists(plotpath+'/'+area):
    #        os.makedirs(plotpath+'/'+area)
    #except OSError:
    #    print ('Error: Creating directory. ' )


    # Make plot array:
    if lin_x:logR=np.linspace(10**(-3),10**3,300)
    else: logR=np.logspace(0,4,100)#*10**(-6)  # should be nmr

    ds_plot=xr.Dataset(coords={'Radius':logR})


    ii=0
    ctrl_dic={}
    ctrl_dic_l={}
    models=list(nested_means.keys())
    cases=list(nested_means[models[0]].keys())

    for model in models:

        fig, axarr = plt.subplots(1, figsize=figsize)
        ax=axarr
        ii=0
        my_ylim=0 # to set ylim equal in all cases


        plt_ds=nested_means[model][case]
        xlogR_ds = get_sizedist_list(dNdD, logR, model, plotType, plt_ds)
        if sectional:
            nlogR_sec_SO4, nlogR_sec_SOA = create_sizedist_sec(plt_ds,logR)
            xlogR_ds['SEC_SOA'] = nlogR_sec_SOA
            xlogR_ds['SEC_SO4'] = nlogR_sec_SO4
        if (case==ctrl_name):

            ctrl_dic_l[model]=xlogR_ds.copy()
        else:
            xlogR_ds = xlogR_ds-ctrl_dic_l[model]

        #labels=['BC_AX'] +mode_names[model]+ ['SEC']
        labels = list(xlogR_ds.columns)
        col = sns.color_palette(cmap, len(labels))

        #xlogR_ds.plot.area(stacked=True, ax=axarr[subp_ind], colormap=cmap, alpha=.8)
        ax.stackplot(logR,xlogR_ds.transpose(), labels=labels,alpha=.8,colors=col)
        if plotType=='number':

            if dNdD:ax.set_ylabel('dN/dDp, #/cm$^3$')
            else: ax.set_ylabel('dN/dlogDp, #/cm$^3$')
            #axarr[subp_ind].legend(fontsize=10,loc=1)
            ax.legend(fontsize=SMALL_SIZE)

        elif plotType=='surface':
            #axarr[subp_ind].plot(logR, slogR,  label=model,linestyle=linestyle_models[model], color=colors_models[model], linewidth=linewidth)
            if dNdD: ax.set_ylabel(r'dS/dDp, um$^2$/cm$^3$')
            else: ax.set_ylabel(r'dS/dlogDp, um$^2$/cm$^3$')
            ax.legend( fontsize=SMALL_SIZE)

        elif plotType=='volume':
           #axarr[subp_ind].plot(logR, vlogR,  label=model,linestyle=linestyle_models[model], color=colors_models[model], linewidth=linewidth)
            if dNdD: ax.set_ylabel(r'dV/dDp, um$^3$/cm$^3$')
            else: ax.set_ylabel(r'dV/dlogDp, um$^3$/cm$^3$')
            ax.legend(fontsize=SMALL_SIZE)
        if y_log: ax.set_yscale('log')#''symlog', linthreshy=linthresh[plotType])
        ax.set_ylim(ylim)



        if not lin_x: ax.set_xscale('log')
    # set symlog so can use logarithmic scale with negative values (start at 10^-4)

        title_txt=case
        ax.set_title(title_txt, fontsize=MEDIUM_SIZE)
        ax.set_xlabel('Diameter [nm]', fontsize=MEDIUM_SIZE)

        ax.tick_params(axis='y',which='both',bottom='on')
        ax.grid(True,which="both",ls="-",axis='both')
        #ax.grid(True,which='minor', linestyle=':',axis='y')#, linewidth='0.5', color='black')
        if not y_log:
            ax.ticklabel_format(axis='y',style='sci', scilimits=(0,4))
            ax.yaxis.offsetText.set_fontsize(MEDIUM_SIZE)
        ii+=1
        ax.set_xlim(xlim)

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=3.0)

    #figure name:
    startyear=nested_means[model][case].attrs['startyear']
    endyear=nested_means[model][case].attrs['endyear']
    if plotType=='number': plotname='sizedistrib'
    elif plotType=='surface': plotname='surfdist'
    elif plotType=='volume': plotname='volumedist'

    figname=plotpath+'/'+area+'/%s_%s_%.0f_%.0f' %(plotname,area,startyear,endyear)
    if avg_over_lev: figname=figname + '_2lev%.0f'%(pmin)
    else: figname=figname + '_atlev%.0f'%(p_level)
    if pressure_adjust: figname=figname + '_pressure_adj'
    if not y_log: figname=figname + '_linearY'
    if lin_x: figname=figname + '_linearX'
    if dNdD: figname=figname + '_dNdD'
    figname=figname+'_'+model+'_stack'
    if (plot_mode=='presentation'): figname=figname+'presm'
    figname=figname+'.png'


    print(figname)
    #plt.savefig(figname)
    plt.show()
    return
#######################################
def plot_sizedist_stacked_n(nested_means,plotpath, area,avg_over_lev, pmin, p_level,pressure_adjust, plotType='number',
                          y_log=True, lin_x=False, dNdD=False , plot_mode='normal', ctrl_name='CTRL',
                          ylim_min = {'number':1e2, 'surface': 1e-2,'volume' : 1e-1}, xlim = [3,1e4],
                          linthresh = {'number':3e1, 'surface': 1e-3,'volume' : 1e-3}, sectional = False, cmap='Paired'):
    [SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE, EVEN_BIGGER, colors_models, linestyle_models, linewidth, figsize] = set_plot_vars(plot_mode)
    #if (plot_mode=='presentation'):
    #    set_presentation_mode()
    #sns.set()
    try:
        if not os.path.exists(plotpath):
            os.makedirs(plotpath)
            os.makedirs(plotpath+'/'+area)
        elif not os.path.exists(plotpath+'/'+area):
            os.makedirs(plotpath+'/'+area)
    except OSError:
        print ('Error: Creating directory. ' )


    # Make plot array:
    if lin_x:logR=np.linspace(10**(-3),10**3,300)
    else: logR=np.logspace(0,4,100)#*10**(-6)  # should be nmr

    ds_plot=xr.Dataset(coords={'Radius':logR})


    ii=0
    ctrl_dic={}
    ctrl_dic_l={}
    models=list(nested_means.keys())
    cases=list(nested_means[models[0]].keys())

    for model in models:

        fig, axarr = plt.subplots(2,3, figsize=figsize)
        ii=0
        my_ylim=0 # to set ylim equal in all cases
        for case in cases:

            subp_ind=tuple(np.array([int(np.floor((ii)/3)),int(ii%3)]).astype(int))

            ii=ii+1


            #print(model)
            #if (case=='CTRL'):
            plt_ds=nested_means[model][case]
            xlogR_ds = get_sizedist_list(dNdD, logR, model, plotType, plt_ds)
            if sectional:
                nlogR_sec_SO4, nlogR_sec_SOA = create_sizedist_sec(plt_ds,logR)
                xlogR_ds['SEC_SOA'] = nlogR_sec_SOA
                xlogR_ds['SEC_SO4'] = nlogR_sec_SO4
            if (case==ctrl_name):

                ctrl_dic_l[model]=xlogR_ds.copy()
            else:
                xlogR_ds = xlogR_ds-ctrl_dic_l[model]

            #labels=['BC_AX'] +mode_names[model]+ ['SEC']
            labels = list(xlogR_ds.columns)
            col = sns.color_palette(cmap, len(labels))

            #xlogR_ds.plot.area(stacked=True, ax=axarr[subp_ind], colormap=cmap, alpha=.8)
            axarr[subp_ind].stackplot(logR,xlogR_ds.transpose(), labels=labels,alpha=.8,colors=col)
            if plotType=='number':

                if dNdD:axarr[subp_ind].set_ylabel('dN/dDp, #/cm$^3$')
                else: axarr[subp_ind].set_ylabel('dN/dlogDp, #/cm$^3$')
                #axarr[subp_ind].legend(fontsize=10,loc=1)
                if case== ctrl_name:# 'CTRL':
                    axarr[subp_ind].legend(loc=4, fontsize=SMALL_SIZE)


            elif plotType=='surface':
                #axarr[subp_ind].plot(logR, slogR,  label=model,linestyle=linestyle_models[model], color=colors_models[model], linewidth=linewidth)
                if dNdD: axarr[subp_ind].set_ylabel(r'dS/dDp, um$^2$/cm$^3$')
                else: axarr[subp_ind].set_ylabel(r'dS/dlogDp, um$^2$/cm$^3$')
                if case==ctrl_name:# 'CTRL':
                    axarr[subp_ind].legend(loc=2, fontsize=SMALL_SIZE)

            elif plotType=='volume':
               #axarr[subp_ind].plot(logR, vlogR,  label=model,linestyle=linestyle_models[model], color=colors_models[model], linewidth=linewidth)
                if dNdD: axarr[subp_ind].set_ylabel(r'dV/dDp, um$^3$/cm$^3$')
                else: axarr[subp_ind].set_ylabel(r'dV/dlogDp, um$^3$/cm$^3$')
                if case== ctrl_name:#'CTRL':
                    axarr[subp_ind].legend(loc=2, fontsize=SMALL_SIZE)
            if y_log: axarr[subp_ind].set_yscale('symlog', linthreshy=linthresh[plotType])



            if not lin_x: axarr[subp_ind].set_xscale('log')
        # set symlog so can use logarithmic scale with negative values (start at 10^-4)

            if (case== ctrl_name): title_txt=case
            else: title_txt=case+ ' - %s'%ctrl_name#CTRL'
            axarr[subp_ind].set_title(title_txt, fontsize=MEDIUM_SIZE)
            axarr[subp_ind].set_xlabel('Diameter [nm]', fontsize=MEDIUM_SIZE)
            if (case !=ctrl_name):#'CTRL'):
                ylim=axarr[subp_ind].get_ylim()
                dummy=max(abs(ylim[0]),abs(ylim[1]))
                my_ylim=max(dummy,my_ylim)


        ii=0
        for case, ax in zip(cases, axarr.flatten()):
            ax.tick_params(axis='y',which='both',bottom='on')
            ax.grid(True,which="both",ls="-",axis='both')
            ax.grid(True,which='minor', linestyle=':',axis='y')#, linewidth='0.5', color='black')
            if not y_log:
                ax.ticklabel_format(axis='y',style='sci', scilimits=(0,4))
                ax.yaxis.offsetText.set_fontsize(MEDIUM_SIZE)
            ii+=1
            ax.set_xlim(xlim)
            if (case !=ctrl_name):#'CTRL'):
                if plotType=='number':
                    if (my_ylim<1e2): my_ylim = ylim_min[plotType]
                elif plotType=='surface':
                    if (my_ylim<1e-2): my_ylim=ylim_min[plotType]
                    print(model)
                    print(my_ylim)
                ax.set_ylim([-my_ylim,my_ylim])



        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=3.0)

        #figure name:
        startyear=nested_means[model][case].attrs['startyear']
        endyear=nested_means[model][case].attrs['endyear']
        if plotType=='number': plotname='sizedistrib'
        elif plotType=='surface': plotname='surfdist'
        elif plotType=='volume': plotname='volumedist'

        figname=plotpath+'/'+area+'/%s_%s_%.0f_%.0f' %(plotname,area,startyear,endyear)
        if avg_over_lev: figname=figname + '_2lev%.0f'%(pmin)
        else: figname=figname + '_atlev%.0f'%(p_level)
        if pressure_adjust: figname=figname + '_pressure_adj'
        if not y_log: figname=figname + '_linearY'
        if lin_x: figname=figname + '_linearX'
        if dNdD: figname=figname + '_dNdD'
        figname=figname+'_'+model+'_stack'
        if (plot_mode=='presentation'): figname=figname+'presm'
        figname=figname+'.png'


        print(figname)
        plt.savefig(figname)
        plt.show()
    return


def get_sizedist_list(dNdD, logR, model, plotType, plt_ds):
    # Names of variables:
    xlogR_ds=pd.DataFrame(index=logR)
    nlogR = np.zeros(logR.shape)
    slogR = np.zeros(logR.shape)
    vlogR = np.zeros(logR.shape)
    #############################################
    ##    NorESM:
    #############################################
    if (model == 'NorESM'):
        xlogR_l = np.zeros([14, len(logR)])
        i = 0

        for i in np.arange(15):
            varN = 'NCONC%02.0f' % (i)
            varSIG = 'SIGMA%02.0f' % (i)
            varNMR = 'NMR%02.0f' % (i)
            NCONC = plt_ds[varN].values  # *10**(-6) #m-3 --> cm-3
            SIGMA = plt_ds[varSIG].values  # case[varSIG][lev]#*10**6
            NMR = plt_ds[varNMR].values * 2  # *1e9 #case[varNMR][lev]*2  #  radius --> diameter
            if varN=='NCONC00':# and len(SIGMA.shape>)
                SIGMA = np.mean(SIGMA)
                NMR = np.mean(NMR)
            if ((i ) not in [3, 11, 13]):  # 3,11 and 13 no longer in use.

                if plotType == 'number':
                    xlogR_ds[mode_names_dic[model][varN]] = NCONC / (np.log(SIGMA) * np.sqrt(2 * np.pi)) * np.exp(
                    #a = NCONC / (np.log(SIGMA) * np.sqrt(2 * np.pi)) * np.exp(
                        -(np.log(logR) - np.log(NMR)) ** 2 / (2 * np.log(SIGMA) ** 2))
                    #xlogR_ds[mode_names_dic[model][varN]] = a

                elif plotType == 'surface':
                    xlogR_ds[mode_names_dic[model][varN]] = 1e-9 * 4. * np.pi * logR ** 2 * (
                                NCONC / (np.log(SIGMA) * np.sqrt(2 * np.pi)) * np.exp(
                            -(np.log(logR) - np.log(NMR)) ** 2 / (2 * np.log(SIGMA) ** 2)))

                elif plotType == 'volume':
                    xlogR_ds[mode_names_dic[model][varN]] = 1e-9 * 1 / 6. * np.pi * logR ** 3 * (
                                NCONC / (np.log(SIGMA) * np.sqrt(2 * np.pi)) * np.exp(
                            -(np.log(logR) - np.log(NMR)) ** 2 / (2 * np.log(SIGMA) ** 2)))
    #############################################
    ##    EC-Earth & EC-Earth
    #############################################
    if (model in ['EC-Earth', 'ECHAM']):
        nrModes = len(N2radi_EC_Earth.keys())
        nlogR = np.zeros(logR.shape)
        slogR = np.zeros(logR.shape)
        xlogR_l = np.zeros([len(vars_N_EC_Earth), len(logR)])
        # slogR_l=np.zeros([nrModes,logR.shape[0]])
        if model == 'EC-Earth':
            radi_list = N2radi_EC_Earth
            sigmaList = N2sigma_EC_Earth
        elif model == 'ECHAM':
            radi_list = N2radi_ECHAM
            sigmaList = N2sigma_ECHAM
        i = 0
        for varN in radi_list:

            varSIG = sigmaList[varN]  # 'SIGMA%02.0f'%(i+1)
            varNMR = radi_list[varN]  # 'NMR%02.0f'%(i+1)
            NCONC = plt_ds[varN].values  # *10**(-6) #m-3 --> cm-3
            SIGMA = sigmaList[varN]  # plt_ds[varSIG].values#case[varSIG][lev]#*10**6
            NMR = plt_ds[varNMR].values * 2  # *1e9 #case[varNMR][lev]*2  #  radius --> diameter
            if plotType == 'number':
                xlogR_ds[mode_names_dic[model][varN]] = NCONC / (np.log(SIGMA) * np.sqrt(2 * np.pi)) * np.exp(
                    -(np.log(logR) - np.log(NMR)) ** 2 / (2 * np.log(SIGMA) ** 2))
                xlogR_ds[mode_names_dic[model][varN]]
            elif plotType == 'surface':
                xlogR_ds[mode_names_dic[model][varN]] = 1e-9 * 4. * np.pi * logR ** 2 * (NCONC / (np.log(SIGMA) * np.sqrt(2 * np.pi)) * np.exp(
                    -(np.log(logR) - np.log(NMR)) ** 2 / (2 * np.log(SIGMA) ** 2)))
            elif plotType == 'volume':
                xlogR_ds[mode_names_dic[model][varN]] = 1e-9 * 1 / 6. * np.pi * logR ** 3 * (
                            NCONC / (np.log(SIGMA) * np.sqrt(2 * np.pi)) * np.exp(
                        -(np.log(logR) - np.log(NMR)) ** 2 / (2 * np.log(SIGMA) ** 2)))

            i += 1
    if dNdD:
        for key in xlogR_ds:
            xlogR_ds[key] = xlogR_ds[key]/logR
    return xlogR_ds


def plot_sizedist_stacked(nested_means,plotpath, area,avg_over_lev, pmin, p_level,pressure_adjust, plotType='number',
                          y_log=True, lin_x=False, dNdD=False , plot_mode='normal', ctrl_name='CTRL',
                          ylim_min = {'number':1e2, 'surface': 1e-2,'volume' : 1e-1},
                          linthresh = {'number':3e1, 'surface': 1e-3,'volume' : 1e-3}, sectional = False):
    [SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE, EVEN_BIGGER, colors_models, linestyle_models, linewidth, figsize] = set_plot_vars(plot_mode)
    #if (plot_mode=='presentation'):
    #    set_presentation_mode()
    #sns.set()
    try:
        if not os.path.exists(plotpath):
            os.makedirs(plotpath)
            os.makedirs(plotpath+'/'+area)
        elif not os.path.exists(plotpath+'/'+area):
            os.makedirs(plotpath+'/'+area)
    except OSError:
        print ('Error: Creating directory. ' )


    # Make plot array:
    if lin_x:logR=np.linspace(10**(-3),10**3,300)
    else: logR=np.logspace(0,4,100)#*10**(-6)  # should be nmr

    ds_plot=xr.Dataset(coords={'Radius':logR})


    ii=0
    ctrl_dic={}
    ctrl_dic_l={}
    models=list(nested_means.keys())
    cases=list(nested_means[models[0]].keys())

    for model in models:

        fig, axarr = plt.subplots(2,3, figsize=figsize)
        ii=0
        my_ylim=0 # to set ylim equal in all cases
        for case in cases:

            subp_ind=tuple(np.array([int(np.floor((ii)/3)),int(ii%3)]).astype(int))

            ii=ii+1


            #print(model)
            #if (case=='CTRL'):
            plt_ds=nested_means[model][case]
            # Names of variables:
            nlogR=np.zeros(logR.shape)
            slogR=np.zeros(logR.shape)
            vlogR=np.zeros(logR.shape)
            #############################################
            ##    NorESM:
            #############################################
            if (model=='NorESM'):
                xlogR_l=np.zeros([14, len(logR)])
                i=0

                for i in np.arange(14):
                    varN='NCONC%02.0f'%(i+1)
                    varSIG = 'SIGMA%02.0f'%(i+1)
                    varNMR = 'NMR%02.0f'%(i+1)
                    NCONC = plt_ds[varN].values#*10**(-6) #m-3 --> cm-3
                    SIGMA = plt_ds[varSIG].values#case[varSIG][lev]#*10**6
                    NMR = plt_ds[varNMR].values*2#*1e9 #case[varNMR][lev]*2  #  radius --> diameter
                    if ((i+1) not in [3,11,13]): #3,11 and 13 no longer in use.

                        nlogR += NCONC/(np.log(SIGMA)*np.sqrt(2*np.pi))*np.exp(-(np.log(logR)-np.log(NMR))**2/(2*np.log(SIGMA)**2))
                        slogR += 1e-9*4.*np.pi*logR**2*(NCONC/(np.log(SIGMA)*np.sqrt(2*np.pi))*np.exp(-(np.log(logR)-np.log(NMR))**2/(2*np.log(SIGMA)**2)))
                        vlogR += 1e-9*1/6.*np.pi*logR**3*(NCONC/(np.log(SIGMA)*np.sqrt(2*np.pi))*np.exp(-(np.log(logR)-np.log(NMR))**2/(2*np.log(SIGMA)**2)))
                        if plotType == 'number':
                            xlogR_l[i,:] =NCONC/(np.log(SIGMA)*np.sqrt(2*np.pi))*np.exp(-(np.log(logR)-np.log(NMR))**2/(2*np.log(SIGMA)**2))
                        elif plotType == 'surface':
                            xlogR_l[i,:] = 1e-9*4.*np.pi*logR**2*(NCONC/(np.log(SIGMA)*np.sqrt(2*np.pi))*np.exp(-(np.log(logR)-np.log(NMR))**2/(2*np.log(SIGMA)**2)))

                        elif plotType == 'volume':
                            xlogR_l[i,:] =1e-9*1/6.*np.pi*logR**3*(NCONC/(np.log(SIGMA)*np.sqrt(2*np.pi))*np.exp(-(np.log(logR)-np.log(NMR))**2/(2*np.log(SIGMA)**2)))

            #############################################
            ##    EC-Earth & EC-Earth
            #############################################
            if (model in ['EC-Earth', 'ECHAM']):
                nrModes=len(N2radi_EC_Earth.keys())
                nlogR=np.zeros(logR.shape)
                slogR=np.zeros(logR.shape)
                xlogR_l=np.zeros([len(vars_N_EC_Earth),len(logR)])
                #slogR_l=np.zeros([nrModes,logR.shape[0]])
                if model=='EC-Earth':
                    radi_list = N2radi_EC_Earth
                    sigmaList=N2sigma_EC_Earth
                elif model =='ECHAM':
                    radi_list = N2radi_ECHAM
                    sigmaList=N2sigma_ECHAM
                i=0
                for varN in radi_list:

                    varSIG = sigmaList[varN]#'SIGMA%02.0f'%(i+1)
                    varNMR = radi_list[varN]#'NMR%02.0f'%(i+1)
                    NCONC = plt_ds[varN].values#*10**(-6) #m-3 --> cm-3
                    SIGMA =  sigmaList[varN]#plt_ds[varSIG].values#case[varSIG][lev]#*10**6
                    NMR = plt_ds[varNMR].values*2#*1e9 #case[varNMR][lev]*2  #  radius --> diameter
                    nlogR += NCONC/(np.log(SIGMA)*np.sqrt(2*np.pi))*np.exp(-(np.log(logR)-np.log(NMR))**2/(2*np.log(SIGMA)**2))
                    slogR += 1e-9*4.*np.pi*logR**2*(NCONC/(np.log(SIGMA)*np.sqrt(2*np.pi))*np.exp(-(np.log(logR)-np.log(NMR))**2/(2*np.log(SIGMA)**2)))
                    vlogR += 1e-9*1/6.*np.pi*logR**3*(NCONC/(np.log(SIGMA)*np.sqrt(2*np.pi))*np.exp(-(np.log(logR)-np.log(NMR))**2/(2*np.log(SIGMA)**2)))

                    if plotType=='number':
                        xlogR_l[i,:] =NCONC/(np.log(SIGMA)*np.sqrt(2*np.pi))*np.exp(-(np.log(logR)-np.log(NMR))**2/(2*np.log(SIGMA)**2))
                    elif plotType == 'surface':
                        xlogR_l[i,:] = 1e-9*4.*np.pi*logR**2*(NCONC/(np.log(SIGMA)*np.sqrt(2*np.pi))*np.exp(-(np.log(logR)-np.log(NMR))**2/(2*np.log(SIGMA)**2)))
                    elif plotType == 'volume':
                        xlogR_l[i,:] = 1e-9*1/6.*np.pi*logR**3*(NCONC/(np.log(SIGMA)*np.sqrt(2*np.pi))*np.exp(-(np.log(logR)-np.log(NMR))**2/(2*np.log(SIGMA)**2)))

                    i+=1

            if dNdD:
                nlogR=nlogR/logR
                slogR=slogR/logR
                vlogR=vlogR/logR
            if (case==ctrl_name):

                #ctrl_dic[model]=nlogR.copy()
                ctrl_dic_l[model]=xlogR_l.copy()
                #elif plotType=='surface': ctrl_dic[model]=slogR.copy()
                #elif plotType=='volume': ctrl_dic[model]=vlogR.copy()
            else:
                #nlogR=nlogR-ctrl_dic[model]
                #slogR=slogR-ctrl_dic[model]
                #vlogR=vlogR-ctrl_dic[model]
                xlogR_l=xlogR_l-ctrl_dic_l[model]

            labels=mode_names[model]
            col = sns.color_palette("Paired", len(labels))
            axarr[subp_ind].stackplot(logR,xlogR_l, labels=labels,alpha=.8,colors=col)
            if plotType=='number':

                if dNdD:axarr[subp_ind].set_ylabel('dN/dDp, #/cm$^3$')
                else: axarr[subp_ind].set_ylabel('dN/dlogDp, #/cm$^3$')
                #axarr[subp_ind].legend(fontsize=10,loc=1)
                if case== ctrl_name:# 'CTRL':
                    axarr[subp_ind].legend(loc=4, fontsize=SMALL_SIZE)


            elif plotType=='surface':
                #axarr[subp_ind].plot(logR, slogR,  label=model,linestyle=linestyle_models[model], color=colors_models[model], linewidth=linewidth)
                if dNdD: axarr[subp_ind].set_ylabel(r'dS/dDp, um$^2$/cm$^3$')
                else: axarr[subp_ind].set_ylabel(r'dS/dlogDp, um$^2$/cm$^3$')
                if case==ctrl_name:# 'CTRL':
                    axarr[subp_ind].legend(loc=2, fontsize=SMALL_SIZE)

            elif plotType=='volume':
                #axarr[subp_ind].plot(logR, vlogR,  label=model,linestyle=linestyle_models[model], color=colors_models[model], linewidth=linewidth)
                if dNdD: axarr[subp_ind].set_ylabel(r'dV/dDp, um$^3$/cm$^3$')
                else: axarr[subp_ind].set_ylabel(r'dV/dlogDp, um$^3$/cm$^3$')
                if case== ctrl_name:#'CTRL':
                    axarr[subp_ind].legend(loc=2, fontsize=SMALL_SIZE)
            if y_log: axarr[subp_ind].set_yscale('symlog', linthreshy=linthresh[plotType])



            if not lin_x: axarr[subp_ind].set_xscale('log')
        # set symlog so can use logarithmic scale with negative values (start at 10^-4)

            if (case== ctrl_name): title_txt=case
            else: title_txt=case+ ' - %s'%ctrl_name#CTRL'
            axarr[subp_ind].set_title(title_txt, fontsize=MEDIUM_SIZE)
            axarr[subp_ind].set_xlabel('Diameter [nm]', fontsize=MEDIUM_SIZE)
            if (case !=ctrl_name):#'CTRL'):
                ylim=axarr[subp_ind].get_ylim()
                dummy=max(abs(ylim[0]),abs(ylim[1]))
                my_ylim=max(dummy,my_ylim)


        ii=0
        for case in cases:
            subp_ind=tuple(np.array([int(np.floor((ii)/3)),int(ii%3)]).astype(int))
            axarr[subp_ind].tick_params(axis='y',which='both',bottom='on')
            axarr[subp_ind].grid(True,which="both",ls="-",axis='both')
            axarr[subp_ind].grid(True,which='minor', linestyle=':',axis='y')#, linewidth='0.5', color='black')
            if not y_log:
                axarr[subp_ind].ticklabel_format(axis='y',style='sci', scilimits=(0,4))
                axarr[subp_ind].yaxis.offsetText.set_fontsize(MEDIUM_SIZE)
            ii+=1
            if (case !=ctrl_name):#'CTRL'):
                if plotType=='number':
                    if (my_ylim<1e2): my_ylim = ylim_min[plotType]
                elif plotType=='surface':
                    if (my_ylim<1e-2): my_ylim=ylim_min[plotType]
                    print(model)
                    print(my_ylim)
                axarr[subp_ind].set_ylim([-my_ylim,my_ylim])



        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=3.0)

        #figure name:
        startyear=nested_means[model][case].attrs['startyear']
        endyear=nested_means[model][case].attrs['endyear']
        if plotType=='number': plotname='sizedistrib'
        elif plotType=='surface': plotname='surfdist'
        elif plotType=='volume': plotname='volumedist'

        figname=plotpath+'/'+area+'/%s_%s_%.0f_%.0f' %(plotname,area,startyear,endyear)
        if avg_over_lev: figname=figname + '_2lev%.0f'%(pmin)
        else: figname=figname + '_atlev%.0f'%(p_level)
        if pressure_adjust: figname=figname + '_pressure_adj'
        if not y_log: figname=figname + '_linearY'
        if lin_x: figname=figname + '_linearX'
        if dNdD: figname=figname + '_dNdD'
        figname=figname+'_'+model+'_stack'
        if (plot_mode=='presentation'): figname=figname+'presm'
        figname=figname+'.png'


        print(figname)
        plt.savefig(figname)
        plt.show()
    return


######################################################################################################

def plot_sizedist_ctrl(nested_means, plotpath,
                       area, avg_over_lev, pmin, p_level, pressure_adjust,
                       plotType='number', y_log=True, lin_x=False, dNdD=False,
                       plot_mode='normal', ctrl_name = 'CTRL',
                       xlim=[3,1e3], ylim=[10,4e4], figsize=[5,4], per_micron=False):
    # nested_means should be nested dir with nested[model][case] containing xarray objects
    # plotType='number'/'surface'/'volume'
    # y_log=True decides if y-axis should be log
    # lin_x=False decides if linear x axis
    # dNdD=False if true prints dN/dD instead of dN/dlogD
    [SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE, EVEN_BIGGER, colors_models, linestyle_models, linewidth, figsize2] = set_plot_vars(plot_mode)
    #if (plot_mode=='presentation'):
    #    set_presentation_mode()
    #sns.set(palette='bright')
    if per_micron and not dNdD:
        sys.exit('Cannot have per_micron True and not dNdD True')


    if lin_x:logR=np.linspace(10**(-3),10**3,300)
    else: logR=np.logspace(0,4,300)#*10**(-6)  # should be nmr

    ds_plot=xr.Dataset(coords={'Radius':logR})

    #for var in varList:
    fig, axarr = plt.subplots(1, figsize=figsize)
    ii=0
    ctrl_dic={} # keep ctr to substract from others
    my_ylim=0   # keep ylimits to use for all plots
    models=list(nested_means.keys())
    #cases=list(nested_means[models[0]].keys())
    for case in [ctrl_name]:#['CTRL']:

        ii=ii+1

        for model in models:
            #print(model)
            #if (case=='CTRL'):
            plt_ds=nested_means[model][case]
            # Names of variables:
            nlogR=np.zeros(logR.shape)
            slogR=np.zeros(logR.shape)
            vlogR=np.zeros(logR.shape)

            #############################################
            ##    NorESM
            #############################################

            if (model=='NorESM'):
                for i in np.arange(15):
                    varN='NCONC%02.0f'%(i)
                    varSIG = 'SIGMA%02.0f'%(i)
                    varNMR = 'NMR%02.0f'%(i)
                    NCONC = plt_ds[varN].values#*10**(-6) #m-3 --> cm-3
                    SIGMA = plt_ds[varSIG].values#case[varSIG][lev]#*10**6
                    NMR = plt_ds[varNMR].values*2#*1e9 #case[varNMR][lev]*2  #  radius --> diameter
                    if ((i) not in [3,11,13]): #3,11 and 13 no longer in use.
                        #slogR += NCONC/(np.log(SIGMA)*np.sqrt(2*np.pi))*np.exp(-(np.log(logR)-np.log(NMR))**2/(2*np.log(SIGMA)**2))*0 #OBS: no meaning
                        nlogR += NCONC/(np.log(SIGMA)*np.sqrt(2*np.pi))*np.exp(-(np.log(logR)-np.log(NMR))**2/(2*np.log(SIGMA)**2))
                        slogR += 1e-9*4.*np.pi*logR**2*(NCONC/(np.log(SIGMA)*np.sqrt(2*np.pi))*np.exp(-(np.log(logR)-np.log(NMR))**2/(2*np.log(SIGMA)**2)))
                        vlogR += 1e-9*1/6.*np.pi*logR**3*(NCONC/(np.log(SIGMA)*np.sqrt(2*np.pi))*np.exp(-(np.log(logR)-np.log(NMR))**2/(2*np.log(SIGMA)**2)))

            #############################################
            ##    EC-Earth & EC-Earth
            #############################################
            if (model in ['EC-Earth', 'ECHAM']):
                nrModes=len(N2radi_EC_Earth.keys())
                nlogR=np.zeros(logR.shape)
                slogR=np.zeros(logR.shape)

                if model=='EC-Earth':
                    radi_list = N2radi_EC_Earth
                    sigmaList=N2sigma_EC_Earth
                elif model =='ECHAM':
                    radi_list = N2radi_ECHAM
                    sigmaList=N2sigma_ECHAM
                for varN in radi_list:
                    varSIG = sigmaList[varN]#'SIGMA%02.0f'%(i+1)
                    varNMR = radi_list[varN]#'NMR%02.0f'%(i+1)
                    NCONC = plt_ds[varN].values#*10**(-6) #m-3 --> cm-3
                    SIGMA =  sigmaList[varN]#plt_ds[varSIG].values#case[varSIG][lev]#*10**6
                    NMR = plt_ds[varNMR].values*2#*1e9 #case[varNMR][lev]*2  #  radius --> diameter

                    nlogR += NCONC/(np.log(SIGMA)*np.sqrt(2*np.pi))*np.exp(-(np.log(logR)-np.log(NMR))**2/(2*np.log(SIGMA)**2))
                    slogR += 1e-9*4.*np.pi*logR**2*(NCONC/(np.log(SIGMA)*np.sqrt(2*np.pi))*np.exp(-(np.log(logR)-np.log(NMR))**2/(2*np.log(SIGMA)**2)))
                    vlogR += 1e-9*1/6.*np.pi*logR**3*(NCONC/(np.log(SIGMA)*np.sqrt(2*np.pi))*np.exp(-(np.log(logR)-np.log(NMR))**2/(2*np.log(SIGMA)**2)))
            plt_logR = logR.copy()
            if dNdD:
                nlogR=nlogR/logR
                slogR=slogR/logR
                vlogR=vlogR/logR
                if per_micron:
                    nlogR = 1e3*nlogR
                    slogR = 1e3*slogR
                    vlogR = 1e3*vlogR
                    #for xlogR in [nlogR, slogR, vlogR]:
                    #    print('HEY')
                    #    xlogR = 1e3*xlogR #per nm --> per um
                    plt_logR = 1e-3*logR # nm --> um

            if (case== ctrl_name):#'CTRL'):
                if plotType=='number': ctrl_dic[model]=nlogR.copy()
                elif plotType=='surface': ctrl_dic[model]=slogR.copy()
                elif plotType=='volume': ctrl_dic[model]=vlogR.copy()
            else:
                nlogR=nlogR-ctrl_dic[model]
                slogR=slogR-ctrl_dic[model]
                vlogR=vlogR-ctrl_dic[model]

            if plotType=='number':
                if y_log and not(lin_x):
                    axarr.loglog(plt_logR, nlogR, label=model, linestyle=linestyle_models[model], color=colors_models[model], linewidth=linewidth)
                else:
                    axarr.plot(plt_logR, nlogR, label=model, linestyle=linestyle_models[model], color=colors_models[model], linewidth=linewidth)
                if dNdD:
                    if per_micron: axarr.set_ylabel('dN/dDp, $\mu$m$^{-1}$#/cm$^3$')#, fontsize=MEDIUM_SIZE)
                    else: axarr.set_ylabel('dN/dDp, nm$^{-1}$#/cm$^3$')#, fontsize=MEDIUM_SIZE)
                else: axarr.set_ylabel('dN/dlogDp, #/cm$^3$')#, fontsize=MEDIUM_SIZE)
                if y_log: axarr.set_yscale('log')
                if case== ctrl_name:#'CTRL':
                    axarr.legend(fontsize=SMALL_SIZE, loc=1)

            elif plotType=='surface':
                axarr.plot(plt_logR, slogR, label=model, linestyle=linestyle_models[model], color=colors_models[model], linewidth=linewidth)
                if dNdD:
                    if per_micron: axarr.set_ylabel(r'dS/dDp, $\mu$m$^{-1}$ $\mu$m$^2$/cm$^3$')#, fontsize=MEDIUM_SIZE)
                    else: axarr.set_ylabel('dN/dDp, nm$^{-1}$$\mu$m$^2$/cm$^3$')#, fontsize=MEDIUM_SIZE)
                else: axarr.set_ylabel(r'dS/dlogDp, um$^2$/cm$^3$', fontsize=MEDIUM_SIZE)
                if y_log: axarr.set_yscale('log')
                if case== ctrl_name:#'CTRL':
                    axarr.legend(fontsize=SMALL_SIZE, loc=2)

            elif plotType=='volume':
                axarr.plot(plt_logR, vlogR, label=model, linestyle=linestyle_models[model], color=colors_models[model], linewidth=linewidth)
                if dNdD:
                    if per_micron: axarr.set_ylabel(r'dS/dDp, $\mu$m$^{-1}$ $\mu$m$^3$/cm$^3$')#, fontsize=MEDIUM_SIZE)
                    else: axarr.set_ylabel('dN/dDp, nm$^{-1}$$\mu$m$^3$/cm$^3$')#, fontsize=MEDIUM_SIZE)
                else: axarr.set_ylabel(r'dV/dlogDp, um$^3$/cm$^3$', fontsize=MEDIUM_SIZE)
                if y_log: axarr.set_yscale('log', linthreshy=.01)
                if case== ctrl_name:#'CTRL':
                    axarr.legend(fontsize=SMALL_SIZE, loc=2)


            if not lin_x: axarr.set_xscale('log')

            if (case==ctrl_name): title_txt=case
            else: title_txt=case+ ' - %s'%ctrl_name#CTRL'
            axarr.set_title(title_txt)#, fontsize=BIGGER_SIZE)
            if per_micron:
                axarr.set_xlabel('Diameter [$\mu$m]')#, fontsize=MEDIUM_SIZE)
            else:
                axarr.set_xlabel('Diameter [nm]')#, fontsize=MEDIUM_SIZE)
            #axarr.tick_params(labelsize=SMALL_SIZE)


    ii=0
    axarr.grid(b=True,which="both",color='k',alpha=0.1, ls="-",axis='both')
    #axarr.tick_params(axis='both', which='both', colors='g')#, visible=True)

    #plt.setp(axarr.get_xticklabels(), visible=True)

    if not y_log:
        axarr[subp_ind].ticklabel_format(axis='y',style='sci', scilimits=(0,4))
        axarr[subp_ind].yaxis.offsetText.set_fontsize(MEDIUM_SIZE)
    axarr.set_xlim(xlim)
    axarr.set_ylim(ylim)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=3.0)
    startyear=nested_means[model][case].attrs['startyear']
    endyear=nested_means[model][case].attrs['endyear']
    if plotType=='number': plotname='sizedistrib'
    elif plotType=='surface': plotname='surfdist'
    elif plotType=='volume': plotname='volumedist'

    figname=plotpath+'/CTRL_ONLY'+area+'/%s_%s_%s_%.0f_%.0f' %(ctrl_name, plotname,area,startyear,endyear)
    if avg_over_lev: figname=figname + '_2lev%.0f'%(pmin)
    else: figname=figname + '_atlev%.0f'%(p_level)
    if pressure_adjust: figname=figname + '_pressure_adj'
    if not y_log: figname=figname + '_linearY'
    if lin_x: figname=figname + '_linearX'
    if dNdD: figname=figname + '_dNdD'
    if (plot_mode=='presentation'): figname=figname+'presm'
    figname=figname+'.png'
    make_folders(figname)
    print(figname)
    plt.savefig(figname, dpi=300)
    plt.show()
    return






def plot_sizedist_ctr_diff(nested_means,plotpath, area,avg_over_lev, pmin,
                           p_level,pressure_adjust, plotType='number',
                           y_log=True, lin_x=False, dNdD=False, ctrl_name='CTRL',
                           plot_mode='normal', ylim_diff='',
                           figsize=[10,5], xlim=[1,1e4], ylim_ctr=[1,1e4],
                           plt_ctr=True):
    # nested_means should be nested dir with nested[model][case] containing xarray objects
    # plotType='number'/'surface'/'volume'
    # y_log=True decides if y-axis should be log
    # lin_x=False decides if linear x axis
    # dNdD=False if true prints dN/dD instead of dN/dlogD
    [SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE, EVEN_BIGGER, colors_models, linestyle_models, linewidth, figsize_d] = set_plot_vars(plot_mode)
    #if (plot_mode=='presentation'):
    #    set_presentation_mode()
    #sns.set(palette='bright')
    plotpath=plotpath+'/2cases/'
    try:
        make_folders(plotpath)
    except OSError:
        print ('Error: Creating directory. ' )
    # Make plot array:
    if lin_x:logR=np.linspace(10**(-3),10**3,300)
    else: logR=np.logspace(0,4,300)#*10**(-6)  # should be nmr

    ds_plot=xr.Dataset(coords={'Radius':logR})

    #for var in varList:
    #fig.set_size_inches(13,10)#,foreward=True)
    ii=0
    ctrl_dic={} # keep ctr to substract from others
    my_ylim=0   # keep ylimits to use for all plots
    models=list(nested_means.keys())
    cases=list(nested_means[models[0]].keys())
    if plt_ctr:
        fig, axarr = plt.subplots(1,len(cases), figsize=figsize)
    else:
        fig, axarr = plt.subplots(1,1, figsize=figsize)
    for case in cases:
        subp_ind=ii#tuple(np.array([int(np.floor((ii)/3)),int(ii%3)]).astype(int))
        if plt_ctr and len(cases)>1:
            ax_plt=axarr[ii]
        else:
            ax_plt=axarr

        ii=ii+1

        for model in models:
            #print(model)
            #if (case=='CTRL'):
            plt_ds=nested_means[model][case]
            # Names of variables:
            nlogR=np.zeros(logR.shape)
            slogR=np.zeros(logR.shape)
            vlogR=np.zeros(logR.shape)

            #############################################
            ##    NorESM
            #############################################

            if (model=='NorESM'):
                for i in np.arange(15):
                    varN='NCONC%02.0f'%(i)
                    varSIG = 'SIGMA%02.0f'%(i)
                    varNMR = 'NMR%02.0f'%(i)
                    NCONC = plt_ds[varN].values#*10**(-6) #m-3 --> cm-3
                    SIGMA = plt_ds[varSIG].values#case[varSIG][lev]#*10**6
                    NMR = plt_ds[varNMR].values*2#*1e9 #case[varNMR][lev]*2  #  radius --> diameter
                    if ((i) not in [3,11,13]): #3,11 and 13 no longer in use.
                        #slogR += NCONC/(np.log(SIGMA)*np.sqrt(2*np.pi))*np.exp(-(np.log(logR)-np.log(NMR))**2/(2*np.log(SIGMA)**2))*0 #OBS: no meaning
                        nlogR += NCONC/(np.log(SIGMA)*np.sqrt(2*np.pi))*np.exp(-(np.log(logR)-np.log(NMR))**2/(2*np.log(SIGMA)**2))
                        slogR += 1e-9*4.*np.pi*logR**2*(NCONC/(np.log(SIGMA)*np.sqrt(2*np.pi))*np.exp(-(np.log(logR)-np.log(NMR))**2/(2*np.log(SIGMA)**2)))
                        vlogR += 1e-9*1/6.*np.pi*logR**3*(NCONC/(np.log(SIGMA)*np.sqrt(2*np.pi))*np.exp(-(np.log(logR)-np.log(NMR))**2/(2*np.log(SIGMA)**2)))

            #############################################
            ##    EC-Earth & EC-Earth
            #############################################
            if (model in ['EC-Earth', 'ECHAM']):
                nrModes=len(N2radi_EC_Earth.keys())
                nlogR=np.zeros(logR.shape)
                slogR=np.zeros(logR.shape)

                if model=='EC-Earth':
                    radi_list = N2radi_EC_Earth
                    sigmaList=N2sigma_EC_Earth
                elif model =='ECHAM':
                    radi_list = N2radi_ECHAM
                    sigmaList=N2sigma_ECHAM
                for varN in radi_list:
                    varSIG = sigmaList[varN]#'SIGMA%02.0f'%(i+1)
                    varNMR = radi_list[varN]#'NMR%02.0f'%(i+1)
                    NCONC = plt_ds[varN].values#*10**(-6) #m-3 --> cm-3
                    SIGMA =  sigmaList[varN]#plt_ds[varSIG].values#case[varSIG][lev]#*10**6
                    NMR = plt_ds[varNMR].values*2#*1e9 #case[varNMR][lev]*2  #  radius --> diameter

                    nlogR += NCONC/(np.log(SIGMA)*np.sqrt(2*np.pi))*np.exp(-(np.log(logR)-np.log(NMR))**2/(2*np.log(SIGMA)**2))
                    slogR += 1e-9*4.*np.pi*logR**2*(NCONC/(np.log(SIGMA)*np.sqrt(2*np.pi))*np.exp(-(np.log(logR)-np.log(NMR))**2/(2*np.log(SIGMA)**2)))
                    vlogR += 1e-9*1/6.*np.pi*logR**3*(NCONC/(np.log(SIGMA)*np.sqrt(2*np.pi))*np.exp(-(np.log(logR)-np.log(NMR))**2/(2*np.log(SIGMA)**2)))

            if dNdD:
                nlogR=nlogR/logR
                slogR=slogR/logR
                vlogR=vlogR/logR
            if (case== ctrl_name):#'CTRL'):

                if plotType=='number': ctrl_dic[model]=nlogR.copy()
                elif plotType=='surface': ctrl_dic[model]=slogR.copy()
                elif plotType=='volume': ctrl_dic[model]=vlogR.copy()
            else:
                nlogR=nlogR-ctrl_dic[model]
                slogR=slogR-ctrl_dic[model]
                vlogR=vlogR-ctrl_dic[model]

            if case==ctrl_name and not plt_ctr: continue
            if plotType=='number':
                ax_plt.plot(logR, nlogR, label=model, linestyle=linestyle_models[model], color=colors_models[model], linewidth=linewidth)

                if dNdD:ax_plt.set_ylabel('dN/dDp, #/cm$^3$', fontsize=MEDIUM_SIZE)
                else: ax_plt.set_ylabel('dN/dlogDp, #/cm$^3$', fontsize=MEDIUM_SIZE)

                if y_log:
                    if case== ctrl_name:#'CTRL':
                        ax_plt.set_yscale('log')#, linthreshy=30)
                    else:
                        ax_plt.set_yscale('symlog', linthreshy=30)
                #if case=='CTRL':
                ax_plt.legend(fontsize=SMALL_SIZE, loc=1)

            elif plotType=='surface':
                ax_plt.plot(logR, slogR, label=model, linestyle=linestyle_models[model], color=colors_models[model], linewidth=linewidth)
                if dNdD: ax_plt.set_ylabel(r'dS/dDp, um$^2$/cm$^3$', fontsize=MEDIUM_SIZE)
                else: ax_plt.set_ylabel(r'dS/dlogDp, um$^2$/cm$^3$', fontsize=MEDIUM_SIZE)
                if y_log: ax_plt.set_yscale('symlog', linthreshy=.003)
                if case== ctrl_name:#'CTRL':
                    ax_plt.legend(fontsize=SMALL_SIZE, loc=2)

            elif plotType=='volume':
                ax_plt.plot(logR, vlogR, label=model, linestyle=linestyle_models[model], color=colors_models[model], linewidth=linewidth)
                if dNdD: ax_plt.set_ylabel(r'dV/dDp, um$^3$/cm$^3$', fontsize=MEDIUM_SIZE)
                else: ax_plt.set_ylabel(r'dV/dlogDp, um$^3$/cm$^3$', fontsize=MEDIUM_SIZE)
                if y_log: ax_plt.set_yscale('symlog', linthreshy=.01)
                #if case=='CTRL':
                ax_plt.legend(fontsize=SMALL_SIZE, loc=2)


            if not lin_x: ax_plt.set_xscale('log')

            if (case==ctrl_name): title_txt=case
            else: title_txt=case+ ' - %s'%ctrl_name#CTRL'
            ax_plt.set_title(title_txt, fontsize=BIGGER_SIZE)
            ax_plt.set_xlabel('Diameter [nm]', fontsize=MEDIUM_SIZE)
            ax_plt.tick_params(labelsize=SMALL_SIZE)
            if (case !=ctrl_name):#'CTRL'):
                ylim = ax_plt.get_ylim()
                dummy = max(abs(ylim[0]),abs(ylim[1]))
                my_ylim = max(dummy,my_ylim)


    ii=0
    for case in cases:
        subp_ind = ii#tuple(np.array([int(np.floor((ii)/3)),int(ii%3)]).astype(int))
        if plt_ctr and len(cases)>1:
            ax_plt=axarr[ii]
        else:
            ax_plt=axarr
        ax_plt.grid(b=True,which="both",color='k',alpha=0.1, ls="-",axis='both')
        ii+=1
        ax_plt.set_xlim(xlim)
        if not y_log:
            ax_plt.ticklabel_format(axis='y',style='sci', scilimits=(0,4))
            ax_plt.yaxis.offsetText.set_fontsize(MEDIUM_SIZE)

        # set same ylim for all plots:
        if (case !=ctrl_name):#'CTRL'):
            ax_plt.set_ylim([-my_ylim,my_ylim])
        else:
            ax_plt.set_ylim(ylim_ctr)


    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=3.0)
    startyear=nested_means[model][case].attrs['startyear']
    endyear=nested_means[model][case].attrs['endyear']
    if plotType=='number': plotname='sizedistrib'
    elif plotType=='surface': plotname='surfdist'
    elif plotType=='volume': plotname='volumedist'

    figname=plotpath+'/'+area+'/%s_%s_%.0f_%.0f' %(plotname,area,startyear,endyear)
    for case in cases:
        figname=figname+case
    if avg_over_lev: figname=figname + '_2lev%.0f'%(pmin)
    else: figname=figname + '_atlev%.0f'%(p_level)
    if pressure_adjust: figname=figname + '_pressure_adj'
    if not y_log: figname=figname + '_linearY'
    if lin_x: figname=figname + '_linearX'
    if dNdD: figname=figname + '_dNdD'
    if (plot_mode=='presentation'): figname=figname+'presm'
    figname=figname+'.png'
    make_folders(figname)
    print(figname)
    plt.savefig(figname, dpi=300)
    plt.show()
    return

def plot_sizedist_ctr_diff_plot_types(nested_means,plotpath, area,avg_over_lev, pmin,
                           p_level,pressure_adjust, plotType=['number','surface','volume'],
                           y_log=True, lin_x=False, dNdD=False, ctrl_name='CTRL',
                           plot_mode='normal', ylim_diff='',
                           figsize=[10,5], xlim=[1,1e4], ylim_ctr=[1,1e4],
                           plt_ctr=True):
    # nested_means should be nested dir with nested[model][case] containing xarray objects
    # plotType='number'/'surface'/'volume'
    # y_log=True decides if y-axis should be log
    # lin_x=False decides if linear x axis
    # dNdD=False if true prints dN/dD instead of dN/dlogD
    [SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE, EVEN_BIGGER, colors_models, linestyle_models, linewidth, figsize_d] = set_plot_vars(plot_mode)
    #if (plot_mode=='presentation'):
    #    set_presentation_mode()
    #sns.set(palette='bright')
    plotpath=plotpath+'/2cases/'
    try:
        make_folders(plotpath)
    except OSError:
        print ('Error: Creating directory. ' )
    # Make plot array:
    if lin_x:logR=np.linspace(10**(-3),10**3,300)
    else: logR=np.logspace(0,4,300)#*10**(-6)  # should be nmr

    ds_plot=xr.Dataset(coords={'Radius':logR})

    #for var in varList:
    #fig.set_size_inches(13,10)#,foreward=True)
    ii=0
    ctrl_dic={} # keep ctr to substract from others
    my_ylim=0   # keep ylimits to use for all plots
    models=list(nested_means.keys())
    cases=list(nested_means[models[0]].keys())
    if plt_ctr:
        fig, axarr = plt.subplots(len(plotType),len(cases), figsize=figsize)
    else:
        fig, axarr = plt.subplots(len(plotType),1, figsize=figsize)
    for case in cases:
        subp_ind=ii#tuple(np.array([int(np.floor((ii)/3)),int(ii%3)]).astype(int))
        if plt_ctr and len(cases)>1:
            ax_plt=axarr[ii]
        else:
            ax_plt=axarr

        ii=ii+1

        for model in models:
            #print(model)
            #if (case=='CTRL'):
            plt_ds=nested_means[model][case]
            # Names of variables:
            nlogR=np.zeros(logR.shape)
            slogR=np.zeros(logR.shape)
            vlogR=np.zeros(logR.shape)

            #############################################
            ##    NorESM
            #############################################

            if (model=='NorESM'):
                for i in np.arange(15):
                    varN='NCONC%02.0f'%(i)
                    varSIG = 'SIGMA%02.0f'%(i)
                    varNMR = 'NMR%02.0f'%(i)
                    NCONC = plt_ds[varN].values#*10**(-6) #m-3 --> cm-3
                    SIGMA = plt_ds[varSIG].values#case[varSIG][lev]#*10**6
                    NMR = plt_ds[varNMR].values*2#*1e9 #case[varNMR][lev]*2  #  radius --> diameter
                    if ((i) not in [3,11,13]): #3,11 and 13 no longer in use.
                        #slogR += NCONC/(np.log(SIGMA)*np.sqrt(2*np.pi))*np.exp(-(np.log(logR)-np.log(NMR))**2/(2*np.log(SIGMA)**2))*0 #OBS: no meaning
                        nlogR += NCONC/(np.log(SIGMA)*np.sqrt(2*np.pi))*np.exp(-(np.log(logR)-np.log(NMR))**2/(2*np.log(SIGMA)**2))
                        slogR += 1e-9*4.*np.pi*logR**2*(NCONC/(np.log(SIGMA)*np.sqrt(2*np.pi))*np.exp(-(np.log(logR)-np.log(NMR))**2/(2*np.log(SIGMA)**2)))
                        vlogR += 1e-9*1/6.*np.pi*logR**3*(NCONC/(np.log(SIGMA)*np.sqrt(2*np.pi))*np.exp(-(np.log(logR)-np.log(NMR))**2/(2*np.log(SIGMA)**2)))

            #############################################
            ##    EC-Earth & EC-Earth
            #############################################
            if (model in ['EC-Earth', 'ECHAM']):
                nrModes=len(N2radi_EC_Earth.keys())
                nlogR=np.zeros(logR.shape)
                slogR=np.zeros(logR.shape)

                if model=='EC-Earth':
                    radi_list = N2radi_EC_Earth
                    sigmaList=N2sigma_EC_Earth
                elif model =='ECHAM':
                    radi_list = N2radi_ECHAM
                    sigmaList=N2sigma_ECHAM
                for varN in radi_list:
                    varSIG = sigmaList[varN]#'SIGMA%02.0f'%(i+1)
                    varNMR = radi_list[varN]#'NMR%02.0f'%(i+1)
                    NCONC = plt_ds[varN].values#*10**(-6) #m-3 --> cm-3
                    SIGMA =  sigmaList[varN]#plt_ds[varSIG].values#case[varSIG][lev]#*10**6
                    NMR = plt_ds[varNMR].values*2#*1e9 #case[varNMR][lev]*2  #  radius --> diameter

                    nlogR += NCONC/(np.log(SIGMA)*np.sqrt(2*np.pi))*np.exp(-(np.log(logR)-np.log(NMR))**2/(2*np.log(SIGMA)**2))
                    slogR += 1e-9*4.*np.pi*logR**2*(NCONC/(np.log(SIGMA)*np.sqrt(2*np.pi))*np.exp(-(np.log(logR)-np.log(NMR))**2/(2*np.log(SIGMA)**2)))
                    vlogR += 1e-9*1/6.*np.pi*logR**3*(NCONC/(np.log(SIGMA)*np.sqrt(2*np.pi))*np.exp(-(np.log(logR)-np.log(NMR))**2/(2*np.log(SIGMA)**2)))

            if dNdD:
                nlogR=nlogR/logR
                slogR=slogR/logR
                vlogR=vlogR/logR
            if (case==ctrl_name):#''CTRL'):

                if plotType=='number': ctrl_dic[model]=nlogR.copy()
                elif plotType=='surface': ctrl_dic[model]=slogR.copy()
                elif plotType=='volume': ctrl_dic[model]=vlogR.copy()
            else:
                nlogR=nlogR-ctrl_dic[model]
                slogR=slogR-ctrl_dic[model]
                vlogR=vlogR-ctrl_dic[model]

            if case==ctrl_name and not plt_ctr: continue
            if plotType=='number':
                ax_plt.plot(logR, nlogR, label=model, linestyle=linestyle_models[model], color=colors_models[model], linewidth=linewidth)

                if dNdD:ax_plt.set_ylabel('dN/dDp, #/cm$^3$', fontsize=MEDIUM_SIZE)
                else: ax_plt.set_ylabel('dN/dlogDp, #/cm$^3$', fontsize=MEDIUM_SIZE)

                if y_log:
                    if case==ctrl_name:#'CTRL':
                        ax_plt.set_yscale('log')#, linthreshy=30)
                    else:
                        ax_plt.set_yscale('symlog', linthreshy=30)
                #if case=='CTRL':
                ax_plt.legend(fontsize=SMALL_SIZE, loc=1)

            elif plotType=='surface':
                ax_plt.plot(logR, slogR, label=model, linestyle=linestyle_models[model], color=colors_models[model], linewidth=linewidth)
                if dNdD: ax_plt.set_ylabel(r'dS/dDp, um$^2$/cm$^3$', fontsize=MEDIUM_SIZE)
                else: ax_plt.set_ylabel(r'dS/dlogDp, um$^2$/cm$^3$', fontsize=MEDIUM_SIZE)
                if y_log: ax_plt.set_yscale('symlog', linthreshy=.003)
                if case==ctrl_name:#'CTRL':
                    ax_plt.legend(fontsize=SMALL_SIZE, loc=2)

            elif plotType=='volume':
                ax_plt.plot(logR, vlogR, label=model, linestyle=linestyle_models[model], color=colors_models[model], linewidth=linewidth)
                if dNdD: ax_plt.set_ylabel(r'dV/dDp, um$^3$/cm$^3$', fontsize=MEDIUM_SIZE)
                else: ax_plt.set_ylabel(r'dV/dlogDp, um$^3$/cm$^3$', fontsize=MEDIUM_SIZE)
                if y_log: ax_plt.set_yscale('symlog', linthreshy=.01)
                #if case=='CTRL':
                ax_plt.legend(fontsize=SMALL_SIZE, loc=2)


            if not lin_x: ax_plt.set_xscale('log')

            if (case==ctrl_name): title_txt=case
            else: title_txt=case+ ' - %s'%ctrl_name#CTRL'
            ax_plt.set_title(title_txt, fontsize=BIGGER_SIZE)
            ax_plt.set_xlabel('Diameter [nm]', fontsize=MEDIUM_SIZE)
            ax_plt.tick_params(labelsize=SMALL_SIZE)
            if (case !=ctrl_name):#'CTRL'):
                ylim = ax_plt.get_ylim()
                dummy = max(abs(ylim[0]),abs(ylim[1]))
                my_ylim = max(dummy,my_ylim)


    ii=0
    for case in cases:
        subp_ind = ii#tuple(np.array([int(np.floor((ii)/3)),int(ii%3)]).astype(int))
        if plt_ctr and len(cases)>1:
            ax_plt=axarr[ii]
        else:
            ax_plt=axarr
        ax_plt.grid(b=True,which="both",color='k',alpha=0.1, ls="-",axis='both')
        ii+=1
        ax_plt.set_xlim(xlim)
        if not y_log:
            ax_plt.ticklabel_format(axis='y',style='sci', scilimits=(0,4))
            ax_plt.yaxis.offsetText.set_fontsize(MEDIUM_SIZE)

        # set same ylim for all plots:
        if (case !=ctrl_name):#'CTRL'):
            ax_plt.set_ylim([-my_ylim,my_ylim])
        else:
            ax_plt.set_ylim(ylim_ctr)


    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=3.0)
    startyear=nested_means[model][case].attrs['startyear']
    endyear=nested_means[model][case].attrs['endyear']
    if plotType=='number': plotname='sizedistrib'
    elif plotType=='surface': plotname='surfdist'
    elif plotType=='volume': plotname='volumedist'

    figname=plotpath+'/'+area+'/%s_%s_%.0f_%.0f' %(plotname,area,startyear,endyear)
    for case in cases:
        figname=figname+case
    if avg_over_lev: figname=figname + '_2lev%.0f'%(pmin)
    else: figname=figname + '_atlev%.0f'%(p_level)
    if pressure_adjust: figname=figname + '_pressure_adj'
    if not y_log: figname=figname + '_linearY'
    if lin_x: figname=figname + '_linearX'
    if dNdD: figname=figname + '_dNdD'
    if (plot_mode=='presentation'): figname=figname+'presm'
    figname=figname+'.png'
    make_folders(figname)
    print(figname)
    plt.savefig(figname, dpi=300)
    plt.show()
    return



