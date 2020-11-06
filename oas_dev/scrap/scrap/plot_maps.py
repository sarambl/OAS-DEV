import numpy as np
import useful_scit.plot
from cartopy import crs as ccrs
from matplotlib import pyplot as plt, colors as colors

import sectional_v2.util
from sectional_v2.util import practical_functions as practical_functions
from sectional_v2.util.naming_conventions.var_info import get_fancy_var_name
from sectional_v2.util.plot.plot_maps import subplots_map, get_global_avg_map, frelative, fdifference, set_vmin_vmax_diff, \
    get_avg_diff, fix_axis4map_plot, map_projection, save_map_name


def plot_map_diff(var, cases, cases_dic, relative=False, n_rows=2, figsize=[20, 6], cbar_equal=True,
                  kwargs_diff={},
                  kwargs_ctr={}):
    """
    Plot diff only
    :param var:
    :param cases:
    :param cases_dic:
    :param relative:
    :param n_rows:
    :param figsize:
    :param cbar_equal:
    :param kwargs_diff:
    :param kwargs_ctr:
    :return:
    """
    ctrl_case = cases[0]
    n_cases = len(cases)
    n_cols = int(np.ceil(n_cases / n_rows))
    fig, axs = subplots_map(n_rows, n_cols, figsize=figsize, )  # Orthographic(10, 0))
    ctrl = cases_dic[ctrl_case][var]
    if len(cases) == 1:
        axs = [axs]
    else:
        axs = axs.flatten()
    # Get avg diff from ctrl
    glob_avg_ctrl = get_global_avg_map(ctrl_case, cases_dic, var)
    if relative:
        func = frelative
        unit = ' [%]'
        title_ext = ' to %s' % ctrl_case
    else:
        func = fdifference
        unit = ' [%s]' % sectional_v2.util.naming_conventions.var_info.get_fancy_unit_xr(ctrl, var)
        title_ext = '-%s' % ctrl_case

    # Difference to ctr
    set_vmin_vmax_diff(cases, cbar_equal, ctrl_case, func, kwargs_diff, cases_dic, var)

    for case, ax in zip(cases, axs):
        if case == ctrl_case:
            plt_var = cases_dic[case][var]
            im = plt_var.plot(ax=ax, transform=ccrs.PlateCarree(), **kwargs_ctr)  # vmin=vmin, vmax=vmax)
            cb = im.colorbar
            cb.set_label(
                get_fancy_var_name(var) + ' [%s]' % sectional_v2.util.naming_conventions.var_info.get_fancy_unit_xr(
                    plt_var, var))
        else:

            plt_var = func(cases_dic[case][var], cases_dic[ctrl_case][var])
            im = plt_var.plot(ax=ax, transform=ccrs.PlateCarree(), **kwargs_diff)
            cb = im.colorbar
            cb.set_label(get_fancy_var_name(var) + unit)
        if case == ctrl_case:
            ax.set_title(case + ', mean:%.2E' % glob_avg_ctrl)
        else:

            glob_diff = get_avg_diff(case, ctrl_case, cases_dic, relative, var)
            ax.set_title('%s ' % case + title_ext + glob_diff)
        fix_axis4map_plot(ax)
    plt.tight_layout()
    return axs, kwargs_diff.copy()


def plot_map_vardiff(varList, case, nested_cases, relative=False, n_rows=5, figsize=[15, 8], cbar_equal=True,
                     symzero=True):
    n_cases = len(varList)
    n_cols = int(np.ceil(n_cases / n_rows))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize,
                            subplot_kw={'projection': ccrs.PlateCarree()})  # Orthographic(10, 0))
    axs = axs.flatten()
    ii = 0

    def frelative(xr1, xr2):
        out_xr = (xr1 - xr2) / xr2 * 100
        out_xr.attrs['units'] = '%'
        return out_xr

    def fdifference(xr1, xr2):
        return (xr1 - xr2)

    if relative:
        func = frelative
    else:
        func = fdifference

    plt_not_ctrl = [func(nested_cases[case][var], nested_cases['CTRL'][var]) for var in varList]
    vmin, vmax = useful_scit.plot.plot.calc_vmin_vmax(plt_not_ctrl)
    if vmin < 0 and vmax > 0:
        minmax = max(abs(vmin), vmax)
        vmin = -minmax
        vmax = minmax
    kwargs = {'robust': True}
    if cbar_equal:
        kwargs['vmin'] = vmin
        kwargs['vmax'] = vmax
    for var in varList:
        ax = axs[ii]
        if case != 'CTRL':
            plt_var = func(nested_cases[case][var], nested_cases['CTRL'][var])
        else:
            plt_var = nested_cases[case][var]  # , nested_cases['CTRL'][var]

        im = plt_var.plot(ax=ax, transform=ccrs.PlateCarree(), **kwargs)
        ii += 1
        ax.set_title(case + ': ' + var)
        fix_axis4map_plot(ax)
    plt.tight_layout()
    return axs


def plot_map_cases(case_dic, cases, model_name, varList, pressure, pressure_coord=False, logscale=False,
                   figsize=[30, 10],
                   relative=False):
    print('DEPRICATED')
    two_rows = len(cases) > 4

    for var in varList:
        # maxmin_cases={}
        maxmin = 0.
        for case in cases[1::]:
            if relative:
                plot_xr = 100. * (case_dic[case][var] - case_dic[cases[0]][var]) / np.abs(
                    case_dic[cases[0]][var].values)
            else:
                plot_xr = case_dic[case][var] - case_dic[cases[0]][var]
            field = plot_xr.values  # (case_dic[case][var]-case_dic[cases[0]][var]).values
            min_tmp, max_tmp = practical_functions.max_min_without_outliers(field, np.min(field), np.max(field))
            maxmin = max(max(np.abs(min_tmp), np.abs(max_tmp)), maxmin)
        print('%s maxmin: %f' % (var, maxmin))

        if two_rows:
            fig, axs = plt.subplots(2, 4, figsize=figsize,
                                    subplot_kw={'projection': map_projection})  # ccrs.PlateCarree()})
        else:
            fig, axs = plt.subplots(1, 4, figsize=[figsize[0], figsize[1] / 2],
                                    subplot_kw={'projection': map_projection})  # ccrs.PlateCarree()})
        ii = 0
        for case in cases:
            if two_rows:
                subp_ind = tuple(np.array([int(np.floor((ii) / 4)), int(ii % 4)]).astype(int))
            else:
                subp_ind = ii  # tuple(np.array([int(np.floor((ii)/4)),int(ii%4)]).astype(int))
            if case == cases[0]:
                min_tmp, max_tmp = practical_functions.max_min_without_outliers(case_dic[case][var].values,
                                                                                case_dic[case][var].min(),
                                                                                case_dic[case][var].max())

                if logscale:
                    im = case_dic[case][var].plot(ax=axs[subp_ind], norm=colors.SymLogNorm(linthresh=10, linscale=1),
                                                  vmin=min_tmp, vmax=max_tmp, transform=ccrs.PlateCarree(), robust=True)
                else:
                    im = case_dic[case][var].plot(ax=axs[subp_ind], transform=ccrs.PlateCarree(),
                                                  robust=True)  # , norm=colors.SymLogNorm(linthresh=10, linscale=1))
                axs[subp_ind].set_title(cases[0])
            else:
                if relative:
                    plot_xr = 100. * (case_dic[case][var] - case_dic[cases[0]][var]) / np.abs(
                        case_dic[cases[0]][var].values)
                else:
                    plot_xr = case_dic[case][var] - case_dic[cases[0]][var]
                if logscale:
                    im = plot_xr.plot(cmap='bwr', ax=axs[subp_ind], vmin=-maxmin, vmax=maxmin,
                                      norm=colors.SymLogNorm(linthresh=10, linscale=1),
                                      transform=ccrs.PlateCarree(), robust=True)
                else:
                    im = plot_xr.plot(cmap='bwr', ax=axs[subp_ind], vmin=-maxmin, vmax=maxmin,
                                      # norm=colors.SymLogNorm(linthresh=10, linscale=1),
                                      transform=ccrs.PlateCarree(), robust=True)

                if relative:
                    title = '%s  change' % (case)
                    clabel = '%s [%%]' % (get_fancy_var_name(var))
                else:
                    title = '%s-%s' % (case, cases[0])
                    clabel = '%s [%s]' % (get_fancy_var_name(var),
                                          sectional_v2.util.naming_conventions.var_info.get_fancy_unit_xr(
                                              case_dic[case][var], var))
                axs[subp_ind].set_title(title)
                cb = im.colorbar
                cb.set_label(clabel)
            ii += 1
            axs[subp_ind].set_global()
            axs[subp_ind].coastlines()
        filen = save_map_name(var, cases, case_dic[case]['lev'].values, pressure_coord, logscale=logscale,
                              relative=relative)
        print(filen)
        practical_functions.make_folders(filen)
        plt.savefig(filen, dpi=300)
        plt.show()