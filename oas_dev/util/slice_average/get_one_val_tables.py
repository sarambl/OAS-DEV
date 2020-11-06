import sys

from sectional_v2.util.slice_average.avg_pkg import yearly_mean_dic
import pandas as pd
from sectional_v2.data_info import get_nice_name_case
from sectional_v2.util.naming_conventions.var_info import get_fancy_var_name, get_fancy_unit_xr


# %%
def get_pd_yearly_mean(varl,
                       cases,
                       startyear,
                       endyear,
                       pmin=850.,
                       pressure_adjust=True,
                       average_over_lev=True,
                       groupby='time.year',
                       dims=None,
                       area='Global'
                       ):

    if groupby is None:
        avg_dim = 'time'
    else:
        avg_dim = groupby.split('.')[-1]
    dummy_dic = yearly_mean_dic(varl,
                                cases,
                                startyear,
                                endyear,
                                pmin,
                                pressure_adjust,
                                model='NorESM',
                                avg_over_lev=average_over_lev,
                                groupby=groupby,
                                dims=dims,
                                area=area)
    # varl, cases, avg_over_lev=average_over_lev, groupby=groupby, dims=dims, area=area)
    dummy_dic
    di = {}
    for var in varl:
        _di_tmp = {}
        for case in cases:
            ncase = get_nice_name_case(case)
            _ds = dummy_dic[case]
            _di_tmp[ncase] = {}
            _di_tmp[ncase]['$\mu$'] = float(_ds[var].mean(avg_dim).values)
            _di_tmp[ncase]['$\sigma$'] = float(_ds[var].std(avg_dim).values)
        nvar = get_fancy_var_name(var)
        un = get_fancy_unit_xr(_ds[var], var)
        di[f'{nvar} [{un}]'] = _di_tmp.copy()
    d = {(i, j): di[i][j]
         for i in di.keys()
         for j in di[i].keys()}
    df = pd.DataFrame(d)
    return df
    # %%
def get_pd_yearly_stat(varl,
                       cases,
                       startyear,
                       endyear,
                       pmin=850.,
                       stat='mean',
                       pressure_adjust=True,
                       average_over_lev=True,
                       groupby='time.year',
                       dims=None,
                       area='Global'
                       ):

    if groupby is None:
        avg_dim = 'time'
    else:
        avg_dim = groupby.split('.')[-1]
    dummy_dic = yearly_mean_dic(varl,
                                cases,
                                startyear,
                                endyear,
                                pmin,
                                pressure_adjust,
                                model='NorESM',
                                avg_over_lev=average_over_lev,
                                groupby=groupby,
                                dims=dims,
                                area=area)
    # varl, cases, avg_over_lev=average_over_lev, groupby=groupby, dims=dims, area=area)
    dummy_dic
    di = {}
    for var in varl:
        _di_tmp = {}
        for case in cases:
            ncase = get_nice_name_case(case)
            _ds = dummy_dic[case]
            _di_tmp[ncase] = {}
            if stat=='mean':
                _di_tmp[ncase]['$\mu$'] = float(_ds[var].mean(avg_dim).values)
            elif stat=='std':
                _di_tmp[ncase]['$\sigma$'] = float(_ds[var].std(avg_dim).values)
            else:
                sys.exit(f'Cannot recognize statistic {stat}')
        nvar = get_fancy_var_name(var)
        un = get_fancy_unit_xr(_ds[var], var)
        di[f'{nvar} [{un}]'] = _di_tmp.copy()
    d = {(i, j): di[i][j]
         for i in di.keys()
         for j in di[i].keys()}
    df = pd.DataFrame(d)
    return df

# %%
def test():
    # %%
    startyear = '2008-01'
    endyear = '2010-12'
    pmin = 850.
    cases = ['SECTv21_ctrl_koagD', 'noSECTv21_ox_ricc_dd','noSECTv21_default_dd']
    varl = ['N_AER', 'NCONC01']  # ,'ACTREL']
    groupby = 'time.year'

    pressure_adjust = True

    dims = None
    area = 'Global'
    average_over_lev = True
    # %%
    df_mean = get_pd_yearly_stat(varl,
                            cases,
                            startyear,
                            endyear,
                            pmin=pmin,
                            stat='mean',
                            pressure_adjust=pressure_adjust,
                            average_over_lev=average_over_lev,
                            groupby=groupby,
                            dims=dims,
                            area=area
                            )
    # %%
    df_std = get_pd_yearly_stat(varl,
                                 cases,
                                 startyear,
                                 endyear,
                                 pmin=pmin,
                                 stat='std',
                                 pressure_adjust=pressure_adjust,
                                 average_over_lev=average_over_lev,
                                 groupby=groupby,
                                 dims=dims,
                                 area=area
                                 )
    # %%
    df = get_pd_yearly_mean(varl,
                       cases,
                       startyear,
                       endyear,
                       pmin=pmin,
                       pressure_adjust=pressure_adjust,
                       average_over_lev=average_over_lev,
                       groupby=groupby,
                       dims=dims,
                       area=area
                       )

    # %%