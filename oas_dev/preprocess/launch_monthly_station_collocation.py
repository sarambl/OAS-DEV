# %%
import sys
import time

from sectional_v2.constants import project_base_path
import subprocess
import pandas as pd
import useful_scit.util.log as log


def check_nrproc(l_df: pd.DataFrame):
    _b = l_df['process'].isnull()
    return len(l_df[~_b])


def check_stat_proc(l_df):
    """
    Check nr of processes done/running or not running
    :param l_df:
    :return:
    """
    nrunning = l_df['status'][l_df['status'] == 'running'].count()
    ndone = l_df['status'][l_df['status'] == 'done'].count()
    nnrunning = l_df['status'][l_df['status'] == 'not running'].count()
    return nrunning, ndone, nnrunning


# %%
def update_stat_proc(r):
    """
    Update row if process is done
    :param r:
    :return:
    """
    if r['status'] == 'done': return 'done'
    if r['status'] == 'not running':
        return 'not running'
    else:
        p = r['process']
        p: subprocess.Popen
        if p.poll() is None:
            return 'running'
        else:
            return 'done'


# %%
def launch_monthly_station_output(case, isSectional, max_launches=5, from_time='2008-01-01', to_time='2009-01-01'):
    """
    Launch a number of processes to calculate monthly files for file.
    :param case: Case to be run
    :param isSectional: Whether sectional or not
    :param max_launches: maximum launched subprocesses at a time
    :param from_time: From time
    :param to_time: To time
    :return:
    """
    # Setup dataframe to keep track of processes
    _dr = pd.date_range(from_time, to_time, freq='MS')[:-1]  # DataFrame(range())
    l_df = pd.DataFrame(index=_dr, columns=['process', 'status'])
    l_df['status'] = 'not running'
    l_df['status'] = l_df.apply(update_stat_proc, axis=1)
    check_stat_proc(l_df)
    pyf = sys.executable  # "/persistent01/miniconda3/envs/env_sec_v2/bin/python3"
    file = project_base_path + 'OAS-DEV/sectional_v2/notebooks/eusaari/01-preprocess/01-01-preprocess-model-output_station_subproc.py'
    log.ger.info(f'Calculating with {pyf}')
    # while loop:
    notDone = True
    while notDone:
        # Update status
        l_df['status'] = l_df.apply(update_stat_proc, axis=1)
        nrunning, ndone, nnrunning = check_stat_proc(l_df)
        # check if done, if so break
        notDone = len(l_df) != ndone
        if notDone is False:
            break
        # If room for one more process:
        if (nrunning < max_launches) and (nnrunning > 0):
            nr_row = l_df[l_df['status'] == 'not running'].iloc[0]
            # from time to time
            time1 = nr_row.name.strftime(format='%Y-%m-%d')
            time2 = (nr_row.name + pd.DateOffset(months=1)).strftime(format='%Y-%m-%d')
            # Launch subprocess:
            p1 = subprocess.Popen([pyf, file, str(time1), str(time2), case, str(isSectional)])
            # put subprocess in dataframe
            l_df.loc[nr_row.name, 'process'] = p1
            l_df.loc[nr_row.name, 'status'] = 'running'

        log.ger.info(l_df)
        time.sleep(4)
