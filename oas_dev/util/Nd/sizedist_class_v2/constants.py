from oas_dev.constants import project_base_path
import pandas as pd
path_Nd_var_name = project_base_path + 'OAS-DEV/oas_dev/data_info/Nd_bins.csv'

diameter_obs_df = pd.read_csv(path_Nd_var_name, index_col=0)


# %%
diameters_observation ={'N30-50':[30,50], 'N50':[50,500],'N100':[100,500], 'N250':[250,500]}