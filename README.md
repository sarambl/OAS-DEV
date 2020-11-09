# Analysis code for Blichner et al 2021
[![DOI](https://zenodo.org/badge/310578324.svg)](https://zenodo.org/badge/latestdoi/310578324)

This is the analysis code assosiated with the paper:
Blichner, S.M., Sporre M.K., Makkonen, R. and Berntsen, T. "Implementing a sectional scheme for early aerosol growth from new particle formation in an Earth System Model: comparison to observations and climate impact
", submitted to GMD oct 2020. 

The EUSAAR comparison needs to be run in a specific order, please see
sectional_v2/notebooks/eusaari/code_usage.ipynb

Please direct any questions to Sara Blichner, s.m.blichner@geo.uio.no

## Software environment

To replicate the software environment, please do:
```bash 
conda env create -f env_oas_dev.yml
conda activate env_oas_dev
conda conda develop sectional_v2
```

Please note that some of the data processing is memory heavy

## To reproduce results:
Please install the software environment as described above.
1. Download the eusaar dataset https://acp.copernicus.org/articles/11/5505/2011/acp-11-5505-2011.pdf
2. Download noresm output (link coming)
3. Edit oas_dev/constants.py with the correct paths
4. Go to oas_dev/notebooks and run notebooks. 
Note that notebooks in oas_dev/eusaari/ must be run with code order in oas_dev/eusaari/01-preprocess first
 
 
