# Analysis code for Blichner et al 2021

This is the analysis code assosiated with the paper:
Blichner, S.M., Sporre M.K., Makkonen, R. and Berntsen, T. "Implementing a sectional scheme for early aerosol growth from new particle formation in an Earth System Model: comparison to observations and climate impact
", submitted to GMD oct 2020. 

Please not that to reproduce the results from the paper, go to 
sectional_v2/notebooks/ and run the notebooks there. 

The EUSAAR comparison needs to be run in a specific order, please see
sectional_v2/notebooks/eusaari/code_usage.ipynb

Please direct any questions to Sara Blichner, s.m.blichner@geo.uio.no

## Software environment

To replicate the software environment, please do:
```bash 
conda env create -f env_sec_v2.yml
conda conda develop sectional_v2
```

Please note that some of the data processing is memory heavy