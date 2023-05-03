# Regime Identification
Segmentation of Multivariate Non-stationary Time series 


This repository contains code for our work on regime identification for discovering causal graph in geo-climate time series.


## Overview

We extract covariances &Sigma matrics discover full causal graph in multivariate nonlinear systems by testing model invariance against Knockoffs-based interventional environments:
1. First we train deep network <img src="https://render.githubusercontent.com/render/math?math=f_i"> using data from observational environment <img src="https://render.githubusercontent.com/render/math?math=E_i">.
2. Then we expose the model to Knockoffs-based interventional environments <img src="https://render.githubusercontent.com/render/math?math=E_k">. 
3. For each pair variables {<img src="https://render.githubusercontent.com/render/math?math=z_i">, <img src="https://render.githubusercontent.com/render/math?math=z_j">} in nonlinear system, we test model invariance across environments. 
4. We perform KS test over distribution <img src="https://render.githubusercontent.com/render/math?math=R_i">, <img src="https://render.githubusercontent.com/render/math?math=R_k"> of model residuals in various environments. 
Our NULL hypothesis is that variable <img src="https://render.githubusercontent.com/render/math?math=z_i"> does not cause <img src="https://render.githubusercontent.com/render/math?math=z_j">, 
<img src="https://render.githubusercontent.com/render/math?math=H_0">: <img src="https://render.githubusercontent.com/render/math?math=R_i"> = <img src="https://render.githubusercontent.com/render/math?math=R_k">, 
else the alternate hypothesis <img src="https://render.githubusercontent.com/render/math?math=H_1">: <img src="https://render.githubusercontent.com/render/math?math=R_i"> != <img src="https://render.githubusercontent.com/render/math?math=R_k">  is accepted.

<p align="center">
<img src="res/synregimes.png" width=100% />
</p>

## Data
We test our method on synthetically generated multivariate nonlinear non-stationary time series as well as geo-climate time series (Recorded at Moxa Geodynamic Observatory) which can be found under `datasets/` directory. The synthetic data is generated using file `src/synthetic_dataset.py`. 


## Code
`src/regimes.py` is our main file, where we segment time series into batches that have specific dynamics.
- `src/regimes_identification.ipynb` for actual and counterfactual outcome generation using interventions.
- `src/plot_regimes.py` for visualization of time series segmention.
- `datasets/` contains the generated synthetic data and real dataset.


## Dependencies
`requirements.txt` contains all the packages that are related to the project.
To install them, simply create a new [conda](https://docs.conda.io/en/latest/) environment and type
```
pip install -r requirements.txt
```

