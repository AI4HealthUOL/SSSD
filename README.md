# Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models


This is the official repository for the paper [Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models](https://.com). In combination with (conditional) diffusion and state-space models, we put forward diverse algorithms, particualary, we propose the generative model $SSSD^{S4}$, which is suited to capture long-term dependencies and demonstrates ***state-of-the-art*** results in time series across diverse missing scenarios and datasets. 

## Datasets and experiments
Visit the source directory to get datasets download and experiments reproducibility instructions.

## $SSSD^{S4}$ robustness on diverse scenarios.

### Random Missing
![alt text](https://github.com/AI4HealthUOL/SSSD/blob/main/reports/plots_merged_001.png?style=centerme)

### Missing not at random
![alt text](https://github.com/AI4HealthUOL/SSSD/blob/main/reports/plots_merged_002.png?style=centerme)

### Black-out missing
![alt text](https://github.com/AI4HealthUOL/SSSD/blob/main/reports/plots_merged_003.png?style=centerme)

### Forecast
![alt text](https://github.com/AI4HealthUOL/SSSD/blob/main/reports/plots_merged_004.png?style=centerme)



## Our proposed $SSSD^{S4}$ model architecture:
![alt text](https://github.com/AI4HealthUOL/SSSD/blob/main/reports/SSSDS4architecture.png?style=centerme)


### Please cite our publication if you found our research to be helpful.

```
@misc{}
```

### Acknowledgments
We would like thank the authors of the the S4 model for releasing and maintaining the
source code for [Structured State Space Models](https://github.com/HazyResearch/state-spaces). Similarly, our $SSSD^{S4}$ code builds on the implementation provided by [DiffWave](https://github.com/philsyn/DiffWave-Vocoder).
