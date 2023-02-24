# Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models


This is the official repository for the paper [Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models](https://arxiv.org/abs/2208.09399). In combination with (conditional) diffusion and state-space models, we put forward diverse algorithms, particualary, we propose the generative model $SSSD^{S4}$, which is suited to capture long-term dependencies and demonstrates ***state-of-the-art*** results in time series across diverse missing scenarios and datasets. 

## Datasets and experiments
Visit the source directory to get datasets download and experiments reproducibility instructions. (here is an [example](https://github.com/AI4HealthUOL/SSSD/blob/main/docs/instructions/PEMS-Bay%20and%20METR-LA/feature_sample_process.ipynb) of the feature sampling approach for the datasets with large number of channels )


## Our proposed $SSSD^{S4}$ model architecture:
![alt text](https://github.com/AI4HealthUOL/SSSD/blob/main/reports/updated_architecture.png?style=centerme)

## $SSSD^{S4}$ robustness on diverse scenarios:

### Random Missing
![alt text](https://github.com/AI4HealthUOL/SSSD/blob/main/reports/plots_merged_001.png?style=centerme)

### Missing not at random
![alt text](https://github.com/AI4HealthUOL/SSSD/blob/main/reports/plots_merged_002.png?style=centerme)

### Black-out missing
![alt text](https://github.com/AI4HealthUOL/SSSD/blob/main/reports/plots_merged_003.png?style=centerme)

### Forecast
![alt text](https://github.com/AI4HealthUOL/SSSD/blob/main/reports/plots_merged_004.png?style=centerme)




### Please cite our publication if you found our research to be helpful.

```bibtex
@article{
lopez alcaraz2022diffusionbased,
title={Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models},
author={Juan Lopez Alcaraz and Nils Strodthoff},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2022},
url={https://openreview.net/forum?id=hHiIbk7ApW},
}

```

### Acknowledgments
We would like thank the authors of the the S4 model for releasing and maintaining the
source code for [Structured State Space Models](https://github.com/HazyResearch/state-spaces). Similarly, our proposed model code builds on the implementation provided by [DiffWave](https://github.com/philsyn/DiffWave-Vocoder).
