Python Research Code - computational experiments with Dataset D1 — M4 Quarterly Finance and Dataset D2 — Exchange-Rate 

Overview

This repository contains the Python implementation of the proposed forecasting framework and the computational experiments conducted using the M4 Quarterly Finance dataset and the Exchange-Rate dataset.

The implementation includes phase-space reconstruction (PSR), topological data analysis (TDA), transfer entropy (TE), causal graph construction, Neural Ordinary Differential Equation (Neural ODE) forecasting, and conformal prediction (CP).

The repository provides scripts for data preprocessing, model training, baseline comparison, ablation studies, robustness experiments, and statistical analysis of forecasting results.

Datasets

The experiments use the following datasets:

M4 Quarterly Finance — quarterly financial time series from the M4 Competition.

Exchange-Rate — multivariate time series containing exchange rates of eight currencies.

Data Preparation

The datasets should be downloaded separately and placed in a data directory located one level above the repository directory.

Expected directory structure:

project/
├── data/
│   ├── M4-info.csv
│   ├── Quarterly-train.csv
│   ├── Quarterly-test.csv
│   └── exchange_rate.txt.gz
│
└── python-research-code/
    ├── README.md
    ├── d1_prep.py
    ├── d2_prep.py
    └── ...

The M4 dataset can be obtained from:

https://github.com/Mcompetitions/M4-methods

The Exchange-Rate dataset can be obtained from:

https://github.com/laiguokun/multivariate-time-series-data

Requirements

The implementation requires Python 3.10 or later and the following libraries:

NumPy

Pandas

SciPy

PyTorch

Ripser

StatsForecast

NeuralForecast

Install the dependencies using:

pip install -r requirements.txt

Repository Structure

File

Description

common.py

Core algorithms, feature extraction, causal analysis, and Neural ODE implementation

d1_prep.py

Data preparation for Dataset D1 (M4 Quarterly Finance)

d2_prep.py

Data preparation for Dataset D2 (Exchange-Rate)

run_d1.py

Training and evaluation of the proposed method and ablation variants on D1

run_d2.py

Training and evaluation of the proposed method and ablation variants on D2

base_d1.py

Baseline forecasting experiments on D1

base_d2.py

Baseline forecasting experiments on D2

dist_bench.py

Computational performance benchmarking

analyze.py

Aggregation and statistical analysis of experimental results

summary.json

Summary of experimental results

dist_bench.json

Benchmark results

Experimental Workflow

Dataset D1: M4 Quarterly Finance

Prepare the dataset:

python d1_prep.py

Prepare the limited-history experimental condition:

python d1_prep.py 20

Run the proposed method:

python run_d1.py full 1

Run the ablation experiments:

python run_d1.py noTDA 1
python run_d1.py nodeonly 1
python run_d1.py rawTDA 1

For the limited-history condition:

python run_d1.py full 1 _h20

Dataset D2: Exchange-Rate

Prepare the dataset for a forecasting horizon of 5:

python d2_prep.py 5 clean

Prepare the dataset for a forecasting horizon of 20:

python d2_prep.py 20 clean

Robustness experiments:

python d2_prep.py 5 missing
python d2_prep.py 5 noise

Run the proposed method:

python run_d2.py full 1 5 clean

Ablation experiments:

python run_d2.py noTDA 1 5 clean
python run_d2.py noCG 1 5 clean
python run_d2.py noTE 1 5 clean
python run_d2.py nodeonly 1 5 clean

Baseline Models

The repository includes scripts for evaluating baseline forecasting models.

For Dataset D1:

python base_d1.py stat
python base_d1.py LSTM 1
python base_d1.py DeepAR 1
python base_d1.py TFT 1

For Dataset D2:

python base_d2.py stat 1 5 clean
python base_d2.py LSTM 1 5 clean
python base_d2.py DeepAR 1 5 clean
python base_d2.py TFT 1 5 clean

Results and Statistical Analysis

The experimental results are saved in the res directory.

The analysis script aggregates results across experimental runs and computes forecasting metrics, confidence interval coverage, and statistical comparisons between models.

Run the analysis:

python analyze.py

Reproducibility

The scripts support repeated experiments with different random seeds.

The results depend on the selected experimental configuration, dataset condition, forecasting horizon, and random seed.
