# Lipo_MF_Model

## Purpose
This repository contains code and data for predicting the lipophilicity of chemical compounds using an MLP Regressor on Morgan Fingerprints. 

## Contents
- `HW5.py`
- `Lipophilicity.csv`
- `environment.yml`
- `README.md`
- `LICENSE.txt`


## Installation Instructions

1. **Clone the repository**:
```sh
git clone https://github.com/yourusername/lipophilicity-prediction.git
cd lipophilicity-prediction
```
2. **Create and activate a conda environment**:
```sh
conda env create -f environment.yml
conda activate HW5
```
3. **Run the Python Script**:
```sh
python HW5.py
```

## Usage

To run the project from the command line, use the HW5.py script. This script will preprocess the data, train the model, and save the results to results.txt.

## Hyperparameters

- --hidden_layer_sizes (int): Each consecutive int input adds a hidden layer of that size
- --max_iter (int): Maximum number of iterations for MLPRegressor

## License
This project is licensed under ____. See the LICENSE.txt file for more details. 