import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
import argparse

dataset = "../Lipophilicity.csv"
env_name = os.getenv("CONDA_DEFAULT_ENV")
print(f'Current Environment: {env_name}')

# Load the dataset
data = pd.read_csv(dataset)
data.drop('CMPD_CHEMBLID', axis=1, inplace=True)
data.head()

#train and test split
X = data.drop('exp', axis=1)
y = data['exp']
features, features_test, targets, targets_test = train_test_split(X, y, test_size=0.2, random_state=42)
#features.head()
targets_test.head()

#scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


#targets scaling
targets_scaled = scaler.fit_transform(targets.values.reshape(-1, 1))
targets_test_scaled = scaler.transform(targets_test.values.reshape(-1, 1))

X_test = features_test
y_test = targets_test_scaled
X_train = features
y_train = targets_scaled



# Parse command line arguments for hyperparameters
parser = argparse.ArgumentParser(description='Train MLPRegressor with user-defined hyperparameters.')
parser.add_argument('--hidden_layer_sizes', type=int, nargs='+', default=[100, 100], help='Hidden layer sizes for MLPRegressor')
parser.add_argument('--max_iter', type=int, default=1000, help='Maximum number of iterations for MLPRegressor')
args = parser.parse_args()

# Extract hyperparameters from arguments
hidden_layer_sizes = tuple(args.hidden_layer_sizes)
max_iter = args.max_iter

print(f'Using hidden_layer_sizes: {hidden_layer_sizes} and max_iter: {max_iter}')

def morgan_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    return generator.GetFingerprintAsNumPy(mol)

def maccs_keys(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return [int(x) for x in AllChem.GetMACCSKeysFingerprint(mol)]

def calculate_rmse(y_true, y_pred):
    residuals = y_true - y_pred
    squared_residuals = residuals ** 2
    mean_squared_residuals = np.mean(squared_residuals)
    rmse = np.sqrt(mean_squared_residuals)
    return rmse


#Morgan Fingerprint

train = X_train['smiles'].apply(morgan_fingerprint)
test = X_test['smiles'].apply(morgan_fingerprint)

train_list = train.apply(lambda x: list(x))
test_list = test.apply(lambda x: list(x))

X_train = pd.DataFrame(train_list.tolist())
X_test = pd.DataFrame(test_list.tolist())

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

y_test = y_test.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)

X_train


#morgan fingerprint model
model = MLPRegressor(hidden_layer_sizes=(hidden_layer_sizes), max_iter=max_iter, random_state=42)
model.fit(X_train, y_train.ravel())
#morgan_score = model.score(X_test, y_test)
#print(f'Morgan score: {morgan_score}')

#RMSE unscaled targets
y_pred = model.predict(X_test)
morgan_rmse = calculate_rmse(y_test, y_pred)
print(f'Morgan RMSE: {morgan_rmse}')


#MACCS keys
X_test = features_test
y_test = targets_test_scaled
X_train = features
y_train = targets_scaled



X_train['MACCS'] = X_train['smiles'].apply(maccs_keys)
X_test['MACCS'] = X_test['smiles'].apply(maccs_keys)

X_train = pd.concat([X_train.drop('MACCS', axis=1), X_train['MACCS'].apply(pd.Series)], axis=1)
X_test = pd.concat([X_test.drop('MACCS', axis=1), X_test['MACCS'].apply(pd.Series)], axis=1)

X_train = X_train.drop(['smiles'], axis=1)
X_test = X_test.drop(['smiles'], axis=1)

X_train.head()


#MACCS keys model
model = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=42)
model.fit(X_train, y_train.ravel())
#maccs_score = model.score(X_test, y_test)
#print(f'MACCS score: {maccs_score}')

#RMSE unscaled targets
y_pred = model.predict(X_test)
maccs_rmse = calculate_rmse(y_test, y_pred)
print(f'MACCS RMSE: {maccs_rmse}')

# Save the test set RMSE, conda environment name, and hyperparameters to a file
with open('results.txt', 'w') as f:
    f.write(f'Environment: {env_name}\n')
    f.write('Hyperparameters:\n')
    f.write(f'Hidden Layer Sizes: {hidden_layer_sizes}\n')
    f.write(f'Max Iterations: {max_iter}\n')
    f.write(f'Morgan RMSE: {morgan_rmse}\n')
    f.write(f'MACCS RMSE: {maccs_rmse}\n')
