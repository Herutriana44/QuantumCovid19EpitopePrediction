#@title Import Library
import pandas as pd
import numpy as np
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

from qiskit.visualization import circuit_drawer
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms import VQC

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim

from qiskit_machine_learning.utils import algorithm_globals

from qiskit.circuit.library import ZZFeatureMap, TwoLocal, RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.primitives import BackendSamplerV2

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

# documentation QiskitRuntimeService https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/qiskit-runtime-service
service = QiskitRuntimeService(channel="local", token="9991f244dd7af5994186deb7974e46860dec16b86cc896b31ed4ea6fb39ed72818f7f5e3d80daf2fb00508f755b1f8484d8711ebd47a64136eb3364be57299a7")

# for check backend open
# for backend in service.backends():
#     print(backend.name)

# implement ibm cloud and fake ibm cloud
# backend = service.backend("fake_osaka")
# sampler = BackendSamplerV2(backend=backend)

from qiskit_aer.primitives import Sampler as AerSampler

if use_gpu:
  if gpu_type == "T4":
    sampler = AerSampler()
    sampler.set_options(device='GPU', max_parallel_threads=2)
  elif gpu_type == "A100":
    sampler = AerSampler()
    sampler.set_options(
        device='GPU',
        max_parallel_threads=16,        # gunakan semua vCPU yang tersedia
        max_parallel_experiments=8,     # batch eksekusi circuit
        blocking_enable=True,            # sinkronisasi batch
    )
else:
  sampler = AerSampler()

# from qiskit.primitives import StatevectorSampler
# sampler = StatevectorSampler()

#@title Load dataset
df = pd.read_csv("/content/QuantumCovid19EpitopePrediction/clean_amino_data.csv")
df = df.dropna()
df = df.reset_index(drop=True)
print(len(df))

# df['label'].value_counts().plot(kind='bar')

# #@title Resample data based on 'label' column
# def resample_by_label(df, label_col, target_count_per_label):
#     """
#     Resamples the DataFrame to have a specified number of samples for each label.

#     Args:
#         df (pd.DataFrame): The input DataFrame.
#         label_col (str): The name of the column containing the labels.
#         target_count_per_label (int): The desired number of samples for each label.

#     Returns:
#         pd.DataFrame: The resampled DataFrame.
#     """
#     resampled_df = pd.DataFrame()
#     for label_value in df[label_col].unique():
#         label_df = df[df[label_col] == label_value]
#         # Use replace=True for oversampling if needed, or adjust if target_count_per_label > len(label_df)
#         resampled_label_df = label_df.sample(n=target_count_per_label, replace=False, random_state=42)
#         resampled_df = pd.concat([resampled_df, resampled_label_df])

#     # Shuffle the resampled data to mix the labels
#     resampled_df = resampled_df.sample(frac=1, random_state=42).reset_index(drop=True)
#     return resampled_df

# # Resample the DataFrame to have 50 'E' and 50 '.' labels
# target_samples_per_label = 5000
# df_resampled = resample_by_label(df, 'label', target_samples_per_label)

# print(f"Original DataFrame length: {len(df)}")
# print(f"Resampled DataFrame length: {len(df_resampled)}")
# print("Value counts in resampled DataFrame:")
# print(df_resampled['label'].value_counts())

# # Update the original dataframe reference to the resampled one for subsequent steps
# df = df_resampled

def label_to_numeric(str):
  if str == 'E':
    return 1
  else:
    return 0

df['label_numeric'] = df['label'].apply(label_to_numeric)

# all numeric column float to int
df['length_sequence'] = df['length_sequence'].astype(int)
df['Position'] = df['Position'].astype(int)

#@title Define features and label
features = ['Position', 'length_sequence', 'numerical_amino_acid']
label = 'label_numeric'
X = df[features].values
y = df[label].values

df.head()

len(df)

#@title Split data (70:30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

#@title Define Quantum Feature Map and Ansatz
num_qubits = X_train.shape[1]

feature_map = ZZFeatureMap(num_qubits, reps=2, entanglement="full")  # Feature Map
ansatz = RealAmplitudes(num_qubits, entanglement='full', reps=3)     # Ansatz

#@title Save Quantum Circuits as images
# feature_map.draw(output='mpl')
# plt.savefig("feature_map_circuit.png")

# ansatz.draw(output='mpl')
# plt.savefig("ansatz_circuit.png")

# feature_map.decompose().draw(output='mpl')

# ansatz.decompose().draw(output='mpl')

#@title Define Optimizer
optimizer = COBYLA(maxiter=100)

#@title Define Quantum Classifier (VQC)
vqc = VQC(feature_map=feature_map, ansatz=ansatz, optimizer=optimizer, sampler=sampler)

# VQC draw circuit
# vqc.circuit.decompose().draw(output='mpl')

#@title Train Model with tqdm Progress Bar
import time

start_time = time.time()
print("Training VQC Model...")
vqc.fit(X_train, y_train)

end_time = time.time()
training_time = end_time - start_time
print(f"Training time: {training_time:.2f} seconds")

# slicing data & time
100 = 27.36 detik | 20.13 detik(versi vCPU=2) | 13.21 detik(A100, vCPU=16) <br>
1000 = 197.77 detik | | 125.26 detik(A100, vCPU=16) <br>
10000 = 2079.54 detik | | 1263.36 detik(A100, vCPU=16) | 650.39 detik <br>
full = 59709.20 detik(full rata kiri)

#@title Predict
y_pred = vqc.predict(X_test)

#@title Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
if use_drive:
  plt.savefig(os.path.join(base_dir, "confusion_matrix.png"))
else:
  plt.savefig("confusion_matrix.png")
plt.show()

#@title Classification Report
report = classification_report(y_test, y_pred, output_dict=True)
if use_drive:
  with open(os.path.join(base_dir, "classification_report.json"), "w") as f:
    json.dump(report, f)
    print("Classification report saved to", os.path.join(base_dir, "classification_report.json"))
else:
  with open("classification_report.json", "w") as f:
    json.dump(report, f)
    print("Classification report saved to", "classification_report.json")

print(classification_report(y_test, y_pred))