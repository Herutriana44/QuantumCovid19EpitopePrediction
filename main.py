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

# Load dataset
df = pd.read_csv("clean_amino_data.csv")
df = df[:100]
print(len(df))

# Define features and label
features = ['Position', 'length sequence', 'amino_numeric']
label = 'label_numeric'
X = df[features].values
y = df[label].values

# Normalize data using StandardScaler (sebelum split)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save the scaler for future use
joblib.dump(scaler, "scaler.pkl")

# Split data (70:30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define Quantum Feature Map and Ansatz
num_qubits = X_train.shape[1]

feature_map = ZZFeatureMap(num_qubits, reps=2, entanglement="full")  # Feature Map
ansatz = RealAmplitudes(num_qubits, entanglement='full', reps=3)     # Ansatz

# Save Quantum Circuits as images
feature_map.draw(output='mpl')
plt.savefig("feature_map_circuit.png")

ansatz.draw(output='mpl')
plt.savefig("ansatz_circuit.png")

# Define Optimizer
optimizer = COBYLA(maxiter=100)

# Define Quantum Classifier (VQC)
vqc = VQC(feature_map=feature_map, ansatz=ansatz, optimizer=optimizer)

# Train Model with tqdm Progress Bar
print("Training VQC Model...")
for _ in tqdm(range(1), desc="Training Progress"):
    vqc.fit(X_train, y_train)

# Predict
y_pred = vqc.predict(X_test)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig("confusion_matrix.png")

# Classification Report
report = classification_report(y_test, y_pred, output_dict=True)
with open("classification_report.json", "w") as f:
    json.dump(report, f, indent=4)

# Save Model
vqc.save('vqc_model')

print("Training Complete. Files saved.")
