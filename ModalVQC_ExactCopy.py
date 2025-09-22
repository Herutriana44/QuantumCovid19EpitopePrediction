#!/usr/bin/env python3
"""
COVID-19 Epitope Prediction VQC - Modal Deployment (Exact Copy Structure)
Quantum Machine Learning untuk prediksi epitope COVID-19 menggunakan VQC (Variational Quantum Classifier)
dari qiskit-machine-learning dengan deployment di Modal platform.
Struktur mengikuti QuantumCovid19EpitopePrediction_Qiskit copy.py
"""

import modal
import numpy as np
from typing import Dict, Any
import json
import os
import datetime
import random

# Define Modal image dengan semua dependencies
image = (
    modal.Image.debian_slim()
    .pip_install([
        "qiskit",
        "qiskit-machine-learning", 
        "qiskit-aer-gpu",
        "qiskit-algorithms",
        "qiskit_ibm_runtime",
        "numpy",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "pandas",
        "seaborn",
        "tqdm",
        "joblib",
        "requests",
        "torch"
    ])
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
)

# Create Modal app
app = modal.App("vqc-exact-copy", image=image)

# Shared volume untuk menyimpan results
volume = modal.Volume.from_name("vqc-exact-results", create_if_missing=True)

@app.function(
    gpu="A100-80GB",
    timeout=3600*20,
    volumes={"/results": volume},
    cpu=8,
    memory=16384  # 16GB memory
)
def train_vqc_exact() -> Dict[str, Any]:
    """
    Train VQC model untuk COVID-19 Epitope Prediction menggunakan VQC dari qiskit-machine-learning.
    Struktur mengikuti QuantumCovid19EpitopePrediction_Qiskit copy.py
    """
    
    # Setup base directory (mirroring original structure)
    base_dir = 'QuantumEpitopeCovid19Logs'
    os.makedirs(f'/results/{base_dir}', exist_ok=True)
    
    # GPU Configuration (exact copy from original)
    use_gpu = True
    gpu_type = "A100"  # Fixed to B200 for Modal
    
    # Import Library (exact copy from original)
    import pandas as pd
    import numpy as np
    import json
    import joblib
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    from tqdm import tqdm
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import confusion_matrix, classification_report
    
    # Qiskit imports (exact copy from original)
    from qiskit.visualization import circuit_drawer
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit_machine_learning.algorithms import VQC
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    
    from qiskit_machine_learning.utils import algorithm_globals
    from qiskit.circuit.library import TwoLocal
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.primitives import StatevectorSampler
    from qiskit_machine_learning.neural_networks import SamplerQNN
    from qiskit_machine_learning.connectors import TorchConnector
    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit.primitives import BackendSamplerV2
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit_aer.primitives import Sampler as AerSampler
    
    # Setup logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_num = random.randint(100, 999)
    log_filename = f"{timestamp}_{random_num}.txt"
    
    # Create log file paths
    modal_log_path = f"/results/{log_filename}"
    local_log_path = f"./{log_filename}"
    
    # Custom print function
    def log_print(*args, **kwargs):
        print(*args, **kwargs)
        
        # Write to Modal volume
        try:
            with open(modal_log_path, 'a', encoding='utf-8') as f:
                print(*args, **kwargs, file=f)
        except Exception as e:
            print(f"Warning: Could not write to Modal log: {e}")
        
        # Write to local
        try:
            with open(local_log_path, 'a', encoding='utf-8') as f:
                print(*args, **kwargs, file=f)
        except Exception as e:
            print(f"Warning: Could not write to local log: {e}")
    
    # Initialize log files
    log_header = f"COVID-19 Epitope Prediction VQC Training Log\nStarted at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nLog file: {log_filename}\n" + "="*80 + "\n\n"
    
    try:
        with open(modal_log_path, 'w', encoding='utf-8') as f:
            f.write(log_header)
        print(f"✅ Modal log file created: {modal_log_path}")
    except Exception as e:
        print(f"⚠️ Could not create Modal log file: {e}")
    
    try:
        with open(local_log_path, 'w', encoding='utf-8') as f:
            f.write(log_header)
        print(f"✅ Local log file created: {local_log_path}")
    except Exception as e:
        print(f"⚠️ Could not create local log file: {e}")
    
    log_print(f"📝 Training log will be saved to: {log_filename}")
    log_print(f"Base directory: {base_dir}")
    
    # GPU Configuration (exact copy from original)
    log_print(f"🚀 Starting VQC training for COVID-19 Epitope Prediction")
    log_print(f"GPU Type: {gpu_type}")
    
    if use_gpu:
        if gpu_type == "T4":
            sampler = AerSampler()
            sampler.set_options(device='GPU', max_parallel_threads=2)
            log_print("✅ Aer GPU T4 configuration applied")
        elif gpu_type == "A100":
            sampler = AerSampler()
            sampler.set_options(
                device='GPU',
                max_parallel_threads=16,        # gunakan semua vCPU yang tersedia
                max_parallel_experiments=8,     # batch eksekusi circuit
                blocking_enable=True,            # sinkronisasi batch
            )
            log_print("✅ Aer GPU A100 configuration applied")
        elif gpu_type == "B200":
            sampler = AerSampler()
            sampler.set_options(
                device='GPU',
                max_parallel_threads=64,         # Optimized untuk single GPU
                max_parallel_experiments=32,     # Enhanced batch processing
                max_parallel_shots=100000,       # Large shot counts untuk single GPU
                blocking_enable=True,
                blocking_qubits=12,              # Optimal blocking untuk complex circuits
                precision='single',              # Optimize untuk speed
                batched_optimization=True,       # Enable batch optimization
                memory_mb=4096                   # 4GB memory allocation
            )
            log_print("✅ Aer GPU B200 configuration applied")
    else:
        sampler = AerSampler()
        log_print("✅ Aer CPU configuration")
    
    # Alternative sampler options (commented out like original)
    # from qiskit.primitives import StatevectorSampler
    # sampler = StatevectorSampler()
    
    # Load dataset (modified for Modal - download from GitHub)
    log_print("📊 Loading dataset...")
    dataset_url = "https://raw.githubusercontent.com/Herutriana44/QuantumCovid19EpitopePrediction/main/clean_after_balanced_amino_data.csv"
    
    try:
        import requests
        response = requests.get(dataset_url)
        response.raise_for_status()
        
        temp_file = "/tmp/clean_after_balanced_amino_data.csv"
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        df = pd.read_csv(temp_file)
        df = df.dropna()
        df = df.reset_index(drop=True)
        log_print(f"Dataset loaded: {len(df)} samples")
        os.remove(temp_file)
        
    except Exception as e:
        log_print(f"❌ Failed to load dataset: {str(e)}")
        return {"error": f"Dataset load failed: {str(e)}"}
    
    # df['label'].value_counts().plot(kind='bar')  # Commented out like original
    
    # Resample data based on 'label' column (exact copy from original)
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

    # # Resample the DataFrame to have 500 'E' and 500 '.' labels (exact copy from original)
    # target_samples_per_label = 50
    # df_resampled = resample_by_label(df, 'label', target_samples_per_label)

    # log_print(f"Original DataFrame length: {len(df)}")
    # log_print(f"Resampled DataFrame length: {len(df_resampled)}")
    # log_print("Value counts in resampled DataFrame:")
    # log_print(df_resampled['label'].value_counts())

    # # Update the original dataframe reference to the resampled one for subsequent steps
    # df = df_resampled

    # Label conversion function (exact copy from original)
    def label_to_numeric(str):
        if str == 'E':
            return 1
        else:
            return 0

    df['label_numeric'] = df['label'].apply(label_to_numeric)

    # all numeric column float to int (exact copy from original)
    df['length_sequence'] = df['length_sequence'].astype(int)
    df['Position'] = df['Position'].astype(int)

    # Define features and label (exact copy from original)
    features = ['Position', 'length_sequence', 'numerical_amino_acid']
    label = 'label_numeric'
    X = df[features].values
    y = df[label].values

    log_print(f"Features shape: {X.shape}")
    log_print(f"Labels shape: {y.shape}")
    log_print(f"Label distribution: {np.bincount(y)}")

    # df.head()  # Commented out like original
    # len(df)    # Commented out like original

    # Split data (70:30) (exact copy from original)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

    # Separate data by label
    df_label_0 = df[df['label_numeric'] == 0]
    df_label_1 = df[df['label_numeric'] == 1]

    # Determine the minimum count between the two labels
    min_count = min(len(df_label_0), len(df_label_1))

    # Randomly sample an equal number of instances for each label
    df_balanced_0 = df_label_0.sample(n=min_count, random_state=42, replace=False)
    df_balanced_1 = df_label_1.sample(n=min_count, random_state=42, replace=False)

    # Concatenate the balanced dataframes
    df_balanced = pd.concat([df_balanced_0, df_balanced_1])

    # Shuffle the balanced dataframe
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    # Define features and label for the balanced data
    X_balanced = df_balanced[features].values
    y_balanced = df_balanced[label].values

    # Split the balanced data into training and testing sets (70:30)
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.3, random_state=42, shuffle=True, stratify=y_balanced)

    print("Value counts in y_train:")
    print(pd.Series(y_train).value_counts())

    print("\nValue counts in y_test:")
    print(pd.Series(y_test).value_counts())
    
    log_print(f"Train samples: {len(X_train)}")
    log_print(f"Test samples: {len(X_test)}")

    # Define Quantum Feature Map and Ansatz (exact copy from original)
    num_qubits = X_train.shape[1]
    log_print(f"Number of qubits: {num_qubits}")

    feature_map = ZZFeatureMap(num_qubits, reps=2, entanglement="full")  # Feature Map
    ansatz = RealAmplitudes(num_qubits, entanglement='full', reps=3)     # Ansatz

    log_print("✅ Quantum circuits created:")
    log_print(f"  Feature Map: ZZFeatureMap with {num_qubits} qubits, reps=2")
    log_print(f"  Ansatz: RealAmplitudes with {num_qubits} qubits, reps=3")

    # Save Quantum Circuits as images (commented out like original)
    # feature_map.draw(output='mpl')
    # plt.savefig("feature_map_circuit.png")

    # ansatz.draw(output='mpl')
    # plt.savefig("ansatz_circuit.png")

    # feature_map.decompose().draw(output='mpl')
    # ansatz.decompose().draw(output='mpl')

    # Define Optimizer (exact copy from original)
    optimizer = COBYLA(maxiter=100)
    log_print(f"✅ Optimizer: COBYLA with maxiter=100")

    # Define Quantum Classifier (VQC) (exact copy from original)
    vqc = VQC(feature_map=feature_map, ansatz=ansatz, optimizer=optimizer, sampler=sampler)
    log_print("✅ VQC model created")

    # VQC draw circuit (commented out like original)
    # vqc.circuit.decompose().draw(output='mpl')

    # Train Model with tqdm Progress Bar (exact copy from original)
    import time

    start_time = time.time()
    log_print("Training VQC Model...")
    
    try:
        vqc.fit(X_train, y_train)
        end_time = time.time()
        training_time = end_time - start_time
        log_print(f"Training time: {training_time:.2f} seconds")
    except Exception as e:
        log_print(f"❌ Training failed: {str(e)}")
        return {"error": f"Training failed: {str(e)}"}

    # Predict (exact copy from original)
    log_print("🔮 Making predictions...")
    y_pred = vqc.predict(X_test)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test)
    log_print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Confusion Matrix (exact copy from original)
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(f'/results/{base_dir}/confusion_matrix.png')
    log_print(f"✅ Confusion matrix saved to /results/{base_dir}/confusion_matrix.png")
    plt.close()

    # Classification Report (exact copy from original)
    report = classification_report(y_test, y_pred, output_dict=True)
    with open(f'/results/{base_dir}/classification_report.json', "w") as f:
        json.dump(report, f)
        log_print(f"Classification report saved to /results/{base_dir}/classification_report.json")

    log_print("\n📋 CLASSIFICATION REPORT:")
    log_print("="*60)
    log_print(classification_report(y_test, y_pred))
    log_print("="*60)
    
    # Confusion Matrix Details
    log_print("\n📊 CONFUSION MATRIX:")
    log_print("="*40)
    log_print("Predicted ->")
    log_print("Actual   0   1")
    log_print("  0    {:3d} {:3d}".format(conf_matrix[0,0], conf_matrix[0,1]))
    log_print("  1    {:3d} {:3d}".format(conf_matrix[1,0], conf_matrix[1,1]))
    log_print("="*40)

    # Save results
    results = {
        'accuracy': float(accuracy),
        'training_time': float(training_time),
        'num_samples': len(df),
        'num_qubits': num_qubits,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'log_filename': log_filename,
        'gpu_type': gpu_type,
        'base_dir': base_dir,
        'classification_report': report,
        'confusion_matrix': conf_matrix.tolist(),
        'class_distribution': {
            'train': np.bincount(y_train).tolist(),
            'test': np.bincount(y_test).tolist()
        }
    }
    
    # Save main results JSON
    with open(f'/results/{base_dir}/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    log_print(f"✅ Results saved to /results/{base_dir}/results.json")
    
    # Final log entry
    final_log_entry = f"\nTraining completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n" + "="*80 + "\n"
    
    try:
        with open(modal_log_path, 'a', encoding='utf-8') as f:
            f.write(final_log_entry)
        log_print(f"✅ Final log entry written to Modal: {modal_log_path}")
    except Exception as e:
        log_print(f"⚠️ Could not write final entry to Modal log: {e}")
    
    try:
        with open(local_log_path, 'a', encoding='utf-8') as f:
            f.write(final_log_entry)
        log_print(f"✅ Final log entry written to local: {local_log_path}")
    except Exception as e:
        log_print(f"⚠️ Could not write final entry to local log: {e}")
    
    log_print(f"✅ VQC Training completed!")
    log_print(f"Final Accuracy: {accuracy:.4f}")
    log_print(f"Training Time: {training_time:.2f}s")
    log_print(f"Samples processed: {len(df)} total, {len(X_train)} train, {len(X_test)} test")
    log_print(f"📝 Training log saved to: {log_filename}")
    log_print(f"📁 Results saved to: /results/{base_dir}/")
    
    return results

@app.function(
    cpu=2,
    volumes={"/results": volume}
)
def download_results() -> Dict[str, Any]:
    """Download hasil training dari Modal volume."""
    
    import json
    import os
    import base64
    import glob
    from pathlib import Path
    
    print("📥 Preparing results for download...")
    
    files_data = {}
    results_info = {}
    
    # Files to download from base directory
    base_dir = 'QuantumEpitopeCovid19Logs'
    files_to_download = [
        f'{base_dir}/results.json',
        f'{base_dir}/classification_report.json',
        f'{base_dir}/confusion_matrix.png'
    ]
    
    # Add log files - check both root and subdirectory
    log_files_root = glob.glob('/results/*.txt')
    log_files_subdir = glob.glob(f'/results/{base_dir}/*.txt')
    all_log_files = log_files_root + log_files_subdir
    
    for log_file in all_log_files:
        log_filename = os.path.basename(log_file)
        # Avoid duplicate entries
        if log_filename not in files_to_download:
            files_to_download.append(log_filename)
    
    print(f"📋 Files to check for download:")
    for filename in files_to_download:
        print(f"  - {filename}")
    
    # List all files in /results directory for debugging
    print(f"\n📁 Contents of /results directory:")
    try:
        all_files = []
        for root, dirs, files in os.walk('/results'):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, '/results')
                all_files.append(rel_path)
                print(f"  - {rel_path}")
    except Exception as e:
        print(f"  ❌ Error listing directory: {e}")
    
    # Read and encode files
    for filename in files_to_download:
        modal_file_path = f'/results/{filename}'
        
        if os.path.exists(modal_file_path):
            try:
                with open(modal_file_path, 'rb') as f:
                    file_content = f.read()
                    files_data[filename] = base64.b64encode(file_content).decode('utf-8')
                    print(f"  ✅ Prepared: {filename} ({len(file_content)} bytes)")
                    
                    # Extract info from JSON files
                    if filename.endswith('.json'):
                        try:
                            json_data = json.loads(file_content.decode('utf-8'))
                            if 'accuracy' in json_data:
                                results_info['accuracy'] = json_data['accuracy']
                                results_info['training_time'] = json_data['training_time']
                        except Exception as e:
                            print(f"    ⚠️ Could not parse JSON info from {filename}: {e}")
                            
            except Exception as e:
                print(f"  ❌ Failed to prepare {filename}: {str(e)}")
        else:
            print(f"  ⚠️ File not found: {modal_file_path}")
    
    download_summary = {
        'files_data': files_data,
        'results_info': results_info,
        'total_files': len(files_data)
    }
    
    print(f"\n📦 Prepared {len(files_data)} files for download")
    
    # Count log files
    log_file_count = sum(1 for filename in files_data.keys() if filename.endswith('.txt'))
    if log_file_count > 0:
        print(f"📝 Including {log_file_count} log file(s) in download")
    
    return download_summary

def save_files_locally(files_data: Dict[str, str], local_dir: str = "./results") -> Dict[str, Any]:
    """Save base64 encoded files to local directory (runs locally, not on Modal)."""
    
    import json
    import os
    import base64
    from pathlib import Path
    
    print(f"💾 Saving files to local directory: {local_dir}")
    print(f"📦 Received {len(files_data)} files to save")
    
    # Create local results directory
    local_path = Path(local_dir)
    local_path.mkdir(exist_ok=True)
    
    saved_files = []
    failed_files = []
    
    for filename, encoded_content in files_data.items():
        print(f"  📁 Processing: {filename}")
        
        # Handle subdirectory structure
        local_file_path = local_path / filename
        
        # Create subdirectory if needed
        try:
            local_file_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"    📂 Directory created: {local_file_path.parent}")
        except Exception as e:
            print(f"    ⚠️ Directory creation warning: {e}")
        
        try:
            # Decode base64 and save file
            file_content = base64.b64decode(encoded_content)
            with open(local_file_path, 'wb') as f:
                f.write(file_content)
            saved_files.append(filename)
            print(f"    ✅ Saved: {filename} ({len(file_content)} bytes)")
            
        except Exception as e:
            failed_files.append(filename)
            print(f"    ❌ Failed to save {filename}: {str(e)}")
    
    save_summary = {
        'local_directory': str(local_path.absolute()),
        'saved_files': saved_files,
        'failed_files': failed_files,
        'total_files': len(saved_files),
        'total_failed': len(failed_files)
    }
    
    print(f"\n✅ Successfully saved {len(saved_files)} files to {local_path.absolute()}")
    if failed_files:
        print(f"❌ Failed to save {len(failed_files)} files: {failed_files}")
    
    return save_summary

# Main entrypoint - no parameters needed
@app.local_entrypoint()
def main():
    """Main entrypoint untuk menjalankan VQC training."""
    
    print("🚀 COVID-19 Epitope Prediction VQC - Modal Deployment (Exact Copy Structure)")
    print("="*70)
    print("Starting training...")
    
    # Run training
    result = train_vqc_exact.remote()
    
    if 'error' in result:
        print(f"❌ Training failed: {result['error']}")
        return
    
    print(f"✅ Training completed with accuracy: {result['accuracy']:.4f}")
    print(f"⏱️ Training time: {result['training_time']:.2f} seconds")
    
    # Download results
    print("\n📥 Downloading results...")
    try:
        download_summary = download_results.remote()
        print(f"📦 Download summary: {download_summary.get('total_files', 0)} files found")
        
        # Save files locally (run locally, not on Modal)
        if download_summary.get('files_data'):
            save_summary = save_files_locally(download_summary['files_data'])
            
            print(f"\n✅ Results downloaded to: {save_summary['local_directory']}")
            print(f"✅ Files successfully saved: {', '.join(save_summary.get('saved_files', []))}")
            
            if save_summary.get('failed_files'):
                print(f"❌ Files failed to save: {', '.join(save_summary['failed_files'])}")
            
            if download_summary.get('results_info'):
                print(f"\n📊 Training Results:")
                print(f"  Accuracy: {download_summary['results_info'].get('accuracy', 'N/A'):.4f}")
                print(f"  Training Time: {download_summary['results_info'].get('training_time', 'N/A'):.2f}s")
        else:
            print("⚠️ No files available for download")
            
    except Exception as e:
        print(f"❌ Download failed: {str(e)}")
        print("💡 You can manually download files from Modal volume if needed")

if __name__ == "__main__":
    print("COVID-19 Epitope Prediction VQC - Modal Deployment (Exact Copy Structure)")
    print("Dataset will be downloaded from GitHub automatically")
    print("Results will be saved to ./results/QuantumEpitopeCovid19Logs directory")
    print("")
    print("🚀 VQC FEATURES:")
    print("  ✅ Variational Quantum Classifier (VQC) from qiskit-machine-learning")
    print("  ✅ B200 GPU optimization")
    print("  ✅ Automatic dataset download")
    print("  ✅ Balanced data resampling (500 samples per class)")
    print("  ✅ Comprehensive logging")
    print("  ✅ Auto-download results")
    print("  ✅ Exact structure from original copy.py")
    print("")
    print("📁 Files that will be saved locally:")
    print("  - results/QuantumEpitopeCovid19Logs/results.json")
    print("  - results/QuantumEpitopeCovid19Logs/classification_report.json")
    print("  - results/QuantumEpitopeCovid19Logs/confusion_matrix.png")
    print("  - YYYYMMDD_HHMMSS_XXX.txt (training log file)")
    print("")
    print("To run: python ModalVQC_ExactCopy.py")

    main()

