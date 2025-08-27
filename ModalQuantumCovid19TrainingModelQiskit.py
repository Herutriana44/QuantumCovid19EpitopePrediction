#!/usr/bin/env python3
"""
COVID-19 Epitope Prediction VQC - Modal Deployment
Quantum Machine Learning untuk prediksi epitope COVID-19 dengan GPU acceleration di Modal platform.
Menggunakan VQC (Variational Quantum Classifier) dari qiskit-machine-learning.
"""

import modal
import numpy as np
from typing import Dict, Any, Optional
import json
import os

# Define Modal image dengan semua dependencies
image = (
    modal.Image.debian_slim()
    .pip_install([
        "qiskit",
        "qiskit-machine-learning", 
        # "qiskit-aer",
        "qiskit-aer-gpu",
        "qiskit_ibm_runtime",
        "qiskit-algorithms",
        "torch",
        "numpy",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "pandas",
        "seaborn",
        "tqdm",
        "joblib",
        "requests"  # For downloading dataset from GitHub
    ])
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")  # For matplotlib backend
)

# Create Modal app
app = modal.App("vqc-quantum-ml", image=image)

# Shared volume untuk menyimpan results
volume = modal.Volume.from_name("vqc-results", create_if_missing=True)

@app.function(
    gpu=["B200:8"],  # Multi-GPU B200 priority for maximum performance
    timeout=3600*4,  # 4 hour timeout for maximum training
    volumes={"/results": volume},
    cpu=32,  # Maximum CPU cores for multi-GPU support
    memory=65536,  # 64GB memory for multi-GPU training
    # secrets=[modal.Secret.from_name("wandb-secret", required=False)]  # Optional W&B logging
)
def train_vqc_gpu(
    dataset_url: str = "https://raw.githubusercontent.com/Herutriana44/QuantumCovid19EpitopePrediction/main/clean_after_balanced_amino_data.csv",
    num_qubits: Optional[int] = None,  # Will be determined from data features
    num_epochs: int = 2,  # Default to 2 epochs for quick training
    test_size: float = 0.3,  # Reduced for more training data
    sampler_type: str = "aer",
    use_gpu: bool = True,
    gpu_type: str = "B200:8"  # Default to B200 for maximum performance
) -> Dict[str, Any]:
    """
    Train VQC model untuk COVID-19 Epitope Prediction dengan GPU acceleration di Modal.
    
    Args:
        dataset_url: URL ke dataset CSV file di GitHub
        num_qubits: Jumlah qubits untuk VQC (auto-detected from features jika None)
        num_epochs: Number of training epochs (default: 2)
        test_size: Proporsi data untuk testing
        sampler_type: Type sampler ('aer', 'statevector')
        use_gpu: Gunakan GPU acceleration
        gpu_type: Type GPU untuk optimization
        
    Returns:
        Dictionary dengan results dan metrics
    """
    
    # Import semua dependencies di dalam function
    import pandas as pd
    import time
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import requests
    from tqdm import tqdm
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.distributed as dist
    import torch.multiprocessing as mp
    from torch.utils.data.distributed import DistributedSampler
    
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
    from qiskit.primitives import Sampler, StatevectorSampler
    
    from qiskit_machine_learning.neural_networks import SamplerQNN
    from qiskit_machine_learning.connectors import TorchConnector
    
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
    
    # Setup Multi-GPU Training
    def setup_multi_gpu():
        """Setup multi-GPU training dengan PyTorch DDP"""
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print(f"🚀 Multi-GPU Setup: {num_gpus} GPUs detected")
            
            for i in range(num_gpus):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
            
            # Setup distributed training environment
            if num_gpus > 1:
                os.environ['MASTER_ADDR'] = 'localhost'
                os.environ['MASTER_PORT'] = '12355'
                os.environ['WORLD_SIZE'] = str(num_gpus)
                
                # Initialize process group for DDP
                try:
                    if not dist.is_initialized():
                        dist.init_process_group(backend='nccl', rank=0, world_size=num_gpus)
                    print(f"✅ Distributed training initialized for {num_gpus} GPUs")
                except Exception as e:
                    print(f"⚠️ DDP initialization failed: {e}, using DataParallel instead")
                    return False, num_gpus
                
                return True, num_gpus
            else:
                print("ℹ️ Single GPU training")
                return False, 1
        else:
            print("⚠️ No GPU available, using CPU")
            return False, 0
    
    # Check GPU availability and setup multi-GPU
    print(f"🚀 Starting VQC training for COVID-19 Epitope Prediction on Modal")
    use_ddp, num_gpus = setup_multi_gpu()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Primary device: {device}")
    
    # Setup Aer sampler dengan GPU (dengan Colab compatibility)
    def setup_sampler_safe(sampler_type: str, use_gpu: bool, gpu_type: str):
        """Setup sampler dengan error handling untuk different environments."""
        
        if sampler_type == 'aer':
            try:
                from qiskit_aer.primitives import Sampler as AerSampler
                sampler = AerSampler()
                
                # Test GPU availability untuk Aer
                if use_gpu and torch.cuda.is_available():
                    try:
                        # Test GPU configuration dengan simple circuit
                        from qiskit import QuantumCircuit
                        test_qc = QuantumCircuit(1)
                        test_qc.h(0)
                        test_qc.measure_all()
                        
                        # Try GPU configuration
                        test_sampler = AerSampler()
                        test_sampler.set_options(device='GPU')
                        
                        # Test run to verify GPU works
                        test_job = test_sampler.run([test_qc], shots=10)
                        test_result = test_job.result()
                        
                        # If test successful, apply GPU settings based on architecture
                        if gpu_type == "B200" or gpu_type == "B200:8":
                            # NVIDIA Blackwell B200 - Ultimate performance dengan 8-GPU optimization
                            sampler.set_options(
                                device='GPU',
                                max_parallel_threads=256,        # Increased untuk 8-GPU setup
                                max_parallel_experiments=128,    # Enhanced batch processing
                                max_parallel_shots=1000000,      # Large shot counts untuk parallel execution
                                blocking_enable=True,
                                blocking_qubits=12,               # Optimal blocking untuk complex circuits
                                precision='single',               # Optimize untuk speed
                                batched_optimization=True,       # Enable batch optimization
                                memory_mb=8192                    # 8GB memory per GPU
                            )
                            print("✓ Aer GPU B200:8 (Blackwell Multi-GPU) configuration applied")
                            # if torch.cuda.is_available():
                            print(f"GPU Device: {torch.cuda.get_device_name(0)}")
                            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
                        
                        elif gpu_type == "H200":
                            # NVIDIA Hopper H200 - Enhanced memory
                            sampler.set_options(
                                device='GPU',
                                max_parallel_threads=48,
                                max_parallel_experiments=24,
                                blocking_enable=True,
                                precision='single'
                            )
                            print("✓ Aer GPU H200 configuration applied")
                        elif gpu_type == "H100":
                            # NVIDIA Hopper H100 - High performance
                            sampler.set_options(
                                device='GPU', 
                                max_parallel_threads=32,
                                max_parallel_experiments=16,
                                blocking_enable=True
                            )
                            print("✓ Aer GPU H100 configuration applied")
                        elif gpu_type in ["A100", "A100-80GB"]:
                            # NVIDIA Ampere A100 - Balanced performance
                            sampler.set_options(
                                device='GPU',
                                max_parallel_threads=16,
                                max_parallel_experiments=8,
                                blocking_enable=True
                            )
                            print("✓ Aer GPU A100 configuration applied")
                        elif gpu_type == "L40S":
                            # NVIDIA Ada Lovelace L40S - Optimized for inference
                            sampler.set_options(
                                device='GPU',
                                max_parallel_threads=24,
                                max_parallel_experiments=12,
                                blocking_enable=True
                            )
                            print("✓ Aer GPU L40S configuration applied")
                        else:
                            sampler.set_options(device='GPU')
                            print(f"✓ Aer GPU {gpu_type} default configuration applied")
                            
                    except Exception as gpu_error:
                        print(f"⚠ GPU configuration failed: {str(gpu_error)}")
                        print("  → Falling back to Aer CPU")
                        sampler = AerSampler()  # Reset to CPU-only
                        sampler.set_options(device='CPU')
                        print("✓ Aer CPU configuration applied")
                else:
                    print("✓ Aer CPU configuration")
                    
                return sampler
                
            except ImportError:
                print("⚠ Qiskit Aer not available, using StatevectorSampler")
                return StatevectorSampler()
                
        elif sampler_type == 'statevector':
            return StatevectorSampler()
        else:
            return Sampler()
    
    try:
        sampler = setup_sampler_safe(sampler_type, use_gpu, gpu_type)
        print(f"✓ Sampler setup successful: {type(sampler).__name__}")
            
    except Exception as e:
        print(f"⚠ Sampler setup failed: {str(e)}")
        print("  → Using default Qiskit sampler")
        sampler = Sampler()
        print("✓ Default sampler configured")
    
    # Download and prepare COVID-19 epitope dataset
    print(f"📊 Downloading COVID-19 epitope dataset from GitHub...")
    print(f"URL: {dataset_url}")
    
    try:
        # Download dataset from GitHub
        response = requests.get(dataset_url)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Save to temporary file
        temp_file = "/tmp/clean_after_balanced_amino_data.csv"
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        # Load dataset from temporary file
        df = pd.read_csv(temp_file)
        df = df.dropna()
        df = df.reset_index(drop=True)
        
        print(f"✅ Successfully downloaded and loaded {len(df)} samples from dataset")
        
        # Clean up temporary file
        os.remove(temp_file)
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to download dataset from {dataset_url}: {str(e)}")
    except Exception as e:
        raise Exception(f"Failed to process dataset: {str(e)}")

    #@title Resample data based on 'label' column
    def resample_by_label(df, label_col, target_count_per_label):
        """
        Resamples the DataFrame to have a specified number of samples for each label.

        Args:
            df (pd.DataFrame): The input DataFrame.
            label_col (str): The name of the column containing the labels.
            target_count_per_label (int): The desired number of samples for each label.

        Returns:
            pd.DataFrame: The resampled DataFrame.
        """
        resampled_df = pd.DataFrame()
        for label_value in df[label_col].unique():
            label_df = df[df[label_col] == label_value]
            # Use replace=True for oversampling if needed, or adjust if target_count_per_label > len(label_df)
            resampled_label_df = label_df.sample(n=target_count_per_label, replace=False, random_state=42)
            resampled_df = pd.concat([resampled_df, resampled_label_df])

        # Shuffle the resampled data to mix the labels
        resampled_df = resampled_df.sample(frac=1, random_state=42).reset_index(drop=True)
        return resampled_df

    # Resample the DataFrame to have 50 'E' and 50 '.' labels
    target_samples_per_label = 500
    df_resampled = resample_by_label(df, 'label', target_samples_per_label)

    print(f"Original DataFrame length: {len(df)}")
    print(f"Resampled DataFrame length: {len(df_resampled)}")
    print("Value counts in resampled DataFrame:")
    print(df_resampled['label'].value_counts())

    # Update the original dataframe reference to the resampled one for subsequent steps
    df = df_resampled
    
    # Label conversion function
    def label_to_numeric(label_str):
        return 1 if label_str == 'E' else 0
    
    df['label_numeric'] = df['label'].apply(label_to_numeric)
    
    # Convert data types
    df['length_sequence'] = df['length_sequence'].astype(int)
    df['Position'] = df['Position'].astype(int)
    
    # Define features and labels
    features = ['Position', 'length_sequence', 'numerical_amino_acid']
    label_col = 'label_numeric'
    
    X = df[features].values
    y = df[label_col].values
    
    print(f"Feature shape: {X.shape}")
    print(f"Label distribution: {np.bincount(y)}")
    
    # Determine number of qubits from features if not specified
    if num_qubits is None:
        num_qubits = X.shape[1]
        print(f"Auto-detected num_qubits: {num_qubits}")
    
    # Split data (following original approach with shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    
    # Convert to PyTorch tensors (without scaling)
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Setup DataLoaders dengan multi-GPU optimization
    # Batch size scaling untuk multi-GPU: base_batch_size * num_gpus
    base_batch_size = 64  # Base batch size per GPU
    effective_batch_size = base_batch_size * max(1, num_gpus)
    print(f"Batch size: {base_batch_size} per GPU, effective: {effective_batch_size}")
    
    if use_ddp and num_gpus > 1:
        # Distributed training dengan DistributedSampler
        train_sampler = DistributedSampler(train_dataset, num_replicas=num_gpus, rank=0)
        test_sampler = DistributedSampler(test_dataset, num_replicas=num_gpus, rank=0, shuffle=False)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=base_batch_size,  # Per GPU batch size
            sampler=train_sampler,
            num_workers=4,  # Optimal untuk B200
            pin_memory=True,
            persistent_workers=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=base_batch_size,  # Per GPU batch size
            sampler=test_sampler,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
        print(f"✅ DistributedDataLoader setup for {num_gpus} GPUs")
    else:
        # Single GPU atau DataParallel
        train_loader = DataLoader(
            train_dataset, 
            batch_size=effective_batch_size, 
            shuffle=True,
            num_workers=8,  # More workers for single/DataParallel mode
            pin_memory=True,
            persistent_workers=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=effective_batch_size, 
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True
        )
        print(f"✅ Standard DataLoader setup with batch size: {effective_batch_size}")

    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Create VQC Model with PyTorch
    print("🔬 Creating VQC model...")
    
    class COVID19VQC(nn.Module):
        def __init__(self, input_dim: int, num_qubits: int, sampler, num_gpus: int = 1):
            super().__init__()
            
            self.num_gpus = num_gpus
            
            # Classical preprocessing dengan batch normalization untuk stability
            self.classical_input = nn.Linear(input_dim, num_qubits)
            self.batch_norm = nn.BatchNorm1d(num_qubits)
            self.activation = nn.ReLU()
            self.dropout = nn.Dropout(0.1)  # Small dropout untuk regularization
            
            # Quantum layer - optimized untuk multi-GPU
            feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=2, entanglement="full")
            ansatz = RealAmplitudes(num_qubits=num_qubits, entanglement='full', reps=3)
            
            qc = QuantumCircuit(num_qubits)
            qc.compose(feature_map, inplace=True)
            qc.compose(ansatz, inplace=True)
            
            # Optimize quantum execution untuk multi-GPU
            qnn = SamplerQNN(
                sampler=sampler,
                circuit=qc,
                input_params=feature_map.parameters,
                weight_params=ansatz.parameters,
                interpret=lambda x: x % 2,
                output_shape=2,
                input_gradients=True
            )
            
            # Improved weight initialization untuk better convergence
            initial_weights = np.random.normal(0, 0.05, qnn.num_weights)  # Smaller variance
            self.quantum_layer = TorchConnector(qnn, initial_weights)
            
            # Enhanced classical output dengan residual connection
            self.output_layer = nn.Sequential(
                nn.Linear(2, 8),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(8, 2)
            )
            
        def forward(self, x):
            # Classical preprocessing
            x_classical = self.classical_input(x)
            if x_classical.size(0) > 1:  # Batch norm only if batch size > 1
                x_classical = self.batch_norm(x_classical)
            x_classical = self.activation(x_classical)
            x_classical = self.dropout(x_classical)
            
            # Quantum processing
            x_quantum = self.quantum_layer(x_classical)
            
            # Classical output
            output = self.output_layer(x_quantum)
            
            return output
    
    # Initialize model dengan multi-GPU support
    model = COVID19VQC(X.shape[1], num_qubits, sampler, num_gpus=num_gpus)
    
    # Multi-GPU model setup
    if torch.cuda.is_available():
        model = model.to(device)
        
        if use_ddp and num_gpus > 1:
            # DistributedDataParallel untuk optimal multi-GPU training
            model = DDP(model, device_ids=[0], find_unused_parameters=True)
            print(f"✅ DistributedDataParallel setup with {num_gpus} GPUs")
        elif num_gpus > 1:
            # DataParallel sebagai fallback
            model = nn.DataParallel(model)
            print(f"✅ DataParallel setup with {num_gpus} GPUs")
        else:
            print(f"✅ Single GPU setup")
    
    # Optimizer dengan learning rate scaling untuk multi-GPU
    base_lr = 0.01
    scaled_lr = base_lr * max(1, num_gpus)  # Scale learning rate dengan number of GPUs
    optimizer = optim.AdamW(  # AdamW untuk better regularization
        model.parameters(), 
        lr=scaled_lr,
        weight_decay=1e-4,  # L2 regularization
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler untuk adaptive learning
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,  # Restart every 10 epochs
        T_mult=2,
        eta_min=1e-6
    )
    
    # Loss function dengan label smoothing untuk better generalization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Gradient scaler untuk mixed precision training (optional optimization)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    print(f"VQC created with {num_qubits} qubits")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Training loop
    print(f"🚀 Starting training for {num_epochs} epochs...")
    
    training_history = {'loss': [], 'accuracy': [], 'epoch_times': []}
    start_time = time.time()
    
    # Use tqdm for epoch progress
    for epoch in tqdm(range(num_epochs), desc="Training Epochs", unit="epoch"):
        epoch_start = time.time()
        
        # Training
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", leave=False)
        for batch_idx, (data, target) in enumerate(train_pbar):
            # Move data to device
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Mixed precision training untuk memory efficiency
            if scaler and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    output = model(data)
                    loss = criterion(output, target)
                
                # Scaled backward pass
                scaler.scale(loss).backward()
                
                # Gradient clipping dengan scaling
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimizer step dengan scaling
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
            
            # Synchronize gradients untuk DDP
            if use_ddp and num_gpus > 1:
                torch.distributed.barrier()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Update progress bar with current metrics
            current_acc = 100. * correct / total if total > 0 else 0
            current_lr = optimizer.param_groups[0]['lr']
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%',
                'LR': f'{current_lr:.6f}'
            })
            
            # Memory cleanup untuk large batches
            if batch_idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        train_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        
        with torch.no_grad():
            val_pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation", leave=False)
            for data, target in val_pbar:
                # Move data to device
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                
                # Mixed precision untuk validation
                if scaler and torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        output = model(data)
                        loss = criterion(output, target)
                else:
                    output = model(data)
                    loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
                
                # Update validation progress bar
                current_val_acc = 100. * val_correct / val_total if val_total > 0 else 0
                val_pbar.set_postfix({
                    'Val Loss': f'{loss.item():.4f}',
                    'Val Acc': f'{current_val_acc:.2f}%'
                })
        
        val_loss = val_loss / len(test_loader)
        val_acc = 100. * val_correct / val_total
        
        epoch_time = time.time() - epoch_start
        
        training_history['loss'].append((train_loss, val_loss))
        training_history['accuracy'].append((train_acc, val_acc))
        training_history['epoch_times'].append(epoch_time)
        
        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1:2d}/{num_epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
              f"Time: {epoch_time:.1f}s | LR: {current_lr:.6f}")
        
        # Memory cleanup setelah each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Update distributed sampler epoch untuk shuffling
        if use_ddp and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
    
    total_training_time = time.time() - start_time
    
    # Final evaluation
    print("🔮 Making predictions...")
    model.eval()
    final_predictions = []
    final_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            # Move data to device
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            # Mixed precision untuk final evaluation
            if scaler and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    output = model(data)
            else:
                output = model(data)
                
            _, predicted = torch.max(output.data, 1)
            final_predictions.extend(predicted.cpu().numpy())
            final_targets.extend(target.cpu().numpy())
    
    final_accuracy = accuracy_score(final_targets, final_predictions)
    
    print(f"Final Accuracy: {final_accuracy:.4f}")
    
    # Create training plots and confusion matrix
    print("📈 Creating training plots and confusion matrix...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    epochs = range(1, len(training_history['loss']) + 1)
    train_losses = [x[0] for x in training_history['loss']]
    val_losses = [x[1] for x in training_history['loss']]
    train_accs = [x[0] for x in training_history['accuracy']]
    val_accs = [x[1] for x in training_history['accuracy']]
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Val Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(final_targets, final_predictions)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax3)
    ax3.set_xlabel('Predicted Label')
    ax3.set_ylabel('True Label')
    ax3.set_title('Confusion Matrix - COVID-19 Epitope Prediction')
    
    # Epoch times
    ax4.plot(epochs, training_history['epoch_times'], 'g-', linewidth=2)
    ax4.set_title('Training Time per Epoch')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Time (s)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot to Modal volume
    plt.savefig('/results/modal_vqc_training_results.png', dpi=150, bbox_inches='tight')
    print("✅ Training results saved to /results/modal_vqc_training_results.png")
    
    # Also save plot to current directory for potential access
    try:
        plt.savefig('modal_vqc_training_results.png', dpi=150, bbox_inches='tight')
        print("✅ Training results also saved to container: modal_vqc_training_results.png")
    except Exception as e:
        print(f"⚠️ Could not save to container directory: {str(e)}")
    
    plt.close()
    
    # Save detailed results dengan multi-GPU information
    results = {
        'final_accuracy': float(final_accuracy),
        'training_time': total_training_time,
        'num_epochs': num_epochs,
        'num_samples': len(df),
        'num_qubits': num_qubits,
        'test_size': test_size,
        'sampler_type': sampler_type,
        'gpu_type': gpu_type,
        'device': device,
        'num_train_samples': len(X_train),
        'num_test_samples': len(X_test),
        'model_parameters': sum(p.numel() for p in model.parameters()),
        # Multi-GPU training information
        'multi_gpu_info': {
            'num_gpus': num_gpus,
            'use_ddp': use_ddp,
            'base_batch_size': base_batch_size,
            'effective_batch_size': effective_batch_size,
            'base_lr': base_lr,
            'scaled_lr': scaled_lr,
            'parallel_training': 'DistributedDataParallel' if use_ddp else ('DataParallel' if num_gpus > 1 else 'Single GPU'),
            'memory_optimization': 'Mixed Precision + Auto Cleanup',
            'quantum_optimization': 'B200 Multi-GPU Sampler Configuration'
        },
        'training_history': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accs,
            'val_accuracies': val_accs,
            'epoch_times': training_history['epoch_times']
        },
        'performance_metrics': {
            'avg_epoch_time': float(np.mean(training_history['epoch_times'])),
            'total_training_time': total_training_time,
            'samples_per_second': len(X_train) * num_epochs / total_training_time,
            'gpu_efficiency': f"{effective_batch_size}/{num_gpus} samples per GPU per batch"
        },
        'classification_report': classification_report(final_targets, final_predictions, output_dict=True),
        'confusion_matrix': conf_matrix.tolist()
    }
    
    # Save results to Modal volume
    with open('/results/modal_vqc_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Also save results to current directory for potential access
    try:
        with open('modal_vqc_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("✅ Training results also saved to container: modal_vqc_results.json")
    except Exception as e:
        print(f"⚠️ Could not save JSON to container directory: {str(e)}")
    
    # Save trained model state
    try:
        # Extract model state dari DDP/DataParallel wrapper
        if hasattr(model, 'module'):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
            
        # Save to Modal volume
        model_data = {
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'final_accuracy': final_accuracy,
            'num_qubits': num_qubits,
            'num_gpus': num_gpus,
            'use_ddp': use_ddp,
            'training_params': {
                'num_epochs': num_epochs,
                'test_size': test_size,
                'sampler_type': sampler_type,
                'gpu_type': gpu_type,
                'base_batch_size': base_batch_size,
                'effective_batch_size': effective_batch_size,
                'scaled_lr': scaled_lr
            }
        }
        
        torch.save(model_data, '/results/modal_vqc_model.pth')
        print("✅ Trained model saved to /results/modal_vqc_model.pth")
        
        # Also save to current directory for potential access
        torch.save(model_data, 'modal_vqc_model.pth')
        print("✅ Trained model also saved to container: modal_vqc_model.pth")
        
    except Exception as e:
        print(f"⚠️ Could not save trained model: {str(e)}")
    
    # Cleanup distributed training
    if use_ddp and dist.is_initialized():
        dist.destroy_process_group()
        print("✅ Distributed training cleaned up")
    
    # Final memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    print(f"✅ VQC Training completed!")
    print(f"Final Accuracy: {final_accuracy:.4f}")
    print(f"Training Time: {total_training_time:.1f}s")
    print(f"Average Epoch Time: {np.mean(training_history['epoch_times']):.1f}s")
    print(f"Multi-GPU Setup: {num_gpus} GPUs, DDP: {use_ddp}")
    print(f"Batch Size: {base_batch_size} per GPU, Effective: {effective_batch_size}")
    print(f"Learning Rate: {scaled_lr:.6f} (scaled from {base_lr})")
    print(f"Samples processed: {len(df)} total, {len(X_train)} train, {len(X_test)} test")
    print(f"Classification Report:")
    print(classification_report(final_targets, final_predictions))
    
    return results


@app.function(
    gpu=["B200:8"],  # B200 with multi-GPU support
    timeout=3600*4,  # 4 hours for comprehensive benchmark with maximum parameters
    volumes={"/results": volume},
    cpu=32,  # Maximum CPU cores untuk multi-GPU
    memory=65536  # 64GB memory untuk multi-GPU benchmark
)
def benchmark_samplers() -> Dict[str, Any]:
    """
    Benchmark berbagai sampler configurations di Modal.
    """
    
    import torch
    import time
    from sklearn.datasets import make_circles
    
    print("🔥 Starting comprehensive sampler benchmark on Modal...")
    
    # Test configurations including B200:8 multi-GPU
    configs = [
        {'sampler_type': 'aer', 'use_gpu': False, 'name': 'Aer CPU'},
        {'sampler_type': 'aer', 'use_gpu': True, 'gpu_type': 'B200:8', 'name': 'Aer GPU B200:8 (Multi-GPU)'},
        {'sampler_type': 'aer', 'use_gpu': True, 'gpu_type': 'B200', 'name': 'Aer GPU B200 (Single)'},
        {'sampler_type': 'aer', 'use_gpu': True, 'gpu_type': 'H200', 'name': 'Aer GPU H200'},
        {'sampler_type': 'aer', 'use_gpu': True, 'gpu_type': 'H100', 'name': 'Aer GPU H100'},
        {'sampler_type': 'aer', 'use_gpu': True, 'gpu_type': 'A100', 'name': 'Aer GPU A100'},
        {'sampler_type': 'aer', 'use_gpu': True, 'gpu_type': 'L40S', 'name': 'Aer GPU L40S'},
        {'sampler_type': 'statevector', 'use_gpu': False, 'name': 'StatevectorSampler'},
    ]
    
    benchmark_results = {}
    
    for config in configs:
        print(f"\n🧪 Testing {config['name']}...")
        
        try:
            start_time = time.time()
            
            # Run training with specific config
            result = train_vqc_gpu(
                dataset_url="https://raw.githubusercontent.com/Herutriana44/QuantumCovid19EpitopePrediction/main/clean_after_balanced_amino_data.csv",
                num_qubits=3,  # Fixed for benchmark
                num_epochs=1,  # Quick benchmark with 1 epoch
                test_size=0.3,
                sampler_type=config['sampler_type'],
                use_gpu=config.get('use_gpu', False),
                gpu_type=config.get('gpu_type', 'B200:8')
            )
            
            benchmark_time = time.time() - start_time
            
            benchmark_results[config['name']] = {
                'accuracy': result['final_accuracy'],
                'training_time': result['training_time'],
                'benchmark_time': benchmark_time,
                'success': True,
                'config': config
            }
            
            print(f"  ✅ {config['name']}: {result['final_accuracy']:.3f} accuracy in {benchmark_time:.1f}s")
            
        except Exception as e:
            print(f"  ❌ {config['name']}: Failed - {str(e)}")
            benchmark_results[config['name']] = {
                'accuracy': 0,
                'training_time': 0,
                'benchmark_time': 0,
                'success': False,
                'error': str(e),
                'config': config
            }
    
    # Save benchmark results
    with open('/results/modal_sampler_benchmark.json', 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("MODAL SAMPLER BENCHMARK RESULTS")
    print("="*60)
    
    successful_results = {k: v for k, v in benchmark_results.items() if v['success']}
    
    if successful_results:
        best_sampler = max(successful_results.items(), key=lambda x: x[1]['accuracy'])
        fastest_sampler = min(successful_results.items(), key=lambda x: x[1]['training_time'])
        
        print(f"Best Accuracy: {best_sampler[0]} ({best_sampler[1]['accuracy']:.3f})")
        print(f"Fastest Training: {fastest_sampler[0]} ({fastest_sampler[1]['training_time']:.1f}s)")
    
    return benchmark_results


@app.function(
    cpu=2,
    volumes={"/results": volume}
)
def download_results_to_local(local_dir: str = "./results") -> Dict[str, Any]:
    """
    Download hasil training dari Modal volume ke local storage menggunakan base64 encoding.
    """
    
    import json
    import os
    import base64
    from pathlib import Path
    
    print(f"📥 Preparing results for download to local directory: {local_dir}")
    
    files_data = {}
    results_info = {}
    
    # Files to download from Modal volume
    files_to_download = [
        'modal_vqc_training_results.png',
        'modal_vqc_results.json',
        'modal_vqc_model.pth',
        'modal_sampler_benchmark.json'
    ]
    
    # Read files from Modal volume and encode as base64
    for filename in files_to_download:
        modal_file_path = f'/results/{filename}'
        
        if os.path.exists(modal_file_path):
            try:
                with open(modal_file_path, 'rb') as f:
                    file_content = f.read()
                    files_data[filename] = base64.b64encode(file_content).decode('utf-8')
                    print(f"  ✅ Prepared for download: {filename} ({len(file_content)} bytes)")
                    
                    # Extract info from JSON files
                    if filename.endswith('.json'):
                        try:
                            json_data = json.loads(file_content.decode('utf-8'))
                            if 'final_accuracy' in json_data:
                                results_info['training_accuracy'] = json_data['final_accuracy']
                                results_info['training_time'] = json_data['training_time']
                                results_info['num_epochs'] = json_data['num_epochs']
                        except:
                            pass
                            
            except Exception as e:
                print(f"  ❌ Failed to prepare {filename}: {str(e)}")
        else:
            print(f"  ⚠️ File not found: {filename}")
    
    download_summary = {
        'local_directory': local_dir,
        'files_data': files_data,
        'results_info': results_info,
        'total_files': len(files_data)
    }
    
    print(f"📦 Prepared {len(files_data)} files for download")
    
    return download_summary


@app.function(cpu=1)
def save_files_locally(files_data: Dict[str, str], local_dir: str = "./results") -> Dict[str, Any]:
    """
    Save base64 encoded files to local directory (this runs locally).
    """
    
    import json
    import os
    import base64
    from pathlib import Path
    
    print(f"💾 Saving files to local directory: {local_dir}")
    
    # Create local results directory if it doesn't exist
    local_path = Path(local_dir)
    local_path.mkdir(exist_ok=True)
    
    saved_files = []
    
    for filename, encoded_content in files_data.items():
        local_file_path = local_path / filename
        
        try:
            # Decode base64 and save file
            file_content = base64.b64decode(encoded_content)
            with open(local_file_path, 'wb') as f:
                f.write(file_content)
            saved_files.append(filename)
            print(f"  ✅ Saved: {filename}")
            
        except Exception as e:
            print(f"  ❌ Failed to save {filename}: {str(e)}")
    
    save_summary = {
        'local_directory': str(local_path.absolute()),
        'saved_files': saved_files,
        'total_files': len(saved_files)
    }
    
    print(f"✅ Successfully saved {len(saved_files)} files to {local_path.absolute()}")
    
    return save_summary


@app.function(cpu=2)
def analyze_results() -> Dict[str, Any]:
    """
    Analyze saved results dari training runs.
    """
    
    import json
    import os
    
    results_summary = {}
    
    # Check for training results
    if os.path.exists('/results/modal_vqc_results.json'):
        with open('/results/modal_vqc_results.json', 'r') as f:
            training_results = json.load(f)
            results_summary['latest_training'] = {
                'accuracy': training_results['final_accuracy'],
                'training_time': training_results['training_time'],
                'gpu_type': training_results.get('gpu_type', 'unknown'),
                'num_epochs': training_results['num_epochs']
            }
    
    # Check for benchmark results
    if os.path.exists('/results/modal_sampler_benchmark.json'):
        with open('/results/modal_sampler_benchmark.json', 'r') as f:
            benchmark_results = json.load(f)
            
            successful_benchmarks = {k: v for k, v in benchmark_results.items() if v.get('success', False)}
            
            if successful_benchmarks:
                best_config = max(successful_benchmarks.items(), key=lambda x: x[1]['accuracy'])
                results_summary['best_sampler'] = {
                    'name': best_config[0],
                    'accuracy': best_config[1]['accuracy'],
                    'training_time': best_config[1]['training_time']
                }
    
    # List available files
    if os.path.exists('/results'):
        results_summary['available_files'] = os.listdir('/results')
    
    return results_summary


# CLI interface untuk local testing
@app.local_entrypoint()
def main(
    action: str = "train",
    dataset_url: str = "https://raw.githubusercontent.com/Herutriana44/QuantumCovid19EpitopePrediction/main/clean_after_balanced_amino_data.csv", 
    epochs: int = 2,  # Default 2 epochs for quick training
    test_size: float = 0.3,  # More training data
    gpu_type: str = "B200:8",  # Maximum GPU performance
    local_results_dir: str = "./results",  # Local directory untuk menyimpan hasil
    auto_download: bool = True,  # Otomatis download hasil setelah training
    no_auto_download: bool = False  # Flag untuk disable auto download
):
    """
    Main entrypoint untuk menjalankan COVID-19 Epitope VQC training di Modal.
    
    Args:
        action: Action to perform ('train', 'benchmark', 'analyze', 'download')
        dataset_url: URL to COVID-19 epitope dataset CSV on GitHub
        epochs: Number of training epochs
        test_size: Test split ratio
        gpu_type: GPU type preference
        local_results_dir: Local directory untuk menyimpan hasil training
        auto_download: Otomatis download hasil setelah training selesai (default: True)
        no_auto_download: Flag untuk disable auto download (use --no-auto-download)
    """
    
    # Handle auto_download logic
    if no_auto_download:
        auto_download = False
    
    if action == "train":
        print(f"🚀 Starting COVID-19 Epitope VQC training on Modal with {gpu_type}...")
        result = train_vqc_gpu.remote(
            dataset_url=dataset_url,
            num_epochs=epochs,
            test_size=test_size,
            gpu_type=gpu_type
        )
        print(f"Training completed with accuracy: {result['final_accuracy']:.4f}")
        
        # Auto-download results to local if enabled
        if auto_download:
            print("\n📥 Auto-downloading results to local storage...")
            download_summary = download_results_to_local.remote(local_results_dir)
            
            # Save files locally using the returned data
            if download_summary['files_data']:
                import json
                import os
                import base64
                from pathlib import Path
                
                # Create local results directory
                local_path = Path(local_results_dir)
                local_path.mkdir(exist_ok=True)
                
                saved_files = []
                for filename, encoded_content in download_summary['files_data'].items():
                    local_file_path = local_path / filename
                    
                    try:
                        # Decode base64 and save file
                        file_content = base64.b64decode(encoded_content)
                        with open(local_file_path, 'wb') as f:
                            f.write(file_content)
                        saved_files.append(filename)
                        print(f"  ✅ Saved locally: {filename}")
                        
                    except Exception as e:
                        print(f"  ❌ Failed to save {filename}: {str(e)}")
                
                print(f"✅ Results downloaded to: {local_path.absolute()}")
                print(f"Files saved: {', '.join(saved_files)}")
            else:
                print("⚠️ No files available for download")
        
    elif action == "benchmark":
        print("🔥 Starting sampler benchmark...")
        results = benchmark_samplers.remote()
        print("Benchmark completed!")
        
        # Auto-download benchmark results if enabled
        if auto_download:
            print("\n📥 Auto-downloading benchmark results to local storage...")
            download_summary = download_results_to_local.remote(local_results_dir)
            
            # Save files locally using the returned data
            if download_summary['files_data']:
                import json
                import os
                import base64
                from pathlib import Path
                
                # Create local results directory
                local_path = Path(local_results_dir)
                local_path.mkdir(exist_ok=True)
                
                saved_files = []
                for filename, encoded_content in download_summary['files_data'].items():
                    local_file_path = local_path / filename
                    
                    try:
                        # Decode base64 and save file
                        file_content = base64.b64decode(encoded_content)
                        with open(local_file_path, 'wb') as f:
                            f.write(file_content)
                        saved_files.append(filename)
                        print(f"  ✅ Saved locally: {filename}")
                        
                    except Exception as e:
                        print(f"  ❌ Failed to save {filename}: {str(e)}")
                
                print(f"✅ Results downloaded to: {local_path.absolute()}")
            else:
                print("⚠️ No files available for download")
        
    elif action == "download":
        print(f"📥 Downloading results from Modal to local directory: {local_results_dir}")
        download_summary = download_results_to_local.remote(local_results_dir)
        
        # Save files locally using the returned data
        if download_summary['files_data']:
            import json
            import os
            import base64
            from pathlib import Path
            
            # Create local results directory
            local_path = Path(local_results_dir)
            local_path.mkdir(exist_ok=True)
            
            saved_files = []
            for filename, encoded_content in download_summary['files_data'].items():
                local_file_path = local_path / filename
                
                try:
                    # Decode base64 and save file
                    file_content = base64.b64decode(encoded_content)
                    with open(local_file_path, 'wb') as f:
                        f.write(file_content)
                    saved_files.append(filename)
                    print(f"  ✅ Saved locally: {filename}")
                    
                except Exception as e:
                    print(f"  ❌ Failed to save {filename}: {str(e)}")
            
            print(f"✅ Download completed!")
            print(f"Local directory: {local_path.absolute()}")
            print(f"Downloaded files: {', '.join(saved_files)}")
            if download_summary['results_info']:
                print(f"Training info: {download_summary['results_info']}")
        else:
            print("⚠️ No files available for download")
        
    elif action == "analyze":
        print("📊 Analyzing results...")
        summary = analyze_results.remote()
        print("Analysis:", summary)
        
    else:
        print("Available actions: train, benchmark, analyze, download")


if __name__ == "__main__":
    # Example usage
    print("COVID-19 Epitope Prediction VQC on Modal - Multi-GPU Parallel Training Ready!")
    print("Dataset will be downloaded from GitHub automatically")
    print("Results will be saved both in Modal volume and local directory")
    print("")
    print("🚀 MULTI-GPU PARALLEL TRAINING FEATURES:")
    print("  ✅ DistributedDataParallel (DDP) for optimal GPU utilization")
    print("  ✅ Mixed precision training with gradient scaling")
    print("  ✅ Adaptive learning rate scaling based on GPU count")
    print("  ✅ Memory optimization with automatic cleanup")
    print("  ✅ Enhanced quantum circuit parallelization")
    print("")
    print("Example commands:")
    print("  # Multi-GPU training with B200:8 (8 GPUs)")
    print("  modal run ModalQuantumCovid19TrainingModelQiskit.py --action=train --gpu-type=B200:8 --epochs=5")
    print("")
    print("  # Training with specific batch size per GPU")
    print("  modal run ModalQuantumCovid19TrainingModelQiskit.py --action=train --gpu-type=B200:8 --epochs=10 --no-auto-download")
    print("")
    print("  # Comprehensive multi-GPU benchmark")
    print("  modal run ModalQuantumCovid19TrainingModelQiskit.py --action=benchmark")
    print("")
    print("  # Download results after training")
    print("  modal run ModalQuantumCovid19TrainingModelQiskit.py --action=download --local-results-dir=./results")
    print("")
    print("Multi-GPU Optimizations:")
    print("  🔥 Batch size: 64 per GPU × 8 GPUs = 512 effective batch size")
    print("  🔥 Learning rate: Auto-scaled based on GPU count")
    print("  🔥 Quantum circuits: Parallel execution with B200 optimization")
    print("  🔥 Memory: 64GB total with automatic cleanup")
    print("")
    print("Files that will be saved locally:")
    print("  - modal_vqc_training_results.png (training plots & confusion matrix)")
    print("  - modal_vqc_results.json (detailed training metrics + multi-GPU info)")
    print("  - modal_vqc_model.pth (trained PyTorch model with DDP state)")
    print("  - modal_sampler_benchmark.json (multi-GPU benchmark results)")
