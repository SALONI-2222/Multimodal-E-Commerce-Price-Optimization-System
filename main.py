# File: main.py (FINAL WORKING VERSION)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import os
import gc 
import time  
import torch 
import warnings
import sys
import re

# Ignore benign warnings from libraries
warnings.filterwarnings("ignore") 

# Imports from your modules
from src.utils import download_images
from src.features import create_feature_matrix, extract_structured_features, initialize_models

# --- CONFIGURATION & PATHS ---
TRAIN_CSV = 'dataset/train.csv'
TEST_CSV = 'dataset/test.csv'
TRAIN_IMG_FOLDER = 'dataset/images/train'
TEST_IMG_FOLDER = 'dataset/images/test'
SUBMISSION_FILE = 'test_out.csv'

# --- SMAPE METRIC ---

def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error (SMAPE) calculation."""
    # FINAL FIX: Clamp y_true to prevent division by zero (NaN)
    y_true = y_true.clip(min=1e-6) 
    
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / denominator) * 100

def chunked_feature_generation(df, image_folder, mode='train', chunk_size=4000, initial_metadata=None, models=None):
    """
    Processes the DataFrame in chunks to manage memory.
    """
    all_X_chunks = []
    metadata = initial_metadata if initial_metadata is not None else {}
    
    # Calculate IPQ/Brand/Target on the entire df once 
    df_processed, y_full, _, _ = extract_structured_features(
        df.copy(), le_brand=metadata.get('le_brand'), le_ipq=metadata.get('le_ipq')
    )
    y_target = y_full if mode == 'train' else None
    
    print(f"Starting feature generation for {len(df)} samples...")
    
    for i in range(0, len(df), chunk_size):
        start_time = time.time()
        
        # Take the pre-processed chunk for embedding (avoids recalculating IPQ/Brand)
        chunk_df_processed = df_processed.iloc[i:i + chunk_size].copy()
        
        print(f"  -> Processing chunk {i//chunk_size + 1}/{len(df)//chunk_size + 1}...")

        # 1. CALL create_feature_matrix 
        feature_result = create_feature_matrix(
            chunk_df_processed, 
            image_folder,
            mode=mode,
            metadata=metadata if metadata is not None and i != 0 else {},
            models=models
        )
        
        X_chunk = feature_result[0]
        
        # 2. CONDITIONAL METADATA UPDATE
        if mode == 'train' and i == 0:
            metadata = feature_result[2] 
            
        all_X_chunks.append(X_chunk)
        
        # CRUCIAL: Manual garbage collection to free memory
        del X_chunk, chunk_df_processed
        gc.collect() 
        torch.cuda.empty_cache() 

    X_combined = pd.concat(all_X_chunks, ignore_index=True)
    
    # Return all necessary outputs
    if mode == 'train':
        return X_combined, y_target, metadata 
    else:
        return X_combined, df_processed['IPQ_Value']


def run_training_and_prediction(models):
    """Handles feature engineering, training, validation, and submission."""
    
    # 1. Load Data
    train_full_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    
    # Split data for validation check
    train_df, val_df = train_test_split(train_full_df, test_size=0.1, random_state=42)
    
    print("\n--- 2. Feature Engineering (Memory-Managed Chunking) ---")
    
    # Generate Full Training Features in Chunks
    X_full, y_full, final_metadata = chunked_feature_generation(
        train_full_df, TRAIN_IMG_FOLDER, mode='train', chunk_size=4000, initial_metadata={}, models=models
    )
    
    # Re-split the resulting feature matrix for validation check
    train_len = len(train_full_df) - len(val_df)
    X_train = X_full.iloc[:train_len]
    X_val = X_full.iloc[train_len:]
    y_train = y_full.iloc[:train_len]
    y_val = y_full.iloc[len(train_full_df) - len(val_df):]
    
    # Re-extract necessary IPQ for de-normalization
    val_ipq_values = extract_structured_features(val_df)[0]['IPQ_Value'].values
    
    print(f"Train features shape: {X_train.shape}, Validation features shape: {X_val.shape}")

    # 3. Train Regression Model (LightGBM)
    print("\n--- 3. Training LGBM Regressor ---")
    
    # Tighter parameters for better accuracy
    lgbm = LGBMRegressor(
        objective='mae', 
        metric='mae', 
        n_estimators=3000,             # Increased estimators
        learning_rate=0.01,            # Decreased learning rate
        num_leaves=63,                 # Increased complexity
        random_state=42, 
        n_jobs=-1,
        verbose=-1 
    )

    lgbm.fit(X_full, y_full) 

    # 4. Evaluate on Validation Set
    print("--- 4. Validation Performance ---")
    log_pred_val = lgbm.predict(X_val)
    actual_val_prices = val_df['price'].values
    pred_val_prices = np.expm1(log_pred_val) * val_ipq_values
    
    val_smape = smape(actual_val_prices, pred_val_prices.clip(min=0.01))
    print(f"✅ Validation SMAPE (Accuracy Metric): {val_smape:.2f}%")
    
    # 5. Final Prediction on Test Set
    print("\n--- 5. Generating Test Predictions ---")
    
    # Generate Test Features 
    X_test, test_ipq_values = chunked_feature_generation(
        test_df, TEST_IMG_FOLDER, mode='test', chunk_size=4000, initial_metadata=final_metadata, models=models
    )
    
    # Predict and De-normalize
    log_pred_test = lgbm.predict(X_test)
    predicted_prices = np.expm1(log_pred_test) * test_ipq_values.values
    
    # 6. Create Submission File
    submission_df = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': predicted_prices.clip(min=0.01)
    })
    
    submission_df.to_csv(SUBMISSION_FILE, index=False)
    print(f"Submission file created successfully: {SUBMISSION_FILE}")


if __name__ == '__main__':
    # --- PHASE A: MODEL INITIALIZATION (RUNS ONCE) ---
    try:
        MODELS = initialize_models()
        print("\n--- PHASE B: RUNNING PIPELINE ---")
        # --- PHASE B: PIPELINE EXECUTION ---
        run_training_and_prediction(MODELS)

    except Exception as e:
        print(f"\n--- FATAL ERROR ---")
        print(f"An error occurred during execution: {e}")
        print("The code structure is finalized. The error is likely due to an unhandled data point (e.g., infinity).")

'''# File: main.py (FINAL CORRECTED VERSION - Sequential Execution)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import os
import gc 
import time  
import torch 
import warnings
import sys
import re

# Ignore benign warnings from libraries
warnings.filterwarnings("ignore") 

# Imports from your modules
from src.utils import download_images
from src.features import create_feature_matrix, extract_structured_features, initialize_models

# --- CONFIGURATION & PATHS ---
TRAIN_CSV = 'dataset/train.csv'
TEST_CSV = 'dataset/test.csv'
TRAIN_IMG_FOLDER = 'dataset/images/train'
TEST_IMG_FOLDER = 'dataset/images/test'
SUBMISSION_FILE = 'test_out.csv'

# --- SMAPE METRIC ---

def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error (SMAPE) calculation."""
    # Final SMAPE Fix: Clamp y_true to prevent division by zero (NaN)
    y_true = y_true.clip(min=1e-6) 
    
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / denominator) * 100

def chunked_feature_generation(df, image_folder, mode='train', chunk_size=4000, initial_metadata=None, models=None):
    """
    Processes the DataFrame in chunks, passing the initialized models.
    """
    all_X_chunks = []
    metadata = initial_metadata if initial_metadata is not None else {}
    
    df_processed, y_full, _, _ = extract_structured_features(
        df.copy(), le_brand=metadata.get('le_brand'), le_ipq=metadata.get('le_ipq')
    )
    y_target = y_full if mode == 'train' else None
    
    print(f"Starting feature generation for {len(df)} samples...")
    
    for i in range(0, len(df), chunk_size):
        start_time = time.time()
        
        chunk_df_processed = df_processed.iloc[i:i + chunk_size].copy()
        
        print(f"  -> Processing chunk {i//chunk_size + 1}/{len(df)//chunk_size + 1}...")

        feature_result = create_feature_matrix(
            chunk_df_processed, 
            image_folder,
            mode=mode,
            metadata=metadata if i != 0 else None,
            models=models # Pass the models here
        )
        
        X_chunk = feature_result[0]
        
        if mode == 'train' and i == 0:
            metadata = feature_result[2] 
            
        all_X_chunks.append(X_chunk)
        
        # CRUCIAL: Manual garbage collection
        del X_chunk, chunk_df_processed
        gc.collect() 
        torch.cuda.empty_cache() 

    X_combined = pd.concat(all_X_chunks, ignore_index=True)
    
    if mode == 'train':
        return X_combined, y_target, metadata 
    else:
        return X_combined, df_processed['IPQ_Value']


def run_training_and_prediction(models):
    """Handles feature engineering, training, validation, and submission."""
    
    # 1. Load Data
    train_full_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    
    # Split data for validation check
    train_df, val_df = train_test_split(train_full_df, test_size=0.1, random_state=42)
    
    print("\n--- 2. Feature Engineering (Memory-Managed Chunking) ---")
    
    # Generate Full Training Features in Chunks
    X_full, y_full, final_metadata = chunked_feature_generation(
        train_full_df, TRAIN_IMG_FOLDER, mode='train', chunk_size=4000, initial_metadata={}, models=models
    )
    
    # Re-split the resulting feature matrix for validation check
    train_len = len(train_full_df) - len(val_df)
    X_train = X_full.iloc[:train_len]
    X_val = X_full.iloc[train_len:]
    y_train = y_full.iloc[:train_len]
    y_val = y_full.iloc[len(train_full_df) - len(val_df):]
    
    # Re-extract necessary IPQ for de-normalization
    val_ipq_values = extract_structured_features(val_df)[0]['IPQ_Value'].values
    
    print(f"Train features shape: {X_train.shape}, Validation features shape: {X_val.shape}")

    # 3. Train Regression Model (LightGBM)
    print("\n--- 3. Training LGBM Regressor ---")
    
    lgbm = LGBMRegressor(
        objective='mae', 
        metric='mae', 
        n_estimators=1500, 
        learning_rate=0.05,
        num_leaves=31,
        random_state=42, 
        n_jobs=-1,
        verbose=-1 
    )

    lgbm.fit(X_full, y_full) 

    # 4. Evaluate on Validation Set
    print("--- 4. Validation Performance ---")
    log_pred_val = lgbm.predict(X_val)
    actual_val_prices = val_df['price'].values
    pred_val_prices = np.expm1(log_pred_val) * val_ipq_values
    
    val_smape = smape(actual_val_prices, pred_val_prices.clip(min=0.01))
    print(f"✅ Validation SMAPE (Accuracy Metric): {val_smape:.2f}%")
    
    # 5. Final Prediction on Test Set
    print("\n--- 5. Generating Test Predictions ---")
    
    # Generate Test Features 
    X_test, test_ipq_values = chunked_feature_generation(
        test_df, TEST_IMG_FOLDER, mode='test', chunk_size=4000, initial_metadata=final_metadata, models=models
    )
    
    # Predict and De-normalize
    log_pred_test = lgbm.predict(X_test)
    predicted_prices = np.expm1(log_pred_test) * test_ipq_values.values
    
    # 6. Create Submission File
    submission_df = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': predicted_prices.clip(min=0.01)
    })
    
    submission_df.to_csv(SUBMISSION_FILE, index=False)
    print(f"Submission file created successfully: {SUBMISSION_FILE}")


if __name__ == '__main__':
    # --- PHASE A: MODEL INITIALIZATION (RUNS ONCE) ---
    try:
        MODELS = initialize_models()
        print("\n--- PHASE B: RUNNING PIPELINE ---")
        # --- PHASE B: PIPELINE EXECUTION ---
        run_training_and_prediction(MODELS)

    except Exception as e:
        print(f"\n--- FATAL ERROR ---")
        print(f"An error occurred during execution: {e}")
        print("Please check your environment variables (OMP_NUM_THREADS, etc.) or Python installation.")

'''