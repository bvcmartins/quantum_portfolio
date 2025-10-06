#!/usr/bin/env python3
"""
Script to regenerate synthetic dataset with proper stock price scaling
"""

import pandas as pd
import numpy as np
import random
import string
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
import os

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

def generate_random_column_name(length=6):
    """Generate random stock-like column names"""
    prefixes = ['STOCK', 'ASSET', 'FUND', 'BOND', 'REIT', 'ETF', 'INDEX', 'COMP']
    suffixes = ['_RET', '_VOL', '_BETA', '_PRICE', '_YIELD', '_CAP', '_RATIO', '_SCORE']

    if random.random() < 0.6:
        # Stock ticker style
        ticker = ''.join(random.choices(string.ascii_uppercase, k=random.randint(3, 5)))
        if random.random() < 0.3:
            ticker += random.choice(suffixes)
        return ticker
    else:
        # Descriptive name style
        prefix = random.choice(prefixes)
        number = random.randint(1, 999)
        suffix = random.choice(suffixes) if random.random() < 0.5 else ''
        return f"{prefix}_{number}{suffix}"

def normalize_to_stock_range(data, min_val=1.0, max_val=1000.0):
    """Normalize data to typical stock price range (not returns)"""
    # Handle edge case where all values are the same
    if data.max() == data.min():
        return np.full_like(data, (min_val + max_val) / 2)

    data_norm = (data - data.min()) / (data.max() - data.min())
    return data_norm * (max_val - min_val) + min_val

def create_correlated_features(base_data, n_features, correlation_strength=0.7):
    """Create features with specified correlation to base data"""
    new_features = pd.DataFrame()

    for i in range(n_features):
        # Choose random base column
        base_col = np.random.choice(base_data.columns)
        base_values = base_data[base_col].values

        # Generate noise
        noise = np.random.normal(0, 1, len(base_values))

        # Create correlated feature
        if correlation_strength > 0:
            # Positive correlation
            new_values = correlation_strength * base_values + np.sqrt(1 - correlation_strength**2) * noise
        else:
            # Negative correlation
            new_values = abs(correlation_strength) * (-base_values) + np.sqrt(1 - correlation_strength**2) * noise

        # Normalize to stock-like range
        new_values = normalize_to_stock_range(new_values)

        col_name = generate_random_column_name()
        new_features[col_name] = new_values

    return new_features

def create_combined_features(base_data, n_features):
    """Create features by combining existing columns"""
    new_features = pd.DataFrame()

    for i in range(n_features):
        # Choose 2-4 random columns to combine
        n_cols = np.random.randint(2, 5)
        selected_cols = np.random.choice(base_data.columns, n_cols, replace=False)

        # Choose combination method
        method = np.random.choice(['weighted_sum', 'product', 'ratio', 'difference'])

        if method == 'weighted_sum':
            weights = np.random.uniform(-1, 1, n_cols)
            new_values = np.sum([w * base_data[col].values for w, col in zip(weights, selected_cols)], axis=0)

        elif method == 'product':
            new_values = np.prod([base_data[col].values for col in selected_cols], axis=0)
            new_values = np.sign(new_values) * np.log1p(np.abs(new_values))

        elif method == 'ratio':
            if n_cols >= 2:
                num = base_data[selected_cols[0]].values
                denom = base_data[selected_cols[1]].values + 0.001  # Avoid division by zero
                new_values = num / denom
            else:
                new_values = base_data[selected_cols[0]].values

        elif method == 'difference':
            if n_cols >= 2:
                new_values = base_data[selected_cols[0]].values - base_data[selected_cols[1]].values
            else:
                new_values = base_data[selected_cols[0]].values

        # Normalize to stock-like range
        new_values = normalize_to_stock_range(new_values)

        col_name = generate_random_column_name()
        new_features[col_name] = new_values

    return new_features

def create_pca_features(base_data, n_components, n_features):
    """Create features using PCA components"""
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(base_data)

    new_features = pd.DataFrame()

    for i in range(n_features):
        # Select random PCA components and weights
        n_comp_selected = np.random.randint(1, min(n_components, 5) + 1)
        selected_components = np.random.choice(n_components, n_comp_selected, replace=False)
        weights = np.random.uniform(-1, 1, n_comp_selected)

        # Create new feature as weighted sum of PCA components
        new_values = np.sum([w * pca_data[:, comp] for w, comp in zip(weights, selected_components)], axis=0)

        # Add some noise
        noise = np.random.normal(0, 0.1, len(new_values))
        new_values = new_values + noise

        # Normalize to stock-like range
        new_values = normalize_to_stock_range(new_values)

        col_name = generate_random_column_name()
        new_features[col_name] = new_values

    return new_features

def create_noise_features(base_data, n_features):
    """Create features with controlled noise patterns scaled to stock price ranges"""
    new_features = pd.DataFrame()

    for i in range(n_features):
        # Different noise patterns
        noise_type = np.random.choice(['gaussian', 'uniform', 'exponential', 'laplace'])
        n_samples = len(base_data)

        if noise_type == 'gaussian':
            new_values = np.random.normal(50, 30, n_samples)  # Mean ~50, std ~30
        elif noise_type == 'uniform':
            new_values = np.random.uniform(10, 200, n_samples)  # Range 10-200
        elif noise_type == 'exponential':
            new_values = np.random.exponential(25, n_samples)  # Scale ~25
            new_values = np.abs(new_values) + 5  # Ensure positive, min 5
        elif noise_type == 'laplace':
            new_values = np.random.laplace(40, 20, n_samples)  # Location 40, scale 20
            new_values = np.abs(new_values) + 1  # Ensure positive, min 1

        # Ensure reasonable stock price range
        new_values = np.clip(new_values, 0.5, 2000)  # Clip to reasonable range

        col_name = generate_random_column_name()
        new_features[col_name] = new_values

    return new_features

def expand_dataset(data, target_columns=10000):
    """Main function to expand dataset from 680 to target number of columns"""
    print(f"Starting with {data.shape[1]} columns")
    print(f"Target: {target_columns} columns")
    print(f"Preserving date index: {data.index.name}")

    # Keep the original date index
    expanded_data = data.copy()
    columns_to_add = target_columns - data.shape[1]

    # Distribution of new features
    n_positive_corr = int(columns_to_add * 0.25)  # 25% positive correlations
    n_negative_corr = int(columns_to_add * 0.25)  # 25% negative correlations
    n_combined = int(columns_to_add * 0.30)       # 30% combined features
    n_pca = int(columns_to_add * 0.10)            # 10% PCA features
    n_noise = columns_to_add - (n_positive_corr + n_negative_corr + n_combined + n_pca)  # Remaining as noise

    print(f"Creating {n_positive_corr} positive correlation features...")
    pos_corr_features = create_correlated_features(data, n_positive_corr, 0.7)
    # Preserve the date index
    pos_corr_features.index = data.index
    expanded_data = pd.concat([expanded_data, pos_corr_features], axis=1)

    print(f"Creating {n_negative_corr} negative correlation features...")
    neg_corr_features = create_correlated_features(data, n_negative_corr, -0.6)
    neg_corr_features.index = data.index
    expanded_data = pd.concat([expanded_data, neg_corr_features], axis=1)

    print(f"Creating {n_combined} combined features...")
    combined_features = create_combined_features(data, n_combined)
    combined_features.index = data.index
    expanded_data = pd.concat([expanded_data, combined_features], axis=1)

    print(f"Creating {n_pca} PCA-based features...")
    pca_features = create_pca_features(data, min(50, data.shape[1]//2), n_pca)
    pca_features.index = data.index
    expanded_data = pd.concat([expanded_data, pca_features], axis=1)

    print(f"Creating {n_noise} noise features...")
    noise_features = create_noise_features(data, n_noise)
    noise_features.index = data.index
    expanded_data = pd.concat([expanded_data, noise_features], axis=1)

    print(f"Final dataset shape: {expanded_data.shape}")
    print(f"Date index preserved: {expanded_data.index.name}")
    print(f"Date range: {expanded_data.index[0]} to {expanded_data.index[-1]}")
    return expanded_data

if __name__ == "__main__":
    print("Loading original data...")

    # Load the original wide dataset to get the timestamps
    original_stocks_path = '../../data/stocks_adjclose.pkl'
    original_etfs_path = '../../data/etfs_close.pkl'

    # Load original data to get the proper date index
    stocks_data = pd.read_pickle(original_stocks_path)
    etfs_data = pd.read_pickle(original_etfs_path)

    # Create wide dataset like in the portfolio optimization notebook
    wide_data = pd.merge(stocks_data, etfs_data, on='ds', how='left').set_index('ds').dropna()

    print(f"Original wide dataset shape: {wide_data.shape}")
    print(f"Date range: {wide_data.index[0]} to {wide_data.index[-1]}")
    print(f"Sample data range: {wide_data.min().min():.4f} to {wide_data.max().max():.4f}")

    # Use the wide dataset as the base for expansion
    sample_data = wide_data.copy()

    print(f"Base data shape: {sample_data.shape}")
    print(f"Index type: {type(sample_data.index)}")
    print(f"First few dates: {sample_data.index[:5].tolist()}")

    print("\n" + "="*50)
    print("REGENERATING SYNTHETIC DATASET WITH PROPER SCALING")
    print("="*50)

    # Expand the dataset with proper stock price scaling
    expanded_dataset = expand_dataset(sample_data, target_columns=10000)

    print("\nDataset expansion completed!")
    print(f"Original shape: {sample_data.shape}")
    print(f"Expanded shape: {expanded_dataset.shape}")
    print(f"Value range: {expanded_dataset.min().min():.4f} to {expanded_dataset.max().max():.4f}")
    print(f"Original data range: {sample_data.min().min():.4f} to {sample_data.max().max():.4f}")

    # Check synthetic columns specifically (excluding original data)
    synthetic_cols = expanded_dataset.columns[len(sample_data.columns):]
    if len(synthetic_cols) > 0:
        synthetic_data = expanded_dataset[synthetic_cols]
        print(f"Synthetic data range: {synthetic_data.min().min():.4f} to {synthetic_data.max().max():.4f}")
        print(f"Synthetic data mean: {synthetic_data.mean().mean():.4f}")
        print(f"Synthetic data std: {synthetic_data.std().mean():.4f}")

    # Save the updated expanded dataset to data folder as synthetic_close.pkl
    os.makedirs('../../data', exist_ok=True)

    # Ensure the dataset has the proper date index named 'ds'
    expanded_dataset.index.name = 'ds'

    # Reset index to make 'ds' a column, then set it back as index to ensure proper format
    expanded_dataset_with_ds = expanded_dataset.reset_index()
    print(f"\nDataset with ds column shape: {expanded_dataset_with_ds.shape}")
    print(f"Columns include 'ds': {'ds' in expanded_dataset_with_ds.columns}")
    print(f"Date range: {expanded_dataset_with_ds['ds'].min()} to {expanded_dataset_with_ds['ds'].max()}")

    # Check value ranges after update
    numeric_cols = expanded_dataset_with_ds.select_dtypes(include=[np.number]).columns
    print(f"Updated value range: {expanded_dataset_with_ds[numeric_cols].min().min():.4f} to {expanded_dataset_with_ds[numeric_cols].max().max():.4f}")

    # Save with 'ds' as a column (like the original datasets)
    expanded_dataset_with_ds.to_pickle('../../data/synthetic_close.pkl')
    print("Updated expanded dataset saved as 'quantum_portfolio/data/synthetic_close.pkl'")

    # Verify what we saved
    test_load = pd.read_pickle('../../data/synthetic_close.pkl')
    print(f"Verification - loaded dataset shape: {test_load.shape}")
    print(f"Verification - has 'ds' column: {'ds' in test_load.columns}")
    if 'ds' in test_load.columns:
        print(f"Verification - ds column type: {test_load['ds'].dtype}")
        print(f"Verification - first few ds values: {test_load['ds'].head().tolist()}")

    # Check final value range
    numeric_cols = test_load.select_dtypes(include=[np.number]).columns
    final_min = test_load[numeric_cols].min().min()
    final_max = test_load[numeric_cols].max().max()
    final_mean = test_load[numeric_cols].mean().mean()

    print(f"Final verification - value range: {final_min:.4f} to {final_max:.4f}")
    print(f"Final verification - mean value: {final_mean:.4f}")

    print("\nDataset regeneration completed successfully!")
    print(f"Successfully updated synthetic dataset with proper price scaling")

    # Final comparison with real data
    print(f"\nComparison with original real data:")
    print(f"Real data range: {sample_data.min().min():.4f} to {sample_data.max().max():.4f}")
    print(f"Real data mean: {sample_data.mean().mean():.4f}")
    print(f"Synthetic data range: {final_min:.4f} to {final_max:.4f}")
    print(f"Synthetic data mean: {final_mean:.4f}")
    print(f"Scale similarity achieved: ✓" if abs(final_mean/sample_data.mean().mean() - 1) < 2 else "Scale needs adjustment: ✗")