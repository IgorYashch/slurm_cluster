import pandas as pd
import numpy as np

def compute_aggregated_numeric_stats(df):
    """
    Computes aggregated statistics (mean, std, min, max, median) information
    for each numeric column in a pandas DataFrame.
    
    Parameters:
    - df: pandas DataFrame
    
    Returns:
    - dict: A dictionary with aggregated statistics and histogram information.
    """
    results = {}
    basic_stats = df.describe().T
    print(basic_stats)
    if df.shape[0] > 0:
        for col in basic_stats.index:
            for stat in ['mean', 'std', 'min', 'max']:
                results[f'{col}_{stat}'] = basic_stats.at[col, stat]
            results[f'{col}_median'] = df[col].median()
    else:
        for col in basic_stats.index:
            for stat in ['mean', 'std', 'min', 'max', 'median']:
                results[f'{col}_{stat}'] = 0
    return results