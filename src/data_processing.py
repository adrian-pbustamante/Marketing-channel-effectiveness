# src/data_processing.py
import pandas as pd
import numpy as np

def load_and_process_data(filepath):
    """Loads a CSV file and performs initial data cleaning."""
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    df.columns = df.columns.str.strip().str.replace(' ', '_')
    return df

def adstock_transform(series, decay_rate):
    """Applies a geometrically decaying adstock transformation."""
    adstock = series.copy()
    for i in range(1, len(adstock)):
        adstock.iloc[i] += adstock.iloc[i - 1] * decay_rate
    return adstock

def apply_adstock_transformation(df, marketing_channels, decay_rates):
    """
    Applies an adstock transformation to specified channels using a dictionary of decay rates.
    """
    adstock_df = df.copy()
    for col in marketing_channels:
        decay_rate = decay_rates.get(col, 0.5) 
        adstock_df[f'adstock_{col}'] = adstock_transform(adstock_df[col], decay_rate)
    return adstock_df

def get_base_data(df, split_date, y_col):
    """Returns the base train/test data needed for decay rate tuning."""
    train_df = df.loc[df.index < split_date].copy()
    test_df = df.loc[df.index >= split_date].copy()
    
    y_train = train_df[y_col]
    y_test = test_df[y_col]
    
    return train_df, test_df, y_train, y_test