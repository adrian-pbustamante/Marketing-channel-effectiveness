# run_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data_processing import (
    load_and_process_data, 
    apply_adstock_transformation,
    get_base_data
)
from src.modeling import (
    check_multicollinearity, 
    find_optimal_decay_rates,
    run_ols_model,
    run_ridge_model_with_decay,
    plot_predictions, 
    plot_ols_interpretation
)

# --- Configuration ---
DATA_FILE = 'data/marketing_spend_and_revenue_data.csv'
SPLIT_DATE = '2022-12-16'
MARKETING_CHANNELS = [
    'Google_Performance_Max', 'Google_Search_Brand', 'Google_Search_No_Brand', 'Facebook_Conversions', 'Facebook_Others',
    'Facebook_Product_Catalog_Sales', 'Influencers', 'Display_Ads', 'TV_Ads', 'Radio_Ads', 'Magazine_Ads'
]
DECAY_RATES_GRID = np.arange(0.1, 1.0, 0.1) # Grid to test decay rates from 0.1 to 0.9
Y_COL = 'Revenue'

def main():
    #  Load and Process Data
    print("Loading and processing data...")
    df = load_and_process_data(DATA_FILE)
    
    # Find Optimal Adstock Decay Rates
    print("Finding optimal adstock decay rates for each channel...")
    optimal_decay_rates = find_optimal_decay_rates(df, MARKETING_CHANNELS, DECAY_RATES_GRID, SPLIT_DATE, Y_COL)
    print("Optimal Decay Rates Found:")
    print(optimal_decay_rates)
    
    # Apply Adstock with Optimal Rates
    df_adstocked = apply_adstock_transformation(df, MARKETING_CHANNELS, optimal_decay_rates)
    
    # Train/Test Split
    X_COLS = [f'adstock_{col}' for col in MARKETING_CHANNELS] + ['Covid']
    train_df = df_adstocked.loc[df_adstocked.index < SPLIT_DATE].copy()
    test_df = df_adstocked.loc[df_adstocked.index >= SPLIT_DATE].copy()
    
    X_train, y_train = train_df[X_COLS], train_df[Y_COL]
    X_test, y_test = test_df[X_COLS], test_df[Y_COL]
    
    print(f"\nTraining data shape: {X_train.shape}, Test data shape: {X_test.shape}")
    
    # Check Multicollinearity
    print(" Checking for multicollinearity...")
    check_multicollinearity(X_train)
    
    # Run OLS Model
    print(" Running OLS model...")
    y_pred_ols, rmse_ols, r2_ols, ols = run_ols_model(X_train, y_train, X_test, y_test)
    print(f"OLS Test Set RMSE: ${rmse_ols:,.2f}")
    print(f"OLS Test Set R^2: {r2_ols:.4f}")
    
    #  Run Ridge Model with Optimal Decay Rates
    print("Running Ridge Regression model with optimal decay rates...")
    y_pred_ridge, rmse_ridge, r2_ridge = run_ridge_model_with_decay(X_train, y_train, X_test, y_test)
    print(f"Ridge Test Set RMSE: ${rmse_ridge:,.2f}")
    print(f"Ridge Test Set R^2: {r2_ridge:.4f}")

    # Plot contributions of each marketing channel
    print("Contributions of each marketing channel...")
    plot_ols_interpretation(ols, X_train, y_train)
    
    # Plot Results
    print("Plotting results...")
    plot_predictions(y_test, y_pred_ols, y_pred_ridge)

if __name__ == '__main__':
    main()