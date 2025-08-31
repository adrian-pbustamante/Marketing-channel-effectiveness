# src/modeling.py
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import matplotlib.pyplot as plt
from src.data_processing import adstock_transform, get_base_data

def check_multicollinearity(X_train):
    """Calculates and prints VIF scores for the training data."""
    X_train_vif = sm.add_constant(X_train)
    vif_data = pd.DataFrame()
    vif_data['Variable'] = X_train_vif.columns
    vif_data['VIF'] = [variance_inflation_factor(X_train_vif.values, i) for i in range(len(X_train_vif.columns))]
    print(vif_data.round(2))

def find_optimal_decay_rates(df, marketing_channels, decay_rates_grid, split_date, y_col):
    """
    Finds the optimal decay rate for each marketing channel using time-series cross-validation.
    """
    optimal_rates = {}
    df_train_temp, _, _, _ = get_base_data(df, split_date, y_col)

    for channel in marketing_channels:
        best_rmse = float('inf')
        best_rate = None
        
        print(f"  Tuning decay rate for {channel}...")
        
        for rate in decay_rates_grid:
            df_temp_adstocked = adstock_transform(df_train_temp[channel], rate)
            
            X_temp = pd.DataFrame({'adstock_channel': df_temp_adstocked})
            y_temp = df_train_temp[y_col]
            
            tscv = TimeSeriesSplit(n_splits=3)
            rmse_scores = []
            
            for train_index, val_index in tscv.split(X_temp):
                X_train_fold, X_val_fold = X_temp.iloc[train_index], X_temp.iloc[val_index]
                y_train_fold, y_val_fold = y_temp.iloc[train_index], y_temp.iloc[val_index]
                
                model = sm.OLS(y_train_fold, sm.add_constant(X_train_fold)).fit()
                y_pred_val = model.predict(sm.add_constant(X_val_fold))
                rmse_scores.append(np.sqrt(mean_squared_error(y_val_fold, y_pred_val)))
            
            avg_rmse = np.mean(rmse_scores)
            
            if avg_rmse < best_rmse:
                best_rmse = avg_rmse
                best_rate = rate
        
        optimal_rates[channel] = best_rate
        
    return optimal_rates

def run_ols_model(X_train, y_train, X_test, y_test):
    """Trains, interprets, and evaluates an OLS model."""
    X_train_sm = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_train_sm).fit()
    print(model.summary())
    
    X_test_sm = sm.add_constant(X_test)
    y_pred = model.predict(X_test_sm)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return y_pred, rmse, r2, model #i added the model

def run_ridge_model_with_decay(X_train, y_train, X_test, y_test):
    """Trains, tunes, and evaluates a Ridge Regression model using pre-optimized adstock decay rates."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    tscv = TimeSeriesSplit(n_splits=5)
    param_grid = {'alpha': np.logspace(-4, 4, 100)}
    ridge_model = Ridge()
    grid_search = GridSearchCV(ridge_model, param_grid, cv=tscv, scoring='neg_mean_squared_error')
    grid_search.fit(X_train_scaled, y_train)
    best_alpha = grid_search.best_params_['alpha']
    print(f"Optimal Alpha: {best_alpha:.4f}")
    
    final_ridge_model = Ridge(alpha=best_alpha)
    final_ridge_model.fit(X_train_scaled, y_train)
    
    y_pred = final_ridge_model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return y_pred, rmse, r2

def plot_ols_interpretation(model, X_train, y_train):
    """Plots the contribution of each channel and CI"""
    contributions = model.params.drop('const') * X_train
    contributions_lower = model.conf_int()[0].drop('const')*X_train
    contributions_upper = model.conf_int()[1].drop('const')*X_train
    total_contributions = contributions.sum()
    total_contributions_lower = contributions_lower.sum()
    total_contributions_upper = contributions_upper.sum()
    total_revenue = y_train.sum()

    print("\n--- OLS Marketing Mix Model Interpretation ---")
    print("Total Revenue Explained by Model:", total_contributions.sum())
    print("Total Actual Revenue:", total_revenue)
    print('\nPercentage contributions per channel:')

    channel_percentages = {}
    channel_percentages_lower = {}
    channel_percentages_upper = {}
    for channel in total_contributions.drop('Covid').index:
        percentage = (total_contributions[channel] / total_contributions.sum()) * 100
        percentage_lower = (total_contributions_lower[channel] / total_contributions.sum())*100
        percentage_upper = (total_contributions_upper[channel] / total_contributions.sum())*100
        channel_percentages[channel] = percentage
        channel_percentages_lower[channel] = percentage_lower
        channel_percentages_upper[channel] = percentage_upper
        print(f"{channel}: {percentage:.2f}%")

    contributions_df = pd.DataFrame(list(channel_percentages.items()), columns=['Channel', 'Contribution_Percentage'])
    contributions_df['Contribution_Percentage_lower'] = list(channel_percentages_lower.values())
    contributions_df['Contribution_Percentage_upper'] = list(channel_percentages_upper.values())

    contributions_df = contributions_df.sort_values(by='Contribution_Percentage', ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='Contribution_Percentage', y='Channel', data=contributions_df)

    x = contributions_df['Contribution_Percentage'].values
    xerr_left  = x - contributions_df['Contribution_Percentage_lower'].values
    xerr_right = contributions_df['Contribution_Percentage_upper'].values - x
    xerr = np.vstack([xerr_left, xerr_right])
    # y positions: seaborn draws bars in order
    y = np.arange(len(contributions_df))
    # add horizontal error bars
    plt.errorbar(x, y, xerr=xerr, fmt='none', ecolor='black', capsize=4, linewidth=1)
    plt.title('Percentage Revenue Contribution by Marketing Channel and 95% CI')
    plt.xlabel('Contribution (%)')
    plt.ylabel('Marketing Channel')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_predictions(y_test, y_pred_ols, y_pred_ridge):
    """Plots the actual vs. predicted revenue for both models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(y_test.index, y_test, label='Actual Revenue', color='orange')
    ax1.plot(y_pred_ols.index, y_pred_ols, label='OLS Predictions', color='green', linestyle='--')
    ax1.set_title('OLS: Actual vs. Predicted Revenue')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Revenue')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(y_test.index, y_test, label='Actual Revenue', color='orange')
    ax2.plot(y_test.index, y_pred_ridge, label='Ridge Predictions', color='purple', linestyle='--')
    ax2.set_title('Ridge: Actual vs. Predicted Revenue')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Revenue')
    ax2.legend()
    ax2.grid(True)

    fig.suptitle('Test Set vs. Predictions from OLS and Ridge Models', fontsize=16)
    plt.tight_layout()
    plt.show()