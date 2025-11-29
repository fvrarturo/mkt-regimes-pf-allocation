import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import List, Dict, Optional, Any

def load_daily_oil(path):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[['Date', 'Close']].rename(columns={'Close': 'oil_price'})
    df = df.set_index('Date').sort_index()
    df['oil_price'] = df['oil_price'].ffill()

    return df

def make_midas_oil_factor(daily_oil, theta=0.03, K=60):
    """
    Convert daily oil prices to monthly MIDAS factor
    using exponential weights.

    daily_oil: df with daily index
    theta: decay parameter
    K: number of daily lags to use
    """

    weights = np.exp(-theta * np.arange(K))
    weights = weights / weights.sum()

    # Rolling MIDAS convolution
    daily_oil['oil_midas'] = (
        daily_oil['oil_price']
        .rolling(K)
        .apply(lambda x: np.sum(x * weights), raw=True)
    )

    monthly = daily_oil['oil_midas'].resample('M').last()

    monthly.index = monthly.index.to_period("M").to_timestamp()

    return monthly.to_frame('oil_midas')


def merge_macro_and_oil(monthly_macro, oil_midas):
    df = monthly_macro.copy()
    df.index = pd.to_datetime(df.index)
    df = df.join(oil_midas, how='left')
    df['oil_midas'] = df['oil_midas'].ffill()

    return df


def run_midas_regression(df, target='inflation_factor', lags=3):
    """
    df: merged df with monthly factors + oil_midas
    lags: number of monthly lags for macro factors
    """
    X = pd.DataFrame(index=df.index)
    for col in ['inflation_factor','growth_factor','monetary_policy_factor','market_volatility_factor']:
        for k in range(1, lags+1):
            X[f"{col}_lag{k}"] = df[col].shift(k)

    X['oil_midas'] = df['oil_midas']

    y = df[target]

    data = pd.concat([y, X], axis=1).dropna()

    y = data[target]
    X = data.drop(columns=[target])
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    return model


def compute_midas_forecast_metrics(model, df, target='inflation_factor', horizons=[1,3,6]):
    """
    Produce RMSE / MAE / n_forecasts for MIDAS model
    Ensures predictions are aligned to the model's training sample.
    """
    results = []

    # 1) Extract the exact rows used in training
    y_train = model.model.endog
    X_train = model.model.exog
    index_train = model.model.data.row_labels  # <-- the exact index used

    preds = pd.Series(model.predict(X_train), index=index_train)

    df = df.copy()

    for h in horizons:
        # shifted true values
        df[f'{target}_future_{h}'] = df[target].shift(-h)

        # 2) align true values with prediction index
        aligned = pd.DataFrame({
            'y_pred': preds,
            'y_true': df.loc[index_train, f'{target}_future_{h}']
        }).dropna()

        rmse = np.sqrt(np.mean((aligned['y_true'] - aligned['y_pred']) ** 2))
        mae = np.mean(np.abs(aligned['y_true'] - aligned['y_pred']))
        n = len(aligned)

        results.append([h, rmse, mae, n])

    metrics_df = pd.DataFrame(results, columns=['horizon','rmse','mae','n_forecasts'])
    return metrics_df


def fit_rolling_midas_forecast(
    df: pd.DataFrame,
    target: str = 'inflation_factor',
    lags: int = 3,
    train_split: float = 0.65,
    horizons: List[int] = [1, 3, 6],
    window_size: Optional[int] = None,
    min_window: int = 36
    )  -> Dict[str, Any]:
    """
    Fit MIDAS OLS model using rolling/expanding window for each forecast origin.
    
    This prevents look-ahead bias by refitting the model at each forecast date
    using only data available up to that date.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: inflation_factor, growth_factor, 
        monetary_policy_factor, market_volatility_factor, oil_midas
    target : str
        Target variable (e.g., 'inflation_factor')
    lags : int
        Number of lags for macro factors
    train_split : float
        Fraction of data for initial training period (0-1)
    horizons : List[int]
        Forecast horizons (months)
    window_size : Optional[int]
        Rolling window size. If None, uses expanding window.
    min_window : int
        Minimum observations required to fit
    
    Returns:
    --------
    dict with keys:
        - 'forecasts': pd.DataFrame indexed by test_date, columns for each horizon
        - 'metrics': RMSE/MAE summary by horizon
        - 'models': dict of fitted models at each test_date (for inspection)
    """
    
    # Split into train and test periods
    n_total = len(df)
    n_train = int(n_total * train_split)
    
    train_data = df.iloc[:n_train].copy()
    test_dates = df.iloc[n_train:].index
    
    # Initialize output
    forecasts = pd.DataFrame(
        index=test_dates,
        columns=[f'h_{h}' for h in horizons]
    )
    
    models_dict = {}
    
    print(f"\nFitting rolling MIDAS OLS for {len(test_dates)} forecast origins...")
    
    for idx, forecast_date in enumerate(test_dates):
        if (idx + 1) % 20 == 0 or idx == 0:
            print(f"  Forecast origin {idx + 1}/{len(test_dates)}: {forecast_date.date()}")
        
        # Get all data available up to forecast_date
        available_data = df.loc[:forecast_date].copy()
        
        # Apply window (expanding or rolling)
        if window_size is None:
            # Expanding window: use all available data
            window_data = available_data
        else:
            # Rolling window: use last window_size observations
            if len(available_data) > window_size:
                window_data = available_data.iloc[-window_size:].copy()
            else:
                window_data = available_data
        
        # Check minimum window requirement
        if len(window_data) < min_window:
            print(f"    Warning: Insufficient data ({len(window_data)} < {min_window})")
            continue
        
        try:
            # Construct feature matrix for this window
            X = pd.DataFrame(index=window_data.index)
            
            # Add lagged macro factors
            for col in ['inflation_factor', 'growth_factor', 'monetary_policy_factor', 'market_volatility_factor']:
                for k in range(1, lags + 1):
                    X[f"{col}_lag{k}"] = window_data[col].shift(k)
            
            # Add oil MIDAS factor
            X['oil_midas'] = window_data['oil_midas']
            
            # Target variable
            y = window_data[target]
            
            # Combine and drop NAs
            data = pd.concat([y, X], axis=1).dropna()
            y_clean = data[target]
            X_clean = data.drop(columns=[target])
            X_clean = sm.add_constant(X_clean)
            
            # Fit OLS model
            model = sm.OLS(y_clean, X_clean).fit()
            models_dict[forecast_date] = model
            
            # Generate forecasts for this forecast_date
            # For each horizon, use the features at forecast_date to predict
            try:
                # Check if forecast_date is in X_clean index
                if forecast_date in X_clean.index:
                    feat_row = X_clean.loc[[forecast_date]]  # Get features at forecast_date
                    
                    if len(feat_row) > 0:
                        for h in horizons:
                            try:
                                forecast_val = model.predict(feat_row).values[0]
                                forecasts.loc[forecast_date, f'h_{h}'] = forecast_val
                            except Exception as e:
                                forecasts.loc[forecast_date, f'h_{h}'] = np.nan
                else:
                    # forecast_date not in features due to lags, skip
                    for h in horizons:
                        forecasts.loc[forecast_date, f'h_{h}'] = np.nan
            except Exception as e:
                for h in horizons:
                    forecasts.loc[forecast_date, f'h_{h}'] = np.nan
        
        except Exception as e:
            print(f"    Error fitting model at {forecast_date.date()}: {e}")
            continue
    
    # Compute metrics - correct h-step ahead evaluation
    print(f"\nTotal forecast origins: {len(forecasts)}")
    print(f"Non-NaN forecasts per horizon:")
    for h in horizons:
        n_valid = forecasts[f'h_{h}'].notna().sum()
        print(f"  h={h}: {n_valid} valid forecasts")
    
    metrics_list = []
    for h in horizons:
        preds = []
        actuals = []
        
        # For each forecast date, align with h-steps-ahead actual value
        for forecast_date in forecasts.index:
            pred_val = forecasts.loc[forecast_date, f'h_{h}']
            
            # h-step ahead actual value
            target_date = forecast_date + pd.DateOffset(months=h)
            
            if target_date in df.index:
                actual_val = df.loc[target_date, target]
                
                if pd.notna(pred_val) and pd.notna(actual_val):
                    preds.append(pred_val)
                    actuals.append(actual_val)
        
        preds = np.array(preds)
        actuals = np.array(actuals)
        
        if len(preds) > 0:
            rmse = np.sqrt(np.mean((actuals - preds) ** 2))
            mae = np.mean(np.abs(actuals - preds))
            n = len(preds)
            
            print(f"  h={h}: {n} aligned pairs, RMSE={rmse:.6f}, MAE={mae:.6f}")
            
            metrics_list.append({
                'horizon': h,
                'rmse': rmse,
                'mae': mae,
                'n_forecasts': n
            })
    
    metrics_df = pd.DataFrame(metrics_list)
    
    return {
        'forecasts': forecasts,
        'metrics': metrics_df,
        'models': models_dict
    }



# Step 0: load macro_final
monthly_macro = pd.read_csv("/Users/zhangxiaojie/Desktop/MIT 25 Fall/mkt-regimes-pf-allocation/main_project/data/macro_final/final_macro.csv", 
                            index_col=0, parse_dates=True)

# Step 1: load daily oil
daily_oil = load_daily_oil("/Users/zhangxiaojie/Desktop/MIT 25 Fall/mkt-regimes-pf-allocation/main_project/data/macro_processed/daily_factors/daily_wti.csv")

# Step 2: create MIDAS oil factor
oil_midas = make_midas_oil_factor(daily_oil, theta=0.03, K=60)

# Step 3: merge
macro_with_oil = merge_macro_and_oil(monthly_macro, oil_midas)

# Step 4: run rolling MIDAS regression with expanding window
result = fit_rolling_midas_forecast(
    macro_with_oil,
    target='inflation_factor',
    lags=3,
    train_split=0.65,
    window_size=None,  # expanding window
    min_window=36
)

print("\nMIDAS Forecast Metrics (Inflation Factor):")
print(result['metrics'])
print("\n")
result['metrics'].to_csv("midas_inflation_forecast_metrics.csv", index=False)
