"""
Main orchestration script for ERP forecasting and trading.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
import sys

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import (
    load_erp_data,
    load_market_data,
    load_macro_features,
    load_sentiment_groq,
    load_sentiment_openai
)
from models.xgboost_model import XGBoostERPForecaster
from models.lstm_model import LSTMerpForecaster
from trading import run_trading_strategy
from performance import compute_performance_metrics
from plotting import plot_cumulative_returns_all_strategies, plot_performance_comparison


# Configuration
START_DATE = pd.Timestamp("2002-03-31")
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    """Main function."""
    print("="*80)
    print("ERP FORECASTING AND TRADING STRATEGY EVALUATION")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    erp = load_erp_data()
    equity_ret, bond_ret = load_market_data()
    macro_df = load_macro_features()
    
    # Try to load sentiment data
    try:
        sentiment_groq = load_sentiment_groq()
        print("✓ Loaded Groq sentiment data")
    except Exception as e:
        print(f"⚠ Warning: Could not load Groq sentiment: {e}")
        sentiment_groq = None
    
    try:
        sentiment_openai = load_sentiment_openai()
        print("✓ Loaded OpenAI sentiment data")
    except Exception as e:
        print(f"⚠ Warning: Could not load OpenAI sentiment: {e}")
        sentiment_openai = None
    
    # Align all data
    print("\nAligning data...")
    common_dates = erp.index.intersection(macro_df.index).intersection(equity_ret.index).intersection(bond_ret.index)
    if sentiment_groq is not None:
        common_dates = common_dates.intersection(sentiment_groq.index)
    if sentiment_openai is not None:
        common_dates = common_dates.intersection(sentiment_openai.index)
    
    erp = erp.reindex(common_dates)
    macro_df = macro_df.reindex(common_dates)
    equity_ret = equity_ret.reindex(common_dates)
    bond_ret = bond_ret.reindex(common_dates)
    if sentiment_groq is not None:
        sentiment_groq = sentiment_groq.reindex(common_dates)
    if sentiment_openai is not None:
        sentiment_openai = sentiment_openai.reindex(common_dates)
    
    print(f"✓ Aligned data: {len(common_dates)} observations")
    print(f"  Date range: {common_dates.min()} to {common_dates.max()}")
    
    # Generate forecasts
    print("\n" + "="*80)
    print("GENERATING ERP FORECASTS")
    print("="*80)
    
    strategies = {}
    
    # 1. XGBoost (macro only)
    print("\n1. Training XGBoost (macro only)...")
    try:
        xgb_model = XGBoostERPForecaster(
            n_lags=12,
            early_stopping_rounds=20
        )
        xgb_forecasts = xgb_model.forecast_rolling(
            erp, macro_df, sentiment_df=None, start_date=START_DATE
        )
        strategies['xgboost'] = {
            'forecast': xgb_forecasts,
            'model': xgb_model
        }
        print(f"✓ Generated {len(xgb_forecasts.dropna())} forecasts")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. LSTM
    print("\n2. Training LSTM...")
    try:
        lstm_model = LSTMerpForecaster(sequence_length=12)
        lstm_forecasts = lstm_model.forecast_rolling(
            erp, macro_df, sentiment_df=None, start_date=START_DATE
        )
        strategies['lstm'] = {
            'forecast': lstm_forecasts,
            'model': lstm_model
        }
        print(f"✓ Generated {len(lstm_forecasts.dropna())} forecasts")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. XGBoost with Groq sentiment
    if sentiment_groq is not None:
        print("\n3. Training XGBoost with Groq sentiment...")
        try:
            xgb_groq_model = XGBoostERPForecaster(
                n_lags=12,
                early_stopping_rounds=20
            )
            xgb_groq_forecasts = xgb_groq_model.forecast_rolling(
                erp, macro_df, sentiment_df=sentiment_groq, start_date=START_DATE
            )
            strategies['xgboost_groq'] = {
                'forecast': xgb_groq_forecasts,
                'model': xgb_groq_model
            }
            print(f"✓ Generated {len(xgb_groq_forecasts.dropna())} forecasts")
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # 4. XGBoost with OpenAI sentiment
    if sentiment_openai is not None:
        print("\n4. Training XGBoost with OpenAI sentiment...")
        try:
            xgb_openai_model = XGBoostERPForecaster(
                n_lags=12,
                early_stopping_rounds=20
            )
            xgb_openai_forecasts = xgb_openai_model.forecast_rolling(
                erp, macro_df, sentiment_df=sentiment_openai, start_date=START_DATE
            )
            strategies['xgboost_openai'] = {
                'forecast': xgb_openai_forecasts,
                'model': xgb_openai_model
            }
            print(f"✓ Generated {len(xgb_openai_forecasts.dropna())} forecasts")
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Run trading strategies
    print("\n" + "="*80)
    print("RUNNING TRADING STRATEGIES")
    print("="*80)
    
    strategy_results = {}
    metrics_rows = []
    
    for name, strategy_data in strategies.items():
        if 'forecast' not in strategy_data:
            continue
        
        forecast = strategy_data['forecast']
        forecast = forecast[forecast.index >= START_DATE]
        
        if forecast.empty:
            continue
        
        print(f"\nRunning trading strategy: {name}")
        result = run_trading_strategy(
            name=name,
            forecasts=forecast,
            equity_returns=equity_ret,
            bond_returns=bond_ret,
            min_weight=0.1,
            max_weight=0.9
        )
        
        strategy_results[name] = result
        
        # Compute metrics
        metrics = compute_performance_metrics(result.returns)
        metrics['strategy'] = name
        metrics_rows.append(metrics)
        
        # Save individual time series (with _monthly suffix)
        output_file = RESULTS_DIR / f"{name}_returns_monthly.csv"
        if not result.returns.empty and len(result.returns) > 0:
            pd.DataFrame({
                'date': result.returns.index,
                'return': result.returns.values,
                'weight': result.weights.values,
                'forecast': result.forecast.values
            }).to_csv(output_file, index=False)
            print(f"✓ Saved to {output_file}")
        else:
            print(f"⚠ Warning: No returns for {name}, skipping CSV save")
    
    # Save performance summary (with _monthly suffix)
    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_df = metrics_df.sort_values('sharpe_ratio', ascending=False)
        summary_file = RESULTS_DIR / "strategy_performance_summary_monthly.csv"
        metrics_df.to_csv(summary_file, index=False)
        print(f"\n✓ Saved performance summary to {summary_file}")
        print("\nPerformance Summary:")
        print(metrics_df[['strategy', 'sharpe_ratio', 'annualized_return', 'annualized_volatility', 'max_drawdown']].to_string(index=False))
    
    # Create plots
    print("\n" + "="*80)
    print("CREATING PLOTS")
    print("="*80)
    
    # Prepare strategy dict for plotting
    plot_strategies = {}
    for name, result in strategy_results.items():
        plot_strategies[name] = {
            'returns': result.returns,
            'weights': result.weights,
            'metrics': compute_performance_metrics(result.returns)
        }
    
    # Plot cumulative returns (with _monthly suffix)
    print("\nPlotting cumulative returns...")
    plot_cumulative_returns_all_strategies(
        plot_strategies, 
        equity_returns=equity_ret,
        bond_returns=bond_ret,
        output_dir=RESULTS_DIR,
        suffix="_monthly"
    )
    
    # Plot performance comparison (with _monthly suffix)
    print("Plotting performance comparison...")
    plot_performance_comparison(plot_strategies, output_dir=RESULTS_DIR, suffix="_monthly")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

