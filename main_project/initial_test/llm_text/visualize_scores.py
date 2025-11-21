"""Enhanced visualization and analysis of sentiment scores with macro data correlation."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Optional imports
try:
    import seaborn as sns
    sns.set_style("whitegrid")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn not available, using matplotlib defaults")

try:
    from scipy.stats import pearsonr, spearmanr, gaussian_kde
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available, some statistics will be skipped")
    # Fallback functions
    def pearsonr(x, y):
        return np.corrcoef(x, y)[0, 1], 0.0
    def spearmanr(x, y):
        from scipy.stats import rankdata
        return pearsonr(rankdata(x), rankdata(y))

plt.rcParams['figure.dpi'] = 150

# Paths
script_dir = Path(__file__).parent
csv_path = script_dir / "sentiment_scores.csv"
# Macro data is at main_project/data/macro (script is at main_project/initial_test/llm_text/)
macro_dir = script_dir.parent.parent / "data" / "macro"

# Load sentiment scores
print("Loading sentiment scores...")
df = pd.read_csv(csv_path)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

print(f"Loaded {len(df)} sentiment score observations")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# Sentiment columns
sentiment_cols = [
    'inflation_sentiment',
    'ec_growth_sentiment', 
    'monetary_policy_sentiment',
    'market_vol_sentiment'
]

# ============================================================================
# 1. DESCRIPTIVE STATISTICS
# ============================================================================
print("\n" + "="*80)
print("DESCRIPTIVE STATISTICS")
print("="*80)

stats_dict = {}
for col in sentiment_cols:
    stats_dict[col] = {
        'count': df[col].count(),
        'mean': df[col].mean(),
        'std': df[col].std(),
        'min': df[col].min(),
        '25%': df[col].quantile(0.25),
        '50%': df[col].quantile(0.50),
        '75%': df[col].quantile(0.75),
        'max': df[col].max(),
        'skewness': df[col].skew(),
        'kurtosis': df[col].kurtosis(),
        'positive_pct': (df[col] > 0).sum() / len(df) * 100,
        'negative_pct': (df[col] < 0).sum() / len(df) * 100,
        'zero_pct': (df[col] == 0).sum() / len(df) * 100,
    }

stats_df = pd.DataFrame(stats_dict).T
print("\nSummary Statistics:")
print(stats_df.round(3))

# Save statistics
stats_df.to_csv(script_dir / 'sentiment_statistics.csv')
print(f"\nStatistics saved to: {script_dir / 'sentiment_statistics.csv'}")

# ============================================================================
# 2. TIME SERIES PLOTS WITH ROLLING AVERAGES
# ============================================================================
print("\n" + "="*80)
print("CREATING TIME SERIES VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Sentiment Scores Over Time with Rolling Averages', fontsize=16, fontweight='bold')

dimensions = [
    ('inflation_sentiment', 'Inflation Sentiment', axes[0, 0]),
    ('ec_growth_sentiment', 'Economic Growth Sentiment', axes[0, 1]),
    ('monetary_policy_sentiment', 'Monetary Policy Sentiment', axes[1, 0]),
    ('market_vol_sentiment', 'Market Volatility Sentiment', axes[1, 1]),
]

for col, title, ax in dimensions:
    # Raw data
    ax.plot(df['date'], df[col], alpha=0.3, linewidth=1, label='Weekly scores', color='lightblue')
    
    # Rolling averages
    rolling_4w = df[col].rolling(window=4, min_periods=1).mean()
    rolling_12w = df[col].rolling(window=12, min_periods=1).mean()
    ax.plot(df['date'], rolling_4w, linewidth=2, label='4-week MA', color='blue')
    ax.plot(df['date'], rolling_12w, linewidth=2, label='12-week MA', color='darkblue')
    
    # Zero line
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Formatting
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Sentiment Score', fontsize=10)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    
    # Add statistics text
    mean_val = df[col].mean()
    std_val = df[col].std()
    ax.text(0.02, 0.98, f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
            fontsize=9)

plt.tight_layout()
plt.savefig(script_dir / 'sentiment_timeseries.png', dpi=150, bbox_inches='tight')
print(f"Time series plot saved to: {script_dir / 'sentiment_timeseries.png'}")

# ============================================================================
# 3. DISTRIBUTION PLOTS
# ============================================================================
print("\nCreating distribution plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Distribution of Sentiment Scores', fontsize=16, fontweight='bold')

# Redefine dimensions for this figure
dimensions_dist = [
    ('inflation_sentiment', 'Inflation Sentiment', axes[0, 0]),
    ('ec_growth_sentiment', 'Economic Growth Sentiment', axes[0, 1]),
    ('monetary_policy_sentiment', 'Monetary Policy Sentiment', axes[1, 0]),
    ('market_vol_sentiment', 'Market Volatility Sentiment', axes[1, 1]),
]

for col, title, ax in dimensions_dist:
    # Get data
    data = df[col].dropna()
    
    if len(data) > 0:
        # Use adaptive binning - fewer bins if data is sparse
        n_bins = 100
        
        # Histogram with density
        counts, bins, patches = ax.hist(data, bins=n_bins, alpha=0.7, density=True, 
                                        color='steelblue', edgecolor='black', linewidth=0.5)
        
        # Ensure we can see the histogram
        if counts.max() > 0:
            ax.set_ylim(0, counts.max() * 1.15)
        
        # KDE overlay (only if we have enough unique values)
        if HAS_SCIPY and len(data) > 2 and data.nunique() > 3:
            try:
                kde = gaussian_kde(data)
                x_range = np.linspace(data.min(), data.max(), 200)
                kde_vals = kde(x_range)
                ax.plot(x_range, kde_vals, 'r-', linewidth=2, label='KDE', zorder=5)
            except Exception as e:
                pass  # Silently skip KDE if it fails
        
        # Mean and median lines
        mean_val = data.mean()
        median_val = data.median()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_val:.3f}', zorder=4)
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, 
                  label=f'Median: {median_val:.3f}', zorder=4)
        
        # Set axis limits with some padding
        x_min, x_max = data.min(), data.max()
        x_range = x_max - x_min
        if x_range > 0:
            ax.set_xlim(x_min - x_range * 0.05, x_max + x_range * 0.05)
        else:
            ax.set_xlim(x_min - 0.1, x_max + 0.1)
        
        # Add statistics text
        stats_text = f'N={len(data)}\nStd={data.std():.3f}'
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Sentiment Score', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.grid(True, alpha=0.3, zorder=0)
    if len(data) > 0:
        ax.legend(fontsize=8, loc='best', framealpha=0.9)

plt.tight_layout()
plt.savefig(script_dir / 'sentiment_distributions.png', dpi=150, bbox_inches='tight')
print(f"Distribution plot saved to: {script_dir / 'sentiment_distributions.png'}")

# ============================================================================
# 4. CORRELATION MATRIX BETWEEN SENTIMENT SCORES
# ============================================================================
print("\nCreating sentiment correlation matrix...")

fig, ax = plt.subplots(figsize=(10, 8))
corr_matrix = df[sentiment_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

if HAS_SEABORN:
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, mask=mask,
                xticklabels=[c.replace('_sentiment', '').replace('_', ' ').title() for c in sentiment_cols],
                yticklabels=[c.replace('_sentiment', '').replace('_', ' ').title() for c in sentiment_cols],
                ax=ax)
else:
    # Fallback to matplotlib
    im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(range(len(sentiment_cols)))
    ax.set_yticks(range(len(sentiment_cols)))
    ax.set_xticklabels([c.replace('_sentiment', '').replace('_', ' ').title() for c in sentiment_cols])
    ax.set_yticklabels([c.replace('_sentiment', '').replace('_', ' ').title() for c in sentiment_cols])
    for i in range(len(sentiment_cols)):
        for j in range(len(sentiment_cols)):
            if not mask[i, j]:
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.3f}',
                             ha="center", va="center", color="black", fontsize=10)
    plt.colorbar(im, ax=ax)

ax.set_title('Correlation Matrix: Sentiment Scores', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(script_dir / 'sentiment_correlation_matrix.png', dpi=150, bbox_inches='tight')
print(f"Correlation matrix saved to: {script_dir / 'sentiment_correlation_matrix.png'}")

# ============================================================================
# 5. LOAD AND ALIGN MACRO DATA
# ============================================================================
print("\n" + "="*80)
print("LOADING MACRO DATA FOR CORRELATION ANALYSIS")
print("="*80)

def load_macro_data(file_path, value_col=None):
    """Load macro data file and return standardized dataframe."""
    try:
        df_macro = pd.read_csv(file_path)
        if 'observation_date' in df_macro.columns:
            df_macro['date'] = pd.to_datetime(df_macro['observation_date'])
        elif 'date' in df_macro.columns:
            df_macro['date'] = pd.to_datetime(df_macro['date'])
        else:
            return None
        
        # Get value column
        if value_col:
            if value_col in df_macro.columns:
                value_col_name = value_col
            else:
                return None
        else:
            # Auto-detect value column (first non-date column)
            value_col_name = [c for c in df_macro.columns if c not in ['date', 'observation_date']][0]
        
        df_macro = df_macro[['date', value_col_name]].copy()
        df_macro = df_macro.rename(columns={value_col_name: 'value'})
        df_macro = df_macro.dropna()
        df_macro = df_macro.sort_values('date')
        
        return df_macro
    except Exception as e:
        print(f"  Error loading {file_path.name}: {e}")
        return None

# Define macro variables to load for each sentiment category
macro_mapping = {
    'inflation_sentiment': [
        ('inflation/cpi.csv', 'CPIAUCSL'),
        ('inflation/PPI_inflation.csv', None),
        ('inflation/PCE_price_index.csv', None),
    ],
    'ec_growth_sentiment': [
        ('ec_growth/gdp.csv', 'GDP'),
        ('ec_growth/real_gdp.csv', None),
        ('ec_growth/industrial_production.csv', None),
        ('ec_growth/retail_sales.csv', None),
        ('ec_growth/unemployment.csv', None),
    ],
    'monetary_policy_sentiment': [
        ('mon_policy/fedfunds.csv', 'FEDFUNDS'),
        ('mon_policy/fed_reserve_discount_rate.csv', None),
        ('mon_policy/10y_treasury_const_maturity_rate.csv', None),
        ('mon_policy/m2_real_money_supply.csv', None),
    ],
    'market_vol_sentiment': [
        ('mkt_vol/vix.csv', 'VIXCLS'),
        ('mkt_vol/3month_vol_index_sp500.csv', None),
        ('mkt_vol/nasdaq_vol_indx.csv', None),
        ('mkt_vol/nat_fin_condition_indx.csv', None),
    ],
}

# Load all macro data
macro_data = {}
if not macro_dir.exists():
    print(f"  WARNING: Macro directory not found at {macro_dir}")
    print(f"  Please check the path. Expected: main_project/data/macro")
else:
    print(f"  Macro directory found: {macro_dir}")
    for sentiment_col, file_list in macro_mapping.items():
        macro_data[sentiment_col] = {}
        for file_path, value_col in file_list:
            full_path = macro_dir / file_path
            if full_path.exists():
                df_macro = load_macro_data(full_path, value_col)
                if df_macro is not None and len(df_macro) > 0:
                    var_name = Path(file_path).stem
                    macro_data[sentiment_col][var_name] = df_macro
                    print(f"  Loaded {var_name}: {len(df_macro)} observations ({df_macro['date'].min()} to {df_macro['date'].max()})")
                else:
                    print(f"  Failed to load {file_path.name}")
            else:
                print(f"  File not found: {full_path}")

# ============================================================================
# 6. CORRELATION ANALYSIS WITH MACRO VARIABLES
# ============================================================================
print("\n" + "="*80)
print("CORRELATION ANALYSIS WITH MACRO VARIABLES")
print("="*80)

def align_and_correlate(sentiment_series, macro_df, lag_weeks=0):
    """Align sentiment and macro data and calculate correlation."""
    try:
        # Ensure sentiment series has datetime index
        if not isinstance(sentiment_series.index, pd.DatetimeIndex):
            return None, None, None
        
        # Convert sentiment to monthly (take last value of month)
        sentiment_monthly = sentiment_series.resample('M').last()
        
        # Ensure macro_df has date column and convert to datetime index
        if 'date' not in macro_df.columns:
            return None, None, None
        
        macro_df = macro_df.copy()
        macro_df['date'] = pd.to_datetime(macro_df['date'])
        macro_monthly = macro_df.set_index('date').resample('M').last()
        
        # Merge on date index
        merged = pd.merge(sentiment_monthly.to_frame('sentiment'), 
                         macro_monthly[['value']], 
                         left_index=True, right_index=True, how='inner')
        
        if len(merged) < 10:  # Need minimum observations
            return None, None, None
        
        # Remove any remaining NaN values
        merged = merged.dropna()
        
        if len(merged) < 10:
            return None, None, None
        
        sentiment_vals = merged['sentiment'].values
        macro_vals = merged['value'].values
        
        # Calculate correlations
        if HAS_SCIPY:
            pearson_r, pearson_p = pearsonr(sentiment_vals, macro_vals)
            spearman_r, spearman_p = spearmanr(sentiment_vals, macro_vals)
        else:
            # Fallback: use numpy correlation
            pearson_r = np.corrcoef(sentiment_vals, macro_vals)[0, 1]
            pearson_p = 0.0  # Can't calculate p-value without scipy
            spearman_r = pearson_r  # Approximate
            spearman_p = 0.0
        
        return {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'n_obs': len(merged),
            'date_range': (merged.index.min(), merged.index.max())
        }, merged, macro_vals
    except Exception as e:
        print(f"    Error in correlation calculation: {e}")
        return None, None, None

# Store correlation results
correlation_results = []

for sentiment_col in sentiment_cols:
    print(f"\n{sentiment_col.replace('_sentiment', '').replace('_', ' ').title()}:")
    print("-" * 60)
    
    sentiment_series = df.set_index('date')[sentiment_col]
    
    if sentiment_col in macro_data and len(macro_data[sentiment_col]) > 0:
        for var_name, macro_df in macro_data[sentiment_col].items():
            # Try correlation
            corr_info, merged_df, macro_vals = align_and_correlate(sentiment_series, macro_df, lag_weeks=0)
            
            if corr_info and corr_info['n_obs'] >= 10:
                print(f"  {var_name}:")
                print(f"    Pearson r={corr_info['pearson_r']:.3f} (p={corr_info['pearson_p']:.4f})")
                print(f"    Spearman r={corr_info['spearman_r']:.3f} (p={corr_info['spearman_p']:.4f})")
                print(f"    N={corr_info['n_obs']} observations")
                print(f"    Date range: {corr_info['date_range'][0].strftime('%Y-%m-%d')} to {corr_info['date_range'][1].strftime('%Y-%m-%d')}")
                
                correlation_results.append({
                    'sentiment': sentiment_col,
                    'macro_variable': var_name,
                    'pearson_r': corr_info['pearson_r'],
                    'pearson_p': corr_info['pearson_p'],
                    'spearman_r': corr_info['spearman_r'],
                    'spearman_p': corr_info['spearman_p'],
                    'n_obs': corr_info['n_obs'],
                })
            else:
                print(f"  {var_name}: Insufficient overlapping data (need >=10 observations)")
    else:
        print(f"  No macro data available for this sentiment category")

# Save correlation results
if correlation_results:
    corr_df = pd.DataFrame(correlation_results)
    corr_df = corr_df.sort_values('pearson_r', key=abs, ascending=False)
    corr_df.to_csv(script_dir / 'sentiment_macro_correlations.csv', index=False)
    print(f"\nCorrelation results saved to: {script_dir / 'sentiment_macro_correlations.csv'}")

# ============================================================================
# 7. CORRELATION PLOTS WITH KEY MACRO VARIABLES
# ============================================================================
print("\n" + "="*80)
print("CREATING CORRELATION PLOTS")
print("="*80)

# Select top correlations for visualization
if correlation_results:
    corr_df_viz = pd.DataFrame(correlation_results)
    # Sort by absolute value of pearson_r and take top 8
    corr_df_viz['abs_pearson_r'] = corr_df_viz['pearson_r'].abs()
    top_corrs = corr_df_viz.nlargest(8, 'abs_pearson_r')
    
    n_plots = len(top_corrs)
    if n_plots > 0:
        n_cols = 2
        n_rows = (n_plots + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        fig.suptitle('Sentiment Scores vs Macro Variables (Top Correlations)', 
                     fontsize=16, fontweight='bold')
        
        for plot_idx, (_, row) in enumerate(top_corrs.iterrows()):
            sentiment_col = row['sentiment']
            var_name = row['macro_variable']
            
            # Get data
            sentiment_series = df.set_index('date')[sentiment_col]
            macro_df = macro_data[sentiment_col][var_name]
            
            _, merged_df, _ = align_and_correlate(sentiment_series, macro_df)
            
            if merged_df is not None and len(merged_df) > 0:
                ax = axes[plot_idx] if n_plots > 1 else axes[0]
                
                # Create scatter plot
                sentiment_vals = merged_df['sentiment'].values
                macro_vals = merged_df['value'].values
                
                ax.scatter(macro_vals, sentiment_vals, alpha=0.6, s=50)
                
                # Add trend line
                z = np.polyfit(macro_vals, sentiment_vals, 1)
                p = np.poly1d(z)
                x_line = np.linspace(macro_vals.min(), macro_vals.max(), 100)
                ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
                
                # Labels
                ax.set_xlabel(f'{var_name.replace("_", " ").title()}', fontsize=10)
                ax.set_ylabel(f'{sentiment_col.replace("_sentiment", "").replace("_", " ").title()}', fontsize=10)
                
                # Title with correlation
                title = f"{sentiment_col.replace('_sentiment', '').replace('_', ' ').title()} vs {var_name.replace('_', ' ').title()}\n"
                title += f"r={row['pearson_r']:.3f} (p={row['pearson_p']:.4f})"
                ax.set_title(title, fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        if n_plots > 1:
            for idx in range(n_plots, len(axes)):
                axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(script_dir / 'sentiment_macro_correlations.png', dpi=150, bbox_inches='tight')
        print(f"Correlation plots saved to: {script_dir / 'sentiment_macro_correlations.png'}")
    else:
        print("  No correlations found to plot")
else:
    print("  No correlation results available")

# ============================================================================
# 8. SUMMARY REPORT
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"\nGenerated files:")
print(f"  1. sentiment_statistics.csv - Descriptive statistics")
print(f"  2. sentiment_timeseries.png - Time series with rolling averages")
print(f"  3. sentiment_distributions.png - Distribution plots")
print(f"  4. sentiment_correlation_matrix.png - Correlation between sentiment scores")
print(f"  5. sentiment_macro_correlations.csv - Correlation results with macro variables")
if correlation_results:
    print(f"  6. sentiment_macro_correlations.png - Scatter plots of top correlations")
print(f"\nAll files saved to: {script_dir}")
