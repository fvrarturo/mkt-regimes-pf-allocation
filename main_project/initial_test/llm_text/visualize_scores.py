"""Quick visualization of sentiment scores to check for oscillation."""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load scores
csv_path = Path(__file__).parent / "sentiment_scores.csv"
df = pd.read_csv(csv_path)
df['date'] = pd.to_datetime(df['date'])

# Plot all four sentiment dimensions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Sentiment Scores Over Time', fontsize=16, fontweight='bold')

dimensions = [
    ('inflation_sentiment', 'Inflation Sentiment', axes[0, 0]),
    ('ec_growth_sentiment', 'Economic Growth Sentiment', axes[0, 1]),
    ('monetary_policy_sentiment', 'Monetary Policy Sentiment', axes[1, 0]),
    ('market_vol_sentiment', 'Market Volatility Sentiment', axes[1, 1]),
]

for col, title, ax in dimensions:
    ax.plot(df['date'], df[col], marker='o', linewidth=2, markersize=8)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sentiment Score')
    ax.set_ylim(-1, 1)
    ax.grid(True, alpha=0.3)
    
    # Calculate week-to-week changes
    changes = df[col].diff().abs()
    max_change = changes.max()
    ax.text(0.02, 0.98, f'Max week-to-week change: {max_change:.2f}', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(csv_path.parent / 'sentiment_scores_visualization.png', dpi=150, bbox_inches='tight')
print(f"Visualization saved to {csv_path.parent / 'sentiment_scores_visualization.png'}")
print("\nWeek-to-week changes summary:")
print(df[['date'] + [d[0] for d in dimensions]].set_index('date').diff().abs().describe())

