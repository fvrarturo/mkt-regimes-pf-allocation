import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Define macroeconomic keyword categories (excluding Regional/Country)
CATEGORIES = {
    'Inflation & Prices': [
        'inflation', 'deflation', 'disinflation', 'stagflation', 'price', 'prices', 'cpi', 'ppi',
        'consumer', 'wholesale', 'deflationary', 'inflationary'
    ],
    'Interest Rates & Monetary Policy': [
        'interest', 'rate', 'rates', 'monetary', 'policy', 'central', 'bank', 'federal', 'reserve', 'fed',
        'fomc', 'fedfunds', 'federal funds', 'discount', 'prime', 'yield', 'yields', 'treasury',
        'bond', 'bonds', 'tbill', 't-bill', 'tnote'
    ],
    'Economic Growth & Output': [
        'growth', 'gdp', 'recession', 'recovery', 'expansion', 'contraction', 'slowdown',
        'economic', 'economy', 'output', 'production', 'industrial', 'manufacturing'
    ],
    'Employment & Labor': [
        'employment', 'unemployment', 'jobs', 'labor', 'labour', 'wage', 'wages', 'payroll',
        'nonfarm', 'non-farm', 'unemployment rate'
    ],
    'Trade & Commerce': [
        'trade', 'trading', 'export', 'exports', 'import', 'imports', 'deficit', 'surplus',
        'balance', 'commerce', 'tariff', 'tariffs', 'tarrif'
    ],
    'Fiscal Policy & Government': [
        'fiscal', 'government', 'debt', 'deficit', 'surplus', 'budget', 'spending', 'tax', 'taxes',
        'treasury', 'congress', 'senate', 'house'
    ],
    'Markets & Finance': [
        'market', 'markets', 'stock', 'stocks', 'equity', 'equities', 'dollar', 'currency',
        'forex', 'exchange', 'dow', 'nasdaq', 'sp500', 's&p', 'vix', 'volatility'
    ],
    'Investment & Capital': [
        'investment', 'investor', 'investors', 'capital', 'equity', 'credit', 'lending',
        'borrowing', 'loan', 'loans'
    ],
    'Business & Corporate': [
        'business', 'corporate', 'earnings', 'profit', 'profits', 'revenue', 'revenues',
        'company', 'companies', 'firm', 'firms'
    ],
    'Commodities & Energy': [
        'oil', 'crude', 'energy', 'gas', 'gasoline', 'gold', 'silver', 'commodity', 'commodities',
        'petroleum', 'natural gas'
    ],
    'Housing & Real Estate': [
        'housing', 'home', 'homes', 'real estate', 'mortgage', 'mortgages', 'construction',
        'builder', 'builders'
    ],
    'Consumer & Spending': [
        'consumer', 'consumption', 'spending', 'retail', 'sales', 'demand', 'supply',
        'purchasing', 'purchase'
    ],
    'Financial Stability & Risk': [
        'crisis', 'crises', 'risk', 'risks', 'stability', 'volatility', 'uncertainty',
        'recession', 'depression', 'bubble', 'bubbles'
    ]
}

def count_keywords_in_text(text_series, keywords):
    """Count occurrences of keywords in text using substring matching"""
    text_lower = text_series.str.lower()
    total_count = 0
    
    for keyword in keywords:
        # Use regex pattern: not preceded by lowercase letter, keyword, not followed by lowercase letter
        # This catches both standalone words and concatenated words
        pattern = r'(?<![a-z])' + re.escape(keyword.lower()) + r'(?![a-z])'
        count = text_lower.str.contains(pattern, regex=True, na=False).sum()
        total_count += count
    
    return total_count

def analyze_category(df, category_name, keywords):
    """Analyze a single category"""
    print(f"\nAnalyzing category: {category_name}")
    
    # Prepare date columns
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['year_month'] = df['date'].dt.to_period('M')
    
    # Combine text
    df['combined_text'] = (df['headline'].fillna('') + ' ' + df['snippet'].fillna('')).str.lower()
    
    # Count articles per year (for normalization)
    articles_per_year = df.groupby('year').size()
    
    # Count total occurrences by year
    yearly_counts_raw = df.groupby('year')['combined_text'].apply(
        lambda x: count_keywords_in_text(x, keywords)
    )
    
    # Normalize by articles per year (frequency per article)
    yearly_counts_normalized = yearly_counts_raw / articles_per_year
    
    # Count total occurrences by month (for recent years)
    recent_years = df[df['year'] >= 2020]
    articles_per_month = recent_years.groupby('year_month').size()
    monthly_counts_raw = recent_years.groupby('year_month')['combined_text'].apply(
        lambda x: count_keywords_in_text(x, keywords)
    )
    monthly_counts_normalized = monthly_counts_raw / articles_per_month
    
    # Count individual keywords
    keyword_counts = {}
    text_lower = df['combined_text']
    for keyword in keywords:
        pattern = r'(?<![a-z])' + re.escape(keyword.lower()) + r'(?![a-z])'
        count = text_lower.str.contains(pattern, regex=True, na=False).sum()
        if count > 0:
            keyword_counts[keyword] = count
    
    return yearly_counts_raw, yearly_counts_normalized, monthly_counts_normalized, keyword_counts, articles_per_year

def create_category_plot(category_name, yearly_counts_raw, yearly_counts_normalized, monthly_counts_normalized, keyword_counts, articles_per_year, output_dir):
    """Create visualization for a single category"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{category_name} - Keyword Analysis', fontsize=16, fontweight='bold')
    
    # 1. Yearly trend (normalized by articles per year)
    ax1 = axes[0, 0]
    ax1.plot(yearly_counts_normalized.index, yearly_counts_normalized.values, marker='o', linewidth=2, markersize=6, color='steelblue')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Frequency per Article')
    ax1.set_title('Yearly Trend (Normalized by Articles per Year)')
    ax1.grid(True, alpha=0.3)
    
    # 2. Monthly trend (recent years, normalized)
    ax2 = axes[0, 1]
    if len(monthly_counts_normalized) > 0:
        monthly_counts_normalized.plot(ax=ax2, marker='o', linewidth=1.5, markersize=3, color='coral')
        ax2.set_xlabel('Year-Month')
        ax2.set_ylabel('Frequency per Article')
        ax2.set_title('Monthly Trend - Normalized (2020-2025)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No recent data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Monthly Trend (2020-2025)')
    
    # 3. Top keywords bar chart
    ax3 = axes[1, 0]
    if keyword_counts:
        sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        keywords, counts = zip(*sorted_keywords)
        ax3.barh(range(len(keywords)), counts, color='steelblue')
        ax3.set_yticks(range(len(keywords)))
        ax3.set_yticklabels(keywords)
        ax3.set_xlabel('Total Frequency')
        ax3.set_title('Top Keywords in Category')
        ax3.invert_yaxis()
    else:
        ax3.text(0.5, 0.5, 'No keywords found', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Top Keywords in Category')
    
    # 4. Statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    peak_year = yearly_counts_normalized.idxmax()
    peak_value = yearly_counts_normalized.max()
    recent_value = yearly_counts_normalized.get(2024, 0)
    stats_text = f"""
    Category Statistics:
    
    Total Occurrences: {yearly_counts_raw.sum():,}
    Avg per Article: {yearly_counts_normalized.mean():.4f}
    Peak Year: {peak_year} ({peak_value:.4f} per article)
    Recent (2024): {recent_value:.4f} per article
    
    Unique Keywords: {len(keyword_counts)}
    Top Keyword: {max(keyword_counts.items(), key=lambda x: x[1])[0] if keyword_counts else 'N/A'}
    """
    ax4.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center', family='monospace')
    
    plt.tight_layout()
    
    # Save figure
    filename = category_name.lower().replace(' ', '_').replace('&', 'and')
    filepath = os.path.join(output_dir, f'{filename}_analysis.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {filepath}")
    
    return yearly_counts_raw.sum()


def create_combined_trends_plot(category_yearly_normalized, category_totals, output_dir):
    """Create combined plot showing all categories' trends over time (normalized by articles)"""
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    fig.suptitle('All Macroeconomic Categories - Combined Trends Over Time (Normalized by Articles per Year)', fontsize=16, fontweight='bold')
    
    # Get color palette (consistent across all plots)
    sorted_categories = sorted(category_totals.items(), key=lambda x: x[1], reverse=True)
    categories_list, totals_list = zip(*sorted_categories)
    colors = plt.cm.tab20(np.linspace(0, 1, len(categories_list)))
    color_map = {cat: colors[i] for i, cat in enumerate(categories_list)}
    
    # 1. Yearly trends (normalized by articles per year) - top left, spans 2 columns
    ax1 = fig.add_subplot(gs[0, :])
    for category_name, yearly_counts in category_yearly_normalized.items():
        ax1.plot(yearly_counts.index, yearly_counts.values, 
                marker='o', linewidth=2, markersize=4, 
                label=category_name, color=color_map[category_name], alpha=0.8)
    
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Frequency per Article', fontsize=12)
    ax1.set_title('Yearly Trends - Normalized by Articles per Year (1990-2025)', fontsize=14)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # 2. Normalized percentage view (each category as % of total frequency per year)
    ax2 = fig.add_subplot(gs[1, :])
    
    # Calculate percentage of total for each year
    yearly_percentages = {}
    for category_name, yearly_counts in category_yearly_normalized.items():
        yearly_percentages[category_name] = yearly_counts
    
    # Convert to DataFrame and calculate percentages
    yearly_df = pd.DataFrame(yearly_percentages)
    yearly_df_pct = yearly_df.div(yearly_df.sum(axis=1), axis=0) * 100
    
    for category_name in yearly_df_pct.columns:
        ax2.plot(yearly_df_pct.index, yearly_df_pct[category_name].values,
                marker='o', linewidth=2, markersize=4,
                label=category_name, color=color_map[category_name], alpha=0.8)
    
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Percentage of Total Frequency (%)', fontsize=12)
    ax2.set_title('Yearly Trends - Percentage of Total Frequency per Year (1990-2025)', fontsize=14)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3)
    
    # 3. Pie chart showing overall distribution - bottom left
    ax3 = fig.add_subplot(gs[2, 0])
    
    # Prepare pie chart data with labels showing both percentage and count
    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            # Only show percentage if significant enough
            if pct > 1.5:
                return f'{pct:.1f}%'
            else:
                return ''
        return my_autopct
    
    # Create labels - show category name and count, but only for slices > 2%
    labels = []
    for i, cat in enumerate(categories_list):
        pct = (totals_list[i] / sum(totals_list)) * 100
        if pct > 2:
            # Shorten long category names for better display
            short_name = cat.replace(' & ', ' &\n').replace(' and ', ' &\n')
            labels.append(f'{short_name}\n{totals_list[i]:,}')
        else:
            labels.append('')  # Empty label for small slices
    
    wedges, texts, autotexts = ax3.pie(totals_list, labels=labels, autopct=make_autopct(totals_list),
                                       startangle=90, colors=colors, 
                                       textprops={'fontsize': 7, 'ha': 'center'},
                                       pctdistance=0.85)
    
    # Adjust text positions for small slices to avoid overlap
    for i, (wedge, text, autotext) in enumerate(zip(wedges, texts, autotexts)):
        pct = (totals_list[i] / sum(totals_list)) * 100
        if pct < 2:  # Less than 2%
            # Move text further out and make smaller
            angle = np.deg2rad((wedge.theta2 + wedge.theta1) / 2)
            x = 1.4 * np.cos(angle)
            y = 1.4 * np.sin(angle)
            text.set_position((x, y))
            text.set_fontsize(6)
            if autotext:
                autotext.set_fontsize(6)
        elif pct < 4:  # Between 2-4%, make slightly smaller
            text.set_fontsize(6.5)
            if autotext:
                autotext.set_fontsize(6.5)
    
    ax3.set_title('Overall Category Distribution\n(Percentage and Total Occurrences)', fontsize=12)
    
    # 4. Bar chart showing totals - bottom right
    ax4 = fig.add_subplot(gs[2, 1])
    bars = ax4.barh(range(len(categories_list)), totals_list, color=colors)
    ax4.set_yticks(range(len(categories_list)))
    ax4.set_yticklabels(categories_list, fontsize=9)
    ax4.set_xlabel('Total Occurrences', fontsize=11)
    ax4.set_title('Total Keyword Occurrences by Category', fontsize=12)
    ax4.invert_yaxis()
    ax4.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, total) in enumerate(zip(bars, totals_list)):
        ax4.text(total + max(totals_list) * 0.01, i, f'{total:,}', va='center', fontsize=8)
    
    plt.tight_layout()
    
    # Save figure
    filepath = os.path.join(output_dir, 'all_categories_combined_trends.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved combined trends plot: {filepath}")

def main():
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.dirname(script_dir)  # parent directory (news_data)
    output_dir = script_dir  # initial_work directory
    csv_file = os.path.join(data_dir, 'full_factiva.csv')
    
    print("Loading data...")
    df = pd.read_csv(csv_file)
    df['date'] = pd.to_datetime(df['date'])
    print(f"Loaded {len(df):,} articles")
    
    # Get articles per year for normalization (same for all categories)
    df['year'] = pd.to_datetime(df['date']).dt.year
    articles_per_year = df.groupby('year').size()
    
    # Analyze each category
    category_totals = {}
    category_yearly_raw = {}
    category_yearly_normalized = {}
    category_monthly_normalized = {}
    
    for category_name, keywords in CATEGORIES.items():
        yearly_counts_raw, yearly_counts_normalized, monthly_counts_normalized, keyword_counts, _ = analyze_category(df, category_name, keywords)
        
        # Create plot for this category
        total = create_category_plot(category_name, yearly_counts_raw, yearly_counts_normalized, monthly_counts_normalized, keyword_counts, articles_per_year, output_dir)
        category_totals[category_name] = total
        category_yearly_raw[category_name] = yearly_counts_raw
        category_yearly_normalized[category_name] = yearly_counts_normalized
        category_monthly_normalized[category_name] = monthly_counts_normalized
        
        # Save CSV files
        filename = category_name.lower().replace(' ', '_').replace('&', 'and')
        
        # Save yearly trends (both raw and normalized)
        yearly_df = pd.DataFrame({
            'raw_frequency': yearly_counts_raw,
            'normalized_frequency': yearly_counts_normalized,
            'articles_per_year': articles_per_year
        })
        yearly_df.to_csv(os.path.join(output_dir, f'{filename}_yearly.csv'))
        
        # Save monthly trends (normalized)
        if len(monthly_counts_normalized) > 0:
            monthly_df = pd.DataFrame({category_name: monthly_counts_normalized})
            monthly_df.to_csv(os.path.join(output_dir, f'{filename}_monthly.csv'))
        
        # Save keyword counts
        keyword_df = pd.DataFrame(list(keyword_counts.items()), columns=['keyword', 'frequency'])
        keyword_df = keyword_df.sort_values('frequency', ascending=False)
        keyword_df.to_csv(os.path.join(output_dir, f'{filename}_keywords.csv'), index=False)
    
    # Create combined trends plot (normalized) - includes pie chart and bar chart
    create_combined_trends_plot(category_yearly_normalized, category_totals, output_dir)
    
    # Save combined yearly data (normalized)
    combined_yearly = pd.DataFrame(category_yearly_normalized)
    combined_yearly.to_csv(os.path.join(output_dir, 'all_categories_yearly_normalized.csv'))
    
    # Also save raw for reference
    combined_yearly_raw = pd.DataFrame(category_yearly_raw)
    combined_yearly_raw.to_csv(os.path.join(output_dir, 'all_categories_yearly_raw.csv'))
    
    # Print summary
    print("\n" + "="*80)
    print("CATEGORY ANALYSIS SUMMARY")
    print("="*80)
    print(f"\nTotal Categories Analyzed: {len(CATEGORIES)}")
    print(f"\nCategory Totals (sorted by frequency):")
    sorted_totals = sorted(category_totals.items(), key=lambda x: x[1], reverse=True)
    for i, (category, total) in enumerate(sorted_totals, 1):
        print(f"  {i:2d}. {category:35s}: {total:10,} occurrences")
    
    print(f"\nAll results saved to: {output_dir}")
    print("="*80)

if __name__ == '__main__':
    main()

