import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

def extract_keywords(text, min_length=3):
    """Extract keywords from text, filtering common stop words (for non-important keywords only)"""
    if pd.isna(text):
        return []
    
    text_str = str(text).lower()
    
    # Common financial/economics stop words to filter
    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'a', 'an',
                  'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does',
                  'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that',
                  'these', 'those', 'with', 'from', 'by', 'as', 'it', 'its', 'they', 'them',
                  'their', 'there', 'then', 'than', 'more', 'most', 'some', 'any', 'all', 'each',
                  'every', 'other', 'another', 'such', 'only', 'just', 'also', 'very', 'much',
                  'many', 'how', 'what', 'when', 'where', 'why', 'who', 'which', 'about', 'into',
                  'over', 'after', 'before', 'during', 'through', 'between', 'among', 'within',
                  'without', 'under', 'above', 'below', 'up', 'down', 'out', 'off', 'away',
                  'back', 'here', 'there', 'where', 'now', 'then', 'today', 'yesterday',
                  'tomorrow', 'year', 'years', 'month', 'months', 'day', 'days', 'week', 'weeks',
                  'time', 'times', 'new', 'old', 'first', 'last', 'next', 'previous', 'recent',
                  'said', 'says', 'say', 'according', 'report', 'reports', 'reported', 'news'}
    
    # Extract words using word boundaries
    words = re.findall(r'\b[a-zA-Z]{' + str(min_length) + r',}\b', text_str)
    
    # Filter stop words and return
    keywords = [w for w in words if w not in stop_words and len(w) >= min_length]
    return keywords

def analyze_keywords(df, top_n=20):
    """Analyze macroeconomic keywords across the dataset"""
    print("Analyzing macroeconomic keywords from headlines and snippets...")
    
    # Combine headlines and snippets
    all_text = (df['headline'].fillna('') + ' ' + df['snippet'].fillna('')).str.lower()
    
    # Curated list of important macroeconomic keywords
    # Using substring matching to catch concatenated words (e.g., "inflationexpectations")
    macro_keywords = [
        # Inflation & Prices
        'inflation', 'deflation', 'disinflation', 'stagflation', 'price', 'prices', 'cpi', 'ppi',
        'consumer', 'wholesale', 'deflationary', 'inflationary',
        
        # Interest Rates & Monetary Policy
        'interest', 'rate', 'rates', 'monetary', 'policy', 'central', 'bank',
        'fomc', 'fedfunds', 'federal funds', 'discount', 'prime', 'yield', 'yields', 'treasury',
        'bond', 'bonds', 'tbill', 't-bill', 'tbill', 'tnote', 'tnote',
        
        # Economic Growth & Output
        'growth', 'gdp', 'recession', 'recovery', 'expansion', 'contraction', 'slowdown',
        'economic', 'economy', 'output', 'production', 'industrial', 'manufacturing',
        
        # Employment & Labor
        'employment', 'unemployment', 'jobs', 'labor', 'labour', 'wage', 'wages', 'payroll',
        'nonfarm', 'non-farm', 'unemployment rate',
        
        # Trade & Commerce
        'trade', 'trading', 'export', 'exports', 'import', 'imports', 'deficit', 'surplus',
        'balance', 'commerce', 'tariff', 'tariffs', 'tarrif',
        
        # Fiscal Policy & Government
        'fiscal', 'government', 'debt', 'deficit', 'surplus', 'budget', 'spending', 'tax', 'taxes',
        'treasury', 'congress', 'senate', 'house',
        
        # Markets & Finance
        'market', 'markets', 'stock', 'stocks', 'equity', 'equities', 'dollar', 'currency',
        'forex', 'exchange', 'dow', 'nasdaq', 'sp500', 's&p', 'vix', 'volatility',
        
        # Investment & Capital
        'investment', 'investor', 'investors', 'capital', 'equity', 'credit', 'lending',
        'borrowing', 'loan', 'loans',
        
        # Business & Corporate
        'business', 'corporate', 'earnings', 'profit', 'profits', 'revenue', 'revenues',
        'company', 'companies', 'firm', 'firms',
        
        # Commodities & Energy
        'oil', 'crude', 'energy', 'gas', 'gasoline', 'gold', 'silver', 'commodity', 'commodities',
        'petroleum', 'natural gas',
        
        # Housing & Real Estate
        'housing', 'home', 'homes', 'real estate', 'mortgage', 'mortgages', 'construction',
        'builder', 'builders',
        
        # Consumer & Spending
        'consumer', 'consumption', 'spending', 'retail', 'sales', 'demand', 'supply',
        'purchasing', 'purchase',
        
        # Financial Stability & Risk
        'crisis', 'crises', 'risk', 'risks', 'stability', 'volatility', 'uncertainty',
        'recession', 'depression', 'bubble', 'bubbles',
    ]
    
    # Count macroeconomic keywords using substring matching (catches concatenated words)
    keyword_counts = Counter()
    for keyword in macro_keywords:
        # Use regex pattern: not preceded by lowercase letter, keyword, not followed by lowercase letter
        # This catches both standalone words and concatenated words (e.g., "inflationexpectations")
        pattern = r'(?<![a-z])' + re.escape(keyword.lower()) + r'(?![a-z])'
        count = all_text.str.contains(pattern, regex=True, na=False).sum()
        if count > 0:
            keyword_counts[keyword.lower()] = count
    
    # Get top keywords
    top_keywords = keyword_counts.most_common(top_n)
    
    return keyword_counts, top_keywords

def analyze_keywords_over_time(df, keywords, keyword_counts):
    """Analyze keyword density over time"""
    print(f"\nAnalyzing keyword density over time for top {len(keywords)} keywords...")
    
    # Prepare date column
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['year_month'] = df['date'].dt.to_period('M')
    
    # Combine text
    df['combined_text'] = (df['headline'].fillna('') + ' ' + df['snippet'].fillna('')).str.lower()
    
    # Calculate keyword density by year
    # Use regex pattern to catch concatenated words (e.g., "inflationexpectations" contains "inflation")
    yearly_density = {}
    for keyword in keywords:
        keyword_lower = keyword.lower()
        # Pattern: not preceded by lowercase letter, keyword, not followed by lowercase letter
        # This catches both standalone words and concatenated words
        pattern = r'(?<![a-z])' + re.escape(keyword_lower) + r'(?![a-z])'
        yearly_counts = df.groupby('year')['combined_text'].apply(
            lambda x: x.str.contains(pattern, regex=True, case=False, na=False).sum()
        )
        yearly_density[keyword] = yearly_counts
    
    yearly_df = pd.DataFrame(yearly_density)
    
    # Calculate keyword density by month (for recent years)
    recent_years = df[df['year'] >= 2020]
    monthly_density = {}
    for keyword in keywords[:10]:  # Top 10 for monthly
        keyword_lower = keyword.lower()
        pattern = r'(?<![a-z])' + re.escape(keyword_lower) + r'(?![a-z])'
        monthly_counts = recent_years.groupby('year_month')['combined_text'].apply(
            lambda x: x.str.contains(pattern, regex=True, case=False, na=False).sum()
        )
        monthly_density[keyword] = monthly_counts
    
    monthly_df = pd.DataFrame(monthly_density)
    
    return yearly_df, monthly_df

def create_visualizations(keyword_counts, top_keywords, yearly_df, monthly_df, df):
    """Create visualization charts"""
    print("\nCreating visualizations...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Top keywords bar chart
    ax1 = plt.subplot(2, 3, 1)
    keywords, counts = zip(*top_keywords)
    ax1.barh(range(len(keywords)), counts, color='steelblue')
    ax1.set_yticks(range(len(keywords)))
    ax1.set_yticklabels(keywords)
    ax1.set_xlabel('Frequency')
    ax1.set_title('Top 20 Macroeconomic Keywords by Frequency')
    ax1.invert_yaxis()
    
    # 2. News volume over time
    ax2 = plt.subplot(2, 3, 2)
    yearly_volume = df.groupby('year').size()
    ax2.plot(yearly_volume.index, yearly_volume.values, marker='o', linewidth=2, markersize=6)
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Number of Articles')
    ax2.set_title('News Volume Over Time')
    ax2.grid(True, alpha=0.3)
    
    # 3. Top keywords trend over years
    ax3 = plt.subplot(2, 3, 3)
    top_5_keywords = [k[0] for k in top_keywords[:5]]
    for keyword in top_5_keywords:
        if keyword in yearly_df.columns:
            ax3.plot(yearly_df.index, yearly_df[keyword], marker='o', label=keyword, linewidth=2)
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Top 5 Macroeconomic Keywords Trend Over Years')
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. Recent trends (2020-2025) - monthly
    ax4 = plt.subplot(2, 3, 4)
    if not monthly_df.empty:
        top_5_recent = [k[0] for k in top_keywords[:5]]
        for keyword in top_5_recent:
            if keyword in monthly_df.columns:
                monthly_df[keyword].plot(ax=ax4, marker='o', label=keyword, linewidth=1.5, markersize=3)
        ax4.set_xlabel('Year-Month')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Top 5 Macroeconomic Keywords - Monthly Trend (2020-2025)')
        ax4.legend(loc='best', fontsize=7)
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
    
    # 5. Keyword heatmap by year (top 10 keywords)
    ax5 = plt.subplot(2, 3, 5)
    top_10_keywords = [k[0] for k in top_keywords[:10]]
    heatmap_data = yearly_df[top_10_keywords].T
    # Normalize by row (percentage of max for each keyword)
    heatmap_data_norm = heatmap_data.div(heatmap_data.max(axis=1), axis=0)
    sns.heatmap(heatmap_data_norm, annot=False, cmap='YlOrRd', cbar_kws={'label': 'Normalized Frequency'}, ax=ax5)
    ax5.set_title('Macroeconomic Keyword Density Heatmap (Top 10, Normalized)')
    ax5.set_xlabel('Year')
    ax5.set_ylabel('Keyword')
    
    # 6. Distribution of articles by hour of day (for entries with times)
    ax6 = plt.subplot(2, 3, 6)
    df_with_time = df[df['date'].dt.time != pd.Timestamp('00:00:00').time()].copy()
    if len(df_with_time) > 0:
        df_with_time['hour'] = df_with_time['date'].dt.hour
        hourly_dist = df_with_time['hour'].value_counts().sort_index()
        ax6.bar(hourly_dist.index, hourly_dist.values, color='coral', alpha=0.7)
        ax6.set_xlabel('Hour of Day (GMT)')
        ax6.set_ylabel('Number of Articles')
        ax6.set_title('Article Distribution by Hour of Day')
        ax6.set_xticks(range(0, 24, 2))
        ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('keyword_analysis_charts.png', dpi=300, bbox_inches='tight')
    print("Charts saved to 'keyword_analysis_charts.png'")
    
    return fig

def print_statistics(df, keyword_counts, top_keywords, yearly_df):
    """Print summary statistics"""
    print("\n" + "="*80)
    print("MACROECONOMIC KEYWORD ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\nDataset Overview:")
    print(f"  Total articles: {len(df):,}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Total macroeconomic keywords tracked: {len(keyword_counts):,}")
    print(f"  Articles with authors: {df['author'].notna().sum():,} ({df['author'].notna().sum()/len(df)*100:.1f}%)")
    
    print(f"\nTop 20 Macroeconomic Keywords:")
    for i, (keyword, count) in enumerate(top_keywords, 1):
        print(f"  {i:2d}. {keyword:20s} : {count:6,} occurrences")
    
    print(f"\nMacroeconomic Keyword Trends by Year (Top 10):")
    top_10_keywords = [k[0] for k in top_keywords[:10]]
    print(f"{'Year':<8}", end="")
    for keyword in top_10_keywords:
        print(f"{keyword[:12]:<13}", end="")
    print()
    print("-" * 150)
    
    # Show last 10 years
    recent_years = yearly_df.tail(10)
    for year in recent_years.index:
        print(f"{year:<8}", end="")
        for keyword in top_10_keywords:
            if keyword in yearly_df.columns:
                count = yearly_df.loc[year, keyword]
                print(f"{count:<13}", end="")
            else:
                print(f"{'0':<13}", end="")
        print()
    
    # Year-over-year growth for top keywords
    print(f"\nYear-over-Year Growth (2024 vs 2023) - Top 10 Macroeconomic Keywords:")
    if 2023 in yearly_df.index and 2024 in yearly_df.index:
        for keyword in top_10_keywords:
            if keyword in yearly_df.columns:
                count_2023 = yearly_df.loc[2023, keyword] if 2023 in yearly_df.index else 0
                count_2024 = yearly_df.loc[2024, keyword] if 2024 in yearly_df.index else 0
                if count_2023 > 0:
                    growth = ((count_2024 - count_2023) / count_2023) * 100
                    print(f"  {keyword:20s}: {count_2023:4.0f} → {count_2024:4.0f} ({growth:+.1f}%)")
                elif count_2024 > 0:
                    print(f"  {keyword:20s}: {count_2023:4.0f} → {count_2024:4.0f} (new)")
    
    print("\n" + "="*80)

def main():
    print("Loading data...")
    df = pd.read_csv('full_factiva.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"Loaded {len(df):,} articles")
    
    # Analyze keywords
    keyword_counts, top_keywords = analyze_keywords(df, top_n=20)
    
    # Analyze keywords over time
    top_keyword_list = [k[0] for k in top_keywords]
    yearly_df, monthly_df = analyze_keywords_over_time(df, top_keyword_list, keyword_counts)
    
    # Create visualizations
    create_visualizations(keyword_counts, top_keywords, yearly_df, monthly_df, df)
    
    # Print statistics
    print_statistics(df, keyword_counts, top_keywords, yearly_df)
    
    # Save detailed results
    print("\nSaving detailed results...")
    
    # Save top keywords to CSV
    top_keywords_df = pd.DataFrame(top_keywords, columns=['keyword', 'frequency'])
    top_keywords_df.to_csv('macro_keywords.csv', index=False)
    print("  Saved 'macro_keywords.csv'")
    
    # Save yearly trends to CSV
    yearly_df.to_csv('macro_keyword_trends_yearly.csv')
    print("  Saved 'macro_keyword_trends_yearly.csv'")
    
    if not monthly_df.empty:
        monthly_df.to_csv('macro_keyword_trends_monthly.csv')
        print("  Saved 'macro_keyword_trends_monthly.csv'")
    
    print("\nAnalysis complete!")

if __name__ == '__main__':
    main()

