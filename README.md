# Forecasting Economic Variables for Active Asset Allocation

## Project Overview

This project develops forecasting models for economic variables over various horizons and evaluates whether these forecasts provide economic value for active asset allocation. The approach combines traditional econometric methods with agentic AI (LLM-based) forecasting, using both quantitative macroeconomic variables and text-derived sentiment scores from news articles.

**Core Question**: Do our forecasting frameworks generate information that is economically valuable for active asset allocation, beyond a simple static strategy?

## Project Structure

The project is organized into three main steps, each with corresponding subdirectories:

### Step 1: Identifying Important Economic Variables (`s1_macro_vars/`)

We use three complementary lenses to determine which macro variables matter for explaining and forecasting the stock–bond risk premium:

1. **Full-Sample Analysis** (`s11_full_sample/`)
   - Run simple and multiple regressions of the stock–bond risk premium on a broad set of candidate macro variables (interest rates, macro policy proxies, growth, volatility, etc.) using the entire sample
   - Get baseline t-statistics and economic signs to see which variables are relevant on average

2. **Regime-Dependent Analysis (Regimeness)** (`s12_regimeness/`)
   - Define a small number of target regimes representing distinct macroeconomic environments (four types: interest rate, macroeconomic policy, economic growth, market volatility)
   - Calculate regime probabilities for each historical period using distance-based methods
   - Estimate separate forecasting models for each regime using probability-weighted observations
   - Produce weighted average forecasts across all regime-specific models

3. **Extreme Events / Tail-Events Analysis** (`s13_extremeness/`)
   - Identify periods where key macroeconomic variables reach unusually high or low levels
   - Examine how the distribution of market returns behaves in those extreme regions
   - Model extremeness directly to capture periods of heightened risk premia

### Step 2: Developing Forecasting Models (`s2_forecasts/`)

We construct forecasting models for the key variables identified in Step 1, using two complementary approaches:

1. **Econometric Forecasting Models** (`s21_macro/`)
   - Build parsimonious models (OLS, ridge, logistic specifications)
   - Relate selected macro variables, regime probabilities, and/or extremeness indicators to future stock–bond risk premia or tail-event indicators
   - Apply full-sample, regime-weighted, and extreme-state specifications
   - Emphasis on economic interpretability and clarity

2. **LLM-Based / Machine-Learning Forecasts** (`s22_ml_based/`)
   - Employ Large Language Model (LLM)–based or related machine-learning framework (agentic AI setup)
   - Use the same forecast targets as econometric approach for direct comparison
   - Incorporate richer information (survey or textual data) where feasible
   - Provide flexible, data-driven perspective on forecasting

### Step 3: Evaluating Economic Value (`s3_evaluation/`)

We assess whether forecasts are economically useful for active asset allocation:

1. **Performance of Forecast-Based Trading Rules**
   - Design trading/allocation rules that tilt between stocks and Treasuries based on forecasts
   - Compute performance metrics: Sharpe ratio, volatility, maximum drawdown
   - Compare relative to neutral benchmark allocation (50/50 stock-bond mix)

2. **Sensitivity of Large Market Moves to Signals**
   - Identify largest market drawdowns and run-ups over the sample
   - Measure how many extreme events were preceded by conditions meeting model criteria
   - Focus on upside and downside fragility

3. **Comparison Across Forecasting Approaches**
   - Compare econometric and LLM-based approaches on:
     - Sharpe ratios and risk characteristics of trading rules
     - Ability to identify periods preceding major drawdowns and run-ups

## Data Structure

### Macroeconomic Data (`data/macro_raw/` and `data/macro_processed/`)

Organized by category:
- **Inflation**: CPI, PPI, PCE Price Index
- **Economic Growth**: GDP, Real GDP, Unemployment, Industrial Production, Retail Sales, Business Inventories
- **Monetary Policy**: Fed Funds Rate, Discount Rate, 10Y Treasury Rate, M2 Money Supply
- **Market Volatility**: VIX, 10Y-2Y Spread, SP500 Vol Index, Nasdaq Vol Index, Financial Conditions Index
- **Other**: Treasury yields, High-yield spreads, SP500

### News Data (`data/news_data/`)

- **Sentiment Scores** (`sentiment_scores.csv`): Weekly sentiment scores derived from news articles using agentic AI
  - Four sentiment dimensions: inflation, economic growth, monetary policy, market volatility
- **Raw News Data**: Factiva news articles and keyword analysis

### LLM Sentiment Analysis (`initial_test/llm_text/`)

Agentic AI pipeline for processing news articles:
- Multi-agent system for article classification, fact-checking, and sentiment scoring
- Weekly sentiment score generation aligned with macro categories
- Supports multiple LLM providers (Groq, OpenAI, Ollama)

## Key Innovations

1. **Integration of Text and Quantitative Data**: Combines traditional macro variables with AI-derived sentiment scores from news articles, enabling richer regime identification

2. **Probabilistic Regime Framework**: Uses soft regime probabilities rather than hard splits, allowing all observations to contribute while emphasizing regime-representative periods

3. **Dual Forecasting Approach**: Parallel development of econometric and LLM-based models enables direct comparison and validation

4. **Economic Value Focus**: Evaluation emphasizes practical trading performance rather than just statistical significance

## Getting Started

### Prerequisites

- Python 3.11+
- Required packages listed in `requirements.txt`
- LLM API keys (Groq, OpenAI, or Ollama) for sentiment analysis

### Running Sentiment Analysis

```bash
# Set up API keys
export GROQ_API_KEYS="your-keys-here"

# Run sentiment analysis pipeline
python main_project/initial_test/llm_text/main.py
```

### Running Analysis

```bash
# Step 1: Identify important variables
# (Scripts in s1_macro_vars/)

# Step 2: Develop forecasting models
# (Scripts in s2_forecasts/)

# Step 3: Evaluate economic value
# (Scripts in s3_evaluation/)
```

## Project Status

- ✅ **Data Collection**: Macro variables and news data collected and processed
- ✅ **Sentiment Analysis**: LLM-based sentiment scoring pipeline operational
- 🔄 **Step 1**: Identifying important economic variables (in progress)
- ⏳ **Step 2**: Developing forecasting models (planned)
- ⏳ **Step 3**: Evaluating economic value (planned)

## References

See `goal_project.md` for detailed methodology and `main_project/initial_test/idea.tex` for theoretical background.
