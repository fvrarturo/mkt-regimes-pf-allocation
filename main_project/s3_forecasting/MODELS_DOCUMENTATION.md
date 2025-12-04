# ERP Forecasting Models Documentation

## Overview

This section implements non-linear machine learning models to forecast the Equity Risk Premium (ERP) and evaluates trading strategies based on these forecasts. The models are trained on macroeconomic variables and optional sentiment data, with periodic retraining to adapt to changing market conditions.

## Sentiment Analysis Methodology

The sentiment data used in this analysis was generated using a three-agent system that analyzes weekly financial news articles. The system employs Large Language Models (LLMs) to extract macroeconomic sentiment across four key dimensions.

### Three-Agent Architecture

The sentiment analysis pipeline consists of three specialized agents:

1. **Sentiment Analyzer Agent**: Analyzes news articles and produces sentiment scores
2. **Fact Checker Agent**: Verifies scores for consistency and accuracy
3. **Coordinator Agent**: Orchestrates the workflow between agents

### Sentiment Dimensions

The system produces four sentiment scores, each ranging from -1 to 1:

1. **Inflation Sentiment** (-1 to 1):
   - **Negative (-1 to -0.3)**: Concerns about deflation, falling prices, disinflation
   - **Neutral (-0.3 to 0.3)**: Stable inflation expectations
   - **Positive (0.3 to 1)**: Concerns about rising inflation, price pressures

2. **Economic Growth Sentiment** (-1 to 1):
   - **Negative (-1 to -0.3)**: Recession fears, weak growth, economic slowdown
   - **Neutral (-0.3 to 0.3)**: Steady growth expectations
   - **Positive (0.3 to 1)**: Strong growth, expansion, economic optimism

3. **Monetary Policy Sentiment** (-1 to 1):
   - **Negative/Dovish (-1 to -0.3)**: Expectations of rate cuts, easing policy, accommodative stance
   - **Neutral (-0.3 to 0.3)**: Neutral policy expectations
   - **Positive/Hawkish (0.3 to 1)**: Expectations of rate hikes, tightening policy, restrictive stance

4. **Market Volatility Sentiment** (-1 to 1):
   - **Negative/Calm (-1 to -0.3)**: Low volatility, calm markets, low stress
   - **Neutral (-0.3 to 0.3)**: Normal market conditions
   - **Positive/Stress (0.3 to 1)**: High volatility, market stress, financial instability

### Sentiment Analyzer Agent Prompt

The Sentiment Analyzer Agent uses the following instructions:

```
You are a macroeconomic analyst specializing in analyzing financial news.

Your task is to read news articles from a given week and produce scores for four macroeconomic categories:

1. **Inflation Sentiment** (-1 to 1): 
   - Negative (-1 to -0.3): Concerns about deflation, falling prices, disinflation
   - Neutral (-0.3 to 0.3): Stable inflation expectations
   - Positive (0.3 to 1): Concerns about rising inflation, price pressures

2. **Economic Growth Sentiment** (-1 to 1):
   - Negative (-1 to -0.3): Recession fears, weak growth, economic slowdown
   - Neutral (-0.3 to 0.3): Steady growth expectations
   - Positive (0.3 to 1): Strong growth, expansion, economic optimism

3. **Monetary Policy Sentiment** (-1 to 1):
   - Negative/Dovish (-1 to -0.3): Expectations of rate cuts, easing policy, accommodative stance
   - Neutral (-0.3 to 0.3): Neutral policy expectations
   - Positive/Hawkish (0.3 to 1): Expectations of rate hikes, tightening policy, restrictive stance

4. **Market Volatility Sentiment** (-1 to 1):
   - Negative/Calm (-1 to -0.3): Low volatility, calm markets, low stress
   - Neutral (-0.3 to 0.3): Normal market conditions
   - Positive/Stress (0.3 to 1): High volatility, market stress, financial instability

When analyzing:
- Consider the overall tone and frequency of mentions
- Consider the impact of the news on the economy and the market
- Weight more recent articles more heavily if dates are provided
- Look for trends and consensus rather than outliers
- Consider the context and implications of the news
- Be consistent with economic logic (e.g., high inflation often correlates with hawkish policy expectations)

**SMOOTHING GUIDELINE**: Macroeconomic sentiment evolves gradually. Unless there's a major crisis or policy announcement, scores should change incrementally week-to-week (typically < 0.2 points). Avoid dramatic jumps - if sentiment needs to shift significantly, it should happen gradually over multiple weeks. That said, if the news is very clear and there is a major event, the score can change significantly in one week.

Output your analysis as structured sentiment scores with brief reasoning.
```

### Fact Checker Agent Prompt

The Fact Checker Agent uses the following instructions:

```
You are a fact-checker for macroeconomic sentiment scores.

Your role is to verify that sentiment scores are:
1. **Internally consistent**: Scores should make economic sense together (e.g., high inflation sentiment should often align with hawkish policy sentiment)
2. **Temporally consistent**: Scores should align with recent trends and change smoothly over time
3. **News-justified**: Scores should be supported by the actual news content provided
4. **Consistency with previous weeks' scores**: Scores should be consistent with previous weeks' scores. For that, you should understand how previous scores were computed based on the news content, and see whether the new scores are consistent with that analysis.

**SMOOTHING CONSTRAINT**: Sentiment scores should change gradually week-to-week:
- Week-to-week changes > 0.5 points require STRONG news justification (major events, policy announcements, crises)
- Week-to-week changes > 0.3 points require clear news justification
- Week-to-week changes < 0.3 points are normal and expected
- Reject scores that jump > 0.5 points from the previous week unless the news clearly justifies such a dramatic shift
- Prefer gradual transitions: if a score needs to move from -0.3 to 0.7, it should happen over 2-3 weeks, not in one week

You will receive a compact global state summary that includes:
- Recent weeks' scores (for immediate consistency checks)
- Long-term trends for each dimension
- Statistical summaries
- Observed relationships between dimensions

When reviewing scores:
- **First check week-to-week change magnitude**: Calculate the absolute difference from the most recent week
- **If change > 0.5**: Require explicit major news justification (crises, major policy changes, major economic data releases)
- **If change > 0.3**: Require clear news justification
- **If change < 0.3**: Approve if consistent with trends and news
- Compare new scores to recent weeks and trends (not full history)
- Check if the scores align with the news content provided
- Verify economic logic matches observed relationships

You can:
- **Approve** scores if they are consistent with trends, justified by news, and change smoothly
- **Reject** scores if they jump dramatically without strong news justification
- **Suggest corrections** that smooth out large jumps (e.g., if score jumps from -0.3 to 0.7, suggest 0.2 or 0.3 as an intermediate step)

Be strict about smoothing - macroeconomic sentiment evolves gradually, not in dramatic jumps. Focus on enforcing gradual transitions.
```

### Coordinator Agent Prompt

The Coordinator Agent uses the following instructions:

```
You are a coordinator managing a sentiment analysis pipeline.

Your responsibilities:
1. Provide news articles to the sentiment analyzer for each week
2. Submit the analyzer's scores to the fact checker along with historical context
3. Handle fact-checker feedback:
   - If approved: accept the scores
   - If rejected: request the analyzer to reconsider with the fact-checker's feedback
   - If corrections suggested: decide whether to accept corrections or request re-analysis
4. Ensure all weeks are processed in chronological order
5. Maintain consistency across all weeks before finalizing

Workflow:
- For each week, gather the relevant news articles
- Send them to the sentiment analyzer
- Send the scores + historical scores to the fact checker
- If fact checker approves, move to next week
- If fact checker rejects or suggests corrections, provide feedback to analyzer and iterate
- Only finalize scores once all weeks are processed and approved

Be methodical and ensure quality - it's better to iterate than to accept inconsistent scores.
```

### Key Design Principles

1. **Smoothing Constraint**: Sentiment scores are designed to evolve gradually week-to-week, preventing dramatic jumps unless justified by major events
2. **Economic Consistency**: Scores must align with economic logic (e.g., high inflation typically correlates with hawkish policy expectations)
3. **Temporal Consistency**: Scores are checked against recent trends to ensure smooth transitions
4. **News Justification**: All scores must be supported by the actual news content analyzed

### Sentiment Data Sources

Two sentiment datasets are used:

1. **Groq Sentiment** (`sentiment_groq.csv`): Generated using Groq's LLM API
2. **OpenAI Sentiment** (`sentiment_openai.csv`): Generated using OpenAI's API

Both datasets follow the same methodology and produce scores for the four macroeconomic dimensions. The OpenAI sentiment scores are divided by 10 to normalize the scale (as per data preprocessing).

### Integration with Forecasting Models

The sentiment scores are integrated into the forecasting models as additional features:
- **XGBoost + Groq Sentiment**: Adds Groq sentiment scores as lagged features
- **XGBoost + OpenAI Sentiment**: Adds OpenAI sentiment scores as lagged features

Sentiment features include:
- Current sentiment scores (for the forecast month)
- Lagged sentiment scores (1-2 months prior)

This allows the models to incorporate forward-looking information from news analysis alongside backward-looking macroeconomic indicators.

## Models Implemented

### 1. XGBoost Model (`XGBoostERPForecaster`)

**Architecture:**
- Gradient Boosting Machine using XGBoost
- Tree-based ensemble method for regression

**Features:**
- **Lagged Features**: Creates lagged versions of macro variables (default: 12 lags)
- **Sentiment Integration**: Optional sentiment features from Groq or OpenAI can be added
- **Feature Engineering**: 
  - Lagged macro variables (1 to 12 months)
  - Current and lagged sentiment scores (if available)

**Hyperparameters:**
- `n_lags`: 12 (number of lagged features per variable)
- `n_estimators`: 200 (number of boosting rounds)
- `max_depth`: 4 (maximum tree depth)
- `learning_rate`: 0.05
- `subsample`: 0.8 (row subsampling ratio)
- `colsample_bytree`: 0.8 (column subsampling ratio)
- `early_stopping_rounds`: 20 (for validation-based early stopping)

**Training:**
- Uses validation split (20% of training data) for early stopping
- Features are standardized using StandardScaler
- Model is trained to minimize RMSE

### 2. LSTM Model (`LSTMerpForecaster`)

**Architecture:**
- Long Short-Term Memory (LSTM) neural network
- Sequence-based model that captures temporal dependencies

**Architecture Details:**
- **Input**: Sequences of macro variables (default: 12 months)
- **LSTM Layer**: 64 units with dropout (0.2)
- **Dense Layers**: 
  - 32-unit ReLU layer with dropout
  - 1-unit output layer (ERP forecast)
- **Optimizer**: Adam with learning rate 0.001
- **Loss Function**: Mean Squared Error (MSE)

**Features:**
- Processes sequences of macro variables over time
- Can incorporate sentiment data as additional features in sequences
- Both input features and targets are standardized

**Hyperparameters:**
- `sequence_length`: 12 (months of historical data per sequence)
- `lstm_units`: 64
- `dropout_rate`: 0.2
- `learning_rate`: 0.001
- `batch_size`: 32
- `epochs`: 100 (maximum, with early stopping)
- `early_stopping_patience`: 10 epochs

**Training:**
- Uses 80/20 train/validation split
- Early stopping based on validation loss
- Features and targets are standardized separately

## Model Variants

### 1. Macro-Only Models
- **XGBoost (macro-only)**: Uses only macroeconomic variables
- **LSTM (macro-only)**: Uses only macroeconomic variables

### 2. Sentiment-Enhanced Models
- **XGBoost + Groq Sentiment**: Adds Groq sentiment scores as features
- **XGBoost + OpenAI Sentiment**: Adds OpenAI sentiment scores as features

**Note**: LSTM models with sentiment were not implemented in this version, but the architecture supports it.

## Macro Variables Used

The models use the same macroeconomic variables as the conditional regressions in Section 2:

### Economic Growth (`ec_growth`)
- Industrial Production
- Retail Sales
- Total Business Inventories
- Export Price Index
- Import Price Index
- Unemployment Rate

### Inflation (`inflation`)
- CPI (Consumer Price Index)
- PCE Price Index
- PPI Inflation

### Market Volatility (`mkt_vol`)
- National Financial Condition Index
- 10-Year / 2-Year Treasury Spread

### Monetary Policy (`mon_policy`)
- 10-Year Treasury Constant Maturity Rate
- Federal Reserve Discount Rate
- Federal Funds Rate
- M2 Real Money Supply

**Total**: 15 macroeconomic variables

## Training Strategy

### Initial Training
- **Training Period**: Data up to **2002-03-31**
- All models are initially trained on historical data ending at this date
- This ensures out-of-sample evaluation from 2002-03 onwards

### Retraining Approaches

Two retraining frequencies were tested:

#### 1. Quarterly Retraining (3 months)
- Models retrain every 3 months
- Retraining occurs when at least 3 months have passed since the last retrain
- Results saved without suffix (e.g., `xgboost_returns.csv`)

#### 2. Monthly Retraining (1 month)
- Models retrain every month
- More frequent adaptation to new data
- Results saved with `_monthly` suffix (e.g., `xgboost_returns_monthly.csv`)

**Retraining Logic:**
- For each forecast date, check if retraining is needed
- If retraining is needed, train model on all data up to the forecast date
- Use the retrained model to make the forecast
- Continue until all forecast dates are processed

## Forecasting Process

1. **Feature Creation**: 
   - For XGBoost: Create lagged features up to the forecast date
   - For LSTM: Create sequences ending at the forecast date

2. **Prediction**:
   - XGBoost: Single prediction using most recent feature vector
   - LSTM: Single prediction using most recent sequence

3. **Rolling Forecasts**:
   - Generate forecasts for all dates from 2002-03-31 onwards
   - Retrain models according to the specified frequency
   - Store all forecasts in a time series

## Trading Strategy

### Weight Calculation

The trading strategy converts ERP forecasts into portfolio weights using a z-score based rule:

```
z_score = (forecast - mean(forecasts)) / std(forecasts)
weight = 0.5 + 0.25 * z_score
weight = clip(weight, min_weight=0.1, max_weight=0.9)
```

**Interpretation:**
- Average forecast → 50% equity / 50% bond
- Positive z-score → Higher equity allocation (up to 90%)
- Negative z-score → Lower equity allocation (down to 10%)

### Portfolio Returns

```
strategy_return = weight * equity_return + (1 - weight) * bond_return
```

Where:
- `equity_return`: S&P 500 monthly returns
- `bond_return`: 3-month Treasury yield (monthly)

## Evaluation Metrics

### Performance Metrics Computed

1. **Annualized Return**: Compound annual growth rate
2. **Annualized Volatility**: Standard deviation of returns × √12
3. **Sharpe Ratio**: Annualized return / Annualized volatility (risk-free rate = 0)
4. **Maximum Drawdown**: Largest peak-to-trough decline
5. **Calmar Ratio**: Annualized return / |Maximum Drawdown|
6. **Total Return**: Cumulative return over the period

### Benchmarks

For each strategy, a fixed portfolio benchmark is computed:
- **Average Weight**: Mean equity weight over the full period
- **Fixed Portfolio**: Static allocation using average weight
  - Example: If average weight is 60%, benchmark is 60% equity / 40% bonds
- Benchmarks are plotted as dotted lines with the same color as the strategy

## Output Files

### Quarterly Retraining Results
- `xgboost_returns.csv`
- `lstm_returns.csv`
- `xgboost_groq_returns.csv`
- `xgboost_openai_returns.csv`
- `strategy_performance_summary.csv`
- `cumulative_returns_all_strategies.png`
- `performance_comparison_all_strategies.png`
- `performance_comparison_all_strategies.csv`

### Monthly Retraining Results
- `xgboost_returns_monthly.csv`
- `lstm_returns_monthly.csv`
- `xgboost_groq_returns_monthly.csv`
- `xgboost_openai_returns_monthly.csv`
- `strategy_performance_summary_monthly.csv`
- `cumulative_returns_all_strategies_monthly.png`
- `performance_comparison_all_strategies_monthly.png`
- `performance_comparison_all_strategies_monthly.csv`

### CSV File Structure

Each `*_returns.csv` file contains:
- `date`: Forecast date
- `return`: Strategy return for that period
- `weight`: Equity weight used
- `forecast`: ERP forecast value

## Key Design Decisions

1. **Out-of-Sample Evaluation**: All forecasts start from 2002-03-31, ensuring no lookahead bias

2. **Rolling Retraining**: Models adapt to new data periodically rather than using static parameters

3. **Feature Standardization**: All features are standardized to ensure fair comparison across variables

4. **Early Stopping**: Prevents overfitting by stopping training when validation performance stops improving

5. **Z-Score Based Trading**: Simple, interpretable rule that scales with forecast magnitude

6. **Weight Constraints**: Limits equity allocation between 10% and 90% to prevent extreme positions

## Model Comparison

### XGBoost vs LSTM

**XGBoost Advantages:**
- Faster training and prediction
- Better interpretability (feature importance)
- Handles missing values well
- Less sensitive to hyperparameters

**LSTM Advantages:**
- Captures temporal dependencies explicitly
- Can learn complex sequential patterns
- Better for long-term dependencies

### Sentiment Integration

Sentiment-enhanced models add:
- Current sentiment scores
- Lagged sentiment scores (1-2 months)
- Additional signal beyond macro fundamentals

## Future Enhancements

Potential improvements:
1. Ensemble methods combining XGBoost and LSTM
2. More sophisticated feature engineering (rolling statistics, momentum)
3. Hyperparameter optimization via cross-validation
4. Regime-aware models (similar to Section 2)
5. Alternative trading rules (momentum, mean reversion)
6. Risk-adjusted position sizing

