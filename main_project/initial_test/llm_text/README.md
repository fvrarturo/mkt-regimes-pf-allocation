# Weekly News Sentiment Analysis with Multi-Agent System

This directory contains a three-agent system for analyzing macroeconomic sentiment from weekly news data. The system processes news articles week-by-week and generates sentiment scores for four key macroeconomic dimensions.

## Overview

The pipeline analyzes news articles from `full_factiva.csv` and produces weekly sentiment scores using a multi-agent architecture:

1. **Sentiment Analyzer Agent**: Reads weekly news articles and generates sentiment scores
2. **Fact Checker Agent**: Validates scores for consistency and economic logic
3. **Coordinator Agent**: Orchestrates interactions between the agents

**Key Features:**
- ✅ **FREE** - Uses Groq API (fast Llama models, no cost)
- ✅ **Incremental Saving** - Saves progress after each week, can resume if interrupted
- ✅ **API Key Rotation** - Automatically handles rate limits by rotating through multiple keys
- ✅ **Temporal Consistency** - Uses compact state summary to maintain consistency without context overload

## Sentiment Dimensions

The system produces four sentiment scores per week (ranging from -1 to +1):

1. **Inflation Sentiment**: Sentiment about inflation trends and price levels
   - `-1`: Extremely negative (deflation concerns, falling prices)
   - `0`: Neutral
   - `+1`: Extremely positive (high inflation concerns, rising prices)

2. **Economic Growth Sentiment**: Sentiment about economic expansion and output
   - `-1`: Extremely negative (recession, contraction)
   - `0`: Neutral
   - `+1`: Extremely positive (strong growth, expansion)

3. **Monetary Policy Sentiment**: Sentiment about central bank policy stance
   - `-1`: Extremely dovish (easing, rate cuts)
   - `0`: Neutral
   - `+1`: Extremely hawkish (tightening, rate hikes)

4. **Market Volatility Sentiment**: Sentiment about financial market stress and volatility
   - `-1`: Extremely low volatility (calm markets)
   - `0`: Neutral
   - `+1`: Extremely high volatility (market stress, uncertainty)

## Setup

### Prerequisites

1. **Python 3.11+** with virtual environment
2. **Groq API Keys** (FREE - get at https://console.groq.com)
   - Recommended: Get 3-5 keys for automatic rotation
3. **News Data**: `main_project/data/news_data/full_factiva.csv`

### Installation

1. **Create and activate virtual environment**:
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   # Install agents framework (editable)
   pip install -e ../../../../openai/
   
   # Install other dependencies
   pip install -r ../../../../requirements.txt
   ```

3. **Set up Groq API keys**:
   ```bash
   # Single key
   export GROQ_API_KEY="gsk_your-api-key-here"
   
   # OR multiple keys for rotation (recommended)
   export GROQ_API_KEYS="gsk_key1,gsk_key2,gsk_key3,gsk_key4,gsk_key5"
   
   # Optional: Set model (defaults to llama-3.1-8b-instant)
   export GROQ_MODEL="llama-3.1-8b-instant"
   ```

### Optional: Local Development with Ollama

For local testing without API calls, you can use Ollama:

1. **Install Ollama**:
   ```bash
   # macOS
   brew install ollama
   
   # Or download from https://ollama.ai
   ```

2. **Pull Llama3 model**:
   ```bash
   ollama pull llama3
   ```

3. **Start Ollama**:
   ```bash
   ollama serve
   ```

4. **Use Ollama instead of Groq**:
   ```bash
   export USE_OLLAMA=true
   export OLLAMA_BASE_URL="http://localhost:11434/v1"
   export OLLAMA_MODEL="llama3"
   ```

**Note**: Ollama is slower and requires local compute. Groq API is recommended for production runs.

## Usage

### Basic Usage

Run the sentiment analysis pipeline:

```bash
cd main_project/initial_test/llm_text
python main.py
```

The script will:
1. Load news data from `main_project/data/news_data/full_factiva.csv`
2. Group articles by week (Monday as week start)
3. Process each week chronologically through the agent pipeline
4. Save results incrementally to `sentiment_scores.csv`
5. Automatically rotate API keys if rate limits are hit

### Output Format

The output CSV (`sentiment_scores.csv`) contains:

```csv
date,inflation_sentiment,ec_growth_sentiment,monetary_policy_sentiment,market_vol_sentiment
1990-01-01,0.0,-0.2,-0.5,0.0
1990-01-08,0.0,-0.3,-0.7,-0.3
...
```

- `date`: Monday date of the week (YYYY-MM-DD)
- Each sentiment score: Float from -1.0 to 1.0

### Resuming Interrupted Jobs

The script automatically saves progress after each week. If interrupted:

1. **Check current progress**:
   ```bash
   wc -l sentiment_scores.csv
   head -20 sentiment_scores.csv
   ```

2. **Resubmit** - The script will automatically skip already-processed weeks:
   ```bash
   python main.py
   ```

No data loss - all processed weeks are saved incrementally.

## Configuration

### LLM Configuration (`llm_config.py`)

The system automatically selects the LLM based on environment variables (priority order):

1. **Ollama** (if `USE_OLLAMA=true`) - Local, free, slower
2. **Groq API** (if `GROQ_API_KEY` or `GROQ_API_KEYS` set) - Cloud, free, fast ⭐ **Recommended**
3. **OpenAI API** (if `OPENAI_API_KEY` set) - Cloud, paid, reliable

**Groq Models Available:**
- `llama-3.1-8b-instant` (default, fast, 500K TPD limit)
- `llama-3.1-70b-versatile` (more capable, slower)
- `groq/compound-mini` (unlimited TPD, 70K TPM)
- `groq/compound` (unlimited TPD, 70K TPM, more capable)

### Agent Configuration (`sentiment_agents.py`)

**Key parameters:**
- `SENTIMENT_ANALYZER_INSTRUCTIONS`: Instructions for the sentiment analyzer
- `FACT_CHECKER_INSTRUCTIONS`: Instructions for the fact checker (includes smoothing constraints)
- `MAX_ITERATIONS`: Maximum retry attempts per week (default: 5)

**Smoothing Constraints:**
The fact checker enforces smooth transitions:
- Changes >0.5 points require strong justification
- Changes >0.3 points require clear justification
- Prevents excessive week-to-week oscillation

### Processing Options (`main.py`)

**Key functions:**
- `group_news_by_week()`: Groups articles by week (Monday start)
- `process_week()`: Processes a single week through the agent pipeline
- `format_news_for_agent()`: Formats news articles for agent consumption

**To process a subset of weeks**, modify `main()`:
```python
# Process only first 10 weeks (for testing)
weeks_to_process = sorted_weeks[:10]
```

## Architecture

### Agent System

**1. Sentiment Analyzer Agent**
- **Input**: Weekly news articles (headlines + snippets)
- **Output**: `WeeklySentimentScores` with 4 sentiment scores + reasoning
- **Role**: Analyze news and extract macroeconomic sentiment

**2. Fact Checker Agent**
- **Input**: Proposed scores + global state summary (recent weeks, trends)
- **Output**: `FactCheckResult` (approved/rejected + issues)
- **Role**: Validate scores for:
  - Temporal consistency (smooth transitions)
  - Economic logic (e.g., inflation ↔ monetary policy correlation)
  - News justification (scores match news content)

**3. Coordinator Agent**
- **Input**: Analyzer output + fact checker feedback
- **Output**: Final approved scores or requests for revision
- **Role**: Manage workflow and ensure all scores are approved before finalizing

### State Management

**Global State Summary** (prevents context overload):
- Recent weeks (last 8 weeks) with scores
- Long-term trends (mean, std, min, max per dimension)
- Dimension definitions and interpretation guidelines

**Benefits:**
- Maintains temporal consistency without loading full history
- Keeps context small (prevents hardware overload)
- Enables efficient processing of long time series

### API Key Rotation

When a rate limit is hit:
1. System detects rate limit error
2. Rotates to next available Groq API key
3. Recreates agents with new API key
4. Retries the request
5. If all keys exhausted, stops with clear message

**Monitor rotation:**
```bash
tail -f sentiment_analysis.out | grep -E 'rotating|API key|🔑'
```

## Running on MIT Engaging Cluster

For running on MIT Engaging HPC cluster, see:
- `../../../../engaging_setup_llm.md` - Complete setup guide
- `../../../../cluster/run_sentiment_analysis.slurm` - SLURM job script

**Quick start:**
```bash
# On cluster
cd ~/Agents_News
sbatch cluster/run_sentiment_analysis.slurm

# Monitor
tail -f sentiment_analysis.out
```

## Troubleshooting

### "No LLM configured!"

**Solution**: Set Groq API keys:
```bash
export GROQ_API_KEYS="your-key1,your-key2,your-key3"
```

### Rate limit errors

**Solution**: The system automatically rotates keys. If all exhausted:
1. Get more keys from https://console.groq.com
2. Update `GROQ_API_KEYS` environment variable
3. Resubmit job (will resume from last processed week)

### Import errors

**Solution**: Reinstall dependencies:
```bash
pip install -e ../../../../openai/
pip install -r ../../../../requirements.txt
```

### Job stops/resumes

**Normal behavior**: The script saves incrementally. If it stops:
- Check `sentiment_scores.csv` for progress
- Resubmit - it will automatically skip processed weeks
- Check `sentiment_analysis.err` for errors

## Files

- `sentiment_agents.py` - Agent definitions and data models
- `main.py` - Main pipeline script
- `llm_config.py` - LLM configuration (Groq/Ollama/OpenAI)
- `test_setup.py` - Quick test script (processes 1-2 weeks)
- `sentiment_scores.csv` - Output file (created automatically)
- `README.md` - This file

## Notes

- **Processing order**: Weeks are processed chronologically to maintain temporal consistency
- **Approval required**: All scores must be fact-checker approved before finalizing
- **Incremental saving**: Progress saved after each week (no data loss on interruption)
- **API costs**: Groq API is FREE (no cost, generous rate limits)
- **Model choice**: `llama-3.1-8b-instant` is the default (fast, sufficient for sentiment analysis). With 9 API keys, you get 14.4K × 9 = 129,600 requests/day total

## Next Steps

After generating sentiment scores:
1. Visualize trends: `python visualize_scores.py` (if available)
2. Analyze correlations between dimensions
3. Use scores for macro regime identification
4. Integrate with portfolio allocation models

---

For questions or issues, check the main project README or the MIT Engaging setup guide.
