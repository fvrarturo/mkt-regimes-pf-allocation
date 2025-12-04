# Weekly News Sentiment Analysis with Llama3 Agents

This directory contains a three-agent system for analyzing macroeconomic sentiment from weekly news data.

## Agents

1. **Sentiment Analyzer Agent**: Analyzes news articles and produces sentiment scores for:
   - Inflation sentiment
   - Economic growth sentiment
   - Monetary policy sentiment
   - Market volatility sentiment

2. **Fact Checker Agent**: Verifies scores for:
   - Internal consistency (economic logic)
   - Temporal consistency (week-to-week changes)
   - News justification (scores match the news content)

3. **Coordinator Agent**: Orchestrates the workflow between the analyzer and fact checker

## Setup

### Prerequisites

1. **Install Ollama** (if not already installed):
   ```bash
   # macOS
   brew install ollama
   
   # Or download from https://ollama.ai
   ```

2. **Pull Llama3 model**:
   ```bash
   ollama pull llama3
   # or
   ollama pull llama3:8b
   ```

3. **Start Ollama** (if not running):
   ```bash
   ollama serve
   ```

4. **Install Python dependencies**:
   ```bash
   pip install pandas openai
   ```

### Verify Setup

Test that Ollama is running:
```bash
curl http://localhost:11434/api/tags
```

## Usage

Run the main script:
```bash
cd main_project/initial_test/llm_text
python main.py
```

The script will:
1. Load news data from `main_project/data/news_data/full_factiva.csv`
2. Group articles by week (Monday as week start)
3. Process each week through the agent pipeline
4. Save results to `sentiment_scores.csv`

## Output Format

The output CSV contains:
- `date`: Monday date of the week (YYYY-MM-DD)
- `inflation_sentiment`: Score from -1 to 1
- `ec_growth_sentiment`: Score from -1 to 1
- `monetary_policy_sentiment`: Score from -1 to 1
- `market_vol_sentiment`: Score from -1 to 1

## Configuration

### Model Configuration

Edit `agents.py` to change the model:
- Default: Uses Ollama with `llama3` model
- To use a different model: Change the `model` parameter in `LLAMA3_MODEL`
- To use a different API: Change `base_url` in `AsyncOpenAI` initialization

### Processing Options

Edit `main.py` to:
- Change number of weeks processed: Modify `weeks_to_process`
- Adjust max articles per week: Change `max_articles` in `format_news_for_agent`
- Change max iterations: Modify `max_iterations` in `process_week`

## Notes

- The system processes weeks chronologically to maintain temporal consistency
- Fact-checker can reject scores and request re-analysis
- All scores must be approved before finalizing (as per requirements)
- The coordinator manages the workflow but currently the main script calls agents directly for simplicity

