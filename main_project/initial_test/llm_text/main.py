"""
Main script to run the sentiment analysis pipeline.
Processes weekly news data and produces sentiment scores.
"""
import sys
from pathlib import Path
import pandas as pd
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import csv
import json
import os

from agents import Runner, set_tracing_disabled
set_tracing_disabled(True)

from sentiment_agents import (
    WeeklySentimentScores,
    FactCheckResult,
    GlobalStateSummary,
    update_state_summary,
    format_state_summary,
    recreate_agents_with_new_key,
)
# Import agents module so we can reload it and get fresh agents
import sentiment_agents
from llm_config import (
    get_groq_llm_model,
    rotate_to_next_groq_key,
    are_all_groq_keys_exhausted,
    GROQ_API_KEYS,
)


def get_monday_of_week(date: pd.Timestamp) -> pd.Timestamp:
    """Get the Monday of the week for a given date."""
    days_since_monday = date.weekday()
    monday = date - timedelta(days=days_since_monday)
    return monday


def group_news_by_week(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Group news articles by week (Monday as week start)."""
    df['date'] = pd.to_datetime(df['date'])
    df['week_start'] = df['date'].apply(get_monday_of_week)
    
    weekly_groups = {}
    for week_start, group in df.groupby('week_start'):
        week_key = week_start.strftime('%Y-%m-%d')
        weekly_groups[week_key] = group
    
    return weekly_groups


def format_news_for_agent(week_df: pd.DataFrame, max_articles: int = 50) -> str:
    """Format news articles for the agent."""
    # Sort by date within the week
    week_df = week_df.sort_values('date')
    
    # Limit to most recent/relevant articles if too many
    if len(week_df) > max_articles:
        week_df = week_df.tail(max_articles)
    
    formatted = []
    for idx, row in week_df.iterrows():
        date_str = pd.to_datetime(row['date']).strftime('%Y-%m-%d %H:%M')
        author = f" by {row['author']}" if pd.notna(row['author']) else ""
        formatted.append(
            f"[{date_str}]{author}\n"
            f"Headline: {row['headline']}\n"
            f"Snippet: {row['snippet']}\n"
        )
    
    return "\n---\n".join(formatted)


async def process_week(
    week_start: str,
    week_news: pd.DataFrame,
    state_summary: GlobalStateSummary,
    max_iterations: int = 3
) -> Optional[WeeklySentimentScores]:
    """Process a single week through the pipeline using compact state summary."""
    print(f"\nProcessing week starting {week_start}...")
    
    # Format news
    news_text = format_news_for_agent(week_news)
    
    # Format compact state summary (instead of full history)
    state_context = format_state_summary(state_summary)
    
    # Iterate until approved or max iterations
    for iteration in range(max_iterations):
        print(f"  Iteration {iteration + 1}...")
        
        # Step 1: Analyze sentiment
        analyze_prompt = f"""Analyze the following news articles from the week starting {week_start} and produce sentiment scores.

News articles:
{news_text}
"""
        
        if iteration > 0:
            analyze_prompt += f"\nNote: Previous attempt was rejected. Please reconsider based on the feedback."
        
        # Retry logic with API key rotation for rate limits
        sentiment_result = None
        max_retries = len(GROQ_API_KEYS) if GROQ_API_KEYS else 1
        for retry_attempt in range(max_retries):
            # Always get fresh agents from the module (they may have been recreated)
            sentiment_analyzer_agent = sentiment_agents.sentiment_analyzer_agent
            fact_checker_agent = sentiment_agents.fact_checker_agent
            coordinator_agent = sentiment_agents.coordinator_agent
            
            try:
                sentiment_result = await Runner.run(sentiment_analyzer_agent, analyze_prompt)
                break  # Success, exit retry loop
            except Exception as e:
                error_str = str(e)
                error_type_str = str(type(e))
                # Check if it's a rate limit error (TPM or TPD)
                is_rate_limit = (
                    "rate_limit" in error_str.lower() or 
                    "429" in error_str or 
                    "tokens per day" in error_str.lower() or
                    "tokens per minute" in error_str.lower() or
                    "tpm" in error_str.lower() or
                    "tpd" in error_str.lower() or
                    "RateLimitError" in error_type_str
                )
                if is_rate_limit:
                    if retry_attempt < max_retries - 1:
                        # Try next API key
                        next_key = rotate_to_next_groq_key()
                        if next_key:
                            print(f"    ⚠ Rate limit hit, rotating to next API key ({retry_attempt + 2}/{max_retries})...")
                            new_model = get_groq_llm_model(next_key)
                            recreate_agents_with_new_key(new_model)
                            # Agents are now recreated in the module, next iteration will pick them up
                            continue  # Retry with new key
                        else:
                            print(f"    ✗ All API keys exhausted!")
                            raise RuntimeError("All Groq API keys have been exhausted. Cannot continue.")
                    else:
                        # Last attempt failed
                        if are_all_groq_keys_exhausted():
                            raise RuntimeError("All Groq API keys have been exhausted. Cannot continue.")
                        raise
                else:
                    # Not a rate limit error, re-raise
                    raise
        
        if sentiment_result is None:
            raise RuntimeError("Failed to get sentiment result after all retries")
        
        try:
            # For Groq (plain text output), parse JSON manually
            if GROQ_API_KEYS:
                # Groq returns plain text (JSON string), need to parse it
                output_text = sentiment_result.final_output
                if isinstance(output_text, str):
                    # Remove markdown code blocks if present
                    text = output_text.strip()
                    if text.startswith("```json"):
                        text = text[7:]
                    if text.startswith("```"):
                        text = text[3:]
                    if text.endswith("```"):
                        text = text[:-3]
                    text = text.strip()
                    
                    # Try to extract JSON object if there's extra text
                    # Look for JSON object boundaries
                    json_start = text.find('{')
                    json_end = text.rfind('}')
                    if json_start != -1 and json_end != -1 and json_end > json_start:
                        text = text[json_start:json_end+1]
                    
                    # Parse JSON with better error handling
                    try:
                        json_obj = json.loads(text)
                        scores = WeeklySentimentScores(**json_obj)
                    except json.JSONDecodeError as parse_error:
                        # Try to fix common JSON issues
                        print(f"    JSON parsing error: {parse_error}")
                        print(f"    Attempting to fix JSON...")
                        
                        import re
                        # Fix 1: Remove trailing commas
                        text_fixed = re.sub(r',\s*}', '}', text)
                        text_fixed = re.sub(r',\s*]', ']', text_fixed)
                        
                        # Fix 2: Escape control characters in string values (newlines, tabs, etc.)
                        # This handles unescaped newlines in the "reasoning" field
                        def escape_control_chars(match):
                            """Escape control characters within JSON string values."""
                            string_content = match.group(1)
                            # Escape newlines, tabs, and other control characters
                            string_content = string_content.replace('\n', '\\n')
                            string_content = string_content.replace('\r', '\\r')
                            string_content = string_content.replace('\t', '\\t')
                            return f'"{string_content}"'
                        
                        # Match string values (between quotes) and escape control chars
                        # This regex matches: "key": "value" where value may contain newlines
                        def escape_string_value(match):
                            """Escape control characters in a matched string value."""
                            value = match.group(1)
                            value = value.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                            return f': "{value}"'
                        
                        text_fixed = re.sub(r':\s*"([^"]*(?:\\.|[^"\\])*)"', 
                                           escape_string_value,
                                           text_fixed)
                        
                        # Alternative: Use a more lenient approach - replace newlines in reasoning field
                        # Find the reasoning field and escape its content
                        reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*(?:\\.|[^"\\])*)"', text_fixed, re.DOTALL)
                        if reasoning_match:
                            reasoning_content = reasoning_match.group(1)
                            # Escape control characters
                            reasoning_escaped = (reasoning_content
                                                .replace('\n', '\\n')
                                                .replace('\r', '\\r')
                                                .replace('\t', '\\t')
                                                .replace('\b', '\\b')
                                                .replace('\f', '\\f'))
                            text_fixed = re.sub(r'"reasoning"\s*:\s*"[^"]*"', 
                                              f'"reasoning": "{reasoning_escaped}"',
                                              text_fixed, flags=re.DOTALL)
                        
                        try:
                            json_obj = json.loads(text_fixed)
                            scores = WeeklySentimentScores(**json_obj)
                            print(f"    ✓ Fixed JSON successfully")
                        except Exception as fix_error:
                            # Last resort: try using json5 or manual extraction
                            print(f"    ✗ Could not fix JSON automatically: {fix_error}")
                            # Try to extract just the numeric fields manually
                            try:
                                inflation_match = re.search(r'"inflation_sentiment"\s*:\s*([-0-9.]+)', text)
                                growth_match = re.search(r'"ec_growth_sentiment"\s*:\s*([-0-9.]+)', text)
                                policy_match = re.search(r'"monetary_policy_sentiment"\s*:\s*([-0-9.]+)', text)
                                vol_match = re.search(r'"market_vol_sentiment"\s*:\s*([-0-9.]+)', text)
                                date_match = re.search(r'"week_start_date"\s*:\s*"([^"]+)"', text)
                                
                                if all([inflation_match, growth_match, policy_match, vol_match, date_match]):
                                    json_obj = {
                                        "week_start_date": date_match.group(1),
                                        "inflation_sentiment": float(inflation_match.group(1)),
                                        "ec_growth_sentiment": float(growth_match.group(1)),
                                        "monetary_policy_sentiment": float(policy_match.group(1)),
                                        "market_vol_sentiment": float(vol_match.group(1)),
                                        "reasoning": "Extracted from malformed JSON"
                                    }
                                    scores = WeeklySentimentScores(**json_obj)
                                    print(f"    ✓ Extracted scores using regex fallback")
                                else:
                                    raise fix_error
                            except:
                                print(f"    ✗ Could not extract scores. Text preview: {text[:500]}")
                                raise
                    except (TypeError, ValueError) as parse_error:
                        print(f"    Validation error: {parse_error}")
                        print(f"    Text to parse: {text[:500]}")
                        raise
                else:
                    # Not a string, try normal parsing
                    scores = sentiment_result.final_output_as(WeeklySentimentScores)
            else:
                # Not using Groq, use normal parsing
                scores = sentiment_result.final_output_as(WeeklySentimentScores)
            
            # Ensure scores is a WeeklySentimentScores object
            if not isinstance(scores, WeeklySentimentScores):
                raise TypeError(f"Expected WeeklySentimentScores, got {type(scores)}: {scores}")
            
            # Ensure week_start_date is set
            if not scores.week_start_date or scores.week_start_date != week_start:
                scores.week_start_date = week_start
            
        except Exception as e:
            print(f"    Error in analysis: {e}")
            if sentiment_result is not None:
                print(f"    Output type: {type(sentiment_result.final_output)}")
                print(f"    Output preview: {str(sentiment_result.final_output)[:200]}")
            import traceback
            traceback.print_exc()
            return None
        
        # Step 2: Fact check with compact state summary
        fact_check_prompt = f"""Review these sentiment scores for the week starting {week_start}:

Inflation Sentiment: {scores.inflation_sentiment:.3f}
Economic Growth Sentiment: {scores.ec_growth_sentiment:.3f}
Monetary Policy Sentiment: {scores.monetary_policy_sentiment:.3f}
Market Volatility Sentiment: {scores.market_vol_sentiment:.3f}

Reasoning: {scores.reasoning}

News articles analyzed (first 2000 chars):
{news_text[:2000]}

{state_context}
"""
        
        # Retry logic with API key rotation for rate limits
        check_result_obj = None
        max_retries = len(GROQ_API_KEYS) if GROQ_API_KEYS else 1
        for retry_attempt in range(max_retries):
            # Always get fresh agents from the module (they may have been recreated)
            fact_checker_agent = sentiment_agents.fact_checker_agent
            
            try:
                check_result_obj = await Runner.run(fact_checker_agent, fact_check_prompt)
                break  # Success, exit retry loop
            except Exception as e:
                error_str = str(e)
                error_type_str = str(type(e))
                # Check if it's a rate limit error (TPM or TPD)
                is_rate_limit = (
                    "rate_limit" in error_str.lower() or 
                    "429" in error_str or 
                    "tokens per day" in error_str.lower() or
                    "tokens per minute" in error_str.lower() or
                    "tpm" in error_str.lower() or
                    "tpd" in error_str.lower() or
                    "RateLimitError" in error_type_str
                )
                if is_rate_limit:
                    if retry_attempt < max_retries - 1:
                        # Try next API key
                        next_key = rotate_to_next_groq_key()
                        if next_key:
                            print(f"    ⚠ Rate limit hit during fact-check, rotating to next API key ({retry_attempt + 2}/{max_retries})...")
                            new_model = get_groq_llm_model(next_key)
                            recreate_agents_with_new_key(new_model)
                            # Agents are now recreated in the module, next iteration will pick them up
                            continue  # Retry with new key
                        else:
                            print(f"    ✗ All API keys exhausted!")
                            raise RuntimeError("All Groq API keys have been exhausted. Cannot continue.")
                    else:
                        # Last attempt failed
                        if are_all_groq_keys_exhausted():
                            raise RuntimeError("All Groq API keys have been exhausted. Cannot continue.")
                        raise
                else:
                    # Not a rate limit error, re-raise
                    raise
        
        if check_result_obj is None:
            raise RuntimeError("Failed to get fact-check result after all retries")
        
        try:
            # For Groq (plain text output), parse JSON manually
            if GROQ_API_KEYS:
                # Groq returns plain text (JSON string), need to parse it
                output_text = check_result_obj.final_output
                if isinstance(output_text, str):
                    # Remove markdown code blocks if present
                    text = output_text.strip()
                    if text.startswith("```json"):
                        text = text[7:]
                    if text.startswith("```"):
                        text = text[3:]
                    if text.endswith("```"):
                        text = text[:-3]
                    text = text.strip()
                    
                    # Parse JSON
                    json_obj = json.loads(text)
                    check_result = FactCheckResult(**json_obj)
                else:
                    check_result = check_result_obj.final_output_as(FactCheckResult)
            else:
                check_result = check_result_obj.final_output_as(FactCheckResult)
            
        except Exception as e:
            print(f"    Error in fact-checking: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: approve if fact-check fails (for now)
            check_result = FactCheckResult(
                approved=True,
                issues=[],
                reasoning="Fact-check failed, approving by default"
            )
        
        # Step 3: Handle result
        if check_result.approved:
            print(f"  ✓ Approved!")
            return scores
        else:
            print(f"  ✗ Rejected: {', '.join(check_result.issues)}")
            if check_result.suggested_corrections:
                # Apply corrections if suggested
                for key, value in check_result.suggested_corrections.items():
                    if hasattr(scores, key):
                        setattr(scores, key, value)
                print(f"  Applied suggested corrections")
                return scores
            
            # If rejected and no corrections, try again with feedback
            if iteration < max_iterations - 1:
                analyze_prompt += f"\n\nFact-checker feedback: {check_result.reasoning}\nIssues: {', '.join(check_result.issues)}"
                continue
    
    print(f"  ⚠ Max iterations reached, using last attempt")
    return scores


async def main():
    """Main pipeline execution."""
    # Load news data
    data_path = Path(__file__).parent.parent.parent / "data" / "news_data" / "full_factiva.csv"
    print(f"Loading news data from {data_path}...")
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"Loaded {len(df):,} articles")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Group by week
    weekly_news = group_news_by_week(df)
    print(f"\nFound {len(weekly_news)} weeks of data")
    
    # Sort weeks chronologically
    sorted_weeks = sorted(weekly_news.keys())
    
    # Process weeks using compact state summary instead of full history
    all_scores: List[WeeklySentimentScores] = []
    state_summary = GlobalStateSummary()
    
    # Process all weeks (full dataset)
    weeks_to_process = sorted_weeks
    print(f"\n🚀 FULL RUN: Processing {len(weeks_to_process)} weeks...")
    print(f"   Date range: {weeks_to_process[0]} to {weeks_to_process[-1]}")
    print(f"   Estimated time: ~{len(weeks_to_process) * 0.5:.1f} minutes (assuming ~30s per week)")
    
    # Set up output file with incremental saving
    output_path = Path(__file__).parent / "sentiment_scores.csv"
    file_exists = output_path.exists()
    
    # Initialize CSV file if it doesn't exist
    if not file_exists:
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'date',
                'inflation_sentiment',
                'ec_growth_sentiment',
                'monetary_policy_sentiment',
                'market_vol_sentiment'
            ])
    
    # Load existing scores to avoid duplicates
    existing_dates = set()
    if file_exists:
        try:
            existing_df = pd.read_csv(output_path)
            if 'date' in existing_df.columns:
                existing_dates = set(existing_df['date'].astype(str))
                print(f"   Found {len(existing_dates)} existing weeks in CSV, will skip duplicates")
        except:
            pass
    
    processed_count = 0
    skipped_count = 0
    
    for week_start in weeks_to_process:
        # Skip if already processed
        if str(week_start) in existing_dates:
            skipped_count += 1
            continue
            
        week_data = weekly_news[week_start]
        scores = await process_week(week_start, week_data, state_summary)
        
        if scores:
            # Append to CSV immediately (incremental save)
            with open(output_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    scores.week_start_date,
                    scores.inflation_sentiment,
                    scores.ec_growth_sentiment,
                    scores.monetary_policy_sentiment,
                    scores.market_vol_sentiment,
                ])
            
            all_scores.append(scores)
            # Update state summary instead of keeping full history
            state_summary = update_state_summary(state_summary, scores)
            processed_count += 1
            
            # Print progress every 10 weeks
            if processed_count % 10 == 0:
                print(f"   Progress: {processed_count} weeks processed, {skipped_count} skipped")
        else:
            print(f"  Failed to get scores for {week_start}")
    
    print(f"\n✓ Saved {processed_count} new weeks of sentiment scores to {output_path}")
    print(f"   Total weeks in CSV: {processed_count + skipped_count}")
    if skipped_count > 0:
        print(f"   Skipped {skipped_count} duplicate weeks")


if __name__ == "__main__":
    asyncio.run(main())
