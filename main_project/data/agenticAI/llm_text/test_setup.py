"""
Test script to verify the setup is working correctly.
Run this before running the main pipeline.
"""
import sys
from pathlib import Path
import asyncio

from agents import Runner, set_tracing_disabled
set_tracing_disabled(True)

from sentiment_agents import sentiment_analyzer_agent, WeeklySentimentScores


async def test_agent():
    """Test that the sentiment analyzer agent works."""
    print("Testing sentiment analyzer agent...")
    print("(Make sure Ollama is running: 'ollama serve')")
    print()
    
    test_news = """
    [2024-01-08 10:00]
    Headline: Fed Signals Potential Rate Cuts as Inflation Cools
    Snippet: Federal Reserve officials indicated they may consider cutting interest rates as inflation fshows signs of moderating. Economic growth remains steady but concerns about slowing consumer spending persist.
    
    [2024-01-09 14:30]
    Headline: Stock Market Volatility Rises Amid Economic Uncertainty
    Snippet: Major indices experienced increased volatility as investors weigh mixed economic signals. VIX index climbed to elevated levels.
    """
    
    prompt = f"""Analyze the following news articles from the week starting 2024-01-08 and produce sentiment scores.

News articles:
{test_news}
"""
    
    try:
        print("Sending request to agent...")
        result = await Runner.run(sentiment_analyzer_agent, prompt)
        scores = result.final_output_as(WeeklySentimentScores)
        
        print("\n✓ Agent responded successfully!")
        print(f"\nResults:")
        print(f"  Week: {scores.week_start_date}")
        print(f"  Inflation Sentiment: {scores.inflation_sentiment:.3f}")
        print(f"  Economic Growth Sentiment: {scores.ec_growth_sentiment:.3f}")
        print(f"  Monetary Policy Sentiment: {scores.monetary_policy_sentiment:.3f}")
        print(f"  Market Volatility Sentiment: {scores.market_vol_sentiment:.3f}")
        print(f"\nReasoning: {scores.reasoning}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Ollama is running: 'ollama serve'")
        print("2. Make sure llama3 model is installed: 'ollama pull llama3'")
        print("3. Check that the agents framework is accessible")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_agent())
    sys.exit(0 if success else 1)

