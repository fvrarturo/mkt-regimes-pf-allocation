"""
Three agents for weekly news sentiment analysis:
1. Sentiment Analyzer Agent - aggregates news and produces sentiment scores
2. Fact Checker Agent - verifies scores for consistency
3. Coordinator Agent - orchestrates the workflow
"""
import statistics
import json
from typing import Optional, Any
from pydantic import BaseModel, Field
from agents import Agent, AgentOutputSchema, AgentOutputSchemaBase, Runner, function_tool, set_tracing_disabled
from agents.exceptions import ModelBehaviorError
import os

# Import LLM configuration (supports both Ollama local and OpenAI API)
from llm_config import (
    LLM_MODEL, 
    GROQ_API_KEYS, 
    get_groq_llm_model,
    rotate_to_next_groq_key,
    are_all_groq_keys_exhausted,
    get_current_groq_key_index,
)

# Disable tracing for local models
set_tracing_disabled(True)

# Check if we're using Groq (which doesn't support json_schema response format at all)
USE_GROQ = len(GROQ_API_KEYS) > 0


# Custom output schema for Groq that treats output as plain text but parses JSON
class GroqCompatibleOutputSchema(AgentOutputSchemaBase):
    """Output schema that works with Groq by using plain text output but parsing JSON."""
    def __init__(self, output_type: type[BaseModel]):
        self.output_type = output_type
        from pydantic import TypeAdapter
        self._type_adapter = TypeAdapter(output_type)
        self._json_schema = self._type_adapter.json_schema()
    
    def is_plain_text(self) -> bool:
        """Return True so framework doesn't send json_schema format"""
        return True
    
    def name(self) -> str:
        return self.output_type.__name__
    
    def json_schema(self) -> dict:
        """Return schema for validation"""
        return self._json_schema
    
    def is_strict_json_schema(self) -> bool:
        return False
    
    def validate_json(self, json_str: str) -> Any:
        """Parse JSON from text output"""
        # Try to extract JSON from text (might be wrapped in markdown code blocks)
        text = json_str.strip()
        
        # Remove markdown code blocks if present
        if text.startswith("```json"):
            text = text[7:]  # Remove ```json
        if text.startswith("```"):
            text = text[3:]  # Remove ```
        if text.endswith("```"):
            text = text[:-3]  # Remove closing ```
        text = text.strip()
        
        # Parse JSON
        try:
            json_obj = json.loads(text)
        except json.JSONDecodeError as e:
            raise ModelBehaviorError(f"Invalid JSON from model: {text[:200]}... Error: {e}")
        
        # Validate against Pydantic model
        try:
            return self._type_adapter.validate_python(json_obj)
        except Exception as e:
            raise ModelBehaviorError(f"Validation failed for JSON: {json_obj}. Error: {e}")


# ============================================================================
# Data Models
# ============================================================================

class WeeklySentimentScores(BaseModel):
    """Sentiment scores for a single week."""
    week_start_date: str = Field(description="Monday date of the week (YYYY-MM-DD)")
    inflation_sentiment: float = Field(description="Inflation sentiment score (-1 to 1, where -1 is extremely negative, 0 is neutral, 1 is extremely positive)")
    ec_growth_sentiment: float = Field(description="Economic growth sentiment score (-1 extremely negative to 1 extremely positive)")
    monetary_policy_sentiment: float = Field(description="Monetary policy sentiment score (-1 to 1, where -1 is extremely doveish/easing, 1 is extremely hawkish/tightening)")
    market_vol_sentiment: float = Field(description="Market volatility/stress sentiment score (-1 to 1, where -1 is extremely low volatility/calm, 1 is extremely high volatility/stress)")
    reasoning: str = Field(description="Brief explanation of the scores")


class FactCheckResult(BaseModel):
    """Result of fact-checking sentiment scores."""
    approved: bool = Field(description="Whether the scores are approved")
    issues: list[str] = Field(description="List of issues found, if any")
    suggested_corrections: Optional[dict[str, float]] = Field(
        default=None,
        description="Suggested corrected scores if issues found (keys: inflation_sentiment, ec_growth_sentiment, monetary_policy_sentiment, market_vol_sentiment)"
    )
    reasoning: str = Field(description="Explanation of the fact-check decision")


class RecentWeekEntry(BaseModel):
    """Entry for a single week in the recent weeks list."""
    week: str = Field(description="Week start date (YYYY-MM-DD)")
    inflation: float = Field(description="Inflation sentiment score")
    growth: float = Field(description="Economic growth sentiment score")
    policy: float = Field(description="Monetary policy sentiment score")
    vol: float = Field(description="Market volatility sentiment score")


class GlobalStateSummary(BaseModel):
    """Compact summary of sentiment analysis state for consistency checking."""
    # Dimension definitions (static, but included for context)
    dimension_definitions: str = Field(
        default="""Sentiment scores range from -1 to 1:
- Inflation Sentiment: -1 (extremedeflation concerns) to 1 (extreme inflation concerns)
- Economic Growth: -1 (extreme recession fears) to 1 (extreme strong growth)
- Monetary Policy: -1 (extremedovish/easing) to 1 (extreme hawkish/tightening)
- Market Volatility: -1 (extremely low volatility/calm) to 1 (extremely high volatility/stress)""",
        description="Definitions of the four sentiment dimensions"
    )
    
    # Recent context (last 2-3 weeks for immediate consistency)
    recent_weeks: list[RecentWeekEntry] = Field(
        default_factory=list,
        description="Last 2-3 weeks of scores"
    )
    
    # Long-term trends (descriptive summaries)
    inflation_trend: str = Field(
        default="",
        description="Trend description: e.g., 'trending upward over past 3 months', 'stable around neutral', 'volatile'"
    )
    growth_trend: str = Field(
        default="",
        description="Economic growth trend description"
    )
    policy_trend: str = Field(
        default="",
        description="Monetary policy trend description"
    )
    volatility_trend: str = Field(
        default="",
        description="Market volatility trend description"
    )
    
    # Statistical summaries (compact)
    inflation_stats: str = Field(
        default="",
        description="Recent statistics: e.g., 'avg: 0.2, range: [-0.1, 0.5]'"
    )
    growth_stats: str = Field(
        default="",
        description="Economic growth statistics"
    )
    policy_stats: str = Field(
        default="",
        description="Monetary policy statistics"
    )
    volatility_stats: str = Field(
        default="",
        description="Market volatility statistics"
    )
    
    # Cross-dimension relationships observed
    observed_relationships: str = Field(
        default="",
        description="Notable relationships: e.g., 'inflation and policy typically move together', 'growth and volatility inversely correlated'"
    )
    
    # Total weeks processed
    weeks_processed: int = Field(
        default=0,
        description="Total number of weeks processed so far"
    )
    
    # Date range
    date_range: str = Field(
        default="",
        description="Date range covered: 'YYYY-MM-DD to YYYY-MM-DD'"
    )


# ============================================================================
# Agent 1: Sentiment Analyzer Agent
# ============================================================================

SENTIMENT_ANALYZER_INSTRUCTIONS = """You are a macroeconomic analyst specializing in analyzing financial news.

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

TEMPORAL KNOWLEDGE RESTRICTION (NO FUTURE INFORMATION):
You must evaluate the economic and financial sentiment of this week using only the information
contained in the provided articles and your deep macroeconomic reasoning.

You are NOT allowed to use or rely on:
- Knowledge of future events, crisis outcomes, recessions, market peaks, or any later developments.
- Historical interpretations, academic analyses, or widely-known ex-post narratives.
- Any real-world timeline of what eventually happened after this week.

However, you ARE allowed to:
- Use your full economic knowledge, reasoning ability, and understanding of macroeconomic mechanisms.
- Infer risks, structural problems, vulnerabilities, momentum shifts, or early signals based solely on the news.
- Make forward-looking judgments as long as they come strictly from patterns present in the text itself.
- Be smarter, more analytical, and more pattern-aware than real analysts at that time — but not clairvoyant.

In short: you are the best macroeconomic analyst in the world with no access to future facts.
You may infer, but you may not know.

When analyzing:
- Consider the overall tone and frequency of mentions
- Consider the impact of the news on the economy and the market
- Weight more recent articles more heavily if dates are provided
- Look for trends and consensus rather than outliers
- Consider the context and implications of the news
- Be consistent with economic logic (e.g., high inflation often correlates with hawkish policy expectations)

**SMOOTHING GUIDELINE**: Macroeconomic sentiment evolves gradually. Unless there's a major crisis or policy announcement, scores should change incrementally week-to-week (typically < 0.2 points). Avoid dramatic jumps - if sentiment needs to shift significantly, it should happen gradually over multiple weeks. That said, if the news is very clear and there is a major event, the score can change significantly in one week.

Output your analysis as structured sentiment scores with brief reasoning."""

# Use custom schema for Groq (doesn't support json_schema response format at all)
# For Groq, we use plain text output and parse JSON manually
if USE_GROQ:
    # Add JSON format instruction for Groq
    groq_instructions = SENTIMENT_ANALYZER_INSTRUCTIONS + """

IMPORTANT: You must output your response as valid JSON only, matching this exact structure:
{
  "week_start_date": "YYYY-MM-DD",
  "inflation_sentiment": <float between -1 and 1>,
  "ec_growth_sentiment": <float between -1 and 1>,
  "monetary_policy_sentiment": <float between -1 and 1>,
  "market_vol_sentiment": <float between -1 and 1>,
  "reasoning": "<brief explanation>"
}

Output ONLY the JSON object, no other text, no markdown formatting."""
    
    sentiment_analyzer_agent = Agent(
        name="SentimentAnalyzer",
        instructions=groq_instructions,
        model=LLM_MODEL,
        output_type=GroqCompatibleOutputSchema(WeeklySentimentScores),
    )
else:
    sentiment_analyzer_agent = Agent(
        name="SentimentAnalyzer",
        instructions=SENTIMENT_ANALYZER_INSTRUCTIONS,
        model=LLM_MODEL,
        output_type=WeeklySentimentScores,
    )


# ============================================================================
# Agent 2: Fact Checker Agent
# ============================================================================

FACT_CHECKER_INSTRUCTIONS = """You are a fact-checker for macroeconomic sentiment scores.

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

Be strict about smoothing - macroeconomic sentiment evolves gradually, not in dramatic jumps. Focus on enforcing gradual transitions."""

# Use custom schema for Groq
if USE_GROQ:
    # Add JSON format instruction for Groq
    groq_fact_check_instructions = FACT_CHECKER_INSTRUCTIONS + """

IMPORTANT: You must output your response as valid JSON only, matching this exact structure:
{
  "approved": <true or false>,
  "issues": ["<issue1>", "<issue2>", ...],
  "suggested_corrections": {"inflation_sentiment": <float>, ...} or null,
  "reasoning": "<explanation>"
}

Output ONLY the JSON object, no other text, no markdown formatting."""
    
    fact_checker_agent = Agent(
        name="FactChecker",
        instructions=groq_fact_check_instructions,
        model=LLM_MODEL,
        output_type=GroqCompatibleOutputSchema(FactCheckResult),
    )
else:
    fact_checker_agent = Agent(
        name="FactChecker",
        instructions=FACT_CHECKER_INSTRUCTIONS,
        model=LLM_MODEL,
        output_type=FactCheckResult,
    )


# ============================================================================
# Agent 3: Coordinator Agent
# ============================================================================

COORDINATOR_INSTRUCTIONS = """You are a coordinator managing a sentiment analysis pipeline.

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

Be methodical and ensure quality - it's better to iterate than to accept inconsistent scores."""

coordinator_agent = Agent(
    name="Coordinator",
    instructions=COORDINATOR_INSTRUCTIONS,
    model=LLM_MODEL,
    tools=[
        sentiment_analyzer_agent.as_tool(
            tool_name="analyze_sentiment",
            tool_description="Analyze news articles for a given week and produce sentiment scores"
        ),
        fact_checker_agent.as_tool(
            tool_name="fact_check_scores",
            tool_description="Verify sentiment scores for consistency and accuracy"
        ),
    ],
)


# ============================================================================
# Functions to recreate agents with new API keys (for rotation)
# ============================================================================

def recreate_agents_with_new_key(new_llm_model):
    """Recreate all agents with a new LLM model (for API key rotation)."""
    global sentiment_analyzer_agent, fact_checker_agent, coordinator_agent
    
    # Debug: Verify the new model has the correct API key
    api_key = new_llm_model._get_client().api_key if hasattr(new_llm_model, '_get_client') else 'unknown'
    print(f"    🔑 Using API key: {api_key[:20]}...")
    
    # Recreate sentiment analyzer
    if USE_GROQ:
        groq_instructions = SENTIMENT_ANALYZER_INSTRUCTIONS + """

IMPORTANT: You must output your response as valid JSON only, matching this exact structure:
{
  "week_start_date": "YYYY-MM-DD",
  "inflation_sentiment": <float between -1 and 1>,
  "ec_growth_sentiment": <float between -1 and 1>,
  "monetary_policy_sentiment": <float between -1 and 1>,
  "market_vol_sentiment": <float between -1 and 1>,
  "reasoning": "<brief explanation>"
}

Output ONLY the JSON object, no other text, no markdown formatting."""
        sentiment_analyzer_agent = Agent(
            name="SentimentAnalyzer",
            instructions=groq_instructions,
            model=new_llm_model,
            output_type=GroqCompatibleOutputSchema(WeeklySentimentScores),
        )
    else:
        sentiment_analyzer_agent = Agent(
            name="SentimentAnalyzer",
            instructions=SENTIMENT_ANALYZER_INSTRUCTIONS,
            model=new_llm_model,
            output_type=WeeklySentimentScores,
        )
    
    # Recreate fact checker
    if USE_GROQ:
        groq_fact_check_instructions = FACT_CHECKER_INSTRUCTIONS + """

IMPORTANT: You must output your response as valid JSON only, matching this exact structure:
{
  "approved": <true or false>,
  "issues": ["<issue1>", "<issue2>", ...],
  "suggested_corrections": {"inflation_sentiment": <float>, ...} or null,
  "reasoning": "<explanation>"
}

Output ONLY the JSON object, no other text, no markdown formatting."""
        fact_checker_agent = Agent(
            name="FactChecker",
            instructions=groq_fact_check_instructions,
            model=new_llm_model,
            output_type=GroqCompatibleOutputSchema(FactCheckResult),
        )
    else:
        fact_checker_agent = Agent(
            name="FactChecker",
            instructions=FACT_CHECKER_INSTRUCTIONS,
            model=new_llm_model,
            output_type=FactCheckResult,
        )
    
    # Recreate coordinator with updated tools
    coordinator_agent = Agent(
        name="Coordinator",
        instructions=COORDINATOR_INSTRUCTIONS,
        model=new_llm_model,
        tools=[
            sentiment_analyzer_agent.as_tool(
                tool_name="analyze_sentiment",
                tool_description="Analyze news articles for a given week and produce sentiment scores"
            ),
            fact_checker_agent.as_tool(
                tool_name="fact_check_scores",
                tool_description="Verify sentiment scores for consistency and accuracy"
            ),
        ],
    )
    
    print(f"✓ Recreated all agents with new API key")


# ============================================================================
# State Summary Management
# ============================================================================

def update_state_summary(
    state: GlobalStateSummary,
    new_score: WeeklySentimentScores,
    lookback_weeks: int = 8
) -> GlobalStateSummary:
    """
    Update the global state summary with a new week's scores.
    This keeps the context compact by maintaining trends and statistics instead of full history.
    """
    # Add to recent weeks (keep only last 2-3)
    recent_entry = RecentWeekEntry(
        week=new_score.week_start_date,
        inflation=new_score.inflation_sentiment,
        growth=new_score.ec_growth_sentiment,
        policy=new_score.monetary_policy_sentiment,
        vol=new_score.market_vol_sentiment,
    )
    state.recent_weeks.append(recent_entry)
    # Keep only last 2 weeks for immediate consistency checks
    if len(state.recent_weeks) > 2:
        state.recent_weeks = state.recent_weeks[-2:]
    
    # Update weeks processed
    state.weeks_processed += 1
    
    # Update date range
    if not state.date_range:
        state.date_range = f"{new_score.week_start_date} to {new_score.week_start_date}"
    else:
        start_date = state.date_range.split(" to ")[0]
        state.date_range = f"{start_date} to {new_score.week_start_date}"
    
    # For trend calculation, we'd need access to more historical data
    # For now, we'll calculate trends from recent weeks only
    # In a real implementation, you might maintain rolling windows
    
    # Calculate statistics from recent weeks (if we had more data, use lookback_weeks)
    if len(state.recent_weeks) >= 2:
        inflations = [w.inflation for w in state.recent_weeks]
        growths = [w.growth for w in state.recent_weeks]
        policies = [w.policy for w in state.recent_weeks]
        vols = [w.vol for w in state.recent_weeks]
        
        def format_stats(values):
            avg = statistics.mean(values)
            min_val = min(values)
            max_val = max(values)
            return f"avg: {avg:.2f}, range: [{min_val:.2f}, {max_val:.2f}]"
        
        state.inflation_stats = format_stats(inflations)
        state.growth_stats = format_stats(growths)
        state.policy_stats = format_stats(policies)
        state.volatility_stats = format_stats(vols)
        
        # Calculate trends (simple directional)
        if len(inflations) >= 2:
            state.inflation_trend = "trending upward" if inflations[-1] > inflations[-2] else "trending downward" if inflations[-1] < inflations[-2] else "stable"
            state.growth_trend = "trending upward" if growths[-1] > growths[-2] else "trending downward" if growths[-1] < growths[-2] else "stable"
            state.policy_trend = "trending upward" if policies[-1] > policies[-2] else "trending downward" if policies[-1] < policies[-2] else "stable"
            state.volatility_trend = "trending upward" if vols[-1] > vols[-2] else "trending downward" if vols[-1] < vols[-2] else "stable"
    
    # Update observed relationships (simple correlation checks)
    if len(state.recent_weeks) >= 3:
        # Check if inflation and policy move together
        inflations = [w.inflation for w in state.recent_weeks]
        policies = [w.policy for w in state.recent_weeks]
        # Simple check: if both trending same direction
        if (inflations[-1] > inflations[-2] and policies[-1] > policies[-2]) or \
           (inflations[-1] < inflations[-2] and policies[-1] < policies[-2]):
            state.observed_relationships = "Inflation and monetary policy typically move together"
    
    return state


def format_state_summary(state: GlobalStateSummary) -> str:
    """Format the state summary as a compact string for the fact-checker."""
    lines = [
        "=== GLOBAL STATE SUMMARY ===",
        f"Weeks processed: {state.weeks_processed}",
        f"Date range: {state.date_range}",
        "",
        "Dimension Definitions:",
        state.dimension_definitions,
        "",
        "Recent Weeks (last 2):",
    ]
    
    for week in state.recent_weeks:
        lines.append(
            f"  {week.week}: "
            f"inflation={week.inflation:.2f}, "
            f"growth={week.growth:.2f}, "
            f"policy={week.policy:.2f}, "
            f"vol={week.vol:.2f}"
        )
    
    if state.inflation_trend:
        lines.extend([
            "",
            "Trends:",
            f"  Inflation: {state.inflation_trend}",
            f"  Growth: {state.growth_trend}",
            f"  Policy: {state.policy_trend}",
            f"  Volatility: {state.volatility_trend}",
        ])
    
    if state.inflation_stats:
        lines.extend([
            "",
            "Recent Statistics:",
            f"  Inflation: {state.inflation_stats}",
            f"  Growth: {state.growth_stats}",
            f"  Policy: {state.policy_stats}",
            f"  Volatility: {state.volatility_stats}",
        ])
    
    if state.observed_relationships:
        lines.extend([
            "",
            "Observed Relationships:",
            f"  {state.observed_relationships}",
        ])
    
    return "\n".join(lines)

