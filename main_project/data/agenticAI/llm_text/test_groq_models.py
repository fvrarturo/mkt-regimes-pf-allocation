"""
Test script to test different Groq models, including groq/compound
"""
import sys
import os
import asyncio
import json
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from agents import Runner, set_tracing_disabled, Agent, AgentOutputSchemaBase
from agents.exceptions import ModelBehaviorError
set_tracing_disabled(True)

from openai import AsyncOpenAI
from agents import OpenAIChatCompletionsModel
from pydantic import BaseModel, Field, TypeAdapter

# Test model class
class TestScores(BaseModel):
    """Simple test output."""
    test_result: str = Field(description="Test result message")
    model_used: str = Field(description="Model that was used")


# Custom output schema for Groq (same as in sentiment_agents.py)
class GroqCompatibleOutputSchema(AgentOutputSchemaBase):
    """Output schema that works with Groq by using plain text output but parsing JSON."""
    def __init__(self, output_type: type[BaseModel]):
        self.output_type = output_type
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
    
    def validate_json(self, json_str: str) -> any:
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


async def test_groq_model(model_name: str, api_key: str):
    """Test a specific Groq model."""
    print(f"\n{'='*60}")
    print(f"Testing model: {model_name}")
    print(f"{'='*60}")
    
    try:
        # Create model
        model = OpenAIChatCompletionsModel(
            model=model_name,
            openai_client=AsyncOpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=api_key,
            ),
        )
        
        # Create a simple agent with Groq-compatible output schema
        agent = Agent(
            name="TestAgent",
            instructions="You are a helpful assistant. Respond with a simple test message. Output ONLY valid JSON matching this structure: {\"test_result\": \"your message\", \"model_used\": \"model name\"}. No markdown, no other text.",
            model=model,
            output_type=GroqCompatibleOutputSchema(TestScores),
        )
        
        # Test prompt
        prompt = "Say hello and confirm you're working. Return a JSON with test_result='success' and model_used='your model name'"
        
        print(f"Sending test request...")
        result = await Runner.run(agent, prompt)
        
        # Handle Groq's plain text JSON output
        if isinstance(result.final_output, str):
            # Manual JSON parsing for Groq
            text = result.final_output.strip()
            print(f"  Raw output (first 200 chars): {text[:200]}")
            
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            
            # Try to extract JSON from text
            try:
                json_obj = json.loads(text)
                output = TestScores(**json_obj)
            except json.JSONDecodeError:
                # Try to find JSON object in the text
                import re
                json_match = re.search(r'\{[^{}]*"test_result"[^{}]*\}', text)
                if json_match:
                    json_obj = json.loads(json_match.group(0))
                    output = TestScores(**json_obj)
                else:
                    raise ValueError(f"Could not parse JSON from output: {text[:500]}")
        else:
            output = result.final_output_as(TestScores)
        
        print(f"\n✓ SUCCESS!")
        print(f"  Model: {model_name}")
        print(f"  Response: {output.test_result}")
        print(f"  Model confirmed: {output.model_used}")
        return True
        
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Test multiple Groq models."""
    # Get API key from environment
    groq_keys_str = os.getenv("GROQ_API_KEYS", "")
    groq_key = os.getenv("GROQ_API_KEY")
    
    if groq_keys_str:
        groq_key = groq_keys_str.split(",")[0].strip()
    elif not groq_key:
        print("Error: Set GROQ_API_KEY or GROQ_API_KEYS environment variable")
        print("Example: export GROQ_API_KEY='gsk_your-key-here'")
        return False
    
    print(f"Using API key: {groq_key[:20]}...")
    
    # Test models
    models_to_test = [
        "llama-3.1-8b-instant",  # Current default
        "groq/compound",         # New model with unlimited TPD
        "groq/compound-mini",    # Mini version
    ]
    
    results = {}
    for model_name in models_to_test:
        success = await test_groq_model(model_name, groq_key)
        results[model_name] = success
        await asyncio.sleep(1)  # Brief pause between tests
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    for model, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {model}")
    
    return all(results.values())


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

