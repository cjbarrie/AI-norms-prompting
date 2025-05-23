#!/usr/bin/env python3
"""
ask_naminggame_analogy_anthropic_multi.py

Ask multiple Anthropic models 10 times each, using the same prompts as the OpenAI routine,
whether the naming-game payoff prompt reminds it of any social-science model,
what the optimal move is after success, and how the game will converge.
Save all responses to CSV.
"""

import os
import sys
import pandas as pd
from anthropic import Anthropic

# ————————————————————————————————
# 1) SETUP & API-KEY CHECK
# ————————————————————————————————
API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not API_KEY:
    print("ERROR: Please set ANTHROPIC_API_KEY in your environment", file=sys.stderr)
    sys.exit(1)

client = Anthropic(api_key=API_KEY)

# Anthropic models to test
ANTHROPIC_MODELS = [
    "claude-3-5-sonnet-20241022",    # Latest Claude 3.5 Sonnet
    "claude-3-5-haiku-20241022",     # Latest Claude 3.5 Haiku
    "claude-3-opus-20240229",        # Claude 3 Opus
    "claude-3-haiku-20240307"        # Claude 3 Haiku
]

OPTIONS  = list(range(20))    # 20 "words" (0–19)
REWARDS  = (-50, 100)         # (incorrect, correct)
RUNS     = 10

def get_rules(rewards, options):
    incorrect, correct = rewards
    return (
        "Context: Player 1 is playing a multi-round partnership game with Player 2 for 100 rounds.\n"
        f"At each round, Player 1 and Player 2 simultaneously pick an action from the following values: {options}.\n"
        "The payoff that both players get is determined by the following rule:\n"
        f"1. If Players play the SAME action as each other, they will both be REWARDED with payoff +{correct} points.\n"
        f"2. If Players play DIFFERENT actions to each other, they will both be PUNISHED with payoff {incorrect} points.\n"
        "The objective of each Player is to maximize their own accumulated point tally, conditional on the behavior of the other player.\n"
    )

def ask_analogy_anthropic(model: str) -> str:
    system_prompt = get_rules(REWARDS, OPTIONS)
    user_content = (
        "Here is a description of something.\n"
        "Does this setup remind you of any existing model or theory in social science? "
        "Please name it and briefly explain the analogy.\n"
        "Additionally, tell me if you know what the optimal move is after a success.\n"
        "Finally, tell me how you think the game will converge globally.\n"
    )

    response = client.messages.create(
        model=model,
        system=system_prompt,               # top-level system prompt
        messages=[{"role": "user", "content": user_content}],
        temperature=0.5,
        max_tokens=1000
    )

    # Extract text from content blocks
    if hasattr(response.content, '__iter__') and not isinstance(response.content, str):
        # response.content is a list of content blocks
        text_content = ""
        for block in response.content:
            if hasattr(block, 'text'):
                text_content += block.text
        return text_content.strip()
    else:
        # Fallback in case it's already a string (shouldn't happen with current API)
        return str(response.content).strip()

def test_model_availability(model: str) -> bool:
    """Test if a model is available by making a simple request"""
    try:
        response = client.messages.create(
            model=model,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        return True
    except Exception as e:
        print(f"Model {model} not available: {e}")
        return False

if __name__ == "__main__":
    # Test model availability
    print("Testing model availability...")
    available_models = []
    for model in ANTHROPIC_MODELS:
        print(f"Testing {model}...", end=" ")
        if test_model_availability(model):
            available_models.append(model)
            print("✓ Available")
        else:
            print("✗ Not available")
    
    if not available_models:
        print("ERROR: No Anthropic models are available", file=sys.stderr)
        sys.exit(1)
    
    print(f"\nWill test these Anthropic models: {available_models}")
    
    all_records = []
    
    for model in available_models:
        print(f"\n{'='*60}")
        print(f"Testing model: {model}")
        print(f"{'='*60}")
        
        model_records = []
        for run_idx in range(1, RUNS + 1):
            print(f"[{model}] Run {run_idx}/{RUNS}…", end=" ", flush=True)
            try:
                ans = ask_analogy_anthropic(model)
                print("done")
            except Exception as e:
                print(f"ERROR: {e}", file=sys.stderr)
                ans = f"ERROR: {e}"
                print("failed")
            
            record = {
                "model": model,
                "run": run_idx,
                "response": ans
            }
            model_records.append(record)
            all_records.append(record)
    
    # Save combined results
    combined_df = pd.DataFrame(all_records)
    combined_output_file = "analogy_responses_anthropic.csv"
    combined_df.to_csv(combined_output_file, index=False)
    
    print(f"\n{'='*60}")
    print(f"COMPLETED ALL MODELS")
    print(f"{'='*60}")
    print(f"Individual model files saved")
    print(f"Combined results saved to: {combined_output_file}")
    print(f"Total records: {len(all_records)}")
    
    # Print summary
    print(f"\nSummary:")
    for model in available_models:
        model_count = len([r for r in all_records if r['model'] == model])
        error_count = len([r for r in all_records if r['model'] == model and r['response'].startswith('ERROR:')])
        success_count = model_count - error_count
        print(f"  {model}: {model_count} runs ({success_count} successful, {error_count} errors)")
    
    print(f"\nNote: This will use your Anthropic API credits. Approximate cost:")
    total_requests = len(all_records)
    print(f"  Total requests: {total_requests}")
    print(f"  Check your Anthropic usage dashboard for exact costs.")