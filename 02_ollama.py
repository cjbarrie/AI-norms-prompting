#!/usr/bin/env python3
"""
ask_naminggame_analogy_ollama.py

Ask an Ollama model 10 times, using the same prompts as the OpenAI routine,
whether the naming-game payoff prompt reminds it of any social-science model,
what the optimal move is after success, and how the game will converge.
Save all responses to CSV.
"""

import os
import sys
import pandas as pd
import requests
import json

# ————————————————————————————————
# 1) SETUP & CONFIGURATION
# ————————————————————————————————

# Ollama server configuration (default local installation)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Llama models to test (based on your available models)
LLAMA_MODELS = [
    "llama3.2:3b",
    "llama3:instruct",
    "llama3:70b-instruct",
    "deepseek-r1:8b", 
    "mistral:7b",
    "gemma3:4b"]

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

def ask_analogy_ollama(model: str) -> str:
    system_prompt = get_rules(REWARDS, OPTIONS)
    user_content = (
        "Here is a description of something.\n"
        "Does this setup remind you of any existing model or theory in social science? "
        "Please name it and briefly explain the analogy.\n"
        "Additionally, tell me if you know what the optimal move is after a success.\n"
        "Finally, tell me how you think the game will converge globally.\n"
    )
    
    # Combine system and user prompts for models that don't support system messages
    full_prompt = f"{system_prompt}\n\n{user_content}"
    
    # Ollama API payload
    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": False,  # Get complete response at once
        "options": {
            "temperature": 0.5,
            "num_predict": 1000  # Equivalent to max_tokens
        }
    }
    
    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json=payload,
            timeout=120  # 2 minute timeout for large models
        )
        response.raise_for_status()
        
        result = response.json()
        return result.get("response", "").strip()
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Ollama API request failed: {e}")
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse Ollama response: {e}")

def check_ollama_connection():
    """Check if Ollama server is running and return available models"""
    try:
        # Check server status
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
        response.raise_for_status()
        
        # Get available models
        models = response.json().get("models", [])
        model_names = [m.get("name", "") for m in models]
        
        print(f"Available Ollama models: {model_names}")
        return model_names
        
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Cannot connect to Ollama server at {OLLAMA_HOST}", file=sys.stderr)
        print(f"Make sure Ollama is running with: ollama serve", file=sys.stderr)
        print(f"Connection error: {e}", file=sys.stderr)
        return None

if __name__ == "__main__":
    # Check Ollama connection before starting
    available_models = check_ollama_connection()
    if available_models is None:
        sys.exit(1)
    
    # Filter LLAMA_MODELS to only include available ones
    available_llama_models = [model for model in LLAMA_MODELS if model in available_models]
    
    if not available_llama_models:
        print("ERROR: No Llama models found among available models", file=sys.stderr)
        print(f"Available models: {available_models}", file=sys.stderr)
        print(f"Looking for: {LLAMA_MODELS}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Will test these Llama models: {available_llama_models}")
    
    all_records = []
    
    for model in available_llama_models:
        print(f"\n{'='*60}")
        print(f"Testing model: {model}")
        print(f"{'='*60}")
        
        model_records = []
        for run_idx in range(1, RUNS + 1):
            print(f"[{model}] Run {run_idx}/{RUNS}…", end=" ", flush=True)
            try:
                ans = ask_analogy_ollama(model)
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
    combined_output_file = "analogy_responses_ollama.csv"
    combined_df.to_csv(combined_output_file, index=False)
    
    print(f"\n{'='*60}")
    print(f"COMPLETED ALL MODELS")
    print(f"{'='*60}")
    print(f"Individual model files saved")
    print(f"Combined results saved to: {combined_output_file}")
    print(f"Total records: {len(all_records)}")
    
    # Print summary
    print(f"\nSummary:")
    for model in available_llama_models:
        model_count = len([r for r in all_records if r['model'] == model])
        error_count = len([r for r in all_records if r['model'] == model and r['response'].startswith('ERROR:')])
        print(f"  {model}: {model_count} runs ({error_count} errors)")