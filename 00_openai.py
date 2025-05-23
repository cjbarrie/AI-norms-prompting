#!/usr/bin/env python3
"""
ask_models_naminggame_analogy_repeat.py

Ask each OpenAI model 10 times whether the naming-game payoff prompt
reminds it of any social-science model, then save all responses to CSV.
"""

import os
import pandas as pd
from openai import OpenAI

# ————————————————————————————————
# 1) SETUP
# ————————————————————————————————
# grab it from the env
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("Please set the OPENAI_API_KEY environment variable")

client = OpenAI(api_key=API_KEY)

MODELS  = ["gpt-4o-mini", "gpt-4o", "gpt-4.1", "gpt-3.5-turbo"]
OPTIONS = list(range(20))    # 20 “words” (0–19)
REWARDS = (-50, 100)         # (incorrect, correct)

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

def ask_analogy(model_name: str) -> str:
    rules = get_rules(REWARDS, OPTIONS)
    user = {
        "role": "user",
        "content": (
            "Here is a description of something.\n"
            "Does this setup remind you of any existing model or theory in social science?"
            "Please name it and briefly explain the analogy. \n"
            "Additionally, tell me if you know what optimal move is after a success. \n"
            "Finally, tell me how you think the game will converge globally. \n"
        )
    }

    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": rules},
            user
        ],
        temperature=0.5,
        max_tokens=1000
    )
    return resp.choices[0].message.content.strip()

# ————————————————————————————————
# 2) RUN REPEATEDLY & RECORD
# ————————————————————————————————
if __name__ == "__main__":
    records = []

    for model in MODELS:
        for run_idx in range(1, 11):
            print(f"[{model}] Run {run_idx}/10…", end=" ", flush=True)
            try:
                ans = ask_analogy(model)
                print("done")
            except Exception as e:
                ans = f"ERROR: {e}"
                print("failed")
            records.append({
                "model":    model,
                "run":      run_idx,
                "response": ans
            })

    # — Save to CSV
    df = pd.DataFrame(records)
    df.to_csv("analogy_responses_openai.csv", index=False)
    print("\nSaved all responses to analogy_responses_global_convergence.csv")
