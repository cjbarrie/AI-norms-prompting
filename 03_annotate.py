#!/usr/bin/env python3
"""
annotate_all_analogy_responses.py

For each of:
  - analogy_responses_anthropic.csv
  - analogy_responses_openai.csv
  - analogy_responses_ollama.csv

Read <input>, classify every (model, run, response) via OpenAI functions API
into three binary flags (coordination, optimal_move, convergence), and write
out <input>_annotated.csv with columns:
  model, run, coordination, optimal_move, convergence

Partially completed outputs are picked up where they left off.
"""

import os
import sys
import csv
import json
import pandas as pd
import concurrent.futures
from openai import OpenAI

# ────────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────────────────────────
INPUT_FILES      = [
    "analogy_responses_anthropic.csv",
    "analogy_responses_openai.csv",
    "analogy_responses_ollama.csv",
]
ANNOTATION_MODEL = os.getenv("OPENAI_ANNOTATION_MODEL", "gpt-4o-2024-08-06")
FLUSH_EVERY      = 10
MAX_WORKERS      = 10

# pull API key from env
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    print("ERROR: Please set the OPENAI_API_KEY environment variable", file=sys.stderr)
    sys.exit(1)

client = OpenAI(api_key=API_KEY)

# Function‐calling spec
FUNCTION_NAME      = "score_analogy_response"
FUNCTION_DESC_TMPL = (
    "{response}\n\n"
    "Answer these three yes/no questions (as 1 or 0) about the above response, and return JSON:\n"
    "- coordination: did it identify this as a coordination game?\n"
    "- optimal_move: did it state the optimal move after a success?\n"
    "- convergence: did it predict how the game will converge?\n"
)
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "coordination":   {"type": "integer", "enum": [0,1]},
        "optimal_move":   {"type": "integer", "enum": [0,1]},
        "convergence":    {"type": "integer", "enum": [0,1]}
    },
    "required": ["coordination", "optimal_move", "convergence"]
}


def classify_content(response: str) -> dict:
    """Call OpenAI functions API to classify a single response."""
    function = {
        "name": FUNCTION_NAME,
        "description": FUNCTION_DESC_TMPL.format(response=response),
        "parameters": OUTPUT_SCHEMA,
    }
    try:
        resp = client.chat.completions.create(
            model=ANNOTATION_MODEL,
            messages=[{"role": "user", "content": function["description"]}],
            functions=[function],
            temperature=0,
            max_tokens=200
        )
        args = resp.choices[0].message.function_call.arguments
        return json.loads(args)
    except Exception as e:
        print(f"OpenAI error: {e}", file=sys.stderr)
        return {"coordination": 0, "optimal_move": 0, "convergence": 0}


def score_one(task):
    """Task = (model, run, response) → [model, run, c, o, v]."""
    model, run, response = task
    out = classify_content(response)
    return [
        model,
        run,
        int(out.get("coordination", 0)),
        int(out.get("optimal_move", 0)),
        int(out.get("convergence", 0))
    ]


def annotate_file(input_path: str):
    output_path = os.path.splitext(input_path)[0] + "_annotated.csv"
    if not os.path.exists(input_path):
        print(f"⚠️  Skipping missing file: {input_path}")
        return

    print(f"\n→ Annotating {input_path} → {output_path}")

    df = pd.read_csv(input_path, dtype={"model": str, "run": int, "response": str})

    # resume
    done = set()
    if os.path.exists(output_path):
        df_done = pd.read_csv(output_path, dtype={"model": str, "run": int})
        done = set(zip(df_done.model, df_done.run))

    fout = open(output_path, "a", newline="", encoding="utf8")
    writer = csv.writer(fout)
    if not done:
        writer.writerow(["model", "run", "coordination", "optimal_move", "convergence"])
        fout.flush()

    # build tasks
    tasks = [
        (row.model, row.run, row.response)
        for row in df.itertuples(index=False)
        if (row.model, row.run) not in done
    ]

    buffer = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for result in pool.map(score_one, tasks):
            buffer.append(result)
            m, r, c, o, v = result
            print(f"[{m}][run {r}] → coordination={c}, optimal_move={o}, convergence={v}")

            if len(buffer) >= FLUSH_EVERY:
                writer.writerows(buffer)
                fout.flush()
                buffer.clear()

    # final flush
    if buffer:
        writer.writerows(buffer)
        fout.flush()

    fout.close()
    print(f"✔ Done: {output_path}")


if __name__ == "__main__":
    for infile in INPUT_FILES:
        annotate_file(infile)
