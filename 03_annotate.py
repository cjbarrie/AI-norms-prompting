
#!/usr/bin/env python3
"""
annotate_all_analogy_responses.py

For each of:
  - analogy_responses_anthropic.csv
  - analogy_responses_openai.csv
  - analogy_responses_ollama.csv

Read <input>, classify every (model, run, response) via OpenAI functions API
into three binary flags (coordination, optimal_move, convergence), plus a
justification snippet for each, and write out <input>_annotated.csv with columns:
  model, run,
  coordination, coordination_justification, coordination_justification_annotated_manual,
  optimal_move, optimal_move_justification, optimal_move_justification_annotated_manual,
  convergence, convergence_justification, convergence_justification_annotated_manual

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
# switched to gpt-4.1-2025-04-14
ANNOTATION_MODEL = os.getenv("OPENAI_ANNOTATION_MODEL", "gpt-4.1-2025-04-14")
FLUSH_EVERY      = 10
MAX_WORKERS      = 10

# pull API key from env
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    print("ERROR: Please set the OPENAI_API_KEY environment variable", file=sys.stderr)
    sys.exit(1)

client = OpenAI(api_key=API_KEY)

# ────────────────────────────────────────────────────────────────────────────────
# Function‐calling spec: include justification fields
# ────────────────────────────────────────────────────────────────────────────────
FUNCTION_NAME      = "score_analogy_response"
FUNCTION_DESC_TMPL = (
    "{response}\n\n"
    "For each of the following, answer yes (1) or no (0), AND provide a short snippet "
    "from the response that justifies your answer. Return JSON with these keys:\n"
    "- coordination: did it identify this as a coordination game?\n"
    "- coordination_justification: a brief excerpt supporting that answer\n"
    "- optimal_move: did it state that the optimal move after a success is to keep answering the same way?\n"
    "- optimal_move_justification: a brief excerpt supporting that answer\n"
    "- convergence: did it predict that the game will converge to a unique global equilibrium?\n"
    "- convergence_justification: a brief excerpt supporting that answer\n"
)
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "coordination":   {"type": "integer", "enum": [0,1]},
        "coordination_justification": {"type": "string"},
        "optimal_move":   {"type": "integer", "enum": [0,1]},
        "optimal_move_justification": {"type": "string"},
        "convergence":    {"type": "integer", "enum": [0,1]},
        "convergence_justification": {"type": "string"},
    },
    "required": [
        "coordination",
        "coordination_justification",
        "optimal_move",
        "optimal_move_justification",
        "convergence",
        "convergence_justification",
    ]
}


def classify_content(response: str) -> dict:
    """Call OpenAI functions API to classify a single response, returning flags + justifications."""
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
            max_tokens=300
        )
        args = resp.choices[0].message.function_call.arguments
        return json.loads(args)
    except Exception as e:
        print(f"OpenAI error: {e}", file=sys.stderr)
        # on error, return zeros and empty justifications
        return {
            "coordination": 0,
            "coordination_justification": "",
            "optimal_move": 0,
            "optimal_move_justification": "",
            "convergence": 0,
            "convergence_justification": "",
        }


def score_one(task):
    """Task = (model, run, response) → expanded row including justifications + manual placeholders."""
    model, run, response = task
    out = classify_content(response)
    return [
        model,
        run,
        int(out.get("coordination", 0)),
        out.get("coordination_justification", ""),
        "",  # coordination_justification_annotated_manual
        int(out.get("optimal_move", 0)),
        out.get("optimal_move_justification", ""),
        "",  # optimal_move_justification_annotated_manual
        int(out.get("convergence", 0)),
        out.get("convergence_justification", ""),
        "",  # convergence_justification_annotated_manual
    ]


def annotate_file(input_path: str):
    output_path = os.path.splitext(input_path)[0] + "_annotated.csv"
    if not os.path.exists(input_path):
        print(f"⚠️  Skipping missing file: {input_path}")
        return

    print(f"\n→ Annotating {input_path} → {output_path}")

    df = pd.read_csv(input_path, dtype={"model": str, "run": int, "response": str})

    # resume if needed
    done = set()
    if os.path.exists(output_path):
        df_done = pd.read_csv(output_path, dtype={"model": str, "run": int})
        done = set(zip(df_done.model, df_done.run))

    fout = open(output_path, "a", newline="", encoding="utf8")
    writer = csv.writer(fout)
    if not done:
        writer.writerow([
            "model", "run",
            "coordination", "coordination_justification", "coordination_justification_annotated_manual",
            "optimal_move", "optimal_move_justification", "optimal_move_justification_annotated_manual",
            "convergence", "convergence_justification", "convergence_justification_annotated_manual"
        ])
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
            m, r, c, cj, cj_man, o, oj, oj_man, v, vj, vj_man = result
            print(
                f"[{m}][run {r}] → coord={c}({cj!r}) man={cj_man!r}, "
                f"opt={o}({oj!r}) man={oj_man!r}, "
                f"conv={v}({vj!r}) man={vj_man!r}"
            )

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
