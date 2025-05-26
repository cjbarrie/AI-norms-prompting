# AI Norms Prompting Study

This repository contains code and data for a study examining how different AI models understand and respond to coordination game scenarios. The project investigates how various language models (OpenAI, Anthropic, and Ollama) interpret and analyze game-theoretic situations, particularly focusing on coordination games.

## Project Structure

The codebase is organized into several key components:

- `00_openai.py`: Script to query OpenAI models about coordination game scenarios
- `01_anthropic.py`: Script to query Anthropic models about coordination game scenarios
- `02_ollama.py`: Script to query Ollama models about coordination game scenarios
- `03_annotate.py`: Script to analyze and annotate model responses
- `04_plot.R`: R script for visualizing and analyzing the results

## Data Files

- `analogy_responses_openai.csv`: Raw responses from OpenAI models
- `analogy_responses_anthropic.csv`: Raw responses from Anthropic models
- `analogy_responses_ollama.csv`: Raw responses from Ollama models
- `analogy_responses_*_annotated.csv`: Annotated versions of the response files
- `annotation_summary_all_models.png`: Visualization of the analysis results

## Setup and Requirements

1. Set up your environment variables:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export OPENAI_ANNOTATION_MODEL="gpt-4.1-2025-04-14"  # or your preferred model
   ```

2. Install required Python packages:
   ```bash
   pip install openai pandas
   ```

3. Install required R packages (for visualization):
   ```R
   install.packages(c("ggplot2", "dplyr", "tidyr"))
   ```

## Usage

1. Run the data collection scripts in sequence:
   ```bash
   python 00_openai.py
   python 01_anthropic.py
   python 02_ollama.py
   ```

2. Annotate the collected responses:
   ```bash
   python 03_annotate.py
   ```

3. Generate visualizations:
   ```bash
   Rscript 04_plot.R
   ```