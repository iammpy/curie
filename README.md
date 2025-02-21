
Example Code accompanying the paper

**CURIE: Evaluating LLMs On Multitask Scientific Long Context Understanding and Reasoning**

> **TL;DR:** we introduce CURIE (Scientific Long **C**ontext **U**nderstanding **R**easoning and **I**nformation **E**xtraction), benchmark with 10 tasks from 6 science domains specifically designed to test the ability of LLMs to assist scientists in realistic workflows.

## Data
Our data is organized into eight domain-specific subfolders: "biogr", "dft", "pdb", "geo", "mpve", "qecc_65", "hfd", and "hfe".  Each subfolder contains two further subfolders: "ground_truth" and "inputs".  Within these, each data instance is stored in a JSON file named record_id.json, where record_id is a unique identifier. The "biogr" domain also includes image inputs as record_id.png files alongside the corresponding JSON.

Ground truth data varies in structure and content across domains, but all files consistently include a record_id field matching the filename.  Input files have a uniform structure across all domains, containing both a record_id field and a text field representing the input text to LLMs.

## Running Inference.
Our inference Colab notebook is provided at code/curie_inference.ipynb.
To execute it:
Add your API key for the model.
Connect to the default runtime ("Python 3 Google Compute Engine backend").
In the "params" cell, configure the following:
root_path: Path to the data folder.


## Running eval.
Our evaluation Colab notebook is provided at code/curie_run_eval.ipynb. To execute it:
Connect to the default runtime ("Python 3 Google Compute Engine backend").
In the "params" cell, configure the following:
root_path: Path to the data folder.
domain: The target domain (e.g., "biogr", "dft").
llm: The Large Language Model to evaluate.
prompt: The prompt used for the LLM.
record_id: The ID of the record to evaluate.
Run the Colab.  Evaluation metrics will be printed at the end of the notebook.

Note: Evaluating the "dft" and "mpve" tasks using the LLMSim score requires querying LLMs and therefore requires setting up a Google API key.


## Generating tables and plots.

To generate the tables and plots in the paper use the notebook code/curie_generate_tables_figures.ipynb

