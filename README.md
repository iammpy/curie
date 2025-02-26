
Example Code accompanying the paper

**CURIE: Evaluating LLMs On Multitask Scientific Long Context Understanding and Reasoning**

> **TL;DR:** we introduce CURIE (Scientific Long **C**ontext **U**nderstanding **R**easoning and **I**nformation **E**xtraction), benchmark with 10 tasks from 6 science domains specifically designed to test the ability of LLMs to assist scientists in realistic workflows.

## 📝 TODOs

- [ ] Add links to download the data.
- [ ] Update evals to include all metrics.
- [ ] Release responses by baselines to fully reproduce the reported numbers.
- [x] Example Colab to run inference.
- [x] Colab to run evaluation.
- [x] Colab to generate all plots and tables.

## Data
Our data is organized into eight domain-specific subfolders: "biogr", "dft", "pdb", "geo", "mpve", "qecc_65", "hfd", and "hfe".  Each subfolder contains two further subfolders: "ground_truth" and "inputs".  Within these, each data instance is stored in a JSON file named record_id.json, where record_id is a unique identifier. The "biogr" domain also includes image inputs as record_id.png files alongside the corresponding JSON.

Ground truth data varies in structure and content across domains, but all files consistently include a record_id field matching the filename.  Input files have a uniform structure across all domains, containing both a record_id field and a text field representing the input text to LLMs.

## Running Inference.
Example Colab notebook hat runs inference by iterating over all examples and prompts for all tasks is provided at code/curie_inference.ipynb.
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

## ✉️ Contact

This repository is created and maintained by [Subhashini](https://vsubhashini.github.io/). Questions and discussions are welcome under issues.

## 🙏 Acknowledgements

We are grateful to the many domain experts who have contributed to the creation
of the benchmark and evaluations.

## 📄 License

Code in this Github repository is licensed under a [APACHE 2.0 License](./LICENSE).

## 🎓 Citing SPIQA

Coming soon...

```
@article{cui2025curie,
  title={{CURIE}: Evaluating {LLM}s on Multitask Scientific Long Context Understanding and Reasoning },
  author={},
  journal={ICLR 2025},
  year={2025}
}
```
