
Data accompanying the paper

**CURIE: Evaluating LLMs On Multitask Scientific Long Context Understanding and Reasoning**

## 📁 Repository Structure

The contents of this repository are structured as follows:

```bash
data
    ├── task
        ├── inputs
        |   └── record_id.json
        └── ground_truth
            └── record_id.json
    └── difficulty_levels.json

```
The data folder consists of a single forlder for each task. Under each task we
have the `inputs` and `ground_truth` directories that contain the data for each
single data point.

The `difficulty_levels.json` contains the difficulty level (`easy`, `medium`,
`hard`) values for each record for each task.


## 📁 Running notebooks on the data

To run the accompanying notebooks on the dataset, we recommend unzipping the
data folder and uploading it to Google Drive and adding the path to the Folder
in the notebooks.

