# Usage Guidelines for Project Files

## project.py
1. `config`: Dictionary for all global configs.
    - `filename`: Location of dataset for this project.
    - `label`: String label for class column. Poisonous = 1, Edible = 0.
2. `results`: Location where results are stored, such as graphs, hypothesis test results, etc.

## data.py
`data.load(filename, label)`:
- `filename`: String with the relative position of the database (.csv).
- `label`: String label for the class columns.
- returns cleaned numerical data in form `(X, y)`

## kfoldcv.py
`kfoldcv.run(X, y, model, k, verbose=False)`:
- `(X, y)`: Dataset to be performed over.
- `model`: Model Class to be trained.
- `k`: Number of folds.
- `verbose`: False by default. If true, prints CV-output to stdout.
- returns accuracy for each fold

Note: see `kfoldcv_graph.py` for usage examples.
