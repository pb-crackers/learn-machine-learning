# Lab 5.2: Model Building & Training

## Objective

Build, train, and evaluate multiple models on the data you prepared in Lab 5.1. Start with
classical ML baselines, then build a custom PyTorch neural network, run experiments, and
select the best model.

## Prerequisites

- Completed Lab 5.1 with saved data in `data/processed/`
- Saved preprocessing objects (`scaler.joblib`, `num_imputer.joblib`)

## Getting Started

1. Open `starter.py` and follow the numbered steps.
2. Create a `models/` directory to save checkpoints.
3. Create a `results/` directory to save comparison tables.
4. Make sure your `plots/` directory exists for learning curves.

## What to Deliver

By the end of this lab you should have:

- Logistic/Linear Regression baseline with recorded metrics
- Random Forest baseline with recorded metrics
- A custom PyTorch `TabularNet` model trained with early stopping
- A learning curve plot saved to `plots/learning_curves.png`
- At least 3 experiment results logged (changing one variable at a time)
- A model comparison CSV saved to `results/model_comparison.csv`
- Final test set evaluation (run exactly once)
- Best model checkpoint saved to `models/final_model.pt`

## Tips

- Always record your baseline numbers before building the neural network.
- If your neural network is worse than Random Forest on tabular data, that is normal and valid.
  Document the finding.
- Watch for overfitting in the learning curves: if train loss keeps dropping but val loss
  goes up, add more regularization (dropout, weight decay) or reduce model size.
- Change one hyperparameter at a time during experiments so you know what caused the change.
- Do not touch the test set until you have finished all experiments and chosen your best model.
