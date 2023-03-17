# Modeling Helpers

Class to build, evaluate and fine-tune models for a given dataset and model object with scorer.

The objective is to find the best performing model for a dataset by hyperparameter tuning and feature selection.

The steps:

1. Build a baseline model with default parameters
2. Tune hyper-parameters of baseline model
3. Perform Feature Selection on Baseline Model
4. Tune Model
5. Perform Feature Selection on Tuned Model
6. Tune Model
7. The Best model will be the one with the best score among the six models built above