# Model Card
ML model to predict salary.

(For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf)

## Model Details
- Developer: Nontawat Pattanajak
- Model date: 20 August 2022
- Model version: 1
- Model type: Machine Learning model
- Model algorithm: Random Forest with 100 estimators

## Intended Use
- To simulate the use of CI/CD for ML model deployment

## Training Data
- Dataset: Census Income Data Set 
- Data source: https://archive.ics.uci.edu/ml/datasets/census+income
- Date of generated data: 20 August 2022
- Data splitting for training: 80%

## Evaluation Data
- Dataset: Census Income Data Set 
- Data source: https://archive.ics.uci.edu/ml/datasets/census+income
- Date of generated data: 20 August 2022
- Data splitting for training: 20%

## Metrics
- Performance metrics: f1-score, precision, and recall
- Model Performance (test data):
  - Accuracy: 0.97
  - f1-score: 0.96 (macro), 0.97 (weighted)
  - precision: 0.96 (macro), 0.97 (weighted)
  - recall: 0.95 (macro), 0.97 (weighted)

## Ethical Considerations
Analysing and publishing report of salary prediction by race, sex, native-country, and marital-status could be offensive. This should be reviewed and agreed with senior management level.

## Caveats and Recommendations
The dataset seems imbalanced - data with label '>=50', '<50K' are not the same/similar ratio. This may bias the model to make prediction by the label with higher data samples. This problem could be addressed by finding more data for label with low coverage OR downsize the data for the label with high coverage.