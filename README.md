# ML Project: stroke prediction using supervised classifiers

## Dataset used:

An imbalanced medical dataset found here: (10.00 usability score)

[https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

## Abstract

Stroke ranks as the 2nd leading cause of death according to the Global Burden of Diseases (GDB). With the rise of modern technology, machine learning became a popular approach in medical predictions. However, imbalanced medical datasets present as an ongoing challenge, significantly decreasing the potential performance of machine learning models. This study aims to investigate, analyze, and compare the performance of supervised machine learning classifiers, including logistic regression, linear discriminant analysis, k-nearest neighbours, decision tree classifier, stochastic gradient descent, and ensemble classifiers, which consists of random forest classifier, adaptive boosting algorithm, and voting classifier. According to the analysis results, logistic regression and stochastic gradient descent work best as base classifiers for the imbalanced stroke dataset, while the adaptive boosting algorithm work best amongst the ensemble algorithms.

Keywords: stroke prediction, supervised machine learning, machine learning classifiers, imbalanced dataset, medical prediction, hyperparameter tuning

*Note: Please read the paper attached in PDF for more insight (file: stroke_pred.pdf)*

## Brief summary:
- Scatterplot visually identitifed outliers
- Label encoder for categorical data
- Simple imputer with median strategy for interpolation
- Supervised classifiers
- Ensemble classifiers (w/ supervised)
- grid search cross-validation
- accuracy, precision, recall, and f1 score metrics

## Setup

For this project I used Google colab. However, locally powered jupyter notebook should also work.

If using local jupyter notebook, these changes should be made:
```python
# Remove this line:
# from google.colab import drive

# Remove this line:
# drive.mount("/content/drive")

df = pd.read_csv("../folder/stroke.csv")
```

If hosted locally, anaconda is highly recommended for the optimization of dependencies.

Dependencies:
- scitkit-learn (sklearn)
- seaborn
- (pandas,numpy,matplotlib) <- these are always installed for ML
