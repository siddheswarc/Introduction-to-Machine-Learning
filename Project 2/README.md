# Predictive Models for Detection of crime

The goal of this project is to apply Machine Learning to solve the handwriting comparison task in forensics. We formulate this as a problem of Linear Regression where we map a set of input features to a real-valued scalar target.

The task is to find similarity between the handwritten samples of the known and the questioned writer by using Linear Regression.

Each instance in the CEDAR (Center of Excellence for Document Analysis) "AND" training data consists of set of input features for each handwritten "AND" sample. The features are obtained from two different sources:
1. Human Observed Features: Features entered by human document examiners manually.
2. GSC Features: Features extracted using Gradient Structural Concavity (GSC) algorithm.

The target values are scalars that can take two values {1:same writer, 0:different writers}.

We conclude Linear Regression to be an inappropriate algorithm to solve this problem, and hence formulate this as a problem of Logistic Regression.
