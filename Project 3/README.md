# Image Classification with MNIST Dataset

The goal of this project is to implement machine learning methods to classify images in the MNIST dataset. We first implement an ensemble of four classifiers. Then the results of individual classifiers are to be combined to make a final decision.

We train the following four classifiers using MNIST digit images:
1. Logistic Regression
2. Multilayer perceptron Neural Network
3. Random Forest Package
4. SVM (Support Vector Machine) Package

Based on the above implementations, we wish to answer the following questions:
1. We test the MNIST trained models on two different test sets: the test set from MNIST and a test set from the USPS dataset. Do our results support the "No Free Lunch" theorem?
2. Observe the confusion matrix of each classifier and describe the relative strengths/weaknesses of each classifier. Which classifier has the overall best performance?
3. Combine the results of the individual classifiers using a classifier combination method such as majority voting. Is the overall combined performance better than that of any individual classifier?
