# Credit_Card_Analysis_Machine_Learning
# Overview 
## Purpose
Machine learning can be used to predict credit card risk and solve loan approval questions becasue credit risk is ultimatley an unbalanced classification problem. Previous data on loan approval can be used to train machine learning models to determine if someone if worthy of a loan, but good loans undeniably outnumber risky loans, and this class imbalance must be considered when choosing model for prediciting credit risk. The purpose of this project was to use the Python's Sciki-learn (**sklearn**) library to build and evaluate six differnt classification models that use differnt resampling and ensemble learning algorithms. The goal was to determine which of the six models are best at predicting credit risk based evidence from confusion matrices and classification reports. 
## Analysis Roadmap
The [credit card dataset](<insert relative link>) used in this project to train machine learning models came from LendingClub, a peer-to-peer lending services company. This dataset was imported to a DataFrame, cleaned, and encoded using the **Pandas** library. 

This project was broken down into following three main parts found in the Analysis Section:
1. Using Resampling Models to Predict Credit Risk
   - Use the oversampling **RandomOverSampler** algorithm to resample the dataset and train a logisitc classifier.
   - Use the oversampling **SMOTE** algorithm to resample the dataset and train a logisitc classifier.
   - Use the undersampling **ClusterCentroids** algorithm to resample the dataset and train a logisitc classifier.
2. Using the SMOTEENN algorithm to Predict Credit Risk
   - Use the combinatorial over- and under-sampling **SMOTEENN** algorithm to resample the dataset and train a logisitc classifier.
3. Using Ensemble Classifiers to Predict Credit Risk
   - Use the ensemble **BalancedRandomForestClassifier** algorithm to resample the dataset and train an ensemble classifier.
   - Use the ensemble **EasyEnsembleClassifier** algorithm to resample the dataset and train an ensemble classifier. 

The evaluation of each of the six models can be found in the Results section with images of the confusion matrix and classification report generated for each model provided for support.

Finally, the Summary section will provide a summary of the results of the machine learning models and discuss which model would best predict credit risk. 

# Analysis
## Using Resampling Models to Predict Credit Risk
Results of the analysis: Explain the purpose of this analysis.
# Results: Using bulleted lists
- Accuracy scores
Models 1-6 produced the following accuracy scores:
- Precision scores
Models 1-6 produced the following precision scores:
- Recall scores
Models 1-6 produced the following accuracy scores:

# Summary: Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. If you do not recommend any of the models, justify your reasoning.
