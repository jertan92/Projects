# Capstone Project: Predicting Customer Churn

## Executive Summary
First, we analyzed the dataset for missing values in which there were none. We conducted exploratory data analysis (EDA) on the variables in the dataset and conducted statistical significance tests. Based on the EDA and the results from the statistical significance tests conducted, we narrowed down to 11 variables that we will be using to build the classification model. The dataset with the 11 variables will be our filtered dataset.

For the filtered dataset, we converted the y response variable (attrition flag) and categorical data (gender) into binary labels for modeling. We conducted train-test split on our data, with a test size of 0.2, standardize our data and fit 5 models (K-nearest neighbor, Bagging classifier, Random forest, Ada boost classifier and Support vector model).

For the original dataset, we converted the y response variable (attrition flag) and categorical data (gender) into binary labels. We also converted the other categorical data (education level, marital status, income category and card category) into dummy variables for modeling. We conducted train-test split on our data, with a test size of 0.2, and fit 5 models (K-nearest neighbor, Bagging classifier, Random forest, Ada boost classifier and Support vector model).

For our classification model, the main purpose is to predict whether a customer will churn and therefore, we would be focusing on how well the model predicts a customer will churn when a customer really churns. This would refer to the sensitivity/recall of the model, which is to try to reduce the false negatives. However, to also ensure that we optimize our resources in terms of marketing efforts to retain an existing customer, we would also like to try to reduce the false positives. Therefore, we would like to minimize both false positives and false negatives. As both the F1 score and ROC AUC take both false positives and false negatives into account, we will focus more on these 2 metrics to evaluate the models. Therefore, the higher the F1 score and the ROC AUC, the better the model is for our classification problem. We conducted further analysis (feature importance) on our best model and conducted further evaluation. The final model was deployed on streamlit app.

## Background
Customer churn, also known as customer attrition rate, is when a customer chooses to stop using a particular products or services. Customer loyalty is something all brands strive for - understanding and preventing churn is critical to achieving this. Based on research ([Link](https://www.chargebee.com/blog/5-off-beat-lessons-saas-industry-reduce-customer-churn/)), we understand that every year companies lose $1.6 trillion in revenue due to customer churn. Every business deals with churn, and more often than not, it's much easier to keep an existing customer than it is to gain a new customer. It is also much easier to save a customer before they leave than it is to convince the customer to come back. Based on an analysis conducted ([Link](https://startuptalky.com/churn-rate-reducing-strategies/)), acquiring a new customer is 5 times the cost more than retaining an existing customer. Additionally, according to research done by Frederick Reichheld of Bain & Company ([Link](https://media.bain.com/Images/BB_Prescription_cutting_costs.pdf)), in financial services, a 5% increase in customer retention produces more than a 25% increase in profit. Therefore, it would be beneficial to understand and reduce the churn rate as this will impact the revenue of a business.

## Problem Statement
In this project, we will be focusing on churn for bank's credit card customers. We have been engaged by a bank to analyze data collected from existing customers to predict which customer is likely to churn so that the bank can focus their marketing efforts towards such customers and try to provide better services so as to be able to retain existing customers. 

## Dataset Introduction
The dataset was obtained from Kaggle. There were 20 columns and 10127 rows of data in the dataset.
([source](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers))

## SQL
We split the dataset into 3 smaller subsets of data, attrition, customer and card. Attrition consists data on the attrition status of the customers, customer consists mainly of the customer data and card consists mainly of the customer's card data. We conducted some SQL queries on the 3 datasets and join the 3 smaller datasets into a combined dataset using SQL for further processing and exploratory data analysis.

## Data Cleaning and Exploratory Data Analysis
We analyzed the dataset for missing values in which there were none. We split the variables into numerical variables and categorical variables. For the numerical variables, we did some exploratory data analysis in which we utilize histograms to visualize the distributions, boxplots to illustrate attrition by each feature and heatmap and coefficient scores to examine the correlation relationship between the numerical variables. We also conducted statistical significance tests (Anova test for normal distributions and Kolmogorov-Smirnov test for non-normal distributions) on the numerical variables. For the categorical variables, we did some exploratory data analysis in which we utilize countplots to represent the occurrence(counts) of the observation present. We also utilized barplots to illustrate attrition by each feature. We also conducted statistical significance test (chi-square test) on the categorical variables. Based on the EDA and the results from the statistical significance tests conducted, we narrowed down to 11 variables that we will be using to build the classification model. The dataset with the 11 variables will be our filtered dataset.

## Preprocessing and modeling
For the filtered dataset, we converted the y response variable (attrition flag) and categorical data (gender) into binary labels for modeling. We conducted train-test split on our data, with a test size of 0.2, standardize our data and fit 5 models (K-nearest neighbor, Bagging classifier, Random forest, Ada boost classifier and Support vector model).

For the original dataset, we converted the y response variable (attrition flag) and categorical data (gender) into binary labels. We also converted the other categorical data (education level, marital status, income category and card category) into dummy variables for modeling. We conducted train-test split on our data, with a test size of 0.2, and fit 5 models (K-nearest neighbor, Bagging classifier, Random forest, Ada boost classifier and Support vector model).

## Evaluation
For our classification model, the main purpose is to predict whether a customer will churn and therefore, we would be focusing on how well the model predicts a customer will churn when a customer really churns. This would refer to the sensitivity/recall of the model, which is to try to reduce the false negatives. However, to also ensure that we optimize our resources in terms of marketing efforts to retain an existing customer, we would also like to try to reduce the false positives. Therefore, the appropriate success metrics to validate our model would be the F1 score, which evaluates both the model's precision and recall ability. Additionally, as we understand from the documentation provided that the dataset is imbalanced, F1 score would also be appropriate as it provides robust results for imbalanced datasets. <br>

Receiver Operating Characteristic (ROC) Curve plots the True Positive Rate (**sensitivity**) (TPR) vs. False Positive Rate (**1 - specificity**) (FPR) for the range of possible decision thresholds (from 0 to 1). The baseline reference line is depicted to portray how good/bad our model is. If the FPR = TPR, then the model curve (blue line) will lie exactly on the orange line. We want a model that minimizes the FPs and FNs so we can maximize with more TPs and TNs. Thus, we need a higher TPR, at a lower FPR, that is, the blue line must be as far away from the orange line as possible. The area under ROC curve must be as big as possible (closer towards the value of 1). <br>

As both the F1 score and ROC AUC take both false positives and false negatives into account, it will be useful to focus more on these 2 metrics. Therefore, the higher the F1 score and the ROC AUC, the better the model is for our classification problem. Below were the accuracy scores (train and test), F1 score (train and test) and ROC AUC for the 5 models which we constructed on both the original dataset and filtered dataset. <br>

|Model|Dataset|Accuracy (train data set)|Accuracy (test data set)|F1 score (train data set)|F1 score (test data set)|ROC AUC|
|---|---|---|---|---|---|---|
|K-nearest neighbor|Original|0.897|0.876|0.554|0.419|0.856|
|K-nearest neighbor|Filtered|0.941|0.926|0.795|0.739|0.944|
|Bagging Classifier|Original|1.0|0.906|1.0|0.586|0.983|
|Bagging Classifier|Filtered|1.0|0.955|1.0|0.854|0.985|
|Random Forest|Original|0.999|0.956|0.997|0.847|0.985|
|Random Forest|Filtered|0.999|0.956|0.998|0.856|0.985|
|Ada Boost Classifier|Original|0.972|0.957|0.912|0.861|0.984|
|Ada Boost Classifier|Filtered|0.964|0.956|0.887|0.856|0.983|
|Support Vector Model|Original|0.981|0.915|0.938|0.709|0.940|
|Support Vector Model|Filtered|0.965|0.940|0.886|0.799|0.965|

We noted that all of the fitted models are performing much better than the baseline model (predicting majority class). Based on the table above with the accuracy scores (comparing train and test data set) and the F1 score (comparing train and test data set), we noted that all of the models appear to overfit. For the K-nearest neighbor, bagging classifier and random forest models, the filtered dataset performs better than the original dataset. For the support vector model, we noted that the accuracy and F1 score of the train data is higher for the original dataset as compared to the filtered dataset. However, the accuracy and F1 score for the test data fares better for the filtered dataset, which means that it is less of an overfit. For the ada boost classifier model, we noted that all the classification metric scores in the table fare slightly better for the original dataset as compared to the filtered dataset. However, we will select ada boost classifer model with filtered dataset as our final choice as it is slightly less overfit and still provides very close accuracy and F1 score on the test data as well as the ROC AUC value.

## Feature importance
Based on the selected model (Ada Boost Classifier on the filtered dataset), we obtain the top 10 features that are being utilized by the model in classifying on whether the customer will attrite. We noted that there are 3 features which seems to have a larger effect on the model that is being used to predict the target variable (i.e. attrition) as compared to the other features. These 3 features are total transaction amount (last 12 months), total trasaction count (last 12 months) and total amount change (Q4 over Q1). 

## Further Evaluation, Conclusion and Recommendations
|Number of features (Threshold)|Accuracy (train data set)|Accuracy (test data set)|F1 score (train data set)|F1 score (test data set)|ROC AUC|
|---|---|---|---|---|---|
|6 (>0.05)|0.957|0.942|0.861|0.808|0.980|
|7 (>0.04)|0.957|0.947|0.862|0.828|0.982|
|10 (>0.01)|0.964|0.955|0.886|0.855|0.983|
|11 (no threshold)|0.964|0.956|0.887|0.856|0.983|

Based on the thresholds which we have set above to further do feature selection to see if we are able to reduce the number of features without compromising too much of the evaluation metrics of the ada boost classification model, we noted that the model with the top 10 features have very similar evaluation metrics scores with the model, which we have earlier selected, that was fitted on the filtered dataset (containing 11 features). Therefore, our final selection of model will be the ada boost classifier with the top 10 features (based on feature importance). <br>

The top 10 features (based on feature importance) are:
1. Total Transaction Amount
2. Total Transaction Count
3. Total Amount Change (Q4 over Q1)
4. Total Revolving Balance
5. Total Count Change (Q4 over Q1)
6. Months inactive in 12 months
7. Contacts count in 12 months
8. Total Relationship count
9. Credit Limit 
10. Average Utilization Ratio <br>

Based on the correlation heat map with attrition flag, we noted that only 2 features (months inactive in 12 months and contacts count in 12 months) are positively correlated with attrition flag. This means that the higher the values for these 2 features, the higher the chances for the customer to attrite. For the remaining 8 features, they are negatively correlated with attrition flag. This means that the lower the values for these 8 features, the higher the chances for the customer to attrite. <br>

Banks can try to understand the needs of the customer so that they can recommend credit cards with specific purposes which the customer would want (i.e. cash back, air miles, etc.) so that the customer will use the card more often and thereby, having the possibility of increasing both the transaction amount and transaction count. <br>

Banks can also try to provide credit card offers and discounts to potential attriting customers to entice them to stay with the Bank. <br>

With our classification model, the bank will be able to predict which are the customers which are likely to attrite and the bank will be able to take action and reach out to those customers to discuss on what can be done so as to be able to retain those customers with the Bank moving forward.