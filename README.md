# Credit-Scoring-Model
Credit Scoring Model: Predicting Creditworthiness Using Machine Learning

Project Overview:
The Credit Scoring Model project involves building a machine learning model to predict the creditworthiness of individuals. The model evaluates various financial and personal attributes of individuals to determine whether they are likely to repay a loan or default. This kind of model is critical for financial institutions as it aids in automating the decision-making process for loan approvals, reducing risks, and optimizing the loan process.

In this project, I utilized classification algorithms to build the model, evaluated its performance using several metrics, and visualized the results to gain insights into the factors that influence creditworthiness.

Objective:
The primary objective of this project is to predict whether a given individual is creditworthy (i.e., likely to repay a loan) based on their financial data and demographic information. The model outputs a binary classification: creditworthy or non-creditworthy.

Dataset:
The dataset used in this project contains financial and personal data of individuals. Each data point includes features such as:
Monthly Inhand Salary
Outstanding Debt
Amount Invested Monthly
Loan Amount
Number of Loans Taken
Credit History
Age
Gender
Marital Status
Occupation
Country
The target variable in the dataset indicates whether the individual is creditworthy (represented as 1) or non-creditworthy (represented as 0).

Data Preprocessing:
Before building the model, several preprocessing steps were performed to clean and prepare the data:

Handling Missing Values: Missing values were imputed using suitable techniques like mean or median for numerical columns and mode for categorical columns.
Encoding Categorical Variables: Categorical features such as gender, marital status, and occupation were encoded using label encoding or one-hot encoding to transform them into a format suitable for machine learning algorithms.
Feature Scaling: Since the dataset contained features with varying ranges (e.g., salary vs. debt), feature scaling was applied using standardization techniques to ensure that all features contributed equally to the model’s performance.
Data Splitting: The dataset was split into training and testing sets in an 80-20 ratio. The training set was used to train the model, while the testing set was used to evaluate the model’s performance.
Modeling
To build the Credit Scoring Model, I explored various classification algorithms to find the best performing model:

Logistic Regression: As a simple and interpretable model, Logistic Regression was used as the baseline to understand how well basic linear models perform.
Decision Tree Classifier: Decision trees were employed to capture non-linear relationships between the features and the target.
Random Forest Classifier: An ensemble model, Random Forest, was used to improve performance by aggregating the results of multiple decision trees.
Gradient Boosting Classifier: A boosting technique was used to further improve accuracy by focusing on correcting the mistakes made by previous models.
Support Vector Machine (SVM): SVM was tested to separate the data points into different classes based on the margin maximization principle.
Model Evaluation
Several evaluation metrics were used to assess the performance of the models:

Accuracy: The percentage of correct predictions made by the model.
Precision: The percentage of true positive predictions among all positive predictions made.
Recall: The percentage of true positive predictions out of all actual positive cases.
F1 Score: The harmonic mean of precision and recall, providing a balance between the two.
Confusion Matrix: A heatmap was used to visualize true positives, true negatives, false positives, and false negatives.
Visualizations
Various visualizations were created to gain insights into the data and model performance:

Feature Importance: A bar plot showing which features contributed most to the prediction of creditworthiness.
Scatter Plot: A scatter plot visualizing the relationship between Monthly Inhand Salary and Outstanding Debt.
Histogram with KDE: A histogram with Kernel Density Estimation (KDE) showing the distribution of the feature Amount Invested Monthly.
Confusion Matrix Heatmap: A heatmap visualizing the model’s performance in terms of true positives, true negatives, and misclassifications.
Results and Findings
The Random Forest Classifier and Gradient Boosting Classifier yielded the highest performance in terms of accuracy and F1 score, outperforming the baseline logistic regression model. The following were key findings:

Outstanding Debt and Monthly Inhand Salary were the most important features in predicting whether an individual is creditworthy.
Individuals with higher Amount Invested Monthly had a greater likelihood of being classified as creditworthy.
The model achieved an accuracy of X% on the test dataset (you can update this based on your results).
Conclusion
The Credit Scoring Model provides a reliable way to predict creditworthiness, helping financial institutions make better decisions regarding loan approvals. By using machine learning algorithms, we can automate and improve the accuracy of credit scoring systems, reducing risks associated with lending.

This project has been an excellent learning opportunity, allowing me to apply machine learning techniques to a real-world financial problem. With further tuning and optimization, the model could be deployed in production environments to assist in credit scoring.
