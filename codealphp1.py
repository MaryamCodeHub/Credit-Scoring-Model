import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

file_path = r'C:\Users\admin\Downloads\code alpha projects\project 01\train.xlsx'
df = pd.read_excel(file_path, sheet_name='train')

print("Data Sample:")
print(df.head())
print("\nMissing Values:")
print(df.isnull().sum())
numeric_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(include=[object]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

for column in categorical_cols:
    df[column] = df[column].astype(str)

label_encoder = LabelEncoder()
for column in categorical_cols:
    df[column] = label_encoder.fit_transform(df[column])

X = df.drop('Credit_Score', axis=1)
y = df['Credit_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nModel Evaluation Results:")
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(report)

# Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlOrRd',  # Warm tones color palette
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_,
            linewidths=1, linecolor='black')
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.title('Confusion Matrix', fontsize=15, weight='bold')
plt.show()

# Feature Importances
feature_importances = model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df, hue='Feature', palette='magma')  # Use 'magma' for more vibrant colors
plt.title('Feature Importances', fontsize=15, weight='bold')
plt.xticks(color='black')
plt.yticks(color='black')
plt.show()

# Distribution of Credit Score
plt.figure(figsize=(10, 6))
sns.countplot(x=y, hue=y, palette='Spectral')  # 'Spectral' for a satisfying gradient
plt.title('Distribution of Credit Score', fontsize=15, weight='bold')
plt.xlabel('Credit Score', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.show()

# Scatter plot for Monthly Inhand Salary vs. Outstanding Debt
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x="Monthly_Inhand_Salary", y="Outstanding_Debt", color="red")
plt.title('Monthly Inhand Salary vs. Outstanding Debt', fontsize=15, weight='bold')
plt.xlabel('Monthly Inhand Salary', fontsize=12)
plt.ylabel('Outstanding Debt', fontsize=12)
plt.show()

# Histogram with KDE for Amount Invested Monthly
plt.figure(figsize=(7, 5))
sns.histplot(data=df, x="Amount_invested_monthly", kde=True, bins=30, color="purple")
plt.title('Amount Invested Monthly Distribution', fontsize=15, weight='bold')
plt.xlabel('Amount Invested Monthly', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()
