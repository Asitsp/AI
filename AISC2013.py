#!/usr/bin/env python
# coding: utf-8

# # Import Important Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score


# # Load the Data

# In[2]:


df = pd.read_csv("diabetes_prediction_dataset.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# # Descriptive Statistics

# In[5]:


df.describe().transpose()


# In[6]:


df.shape


# # Check for Missing Values

# In[7]:


#Check for null values in the dataset
df.isnull().sum()


# # Check the Number of Unique Values in Integer Columns

# In[8]:


#Checking the number of unique values
df.select_dtypes(include='int64').nunique()


# In[9]:


#check duplicate values
df.duplicated().sum()


# In[10]:


#drop the duplicated values
df = df.drop_duplicates()


# In[11]:


df.shape


# In[12]:


column_names = df.columns.tolist()
print("Column Names:")
print(column_names)


# # Explore Numeric Variables with Histograms

# In[13]:


# Histogram for numeric columns
numeric_columns = df.select_dtypes(include=['int64'])
numeric_columns.hist(bins=10, figsize=(12, 8))
plt.show()


# # Box Plots for Numeric Variables

# In[14]:


numeric_columns = df.select_dtypes(include=['int64'])
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

for i, col in enumerate(numeric_columns.columns):
    sns.boxplot(x=col, data=df, ax=axes[i//2, i%2])
    axes[i//2, i%2].set_title(f'Box Plot of {col}')

plt.tight_layout()
plt.show()


# In[15]:


# Combined side-by-side count plot for categorical variables
categorical_columns = ['blood_glucose_level', 'smoking_history']
fig, axes = plt.subplots(nrows=1, ncols=len(categorical_columns), figsize=(12, 4))

for i, col in enumerate(categorical_columns):
    sns.countplot(x=col, data=df, ax=axes[i], palette='pastel')
    axes[i].set_title(f'Count Plot of {col}')

plt.tight_layout()
plt.show()


# # Explore Categorical Variables with Count Plots

# In[16]:


# Stacked Area Chart for diabetes
crosstab = pd.crosstab(df['age'], df['diabetes'])
crosstab.plot(kind='area', colormap='viridis', alpha=0.7, stacked=True)
plt.title('Stacked Area Chart: Age Category by General Health')
plt.xlabel('Age Category')
plt.ylabel('Count')
plt.show()


# # Pair Plot for Numeric Variables

# In[17]:


sns.pairplot(df[numeric_columns.columns])
plt.show()


# # Distribution of the Target Variable

# In[18]:


sns.countplot(x='diabetes', data=df, palette='pastel')
plt.title('Distribution of Diabetes (Target Variable)')
plt.show()


# 
# # Correlation Heatmap

# In[19]:


# Create a copy of the DataFrame to avoid modifying the original
df_encoded = df.copy()

# Create a label encoder object
label_encoder = LabelEncoder()

# Iterate through each object column and encode its values
for column in df_encoded.select_dtypes(include='object'):
    df_encoded[column] = label_encoder.fit_transform(df_encoded[column])


# In[20]:


target_correlation = df_encoded.corr()['diabetes'].sort_values(ascending=False)
print(target_correlation)


# In[21]:


plt.figure(figsize=(20, 16))
sns.heatmap(df_encoded.corr(), fmt='.2g', annot=True)
plt.title('Pairwise Correlation Heatmap')
plt.show()


# 
# # Data Preprocessing

# In[22]:


# Encode categorical variables
label_encoder = LabelEncoder()
df['gender'] = label_encoder.fit_transform(df['gender'])
df['smoking_history'] = label_encoder.fit_transform(df['smoking_history'])


# # Split the Data into Features (X) and Target Variable (y)

# In[23]:


# Split the data into features (X) and target variable (y)
X = df.drop('diabetes', axis=1)
y = df['diabetes']


# # Split the Data into Training and Testing Sets

# In[24]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[25]:


# Standardize numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# # Train a Logistic Regression Model

# In[26]:


# Train a logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)


# In[27]:


# Make predictions on the test set
y_pred = model.predict(X_test_scaled)


# # Model Evaluation

# In[28]:


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')


# 
# # Plot the Confusion Matrix

# In[29]:


# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# 
# #  Plot the ROC curve

# In[30]:


# Train a Logistic Regression Model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Make Predictions on the Test Set
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

# Calculate the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Calculate the AUC
roc_auc = roc_auc_score(y_test, y_pred_prob)

# Plot the ROC curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()


# In[ ]:




