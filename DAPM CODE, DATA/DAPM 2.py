#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import scipy as sp
from sklearn import *
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore


# In[2]:


df = pd.read_csv(r"C:\Users\abhis\Desktop\ObesityDataSet_raw_and_data_sinthetic.csv")
data = pd.read_csv(r"C:\Users\abhis\Desktop\ObesityDataSet_raw_and_data_sinthetic.csv")


# In[3]:


# Check for null values in the entire DataFrame
null_values = df.isnull().sum()

# Display the count of null values for each column
print("Null Values in Each Column:")
print(null_values)

# Check if there are any null values in the entire DataFrame
any_null_values = df.isnull().any().any()

if any_null_values:
    print("\nThere are null values in the DataFrame.")
else:
    print("\nThere are no null values in the DataFrame.")
    


# In[4]:


# Find and display duplicate rows based on all columns
duplicate_rows = df[df.duplicated()]
print("Duplicate Rows (based on all columns):")
print(duplicate_rows)

# Find and display duplicate rows based on selected columns
selected_columns = ['Gender', 'Age', 'Height', 'Weight', 'Family_hist', 'High_Cl', 'Veg',
       'Main_meals', 'Meals_btw', 'Smoking', 'Water_int', 'Physical_Act',
       'Alcohol', 'M_Trans', 'OB_Level']
duplicate_rows_all_columns = df[df.duplicated(subset=selected_columns)]
print("\nDuplicate Rows (based on selected columns):")
print(duplicate_rows_all_columns)
# Count and display the number of duplicate values in the entire dataset
num_duplicate_all_columns = len(duplicate_rows_all_columns)
print(f"\nNumber of duplicate values based on all columns: {num_duplicate_all_columns}")


# Remove duplicates and keep the first occurrence
df = df.drop_duplicates()


# In[5]:


# Box plots for selected features
selected_features = ['Age', 'Height', 'Weight']
for feature in selected_features:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df[feature])
    plt.title(f"Box Plot for {feature}")
    plt.show()


# In[6]:


# Display basic statistics for numerical features
numerical_stats = df[['Age', 'Height', 'Weight']].describe()
print(numerical_stats)


# In[7]:


column_names = df.columns
print(column_names)
# Print the names of numerical features
numerical_feature_names = df.columns.tolist()
print("Numerical Feature Names:")
print(numerical_feature_names)


# In[8]:


# Select relevant features for outlier detection
features_for_outliers = [ 'Height', 'Weight']


# Extract features for outlier detection
X_outliers = df[features_for_outliers]

# Calculate Z-scores for each feature
z_scores = zscore(X_outliers)

# Define a threshold for identifying outliers (e.g., Z-score greater than 3 or less than -3)
threshold = 3

# Identify and remove outliers
outliers_mask = (abs(z_scores) > threshold).any(axis=1)
df = df[~outliers_mask]

# Display the rows with outliers
print("Rows with Outliers:")
print(df[outliers_mask])

# Box plots for selected features
selected_features = ['Age', 'Height', 'Weight']
for feature in selected_features:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df[feature])
    plt.title(f"Box Plot for {feature}")
    plt.show()


# In[9]:


df.drop(df.index[df['Height'] == 1.98], inplace=True)          
# Box plots for selected features
selected_features = ['Age', 'Height', 'Weight']
for feature in selected_features:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df[feature])
    plt.title(f"Box Plot for {feature}")
    plt.show()


# In[10]:


# Display basic statistics for numerical features
numerical_stats = df[['Age', 'Height', 'Weight']].describe()
print(numerical_stats)


# In[11]:


# Display basic statistics for numerical features
numerical_stats = df.describe()
print(numerical_stats)

# Calculate skewness and kurtosis for each numerical feature
skewness = df.skew()
kurtosis = df.kurtosis()

# Display skewness and kurtosis
print("\nSkewness:")
print(skewness)
print("\nKurtosis:")
print(kurtosis)


# In[12]:


# Correlation heatmap
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
print("Correlation Matrix:")
print(correlation_matrix)


# In[13]:


# Visualize distribution of numerical features using histograms
df.hist(bins=20, figsize=(15, 10))
plt.suptitle("Histograms of Numerical Features", y=0.92)
plt.show()


# In[14]:


# Select categorical variables only
categorical_columns = df.select_dtypes(include=['object'])
categorical_column_names = categorical_columns.columns
# Print the categorical columns
print(categorical_column_names)


# In[15]:


from scipy.stats import chi2_contingency
categorical_features = ['Gender', 'Family_hist', 'High_Cl', 'Meals_btw', 'Smoking', 'Alcohol',
       'M_Trans']


# In[16]:


# Chi-square test for each categorical feature against the target variable
target_variable = 'OB_Level'

for feature in categorical_features:
    contingency_table = pd.crosstab(df[feature], df[target_variable])
    chi2, p, _, _ = chi2_contingency(contingency_table)
    
    print(f"Chi-square test for {feature} vs {target_variable}:")
    print(f"Chi2 Value: {chi2}")
    print(f"P-Value: {p}")
    
    if p < 0.05:
        print(f"The feature {feature} is statistically significant for {target_variable}.\n")
    else:
        print(f"The feature {feature} is not statistically significant for {target_variable}.\n")


# In[17]:


from sklearn.preprocessing import LabelEncoder

# Select the column(s) you want to label encode
columns_to_encode = ['Gender', 'Family_hist', 'High_Cl', 'Meals_btw', 'Smoking', 'Alcohol',
       'M_Trans', 'OB_Level']

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Label encode each selected column
for column in columns_to_encode:
    df[column] = label_encoder.fit_transform(df[column])

# Display the updated DataFrame
print(df)

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Label encode each selected column
for column in columns_to_encode:
    df[column] = label_encoder.fit_transform(df[column])

# Display the updated DataFrame
print(df)

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Label encode each selected column
for column in columns_to_encode:
    df[column] = label_encoder.fit_transform(df[column])

# Display the updated DataFrame
print(df)


# In[18]:


from sklearn.metrics import jaccard_score

# Select the categorical variables for Jaccard similarity calculation
categorical_columns = ['Gender', 'Family_hist', 'High_Cl', 'Meals_btw', 'Smoking', 'Alcohol', 'M_Trans', 'OB_Level']

# Create a subset of the dataframe with only the selected categorical columns
df_subset = df[categorical_columns]

# Convert the categorical values to numerical labels
df_numerical = df_subset.apply(lambda x: pd.factorize(x)[0])

# Initialize an empty Jaccard similarity score matrix
num_columns = len(categorical_columns)
jaccard_similarity_matrix = pd.DataFrame(index=categorical_columns, columns=categorical_columns)

# Calculate Jaccard similarity score matrix
for i in range(num_columns):
    for j in range(i, num_columns):
        col1, col2 = categorical_columns[i], categorical_columns[j]
        jaccard_similarity = jaccard_score(df_numerical[col1], df_numerical[col2], average='weighted')
        jaccard_similarity_matrix.loc[col1, col2] = jaccard_similarity
        jaccard_similarity_matrix.loc[col2, col1] = jaccard_similarity

# Display the Jaccard similarity score matrix
print("Jaccard Similarity Score Matrix:")
print(jaccard_similarity_matrix)
# Heatmap to represent the Jaccard's Similarity Scores
# Set the style of the visualization
sns.set(style="whitegrid")
# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(jaccard_similarity_matrix.astype(float), annot=True, cmap="RdYlGn", linewidths=.5)
# Add labels and title
plt.title("Jaccard Similarity Score Heatmap")
plt.xlabel("Categorical Variables")
plt.ylabel("Categorical Variables")
# Show the plot
plt.show()


# In[19]:


import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances

# Assuming 'Age', 'Height', and 'Weight' are the numerical features
numerical_features = ['Age', 'Height', 'Weight']

# Extract numerical features for proximity analysis
X_proximity_numerical = df[numerical_features]

# Transpose the DataFrame to get column-to-column distances
X_proximity_numerical_transposed = X_proximity_numerical.T

# Calculate pairwise distances between columns using Euclidean and Manhattan metrics
euclidean_distances_matrix = pd.DataFrame(euclidean_distances(X_proximity_numerical_transposed.values), columns=numerical_features, index=numerical_features)
manhattan_distances_matrix = pd.DataFrame(manhattan_distances(X_proximity_numerical_transposed.values), columns=numerical_features, index=numerical_features)

# Display the distance matrices
print("Euclidean Distance Matrix:")
print(euclidean_distances_matrix)

print("\nManhattan Distance Matrix:")
print(manhattan_distances_matrix)

# Heatmap to represent euclidean_distances_matrix
# Set the style of the visualization
sns.set(style="whitegrid")
# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(euclidean_distances_matrix.astype(float), annot=True, cmap="RdYlGn", linewidths=.5)
# Add labels and title
plt.title("Euclidean Distance Score Heatmap")
plt.xlabel("Categorical Variables")
plt.ylabel("Categorical Variables")
# Show the plot
plt.show()

# Heatmap to represent the Jaccard's Similarity Scores
# Set the style of the visualization
sns.set(style="whitegrid")
# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(manhattan_distances_matrix.astype(float), annot=True, cmap="RdYlGn", linewidths=.5)
# Add labels and title
plt.title("Manhattan Distance Score")
plt.xlabel("Categorical Variables")
plt.ylabel("Categorical Variables")
# Show the plot
plt.show()


# In[20]:


#information gain for feature selection
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

# Select features (both categorical and numerical)
selected_features = ['Gender', 'Age', 'Height', 'Weight', 'Family_hist', 'High_Cl', 'Veg',
       'Main_meals', 'Meals_btw', 'Smoking', 'Water_int', 'Physical_Act',
       'Alcohol', 'M_Trans']

# Calculate Information Gain for each feature
information_gain = mutual_info_classif(df[selected_features], df['OB_Level'], discrete_features='auto')

# Create a DataFrame to display the results
information_gain_df = pd.DataFrame({'Feature': selected_features, 'Information Gain': information_gain})
information_gain_df = information_gain_df.sort_values(by='Information Gain', ascending=False)

# Display the results
print(information_gain_df)


# In[21]:


import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Assuming df is your dataframe with the features
features = df.drop('OB_Level', axis=1)  # Drop the target variable

# Standardize the data (important for PCA)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply PCA
pca = PCA()
principal_components = pca.fit_transform(scaled_features)

# Create a DataFrame for the principal components
columns = [f'PC{i+1}' for i in range(principal_components.shape[1])]
principal_df = pd.DataFrame(data=principal_components, columns=columns)

# Add the target variable back to the DataFrame
principal_df['OB_Level'] = df['OB_Level']

# Check the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = explained_variance_ratio.cumsum()



# Print the explained variance ratio and cumulative explained variance
print("\nExplained Variance Ratio:")
print(explained_variance_ratio)

print("\nCumulative Explained Variance:")
print(cumulative_explained_variance)

# Plot the explained variance
plt.plot(cumulative_explained_variance, marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Principal Components')
plt.show()



# In[22]:


# Log transformation
df['Log_Age'] = np.log1p(df['Age'])
df['Log_Weight'] = np.log1p(df['Weight'])
df['Log_Height'] = np.log1p(df['Height'])

# Plot distributions before and after log transformation
plt.figure(figsize=(18, 5))

# Before transformation - Age
plt.subplot(1, 3, 1)
sns.histplot(df['Age'], kde=True)
plt.title('Distribution of Age (Before Log Transformation)')

# Before transformation - Weight
plt.subplot(1, 3, 2)
sns.histplot(df['Weight'], kde=True)
plt.title('Distribution of Weight (Before Log Transformation)')

# Before transformation - Height
plt.subplot(1, 3, 3)
sns.histplot(df['Height'], kde=True)
plt.title('Distribution of Height (Before Log Transformation)')

plt.show()

# Plot distributions after log transformation
plt.figure(figsize=(18, 5))

# After transformation - Log_Age
plt.subplot(1, 3, 1)
sns.histplot(df['Log_Age'], kde=True)
plt.title('Distribution of Log_Age (After Log Transformation)')

# After transformation - Log_Weight
plt.subplot(1, 3, 2)
sns.histplot(df['Log_Weight'], kde=True)
plt.title('Distribution of Log_Weight (After Log Transformation)')

# After transformation - Log_Height
plt.subplot(1, 3, 3)
sns.histplot(df['Log_Height'], kde=True)
plt.title('Distribution of Log_Height (After Log Transformation)')

plt.show()


# In[23]:


from scipy.stats import boxcox
# Box-Cox transformation
df['BoxCox_Age'], _ = boxcox(df['Age'] + 1)  # Adding 1 to handle zero and negative values
df['BoxCox_Weight'], _ = boxcox(df['Weight'] + 1)
df['BoxCox_Height'], _ = boxcox(df['Height'] + 1)

# Plot distributions before and after Box-Cox transformation
plt.figure(figsize=(18, 5))

# Before transformation - Age
plt.subplot(1, 3, 1)
sns.histplot(df['Age'], kde=True)
plt.title('Distribution of Age (Before Box-Cox Transformation)')

# Before transformation - Weight
plt.subplot(1, 3, 2)
sns.histplot(df['Weight'], kde=True)
plt.title('Distribution of Weight (Before Box-Cox Transformation)')

# Before transformation - Height
plt.subplot(1, 3, 3)
sns.histplot(df['Height'], kde=True)
plt.title('Distribution of Height (Before Box-Cox Transformation)')

plt.show()

# Plot distributions after Box-Cox transformation
plt.figure(figsize=(18, 5))

# After transformation - BoxCox_Age
plt.subplot(1, 3, 1)
sns.histplot(df['BoxCox_Age'], kde=True)
plt.title('Distribution of BoxCox_Age (After Box-Cox Transformation)')

# After transformation - BoxCox_Weight
plt.subplot(1, 3, 2)
sns.histplot(df['BoxCox_Weight'], kde=True)
plt.title('Distribution of BoxCox_Weight (After Box-Cox Transformation)')

# After transformation - BoxCox_Height
plt.subplot(1, 3, 3)
sns.histplot(df['BoxCox_Height'], kde=True)
plt.title('Distribution of BoxCox_Height (After Box-Cox Transformation)')

plt.show()


# In[24]:


# Dropping 2 columns
df.drop(['Smoking', 'M_Trans'], axis=1, inplace=True)


# In[25]:


pip install mlxtend


# In[32]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Assuming 'Obesity Level' is your target variable
label_encoder = LabelEncoder()
df['OB_Level'] = label_encoder.fit_transform(df['OB_Level'])

# Select relevant features for classification
features_for_classification = ['Gender', 'Age', 'Height', 'Weight', 'Family_hist', 'High_Cl', 'Veg',
       'Main_meals', 'Meals_btw', 'Water_int', 'Physical_Act',
       'Alcohol']
X_classification = df[features_for_classification]
y_classification = df['OB_Level']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_classification, y_classification, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize classifiers
classifiers = [
    KNeighborsClassifier(n_neighbors=5),
    SVC(kernel='linear', C=1),
    LogisticRegression(),
    GaussianNB(),
    RandomForestClassifier(n_estimators=100, random_state=42),
    DecisionTreeClassifier(random_state=42),
    AdaBoostClassifier(n_estimators=50, random_state=42),
    GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42)
]

# Create a table for results
results_table = pd.DataFrame(columns=['Classifier', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'Recall', 'F1-score'])

# Iterate through classifiers
for classifier in classifiers:
    classifier_name = classifier.__class__.__name__

    # Fit and predict
    classifier.fit(X_train_scaled, y_train)
    y_pred = classifier.predict(X_test_scaled)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    sensitivity = confusion[0, 0] / (confusion[0, 0] + confusion[0, 1])
    specificity = confusion[1, 1] / (confusion[1, 0] + confusion[1, 1])
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Append to results table
    results_table = results_table.append({
        'Classifier': classifier_name,
        'Accuracy': accuracy,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1
    }, ignore_index=True)

# Print the results table
print(results_table)


# In[33]:





# In[34]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.decomposition import PCA

# Assuming 'Obesity Level' is your target variable
label_encoder = LabelEncoder()
df['OB_Level'] = label_encoder.fit_transform(df['OB_Level'])

# Select relevant features for classification
features_for_classification = ['Gender', 'Age', 'Height', 'Weight', 'Family_hist', 'High_Cl', 'Veg',
       'Main_meals', 'Meals_btw', 'Water_int', 'Physical_Act',
       'Alcohol']
X_classification = df[features_for_classification]
y_classification = df['OB_Level']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_classification, y_classification, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA with variance threshold of 0.85
pca = PCA(0.85)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Initialize classifiers
classifiers = [
    KNeighborsClassifier(n_neighbors=5),
    SVC(kernel='linear', C=1),
    LogisticRegression(),
    GaussianNB(),
    RandomForestClassifier(n_estimators=100, random_state=42),
    DecisionTreeClassifier(random_state=42),
    AdaBoostClassifier(n_estimators=50, random_state=42),
    GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42)
]

# Create a table for results
results_table = pd.DataFrame(columns=['Classifier', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'Recall', 'F1-score'])

# Iterate through classifiers
for classifier in classifiers:
    classifier_name = classifier.__class__.__name__

    # Fit and predict using PCA-transformed data
    classifier.fit(X_train_pca, y_train)
    y_pred = classifier.predict(X_test_pca)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    sensitivity = confusion[0, 0] / (confusion[0, 0] + confusion[0, 1])
    specificity = confusion[1, 1] / (confusion[1, 0] + confusion[1, 1])
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Append to results table
    results_table = results_table.append({
        'Classifier': classifier_name,
        'Accuracy': accuracy,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1
    }, ignore_index=True)

# Print the results table
print(results_table)


# In[38]:


from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Assuming 'Obesity Level' is your target variable
label_encoder = LabelEncoder()
df['OB_Level'] = label_encoder.fit_transform(df['OB_Level'])

# Select relevant features for classification
features_for_classification = ['Gender', 'Age', 'Height', 'Weight', 'Family_hist', 'High_Cl', 'Veg',
       'Main_meals', 'Meals_btw', 'Water_int', 'Physical_Act',
       'Alcohol']

# Extract features for classification
X_classification = df[features_for_classification]
y_classification = df['OB_Level']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_classification, y_classification, test_size=0.2, random_state=42)

# List of classifiers
classifiers = [
    KNeighborsClassifier(n_neighbors=5),
    SVC(kernel='linear', C=1),
    LogisticRegression(),
    GaussianNB(),
    RandomForestClassifier(n_estimators=100, random_state=42),
    DecisionTreeClassifier(random_state=42),
    AdaBoostClassifier(n_estimators=50, random_state=42),
    GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42)
]
# Initialize a BaggingClassifier with RandomForestClassifier as base estimator
bagging_clf = BaggingClassifier(base_estimator=RandomForestClassifier(), random_state=42)
classifiers.append(bagging_clf)


# Calculate metrics for each classifier
results = []
for classifier in classifiers:
    clf_name = classifier.__class__.__name__
    
    # Train the classifier
    classifier.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = classifier.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    sensitivity = confusion[0, 0] / (confusion[0, 0] + confusion[0, 1])
    specificity = confusion[1, 1] / (confusion[1, 0] + confusion[1, 1])
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Append to results table
    results_table = results_table.append({
        'Classifier': classifier_name,
        'Accuracy': accuracy,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1
    }, ignore_index=True)

# Print the results table
print(results_table)


# In[44]:


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Assuming 'Obesity Level' is your target variable
label_encoder = LabelEncoder()
df['OB_Level'] = label_encoder.fit_transform(df['OB_Level'])

# Select relevant features for classification
features_for_classification = ['Gender', 'Age', 'Height', 'Weight', 'Family_hist', 'High_Cl', 'Veg',
       'Main_meals', 'Meals_btw', 'Water_int', 'Physical_Act',
       'Alcohol']

# Extract features for classification
X_classification = df[features_for_classification]
y_classification = df['OB_Level']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_classification, y_classification, test_size=0.2, random_state=42)

# List of classifiers
classifiers = [
    KNeighborsClassifier(),
    SVC(),
    LogisticRegression(),
    GaussianNB(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    MLPClassifier(),
    GradientBoostingClassifier()
]

# Initialize a RandomForestClassifier for feature importance
rf_clf = RandomForestClassifier(n_estimators=500, random_state=22)
rf_clf.fit(X_train, y_train)

# Select features based on importance
sfm = SelectFromModel(rf_clf)
sfm.fit(X_train, y_train)
X_train_selected = sfm.transform(X_train)
X_test_selected = sfm.transform(X_test)

# Calculate metrics for each classifier
results = []
for classifier in classifiers:
    clf_name = classifier.__class__.__name__
    
    # Train the classifier
    classifier.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = classifier.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    sensitivity = confusion[0, 0] / (confusion[0, 0] + confusion[0, 1])
    specificity = confusion[1, 1] / (confusion[1, 0] + confusion[1, 1])
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Append to results table
    results_table = results_table.append({
        'Classifier': classifier_name,
        'Accuracy': accuracy,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1
    }, ignore_index=True)

# Print the results table
print(results_table)


# In[ ]:




