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


print(df.head(10))


# In[4]:


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


# In[5]:


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





# In[6]:


# Box plots for selected features
selected_features = ['Age', 'Height', 'Weight']
for feature in selected_features:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df[feature])
    plt.title(f"Box Plot for {feature}")
    plt.show()


# In[7]:


# Display basic statistics for numerical features
numerical_stats = df[['Age', 'Height', 'Weight']].describe()
print(numerical_stats)


# In[8]:


column_names = df.columns
print(column_names)
# Print the names of numerical features
numerical_feature_names = df.columns.tolist()
print("Numerical Feature Names:")
print(numerical_feature_names)


# In[9]:


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


# In[10]:


df.drop(df.index[df['Height'] == 1.98], inplace=True)          
# Box plots for selected features
selected_features = ['Age', 'Height', 'Weight']
for feature in selected_features:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df[feature])
    plt.title(f"Box Plot for {feature}")
    plt.show()


# In[11]:


# Display basic statistics for numerical features
numerical_stats = df[['Age', 'Height', 'Weight']].describe()
print(numerical_stats)


# In[12]:


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


# In[13]:


# Correlation heatmap
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
print("Correlation Matrix:")
print(correlation_matrix)


# In[14]:


# Visualize distribution of numerical features using histograms
df.hist(bins=20, figsize=(15, 10))
plt.suptitle("Histograms of Numerical Features", y=0.92)
plt.show()



# In[15]:


# Select categorical variables only
categorical_columns = df.select_dtypes(include=['object'])
categorical_column_names = categorical_columns.columns
# Print the categorical columns
print(categorical_column_names)


# In[16]:


from scipy.stats import chi2_contingency
categorical_features = ['Gender', 'Family_hist', 'High_Cl', 'Meals_btw', 'Smoking', 'Alcohol',
       'M_Trans']


# In[17]:


df.head(10)


# In[18]:


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


# In[19]:


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


# In[20]:


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


# In[22]:


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


# In[23]:


# Select numeric variables only
numeric_columns = df.select_dtypes(include=['int64', 'float64'])
numeric_column_names = numeric_columns.columns
# Print the categorical columns
print(numeric_column_names)


# In[24]:


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


# In[25]:


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



# In[26]:


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


# In[27]:


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


# In[28]:


# Dropping 2 columns
df.drop(['Smoking', 'M_Trans'], axis=1, inplace=True)


# In[29]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Define features and target
X = df.drop('OB_Level', axis=1)  # Features
y = df['OB_Level']  # Target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[30]:


pip install mlxtend


# In[31]:


import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns


# Assuming 'Obesity Level' is your target variable
label_encoder = LabelEncoder()
df['OB_Level'] = label_encoder.fit_transform(df['OB_Level'])

# Select relevant features for clustering
features_for_clustering = ['Gender', 'Age', 'Height', 'Weight', 'Family_hist']

# Extract features for clustering
X_clustering = df[features_for_clustering]

# Standardize features
scaler = StandardScaler()
X_clustering_scaled = scaler.fit_transform(X_clustering)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['KMeans_Cluster'] = kmeans.fit_predict(X_clustering_scaled)

# Apply Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=3)
df['Hierarchical_Cluster'] = hierarchical.fit_predict(X_clustering_scaled)

# Evaluate clustering validity using Silhouette Score
silhouette_kmeans = silhouette_score(X_clustering_scaled, df['KMeans_Cluster'])
silhouette_hierarchical = silhouette_score(X_clustering_scaled, df['Hierarchical_Cluster'])

print(f"Silhouette Score for K-Means: {silhouette_kmeans:.2f}")
print(f"Silhouette Score for Hierarchical Clustering: {silhouette_hierarchical:.2f}")

# Visualize clusters
sns.scatterplot(x='Age', y='Gender', hue='KMeans_Cluster', data=df)
plt.title('K-Means Clustering')
plt.show()


# In[31]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd


# Assuming 'Obesity Level' is your target variable
label_encoder = LabelEncoder()
df['OB_Level'] = label_encoder.fit_transform(df['OB_Level'])

# Define features and target
X = df.drop('OB_Level', axis=1)  # Features
y = df['OB_Level']  # Target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training set
rf_classifier.fit(X_train, y_train)

# Predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on the test set: {accuracy:.2f}")


# In[32]:


pip install xgboost


# In[33]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier



# Assuming 'Obesity Level' is your target variable
label_encoder = LabelEncoder()
df['OB_Level'] = label_encoder.fit_transform(df['OB_Level'])

# Define features and target
X = df.drop('OB_Level', axis=1)  # Features
y = df['OB_Level']  # Target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the XGBClassifier
xgb_classifier = XGBClassifier(objective='multi:softmax', num_class=len(df['OB_Level'].unique()), random_state=42)

# Train the model on the training set
xgb_classifier.fit(X_train, y_train)

# Predictions on the test set
y_pred = xgb_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on the test set: {accuracy:.2f}")


# In[34]:


from sklearn.linear_model import LogisticRegression

# Initialize the LogisticRegression model
logreg_classifier = LogisticRegression(max_iter=1000, random_state=42)  
# You can adjust parameters like max_iter

# Train the model on the training set
logreg_classifier.fit(X_train, y_train)

# Predictions on the test set
y_pred_logreg = logreg_classifier.predict(X_test)

# Evaluate the Logistic Regression model
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
print(f"Accuracy of Logistic Regression on the test set: {accuracy_logreg:.2f}")


# In[35]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Assuming 'OB_Level' is the target categorical variable
label_encoder = LabelEncoder()
df['OB_Level'] = label_encoder.fit_transform(df['OB_Level'])

# Select features and target variable
features = ['Gender', 'Age', 'Height', 'Weight', 'Family_hist', 'High_Cl', 'Veg',
       'Main_meals', 'Meals_btw', 'Water_int', 'Physical_Act',
       'Alcohol']
target = 'OB_Level'

X = df[features]
y = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Get feature importances
feature_importances = pd.Series(classifier.feature_importances_, index=features)

# Sort features by importance
sorted_features = feature_importances.sort_values(ascending=False)

# Select top features (you can choose a threshold)
top_features = sorted_features.head(15)  # Adjust the number based on your preference

# Print the selected features
print("Selected Features:")
print(top_features)


# In[ ]:





# In[51]:





# In[37]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

# Assuming 'Obesity Level' is your target variable
label_encoder = LabelEncoder()
df['OB_Level'] = label_encoder.fit_transform(df['OB_Level'])

# Define features and target
X = df.drop('OB_Level', axis=1)  # Features
y = df['OB_Level']  # Target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline for Ridge Regression
ridge_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Standardize features for Ridge
    ('ridge', Ridge(alpha=1.0))  # You can adjust the regularization strength (alpha)
])

# Train the Ridge model on the training set
ridge_pipeline.fit(X_train, y_train)

# Predictions on the test set
y_pred_ridge = ridge_pipeline.predict(X_test)

# Evaluate the Ridge model
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
print(f"Mean Squared Error for Ridge Regression: {mse_ridge:.2f}")

# Create a pipeline for Lasso Regression
lasso_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Standardize features for Lasso
    ('lasso', Lasso(alpha=1.0))  # You can adjust the regularization strength (alpha)
])

# Train the Lasso model on the training set
lasso_pipeline.fit(X_train, y_train)

# Predictions on the test set
y_pred_lasso = lasso_pipeline.predict(X_test)

# Evaluate the Lasso model
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
print(f"Mean Squared Error for Lasso Regression: {mse_lasso:.2f}")


# In[42]:


import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Assuming 'Obesity Level' is your target variable
label_encoder = LabelEncoder()
df['NObeyesdad'] = label_encoder.fit_transform(df['NObeyesdad'])

# Select relevant columns for association rule mining
columns_for_apriori = ['family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE',
       'SCC', 'CALC', 'MTRANS']

# Convert categorical columns to string type
for column in columns_for_apriori:
    df[column] = df[column].astype(str)

# Apply one-hot encoding for Apriori
df_apriori = pd.get_dummies(df[columns_for_apriori])

# Apply Apriori algorithm
frequent_itemsets = apriori(df_apriori, min_support=0.1, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# Display the association rules
print("Association Rules:")
print(rules)


# In[50]:


import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns


# Assuming 'Obesity Level' is your target variable
label_encoder = LabelEncoder()
df['OB_Level'] = label_encoder.fit_transform(df['OB_Level'])

# Select relevant features for clustering
features_for_clustering = ['Age', 'Height', 'Weight']

# Extract features for clustering
X_clustering = df[features_for_clustering]

# Standardize features
scaler = StandardScaler()
X_clustering_scaled = scaler.fit_transform(X_clustering)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
df['KMeans_Cluster'] = kmeans.fit_predict(X_clustering_scaled)

# Apply Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=5)
df['Hierarchical_Cluster'] = hierarchical.fit_predict(X_clustering_scaled)

# Evaluate clustering validity using Silhouette Score
silhouette_kmeans = silhouette_score(X_clustering_scaled, df['KMeans_Cluster'])
silhouette_hierarchical = silhouette_score(X_clustering_scaled, df['Hierarchical_Cluster'])

print(f"Silhouette Score for K-Means: {silhouette_kmeans:.2f}")
print(f"Silhouette Score for Hierarchical Clustering: {silhouette_hierarchical:.2f}")

# Visualize clusters (example for K-Means)
sns.scatterplot(x='Age', y='Height', hue='KMeans_Cluster', data=df)
plt.title('K-Means Clustering')
plt.show()


# In[44]:


import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'Obesity Level' is your target variable
label_encoder = LabelEncoder()
df['OB_Level'] = label_encoder.fit_transform(df['OB_Level'])

# Select relevant features for clustering
features_for_clustering = ['Age', 'Height', 'Weight']

# Extract features for clustering
X_clustering = df[features_for_clustering]

# Standardize features
scaler = StandardScaler()
X_clustering_scaled = scaler.fit_transform(X_clustering)

# Apply Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=3)
df['Hierarchical_Cluster'] = hierarchical.fit_predict(X_clustering_scaled)
# Evaluate clustering validity using Silhouette Score
silhouette_kmeans = silhouette_score(X_clustering_scaled, df['KMeans_Cluster'])
silhouette_hierarchical = silhouette_score(X_clustering_scaled, df['Hierarchical_Cluster'])

print(f"Silhouette Score for K-Means: {silhouette_kmeans:.2f}")
print(f"Silhouette Score for Hierarchical Clustering: {silhouette_hierarchical:.2f}")

# Visualize clusters for Hierarchical Clustering
sns.scatterplot(x='Age', y='Height', hue='Hierarchical_Cluster', data=df)
plt.title('Hierarchical Clustering')
plt.show()

# Visualize clusters for Hierarchical Clustering
sns.scatterplot(x='Age', y='Height', hue='KMeans_Cluster', data=df)
plt.title('KMeans Clustering')
plt.show()


# Visualize clusters for Hierarchical Clustering
sns.scatterplot(x='Age', y='Weight', hue='KMeans_Cluster', data=df)
plt.title('KMeans Clustering')
plt.show()




# Visualize clusters for Hierarchical Clustering
sns.scatterplot(x='Age', y='Weight', hue='Hierarchical_Cluster', data=df)
plt.title('Hierarchical Clustering')
plt.show()



# In[ ]:


from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Assuming 'OB_Level' is your target variable
label_encoder = LabelEncoder()
df['OB_Level'] = label_encoder.fit_transform(df['OB_Level'])

# Select relevant features for classification
features_for_classification = ['Age', 'Height', 'Weight']

# Extract features for classification
X_classification = df[features_for_classification]
y_classification = df['OB_Level']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_classification, y_classification, test_size=0.2, random_state=42)

# Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)
y_pred_dt = dt_classifier.predict(X_test)

# Naïve Bayes Classifier
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
y_pred_nb = nb_classifier.predict(X_test)

# Define class names based on unique classes in 'OB_Level'
class_names = label_encoder.classes_

# Evaluate classification models
accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt, average='weighted')
recall_dt = recall_score(y_test, y_pred_dt, average='weighted')

accuracy_nb = accuracy_score(y_test, y_pred_nb)
precision_nb = precision_score(y_test, y_pred_nb, average='weighted')
recall_nb = recall_score(y_test, y_pred_nb, average='weighted')

print("Decision Tree Metrics:")
print(f"Accuracy: {accuracy_dt:.2f}")
print(f"Precision: {precision_dt:.2f}")
print(f"Recall: {recall_dt:.2f}")

print("\nNaïve Bayes Metrics:")
print(f"Accuracy: {accuracy_nb:.2f}")
print(f"Precision: {precision_nb:.2f}")
print(f"Recall: {recall_nb:.2f}")

# Assuming 'Obesity Level' is your target variable
class_names = df['OB_Level'].unique()

# Visualize Decision Tree (example)
from sklearn.tree import plot_tree
plt.figure(figsize=(10, 6))
plot_tree(dt_classifier, feature_names=features_for_classification, class_names=class_names, filled=True, rounded=True)
plt.title('Decision Tree Visualization')
plt.show()


# In[47]:


import pandas as pd
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'Obesity Level' is your target variable
label_encoder = LabelEncoder()
df['OB_Level'] = label_encoder.fit_transform(df['OB_Level'])

# Select relevant features for clustering
features_for_clustering = ['Gender', 'Age', 'Height', 'Weight', 'Family_hist', 'High_Cl', 'Veg',
       'Main_meals', 'Meals_btw', 'Water_int', 'Physical_Act',
       'Alcohol']

# Extract features for clustering
X_clustering = df[features_for_clustering]

# Standardize features
scaler = StandardScaler()
X_clustering_scaled = scaler.fit_transform(X_clustering)

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=3)
df['DBScan_Cluster'] = dbscan.fit_predict(X_clustering_scaled)

# Apply OPTICS clustering
optics = OPTICS(min_samples=3)
df['OPTICS_Cluster'] = optics.fit_predict(X_clustering_scaled)

# Visualize clusters (example for DBSCAN)
sns.scatterplot(x='Weight', y='Height', hue='DBScan_Cluster', data=df)
plt.title('DBSCAN Clustering')
plt.show()

# Visualize clusters (example for OPTICS)
sns.scatterplot(x='Weight', y='Height', hue='OPTICS_Cluster', data=df)
plt.title('OPTICS Clustering')
plt.show()


from sklearn.metrics import silhouette_score

# Evaluate clustering validity using Silhouette Score
silhouette_dbscan = silhouette_score(X_clustering_scaled, df['DBScan_Cluster'])
silhouette_optics = silhouette_score(X_clustering_scaled, df['OPTICS_Cluster'])

print(f"Silhouette Score for DBSCAN: {silhouette_dbscan:.2f}")
print(f"Silhouette Score for OPTICS: {silhouette_optics:.2f}")


# In[39]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
    MLPClassifier(hidden_layer_sizes=(100,), max_iter=500),
    GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42)
]

# Create a table for results
results_table = pd.DataFrame(columns=['Classifier', 'Accuracy', 'Precision', 'Recall', 'F1-score'])

# Iterate through classifiers
for classifier in classifiers:
    classifier_name = classifier.__class__.__name__

    # Fit and predict
    classifier.fit(X_train_scaled, y_train)
    y_pred = classifier.predict(X_test_scaled)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Append to results table
    results_table = results_table.append({
        'Classifier': classifier_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1
    }, ignore_index=True)

# Print the results table
print(results_table)


# In[49]:





# In[33]:


from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



# Select relevant features for classification
features_for_classification = ['Gender', 'Age', 'Height', 'Weight', 'Family_hist', 'High_Cl', 'Veg',
       'Main_meals', 'Meals_btw', 'Water_int', 'Physical_Act',
       'Alcohol']

# Extract features for classification
X_classification = df[features_for_classification]
y_classification = df['OB_Level']

# Standardize features
scaler = StandardScaler()
X_classification_scaled = scaler.fit_transform(X_classification)



# Select features and target variable
features_for_pca = ['Gender', 'Age', 'Height', 'Weight', 'Family_hist', 'High_Cl', 'Veg',
       'Main_meals', 'Meals_btw', 'Water_int', 'Physical_Act',
       'Alcohol']
X_pca = df[features_for_pca]
y_pca = df['OB_Level']

# Standardize the features
scaler = StandardScaler()
X_pca_scaled = scaler.fit_transform(X_pca)

# Apply PCA
pca = PCA(n_components=0.95)  # Choose the desired explained variance
X_pca_transformed = pca.fit_transform(X_classification_scaled)


# Split the dataset into training and testing sets
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca_transformed, y_classification, test_size=0.2, random_state=42)

# Function to evaluate classifiers with PCA
def evaluate_classifier_with_pca(classifier, X_train, X_test, y_train, y_test):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    return accuracy, precision, recall, f1

# Apply classifiers with PCA
classifiers_with_pca = [
    KNeighborsClassifier(),
    SVC(),
    LogisticRegression(),
    GaussianNB(),
    RandomForestClassifier(),
    DecisionTreeClassifier(),
    AdaBoostClassifier(),
    MLPClassifier(),
    GradientBoostingClassifier()
]

# Create a DataFrame to store results
results_with_pca = pd.DataFrame(columns=['Classifier', 'Accuracy', 'Precision', 'Recall', 'F1-score'])

# Loop through classifiers and evaluate with PCA
for classifier in classifiers_with_pca:
    classifier_name = classifier.__class__.__name__
    accuracy, precision, recall, f1 = evaluate_classifier_with_pca(classifier, X_train_pca, X_test_pca, y_train_pca, y_test_pca)
    results_with_pca = results_with_pca.append({'Classifier': classifier_name, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-score': f1}, ignore_index=True)

# Print results
print("Results with PCA:")
print(results_with_pca)


# In[58]:


from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



# Select relevant features for classification
features_for_classification = ['Gender', 'Age', 'Height', 'Weight', 'Family_hist', 'High_Cl', 'Veg',
       'Main_meals', 'Meals_btw', 'Water_int', 'Physical_Act',
       'Alcohol']

# Extract features for classification
X_classification = df[features_for_classification]
y_classification = df['OB_Level']

# Standardize features
scaler = StandardScaler()
X_classification_scaled = scaler.fit_transform(X_classification)



# Select features and target variable
features_for_pca = ['Gender', 'Age', 'Height', 'Weight', 'Family_hist', 'High_Cl', 'Veg',
       'Main_meals', 'Meals_btw', 'Water_int', 'Physical_Act',
       'Alcohol']
X_pca = df[features_for_pca]
y_pca = df['OB_Level']

# Standardize the features
scaler = StandardScaler()
X_pca_scaled = scaler.fit_transform(X_pca)

# Apply PCA
pca = PCA(n_components=10)  # Choose the desired explained variance
X_pca_transformed = pca.fit_transform(X_classification_scaled)


# Split the dataset into training and testing sets
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca_transformed, y_classification, test_size=0.2, random_state=42)

# Function to evaluate classifiers with PCA
def evaluate_classifier_with_pca(classifier, X_train, X_test, y_train, y_test):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    return accuracy, precision, recall, f1

# Apply classifiers with PCA
classifiers_with_pca = [
    KNeighborsClassifier(),
    SVC(),
    LogisticRegression(),
    GaussianNB(),
    RandomForestClassifier(),
    DecisionTreeClassifier(),
    AdaBoostClassifier(),
    MLPClassifier(),
    GradientBoostingClassifier()
]

# Create a DataFrame to store results
results_with_pca = pd.DataFrame(columns=['Classifier', 'Accuracy', 'Precision', 'Recall', 'F1-score'])

# Loop through classifiers and evaluate with PCA
for classifier in classifiers_with_pca:
    classifier_name = classifier.__class__.__name__
    accuracy, precision, recall, f1 = evaluate_classifier_with_pca(classifier, X_train_pca, X_test_pca, y_train_pca, y_test_pca)
    results_with_pca = results_with_pca.append({'Classifier': classifier_name, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-score': f1}, ignore_index=True)

# Print results
print("Results with PCA:")
print(results_with_pca)


# In[45]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

# Assuming 'OB_Level' is your target variable
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

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply Recursive Feature Elimination (RFE) with RandomForestClassifier as the estimator
rfe = RFE(RandomForestClassifier(), n_features_to_select=13)
X_rfe = rfe.fit_transform(X_train_scaled, y_train)

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

# Calculate metrics for each classifier
results = []
for classifier in classifiers:
    clf_name = classifier.__class__.__name__
    
    # Train the classifier
    classifier.fit(X_rfe, y_train)
    
    # Make predictions on the test set
    y_pred = classifier.predict(X_test_scaled[:, rfe.support_])
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results.append([clf_name, accuracy, precision, recall, f1])

# Display results in a table
results_df = pd.DataFrame(results, columns=['Classifier', 'Accuracy', 'Precision', 'Recall', 'F1-score'])
print(results_df)


# In[46]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

# Assuming 'OB_Level' is your target variable
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

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply Recursive Feature Elimination (RFE) with RandomForestClassifier as the estimator
rfe = RFE(RandomForestClassifier(), n_features_to_select=10)
X_rfe = rfe.fit_transform(X_train_scaled, y_train)

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

# Calculate metrics for each classifier
results = []
for classifier in classifiers:
    clf_name = classifier.__class__.__name__
    
    # Train the classifier
    classifier.fit(X_rfe, y_train)
    
    # Make predictions on the test set
    y_pred = classifier.predict(X_test_scaled[:, rfe.support_])
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results.append([clf_name, accuracy, precision, recall, f1])

# Display results in a table
results_df = pd.DataFrame(results, columns=['Classifier', 'Accuracy', 'Precision', 'Recall', 'F1-score'])
print(results_df)


# In[49]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

# Assuming 'OB_Level' is your target variable
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

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply Recursive Feature Elimination (RFE) with RandomForestClassifier as the estimator
rfe = RFE(RandomForestClassifier(), n_features_to_select=8)
X_rfe = rfe.fit_transform(X_train_scaled, y_train)

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

# Calculate metrics for each classifier
results = []
for classifier in classifiers:
    clf_name = classifier.__class__.__name__
    
    # Train the classifier
    classifier.fit(X_rfe, y_train)
    
    # Make predictions on the test set
    y_pred = classifier.predict(X_test_scaled[:, rfe.support_])
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results.append([clf_name, accuracy, precision, recall, f1])

# Display results in a table
results_df = pd.DataFrame(results, columns=['Classifier', 'Accuracy', 'Precision', 'Recall', 'F1-score'])
print(results_df)


# In[50]:


import matplotlib.pyplot as plt

# Apply Recursive Feature Elimination (RFE) with RandomForestClassifier as the estimator
rfe = RFE(RandomForestClassifier(), n_features_to_select=12)
X_rfe = rfe.fit_transform(X_train_scaled, y_train)

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

# Calculate metrics for each classifier
results = []
for classifier in classifiers:
    clf_name = classifier.__class__.__name__
    
    # Train the classifier
    classifier.fit(X_rfe, y_train)
    
    # Make predictions on the test set
    y_pred = classifier.predict(X_test_scaled[:, rfe.support_])
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results.append([clf_name, accuracy, precision, recall, f1])

# Display results in a table
results_df = pd.DataFrame(results, columns=['Classifier', 'Accuracy', 'Precision', 'Recall', 'F1-score'])
print(results_df)

# Plot RFE ranking
ranking = rfe.ranking_
selected_features = [features_for_classification[i] for i in range(len(features_for_classification)) if ranking[i] == 1]
plt.figure(figsize=(10, 6))
plt.bar(range(len(features_for_classification)), ranking)
plt.xticks(range(len(features_for_classification)), features_for_classification, rotation=45)
plt.title('RFE Ranking of Features')
plt.xlabel('Features')
plt.ylabel('Ranking')
plt.show()


# In[52]:


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
rf_clf = RandomForestClassifier(random_state=42)
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
    classifier.fit(X_train_selected, y_train)
    
    # Make predictions on the test set
    y_pred = classifier.predict(X_test_selected)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results.append([clf_name, accuracy, precision, recall, f1])

# Display results in a table
results_df = pd.DataFrame(results, columns=['Classifier', 'Accuracy', 'Precision', 'Recall', 'F1-score'])
print(results_df)

# Plot feature importances
feature_importances = rf_clf.feature_importances_
plt.figure(figsize=(10, 6))
plt.bar(range(len(features_for_classification)), feature_importances)
plt.xticks(range(len(features_for_classification)), features_for_classification)
plt.title('Feature Importances from RandomForestClassifier')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()


# In[54]:


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
    classifier.fit(X_train_selected, y_train)
    
    # Make predictions on the test set
    y_pred = classifier.predict(X_test_selected)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results.append([clf_name, accuracy, precision, recall, f1])

# Display results in a table
results_df = pd.DataFrame(results, columns=['Classifier', 'Accuracy', 'Precision', 'Recall', 'F1-score'])
print(results_df)

# Plot feature importances
feature_importances = rf_clf.feature_importances_
plt.figure(figsize=(10, 6))
plt.bar(range(len(features_for_classification)), feature_importances)
plt.xticks(range(len(features_for_classification)), features_for_classification)
plt.title('Feature Importances from RandomForestClassifier')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()


# In[56]:


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
    KNeighborsClassifier(),
    SVC(),
    LogisticRegression(),
    GaussianNB(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    MLPClassifier(),
    GradientBoostingClassifier()
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
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results.append([clf_name, accuracy, precision, recall, f1])

# Display results in a table
results_df = pd.DataFrame(results, columns=['Classifier', 'Accuracy', 'Precision', 'Recall', 'F1-score'])
print(results_df)


# In[57]:


from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
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
    KNeighborsClassifier(),
    SVC(),
    LogisticRegression(),
    GaussianNB(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    MLPClassifier(),
    GradientBoostingClassifier()
]

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
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results.append([clf_name, accuracy, precision, recall, f1])

# Display results in a table
results_df = pd.DataFrame(results, columns=['Classifier', 'Accuracy', 'Precision', 'Recall', 'F1-score'])
print(results_df)


# In[34]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
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

# Initialize RandomForestClassifier as the base estimator
base_estimator = RandomForestClassifier(n_estimators=100, random_state=42)

# Initialize BaggingClassifier with RandomForestClassifier as base estimator
bagging_classifier = BaggingClassifier(base_estimator=base_estimator, n_estimators=10, random_state=42)

# Fit and predict
bagging_classifier.fit(X_train_scaled, y_train)
y_pred = bagging_classifier.predict(X_test_scaled)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
sensitivity = confusion[0, 0] / (confusion[0, 0] + confusion[0, 1])
specificity = confusion[1, 1] / (confusion[1, 0] + confusion[1, 1])
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print the results
print(f"Accuracy: {accuracy:.2f}")
print(f"Sensitivity: {sensitivity:.2f}")
print(f"Specificity: {specificity:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




