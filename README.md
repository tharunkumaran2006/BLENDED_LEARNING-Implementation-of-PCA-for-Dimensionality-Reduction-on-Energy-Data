# BLENDED LEARNING
# Implementation of Principal Component Analysis (PCA) for Dimensionality Reduction on Energy Data

## AIM:
To implement Principal Component Analysis (PCA) to reduce the dimensionality of the energy data.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. **Load the Data**  
   Import the dataset to begin the dimensionality reduction process.

2. **Explore the Data**  
   Perform an initial analysis to understand data characteristics, distributions, and potential patterns.

3. **Preprocess the Data (Feature Scaling)**  
   Scale features to ensure consistency, preparing the data for principal component analysis (PCA).

4. **Apply PCA for Dimensionality Reduction**  
   Use PCA to reduce the dataset’s dimensionality while retaining the most significant features.

5. **Analyze Explained Variance**  
   Assess the variance explained by each principal component to determine the effectiveness of dimensionality reduction.

6. **Visualize Principal Components**  
   Create visualizations of the principal components to interpret patterns and clusters in reduced dimensions. 

## Program:
### Program to implement Principal Component Analysis (PCA) for dimensionality reduction on the energy data.
### Developed by: THARUN V K
### RegisterNumber: 212223230231

```python
# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset 
data = pd.read_csv('HeightsWeights.csv')

# Step 2: Explore the data
# Display the first few rows and column names for initial inspection
print(data.head())
print(data.columns)

# Step 3: Preprocess the data (Feature Scaling)
# Select the relevant columns for analysis
X = data[['Height(Inches)', 'Weight(Pounds)']]  # Use the appropriate column names

# Standardize the features to bring them to the same scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Apply PCA for dimensionality reduction
# Initialize PCA to reduce the features to 2 components (for simplicity)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Step 5: Analyze the explained variance
# Print the explained variance ratio for each principal component
explained_variance = pca.explained_variance_ratio_
print("Explained Variance Ratio for each Principal Component:", explained_variance)
print("Total Explained Variance:", sum(explained_variance))

# Step 6: Visualize the principal components
# Create a DataFrame to store the principal components
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])

# Plot the first two principal components
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', data=pca_df, alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - Heights and Weights Dataset")
plt.show()
```
## Output:
![Screenshot 2025-05-18 164608](https://github.com/user-attachments/assets/f3d714e4-b71e-4e83-9a2e-03ebb0dbebed)

![Screenshot 2025-05-18 164616](https://github.com/user-attachments/assets/128627e0-a01a-4557-adcf-6299bb64a4d8)


## Result:
Thus, Principal Component Analysis (PCA) was successfully implemented to reduce the dimensionality of the energy dataset.
