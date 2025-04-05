import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# load data
data = pd.read_csv('train.csv')

# list of features to check for outliers
features_to_check = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'YearBuilt']

# outliers removal
def detect_outliers(df, features):
    outlier_indices = []
    
    for feature in features:
        # calculate Q1 and Q3 
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        
        # IQR
        IQR = Q3 - Q1
        
        # outlier limits
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # get outliers
        outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)].index
        outlier_indices.extend(outliers)
    
    # remove duplicates
    outlier_indices = list(set(outlier_indices))
    
    return outlier_indices

# detect outliers
outliers = detect_outliers(data, features_to_check)

# remove outliers
data_cleaned = data.drop(outliers)

# verify the new dataset shape
print(f"Removed {len(outliers)} outliers. New dataset shape: {data_cleaned.shape}")

# select features and target variable
features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'YearBuilt']
target = 'SalePrice'

# drop rows with missing values
data_cleaned = data_cleaned.dropna(subset=features + [target])

# input (X) and output (y)
X = data_cleaned[features]
y = data_cleaned[target]

# split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train the Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# predict
y_pred = lr_model.predict(X_test)

# evaluate
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# display results
print("Model Evaluation after removing outliers:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# visualize predictions vs actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Sale Prices")
plt.ylabel("Predicted Sale Prices")
plt.title("Linear Regression Predictions vs Actuals (After Outlier Removal)")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.show()

# visualize distributions to show outlier removal
plt.figure(figsize=(12, 8))
for feature in features_to_check:
    sns.boxplot(data_cleaned[feature])
    plt.title(f"Boxplot of {feature} (After Outlier Removal)")
    plt.show()
