import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
data = pd.read_csv("../data/student_data.csv")

print("\nDataset Preview:")
print(data.head())

# ----------------------
# EDA
# ----------------------

print("\nDataset Info:")
data.info()


print("\nStatistical Summary:")
print(data.describe())

# Correlation Heatmap
plt.figure()
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# ----------------------
# Model Preparation
# ----------------------

X = data.drop("final_score", axis=1)
y = data["final_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------
# Model 1: Linear Regression
# ----------------------

lr = LinearRegression()
lr.fit(X_train, y_train)
lr_predictions = lr.predict(X_test)

lr_mae = mean_absolute_error(y_test, lr_predictions)
lr_r2 = r2_score(y_test, lr_predictions)



print("\nLinear Regression Results:")
print("MAE:", lr_mae)
print("R2 Score:", lr_r2)
cv_scores_lr = cross_val_score(lr, X, y, cv=5, scoring='r2')
print("\nLinear Regression Cross-Validation R2 Scores:", cv_scores_lr)
print("Average CV R2:", cv_scores_lr.mean())


# ----------------------
# Model 2: Random Forest
# ----------------------

rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    random_state=42
)

rf.fit(X_train, y_train)
rf_predictions = rf.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)

print("\nRandom Forest Results:")
print("MAE:", rf_mae)
print("R2 Score:", rf_r2)

cv_scores_rf = cross_val_score(rf, X, y, cv=5, scoring='r2')
print("\nRandom Forest Cross-Validation R2 Scores:", cv_scores_rf)
print("Average CV R2:", cv_scores_rf.mean())


# ----------------------
# Feature Importance
# ----------------------

feature_importance = pd.Series(
    rf.feature_importances_, index=X.columns
)

print("\nFeature Importance:")
print(feature_importance.sort_values(ascending=False))

plt.figure()
feature_importance.sort_values().plot(kind='barh')
plt.title("Feature Importance (Random Forest)")
plt.show()

# ----------------------
# Actual vs Predicted
# ----------------------

plt.figure()
plt.scatter(y_test, rf_predictions)
plt.xlabel("Actual Scores")
plt.ylabel("Predicted Scores")
plt.title("Actual vs Predicted Scores (Random Forest)")
plt.show()

residuals = y_test - rf_predictions

plt.figure()
plt.scatter(rf_predictions, residuals)
plt.axhline(y=0)
plt.xlabel("Predicted Scores")
plt.ylabel("Residuals")
plt.title("Residual Plot (Random Forest)")
plt.show()
