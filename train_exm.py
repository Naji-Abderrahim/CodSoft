from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

datafile = "./Titanic-Dataset.csv"
init_data = pd.read_csv(datafile)


# clean it from unwanted data, and clean the Nan rows
df = init_data.drop(columns=['PassengerId', 'Name', 'Cabin', 'Embarked', 'Ticket']).dropna()

# remap male->1, female->2
df['Sex'] = df['Sex'].map(gender_mapping)# Assuming you have your data in a DataFrame called 'df'
# First, let's split the data into features (X) and target (y)
X = df.drop('Survived', axis=1)  # Assuming 'Survived' is your target column
y = df['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the decision tree
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# Make predictions
y_pred = dt_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Print detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': dt_classifier.feature_importances_
})
print("\nFeature Importance:")
print(feature_importance.sort_values('importance', ascending=False))

# Visualize the decision tree (optional)
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))
plot_tree(dt_classifier, feature_names=X.columns, class_names=['Not Survived', 'Survived'], 
          filled=True, rounded=True)
plt.show()

# Hyperparameter tuning using GridSearchCV
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), 
                         param_grid, 
                         cv=5, 
                         scoring='accuracy')
grid_search.fit(X_train, y_train)

print("\nBest parameters:", grid_search.best_params_)
print("Best cross-validation accuracy: {:.2f}%".format(grid_search.best_score_ * 100))

# Use the best model
best_dt = grid_search.best_estimator_
y_pred_best = best_dt.predict(X_test)
print("\nBest model accuracy on test set: {:.2f}%".format(
    accuracy_score(y_test, y_pred_best) * 100))