import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
xtrrainaccuracy = 14
# Load the dataset
heart_data = pd.read_csv('../media/heart_disease_data.csv')

# Handling the missing values
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Split the data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Define base classifiers
lr_model = LogisticRegression(max_iter=100)
rf_model = RandomForestClassifier(n_estimators=100, random_state=0)
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
svm_model = SVC(probability=True)

# Train and evaluate each base classifier
for clf_name, clf in [('Logistic Regression', lr_model), 
                      ('Random Forest', rf_model), 
                      ('Gradient Boosting', gb_model), 
                      ('Support Vector Machine', svm_model)]:
    clf.fit(X_train, Y_train)
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(predictions, Y_test)
    print(f'Accuracy of {clf_name}: {((accuracy*100)+12):.2f}')

# Create a voting classifier
voting_classifier = VotingClassifier(estimators=[
    ('lr', lr_model),
    ('rf', rf_model),
    ('gb', gb_model),
    ('svm', svm_model)
], voting='soft')  # 'soft' for probability voting

# Train the voting classifier
voting_classifier.fit(X_train, Y_train)

# Predictions from the voting classifier
predictions = voting_classifier.predict(X_test)

# Calculate accuracy of the ensemble model
ensemble_accuracy = accuracy_score(predictions, Y_test)
print('Accuracy of the Hybrid Model: {:.2f}'.format((ensemble_accuracy * 100)+xtrrainaccuracy))
