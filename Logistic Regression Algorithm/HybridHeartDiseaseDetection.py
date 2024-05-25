import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def train_and_evaluate_models():
    # Load the dataset
    heart_data = pd.read_csv('C:/Users/Mayuresh/Desktop/Heart Attack Detection Hub/Heart Attack Detection Hub/media/heart_disease_data.csv')

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
        print(f'Accuracy of {clf_name}: {accuracy:.2f}%')

    # Create a voting classifier
    voting_classifier = VotingClassifier(estimators=[
        ('lr', lr_model),
        ('rf', rf_model),
        ('gb', gb_model),
        ('svm', svm_model)
    ], voting='soft')  # 'soft' for probability voting

    # Train the voting classifier
    voting_classifier.fit(X_train, Y_train)

    return voting_classifier, X_train.columns

def predict_heart_disease(voting_classifier, feature_names, input_data):
    input_data_df = pd.DataFrame([input_data], columns=feature_names)
    prediction = voting_classifier.predict(input_data_df)
    probability = voting_classifier.predict_proba(input_data_df)
    return prediction, probability

# Example usage
voting_classifier, feature_names = train_and_evaluate_models()

# Example input data
input_data = [57,1,0,190,276,0,0,112,1,0.6,1,1,1]

prediction, probability = predict_heart_disease(voting_classifier, feature_names, input_data)
print("Prediction:", prediction)
print("Probability:", probability)
