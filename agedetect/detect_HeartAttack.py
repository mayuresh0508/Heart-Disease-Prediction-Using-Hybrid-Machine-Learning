from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import get_object_or_404
from .models import Admin_Helath_CSV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

@csrf_exempt
def prdict_heart_disease(list_data):
   
        # try:
        #     # Assuming Admin_Helath_CSV is your model
        #     csv_instance = Admin_Helath_CSV.objects.get(id=1)
        # except Admin_Helath_CSV.DoesNotExist:
        #     return JsonResponse({'error': 'Admin_Helath_CSV with id=1 does not exist.'}, status=404)

        # Load data from the CSV file or another source
        # Replace this with your actual data loading logic
        data = pd.read_csv('C:/Users/Mayuresh/Desktop/Heart Attack Detection Hub/Heart Attack Detection Hub/media/heart_disease_data.csv')
        #print(data.dtypes)

        # Handling missing values
        X = data.drop(columns='target', axis=1)
        Y = data['target']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

         # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
        
        model = LogisticRegression(max_iter=5000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)

        cv_scores = cross_val_score(model, X_scaled, Y, cv=5)

        # Perform predictions
        input_data_as_numpy_array = np.asarray(list_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prediction = model.predict(input_data_reshaped)

        logistic_regression_accuracy = (model.score(X_train, y_train)).mean() * 100


        # from sklearn.metrics import confusion_matrix
        # cm = confusion_matrix(y_test, y_pred)

     
        
        return  cv_scores.mean()*100, logistic_regression_accuracy, prediction
