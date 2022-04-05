# Project <Titanic Machine Learning>
# Made by Daniil Khmelnytskyi
# 05.04.2022

# Imports 
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def predict(model, X_test, train_data_ids):
    """
        Func makes predictions and saves the submission
    """

    predictions = model.predict(X_test)

    output = pd.DataFrame({'PassengerId': train_data_ids, 'Survived': predictions})
    output.to_csv('result/submission.csv', index=False)

    print("Submission was successfully saved")


def calc_accuracy():
    """
        Func checks accuracy of prediction
    """
    y_data = pd.read_csv("result/submission.csv")
    right_data = pd.read_csv("data/gender_submission.csv")

    right_data_refactored = dict(list(zip(list(right_data['PassengerId']), list(right_data['Survived']))))

    amount_of_all = len(right_data_refactored)
    counter_of_right = 0

    for i in range(len(y_data)):
        if right_data_refactored[y_data['PassengerId'][i]] == y_data['Survived'][i]:
            counter_of_right += 1

    print(f"Amount of all test records: {amount_of_all}")
    print(f"Counter of right predicted records: {counter_of_right}")

    accuracy = counter_of_right / amount_of_all * 100
    print(f'Accuracy is: {accuracy}')

if __name__ == "__main__":
    # Get data to train && test
    train_data = pd.read_csv("data/train.csv") # read training data

    test_data = pd.read_csv("data/test.csv") # read testing data

    features = ["Pclass", "Sex", "SibSp", "Parch"] # columns to extract from csv

    X = pd.get_dummies(train_data[features]) # converts categorical data into indicator variables (sex => male && female)
    y = train_data["Survived"] # extract column 'Survived'

    model = DecisionTreeClassifier(random_state=1, max_depth=3) # initializes a model

    model.fit(X, y) # trains the model
    
    X_test = pd.get_dummies(test_data[features]) # converts categorical data into indicator variables (sex => male && female)

    predict(model, X_test, test_data.PassengerId) # makes predictions on trained model

    calc_accuracy() # calculates the accuracy of predictions