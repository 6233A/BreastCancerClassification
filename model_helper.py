import pickle
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

# Returns the test dataset as a Dataframe given the filename (should be "testing.csv" in this repository)
def load_testing(testing_filename):
    return pd.read_csv(testing_filename)

# Returns the machine learning model from the saved files. (One of "logisticRegressor.sav", "naiveBayes.sav", "randomForest.sav", "gradientBoostingMachine.sav")
def load_model(filename):
    return pickle.load(open(filename, 'rb'))

# Returns the input (X) variables for the test dataset, as a numpy array
def get_testing_X_numpy(testing):
    return testing.values[:, testing.columns != "diagnosis"]

# Returns the actual labels (Y - benign or malignant) for the test dataset, as a numpy array
def get_testing_Y_numpy(testing):
    return testing.values[:, testing.columns == "diagnosis"]

# Returns the input (X) variables for the test dataset, as a Dataframe
def get_testing_X_df(testing):
    return testing.loc[:, testing.columns != "diagnosis"]

# Returns the actual labels (Y - benign or malignant) for the test dataset, as a Dataframe
def get_testing_Y_df(testing):
    return testing.loc[:, testing.columns == "diagnosis"]

def evaluate_model(model, testing_X, testing_Y):
    predictions = model.predict(testing_X)

    CM = confusion_matrix(testing_Y, predictions)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]

    # Precision
    PRE = TP/(TP+FP)
    # True positive rate or Recall
    TPR = TP/(TP+FN)
    # True negative rate
    TNR = TN/(TN+FP)
    # Talse positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)

    accuracy = accuracy_score(testing_Y, predictions)
    f1 = f1_score(testing_Y, predictions, average='binary')
    print("Accuracy score: " + str(accuracy))
    print("F1 Score: " + str(f1))
    print("Precision: " + str(PRE))
    print("True positive rate/Recall: " + str(TPR))
    print("True negative rate: " + str(TNR))
    print("False positive rate: " + str(FPR))
    print("False negative rate: " + str(FNR))
    print("True Positives: " + str(TP))
    print("True Negatives: " + str(TN))
    print("False Positives: " + str(FP))
    print("False Negatives: " + str(FN))
