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