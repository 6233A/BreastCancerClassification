import pickle
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

def load_testing(testing_filename):
    return pd.read_csv(testing_filename)

def load_model(filename):
    return pickle.load(open(filename, 'rb'))

def get_testing_X_numpy(testing):
    return testing.values[:, testing.columns != "diagnosis"]

def get_testing_Y_numpy(testing):
    return testing.values[:, testing.columns == "diagnosis"]

def get_testing_X_df(testing):
    return testing.loc[:, testing.columns != "diagnosis"]

def get_testing_Y_df(testing):
    return testing.loc[:, testing.columns == "diagnosis"]