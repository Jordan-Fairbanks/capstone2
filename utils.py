import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

def labels_to_numbers(label_vector):
    """
    labels_to_numbers takes an array-like object and creates a dictionary that maps each
    label to a integer. It also returns a pandas series object of integers corresponding
    the dictionary mapping of the original array.
    PARAMETERS:
        label_vector = pandas.Series or array-like object - array of strings
    RETURNS:
    categories - dict(int:str) - a dictionary with integers as keys that map to each unique
                                 string in the label_vector
    arr - pandas.Series - a mapping of the label_vector according the the categories dictionary
    """
    number_labels =[]
    categories = {}
    for ind, category in enumerate(label_vector.unique()): 
        categories[ind] = category
    for label in label_vector:
        for ind, category in enumerate(label_vector.unique()):  
            if label == category:
                number_labels.append(ind)
    arr = pd.series(number_labels)
    return categories, arr

def test_model(model, X_train, X_test,y_train, y_test):
    """
    test_model takes a categorical classifier model that conforms to the sklearn standard
    and trains and tests the model, then returns performance scores and the prediction
    PARAMETERS:
        model - categorical classifier object with .fit() and .predict() methods
        X_train - pandas.DataFrame or ndarray-like object - the feature data to train the model on
        X_test - pandas.DataFrame or ndarray-like object - the feature data to make predictions on
        y_train - pandas.Series or array-like object - the target labels for the training data
        y_test - pandas.Series or array-like object - the target labels for the testing data
    RETURNS:
        acc - float between 0.0 and 1.0 - float representing the accuracy of the model
        rec - float between 0.0 and 1.0 - float representing the recall of the model
        pre - float between 0.0 and 1.0 - float representing the precision of the model
        f1 - float between 0.0 and 1.0 - float representing the f1 score of the model
        pred - numpy.array - an array representing the predictions of the model
    """
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test,pred)
    rec = recall_score(y_test,pred, average='weighted')
    pre = precision_score(y_test,pred, average='weighted')
    f1 = f1_score(y_test,pred, average='weighted')
    print(f"ACC: {acc}")
    print(f"REC: {rec}")
    print(f"PRE: {pre}")
    print(f"F1: {f1}")
    return acc, rec, pre, f1, pred