import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from ..utils import test_model

if __name__ == '__main__':
    # read in data
    train = pd.read_parquet('data/train')
    test = pd.read_parquet('data/test')
    y_train = train.pop('Target').values
    y_test = test.pop('Target').values
    X_train = train.values
    X_test = test.values

    #create and train model
    logreg = LogisticRegression(solver = 'lbfgs',multi_class='auto')
    acc, rec, pre, f1, pred = test_model(logreg, X_train, X_test, y_train, y_test)
    pickle.dump(logreg, open('Logistic_Regression.pkl','wb'))
