import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from ..utils import test_model

if __name__ == '__main__':
    # read in data
    train = pd.read_parquet('data/train')
    test = pd.read_parquet('data/test')
    X_train = train.copy()
    X_test = test.copy()
    y_train = X_train.pop('Target')
    y_test = X_test.pop('Target')

    # train, test, and save model
    rf = RandomForestClassifier(n_estimators=1000, max_depth=3)
    acc, rec, pre, f1, pred = test_model(rf, X_train, X_test, y_train, y_test)
    pickle.dump(rf, open('Random_Forest.pkl','wb'))
