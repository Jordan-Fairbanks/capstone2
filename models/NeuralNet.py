import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from tensorflow.keras import layers, Sequential

def create_nueral_network(X, y, epochs=8):
    """
    create_nueral_network takes training data and creates neural network with 5 densely connected
    layers and trains the model.
    PARAMETERS:
        X - pandas.DataFrame or ndarray-like object - the feature data for the training set
        y - pandas.Series or array-like object - the target lables for the feature data
        epochs - int (default:3)- number of epochs to train the model
    RETURNS:
        model - tensorflow.Senquential - the trained nueral network 
    """
    model = Sequential()
    model.add(layers.Dense(500, input_dim=X.shape[1]))
    model.add(layers.Dropout(.2))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(.2))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(.2))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(5,activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
    print(model.summary())
    model.fit(X, y, epochs=epochs, batch_size=500)
    return model


def get_scores(model, X, y):
    """
    get_scores takes a neural network that preforms categorical predictions along with a
    testing set of data and target labels and returns performance metrics.
    PARAMETERS:
        model - tensorflow.Sequential - the nueral network with the appropriate shape to make
                                        predictions on the matrix X
        X - pandas.DataFrame or ndarray-like object - the feature data to make predictions on
        y - pandas.Series or array-like object - the target labels for the feature data
    RETURNS:
        acc - float between 0.0 and 1.0 - float representing the accuracy of the model
        rec - float between 0.0 and 1.0 - float representing the recall of the model
        pre - float between 0.0 and 1.0 - float representing the precision of the model
        f1 - float between 0.0 and 1.0 - float representing the f1 score of the model
        pred - pd.DataFrame - a one-hot encoded dataframe representing the predictions of
                              the model
    """
    y_hat = model.predict(X)
    pred = pd.get_dummies(y_hat.argmax(axis=1))
    acc = accuracy_score(y,pred)
    rec = recall_score(y,pred, average='weighted')
    pre = precision_score(y,pred, average='weighted')
    f1 = f1_score(y,pred, average='weighted')
    print(f"ACC: {acc}")
    print(f"REC: {rec}")
    print(f"PRE: {pre}")
    print(f"F1: {f1}")
    return acc, rec, pre, f1, pred

if __name__ =='__main__':
    # read in data
    train = pd.read_parquet('data/train')
    test = pd.read_parquet('data/test')
    y_train = train.pop('Target').values
    y_test = test.pop('Target').values
    X_train = train.values
    X_test = test.values

    # train and test neural network
    model = create_nueral_network(X_train, pd.get_dummies(y_train))
    acc, rec, pre, f1, pred = get_scores(model, X_test, pd.get_dummies(y_test))

    # save model
    model.save('../objects/Neural_Net')

