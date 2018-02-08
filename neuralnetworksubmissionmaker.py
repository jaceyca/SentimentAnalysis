import numpy as np
import pathlib
from datetime import datetime

# classifiers
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, RidgeClassifier, SGDClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import BernoulliRBM, MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF
from sklearn.cross_validation import KFold
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
# import matplotlib.pyplot as plt

def load_data(filename, train=True):
    """
    Function loads data stored in the file filename and returns it as a numpy ndarray.
    
    Inputs:
        filename: given as a string
        (optional) train: used to determine whether this is the training or test set
        
    Outputs:
        Data contained in the file, returned as a numpy ndarray
    """
    X = []
    y = []
    with open(filename) as f:
        for line in f:
            if (train):
                # remove \n, split on space, separate into label and weights
                X.append(line.strip().split(' ')[1:])
                y.append(line.strip().split(' ')[0])
            else:
                X.append(line.strip().split(' '))
                
    # convert to np, cast to int, and remove the headers
    X = np.asarray(X[1:]).astype(int)
    if (train):
        y = np.asarray(y[1:]).astype(int)
        
    return X, y

def split_data(x_train, y_train):
    '''
    Function for cross validiation. 
    
    Inputs: 
        x_train: training data points
        y_train: training labels
        
    Outputs:
        trainX: randomized 4/5 of given data points
        trainY: corresponding labels
        testX: randomized 1/5 of given data points
        testY: corresponding lables
    '''
    dataSplit = ShuffleSplit(n_splits = 1, test_size = 0.2)
    for train, test in dataSplit.split(x_train, y_train):
        return x_train[train], y_train[train], x_train[test], y_train[test] 
       
def normalization(X_train, X_test):
    '''
    Function to normalize training and test data

    Inputs:
        X_train: training set data points
        X_test: test set data points

    Outputs:
        train_norm: normalized training set data points
        test_norm: normalized test set data points
    '''
    normalizer = Normalizer().fit(X_train)
    train_norm = normalizer.transform(X_train)
    test_norm = normalizer.transform(X_test)

    return (train_norm, test_norm)

def make_predictions(clf, X, y, test):
    '''
    Function to train and test our classifier
    
    Inputs:
        clf: classifier
        X: data points
        y: labels
        test: test set
    
    Outputs:
        predictions: predictions from running the clf on the test set
    '''
    clf.fit(X, y)
    predictions = clf.predict(test)
    predictions = predictions.astype(int)
    
    return predictions

def save_data(data, filename="%s.txt" % datetime.today().strftime("%X").replace(":", "")):
    '''
    Function to save the predictions by the classifier
    
    Inputs: predictions, (optional) filename
        If filename isn't specified, then it just uses the current time
    
    Outputs: Does not return anything
        Writes the submisssion to a textfile that should have the same format as the sample_submission.txt
    '''
    
    # Creates a new submissions folder if one doesn't exist
    pathlib.Path('submissions').mkdir(parents=True, exist_ok=True)
    with open("submissions\\%s" % filename, "w") as f:
        f.write("Id,Prediction\n")
        for Id, prediction in enumerate(data, 1):
            string = str(Id) + ',' + str(prediction) + '\n'
            f.write(string)
        
def percentError(yPred, yTrue):
    '''
    Calculates the percent error between two given label sets
    
    Inputs:
        yPred: predicted labels
        yTrue: actual labels
    
    Outputs:
        error: float of the number of mismatches divided by total length
    '''     
    return 1.0-np.sum(np.equal(yPred, yTrue))/len(yTrue)

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras import regularizers

from keras.utils import to_categorical

def main():
    # load the data
    X_train, y_train = load_data("training_data.txt")
    X_test, _ = load_data("test_data.txt", False)

    # normalize training and test data
    X_train_n, X_test_n = normalization(X_train, X_test)    

    # split the data in to training and testing so we can test ourselves
    trainX, trainY, testX, testY = split_data(X_train_n, y_train)

    y_binary = to_categorical(y_train)

    rate = 0.5
    model = Sequential()
    model.add(Dense(300, input_shape=(1000,), activation='sigmoid'))
    # model.add(Activation('sigmoid'))
    # model.add(BatchNormalization())
    model.add(Dropout(rate))

    model.add(Dense(300, activation='relu',
        kernel_regularizer=regularizers.l2(0.001)
        # activity_regularizer=regularizers.l2(0.005)
        ))
    # model.add(LeakyReLU(alpha=.01))
    # model.add(Dropout(rate))

    # output layer
    model.add(Dense(2, activation='softmax'))

    ## Printing a summary of the layers and weights in your model
    model.summary()

    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

    fit = model.fit(X_train_n, y_binary, batch_size=32, epochs=50, verbose=1)

    # testY_binary = to_categorical(testY)

    ## Printing the accuracy of our model, according to the loss function specified in model.compile above
    # score = model.evaluate(testX, testY_binary, verbose=0)
    # print('Test score:', score[0])
    # print('Test accuracy:', score[1])

    seq_binarypredictions = model.predict(X_test_n)
    save_data(seq_binarypredictions, "neuralbinarypreds.txt")

    seq_predictions = []
    for i in seq_binarypredictions:
        if i[0] > i[1]:
            seq_predictions.append(0)
        elif i[0] < i[1]:
            seq_predictions.append(1)
        else:
            print(i[0], i[1], "are equal!!!!\n\n")
    save_data(seq_predictions, "NeuralNetworkSubmission.txt")

if __name__ == '__main__':
    main()

'''
kernelreg l2=0.001
160, 160, 3         .852, 0.8297
160, 160, 10        .85475
160, 160, 30, 64    .84325

kernelreg l2=0.01
160, 160, 5         .8525, 0.85425
160, 160, 6         .855, 0.844

kernelreg l1=0.001
160, 160, 3         0.85325, 0.84075b, 0.8d
160, 160, 5         0.84375, 0.85025
160, 160, 10        .8515, 0.854
160, 160, 15        0.84375
160, 160, 20        .85075

activityreg l2=0.0001
160, 160, 15        .84525

d = 0
150, 4, b           .846, .85125
150, 6, b           .854, .84975, .8465

200, 8              .847
200, 10             .84825
200, 12             .85175

d = 0.3
150, 4, d            .8565, .8535, .85625
150, 6, d            .849
150, 8, d            .852, .83875
150, 10, d           .8575

dropout = 0.5
150, 10, d           .85, 853, .84975
150, 10, d, separate softmax   .84225, .84725

200, 2, no d         .84975, .84575
200, 3, no d         .8385, .851
200, 4, no d         .8495, .85225

200, 8, d            .85575, .855, .85, .8585, .84675
200, 8, no d         .847
200, 10, d           .847, .8505, .85425
200, 10, no d        .84825
200, 10, d, sep      .8525
200, 12, no d        .85175

130, 130, 10         .8465, .8505
130, 130, 12         .84625
130, 130, 15         .862, .851

130, 150, 10         .852, .8465, .84975
130, 150, 12         .85025, .85475, .8455

150, 130, 8          .854         
150, 130, 10         .82925, .852, .8425, .85175
150, 130, 11         .86425, .85325, .85625, .84575, .8435, .84625, .8515, 0.8525, .849
150, 130, 12         .85425, 0.86275, .8505, 0.85925, .8425, 0.85125
150, 130, 13         .852, 0.84475
150, 130, 15         .8455

150, 150, 10, 64     .83825, 0.851
150, 150, 13, 16     .851, 0.8565
150, 150, 14, 64     .849, 0.85475
150, 150, 14, 128    .84925
150, 150, 30, 16     .85025
150, 150, 30, 300    .8595, 0.836
150, 150, 45, 400    .851

150, 150, 8          .862, .85275, .8445, .8495, .847, .847, .8495, .834, .8465, 0.84775
150, 150, 10         .86025, .84425, .856, .85825, .852
150, 150, 11         .85175, .841, .85
150, 150, 12         .85175, .85175, .848
150, 150, 14         .86425, .8575
150, 150, 15         .851

150, 200, 10         .84925, .84875

160, 160, 10         .85975, 0.8485, .85325, 0.8445
160, 160, 12         .84

180, 180, 10         .85125, 0.846
180, 180, 12         .852, 0.84975
180, 180, 14         .848

200, 200, 8          .84475, .84725, .845
200, 200, 10         .85425, .84925, .8595, .8505, .847
200, 200, 12         .84875, .84675, .85375

more nodes = less epochs

leaky relu
12 epochs
100, 100            .8475
150, 150            .856, .83225, .84825, .841, .848
200, 200            .85825, .844, .85775, .859, .84925
300, 300            .84925
alpha = .03         .851   -reverted
sigmoid -> leaky    .8375  -rev
add sigmoid layer   .8535  -rev 
    delete dropout  .8515, .848
alpha = .3          .85625
add batchnorm       .852
epochs = 8          .85175, .85225, .852, .856, .85225, .85525, .85325
epochs = 10         .847
epochs = 9          .84975
epochs = 11         .849
'''