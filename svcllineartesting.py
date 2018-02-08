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

def main():
    # load the data
    X_train, y_train = load_data("training_data.txt")
    X_test, _ = load_data("test_data.txt", False)

    # normalize training and test data
    X_train_n, X_test_n = normalization(X_train, X_test)    

    # split the data in to training and testing so we can test ourselves
    trainX, trainY, testX, testY = split_data(X_train_n, y_train)

    fold = KFold(len(trainY), n_folds=5, shuffle=True)

    gamma_list = [0.5, 1, 2, 4]
    C_list = [0.5, 1, 2, 4]
    # gamma_list = [0.1, 1, 10]
    # C_list = [0.01, 1, 100]
    # clfs = []
    parameters = {'C':C_list, 'gamma':gamma_list, 'kernel':['linear']}
    # lowestError = 1e10
    # bestC = None
    # bestg = None
    clf = GridSearchCV(SVC(), parameters, cv=fold, n_jobs=-1, verbose=2)
    clf.fit(trainX, trainY)

    print(clf.best_score_)
    print()
    print(clf.best_params_)
    print()
    bestC = clf.best_params_['C']
    bestgamma = clf.best_params_['gamma']
    # for C in C_list:
    #     for g in gamma_list:
    #         clf = SVC(C=C, gamma=g)
    #         preds = make_predictions(clf, trainX, trainY, testX)
    #         error = percentError(preds, testY)
    #         if error < lowestError:
    #             lowestError = error
    #             bestC = C
    #             bestg = g
    #         print("SVC error:", C, g, error)
    #         clfs.append((C, g, error, clf))

    # print(lowestError)
    SVclf = SVC(C=bestC, gamma=bestgamma, kernel='linear')

    SV = make_predictions(SVclf, trainX, trainY, testX)
    print("SVC linear error:", percentError(SV, testY))

    SVCsubmission = make_predictions(SVclf, X_train_n, y_train, X_test_n)
    save_data(SVCsubmission, "SVClinearsubmission.txt")


if __name__ == '__main__':
    main()


'''
Done  80 out of  80 | elapsed: 95.4min finished
0.8456875

{'C': 2, 'gamma': 0.5, 'kernel': 'linear'}

SVC linear error: 0.15775
'''