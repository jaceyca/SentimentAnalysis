import numpy as np
import pathlib
from datetime import datetime

# classifiers
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, RidgeClassifier, SGDClassifier
from sklearn.linear_model import LogisticRegressionCV, RidgeClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import BernoulliRBM, MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
# import matplotlib.pyplot as plt


from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score


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


def main():
    # attempt at blending?

    # load the data
    X_train, y_train = load_data("training_data.txt")
    X_test, _ = load_data("test_data.txt", False)

    # normalize training and test data
    X_train_n, X_test_n = normalization(X_train, X_test)    

    # split the data in to training and testing so we can test ourselves
    trainX, trainY, testX, testY = split_data(X_train_n, y_train)

    # PUT THE THINGS WE WANT TO BLEND HERE.
    logclf = LogisticRegression(C=2.7825594)
    SVCclf = SVC(gamma=1, C=2, probability = True)

    # test3 = MLPClassifier(activation = 'logistic', hidden_layer_sizes=(300,))

    # gnbclf = GaussianNB()


    etclf = ExtraTreesClassifier(n_estimators=800, n_jobs=-1, verbose=2)
    adaclf = AdaBoostClassifier(base_estimator=etclf)

    # can add weights to this
    votingclf = VotingClassifier(estimators=[('ada', adaclf),
        ('svc', SVCclf),
        # ('mlp', mlpclf),
        # ('nb', gnbclf),
        ('log', logclf)
        ], voting='hard')

    # print('Fitting to training data...')
    # voting = make_predictions(votingclf, trainX, trainY, testX)
    # print("Voting error:", percentError(voting, testY))

    print('Fitting to testing data...')
    try:
        votingsubmission = make_predictions(votingclf, X_train_n, y_train, X_test_n)
    except Exception as e:
        print(e)
    save_data(votingsubmission, "votingsubmission2.txt")

    print('All done! \n')
    # for clf, label in zip([test1, test2, test3, blend], 
    #                       ['Logistic Regression', 'LogReg', 'MLPClass', 'Ensemble']):
    #     # scores = cross_val_score(clf, X_train_n, y_train, cv=5, scoring = 'accuracy')
    #     print("Accuracy: %0.8f (+/- %0.8f) [%s]" % (scores.mean(), scores.std(), label))

    # # seq_predictions, ridgePred, log
    # # consideration of 3 scores
    # togPred = []
    # for index, element in enumerate(seq_predictions):
    #     if (element + ridgePred[index] + log[index]) > 1:
    #         togPred.append(1)
    #     else: 
    #         togPred.append(0)
            
    # # save_data(togPred)

if __name__ == '__main__':
    main()
