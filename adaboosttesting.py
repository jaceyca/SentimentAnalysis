import numpy as np
import pathlib
from datetime import datetime

# classifiers
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, RidgeClassifier, SGDClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import BernoulliRBM, MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF
from sklearn.cross_validation import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

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

    abc = AdaBoostClassifier()
    parameters = {'n_estimators': [50, 100, 200, 300, 400, 1000], 'learning_rate': [0.1, 1, 2]}

    ABCSearchCV = GridSearchCV(abc, parameters, cv=fold)
    ABCSearchCV.fit(trainX, trainY)
    print(ABCSearchCV.cv_results_)
    print(ABCSearchCV.score(trainX, trainY))
    print()
    print(ABCSearchCV.score(testX, testY))
    print()

    print(clf.best_score_)
    print()
    print(clf.best_params_)
    print()
    bestn = ABCSearchCV.best_params_['n_estimators']
    bestlearningrate = clf.best_params_['learning_rate']

    abclf = AdaBoostClassifier(n_estimators=ABCSearchCV.best_params_['n_estimators'], 
    	learning_rate=ABCSearchCV.best_params_['best_params_'])

    ab = make_predictions(abclf, trainX, trainY, testX)
    print("Adaboost error:", percentError(ab, testY))

    absubmission = make_predictions(abclf, X_train_n, y_train, X_test_n)
    save_data(absubmission, "adaboostsubmission.txt")

if __name__ == '__main__':
    main()


"""
, 'params': [{'learning_rate': 0.1, 'n_estimators': 50}, {'learning_rate': 0.1, 'n_estimators': 100}, {'learning_rate': 0.1, 'n_estimators': 200}, {'learning_rate': 0.1, 'n_estimators': 300}, {'learning_rate': 0.1, 'n_estimators': 400}, {'learning_rate': 0.1, 'n_estimators': 1000}, {'learning_rate': 1, 'n_estimators': 50}, {'learning_rate': 1, 'n_estimators': 100}, {'learning_rate': 1, 'n_estimators': 200}, {'learning_rate': 1, 'n_estimators': 300}, {'learning_rate': 1, 'n_estimators': 400}, {'learning_rate': 1, 'n_estimators': 1000}, {'learning_rate': 2, 'n_estimators': 50}, {'learning_rate': 2, 'n_estimators': 100}, {'learning_rate': 2, 'n_estimators': 200}, {'learning_rate': 2, 'n_estimators': 300}, {'learning_rate': 2, 'n_estimators': 400}, {'learning_rate': 2, 'n_estimators': 1000}], 
    'split0_test_score': array([ 0.70040612,  0.75945017,  0.788816  ,  0.79506404,  0.80131209,
        0.82661668,  0.78819119,  0.80880975,  0.82099344,  0.82161824,
        0.82224305,  0.82099344,  0.50671665,  0.50671665,  0.50671665,
        0.50671665,  0.50671665,  0.50671665]), 
    'split1_test_score': array([ 0.694375 ,  0.7540625,  0.785    ,  0.7996875,  0.8053125,
        0.8296875,  0.7934375,  0.8075   ,  0.823125 ,  0.828125 ,
        0.8253125,  0.825    ,  0.5090625,  0.5090625,  0.5090625,
        0.5090625,  0.5090625,  0.5090625]), 
    'split2_test_score': array([ 0.701875 ,  0.7665625,  0.7903125,  0.798125 ,  0.8009375,
        0.8209375,  0.78875  ,  0.8009375,  0.8165625,  0.821875 ,
        0.820625 ,  0.81625  ,  0.4053125,  0.4053125,  0.4053125,
        0.4053125,  0.4053125,  0.4053125]), 
    'split3_test_score': array([ 0.711875 ,  0.7796875,  0.805    ,  0.8128125,  0.8203125,
        0.84     ,  0.804375 ,  0.8165625,  0.8278125,  0.8296875,
        0.8321875,  0.8346875,  0.39875  ,  0.39875  ,  0.39875  ,
        0.39875  ,  0.39875  ,  0.39875  ]), 
    'split4_test_score': array([ 0.72460144,  0.76023757,  0.79337293,  0.8105658 ,  0.81150359,
        0.82775867,  0.79837449,  0.81931854,  0.82244451,  0.82900907,
        0.8240075 ,  0.81838074,  0.39949984,  0.39949984,  0.39949984,
        0.39949984,  0.39949984,  0.50734605]), 
    'mean_test_score': array([ 0.706625 ,  0.764    ,  0.7925   ,  0.80325  ,  0.807875 ,
        0.829    ,  0.794625 ,  0.810625 ,  0.8221875,  0.8260625,
        0.824875 ,  0.8230625,  0.443875 ,  0.443875 ,  0.443875 ,
        0.443875 ,  0.443875 ,  0.4654375]), 
    'std_test_score': array([ 0.01060118,  0.0087895 ,  0.00680622,  0.00708426,  0.00728882,
        0.00622413,  0.00610654,  0.00659875,  0.0036236 ,  0.00355997,
        0.00398518,  0.00650383,  0.05232894,  0.05232894,  0.05232894,
        0.05232894,  0.05232894,  0.05181825]), 
    'rank_test_score': array([12, 11, 10,  8,  7,  1,  9,  6,  5,  2,  3,  4, 14, 14, 14, 14, 14,
       13], dtype=int32), 
    'split0_train_score': array([ 0.70575826,  0.76779436,  0.79959372,  0.81232909,  0.82107977,
        0.8563169 ,  0.80615673,  0.83053364,  0.85647316,  0.86952106,
        0.87991249,  0.92796312,  0.50738339,  0.50738339,  0.50738339,
        0.50738339,  0.50738339,  0.50738339]), 
    'split1_train_score': array([ 0.70796875,  0.76890625,  0.80203125,  0.81054688,  0.82046875,
        0.85476562,  0.80679687,  0.82921875,  0.86015625,  0.8734375 ,
        0.88617188,  0.92554687,  0.50757812,  0.50757812,  0.50757812,
        0.50757812,  0.50757812,  0.50757812]), 
    'split2_train_score': array([ 0.715     ,  0.7775    ,  0.8       ,  0.81367188,  0.82375   ,
        0.8565625 ,  0.80664062,  0.83367187,  0.85984375,  0.87507813,
        0.88679688,  0.92890625,  0.40015625,  0.40015625,  0.40015625,
        0.40015625,  0.40015625,  0.40015625]), 
    'split3_train_score': array([ 0.70289062,  0.77632813,  0.79945312,  0.81101563,  0.82070312,
        0.85195312,  0.80476563,  0.82742187,  0.85445313,  0.871875  ,
        0.88171875,  0.92671875,  0.40289063,  0.40289063,  0.40289063,
        0.40289063,  0.40289063,  0.40289063]), 
    'split4_train_score': array([ 0.70908523,  0.76025311,  0.79892196,  0.81329584,  0.82251387,
        0.85337083,  0.80759316,  0.83040387,  0.85907351,  0.86883837,
        0.88329037,  0.9247715 ,  0.40582767,  0.40582767,  0.40582767,
        0.40582767,  0.40582767,  0.507226  ]), 
    'mean_train_score': array([ 0.70814057,  0.77015637,  0.80000001,  0.81217186,  0.8217031 ,
        0.8545938 ,  0.8063906 ,  0.83025   ,  0.85799996,  0.87175001,
        0.88357807,  0.9267813 ,  0.44476721,  0.44476721,  0.44476721,
        0.44476721,  0.44476721,  0.46504688]), 
    'std_train_score': array([ 0.00403016,  0.0062807 ,  0.00107259,  0.00122601,  0.00124573,
        0.00175175,  0.00093477,  0.00204329,  0.00219603,  0.00234032,
        0.00261017,  0.0015152 ,  0.05123684,  0.05123684,  0.05123684,
        0.05123684,  0.05123684,  0.051874  ])}

Best params: learning rate = 0.1, n = 1000
2nd Best: learning rate = 1, n = 300

Training accuracy, best params.
0.8506875

Testing accuracy, best params.
0.83075
"""