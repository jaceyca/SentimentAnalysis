import numpy as np

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, RidgeClassifier, SGDClassifier
from sklearn.linear_model import LogisticRegressionCV, RidgeClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import BernoulliRBM, MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import GridSearchCV, ShuffleSplit



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





# attempt at blending?

# load the data
X_train, y_train = load_data("training_data.txt")
X_test, _ = load_data("test_data.txt", False)

# normalize training and test data
X_train_n, X_test_n = normalization(X_train, X_test)    

# split the data in to training and testing so we can test ourselves
trainX, trainY, testX, testY = split_data(X_train_n, y_train)

# PUT THE THINGS WE WANT TO BLEND HERE.
test1 = LogisticRegression(C=2.7825594)
# test2 = LogisticRegression(C=2.7825594)
# test2 = MLPClassifier(activation = 'logistic', hidden_layer_sizes=(300,))
test2 = SVC(gamma=1, C=2)
test3 = ExtraTreesClassifier(criterion='gini', max_depth=None, min_samples_split=2, n_estimators=1000)

blend = VotingClassifier(estimators=[('lr', test1), ('etc', test2), ('r', test3)], voting='hard')

for clf, label in zip([test1, test2, test3, blend], 
                      ['Logistic Regression', 'SVC Classifier', 'Extra Trees Classifier', 'LR_SVC_ET']):
    scores = cross_val_score(clf, X_train_n, y_train, cv=5, scoring = 'accuracy')
    print("Accuracy: %0.8f (+/- %0.8f) [%s]" % (scores.mean(), scores.std(), label))