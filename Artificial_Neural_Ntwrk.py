import numpy as np
import pandas as pnd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import *
from sklearn.metrics import confusion_matrix
from datetime import datetime as dt

######################################### Part-1 Data Pre-processing ###########################################

# Import the dataset
data_set = pnd.read_csv('Churn_Modelling.csv')
X = data_set.iloc[:, 3:13].values
Y = data_set.iloc[:, 13].values

# Encoding the categorical data
labelEncoder_x1 = LabelEncoder()
X[:, 1] = labelEncoder_x1.fit_transform(X[:, 1])
labelEncoder_x2 = LabelEncoder()
X[:, 2] = labelEncoder_x2.fit_transform(X[:, 2])
oneHotEncoder = OneHotEncoder(categorical_features=[1])
X = oneHotEncoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the set into training set and testing set
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

# Scaling the variables
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

############################################## Part - 2 Make ANN ###############################################
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import cross_val_score, GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier


def build_classifier(optmizr,U1,U2):
    classifier = Sequential()
    # Initialize the ANN and adding the input layer and hidden layers
    """Makes the architecture of neural network"""
    classifier.add(Dense(kernel_initializer='uniform', activation='relu', input_dim=11, units=U1))
    classifier.add(Dropout(rate = 0.1))
    classifier.add(Dense(kernel_initializer='uniform', activation='relu', units=U2))
    classifier.add(Dropout(rate =0.1))
    # classifier.add(Dense(kernel_initializer='uniform', activation='relu', units=5))
    classifier.add(Dense(kernel_initializer='uniform', activation='sigmoid', units=1))
    classifier.compile(optimizer=optmizr, loss='binary_crossentropy', metrics=['accuracy'])
    # use categorical_crossentropy for categorial results
    return classifier

# Fitting the ANN to the training set
start_time = dt.now()

"""METHOD 2: Using the GridSearch. It allows us to get the best configuration of
parameters."""
classifier = KerasClassifier(build_fn= build_classifier)
parameters = {'batch_size':[10, 15, 30],
              'optmizr': ['adam','rmsprop'],
              'nb_epoch':[100, 250, 450],
              'U1':[7,8,9,10], 'U2':[7,8,9,10]}

grid_search = GridSearchCV(estimator = classifier, param_grid =parameters, scoring ='accuracy', cv=10, n_jobs=-1)
grid_search.fit(x_train, y_train)
best_parameters = grid_search.best_params_
best_score = grid_search.best_score_
print("\nBest parameters:", best_parameters)
print("\nBest score:", best_score)
end_time = dt.now()
difference = end_time - start_time
print("\nTime taken:", difference)

"""METHOD 1: Using the k-Fold Cross Validation technique""" 
# accuracies = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=10,n_jobs= 1)
# print("\n",accuracies)
# print("\nMEAN:",accuracies.mean())
# print("\nVARIANCE:",accuracies.std())
# Fitting the ANN to the training set
# classifier.fit(x_train,y_train, batch_size= 10, epochs= 100)

# Predicting the test set results
# y_prediction = classifier.predict(x_test)
# y_prediction = (y_prediction > 0.5)
#
#
# # Making the confusion matrix
# conMat = confusion_matrix(y_true=y_test, y_pred=y_prediction)
# print(conMat)
#
# # newPrediction = classifier.predict(scaler.transform(np.array([[0.0, 0, 600, 1, 49, 1, 6000, 1, 0, 0, 5000]])))
# # newPrediction = (newPrediction > 0.5)
# # print(newPrediction)
#
# ####### Evaluating and tuning the ANN #########
# """Done by changing the architecture and fitting the classifier using k-fold Cross Validation"""

