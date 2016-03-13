import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

def preprocess_features(X):
    outX = pd.DataFrame(index=X.index)  # output dataframe, initially empty

    # Check each column
    for col, col_data in X.iteritems():
        # If data type is non-numeric, try to replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])
        # Note: This should change the data type for yes/no columns to int

        # If still non-numeric, convert to one or more dummy variables
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix=col)  # e.g. 'school' => 'school_GP', 'school_MS'

        outX = outX.join(col_data)  # collect column(s) in output dataframe

    return outX

def train_classifier(clf, X_train, y_train):
    print "Training {}...".format(clf.__class__.__name__)
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    print "Done!\nTraining time (secs): {:.3f}".format(end - start)

def predict_labels(clf, features, target):
    print "Predicting labels using {}...".format(clf.__class__.__name__)
    start = time.time()
    y_pred = clf.predict(features)
    end = time.time()
    print "Done!\nPrediction time (secs): {:.3f}".format(end - start)
    accuracy=accuracy_score(target, y_pred)
    print 'accuracy:',accuracy
    return f1_score(target.values, y_pred, pos_label='yes')

student_data = pd.read_csv("student-data.csv")

n_students = student_data['passed'].shape[0]
n_features = len(student_data.columns)-1
n_passed = (student_data['passed'][student_data['passed']=='yes']).shape[0]
n_failed = (student_data['passed'][student_data['passed']=='no']).shape[0]
grad_rate = 100*(float(n_passed)/float(n_students))
print "Total number of students: {}".format(n_students)
print "Number of students who passed: {}".format(n_passed)
print "Number of students who failed: {}".format(n_failed)
print "Number of features: {}".format(n_features)
print "Graduation rate of the class: {:.2f}%".format(grad_rate)

feature_cols = list(student_data.columns[:-1])  # all columns but last are features
target_col = student_data.columns[-1]  # last column is the target/label
print "Feature column(s):-\n{}".format(feature_cols)
print "Target column: {}".format(target_col)

X_all = student_data[feature_cols]  # feature values for all students
y_all = student_data[target_col]  # corresponding targets/labels
print "\nFeature values:-"
print X_all.head()  # print the first 5 rows

num_all = student_data.shape[0]  # same as len(student_data)
num_train = 300  # about 75% of the data
num_test = num_all - num_train

# First, decide how many training vs test samples you want
num_all = student_data.shape[0]  # same as len(student_data)
num_train = 300  # about 75% of the data
num_test = num_all - num_train

# TODO: Then, select features (X) and corresponding labels (y) for the training and test sets
# Note: Shuffle the data or randomly select samples to avoid any bias due to ordering in the dataset
X_train = X_all.sample(num_train)
y_train = (y_all.loc[X_all.index.isin(X_train.index)])
X_test =X_all.loc[~X_all.index.isin(X_train.index)]
y_test = y_all.loc[~X_all.index.isin(X_train.index)]

X_train=X_train.as_matrix()
y_train=y_train.values.tolist()
X_test=X_test.as_matrix()
y_test=y_test.values.tolist()

clf = DecisionTreeClassifier(random_state=0)
train_classifier(clf, X_train, y_train)

train_f1_score = predict_labels(clf, X_train, y_train)
print "F1 score for training set: {}".format(train_f1_score)

#print X_train
#raw_input()
'''
y_train = ?
X_test = ?
y_test = ?
print "Training set: {} samples".format(X_train.shape[0])
print "Test set: {} samples".format(X_test.shape[0])
# Note: If you need a validation set, extract it from within training data


'''

#X_all = preprocess_features(X_all)
#print "Processed feature columns ({}):-\n{}".format(len(X_all.columns), list(X_all.columns))
#raw_input()

'''
for i in range(student_data['passed'].shape[0]):
    if(student_data['passed'][i]=='yes'):
        student_data.loc[i,'passed']=1
    else:
        student_data.loc[i,'passed']=0
'''