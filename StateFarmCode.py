import pandas as pd
import numpy as np
from IPython.display import display
from sklearn.cross_validation import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV
from math import sqrt
from six.moves import cPickle as pickle

seed = 42

data = pd.read_csv("Data for Cleaning & Modeling.csv")
train_num = data.shape[0]
data2 = pd.read_csv("Holdout for Testing.csv")

# Concat training and test data
data = pd.concat([data, data2], axis=0)

# Drop the data instance with all NaN values at #364111
index = data['X11'].index[data['X11'].apply(pd.isnull)]
data.drop(index, axis=0, inplace=True)
#pd.set_option("display.max_columns",50)

# Convert % and $ data to float
data[['X1','X4','X5','X6','X30']] = data[['X1','X4','X5','X6','X30']].replace('[$,%]','',regex=True).astype(float)

# Convert X7 to boolean
data['X7'] = data['X7'].replace([' 36 months', ' 60 months'], [0, 1]).astype('bool')

# Create Dummies for X8
data['X8'].fillna('OTHER', inplace=True)
data_X8 = pd.get_dummies(data['X8'], prefix='X8', drop_first=True).astype('bool')

# Convert X11 to int
data['X11'] = data['X11'].map({'< 1 year': 0, '1 year': 1, '2 years': 2,
                               '3 years': 3, '4 years': 4, '5 years': 5,
                               '6 years': 6, '7 years': 7, '8 years': 8,
                               '9 years': 9, '10+ years': 10, 'n/a': 0}).astype(float)

# Create Dummies for X12
data['X12'].fillna('OTHER', inplace=True)
data_X12 = pd.get_dummies(data['X12'], prefix='X12', drop_first=True).astype('bool')

# Fill NaN in X13 with the median
data['X13'].fillna(data['X13'].median(), inplace=True)

# Create Dummies for X14
data['X14'] = data['X14'].replace(['NONE', 'ANY'], ['OTHER', 'OTHER'])
data['X14'] = data['X14'].replace(['not verified','VERIFIED - income source','VERIFIED - income'], ['NV','VIS','VI'])
data_X14 = pd.get_dummies(data['X14'], prefix='X14', drop_first=True).astype('bool')

# Create Dummies for X17
data_X17 = pd.get_dummies(data['X17'], prefix='X17', drop_first=True).astype('bool')

# Create Dummies for X25, X26
data['X25'].loc[data['X25'] > 0] = 'NONZERO'
data['X25'].loc[data['X25'] == 0] = 'ZERO'
data['X25'].fillna('NA', inplace=True)
data_X25 = pd.get_dummies(data['X25'], prefix='X25', drop_first=True).astype('bool')
data['X26'].loc[data['X26'] > 0] = 'NONZERO'
data['X26'].loc[data['X26'] == 0] = 'ZERO'
data['X26'].fillna('NA', inplace=True)
data_X26 = pd.get_dummies(data['X26'], prefix='X26', drop_first=True).astype('bool')

# Fill NaN of X30 with 0
data['X30'].fillna(0, inplace=True)

# Convert X32 to boolean
data['X32'] = data['X32'].replace(['f', 'w'], [0, 1]).astype('bool')

# Drop unused columns
data.drop(['X2','X3','X8','X9','X10','X12','X14','X15','X16','X17','X18','X19','X20','X23','X25','X26'],
          axis=1, inplace=True)

# Concat Dummies
data = pd.concat([data, data_X8, data_X12, data_X14, data_X17, data_X25, data_X26], axis=1)

y = data['X1']
data.drop('X1', axis=1, inplace=True)

# Scale data columns with float dtypes
scaler = StandardScaler()
data_float = data.select_dtypes(include=['float64'])
data[data_float.columns] = pd.DataFrame(scaler.fit_transform(data_float), columns=data_float.columns)

# Split test_data
X_test = data[train_num-1:]
data_all = data[:train_num-1]
labels_all = y[:train_num-1]

# Drop all data instances with missing labels X1
data_all = data_all[pd.notnull(labels_all)].reset_index(drop=True)
labels_all = labels_all[pd.notnull(labels_all)].reset_index(drop=True)

# Split train_data, valid_data
train_index, valid_index = next(iter(ShuffleSplit(labels_all.shape[0],test_size=0.2,random_state=seed)))
X_train, X_valid = np.array(data_all)[train_index,:], np.array(data_all)[valid_index,:]
y_train, y_valid = np.array(labels_all)[train_index], np.array(labels_all)[valid_index]


def train_predict(clf, X_train, y_train, X_test, y_test):
    print "Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train))
    clf.fit(X_train, y_train)
    print "RMSE score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train))
    print "RMSE score for validation set: {:.4f}.".format(predict_labels(clf, X_test, y_test))
    
def predict_labels(clf, features, target):
    y_pred = clf.predict(features)
    return sqrt(mean_squared_error(target, y_pred))


'''
clf_A = ExtraTreesRegressor(random_state=seed)
clf_B = RandomForestRegressor(random_state=seed)
clf_C = GradientBoostingRegressor(random_state=seed)
clf_D = RidgeCV()
clf_E = LassoCV(random_state=seed)
clf_F = ElasticNetCV(random_state=seed)

for clf in [clf_A, clf_B, clf_C, clf_D, clf_E, clf_F]:
    train_predict(clf, X_train, y_train, X_valid, y_valid)'''
    

# RandomForestRegressor
parameters = {'n_estimators':(10,15,20),
              'min_samples_split':(2,3,4),
              'min_samples_leaf':(1,2,3)}

rfr = RandomForestRegressor(random_state=seed, warm_start=True)
score = make_scorer(mean_squared_error, greater_is_better=False)
grid_obj = GridSearchCV(rfr, param_grid=parameters, scoring=score, verbose=1, n_jobs=4, cv=5)
grid_obj= grid_obj.fit(X_train, y_train)
rfr = grid_obj.best_estimator_
print rfr.get_params(), '\n'
print "Tuned model has a training RMSE score of {:.4f}.".format(predict_labels(rfr, X_train, y_train))
print "Tuned model has a testing RMSE score of {:.4f}.".format(predict_labels(rfr, X_valid, y_valid))

# RidgeCV
ridge = RidgeCV(alphas=(1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0), cv=5)
ridge = ridge.fit(X_train, y_train)
print ridge.get_params(), '\n'
print "Tuned model has a training RMSE score of {:.4f}.".format(predict_labels(ridge, X_train, y_train))
print "Tuned model has a testing RMSE score of {:.4f}.".format(predict_labels(ridge, X_valid, y_valid))

# Save regressors
pickle_file = 'regressor.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'random_forest_regressor': rfr,
    'ridge': ridge,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise

# Load regressor
pickle_file = 'regressor.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  rfr = save['random_forest_regressor']
  ridge = save['ridge']
  del save

# Predict test_data
y_pred_rfr = rfr.predict(X_test)
y_pred_ridge = ridge.predict(X_test)
pd.DataFrame({'X1_Random_Forest': y_pred_rfr, 'X1_Ridge': y_pred_ridge}).to_csv('Results from Hang Yao.csv')

