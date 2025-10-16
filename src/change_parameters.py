from desReg.des.DESRegression import DESRegression
from desReg.dataset import load_Student_Mark
from desReg.utils import measures

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.linear_model import Lasso

from scipy.spatial import distance

""" 
Standart Hyperparameters:

regressors_list = None (DecisionTreeRegressor), 
n_estimators_bag = 10,
random_state= None,
DSEL_perc = 0.95, 
XTRAIN_full = True,
n_jobs= -1,
k = 5,
distance = scipy.spatial.distance.euclidean, 
competence_region = 'knn',
competence_level= desReg.utils.measures.all_errors,
regressor_selection= np.mean,
aggregation_method= np.mean,
ensemble_type= 'DES'
"""
def test(regressors_list = None, 
     n_estimators_bag = 10,
     random_state= None,
     DSEL_perc = 0.95, 
     XTRAIN_full = True,
     n_jobs= -1,
     k = 5,
     distance = distance.euclidean, 
     competence_region = 'knn',
     competence_level= measures.all_errors,
     regressor_selection= np.mean,
     aggregation_method= np.mean,
     ensemble_type= 'DES',
     dataset = 'Student Mark') -> float:
     
     data = pd.DataFrame
     
     match dataset:
          case 'Student Mark':     
               data = load_Student_Mark()
     
     X = data.iloc[:,:-1].to_numpy()
     y = np.ravel(data.iloc[:, -1:]) 
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

     heterogeneous_DES = DESRegression(
          regressors_list=regressors_list,
          n_estimators_bag = n_estimators_bag,
          random_state= random_state,
          DSEL_perc = DSEL_perc, 
          XTRAIN_full = XTRAIN_full,
          n_jobs= n_jobs,
          k = k,
          distance = distance, 
          competence_region = competence_region,
          competence_level= competence_level,
          regressor_selection= regressor_selection,
          aggregation_method= aggregation_method,
          ensemble_type= ensemble_type
          )

     heterogeneous_DES.fit(X_train, y_train)
     y_pred = heterogeneous_DES.predict(X_test)
     # print('MSE error:', mean_squared_error(y_test, y_pred))
     return mean_squared_error(y_test, y_pred)
