from desReg.des.DESRegression import DESRegression
from desReg.dataset import load_Student_Mark
from desReg.utils import measures

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

from os import path, listdir

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

"""
Partitions with Datasets:

partition_name = './Datasets/Abalone/abalone-5-'
partition_name = './Datasets/Concrete/concrete-5-'
partition_name = './Datasets/Liver/liver-5-'
partition_name = './Datasets/Machine_CPU/machineCPU-5-'
partition_name = './Datasets/Real_estate/Real_estate-5-'
partition_name = './Datasets/Student_marks/student_marks-5-' 
partition_name = './Datasets/Wine_quality_red/winequality-red-5-'
partition_name = './Datasets/Wine_quality_white/winequality-white-5-'
partition_name = './Datasets/Yacht/yacht_hydrodynamics-5-'
"""
def test(X_train, y_train, X_test, y_test,
                   regressors_list=None,
                   n_estimators_bag=10,
                   random_state=None,
                   DSEL_perc=0.95,
                   XTRAIN_full=True,
                   n_jobs=-1,
                   k=5,
                   distance=distance.euclidean,
                   competence_region='knn',
                   competence_level=measures.all_errors,
                   regressor_selection=np.mean,
                   aggregation_method=np.mean,
                   ensemble_type='DES') -> float:

    heterogeneous_DES = DESRegression(
        regressors_list=regressors_list,
        n_estimators_bag=n_estimators_bag,
        random_state=random_state,
        DSEL_perc=DSEL_perc,
        XTRAIN_full=XTRAIN_full,
        n_jobs=n_jobs,
        k=k,
        distance=distance,
        competence_region=competence_region,
        competence_level=competence_level,
        regressor_selection=regressor_selection,
        aggregation_method=aggregation_method,
        ensemble_type=ensemble_type
    )

    heterogeneous_DES.fit(X_train, y_train)
    y_pred = heterogeneous_DES.predict(X_test)
    return mean_squared_error(y_test, y_pred)


def load_dataset(dataset='abalone'):
    dataset = dataset.lower()
    datasets_dir = path.join(path.dirname(__file__), "../datasets")
    dataset_files = {file[:-4].lower(): path.join(datasets_dir, file) for file in listdir(datasets_dir) if file.endswith('.csv')}
    
    if dataset not in dataset_files:
        raise ValueError(f"Dataset '{dataset}' não suportado.")
    
    data = pd.read_csv(dataset_files[dataset], low_memory=False)
    X = data.iloc[:, :-1].to_numpy()
    y = np.ravel(data.iloc[:, -1:])
    return X, y
