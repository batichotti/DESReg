from desReg.des.DESRegression import DESRegression
from desReg.dataset import load_Student_Mark
from desReg.utils import measures

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

from os import path

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


def load_dataset(dataset='Student Mark'):
    dataset = dataset.lower()
    match dataset:
        case 'abalone':
            data = pd.read_csv(path.join(path.dirname(__file__), "../datasets", "abalone.csv"), low_memory=False)
        case 'airfoil':
            data = pd.read_csv(path.join(path.dirname(__file__), "../datasets", "airfoil_self_noise.csv"), low_memory=False)
        case 'bank8fm':
            data = pd.read_csv(path.join(path.dirname(__file__), "../datasets", "bank8FM.csv"), low_memory=False)
        case 'bank32nh':
            data = pd.read_csv(path.join(path.dirname(__file__), "../datasets", "bank32nh.csv"), low_memory=False)
        case 'ccpp':
            data = pd.read_csv(path.join(path.dirname(__file__), "../datasets", "CCPP.csv"), low_memory=False)
        case 'concrete':
            data = pd.read_csv(path.join(path.dirname(__file__), "../datasets", "concrete.csv"), low_memory=False)
        case 'delta elevators':
            data = pd.read_csv(path.join(path.dirname(__file__), "../datasets", "delta_elevators.csv"), low_memory=False)
        case 'housing':
            data = pd.read_csv(path.join(path.dirname(__file__), "../datasets", "housing.csv"), low_memory=False)
        case 'liver':
            data = pd.read_csv(path.join(path.dirname(__file__), "../datasets", "liver.csv"), low_memory=False)
        case 'machine':
            data = pd.read_csv(path.join(path.dirname(__file__), "../datasets", "machine.csv"), low_memory=False)
        case 'puma8nh':
            data = pd.read_csv(path.join(path.dirname(__file__), "../datasets", "puma8NH.csv"), low_memory=False)
        case 'puma32NH':
            data = pd.read_csv(path.join(path.dirname(__file__), "../datasets", "puma32NH.csv"), low_memory=False)
        case 'real estate':
            data = pd.read_csv(path.join(path.dirname(__file__), "../datasets", "real_estate.csv"), low_memory=False)
        case 'stock':
            data = pd.read_csv(path.join(path.dirname(__file__), "../datasets", "stock.csv"), low_memory=False)
        case 'student mark':
            data = pd.read_csv(path.join(path.dirname(__file__), "../datasets", "Student_marks.csv"), low_memory=False)
        case 'triazines':
            data = pd.read_csv(path.join(path.dirname(__file__), "../datasets", "triazines.csv"), low_memory=False)
        case 'wine quality red':
            data = pd.read_csv(path.join(path.dirname(__file__), "../datasets", "wine_quality_red.csv"), low_memory=False)
        case 'wine quality white':
            data = pd.read_csv(path.join(path.dirname(__file__), "../datasets", "wine_quality_white.csv"), low_memory=False)
        case 'yatch':
            data = pd.read_csv(path.join(path.dirname(__file__), "../datasets", "yatch.csv"), low_memory=False)
        case _:
            raise ValueError(f"Dataset '{dataset}' não suportado.")
    X = data.iloc[:, :-1].to_numpy()
    y = np.ravel(data.iloc[:, -1:])
    return X, y