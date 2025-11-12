import numpy as np
from sklearn.model_selection import RepeatedKFold
from change_parameters import load_dataset, test_with_data
from scipy.spatial import distance

# from sklearn.svm import SVR
# from sklearn.linear_model import Lasso

COMPETENCE_REGION_LIST = ['knn', 'cluster', 'output_profiles']
DISTANCE_HEURISTICS_LIST = [distance.braycurtis, distance.canberra, distance.chebyshev, distance.cityblock, distance.cosine, distance.euclidean, distance.minkowski, distance.sqeuclidean]
    

def simulate(repeats=2, n_splits=10, dataset='Student Mark') -> list:
    X, y = load_dataset(dataset)
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=repeats, random_state=42)
    mse_list = []

    for train_index, test_index in rkf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        mse = test_with_data(
            X_train, y_train, X_test, y_test,
            k=10,
            n_estimators_bag=100,
            distance=distance.euclidean,
            competence_region='knn'
        )
        mse_list.append(mse)

    return mse_list

if __name__ == "__main__":
    mse_list = simulate(repeats=2, n_splits=10, dataset='Student Mark')

    print("MSEs:", mse_list)
    mse_numpy = np.array(mse_list)
    print("Média:", np.mean(mse_numpy))
    print("Mediana:", np.median(mse_numpy))
    print("Desvio padrão:", np.std(mse_numpy))
    print("Coeficiente de Variação (%):", (np.std(mse_numpy) / np.mean(mse_numpy)) * 100) # CV baixo (< 15%~20%) indica dispersão relativamente baixa
