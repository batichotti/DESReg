import numpy as np
from sklearn.model_selection import RepeatedKFold
from change_parameters import load_dataset, test
from scipy.spatial import distance

COMPETENCE_REGION_LIST = ['knn', 'cluster', 'output_profiles']
DISTANCE_HEURISTICS_LIST = [distance.braycurtis, distance.canberra, distance.chebyshev, distance.cityblock, distance.cosine, distance.euclidean, distance.minkowski, distance.sqeuclidean]

def make_distance_safe(dist_func):
    def safe_dist(u, v):
        u = np.asarray(u).ravel()
        v = np.asarray(v).ravel()
        return dist_func(u, v)
    return safe_dist


def simulate(repeats=2, n_splits=10, dataset='Student Mark', distance=DISTANCE_HEURISTICS_LIST[0], competence_region=COMPETENCE_REGION_LIST[0]) -> list:
    X, y = load_dataset(dataset)
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=repeats, random_state=42)
    mse_list = []

    for train_index, test_index in rkf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        mse = test(
            X_train, y_train, X_test, y_test,
            k=10,
            n_estimators_bag=100,
            distance=make_distance_safe(distance),
            competence_region=competence_region
        )
        mse_list.append(mse)

    return mse_list

if __name__ == "__main__":
    for dt in DISTANCE_HEURISTICS_LIST:
        for cr in COMPETENCE_REGION_LIST:
            print("="*30)
            print(f"CR: {cr} & D: {dt}")
            mse_list = simulate(repeats=2, n_splits=10, dataset='Student Mark', distance=dt, competence_region=cr)
            
            # print("MSEs:", mse_list)
            mse_numpy = np.array(mse_list)
            print("Média:", np.mean(mse_numpy))
            print("Mediana:", np.median(mse_numpy))
            print("Desvio padrão:", np.std(mse_numpy))
            print("Coeficiente de Variação (%):", (np.std(mse_numpy) / np.mean(mse_numpy)) * 100) # CV baixo (< 15%~20%) indica dispersão relativamente baixa
            print("="*30)
            print("\n")
