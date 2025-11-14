import numpy as np
from sklearn.model_selection import RepeatedKFold
from change_parameters import load_dataset, test
from scipy.spatial import distance
from dataclasses import dataclass
import pandas as pd

import time

COMPETENCE_REGION_LIST = ['knn', 'cluster', 'output_profiles']
DISTANCE_HEURISTICS_LIST = [distance.braycurtis, distance.canberra, distance.chebyshev, distance.cityblock, distance.cosine, distance.euclidean, distance.minkowski, distance.sqeuclidean]

def make_distance_safe(dist_func):
    def safe_dist(u, v):
        u = np.asarray(u).ravel()
        v = np.asarray(v).ravel()
        return dist_func(u, v)
    return safe_dist


def simulate(repeats=2, n_splits=10, dataset='student_mark', distance=DISTANCE_HEURISTICS_LIST[0], competence_region=COMPETENCE_REGION_LIST[0]) -> list:
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

@dataclass
class metrics:
    competence: str
    distance: str
    mean: float = 0.0
    median: float = 0.0
    std: float = 0.0
    cv: float = 0.0
    duration: float = 0.0

if __name__ == "__main__":

    metrics_list = []

    for dt in DISTANCE_HEURISTICS_LIST:
        for cr in COMPETENCE_REGION_LIST:
            print("=" * 30)
            print(f"CR: {cr} & D: {str(dt).split(' ')[1].capitalize()}")

            start_time = time.time()
            mse_list = simulate(repeats=2, n_splits=10, dataset='student_marks', distance=dt, competence_region=cr)
            duration = time.time() - start_time

            # print(mse_list)
            mse_numpy = np.array(mse_list)
            mean_val = np.mean(mse_numpy)
            median_val = np.median(mse_numpy)
            std_val = np.std(mse_numpy)
            cv_val = (std_val / mean_val) * 100 if mean_val != 0 else np.nan

            metrics_list.append(metrics(
                competence=cr,
                distance=str(dt).split(' ')[1].capitalize(),
                mean=mean_val,
                median=median_val,
                std=std_val,
                cv=cv_val,
                duration=duration
            ))

            print("Média:", mean_val)
            print("Mediana:", median_val)
            print("Desvio padrão:", std_val)
            print("Coeficiente de Variação (%):", cv_val)
            print("Duração (s):", duration)
            print("=" * 30)
            print("\n")

    df = pd.DataFrame([m.__dict__ for m in metrics_list])
    csv_path = "metrics_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Metrics salvo em: {csv_path}")
