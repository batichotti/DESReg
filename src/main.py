import numpy as np
from sklearn.model_selection import RepeatedKFold
from change_parameters import load_dataset, test
from scipy.spatial import distance
from dataclasses import dataclass
import pandas as pd

from time import time
from tqdm import tqdm

COMPETENCE_REGION_LIST = ['knn', 'cluster', 'output_profiles']
DISTANCE_HEURISTICS_LIST = [distance.braycurtis, distance.canberra, distance.chebyshev, distance.cityblock, distance.cosine, distance.euclidean, distance.minkowski, distance.sqeuclidean]
DATASETS_LIST = ['student_marks', 'liver', 'machine', 'yatch', 'housing', 'real_estate', 'concrete', 'trianzines', 'stock', 'airfoild', 'wine_quality_red', 'abalone', 'wine_quality_white', 'ccpp', 'delta_elevators', 'bank8fm', 'puma8nh', 'puma32h', 'bank32nh']

def make_distance_safe(dist_func):
    def safe_dist(u, v):
        u = np.asarray(u).ravel()
        v = np.asarray(v).ravel()
        return dist_func(u, v)
    return safe_dist


def simulate(repeats=2, n_splits=10, dataset='student_marks', distance=DISTANCE_HEURISTICS_LIST[0], competence_region=COMPETENCE_REGION_LIST[0]) -> list:
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
class Metrics:
    dataset: str
    competence: str
    distance: str
    mean: float = 0.0
    median: float = 0.0
    std: float = 0.0
    cv: float = 0.0
    fit_time: float = 0.0
    predict_time: float = 0.0
    duration: float = 0.0

if __name__ == "__main__":

    metrics_list = []

    for dataset in tqdm(DATASETS_LIST, desc="Datasets"):
        for dt in tqdm(DISTANCE_HEURISTICS_LIST, desc="Distances", leave=False):
            for cr in tqdm(COMPETENCE_REGION_LIST, desc="Competence Region Heuristics", leave=False):
                print('\n\n\n')
                print("=" * 30)
                print(f"DATA: {dataset} -> CR: {cr} & D: {str(dt).split(' ')[1].capitalize()}")

                start_time = time()
                results = simulate(repeats=2, n_splits=10, dataset=dataset, distance=dt, competence_region=cr)
                duration = time() - start_time

                mse_vals = [r["mse"] for r in results]
                fit_times = [r["fit_time"] for r in results]
                predict_times = [r["predict_time"] for r in results]

                mean_val = np.mean(mse_vals)
                median_val = np.median(mse_vals)
                std_val = np.std(mse_vals)
                cv_val = (std_val / mean_val) * 100 if mean_val != 0 else np.nan

                metrics_list.append(Metrics(
                    dataset=dataset,
                    competence=cr,
                    distance=str(dt).split(' ')[1].capitalize(),
                    mean=mean_val,
                    median=median_val,
                    std=std_val,
                    cv=cv_val,
                    fit_time=np.sum(fit_times),
                    predict_time=np.sum(predict_times),
                    duration=duration,
                ))

                print("Média MSE:", mean_val)
                print("Mediana MSE:", median_val)
                print("Desvio padrão MSE:", std_val)
                print("Coeficiente de Variação MSE (%):", cv_val)
                print("Tempo de Fit (s):", np.sum(fit_times))
                print("Tempo de Predict (s):", np.sum(predict_times))
                print("Duração total (s):", duration)
                print("=" * 30)

    df = pd.DataFrame([m.__dict__ for m in metrics_list])
    csv_path = "metrics_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Metrics salvo em: {csv_path}")
