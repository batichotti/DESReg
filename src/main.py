import numpy as np
import change_parameters

# from sklearn.svm import SVR
# from sklearn.linear_model import Lasso

def simulate(amount: int = 100) -> list:
    mse_list = []
    for i in range(100):
        mse_list.append(change_parameters.test(k=10, n_estimators_bag=100))
    return mse_list

    
if __name__ == "__main__":
    mse_list = simulate()

    print(mse_list)
    mse_numpy = np.array(mse_list)
    print("Média:", np.mean(mse_numpy))
    print("Mediana:", np.median(mse_numpy))
    print("Desvio padrão:", np.std(mse_numpy))
    print("Coeficiente de Variação:", (np.std(mse_numpy)/np.mean(mse_list))*100) # CV baixo (< 15%~20%) indica dispersão relativamente baixa
