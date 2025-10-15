import numpy as np
import change_parameters

mse_list = []
for i in range(100):
    mse_list.append(change_parameters.test())

print(mse_list)
mse_numpy = np.array(mse_list)
print("Média:", np.mean(mse_numpy))
print("Mediana:", np.median(mse_numpy))
print("Desvio padrão:", np.std(mse_numpy))
print("Coeficiente de Variação:", (np.std(mse_numpy)/np.mean(mse_list))*100) # CV baixo (< 15%~20%) indica dispersão relativamente baixa
