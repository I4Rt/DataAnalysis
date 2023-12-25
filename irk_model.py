import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("node_54.csv",
                 delimiter=",", dtype=float)

dataset = []
y = []

for sheet in data:
    y.append((sheet[7] + 60)/ 120)
    dataset.append( (np.asarray([[
                                    sheet[0],
                                    sheet[1],
                                    sheet[2],
                                    sheet[3],
                                    sheet[4],
                                    sheet[5],
                                    sheet[6]
                                ]]),
                     np.asarray([(sheet[7] + 60)/ 120])
                     )
                   )
    
dataset = dataset[:len(dataset)*2//3]
dataset_test = dataset[-len(dataset)//3:]

plt.plot(y, label='target')
plt.show()


def softMinusOneToPlusOne(y, dif=False):
    if dif:
        return 1/(1+np.sqrt(y**2))**2
    return y/(1+np.sqrt(y**2))
    
from ClassModules import *

model = Sequential('adam', ALPHA=0.00015, type_='mean_squared_error')
model.add(Dense(72, linear, input_shape=7))
model.add(Dense(32, sin))
model.add(Dense(16, backLinear))
model.add(Dense(64, linear))
model.add(Dense(1, softMinusOneToPlusOne))

model.train(dataset, 1000, False, False, 512)

y_res = [model.predict(dataset_test[i][0])[0][0] for i in range(len(dataset_test))]


plt.plot([dataset_test[i][1] for i in range(len(dataset_test))], label='target')
plt.plot(y_res, label='result')
plt.legend(fontsize=16)
plt.minorticks_on()
plt.show()