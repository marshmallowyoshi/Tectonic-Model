import algorithm as alg
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

name = 'R2 H3'
name = 'data/' + name + '.csv'

df = pd.read_csv(name)
x = np.asarray(df.iloc[:,0])
y = np.asarray(df.iloc[:,3])


plt.plot(alg.resampleToPeak(x,y,sample_rate_override=5))
plt.show()
