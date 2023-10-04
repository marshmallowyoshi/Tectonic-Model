import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import algorithm as alg


name = 'R2 H3'
name = 'data/' + name + '.csv'

df = pd.read_csv(name)
x = np.asarray(df.iloc[:,0])
y = np.asarray(df.iloc[:,3])


plt.plot(alg.resample_to_peak(x,y,sample_rate_override=5))
plt.show()
