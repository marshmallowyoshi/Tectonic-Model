import numpy as np
import pandas as pd
import algorithm as alg

name = 'R2 H3.csv'
df = pd.read_csv(name)
x = np.asarray(df.iloc[:,0])
y = np.asarray(df.iloc[:,3])


alg.findMinimumSampleRate(x, y)
