import numpy as np
import pandas as pd
import algorithm as alg
import matplotlib.pyplot as plt

name = 'R2 H3'
name = 'data/' + name + '.csv'
df = pd.read_csv(name)
x = np.asarray(df.iloc[:,0])
y = np.asarray(df.iloc[:,3])

############################################################################################################################################

print(alg.findMinimumSampleRate(x,y, 
                        plot=True, 
                        verbose=True, 
                        peakRateThreshold=0.03, 
                        samples_per_period=1300, 
                        sampling_period=1, 
                        bands=(1,31),))