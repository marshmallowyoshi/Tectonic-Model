import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import algorithm as alg

name = 'BF1 H2'
name = 'data/' + name + '.csv'

df = pd.read_csv(name)

x = np.asarray(df.iloc[:,0])
y = np.asarray(df.iloc[:,3])

for i in np.arange(4, len(x)-1, 1):
    a, b = alg.resampleToPeak(x,y, sample_rate_override=i, kind='quadratic')
    x2, y2 = alg.resampleToPeak(a, b, sample_rate_override=40, kind='quadratic')
    # x3, y3 = alg.resampleToPeak(x,y, sample_rate_override=i, kind='linear')

    plt.plot(x,y, label='original')
    plt.plot(x2,y2, label='quadratic')
    # plt.plot(x3,y3, label='linear')
    plt.legend()
    plt.savefig('resampling\\' + str(i) + '.png')  # save the figure to file
    plt.close()
