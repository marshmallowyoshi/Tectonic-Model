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
contWav = alg.wavelet_generator(x,y, sampling_period=1)
contWav = list(contWav)
contWav[0] = np.flip(contWav[0], 0)
contWav[1] = contWav[1][::-1]
fig, ax = alg.wavelet_plot_3d(contWav)
ax.set_ylabel('Distance (m)')
ax.set_xlabel('Wavelet Bands')
ax.set_zlabel('Amplitude')

plt.show()
