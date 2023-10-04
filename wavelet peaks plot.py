import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import algorithm as alg

# Read data
df = pd.read_csv('data/R2 H3.csv')
x = np.asarray(df.iloc[:,0])
y = np.asarray(df.iloc[:,3])

contWav = alg.wavelet_generator(x,y)
dips = [list(find_peaks(wavelength*-1)[0]) for wavelength in contWav[0]]
peaks = [list(find_peaks(wavelength)[0]) for wavelength in contWav[0]]
# Plot
fig, ax = plt.subplots(1,1)
ax.imshow(contWav[0], aspect='auto', origin='lower', cmap='gray')
for idx, (peakband, dipband) in enumerate(zip(peaks, dips)):
    ax.scatter(peakband, np.full(len(peakband), idx), color='red', s=30, marker='+', linewidths=0.7)
    ax.scatter(dipband, np.full(len(dipband), idx), color='blue', s=30, marker='_')
plt.show()
