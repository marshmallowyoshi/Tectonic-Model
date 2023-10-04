import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import algorithm as alg

name = 'part4'
name = 'data/' + name + '.csv'

df = pd.read_csv(name)

x = np.asarray(df.iloc[:,0])
y = np.asarray(df.iloc[:,3])
x2, y2 = alg.resample_to_peak(x,y, sample_rate_override=31)
allSegments, segmentRegression = alg.full_segment_analysis(x2,y2,
                                                            plot=False,
                                                            skip_wavelets=True,
                                                            peak_rate_threshold=0.04,
                                                            samples_per_period=1300,
                                                            sampling_period=1,
                                                            bands=(1,31),
                                                            resampling_kind='quadratic',
                                                            join=True,
                                                            segment_mode='poly',
                                                            title='short',
                                                            name=name)
x3, y3 = alg.resample_to_peak(x,y, sample_rate_override=28)
allSegments2, segmentRegression2 = alg.full_segment_analysis(x3,y3,
                                                            plot=False,
                                                            skip_wavelets=True,
                                                            peak_rate_threshold=0.04,
                                                            samples_per_period=1300,
                                                            sampling_period=1,
                                                            bands=(1,31),
                                                            resampling_kind='quadratic',
                                                            join=True,
                                                            segment_mode='poly',
                                                            title='short',
                                                            name=name)
contWav = alg.wavelet_generator(x,y)
    
fig, ax = plt.subplots(1, 1)


ax.plot(x2,y2, color='red', label='Before')
ax.plot(x3,y3, color='blue', label='After')

fig.legend(bbox_to_anchor=(0.99,0.91), loc='upper right', ncol=1, borderaxespad=0., fontsize=10)
ax.set_xlabel('Sample Number')
ax.set_ylabel('Throw')
plt.show()

