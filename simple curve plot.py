import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import algorithm as alg

name = 'L2 H4-1'
nameShort = str(name)
name = 'data/' + name + '.csv'

df = pd.read_csv(name)

x = np.asarray(df.iloc[:,0])
y = np.asarray(df.iloc[:,3])
x2, y2 = x, y
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

plt.rcParams['figure.dpi'] = 300
fig, ax = plt.subplots(1, 1)
x4, y4 = alg.resample_to_peak(x,y, samples_per_period=1300, sampling_period=1, bands=(1,31))
ax.plot(np.arange(len(y4)), y4, color='black', label='Raw Data')

for index, segment in enumerate(allSegments):
    y_predicted = segmentRegression[index][0].predict(segmentRegression[index][2])
    if index == 0:
        ax.plot(segmentRegression[index][1] + segment[0], y_predicted, color='red', label='Original', linewidth=3)
    else:
        ax.plot(segmentRegression[index][1] + segment[0], y_predicted, color='red', linewidth=3)
    

# for index, segment in enumerate(allSegments2):
#     y_predicted = segmentRegression2[index][0].predict(segmentRegression2[index][2])
#     if index == 0:
#         ax.plot(segmentRegression2[index][1] + segment[0], y_predicted, color='blue', label='Reduced')
#     else:
#         ax.plot(segmentRegression2[index][1] + segment[0], y_predicted, color='blue')
# fig.legend(bbox_to_anchor=(0.99,0.91), loc='upper right', ncol=1, borderaxespad=0., fontsize=10)
ax.set_xlabel('Sample Number')
ax.set_ylabel('Throw (m)')
ax.set_title(nameShort)
plt.show()

