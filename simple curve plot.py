import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import algorithm as alg

name = 'R2 H2'
name = 'data/' + name + '.csv'

df = pd.read_csv(name)

x = np.asarray(df.iloc[:,0])
y = np.asarray(df.iloc[:,3])
x2, y2 = alg.resampleToPeak(x,y, sample_rate_override=215)
allSegments, polynomialRegression = alg.fullSegmentAnalysis(x2,y2,
                                                            plot=True,
                                                            skip_wavelets=True,
                                                            peakRateThreshold=0.04,
                                                            samples_per_period=1300,
                                                            sampling_period=1,
                                                            bands=(1,31),
                                                            resampling_kind='quadratic',
                                                            join=True,
                                                            segmentMode='poly',
                                                            title='short',
                                                            name=name)
x3, y3 = alg.resampleToPeak(x,y, sample_rate_override=193)
allSegments, polynomialRegression = alg.fullSegmentAnalysis(x3,y3,
                                                            plot=True,
                                                            skip_wavelets=True,
                                                            peakRateThreshold=0.04,
                                                            samples_per_period=1300,
                                                            sampling_period=1,
                                                            bands=(1,31),
                                                            resampling_kind='quadratic',
                                                            join=True,
                                                            segmentMode='poly',
                                                            title='short',
                                                            name=name)
plt.show()

