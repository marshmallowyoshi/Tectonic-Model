import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import math
import scipy.special as sp
import scipy.optimize as opt
from scipy import stats
import algorithm as alg

name = 'L2 H4-1'
name = 'data/' + name + '.csv'
df = pd.read_csv(name)
x = np.asarray(df.iloc[:,0])
y = np.asarray(df.iloc[:,3])
x, y = alg.resampleToPeak(x, y, sample_rate_override=0.9*len(x), kind='quadratic')
peakRateThreshold = 0.04
contWav = alg.waveletGen(x,y)
allSegments = alg.fullSegmentAnalysis(x,y, 
                                      name=name, 
                                      peakRateThreshold=peakRateThreshold, 
                                      join=True, 
                                      segmentMode='flat', 
                                      plot=True, 
                                      title='short', 
                                      samples_per_period=1300, 
                                      bands=(1,31), 
                                      sampling_period=1,
                                      skip_wavelets=False)
# plt.axis('off')
# plt.title('')
plt.tight_layout()
plt.savefig('plotoutput.png', dpi='figure', bbox_inches='tight', format=None)
# plt.show()

# allSegments, allCurves = alg.newFullSegmentAnalysis(x,y, name=name, peakRateThreshold=peakRateThreshold, join=True, segmentMode='poly', plot=False)
# contWav = alg.waveletGen(x,y)
# plt.plot(np.arange(250),signal.resample(y, 250) * 0.8 * contWav[0].shape[0] / np.max(y), color='black', linewidth=0.5)
# alg.regressionPlot(allSegments, allCurves, y, contWav)
# plt.show()

############################################################################################################################################

# allSegments, allCurves = alg.newFullSegmentAnalysis(x,y, name=name, peakRateThreshold=peakRateThreshold, join=True, segmentMode='poly', plot=True)
# plt.show()
